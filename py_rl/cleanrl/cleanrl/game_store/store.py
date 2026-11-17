"""A bounded, prioritised collection of finished games.

Memory is bounded two ways:

* `budget_mb`: bytes held by training payloads. When it is exceeded, the lowest-priority games
  lose their payload first (`demote=True`: they stay as replayable action lists) or are dropped.
* `max_games`: total records, including action-only ones. Beyond it the lowest-priority record is
  dropped.

Priorities can decay with age (`half_life` in games added), so the store prefers recent play
without forgetting rare, high-priority games (a long-ago win against a hard opponent).

Sampling draws games with probability proportional to (priority + eps) ** alpha (alpha=0 is
uniform), then steps uniformly within each game. A small LRU cache keeps recently decompressed
games so a batch that revisits a game does not pay for decompression twice.
"""

from __future__ import annotations

import collections
import glob
import os
import pickle
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

from .record import GameRecord, discounted_returns

STORE_VERSION = "game_store_v1"


class GameStore:
    def __init__(
        self,
        budget_mb: float = 1024.0,
        max_games: int = 100_000,
        alpha: float = 1.0,
        eps: float = 1e-3,
        half_life: Optional[float] = None,
        demote: bool = True,
        cache_games: int = 64,
        seed: int = 0,
    ):
        self.budget_bytes = int(float(budget_mb) * 1024 * 1024)
        self.max_games = int(max_games)
        self.alpha = float(alpha)
        self.eps = float(eps)
        self.half_life = None if half_life is None else float(half_life)
        self.demote = bool(demote)
        self.rng = np.random.default_rng(int(seed))
        self._records: Dict[int, GameRecord] = {}
        self._added_at: Dict[int, int] = {}
        self._payload_bytes = 0
        self._next_id = 0
        self._cache: "collections.OrderedDict[int, Dict[str, np.ndarray]]" = collections.OrderedDict()
        self._cache_games = int(cache_games)
        self.added = 0
        self.evicted = 0
        self.demoted = 0

    # ---- size and stats ---------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._records)

    @property
    def payload_bytes(self) -> int:
        return self._payload_bytes

    def records(self) -> Iterable[GameRecord]:
        return self._records.values()

    def get(self, game_id: int) -> GameRecord:
        return self._records[game_id]

    def stats(self) -> Dict[str, float]:
        recs = list(self._records.values())
        with_payload = [r for r in recs if r.has_payload]
        out = {
            "games": float(len(recs)),
            "games_with_payload": float(len(with_payload)),
            "payload_mb": self._payload_bytes / (1024 * 1024),
            "steps_with_payload": float(sum(r.length for r in with_payload)),
            "added": float(self.added),
            "evicted": float(self.evicted),
            "demoted": float(self.demoted),
        }
        if recs:
            out["mean_priority"] = float(np.mean([r.priority for r in recs]))
            outcomes = collections.Counter(r.outcome for r in recs)
            for name in ("win", "loss", "timeout"):
                out[f"frac_{name}"] = outcomes.get(name, 0) / len(recs)
        if with_payload:
            out["bytes_per_step"] = self._payload_bytes / max(1.0, out["steps_with_payload"])
        return out

    # ---- adding and evicting --------------------------------------------------------------
    def effective_priority(self, game_id: int) -> float:
        rec = self._records[game_id]
        p = float(rec.priority)
        if self.half_life is not None:
            age = self.added - self._added_at[game_id]
            p *= 0.5 ** (age / self.half_life)
        return p

    def add(self, record: GameRecord) -> int:
        record.game_id = self._next_id
        self._next_id += 1
        self._records[record.game_id] = record
        self._added_at[record.game_id] = self.added
        self.added += 1
        if record.has_payload:
            self._payload_bytes += record.nbytes()
        self._enforce_limits()
        return record.game_id

    def _lowest(self, with_payload: bool) -> Optional[int]:
        best_id, best_p = None, np.inf
        for gid, rec in self._records.items():
            if with_payload and not rec.has_payload:
                continue
            p = self.effective_priority(gid)
            if p < best_p:
                best_id, best_p = gid, p
        return best_id

    def _enforce_limits(self) -> None:
        while self._payload_bytes > self.budget_bytes:
            gid = self._lowest(with_payload=True)
            if gid is None:
                break
            if self.demote:
                self._demote(gid)
            else:
                self._remove(gid)
        while len(self._records) > self.max_games:
            gid = self._lowest(with_payload=False)
            if gid is None:
                break
            self._remove(gid)

    def _demote(self, gid: int) -> None:
        rec = self._records[gid]
        if rec.has_payload:
            self._payload_bytes -= rec.nbytes()
            self._records[gid] = rec.strip_payload()
            self._cache.pop(gid, None)
            self.demoted += 1

    def _remove(self, gid: int) -> None:
        rec = self._records.pop(gid)
        self._added_at.pop(gid, None)
        self._cache.pop(gid, None)
        if rec.has_payload:
            self._payload_bytes -= rec.nbytes()
        self.evicted += 1

    def set_priority(self, game_id: int, priority: float) -> None:
        """Update a game's priority (e.g. a learner re-scores games by their current TD error)."""
        if game_id in self._records:
            self._records[game_id].priority = float(priority)

    # ---- sampling ---------------------------------------------------------------------------
    def tensors(self, game_id: int) -> Dict[str, np.ndarray]:
        cached = self._cache.get(game_id)
        if cached is not None:
            self._cache.move_to_end(game_id)
            return cached
        data = self._records[game_id].tensors()
        self._cache[game_id] = data
        while len(self._cache) > self._cache_games:
            self._cache.popitem(last=False)
        return data

    def sample_game_ids(self, k: int, require_payload: bool = True, filter_fn=None) -> List[int]:
        ids = [gid for gid, r in self._records.items()
               if (r.has_payload or not require_payload) and (filter_fn is None or filter_fn(r))]
        if not ids:
            return []
        if self.alpha == 0.0:
            probs = None
        else:
            p = np.array([max(self.effective_priority(g), 0.0) + self.eps for g in ids]) ** self.alpha
            probs = p / p.sum()
        replace = k > len(ids)
        picks = self.rng.choice(len(ids), size=int(k), replace=replace, p=probs)
        return [ids[i] for i in picks]

    def sample_steps(
        self,
        batch_size: int,
        games_per_batch: int = 16,
        gamma: float = 0.99,
        per_turn: bool = False,
        reward_scale_now: float = 1.0,
        filter_fn=None,
    ) -> Optional[Dict[str, np.ndarray]]:
        """A padded batch of single steps, for step-wise learners (self-imitation, AWR, BC, value).

        `return` is the Monte Carlo return from that step to the end of its game, rescaled from the
        reward scale the game was recorded in to `reward_scale_now`, so it is comparable with the
        current critic when `--norm-returns` is on.
        """
        game_ids = self.sample_game_ids(games_per_batch, filter_fn=filter_fn)
        if not game_ids:
            return None
        per_game = np.bincount(self.rng.integers(0, len(game_ids), size=int(batch_size)), minlength=len(game_ids))
        rows = []
        for gid, count in zip(game_ids, per_game):
            if count == 0:
                continue
            data = self.tensors(gid)
            rec = self._records[gid]
            ret_key = ("_return", float(gamma), bool(per_turn))
            if ret_key not in data:
                data[ret_key] = discounted_returns(data["reward"], gamma, data["end_turn"] if per_turn else None)
            scale = float(rec.meta.get("reward_scale", 1.0)) / max(float(reward_scale_now), 1e-8)
            for t in self.rng.integers(0, rec.length, size=int(count)):
                rows.append((gid, int(t), data, data[ret_key][t] * scale))
        return collate_steps(rows)

    def sample_games(self, k: int, filter_fn=None) -> List[Dict[str, Any]]:
        """Whole games (decompressed), for sequence learners such as V-trace."""
        out = []
        for gid in self.sample_game_ids(k, filter_fn=filter_fn):
            out.append({"game_id": gid, "meta": self._records[gid].meta, **self.tensors(gid)})
        return out

    # ---- persistence ------------------------------------------------------------------------
    def save(self, directory: str, shard_games: int = 500) -> List[str]:
        """Write the store to `directory` as pickle shards (overwrites earlier shards)."""
        os.makedirs(directory, exist_ok=True)
        for old in glob.glob(os.path.join(directory, "games_*.pkl")):
            os.remove(old)
        recs = list(self._records.values())
        paths = []
        for i in range(0, len(recs), shard_games):
            path = os.path.join(directory, f"games_{i // shard_games:05d}.pkl")
            with open(path, "wb") as f:
                pickle.dump({"version": STORE_VERSION, "records": [r.to_state() for r in recs[i:i + shard_games]]},
                            f, protocol=pickle.HIGHEST_PROTOCOL)
            paths.append(path)
        return paths

    def load(self, directory: str, filter_fn=None) -> int:
        """Add the games saved in `directory` (only load stores you wrote: this unpickles)."""
        loaded = 0
        for path in sorted(glob.glob(os.path.join(directory, "games_*.pkl"))):
            with open(path, "rb") as f:
                blob = pickle.load(f)
            if blob.get("version") != STORE_VERSION:
                raise ValueError(f"{path}: unsupported store version {blob.get('version')!r}")
            for state in blob["records"]:
                rec = GameRecord.from_state(state)
                if filter_fn is None or filter_fn(rec):
                    self.add(rec)
                    loaded += 1
        return loaded


def collate_steps(rows) -> Optional[Dict[str, np.ndarray]]:
    """Pad a list of (game_id, t, tensors, return) to the widest legal set in the batch."""
    if not rows:
        return None
    widths = [int(d["legal_offsets"][t + 1] - d["legal_offsets"][t]) for _, t, d, _ in rows]
    width = max(1, max(widths))
    b = len(rows)
    first = rows[0][2]
    obs = np.zeros((b,) + first["obs"].shape[1:], dtype=np.float32)
    ids = np.zeros((b, width), dtype=np.int64)
    valid = np.zeros((b, width), dtype=bool)
    has_feats = "legal_features" in first
    feats = np.zeros((b, width, first["legal_features"].shape[-1]), dtype=np.float32) if has_feats else None
    out = {k: np.zeros(b, dtype=np.float32) for k in ("logprob", "value", "reward", "return")}
    slot = np.zeros(b, dtype=np.int64)
    game_ids = np.zeros(b, dtype=np.int64)
    steps = np.zeros(b, dtype=np.int64)
    for i, ((gid, t, d, ret), n) in enumerate(zip(rows, widths)):
        r0 = int(d["legal_offsets"][t])
        obs[i] = d["obs"][t]
        ids[i, :n] = d["legal_ids"][r0:r0 + n]
        valid[i, :n] = True
        if has_feats:
            feats[i, :n] = d["legal_features"][r0:r0 + n]
        slot[i] = int(d["slot"][t])
        out["logprob"][i] = d["logprob"][t]
        out["value"][i] = d["value"][t]
        out["reward"][i] = d["reward"][t]
        out["return"][i] = ret
        game_ids[i], steps[i] = gid, t
    batch = {"obs": obs, "legal_ids": ids, "legal_valid": valid, "slot": slot, "game_id": game_ids, "step": steps, **out}
    if has_feats:
        batch["legal_features"] = feats
    return batch
