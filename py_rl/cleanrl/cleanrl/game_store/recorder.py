"""Turn PPO rollout buffers into finished-game records, with almost no extra work in the loop.

The trainer makes two calls:

* `note_infos(step, infos)` inside the rollout loop, which only looks at `final_info` (it is a
  no-op on steps where no game finished);
* `ingest(...)` once per iteration, after the rollout, with the rollout buffers it already has.

`ingest` drops the padded legal-action rows on the device (typically 20-60 of 256-512 rows are
valid), copies the rest to the CPU once, and splits each env's column at episode boundaries. Games
can span iterations, so each env keeps its unfinished segment until the game ends. A finished game
is first offered to the selector on its raw arrays; only a kept game is compressed and stored, so
discarded games cost no compression time.

Buffer convention (CleanRL): `dones[t] == 1` means the observation at step t starts a new game, so
the game that just ended owns steps up to t-1 and its terminal reward is `rewards[t-1]`.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

import numpy as np

from .record import PAYLOAD_BIG, Codec, GameRecord

try:  # torch is optional so the recorder can be unit-tested with numpy arrays
    import torch
except ImportError:  # pragma: no cover
    torch = None

END_TURN_GLOBAL_ID = 0

_SKIP_META_PREFIXES = ("legal_", "action_mask", "_")


def _to_numpy(x):
    if x is None:
        return None
    if torch is not None and isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def scalar_meta(info: Optional[Dict[str, Any]], max_keys: int = 400) -> Dict[str, Any]:
    """The scalar entries of a final info dict (outcome, map, seed, end-of-game statistics)."""
    out: Dict[str, Any] = {}
    if not isinstance(info, dict):
        return out
    for key, value in info.items():
        if len(out) >= max_keys:
            break
        if not isinstance(key, str) or key.startswith(_SKIP_META_PREFIXES):
            continue
        if isinstance(value, (bool, int, float, str)) or (isinstance(value, np.generic) and value.ndim == 0):
            out[key] = value.item() if isinstance(value, np.generic) else value
        elif key == "episode" and isinstance(value, dict):
            for sub in ("r", "l"):
                if sub in value:
                    out[f"episode_{sub}"] = float(np.asarray(value[sub]).reshape(-1)[0])
    if "combat_outcome" in out and "outcome" not in out:
        out["outcome"] = out["combat_outcome"]
    return out


class EpisodeRecorder:
    """Accumulates per-env rollout segments and emits `GameRecord`s for finished games.

    sink(record) is called for every kept game (usually `GameStore.add`).
    selector: object with `score(meta, arrays) -> float | None` (None discards the game).
    payload: "full" stores the training tensors; "actions" stores only the replayable action list.
    obs_dtype / feature_dtype: storage precision (float16 halves memory; float32 is exact).
    """

    def __init__(
        self,
        num_envs: int,
        sink: Callable[[GameRecord], Any],
        selector=None,
        payload: str = "full",
        codec: str = "xor_zlib",
        obs_dtype=np.float16,
        feature_dtype=np.float16,
        gamma: float = 0.99,
        extra_meta: Optional[Dict[str, Any]] = None,
    ):
        if payload not in ("full", "actions"):
            raise ValueError("payload must be 'full' or 'actions'")
        self.num_envs = int(num_envs)
        self.sink = sink
        self.selector = selector
        self.payload = payload
        self.codec = Codec(codec)
        self.obs_dtype = np.dtype(obs_dtype)
        self.feature_dtype = np.dtype(feature_dtype)
        self.gamma = float(gamma)
        self.extra_meta = dict(extra_meta or {})
        self._segments: List[List[Dict[str, np.ndarray]]] = [[] for _ in range(self.num_envs)]
        self._final_infos: List[List[Dict[str, Any]]] = [[] for _ in range(self.num_envs)]
        self.games_seen = 0
        self.games_kept = 0

    # ---- called inside the rollout loop -------------------------------------------------
    def note_infos(self, step: int, infos: Dict[str, Any]) -> None:
        final_infos = infos.get("final_info") if isinstance(infos, dict) else None
        if final_infos is None:
            return
        mask = infos.get("_final_info")
        for env_idx, info in enumerate(final_infos):
            if env_idx >= self.num_envs:
                break
            if mask is not None and not bool(mask[env_idx]):
                continue
            if isinstance(info, dict):
                self._final_infos[env_idx].append(scalar_meta(info))

    # ---- called once per iteration ------------------------------------------------------
    def ingest(
        self,
        obs,
        legal_ids,
        legal_valid,
        selected_slots,
        actions,
        logprobs,
        values,
        rewards,
        dones,
        next_done,
        legal_features=None,
        global_step: int = 0,
        reward_scale: float = 1.0,
        extras: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Split the iteration's buffers into games. Returns the number of games finished."""
        num_steps = int(_to_numpy(dones).shape[0])
        step_cols = self._compact(obs, legal_ids, legal_valid, selected_slots, legal_features)
        actions_np = _to_numpy(actions).reshape(num_steps, self.num_envs).astype(np.int32)
        small = {
            "logprob": _to_numpy(logprobs).reshape(num_steps, self.num_envs).astype(np.float32),
            "value": _to_numpy(values).reshape(num_steps, self.num_envs).astype(np.float32),
            "reward": _to_numpy(rewards).reshape(num_steps, self.num_envs).astype(np.float32),
        }
        extras_np = {k: _to_numpy(v) for k, v in (extras or {}).items()}
        dones_np = _to_numpy(dones).reshape(num_steps, self.num_envs) > 0.5
        next_done_np = _to_numpy(next_done).reshape(self.num_envs) > 0.5

        finished = 0
        for env in range(self.num_envs):
            col = step_cols[env]
            # Cut points: a game ends before every step whose observation starts a new game, and after the
            # last step if the env finished on it.
            starts = [t for t in range(1, num_steps) if dones_np[t, env]]
            cuts = starts + ([num_steps] if next_done_np[env] else [])
            begin = 0
            for cut in cuts:
                self._segments[env].append(self._slice(col, actions_np, small, extras_np, env, begin, cut))
                self._finish(env, global_step, reward_scale)
                finished += 1
                begin = cut
            if begin < num_steps:
                self._segments[env].append(self._slice(col, actions_np, small, extras_np, env, begin, num_steps))
        return finished

    # ---- internals -----------------------------------------------------------------------
    def _compact(self, obs, legal_ids, legal_valid, selected_slots, legal_features):
        """Per env: obs [S,D], ragged legal ids/features, slot remapped to the valid-row index."""
        valid = legal_valid
        if torch is not None and isinstance(valid, torch.Tensor):
            valid = valid.bool()
            # env-major so a boolean gather yields each env's rows contiguously and in step order
            valid_e = valid.transpose(0, 1)
            ids_e = legal_ids.transpose(0, 1)[valid_e].to(torch.int32)
            counts = valid_e.sum(-1).to(torch.int32)  # [N,S]
            rank = torch.cumsum(valid_e.to(torch.int32), dim=-1) - 1
            slot_e = rank.gather(-1, selected_slots.long().transpose(0, 1).unsqueeze(-1)).squeeze(-1)
            obs_e = obs.transpose(0, 1).reshape(self.num_envs, obs.shape[0], -1)
            feats_e = None
            if legal_features is not None and self.payload == "full":
                feats_e = legal_features.transpose(0, 1)[valid_e]
            if self.payload == "full":
                obs_np = obs_e.to(torch.float32).cpu().numpy().astype(self.obs_dtype, copy=False)
                feats_np = None if feats_e is None else feats_e.to(torch.float32).cpu().numpy().astype(self.feature_dtype, copy=False)
                ids_np = ids_e.cpu().numpy()
            else:
                obs_np, feats_np, ids_np = None, None, None
            counts_np = counts.cpu().numpy()
            slot_np = slot_e.cpu().numpy().astype(np.int16)
        else:
            valid_np = np.asarray(valid).astype(bool)
            valid_e = valid_np.transpose(1, 0, 2)
            counts_np = valid_e.sum(-1).astype(np.int32)
            rank = np.cumsum(valid_e, axis=-1) - 1
            slots = np.asarray(selected_slots).astype(np.int64).transpose(1, 0)
            slot_np = np.take_along_axis(rank, slots[..., None], axis=-1)[..., 0].astype(np.int16)
            if self.payload == "full":
                ids_np = np.asarray(legal_ids).transpose(1, 0, 2)[valid_e].astype(np.int32)
                obs_arr = np.asarray(obs)
                obs_np = obs_arr.transpose(1, 0, *range(2, obs_arr.ndim)).reshape(self.num_envs, obs_arr.shape[0], -1)
                obs_np = obs_np.astype(self.obs_dtype)
                feats_np = None
                if legal_features is not None:
                    feats_np = np.asarray(legal_features).transpose(1, 0, 2, 3)[valid_e].astype(self.feature_dtype)
            else:
                obs_np, feats_np, ids_np = None, None, None

        cols = []
        row = 0
        for env in range(self.num_envs):
            n_rows = int(counts_np[env].sum())
            col = {"counts": counts_np[env], "slot": slot_np[env]}
            if self.payload == "full":
                col["obs"] = obs_np[env]
                col["legal_ids"] = ids_np[row:row + n_rows]
                if feats_np is not None:
                    col["legal_features"] = feats_np[row:row + n_rows]
            col["row_offsets"] = np.concatenate([[0], np.cumsum(counts_np[env])]).astype(np.int64)
            cols.append(col)
            row += n_rows
        return cols

    def _slice(self, col, actions_np, small, extras_np, env, begin, end):
        seg = {
            "actions": actions_np[begin:end, env],
            "slot": col["slot"][begin:end],
            "counts": col["counts"][begin:end],
        }
        for key, arr in small.items():
            seg[key] = arr[begin:end, env]
        if "obs" in col:
            r0, r1 = int(col["row_offsets"][begin]), int(col["row_offsets"][end])
            seg["obs"] = col["obs"][begin:end]
            seg["legal_ids"] = col["legal_ids"][r0:r1]
            if "legal_features" in col:
                seg["legal_features"] = col["legal_features"][r0:r1]
        for key, arr in extras_np.items():
            seg["x_" + key] = arr[begin:end, env]
        # Copy so the record does not pin the whole iteration's arrays in memory.
        return {k: np.array(v, copy=True) for k, v in seg.items()}

    def _finish(self, env, global_step, reward_scale):
        parts = self._segments[env]
        self._segments[env] = []
        infos = self._final_infos[env]
        meta = infos.pop(0) if infos else {}
        game = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
        self.games_seen += 1

        length = int(game["actions"].shape[0])
        meta.update(self.extra_meta)
        meta.update({
            "length": length,
            "global_step": int(global_step),
            "reward_scale": float(reward_scale),
            "train_return": float(game["reward"].sum()),
            "first_value": float(game["value"][0]) if length else 0.0,
        })
        meta.setdefault("outcome", meta.get("combat_outcome"))
        game["end_turn"] = game["actions"] == END_TURN_GLOBAL_ID

        score = 0.0 if self.selector is None else self.selector.score(meta, game)
        if score is None:
            return
        record = self._encode(game, meta)
        record.priority = float(score)
        self.games_kept += 1
        self.sink(record)

    def _encode(self, game, meta) -> GameRecord:
        actions = game["actions"].astype(np.int32)
        if self.payload == "actions":
            return GameRecord(actions=actions, meta=meta)
        offsets = np.concatenate([[0], np.cumsum(game["counts"])]).astype(np.int32)
        payload = {
            "legal_offsets": offsets,
            "legal_ids": game["legal_ids"].astype(np.int32),
            "slot": game["slot"].astype(np.int16),
            "logprob": game["logprob"],
            "value": game["value"],
            "reward": game["reward"],
            "end_turn": game["end_turn"],
        }
        for key in PAYLOAD_BIG:
            if key in game:
                payload[key] = self.codec.encode(game[key])
        extras = {k[2:]: self.codec.encode(v) if v.ndim >= 2 else v for k, v in game.items() if k.startswith("x_")}
        return GameRecord(actions=actions, meta=meta, payload=payload, extras=extras)
