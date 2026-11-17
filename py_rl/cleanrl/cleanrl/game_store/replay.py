"""Re-play a stored game in the engine, from its map, episode seed and action list.

The environment is deterministic given (map, episode seed, action sequence): the game RNG is
`new Random(seed)` and SimpleAgent is seeded with the same seed (see tools/hash_env_trajectory.py,
which checks this). So an action-only record costs a few hundred bytes and can be turned back
into full training tensors whenever a learner wants them:

* `replay_game(env, record)` steps the env through the stored actions and returns the same per-step
  arrays the recorder produces. It raises `ReplayMismatch` if a stored action is not legal at its
  step or the outcome differs, which means the engine, rules or contract changed since the game
  was recorded.
* with `agent=`, the current network also scores every step (log-probability and value). That is
  "reanalyse" (MuZero Reanalyse): fresh targets for old games from the newer network, without
  playing new games.
* `materialize(record, arrays)` packs the arrays back into a record with a payload.

Needs a single (non-vector) `TribesGymWrapper` with the compiled Java engine.
"""

from __future__ import annotations

import contextlib
from typing import Any, Dict, Optional

import numpy as np

from .record import PAYLOAD_BIG, Codec, GameRecord
from .recorder import END_TURN_GLOBAL_ID, scalar_meta


class ReplayMismatch(RuntimeError):
    pass


@contextlib.contextmanager
def pinned_level(env, level_file: str):
    """Make the next reset of `env` load `level_file`, whatever its level pool and selection mode."""
    base = getattr(env, "unwrapped", env)
    saved = (base._level_pool, base._level_pool_size)
    base._level_pool, base._level_pool_size = [level_file], 1
    try:
        yield base
    finally:
        base._level_pool, base._level_pool_size = saved


def _legal_rows(info):
    ids = np.asarray(info["legal_global_ids_padded"])
    valid = np.asarray(info["legal_action_valid_mask"]).astype(bool)
    feats = np.asarray(info.get("legal_action_features_padded")) if "legal_action_features_padded" in info else None
    return ids[valid], (feats[valid] if feats is not None else None)


def replay_game(env, record: GameRecord, agent=None, device=None, check_outcome: bool = True) -> Dict[str, Any]:
    meta = record.meta
    level = meta.get("map_path")
    seed = meta.get("episode_seed")
    if level is None or seed is None:
        raise ValueError("record has no map_path / episode_seed; it cannot be replayed")
    with pinned_level(env, level):
        obs, info = env.reset(seed=int(seed))

    cols = {k: [] for k in ("obs", "legal_ids", "legal_features", "counts", "slot", "reward", "logprob", "value")}
    final_info: Dict[str, Any] = {}
    for t, action in enumerate(record.actions.tolist()):
        ids, feats = _legal_rows(info)
        where = np.nonzero(ids == action)[0]
        if where.size != 1:
            raise ReplayMismatch(f"step {t}: stored action {action} is not legal here (map={level}, seed={seed})")
        slot = int(where[0])
        cols["obs"].append(np.asarray(obs, dtype=np.float32).reshape(-1))
        cols["legal_ids"].append(ids.astype(np.int32))
        if feats is not None:
            cols["legal_features"].append(feats.astype(np.float32))
        cols["counts"].append(ids.shape[0])
        cols["slot"].append(slot)
        if agent is not None:
            lp, v = _score(agent, obs, ids, feats, slot, device)
            cols["logprob"].append(lp)
            cols["value"].append(v)
        obs, reward, terminated, truncated, info = env.step(int(action))
        cols["reward"].append(float(reward))
        if terminated or truncated:
            final_info = info
            if t != record.length - 1:
                raise ReplayMismatch(f"game ended at step {t}, record has {record.length} steps")
            break
    else:
        raise ReplayMismatch(f"game did not end after the stored {record.length} steps")

    replay_meta = scalar_meta(final_info)
    if check_outcome and meta.get("outcome") is not None and replay_meta.get("outcome") != meta.get("outcome"):
        raise ReplayMismatch(f"outcome {replay_meta.get('outcome')!r} != recorded {meta.get('outcome')!r}")

    n = len(cols["slot"])
    game = {
        "actions": record.actions.astype(np.int32),
        "obs": np.stack(cols["obs"]),
        "legal_ids": np.concatenate(cols["legal_ids"]),
        "counts": np.asarray(cols["counts"], dtype=np.int32),
        "slot": np.asarray(cols["slot"], dtype=np.int16),
        "reward": np.asarray(cols["reward"], dtype=np.float32),
        "logprob": np.asarray(cols["logprob"] or np.zeros(n), dtype=np.float32),
        "value": np.asarray(cols["value"] or np.zeros(n), dtype=np.float32),
        "end_turn": record.actions == END_TURN_GLOBAL_ID,
        "meta": replay_meta,
    }
    if cols["legal_features"]:
        game["legal_features"] = np.concatenate(cols["legal_features"])
    return game


def _score(agent, obs, ids, feats, slot, device):
    import torch

    with torch.no_grad():
        x = torch.as_tensor(np.asarray(obs, dtype=np.float32).reshape(1, -1), device=device)
        lid = torch.as_tensor(ids.reshape(1, -1).astype(np.int64), device=device)
        valid = torch.ones_like(lid, dtype=torch.bool)
        f = None if feats is None else torch.as_tensor(feats[None].astype(np.float32), device=device)
        s = torch.as_tensor([slot], device=device)
        _, _, lp, _, v = agent.get_action_and_value(x, legal_global_ids=lid, legal_action_valid_mask=valid,
                                                    legal_action_features=f, selected_slot=s)
    return float(lp.item()), float(v.reshape(-1)[0].item())


def materialize(record: GameRecord, game: Dict[str, Any], codec: str = "xor_zlib", obs_dtype=np.float16,
                reward_scale: float = 1.0, behaviour: Optional[str] = None) -> GameRecord:
    """A copy of `record` with a training payload built from `replay_game` output.

    The replayed rewards are raw, so the record's `reward_scale` becomes `reward_scale` (pass the
    trainer's current scale when `--norm-returns` is on and the rewards should be divided by it).
    `behaviour` notes where the stored log-probabilities came from (e.g. "reanalyse").
    """
    c = Codec(codec)
    offsets = np.concatenate([[0], np.cumsum(game["counts"])]).astype(np.int32)
    scale = float(reward_scale)
    payload = {
        "legal_offsets": offsets,
        "legal_ids": game["legal_ids"].astype(np.int32),
        "slot": game["slot"].astype(np.int16),
        "logprob": game["logprob"].astype(np.float32),
        "value": game["value"].astype(np.float32),
        "reward": (game["reward"] / scale).astype(np.float32),
        "end_turn": game["end_turn"],
    }
    for key in PAYLOAD_BIG:
        if key in game:
            payload[key] = c.encode(game[key].astype(obs_dtype))
    meta = dict(record.meta)
    meta["reward_scale"] = scale
    if behaviour:
        meta["behaviour"] = behaviour
    return GameRecord(actions=record.actions.copy(), meta=meta, payload=payload, priority=record.priority,
                      game_id=record.game_id)
