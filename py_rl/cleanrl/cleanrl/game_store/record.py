"""One finished game, stored compactly.

A `GameRecord` always has two parts:

* **metadata and the action list** (a few hundred bytes): map, episode seed, outcome, return, the
  global action id chosen at every step. That is enough to replay the game in the engine
  (`replay.py`), so a store can keep millions of games at this level.
* **an optional training payload**: the tensors a learner needs without touching the engine
  (observations, the legal actions and their features, the chosen slot, the behaviour policy's
  log-probability and value, rewards). The legal actions are stored *ragged* (only the valid rows,
  not the padded `max_legal_actions` block), and the large arrays are compressed. See `Codec`.

Torch-free: everything here is numpy so the store can be tested and inspected without a GPU.
"""

from __future__ import annotations

import dataclasses
import zlib
from typing import Any, Dict, Optional

import numpy as np

RECORD_VERSION = "game_record_v1"


# --------------------------------------------------------------------------------------------
# Compression
# --------------------------------------------------------------------------------------------
class Codec:
    """Lossless array compression.

    `xor_zlib` XORs each row's bit pattern with the previous row before zlib. Consecutive
    observations within a turn differ in a few entries, so the XOR is mostly zeros and compresses
    far better than the raw rows. `zlib` compresses the raw bytes; `none` keeps them as they are.
    """

    NAMES = ("none", "zlib", "xor_zlib")

    def __init__(self, name: str = "xor_zlib", level: int = 1):
        if name not in self.NAMES:
            raise ValueError(f"unknown codec {name!r}; choose from {self.NAMES}")
        self.name = name
        self.level = int(level)

    @staticmethod
    def _bits(arr: np.ndarray) -> np.ndarray:
        width = arr.dtype.itemsize
        uint = {1: np.uint8, 2: np.uint16, 4: np.uint32, 8: np.uint64}[width]
        return np.ascontiguousarray(arr).view(uint)

    def encode(self, arr: np.ndarray) -> Dict[str, Any]:
        arr = np.ascontiguousarray(arr)
        packed = {"shape": tuple(arr.shape), "dtype": arr.dtype.str, "codec": self.name}
        if self.name == "none":
            packed["data"] = arr
            return packed
        if self.name == "xor_zlib" and arr.ndim >= 2 and arr.shape[0] > 1:
            bits = self._bits(arr).reshape(arr.shape[0], -1)
            delta = bits.copy()
            delta[1:] ^= bits[:-1]
            raw = delta.tobytes()
        else:
            packed["codec"] = "zlib"
            raw = arr.tobytes()
        packed["data"] = zlib.compress(raw, self.level)
        return packed

    @staticmethod
    def decode(packed: Dict[str, Any]) -> np.ndarray:
        codec = packed["codec"]
        dtype = np.dtype(packed["dtype"])
        shape = tuple(packed["shape"])
        if codec == "none":
            return packed["data"]
        raw = zlib.decompress(packed["data"])
        if codec == "zlib":
            return np.frombuffer(raw, dtype=dtype).reshape(shape).copy()
        uint = {1: np.uint8, 2: np.uint16, 4: np.uint32, 8: np.uint64}[dtype.itemsize]
        delta = np.frombuffer(raw, dtype=uint).reshape(shape[0], -1)
        bits = np.bitwise_xor.accumulate(delta, axis=0)
        return bits.view(dtype).reshape(shape)

    @staticmethod
    def nbytes(packed: Dict[str, Any]) -> int:
        data = packed["data"]
        return int(data.nbytes if isinstance(data, np.ndarray) else len(data))


# --------------------------------------------------------------------------------------------
# Record
# --------------------------------------------------------------------------------------------
# Payload fields and how they are kept. "big" fields go through the codec; the rest are small.
PAYLOAD_BIG = ("obs", "legal_features")
PAYLOAD_SMALL = (
    "legal_offsets",  # int32 [T+1]: rows of step t are legal_ids[legal_offsets[t]:legal_offsets[t+1]]
    "legal_ids",  # int32 [sum n_t]
    "slot",  # int16 [T]: index of the chosen action among the step's valid rows
    "logprob",  # float32 [T]: behaviour policy log-probability of the chosen action
    "value",  # float32 [T]: behaviour critic value
    "reward",  # float32 [T]: training reward of the step (in the scale given by `reward_scale`)
    "end_turn",  # bool [T]: the step ended the turn (for per-turn discounting)
)


@dataclasses.dataclass
class GameRecord:
    """A finished game. `payload` is None for an action-only record."""

    actions: np.ndarray  # int32 [T]: global action id at every step (the replay key)
    meta: Dict[str, Any]  # map_id, map_path, episode_seed, outcome, episode_return, length, global_step, ...
    payload: Optional[Dict[str, Any]] = None  # compressed training tensors, see PAYLOAD_*
    extras: Dict[str, Any] = dataclasses.field(default_factory=dict)  # learner-specific per-step arrays
    priority: float = 0.0  # set by the selector; higher is kept longer and sampled more
    game_id: int = -1  # assigned by the store

    @property
    def length(self) -> int:
        return int(self.actions.shape[0])

    @property
    def outcome(self) -> Optional[str]:
        return self.meta.get("outcome")

    @property
    def has_payload(self) -> bool:
        return self.payload is not None

    def nbytes(self) -> int:
        total = int(self.actions.nbytes) + 256  # rough metadata allowance
        for part in (self.payload or {}, self.extras):
            for value in part.values():
                if isinstance(value, dict) and "codec" in value:
                    total += Codec.nbytes(value)
                elif isinstance(value, np.ndarray):
                    total += int(value.nbytes)
        return total

    # ---- payload access ------------------------------------------------------------------
    def tensors(self) -> Dict[str, np.ndarray]:
        """Decompressed payload (numpy). Raises if this is an action-only record."""
        if self.payload is None:
            raise ValueError("action-only record: replay it through the engine to get tensors")
        out = {}
        for key, value in self.payload.items():
            out[key] = Codec.decode(value) if isinstance(value, dict) and "codec" in value else value
        for key, value in self.extras.items():
            out[key] = Codec.decode(value) if isinstance(value, dict) and "codec" in value else value
        return out

    def strip_payload(self) -> "GameRecord":
        """Drop the tensors and keep the replayable action list (used when the byte budget is tight)."""
        return dataclasses.replace(self, payload=None, extras={})

    # ---- serialisation -------------------------------------------------------------------
    def to_state(self) -> Dict[str, Any]:
        return {
            "version": RECORD_VERSION,
            "actions": self.actions,
            "meta": self.meta,
            "payload": self.payload,
            "extras": self.extras,
            "priority": float(self.priority),
            "game_id": int(self.game_id),
        }

    @classmethod
    def from_state(cls, state: Dict[str, Any]) -> "GameRecord":
        if state.get("version") != RECORD_VERSION:
            raise ValueError(f"unsupported game record version {state.get('version')!r}")
        return cls(
            actions=np.asarray(state["actions"], dtype=np.int32),
            meta=dict(state["meta"]),
            payload=state.get("payload"),
            extras=dict(state.get("extras") or {}),
            priority=float(state.get("priority", 0.0)),
            game_id=int(state.get("game_id", -1)),
        )


def discounted_returns(rewards: np.ndarray, gamma: float, end_turn: Optional[np.ndarray] = None) -> np.ndarray:
    """Monte Carlo return of every step of one finished game.

    With `end_turn`, the discount applies only after steps that ended the turn (matching
    `gae.per_turn_discounts`); otherwise every step is discounted by gamma.
    """
    rewards = np.asarray(rewards, dtype=np.float64)
    if end_turn is None:
        discounts = np.full(rewards.shape, gamma)
    else:
        discounts = np.where(np.asarray(end_turn, dtype=bool), gamma, 1.0)
    out = np.zeros_like(rewards)
    running = 0.0
    for t in range(rewards.shape[0] - 1, -1, -1):
        running = rewards[t] + discounts[t] * running
        out[t] = running
    return out.astype(np.float32)
