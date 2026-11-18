"""Saved-game engine: record finished games, keep the useful ones, and learn from them again.

    selectors.py  what to keep        (all, win, outcome, top_return, surprise, upset, reservoir, ...)
    store.py      how much to keep    (byte budget, max games, priority + age eviction, persistence)
    learners.py   what to learn       (sil, awr, bc, value, replay_ppo, vtrace)
    recorder.py   rollout buffers -> finished-game records
    replay.py     action list -> engine replay -> tensors (and reanalyse with the current network)
    engine.py     the trainer-facing wrapper (--game-store in ppo.py)

See docs/game-store.md.
"""

from .record import Codec, GameRecord, discounted_returns
from .recorder import EpisodeRecorder
from .registry import make_learner, make_selector, parse_spec
from .selectors import SELECTORS, Selector, register_selector
from .store import GameStore

__all__ = [
    "Codec",
    "GameRecord",
    "discounted_returns",
    "EpisodeRecorder",
    "GameStore",
    "make_learner",
    "make_selector",
    "parse_spec",
    "SELECTORS",
    "Selector",
    "register_selector",
]
