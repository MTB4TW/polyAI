"""Which finished games to keep, and how much they matter.

A selector has one method, `score(meta, game) -> float | None`:

* `meta` is the game's metadata (outcome, map, seed, length, `train_return`, `first_value`,
  end-of-game statistics from the final info);
* `game` holds the raw per-step arrays (`reward`, `value`, `logprob`, `actions`, ...) before
  compression;
* the return value is the game's priority (higher means kept longer and sampled more often), or
  None to discard it. A discarded game is never compressed.

Selectors compose: `AnyOf`, `AllOf` and `Weighted`. `make_selector("win|surprise")` builds one
from a short spec (see `registry.py`). Add a selector by subclassing `Selector` and registering it
with `@register_selector("name")`.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional

import numpy as np

from .record import discounted_returns

SELECTORS: Dict[str, type] = {}


def register_selector(name: str):
    def deco(cls):
        SELECTORS[name] = cls
        cls.registry_name = name
        return cls

    return deco


class Selector:
    registry_name = "base"

    def score(self, meta: Dict[str, Any], game: Dict[str, np.ndarray]) -> Optional[float]:
        raise NotImplementedError

    def describe(self) -> str:
        return self.registry_name


@register_selector("all")
class KeepAll(Selector):
    """Keep every game at the same priority (the store's eviction then keeps the most recent)."""

    def __init__(self, priority: float = 1.0):
        self.priority = float(priority)

    def score(self, meta, game):
        return self.priority


@register_selector("outcome")
class OutcomeFilter(Selector):
    """Keep games with one of the given outcomes (combat: win / loss / timeout)."""

    def __init__(self, keep: Iterable[str] = ("win",), priority: float = 1.0):
        self.keep = {keep} if isinstance(keep, str) else set(keep)
        self.priority = float(priority)

    def score(self, meta, game):
        return self.priority if meta.get("outcome") in self.keep else None


@register_selector("win")
class WinsOnly(OutcomeFilter):
    def __init__(self, priority: float = 1.0):
        super().__init__(("win",), priority)


@register_selector("top_return")
class TopReturn(Selector):
    """Keep games whose return is in the top `fraction` of recent games.

    The threshold is a running quantile over the last `window` games, so it keeps working as the
    policy improves. `key` is a meta field (default `train_return`; Phase 1 can use an SPT field).
    The priority is the game's rank within the window, in [0, 1].
    """

    def __init__(self, fraction: float = 0.25, window: int = 500, key: str = "train_return", warmup: int = 20):
        self.fraction = float(fraction)
        self.window = int(window)
        self.key = key
        self.warmup = int(warmup)
        self._recent = []

    def score(self, meta, game):
        value = meta.get(self.key)
        if value is None:
            return None
        value = float(value)
        self._recent.append(value)
        if len(self._recent) > self.window:
            self._recent.pop(0)
        rank = float(np.mean(np.asarray(self._recent) <= value))
        if len(self._recent) < self.warmup:
            return rank
        return rank if rank >= 1.0 - self.fraction else None


@register_selector("surprise")
class Surprise(Selector):
    """Keep games the critic got wrong: mean |Monte Carlo return - predicted value| over the game.

    These are the games with the most left to learn from (the idea behind prioritised replay). Games
    the critic already predicts well are skipped. `min_error` is in training-reward units.
    """

    def __init__(self, gamma: float = 0.99, per_turn: bool = False, min_error: float = 0.0, power: float = 1.0):
        self.gamma = float(gamma)
        self.per_turn = bool(per_turn)
        self.min_error = float(min_error)
        self.power = float(power)

    def score(self, meta, game):
        returns = discounted_returns(game["reward"], self.gamma, game.get("end_turn") if self.per_turn else None)
        err = float(np.mean(np.abs(returns - game["value"]))) if returns.size else 0.0
        meta["value_error"] = err
        return err ** self.power if err >= self.min_error else None


@register_selector("upset")
class Upset(Selector):
    """Keep games whose result disagreed with the critic's opening estimate.

    A win the critic expected to lose (or a loss it expected to win) is the most informative kind of
    game for both the critic and the policy. The priority is |sign-adjusted result - first value|.
    """

    def __init__(self, win_value: float = 1.0, loss_value: float = -1.0, timeout_value: float = 0.0, min_gap: float = 0.0):
        self.targets = {"win": float(win_value), "loss": float(loss_value), "timeout": float(timeout_value)}
        self.min_gap = float(min_gap)

    def score(self, meta, game):
        target = self.targets.get(meta.get("outcome"))
        if target is None:
            return None
        gap = abs(target - float(np.tanh(meta.get("first_value", 0.0))))
        return gap if gap >= self.min_gap else None


@register_selector("reservoir")
class Reservoir(Selector):
    """Keep a uniform random sample: each game gets a random priority, so priority eviction behaves
    like reservoir sampling over everything seen (good for diversity and an unbiased value target)."""

    def __init__(self, seed: int = 0, keep_prob: float = 1.0):
        self.rng = np.random.default_rng(int(seed))
        self.keep_prob = float(keep_prob)

    def score(self, meta, game):
        if self.rng.random() >= self.keep_prob:
            return None
        return float(self.rng.random())


@register_selector("short_win")
class ShortWin(Selector):
    """Keep wins, prioritising fast ones (fewer steps means a cleaner plan to imitate)."""

    def __init__(self, max_length: Optional[int] = None):
        self.max_length = None if max_length is None else int(max_length)

    def score(self, meta, game):
        if meta.get("outcome") != "win":
            return None
        length = int(meta.get("length", 0))
        if self.max_length is not None and length > self.max_length:
            return None
        return 1.0 / (1.0 + length / 100.0)


@register_selector("field")
class FieldFilter(Selector):
    """Keep games where a meta field is in [low, high]; the priority is the field value itself.

    e.g. Phase 1 games above 25 SPT: `field:key=spt,low=25`.
    """

    def __init__(self, key: str, low: float = -np.inf, high: float = np.inf):
        self.key = key
        self.low = float(low)
        self.high = float(high)

    def score(self, meta, game):
        value = meta.get(self.key)
        if value is None:
            return None
        value = float(value)
        return value if self.low <= value <= self.high else None


# ---- combinators ---------------------------------------------------------------------------
class AnyOf(Selector):
    """Keep a game if any part keeps it; priority = max over the parts that keep it."""

    registry_name = "any"

    def __init__(self, parts):
        self.parts = list(parts)

    def score(self, meta, game):
        scores = [s for s in (p.score(meta, game) for p in self.parts) if s is not None]
        return max(scores) if scores else None

    def describe(self):
        return "|".join(p.describe() for p in self.parts)


class AllOf(Selector):
    """Keep a game only if every part keeps it; priority = product."""

    registry_name = "all_of"

    def __init__(self, parts):
        self.parts = list(parts)

    def score(self, meta, game):
        total = 1.0
        for part in self.parts:
            s = part.score(meta, game)
            if s is None:
                return None
            total *= float(s)
        return total

    def describe(self):
        return "&".join(p.describe() for p in self.parts)
