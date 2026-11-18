"""Build selectors and learners from short command-line specs.

Spec grammar:  name[:key=value,key=value]   combined with  |  (keep if any)  or  &  (keep if all).

    win                          keep wins
    outcome:keep=win+timeout     keep wins and timeouts (list values are joined with +)
    surprise:min_error=0.5|win   keep surprising games and every win
    top_return:fraction=0.2
    sil:value_coef=0.01          self-imitation learner
    vtrace:lam=0.9
"""

from __future__ import annotations

from typing import Any, Dict

from .selectors import SELECTORS, AllOf, AnyOf


def _parse_value(text: str) -> Any:
    if "+" in text:
        return [_parse_value(p) for p in text.split("+")]
    low = text.lower()
    if low in ("true", "false"):
        return low == "true"
    if low in ("none", "null"):
        return None
    for cast in (int, float):
        try:
            return cast(text)
        except ValueError:
            pass
    return text


def parse_spec(spec: str):
    name, _, rest = spec.strip().partition(":")
    kwargs: Dict[str, Any] = {}
    if rest:
        for item in rest.split(","):
            if not item:
                continue
            key, eq, value = item.partition("=")
            if not eq:
                raise ValueError(f"bad option {item!r} in {spec!r} (expected key=value)")
            kwargs[key.strip()] = _parse_value(value.strip())
    return name.strip(), kwargs


def make_selector(spec: str):
    if "|" in spec:
        return AnyOf([make_selector(p) for p in spec.split("|")])
    if "&" in spec:
        return AllOf([make_selector(p) for p in spec.split("&")])
    name, kwargs = parse_spec(spec)
    if name not in SELECTORS:
        raise ValueError(f"unknown game selector {name!r}; choose from {sorted(SELECTORS)}")
    return SELECTORS[name](**kwargs)


def make_learner(spec: str):
    from .learners import LEARNERS  # imports torch

    name, kwargs = parse_spec(spec)
    if name not in LEARNERS:
        raise ValueError(f"unknown replay learner {name!r}; choose from {sorted(LEARNERS)}")
    return LEARNERS[name](**kwargs)
