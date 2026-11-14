"""Phase 1 search teacher: one-step heuristic lookahead over the policy-visible legal set.

Shared by `tools/evaluate_search_policy.py` (which plays the teacher as a diagnostic policy) and
`TribesGymWrapper` (which, with POLYVISION_TEACHER=lookahead_visible, reports the teacher's choice in `info`
for kickstarting PPO). Both use `select_lookahead_slot`, so the teacher plays identically in both places.

The scores come from `PythonEnv.lookaheadScores` (Java, `players/heuristics/Phase1SptHeuristic.java`), which
evaluates every raw Java action on a copy of the state. With `fullInformation=false` the copy has Bardur's fog
applied, so the teacher only knows what the player can see.

State copies get an unseeded java.util.Random, so actions with random outcomes (EXAMINE ruins, LEVEL_UP with an
EXPLORER) score differently on each call and the PR #11 evaluator is not reproducible run to run. The training
teacher therefore uses `PythonEnv.lookaheadScoresSeededPacked`, which reseeds each copy from
`decision_seed(episode_seed, decision_index)`; `tools/evaluate_search_policy.py --search-seeding per_decision`
plays the same seeded teacher.
"""
from __future__ import annotations

import hashlib
from typing import Mapping, Optional, Sequence

import numpy as np

# Order must match players.heuristics.Phase1SptHeuristic.FEATURE_NAMES.
FEATURE_NAMES = (
    "spt", "pop_progress", "stars_usable", "units_on_village", "village_proximity", "techs", "cities",
    "visible_tiles", "units", "unlocked_pop", "frontier_proximity",
)

# Hand-set weights. SPT is the target; one population point is worth a fraction of a city level (+1 SPT); stars,
# units and unlocked population are worth what they can still buy or reach before Turn 10; village distance and
# vision guide multi-turn movement.
DEFAULT_WEIGHTS = {
    "spt": 10.0, "pop_progress": 8.0, "stars_usable": 0.5, "units_on_village": 6.0,
    "village_proximity": 0.5, "techs": 0.0, "cities": 5.0, "visible_tiles": 15.0,
    "units": 1.5, "unlocked_pop": 0.5, "frontier_proximity": 0.3,
}

# Weights tuned on 60 TRAINING maps by tools/tune_search_heuristic.py --information visible, exactly as recorded in
# outputs/evaluations/20261112_lookahead_visible_tuned_validation250_visible-search-diagnostic/config.json
# (21.84 mean SPT on the 250 validation maps). Values are copied bit for bit (0.44999999999999996 included).
TUNED_VISIBLE_WEIGHTS = {
    "cities": 5.0, "frontier_proximity": 0.44999999999999996, "pop_progress": 20.0, "spt": 10.0,
    "stars_usable": -0.5, "techs": -0.5, "units": 1.5, "units_on_village": 3.75, "unlocked_pop": 0.75,
    "village_proximity": 0.75, "visible_tiles": 37.5,
}

# POLYVISION_TEACHER values -> (weights, fullInformation). Only the fog-respecting teacher is offered for training.
TEACHER_MODES = {
    "lookahead_visible": (TUNED_VISIBLE_WEIGHTS, False),
}
TEACHER_ENV_VAR = "POLYVISION_TEACHER"
TEACHER_SLOT_INFO_KEY = "teacher_slot"

LOOKAHEAD_EPS = 1e-9


def decision_seed(episode_seed: int, decision_index: int) -> int:
    """A reproducible 62-bit Java long for the teacher search at one decision of one episode."""
    digest = hashlib.sha256(f"phase1_teacher:{int(episode_seed)}:{int(decision_index)}".encode()).digest()
    return int.from_bytes(digest[:8], "big") >> 2


def lookahead_scores_seeded(java_env, java_weights, turns_remaining: int, full_information: bool, seed: int):
    """Seeded lookahead scores (np.float64, indexed by raw Java action) in one Py4J transfer."""
    packed = java_env.lookaheadScoresSeededPacked(java_weights, int(turns_remaining), bool(full_information), int(seed))
    return np.frombuffer(bytes(packed), dtype=">f8").astype(np.float64)


def parse_teacher_mode(raw: Optional[str]) -> Optional[str]:
    """Normalise a POLYVISION_TEACHER value: None/''/'0'/'none'/'off' -> None; unknown values raise."""
    value = str(raw if raw is not None else "").strip().lower()
    if value in ("", "0", "none", "off", "false", "no"):
        return None
    if value not in TEACHER_MODES:
        raise ValueError(f"Unknown {TEACHER_ENV_VAR}={raw!r}; expected one of {sorted(TEACHER_MODES)} or unset")
    return value


def java_weight_array(gateway, weights: Mapping[str, float]):
    """A Java double[] holding `weights` in FEATURE_NAMES order (every name is required)."""
    arr = gateway.new_array(gateway.jvm.double, len(FEATURE_NAMES))
    for i, k in enumerate(FEATURE_NAMES):
        arr[i] = float(weights[k])
    return arr


def select_lookahead_slot(
    scores: Sequence[float],
    ids: np.ndarray,
    valid: np.ndarray,
    is_end_turn: np.ndarray,
    raw_map: Mapping[int, int],
    eps: float = LOOKAHEAD_EPS,
) -> int:
    """The lookahead's chosen legal slot.

    scores: lookaheadScores output, indexed by raw Java action index. ids / valid / is_end_turn: per legal slot
    (global id, validity, END_TURN flag). raw_map: global id -> raw Java index for the current legal set.
    Picks the first slot with the strictly best score, but ends the turn unless some action strictly improves on
    stopping now (END_TURN scores the current state).
    """
    slots = np.flatnonzero(valid)
    end_slots = [int(s) for s in slots if is_end_turn[s]]
    best_slot, best_score = None, -np.inf
    for s in slots:
        raw = raw_map.get(int(ids[s]))
        if raw is None or raw >= len(scores):
            continue
        v = float(scores[raw])
        if v > best_score + eps:
            best_slot, best_score = int(s), v
    if end_slots:
        end_raw = raw_map.get(int(ids[end_slots[0]]))
        end_score = float(scores[end_raw]) if end_raw is not None and end_raw < len(scores) else -np.inf
        if best_slot is None or best_score <= end_score + eps:
            best_slot = end_slots[0]
    if best_slot is None:
        best_slot = int(slots[0])
    return int(best_slot)
