"""Kickstarting PPO from a search teacher (Schmitt et al. 2018, "Kickstarting Deep Reinforcement Learning").

With `--teacher lookahead_visible`, every Phase 1 env worker reports in `info["teacher_slot"]` the legal slot the
fog-respecting one-step lookahead would choose (pol_env/Tribes/py/search_teacher.py). The PPO loss gains

    coef(t) * mean_over_labelled_steps( -log pi(teacher_slot | s) )

i.e. the cross-entropy between the policy's distribution over legal slots and the teacher's one-hot choice, with
coef(t) decaying linearly from --teacher-coef to 0 over the first --teacher-decay-frac of training. Training only:
the network, checkpoint format and evaluation are unchanged.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from pol_env.Tribes.py.search_teacher import TEACHER_ENV_VAR, TEACHER_SLOT_INFO_KEY, parse_teacher_mode

NO_TEACHER_SLOT = -1


def validate_teacher_args(teacher: Optional[str], teacher_coef: float, teacher_decay_frac: float,
                          actor_mode: str, combat: bool) -> Optional[str]:
    """Normalised teacher mode (None when off); raises on settings that cannot work."""
    mode = parse_teacher_mode(teacher)
    if mode is None:
        return None
    if combat:
        raise RuntimeError("--teacher is Phase 1 only (the search heuristic scores Phase 1 SPT); unset it for combat")
    if actor_mode not in ("legal_only", "legal_features"):
        raise RuntimeError("--teacher needs a legal-slot actor mode (legal_only or legal_features)")
    if not np.isfinite(teacher_coef) or teacher_coef < 0:
        raise RuntimeError("--teacher-coef must be a finite number >= 0")
    if not np.isfinite(teacher_decay_frac) or teacher_decay_frac < 0:
        raise RuntimeError("--teacher-decay-frac must be >= 0 (0 keeps the coefficient constant)")
    return mode


def teacher_coef_at(iteration: int, num_iterations: int, coef: float, decay_frac: float) -> float:
    """Coefficient for 1-based `iteration`: linear from `coef` at the start to 0 at `decay_frac` of training.

    decay_frac <= 0 keeps it constant; decay_frac >= 1 decays over the whole run (reaching 0 only after it).
    """
    if coef <= 0:
        return 0.0
    if decay_frac <= 0:
        return float(coef)
    progress = (int(iteration) - 1) / max(1, int(num_iterations))
    return float(coef) * max(0.0, 1.0 - progress / float(decay_frac))


def extract_teacher_slots(infos, num_envs: int, legal_action_valid_mask: torch.Tensor) -> torch.Tensor:
    """Per-env teacher slot (long, NO_TEACHER_SLOT where an env reported none) from vector-env infos.

    Uses the current (post-autoreset) infos, never final_info: the teacher labels the state the policy acts on.
    A reported slot that is out of range or points at an invalid/padded slot is a bug and raises.
    """
    out = np.full((num_envs,), NO_TEACHER_SLOT, dtype=np.int64)
    if isinstance(infos, dict) and TEACHER_SLOT_INFO_KEY in infos:
        raw = np.asarray(infos[TEACHER_SLOT_INFO_KEY]).reshape(-1)
        present = infos.get("_" + TEACHER_SLOT_INFO_KEY)
        present = np.ones((num_envs,), dtype=bool) if present is None else np.asarray(present, dtype=bool).reshape(-1)
        if raw.shape[0] != num_envs or present.shape[0] != num_envs:
            raise RuntimeError(f"{TEACHER_SLOT_INFO_KEY} has {raw.shape[0]} entries for {num_envs} envs")
        for i in range(num_envs):
            if present[i] and raw[i] is not None:
                out[i] = int(raw[i])
    slots = torch.as_tensor(out, dtype=torch.long, device=legal_action_valid_mask.device)
    labelled = slots >= 0
    if bool(labelled.any()):
        if bool((slots[labelled] >= legal_action_valid_mask.shape[1]).any()):
            raise RuntimeError("teacher_slot out of range of the legal slot tensor")
        ok = legal_action_valid_mask.bool().gather(1, slots.clamp(min=0).unsqueeze(1)).squeeze(1)
        if not bool(ok[labelled].all()):
            raise RuntimeError("teacher_slot points at an invalid/padded legal slot")
    return slots


def kickstart_cross_entropy(dist: torch.distributions.Categorical, teacher_slots: torch.Tensor):
    """(mean -log pi(teacher_slot) over labelled rows, number of labelled rows). Mean is 0 when none are labelled."""
    labelled = teacher_slots >= 0
    n = int(labelled.sum().item())
    if n == 0:
        return dist.logits.sum() * 0.0, 0
    logp = dist.log_prob(teacher_slots.clamp(min=0))
    return -(logp[labelled]).mean(), n


__all__ = [
    "NO_TEACHER_SLOT", "TEACHER_ENV_VAR", "TEACHER_SLOT_INFO_KEY", "extract_teacher_slots",
    "kickstart_cross_entropy", "teacher_coef_at", "validate_teacher_args",
]
