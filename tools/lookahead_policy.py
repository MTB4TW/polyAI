"""Heuristic action reranking using policy logits and visible action features.

The historical class name remains for compatibility. This policy does not
simulate successor states or perform tree search.
"""

from __future__ import annotations

from typing import Any, Mapping
import numpy as np
import torch

from tools.phase1_eval_core import EvaluationPolicy, legal_tensors
from pol_env.Tribes.py.register_env import TribesGymWrapper


class TacticalLookaheadPolicy(EvaluationPolicy):
    """Action-feature heuristic reranking; no future states are simulated."""

    def __init__(
        self,
        agent,
        device: torch.device,
        spt_weight: float = 3.0,
        pop_weight: float = 1.5,
        reveal_weight: float = 0.5,
        cost_penalty: float = 0.2,
    ):
        self.agent = agent
        self.device = device
        self.spt_weight = spt_weight
        self.pop_weight = pop_weight
        self.reveal_weight = reveal_weight
        self.cost_penalty = cost_penalty
        self.name = "tactical_heuristic"

    def choose_action(self, obs: np.ndarray, info: Mapping[str, Any]) -> tuple[int, int]:
        ids, valid, feats = legal_tensors(info)
        x = torch.as_tensor(obs, dtype=torch.float32, device=self.device).reshape(1, -1)
        ids_t = torch.as_tensor(ids, dtype=torch.long, device=self.device).reshape(1, -1)
        valid_t = torch.as_tensor(valid, dtype=torch.bool, device=self.device).reshape(1, -1)
        feats_t = torch.as_tensor(feats, dtype=torch.float32, device=self.device).unsqueeze(0)

        with torch.inference_mode():
            dist = self.agent.get_action_distribution(
                x,
                legal_global_ids=ids_t,
                legal_action_valid_mask=valid_t,
                legal_action_features=feats_t,
            )
            base_logits = dist.logits.squeeze(0).cpu().numpy()

        # Extract tactical features across valid slots
        # feats shape: [max_legal_actions, 47]
        names = TribesGymWrapper.LEGAL_ACTION_FEATURE_NAMES
        reveal_signal = feats[:, names.index("newly_revealed_tiles_if_move_norm")]
        pop_gain = feats[:, names.index("expected_population_delta_norm")]
        spt_gain = feats[:, names.index("expected_immediate_spt_delta_norm")]
        action_cost = feats[:, names.index("action_star_cost_norm")]

        # Compute search-augmented tactical scores
        tactical_score = (
            base_logits
            + (self.spt_weight * spt_gain)
            + (self.pop_weight * pop_gain)
            + (self.reveal_weight * reveal_signal)
            - (self.cost_penalty * action_cost)
        )
        tactical_score[~valid] = -1e9

        chosen_slot = int(np.argmax(tactical_score))
        return int(ids[chosen_slot]), chosen_slot
