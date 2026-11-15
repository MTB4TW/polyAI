"""Privileged (full-information) critic for combat PPO (opt-in `--privileged-critic`).

Under fog of war the policy's own value head has to guess what it cannot see, so its advantage estimates are
noisy. The critic is only used during training (to compute advantages), never when the policy plays, so it may
see more than the policy: here it reads the same observation layout built from the true, unfogged game state,
which the env sends in info["privileged_obs"] when POLYVISION_PRIVILEGED_OBS=1 (combat only). This is the
"asymmetric actor-critic" used by AlphaStar (its value function saw the opponent's observations).

The policy network is unchanged and its checkpoint keeps exactly the same keys, so evaluation and every loader
work as before. The critic's own weights go to a separate training-only file next to the checkpoint
(`X.cleanrl_model.privileged_critic.pt`); losing it only means a resumed run retrains the critic.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from py_rl.cleanrl.cleanrl.obs_normalization import RunningObsNormalizer
from py_rl.cleanrl.cleanrl.spatial_agent import (
    BOARD_HEIGHT,
    BOARD_WIDTH,
    SpatialResNetBackbone,
    layer_init,
    observation_sizes,
)

PRIVILEGED_CRITIC_VERSION = "privileged_critic_v1"
PRIVILEGED_CRITIC_SUFFIX = ".privileged_critic.pt"


def privileged_critic_path(model_path: str) -> str:
    return f"{model_path}{PRIVILEGED_CRITIC_SUFFIX}"


class PrivilegedCritic(nn.Module):
    """V(s) from the unfogged observation: spatial ResNet + scalar MLP + value MLP (the spatial agent's critic
    shape, with its own encoder so the policy's encoder is trained by the policy loss alone)."""

    def __init__(self, obs_dim: int, *, norm_obs: bool = False, hidden_channels: int = 64):
        super().__init__()
        channels, scalar_dim = observation_sizes()
        spatial_dim = channels * BOARD_WIDTH * BOARD_HEIGHT
        if int(obs_dim) != spatial_dim + scalar_dim:
            raise ValueError(
                f"Privileged critic needs the 11x11 spatial observation ({spatial_dim} + {scalar_dim}); got {obs_dim}"
            )
        self.obs_dim = int(obs_dim)
        self.spatial_channels = channels
        self.spatial_dim = spatial_dim
        self.obs_normalizer = RunningObsNormalizer(self.obs_dim) if norm_obs else None
        self.spatial_backbone = SpatialResNetBackbone(in_channels=channels, hidden_channels=hidden_channels, out_dim=128)
        self.scalar_backbone = nn.Sequential(
            layer_init(nn.Linear(scalar_dim, 64)),
            nn.ReLU(inplace=True),
            layer_init(nn.Linear(64, 64)),
            nn.ReLU(inplace=True),
        )
        self.value_head = nn.Sequential(
            layer_init(nn.Linear(128 + 64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """[B, obs_dim] -> [B, 1]."""
        if self.obs_normalizer is not None:
            x = self.obs_normalizer(x)
        spatial = x[:, : self.spatial_dim].contiguous().view(-1, self.spatial_channels, BOARD_HEIGHT, BOARD_WIDTH)
        scalars = x[:, self.spatial_dim :].contiguous()
        h = torch.cat([self.spatial_backbone(spatial), self.scalar_backbone(scalars)], dim=-1)
        return self.value_head(h)

    @torch.no_grad()
    def values(self, x: torch.Tensor, chunk: int = 4096) -> torch.Tensor:
        """Batched no-grad V(s) for a whole rollout ([N, obs_dim] -> [N])."""
        out = [self(x[i : i + chunk]).view(-1) for i in range(0, x.shape[0], chunk)]
        return torch.cat(out) if out else x.new_zeros((0,))

    @torch.no_grad()
    def update_normalizer(self, x: torch.Tensor) -> None:
        if self.obs_normalizer is not None:
            self.obs_normalizer.update(x)


def extract_privileged_obs(infos, num_envs: int, obs_dim: int, device) -> torch.Tensor:
    """Stack info["privileged_obs"] from a vector-env info dict into [num_envs, obs_dim]. Every env must send it."""
    from pol_env.Tribes.py.environment_contract import PRIVILEGED_OBS_INFO_KEY

    if not isinstance(infos, dict) or PRIVILEGED_OBS_INFO_KEY not in infos:
        raise RuntimeError(
            "--privileged-critic: the env info has no 'privileged_obs' (POLYVISION_PRIVILEGED_OBS=1 not reaching the env?)"
        )
    present = infos.get("_" + PRIVILEGED_OBS_INFO_KEY, None)
    if present is not None and not np.asarray(present, dtype=bool).all():
        raise RuntimeError("--privileged-critic: some envs sent no privileged_obs this step")
    rows = infos[PRIVILEGED_OBS_INFO_KEY]
    arr = np.asarray(np.stack([np.asarray(r, dtype=np.float32) for r in rows]) if not isinstance(rows, np.ndarray) else rows,
                     dtype=np.float32)
    if arr.shape != (int(num_envs), int(obs_dim)):
        raise RuntimeError(f"--privileged-critic: privileged_obs batch shape {arr.shape} != {(num_envs, obs_dim)}")
    return torch.from_numpy(arr).to(device)
