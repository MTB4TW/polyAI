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

PRIVILEGED_CRITIC_VERSION = "privileged_critic_v2"
PRIVILEGED_CRITIC_SUFFIX = ".privileged_critic.pt"


def privileged_critic_path(model_path: str) -> str:
    return f"{model_path}{PRIVILEGED_CRITIC_SUFFIX}"


def load_privileged_critic_for_init(critic: nn.Module, init_from: str, map_location="cpu") -> bool:
    """With --init-from X: load the critic saved next to X (X.privileged_critic.pt) into `critic`.

    Returns True if it was loaded, False if X has no critic file (the critic then stays freshly initialised).
    Loading is strict: missing or unexpected keys (e.g. --norm-obs differing from the source run) and mismatched
    shapes raise instead of being skipped. Prints one line either way.
    """
    import os

    path = privileged_critic_path(init_from)
    if not os.path.exists(path):
        print(f"[PRIVILEGED_CRITIC] init: no saved critic at {path}; starting a fresh critic")
        return False
    state = torch.load(path, map_location=map_location)
    if not isinstance(state, dict):
        raise RuntimeError(f"[PRIVILEGED_CRITIC] {path} is not a state dict")
    own = critic.state_dict()
    stat_keys = ("return_mean", "return_sq_mean", "return_stats_initialised")
    has_stats = all(k in state for k in stat_keys)
    run_norm = bool(getattr(critic, "value_norm", False))
    if run_norm and not any(k in state for k in stat_keys):
        # Saved before value normalisation (CHANGELOG 071): its head emits raw returns. Mean 0 / std 1 (the std
        # floor) gives V exactly as saved; marking the statistics initialised makes the first PopArt update
        # preserve those outputs instead of overwriting them.
        state = dict(state)
        state["return_mean"] = torch.zeros((), dtype=torch.float32)
        state["return_sq_mean"] = torch.ones((), dtype=torch.float32)
        state["return_stats_initialised"] = torch.ones((), dtype=torch.float32)
        print(f"[PRIVILEGED_CRITIC] init: {path} has no value-normalisation statistics; starting them at mean 0, std 1")
    elif not run_norm and has_stats:
        # Saved with value normalisation, loaded without: fold mean/std into the output layer.
        state = dict(state)
        mean = state.pop("return_mean").float()
        var = (state.pop("return_sq_mean").float() - mean ** 2).clamp_min(0.0)
        std = var.sqrt().clamp_min(float(getattr(critic, "value_norm_min_std", 1.0)))
        state.pop("return_stats_initialised")
        state["value_head.4.weight"] = state["value_head.4.weight"] * std
        state["value_head.4.bias"] = state["value_head.4.bias"] * std + mean
        print(f"[PRIVILEGED_CRITIC] init: folded saved value-normalisation statistics into the output layer")
    bad = [f"{k}: saved {tuple(v.shape)} vs run {tuple(own[k].shape)}" for k, v in state.items()
           if k in own and hasattr(v, "shape") and tuple(v.shape) != tuple(own[k].shape)]
    if bad:
        raise RuntimeError(f"[PRIVILEGED_CRITIC] {path}: shape mismatch: " + "; ".join(bad))
    critic.load_state_dict(state, strict=True)  # raises on missing / unexpected keys
    n = sum(int(v.numel()) for v in state.values() if hasattr(v, "numel"))
    print(f"[PRIVILEGED_CRITIC] init: loaded saved critic from {path} ({len(state)} tensors, {n} values)")
    return True


class PrivilegedCritic(nn.Module):
    """V(s) from the unfogged observation: spatial ResNet + scalar MLP + value MLP (the spatial agent's critic
    shape, with its own encoder so the policy's encoder is trained by the policy loss alone)."""

    def __init__(self, obs_dim: int, *, norm_obs: bool = False, hidden_channels: int = 64, value_norm: bool = True,
                 value_norm_beta: float = 0.1, value_norm_min_std: float = 1.0):
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
        self.hidden_channels = int(hidden_channels)
        self.obs_normalizer = RunningObsNormalizer(self.obs_dim) if norm_obs else None
        # Value normalisation (PopArt, van Hasselt et al. 2016): the network predicts standardised returns and
        # V = mean + std * output. Without it the head has to emit raw returns (about -25 to +100 in combat) from
        # unit-scale activations, which drives its tanh layers into saturation and the critic goes flat.
        self.value_norm = bool(value_norm)
        self.value_norm_beta = float(value_norm_beta)
        self.value_norm_min_std = float(value_norm_min_std)
        if self.value_norm:
            self.register_buffer("return_mean", torch.zeros((), dtype=torch.float32))
            self.register_buffer("return_sq_mean", torch.ones((), dtype=torch.float32))
            self.register_buffer("return_stats_initialised", torch.zeros((), dtype=torch.float32))
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

    def extra_init_kwargs(self) -> dict:
        return {"value_norm": self.value_norm, "value_norm_beta": self.value_norm_beta,
                "value_norm_min_std": self.value_norm_min_std}

    @staticmethod
    def is_statistic_key(key: str) -> bool:
        """State-dict entries that are running statistics, not learned weights."""
        return key.startswith("obs_normalizer.") or key.startswith("return_")

    def return_std(self) -> torch.Tensor:
        if not self.value_norm:
            return torch.ones((), device=next(self.parameters()).device)
        var = (self.return_sq_mean - self.return_mean ** 2).clamp_min(0.0)
        return var.sqrt().clamp_min(self.value_norm_min_std)

    def normalized(self, x: torch.Tensor) -> torch.Tensor:
        """[B, obs_dim] -> [B, 1] standardised value (the raw value when value normalisation is off)."""
        if self.obs_normalizer is not None:
            x = self.obs_normalizer(x)
        spatial = x[:, : self.spatial_dim].contiguous().view(-1, self.spatial_channels, BOARD_HEIGHT, BOARD_WIDTH)
        scalars = x[:, self.spatial_dim :].contiguous()
        h = torch.cat([self.spatial_backbone(spatial), self.scalar_backbone(scalars)], dim=-1)
        return self.value_head(h)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """[B, obs_dim] -> [B, 1] value in return units."""
        out = self.normalized(x)
        if self.value_norm:
            out = self.return_mean + self.return_std() * out
        return out

    def normalize_returns(self, returns: torch.Tensor) -> torch.Tensor:
        if not self.value_norm:
            return returns
        return (returns - self.return_mean) / self.return_std()

    def value_loss(self, x: torch.Tensor, returns: torch.Tensor) -> torch.Tensor:
        """0.5 * mean squared error, measured in standardised units under value normalisation."""
        pred = self.normalized(x).view(-1)
        return 0.5 * ((pred - self.normalize_returns(returns.view(-1))) ** 2).mean()

    @torch.no_grad()
    def update_return_stats(self, returns: torch.Tensor) -> None:
        """Move the return statistics towards this rollout's returns and rescale the output layer so that every
        V(s) is unchanged (PopArt's "preserving outputs"). Call once per rollout, before the PPO update."""
        if not self.value_norm:
            return
        r = returns.detach().reshape(-1).double()
        if r.numel() == 0:
            return
        b_mean, b_sq = r.mean().float(), (r ** 2).mean().float()
        old_mean, old_std = self.return_mean.clone(), self.return_std().clone()
        if float(self.return_stats_initialised) == 0.0:
            # First rollout: take its statistics as they are, without preserving outputs. The fresh head's outputs
            # are unit-scale noise, which is what a standardised value should start as; preserving them would turn
            # the raw return offset (about -12 in combat) into a large standardised one the head must undo.
            self.return_mean.copy_(b_mean)
            self.return_sq_mean.copy_(b_sq)
            self.return_stats_initialised.fill_(1.0)
            return
        beta = self.value_norm_beta
        self.return_mean.mul_(1.0 - beta).add_(beta * b_mean)
        self.return_sq_mean.mul_(1.0 - beta).add_(beta * b_sq)
        new_mean, new_std = self.return_mean, self.return_std()
        last = self.value_head[-1]
        last.weight.mul_(old_std / new_std)
        last.bias.mul_(old_std).add_(old_mean - new_mean).div_(new_std)

    @torch.no_grad()
    def diagnostics(self, x: torch.Tensor) -> dict:
        """Health of the critic's features on a batch (a collapsed critic gives the same output for every state)."""
        if self.obs_normalizer is not None:
            x = self.obs_normalizer(x)
        spatial = x[:, : self.spatial_dim].contiguous().view(-1, self.spatial_channels, BOARD_HEIGHT, BOARD_WIDTH)
        s = self.spatial_backbone(spatial)
        c = self.scalar_backbone(x[:, self.spatial_dim :].contiguous())
        h = torch.cat([s, c], dim=-1)
        pre1 = self.value_head[0](h)
        a1 = torch.tanh(pre1)
        pre2 = self.value_head[2](a1)
        a2 = torch.tanh(pre2)
        v = self.value_head[4](a2).view(-1) * (self.return_std() if self.value_norm else 1.0)
        return {
            "spatial_dead_frac": float((s.amax(dim=0) <= 0).float().mean()),
            "scalar_dead_frac": float((c.amax(dim=0) <= 0).float().mean()),
            "feature_std": float(h.std(dim=0).mean()),
            "tanh1_sat_frac": float((a1.abs() > 0.99).float().mean()),
            "tanh2_sat_frac": float((a2.abs() > 0.99).float().mean()),
            "hidden_std": float(a2.std(dim=0).mean()),
            "value_std": float(v.std()),
        }

    @torch.no_grad()
    def values(self, x: torch.Tensor, chunk: int = 4096) -> torch.Tensor:
        """Batched no-grad V(s) for a whole rollout ([N, obs_dim] -> [N])."""
        out = [self(x[i : i + chunk]).view(-1) for i in range(0, x.shape[0], chunk)]
        return torch.cat(out) if out else x.new_zeros((0,))

    @torch.no_grad()
    def update_normalizer(self, x: torch.Tensor) -> None:
        if self.obs_normalizer is not None:
            self.obs_normalizer.update(x)


class FlatCriticGuard:
    """Detects a collapsed critic: its predictions have (almost) no spread across the rollout while the returns do.

    `observe()` is called once per PPO update with the std of the rollout's value predictions and of its returns;
    it returns True on the update where the critic has been flat for `patience` updates in a row."""

    def __init__(self, patience: int, min_value_std: float = 1e-3, min_return_std: float = 1e-2):
        self.patience = int(patience)
        self.min_value_std = float(min_value_std)
        self.min_return_std = float(min_return_std)
        self.flat_updates = 0

    def is_flat(self, value_std: float, return_std: float) -> bool:
        return value_std < self.min_value_std and return_std > self.min_return_std

    def observe(self, value_std: float, return_std: float) -> bool:
        if self.patience <= 0:
            return False
        self.flat_updates = self.flat_updates + 1 if self.is_flat(value_std, return_std) else 0
        if self.flat_updates >= self.patience:
            self.flat_updates = 0
            return True
        return False


def reinitialize_critic(critic: PrivilegedCritic, optimizer: torch.optim.Optimizer | None = None, seed: int | None = None) -> None:
    """Fresh weights for every learned layer (normaliser statistics kept) and no optimiser state for them, so Adam
    restarts its moment estimates for the critic. Parameter tensors are updated in place, so the optimiser keeps
    pointing at them."""
    if seed is not None:
        torch.manual_seed(int(seed))
    fresh = PrivilegedCritic(critic.obs_dim, norm_obs=critic.obs_normalizer is not None,
                             hidden_channels=critic.hidden_channels, **critic.extra_init_kwargs())
    current = critic.state_dict()
    keep = {k: v for k, v in current.items() if critic.is_statistic_key(k)}
    state = fresh.state_dict()
    state.update(keep)
    with torch.no_grad():
        critic.load_state_dict({k: v.to(current[k].device) for k, v in state.items()})
    if optimizer is not None:
        for p in critic.parameters():
            optimizer.state.pop(p, None)


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
