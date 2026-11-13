"""Observation normalisation that travels with the checkpoint (opt-in `--norm-obs`).

`ObsNormalizedAgent` wraps any policy from `agent_factory.make_agent`. It owns a
`RunningObsNormalizer` whose statistics are registered buffers, so they are saved in the
checkpoint's state_dict and restored by `load_state_dict`:

    policy.*              the wrapped network's parameters (unchanged names, prefixed)
    obs_normalizer.mean   per-dimension running mean           (float32, [obs_dim])
    obs_normalizer.var    per-dimension running variance       (float32, [obs_dim])
    obs_normalizer.count  number of observations merged so far (float32, scalar)
    obs_normalizer.epsilon, obs_normalizer.clip               (float32 scalars)

The normaliser only changes when the trainer calls `update()` explicitly (once per PPO iteration,
after the update), so rollouts, the PPO update and evaluation all see fixed statistics.

The sidecar records `"observation_normalization": OBS_NORMALIZATION_VERSION`; a sidecar without the
field means no normalisation (every checkpoint before this change). A normalised checkpoint loaded
without the wrapper, or a plain checkpoint loaded with it, fails `load_state_dict(strict=True)`
because the key sets differ, so mismatches are rejected rather than silently mis-evaluated.
"""

from typing import Mapping, Optional

import torch
import torch.nn as nn

OBS_NORMALIZATION_VERSION = "running_mean_std_v1"
SIDECAR_FIELD = "observation_normalization"
SUPPORTED_OBS_NORMALIZATIONS = (None, OBS_NORMALIZATION_VERSION)


def observation_normalization_from_metadata(meta: Optional[Mapping]) -> Optional[str]:
    """Return the normalisation recorded in a checkpoint sidecar (None if absent); reject unknown values."""
    if not meta:
        return None
    value = meta.get(SIDECAR_FIELD)
    if value in (None, "", "none"):
        return None
    if value not in SUPPORTED_OBS_NORMALIZATIONS:
        raise ValueError(
            f"Checkpoint uses observation normalisation {value!r}, which this code does not support "
            f"(supported: {OBS_NORMALIZATION_VERSION!r})."
        )
    return value


class RunningObsNormalizer(nn.Module):
    """(x - mean) / sqrt(var + epsilon), clipped to [-clip, clip]; statistics merged in float64."""

    def __init__(self, obs_dim: int, epsilon: float = 1e-8, clip: float = 10.0, initial_count: float = 1e-4):
        super().__init__()
        self.register_buffer("mean", torch.zeros(int(obs_dim), dtype=torch.float32))
        self.register_buffer("var", torch.ones(int(obs_dim), dtype=torch.float32))
        self.register_buffer("count", torch.tensor(float(initial_count), dtype=torch.float32))
        self.register_buffer("epsilon", torch.tensor(float(epsilon), dtype=torch.float32))
        self.register_buffer("clip", torch.tensor(float(clip), dtype=torch.float32))

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        x = x.reshape(-1, self.mean.shape[0]).float()
        n = x.shape[0]
        if n == 0:
            return
        batch_mean = x.mean(dim=0).cpu().double()
        batch_var = x.var(dim=0, unbiased=False).cpu().double()
        mean = self.mean.cpu().double()
        var = self.var.cpu().double()
        count = float(self.count.item())
        total = count + n
        delta = batch_mean - mean
        new_mean = mean + delta * (n / total)
        m2 = var * count + batch_var * n + delta.pow(2) * (count * n / total)
        self.mean.copy_(new_mean.float())
        self.var.copy_((m2 / total).float())
        self.count.fill_(total)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normed = (x - self.mean) / torch.sqrt(self.var + self.epsilon)
        return torch.clamp(normed, -self.clip, self.clip)


class ObsNormalizedAgent(nn.Module):
    """Normalises the observation argument, then calls the wrapped policy.

    Wrapping (rather than patching the policy's methods) keeps the policy's internal calls, e.g.
    get_action_and_value -> get_action_distribution, from normalising twice. Other attributes are
    delegated to the wrapped policy.
    """

    _OBS_METHODS = ("get_value", "get_action_distribution", "get_action_and_value", "encode_state")

    def __init__(self, policy: nn.Module, obs_dim: int, epsilon: float = 1e-8, clip: float = 10.0):
        super().__init__()
        self.policy = policy
        self.obs_normalizer = RunningObsNormalizer(obs_dim, epsilon=epsilon, clip=clip)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return self.obs_normalizer(x)

    def get_value(self, x, *args, **kwargs):
        return self.policy.get_value(self.normalize(x), *args, **kwargs)

    def get_action_distribution(self, x, *args, **kwargs):
        return self.policy.get_action_distribution(self.normalize(x), *args, **kwargs)

    def get_action_and_value(self, x, *args, **kwargs):
        return self.policy.get_action_and_value(self.normalize(x), *args, **kwargs)

    def encode_state(self, x, *args, **kwargs):
        return self.policy.encode_state(self.normalize(x), *args, **kwargs)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            pass
        value = getattr(super().__getattr__("policy"), name)
        # Plain attributes (actor_mode, max_legal_actions, ...) are delegated. Submodules and other
        # methods are not: calling them directly would skip normalisation, so fail loudly instead.
        if callable(value):
            raise AttributeError(
                f"{name!r} is not exposed by ObsNormalizedAgent (calling it would bypass observation "
                f"normalisation); use agent.policy.{name} on agent.normalize(obs) explicitly."
            )
        return value
