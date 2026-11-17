"""Construct policies using recorded architecture metadata (legacy defaults to MLP)."""
def make_agent(envs, *, network_arch="mlp", observation_normalization=None, **kwargs):
    """Build the policy for `network_arch`.

    observation_normalization: None (default, every checkpoint before 053) or
    obs_normalization.OBS_NORMALIZATION_VERSION, which wraps the policy in ObsNormalizedAgent so the
    normalisation statistics are part of the state_dict. Pass the value recorded in the checkpoint
    sidecar (see obs_normalization.observation_normalization_from_metadata).
    """
    agent = _make_policy(envs, network_arch=network_arch, **kwargs)
    if observation_normalization is None:
        return agent
    from py_rl.cleanrl.cleanrl.obs_normalization import OBS_NORMALIZATION_VERSION, ObsNormalizedAgent
    if observation_normalization != OBS_NORMALIZATION_VERSION:
        raise ValueError(f"Unknown observation normalisation: {observation_normalization!r}")
    import numpy as np
    obs_dim = int(np.prod(envs.single_observation_space.shape))
    return ObsNormalizedAgent(agent, obs_dim)


def _make_policy(envs, *, network_arch="mlp", **kwargs):
    if network_arch == "mlp":
        from py_rl.cleanrl.cleanrl.ppo import Agent
        return Agent(envs, **kwargs)
    if network_arch in ("spatial_resnet", "spatial_resnet_local"):
        from py_rl.cleanrl.cleanrl.spatial_agent import SpatialResNetAgent
        return SpatialResNetAgent(envs, local_action_features=network_arch == "spatial_resnet_local", **kwargs)
    raise ValueError(f"Unknown network architecture: {network_arch}")
