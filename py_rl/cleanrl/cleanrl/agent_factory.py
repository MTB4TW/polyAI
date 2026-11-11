"""Construct policies using recorded architecture metadata (legacy defaults to MLP)."""
def make_agent(envs, *, network_arch="mlp", **kwargs):
    if network_arch == "mlp":
        from py_rl.cleanrl.cleanrl.ppo import Agent
        return Agent(envs, **kwargs)
    if network_arch in ("spatial_resnet", "spatial_resnet_local"):
        from py_rl.cleanrl.cleanrl.spatial_agent import SpatialResNetAgent
        return SpatialResNetAgent(envs, local_action_features=network_arch == "spatial_resnet_local", **kwargs)
    raise ValueError(f"Unknown network architecture: {network_arch}")
