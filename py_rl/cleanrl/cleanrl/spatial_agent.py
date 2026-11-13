"""Spatial ResNet Agent for Polytopia (PolyVision).

Implements 2D multi-channel convolutional/residual spatial encoding
alongside scalar feature networks to preserve map topology, road connectivity,
and spatial adjacency on the 11x11 board grid.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical

SPATIAL_CHANNELS = 52
BOARD_WIDTH = 11
BOARD_HEIGHT = 11
SPATIAL_DIM = SPATIAL_CHANNELS * BOARD_WIDTH * BOARD_HEIGHT  # 6292
SCALAR_DIM = 132  # 6424 - 6292


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    if hasattr(layer, "weight") and layer.weight is not None:
        torch.nn.init.orthogonal_(layer.weight, std)
    if hasattr(layer, "bias") and layer.bias is not None:
        torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class ResBlock(nn.Module):
    """Basic 2D Residual Block with orthogonal initialization."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = layer_init(nn.Conv2d(channels, channels, kernel_size=3, padding=1))
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = layer_init(nn.Conv2d(channels, channels, kernel_size=3, padding=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        out = self.relu(out + residual)
        return out


class SpatialResNetBackbone(nn.Module):
    """Processes 52-channel 11x11 grid planes into a spatial embedding vector."""

    def __init__(self, in_channels: int = SPATIAL_CHANNELS, hidden_channels: int = 64, out_dim: int = 128):
        super().__init__()
        self.stem = nn.Sequential(
            layer_init(nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1)),
            nn.ReLU(inplace=True),
        )
        self.res1 = ResBlock(hidden_channels)
        self.res2 = ResBlock(hidden_channels)
        self.res3 = ResBlock(hidden_channels)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Sequential(
            layer_init(nn.Linear(hidden_channels, out_dim)),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, return_grid=False):
        # x: [B, 52, 11, 11]
        out = self.stem(x)
        out = self.res1(out)
        out = self.res2(out)
        out = self.res3(out)
        grid = out
        out = self.head(self.pool(out).flatten(1))
        return (out, grid) if return_grid else out


class SpatialResNetAgent(nn.Module):
    """Superhuman candidate agent architecture combining spatial ResNet and scalar MLP.

    Compatible with existing PolyVision action spaces and checkpoint loaders.
    """

    def __init__(
        self,
        envs=None,
        action_space_n: int = 63913,
        actor_mode: str = "legal_features",
        max_legal_actions: int = 256,
        legal_action_feature_dim: int = 47,
        spatial_channels: int = SPATIAL_CHANNELS,
        scalar_dim: int = SCALAR_DIM,
        local_action_features: bool = False,
        composed_actions: bool = False,
    ):
        super().__init__()
        # composed_actions replaces the per-ID embedding with action type + local grid features.
        self.composed_actions = bool(composed_actions)
        self.local_action_features = bool(local_action_features) or self.composed_actions
        if spatial_channels != SPATIAL_CHANNELS or scalar_dim != SCALAR_DIM:
            raise ValueError("Spatial agent requires the 52-channel 11x11 observation contract")
        self.actor_mode = str(actor_mode).strip().lower()
        if self.actor_mode not in ("legal_only", "legal_features"):
            raise ValueError(f"Unsupported actor_mode={actor_mode}")

        if self.local_action_features and self.actor_mode != "legal_features":
            raise ValueError("Local spatial action features require legal_features actor mode")
        if self.composed_actions and self.actor_mode != "legal_features":
            raise ValueError("Composed action representations require legal_features actor mode")
        self.max_legal_actions = int(max_legal_actions)
        self.legal_action_feature_dim = int(legal_action_feature_dim)
        n_actions = envs.single_action_space.n if envs is not None else action_space_n

        self.spatial_backbone = SpatialResNetBackbone(in_channels=spatial_channels, hidden_channels=64, out_dim=128)
        self.scalar_backbone = nn.Sequential(
            layer_init(nn.Linear(scalar_dim, 64)),
            nn.ReLU(inplace=True),
            layer_init(nn.Linear(64, 64)),
            nn.ReLU(inplace=True),
        )

        # Combined state representation
        combined_dim = 128 + 64  # 192
        self.state_fusion = nn.Sequential(
            layer_init(nn.Linear(combined_dim, 64)),
            nn.Tanh(),
        )

        # Critic evaluates combined spatial + scalar state
        self.critic = nn.Sequential(
            layer_init(nn.Linear(combined_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )

        # Action representations
        if self.composed_actions:
            # No per-ID table: an action is its type (catalog block) plus the grid features at
            # its source/target tiles plus the hand-made legal-action features.
            from py_rl.cleanrl.cleanrl.action_types import action_type_table_for_action_space

            table = action_type_table_for_action_space(n_actions, BOARD_WIDTH, BOARD_HEIGHT)
            self.action_type_names = [name for name, _, _ in table]
            self.register_buffer("action_type_starts", torch.tensor([s for _, s, _ in table], dtype=torch.long))
            self.register_buffer("action_type_has_tile", torch.tensor([t for _, _, t in table], dtype=torch.bool))
            self.action_embedding = None
            self.action_type_embedding = nn.Embedding(len(table), 64)
            nn.init.normal_(self.action_type_embedding.weight, mean=0.0, std=0.02)
        else:
            self.action_embedding = nn.Embedding(n_actions, 64)
            nn.init.normal_(self.action_embedding.weight, mean=0.0, std=0.02)

        if self.actor_mode == "legal_features":
            self.action_feature_encoder = nn.Sequential(
                layer_init(nn.Linear(self.legal_action_feature_dim, 32)),
                nn.Tanh(),
                layer_init(nn.Linear(32, 32)),
                nn.Tanh(),
            )
            self.action_scorer = nn.Sequential(
                layer_init(nn.Linear(64 + 64 + 32 + (128 if self.local_action_features else 0), 64)),
                nn.Tanh(),
                layer_init(nn.Linear(64, 1), std=0.01),
            )
        else:
            self.action_feature_encoder = None
            self.action_scorer = None

    def encode_state(self, x: torch.Tensor, return_grid=False):
        """Splits raw 6424-dim obs into spatial grid and scalars, returning fused representation and combined embedding."""
        spatial_raw = x[:, :SPATIAL_DIM].contiguous().view(-1, SPATIAL_CHANNELS, BOARD_HEIGHT, BOARD_WIDTH)
        scalar_raw = x[:, SPATIAL_DIM:].contiguous()

        spatial_emb, grid = self.spatial_backbone(spatial_raw, return_grid=True)
        scalar_emb = self.scalar_backbone(scalar_raw)
        combined = torch.cat([spatial_emb, scalar_emb], dim=-1)
        h = self.state_fusion(combined)
        return (h, combined, grid) if return_grid else (h, combined)

    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        _, combined = self.encode_state(x)
        return self.critic(combined)

    def get_action_distribution(
        self,
        x: torch.Tensor,
        action_mask=None,
        legal_global_ids=None,
        legal_action_valid_mask=None,
        legal_action_features=None,
    ) -> Categorical:
        h, _, grid = self.encode_state(x, return_grid=True)
        return self._distribution(h, grid, legal_global_ids, legal_action_valid_mask, legal_action_features)

    @staticmethod
    def gather_local_features(grid, features):
        # Observation planes are flattened as [x, y], not [y, x].
        coords = features[..., 43:47].clamp(0, 1)
        xy = (coords * (BOARD_WIDTH - 1)).round().long()
        flat = grid.flatten(2).transpose(1, 2)
        parts = []
        for offset in (0, 2):
            index = xy[..., offset] * BOARD_HEIGHT + xy[..., offset + 1]
            parts.append(flat.gather(1, index.unsqueeze(-1).expand(-1, -1, flat.shape[-1])))
        # Missing coordinates share (0,0) in the established feature contract;
        # action-type features allow the scorer to distinguish global actions.
        return torch.cat(parts, dim=-1)

    def action_types(self, legal_ids):
        """Map global action IDs to action-type indices (catalog block containing each ID)."""
        ids = legal_ids.long().clamp(min=0).contiguous()
        return torch.bucketize(ids, self.action_type_starts, right=True) - 1

    def _distribution(self, h, grid, legal_global_ids, legal_action_valid_mask, legal_action_features):
        legal_ids = legal_global_ids.long()
        valid = legal_action_valid_mask.bool()
        if not valid.any(dim=-1).all():
            raise ValueError("Each policy decision requires at least one valid legal action")
        if self.composed_actions:
            types = self.action_types(legal_ids)
            legal_emb = self.action_type_embedding(types)
        else:
            legal_emb = self.action_embedding(legal_ids)
        if self.actor_mode == "legal_only":
            logits = torch.einsum("bd,bkd->bk", h, legal_emb)
        else:
            features = legal_action_features.float()
            feat_emb = self.action_feature_encoder(features)
            parts = [h.unsqueeze(1).expand(-1, legal_ids.shape[1], -1), legal_emb, feat_emb]
            if self.composed_actions:
                # Tile-less actions (END_TURN, RESEARCH_TECH) have zero coordinates; do not let
                # them read tile (0,0).
                local = self.gather_local_features(grid, features)
                parts.append(local * self.action_type_has_tile[types].unsqueeze(-1).to(local.dtype))
            elif self.local_action_features:
                parts.append(self.gather_local_features(grid, features))
            logits = self.action_scorer(torch.cat(parts, dim=-1)).squeeze(-1)
        return Categorical(logits=logits.masked_fill(~valid, -1e8))

    def get_action_and_value(
        self,
        x: torch.Tensor,
        action=None,
        action_mask=None,
        legal_global_ids=None,
        legal_action_valid_mask=None,
        legal_action_features=None,
        selected_slot=None,
        return_distribution=False,
    ):
        """return_distribution=True appends the policy distribution (over legal slots) as a sixth value."""
        h, combined, grid = self.encode_state(x, return_grid=True)
        value = self.critic(combined)

        legal_ids = legal_global_ids.long()
        valid = legal_action_valid_mask.bool()

        probs = self._distribution(h, grid, legal_global_ids, legal_action_valid_mask, legal_action_features)

        if selected_slot is None:
            selected_slot = probs.sample()
        else:
            selected_slot = selected_slot.long()
        if selected_slot.ndim != 1:
            selected_slot = selected_slot.view(-1)

        selected_global_id = legal_ids.gather(1, selected_slot.unsqueeze(1)).squeeze(1)
        out = (selected_global_id, selected_slot, probs.log_prob(selected_slot), probs.entropy(), value)
        return out + (probs,) if return_distribution else out
