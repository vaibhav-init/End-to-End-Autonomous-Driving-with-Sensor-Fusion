"""Point-set baselines for per-detection multipath classification."""

import torch
from torch import nn

from .features import FEATURE_NAMES


class PointMLP(nn.Module):
    """Independent per-detection baseline with no spatial/temporal context."""

    def __init__(self, input_dim=len(FEATURE_NAMES), hidden_dim=96, dropout=0.1):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features, point_mask=None):
        batch, points, channels = features.shape
        logits = self.network(features.reshape(batch * points, channels))
        return logits.reshape(batch, points)


class TemporalPointNet(nn.Module):
    """Per-point classifier conditioned on a temporal point-set summary."""

    def __init__(
        self,
        input_dim=len(FEATURE_NAMES),
        hidden_dim=128,
        context_dim=192,
        dropout=0.15,
    ):
        super().__init__()
        self.local = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, context_dim),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(context_dim * 2, context_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(context_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features, point_mask=None):
        local = self.local(features)
        if point_mask is None:
            context = local.max(dim=1, keepdim=True).values
        else:
            masked = local.masked_fill(~point_mask.unsqueeze(-1), -1.0e9)
            context = masked.max(dim=1, keepdim=True).values
        context = context.expand(-1, local.shape[1], -1)
        return self.head(torch.cat((local, context), dim=-1)).squeeze(-1)


def create_ghost_model(model_name, **kwargs):
    if model_name == "point_mlp":
        return PointMLP(**kwargs)
    if model_name == "temporal_pointnet":
        return TemporalPointNet(**kwargs)
    raise ValueError(f"Unknown ghost detector model {model_name!r}")
