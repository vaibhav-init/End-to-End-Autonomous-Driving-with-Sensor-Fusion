#!/usr/bin/env python3
"""
Shared target-speed model utilities.
"""

import torch
import torch.nn as nn


BASE_FEATURE_COLS = [
    "ego_speed",
    "ego_acceleration",
    "distance",
    "relative_velocity",
    "ttc",
    "obstacle_speed",
    "traffic_light_state",
    "tl_confidence",
    "tl_bbox_area",
    "tl_center_x",
]


def feature_sort_key(name):
    base, lag = name.rsplit("_t-", 1)
    return int(lag), base


def flatten_history(history, base_cols):
    row = {}
    frames = list(history)
    for lag, feature_dict in enumerate(reversed(frames)):
        for name in base_cols:
            row[f"{name}_t-{lag}"] = feature_dict[name]
    return row


class TargetSpeedMLP(nn.Module):
    """
    Camera encoder plus speed head.

    The encoder output is kept explicit so a later radar branch can be fused
    into the head without reworking the rest of the training/inference code.
    """

    def __init__(self, input_dim, encoder_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(128, encoder_dim),
            nn.ReLU(),
        )
        self.speed_head = nn.Linear(encoder_dim, 1)

    def encode(self, x):
        return self.encoder(x)

    def forward(self, x):
        embedding = self.encode(x)
        return self.speed_head(embedding).squeeze(-1)
