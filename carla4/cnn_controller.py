#!/usr/bin/env python3
"""Target-speed CNN over a range-azimuth occupancy grid.

The transformer sees the detection list as an unordered bag of points and has
to learn every spatial relation from scratch. A multipath ghost is only
identifiable *relative to its parent*: same bearing, longer range, weaker,
with a Doppler that is a scaled copy. In a range-azimuth grid that relation is
a fixed offset between two cells, which is exactly the pattern a convolution
kernel detects for free. Range-azimuth rather than Cartesian precisely because
the ghost sits at the same bearing, so the relation is translation-invariant
along one axis.

It also removes the transformer's point budget. The grid holds every point of
every scan, so the parent can never be dropped while the ghost is kept.

Output is identical to the transformer's: one scalar, the change in target
speed, into the same PID. That is what keeps the arms comparable.

Everything except the model class works without torch so the rasteriser can be
unit tested on a machine with no GPU.
"""

import json
import os

import numpy as np

from radar.detection_log import DETECTION_DTYPE
from radar.ghost_detection.features import (
    frame_context_statistics,
    snr_db_to_amplitude,
)
from transformer_controller import SOURCE_CODES, _scan_columns

MODEL_TYPE = "cnn"
CHECKPOINT_NAME = "target_speed_cnn.pt"

# Grid geometry. 1 m x 2 deg is coarse enough that occupancy is a few percent
# rather than the 0.7% of the sensor's native 0.15 m x 1.8 deg resolution,
# and fine enough to separate a ghost from its parent (real second-order
# offsets are ~1.7 m).
RANGE_MAX_M = 100.0
RANGE_BIN_M = 1.0
AZIMUTH_MAX_DEG = 70.0
AZIMUTH_BIN_DEG = 2.0
RANGE_BINS = int(round(RANGE_MAX_M / RANGE_BIN_M))            # 100
AZIMUTH_BINS = int(round(2 * AZIMUTH_MAX_DEG / AZIMUTH_BIN_DEG))  # 70

# Points are bucketed by age rather than by which scan they came from, so the
# tensor has a fixed shape no matter how many scans the history holds.
AGE_SLOTS = 5
AGE_SLOT_S = 0.1
CHANNELS_PER_SLOT = 4          # occupancy, amplitude, Doppler, Doppler spread
GRID_CHANNELS = AGE_SLOTS * CHANNELS_PER_SLOT

SPEED_SCALE_MPS = 20.0
DOPPLER_SCALE_MPS = 40.0
OUTPUT_MODE = "delta"


def window_points(scans):
    """Pool a window into flat arrays, with per-scan frame statistics.

    Returns ``(range, azimuth, doppler, age, relative_amplitude, source_code)``.
    The relative amplitude is computed per scan over every point of that scan,
    which is its definition; nothing is dropped here.
    """

    r_all, az_all, vr_all, age_all, rel_all, src_all = [], [], [], [], [], []
    for age_s, scan in scans:
        r, az, vr, snr, src = _scan_columns(scan)
        if r.size == 0:
            continue
        rel, _residual, _density = frame_context_statistics(
            r, az, vr, snr_db_to_amplitude(snr)
        )
        r_all.append(r)
        az_all.append(az)
        vr_all.append(vr)
        age_all.append(np.full(r.size, max(0.0, float(age_s))))
        rel_all.append(rel)
        src_all.append(
            np.array(
                [SOURCE_CODES.get(str(s), SOURCE_CODES["other"]) for s in src],
                dtype=np.int8,
            )
        )
    if not r_all:
        empty = np.zeros(0, dtype=np.float64)
        return empty, empty.copy(), empty.copy(), empty.copy(), empty.copy(), np.zeros(0, dtype=np.int8)
    return (
        np.concatenate(r_all),
        np.concatenate(az_all),
        np.concatenate(vr_all),
        np.concatenate(age_all),
        np.concatenate(rel_all),
        np.concatenate(src_all),
    )


def rasterize(range_m, azimuth_rad, doppler_mps, age_s, relative_amplitude, keep=None):
    """Scatter points into the (GRID_CHANNELS, RANGE_BINS, AZIMUTH_BINS) tensor.

    ``keep`` is an optional boolean mask, which is how the counterfactual test
    removes ghost points and re-renders the same scene without them.
    """

    grid = np.zeros((GRID_CHANNELS, RANGE_BINS, AZIMUTH_BINS), dtype=np.float32)
    r = np.asarray(range_m, dtype=np.float64)
    if r.size == 0:
        return grid
    az = np.asarray(azimuth_rad, dtype=np.float64)
    vr = np.asarray(doppler_mps, dtype=np.float64)
    age = np.asarray(age_s, dtype=np.float64)
    rel = np.asarray(relative_amplitude, dtype=np.float64)
    if keep is not None:
        keep = np.asarray(keep, dtype=bool)
        r, az, vr, age, rel = r[keep], az[keep], vr[keep], age[keep], rel[keep]
        if r.size == 0:
            return grid

    range_bin = np.floor(r / RANGE_BIN_M).astype(np.int64)
    azimuth_bin = np.floor(
        (np.degrees(az) + AZIMUTH_MAX_DEG) / AZIMUTH_BIN_DEG
    ).astype(np.int64)
    # The epsilon keeps an age that is exactly on a slot boundary, which is
    # what a 10 Hz sensor inside a 20 Hz loop produces constantly, from
    # falling into the slot below through binary rounding (0.3 / 0.1 is
    # 2.9999... in floating point).
    slot = np.clip(
        np.floor(age / AGE_SLOT_S + 1.0e-6).astype(np.int64), 0, AGE_SLOTS - 1
    )
    inside = (
        (range_bin >= 0) & (range_bin < RANGE_BINS)
        & (azimuth_bin >= 0) & (azimuth_bin < AZIMUTH_BINS)
    )
    if not np.any(inside):
        return grid
    range_bin, azimuth_bin, slot = range_bin[inside], azimuth_bin[inside], slot[inside]
    vr, rel = vr[inside], rel[inside]

    flat = (slot * RANGE_BINS + range_bin) * AZIMUTH_BINS + azimuth_bin
    cells = AGE_SLOTS * RANGE_BINS * AZIMUTH_BINS
    count = np.bincount(flat, minlength=cells).astype(np.float64)
    amp_sum = np.bincount(flat, weights=rel, minlength=cells)
    vr_sum = np.bincount(flat, weights=vr / DOPPLER_SCALE_MPS, minlength=cells)
    vr_sq = np.bincount(flat, weights=(vr / DOPPLER_SCALE_MPS) ** 2, minlength=cells)
    occupied = count > 0
    mean_amp = np.zeros_like(amp_sum)
    mean_vr = np.zeros_like(vr_sum)
    spread = np.zeros_like(vr_sq)
    mean_amp[occupied] = amp_sum[occupied] / count[occupied]
    mean_vr[occupied] = vr_sum[occupied] / count[occupied]
    spread[occupied] = np.sqrt(
        np.maximum(vr_sq[occupied] / count[occupied] - mean_vr[occupied] ** 2, 0.0)
    )

    shape = (AGE_SLOTS, RANGE_BINS, AZIMUTH_BINS)
    planes = (
        np.log1p(count).reshape(shape),   # occupancy, compressed
        mean_amp.reshape(shape),          # already frame-relative, near unit scale
        mean_vr.reshape(shape),
        spread.reshape(shape),
    )
    for channel, plane in enumerate(planes):
        grid[channel::CHANNELS_PER_SLOT] = plane
    return grid


def build_window_grid(scans, ego_speed_mps, drop_sources=()):
    """Model input for one decision.

    Returns ``grid`` (GRID_CHANNELS x RANGE_BINS x AZIMUTH_BINS float32),
    ``ego`` (normalised ego speed) and ``sources`` (per-point codes), the last
    so evaluation can re-render the window with a class of points removed.
    """

    r, az, vr, age, rel, src = window_points(scans)
    keep = None
    if len(drop_sources) and r.size:
        keep = ~np.isin(src, list(drop_sources))
    return {
        "grid": rasterize(r, az, vr, age, rel, keep=keep),
        "ego": np.float32(float(ego_speed_mps) / SPEED_SCALE_MPS),
        "sources": src,
        "point_count": int(r.size if keep is None else int(keep.sum())),
        "arrays": (r, az, vr, age, rel),
    }


def default_model_kwargs():
    return {
        "in_channels": GRID_CHANNELS,
        "width": 32,
        "dropout": 0.1,
        "output_mode": OUTPUT_MODE,
    }


def _torch():
    import torch

    return torch


def create_model(**kwargs):
    import torch.nn as nn

    class TargetSpeedCNN(nn.Module):
        """Convolutions over the range-azimuth grid, ego speed fused at the head."""

        def __init__(self, in_channels, width, dropout, output_mode=OUTPUT_MODE):
            if output_mode != OUTPUT_MODE:
                raise ValueError(f"unsupported output_mode {output_mode!r}")
            super().__init__()

            def block(cin, cout, stride=1):
                return nn.Sequential(
                    nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False),
                    nn.BatchNorm2d(cout),
                    nn.ReLU(inplace=True),
                )

            self.features = nn.Sequential(
                block(in_channels, width),
                block(width, width, stride=2),
                block(width, width * 2),
                block(width * 2, width * 2, stride=2),
                block(width * 2, width * 2),
                nn.AdaptiveAvgPool2d((4, 3)),
            )
            self.head = nn.Sequential(
                nn.Flatten(),
                nn.Linear(width * 2 * 12 + 1, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(128, 1),
            )

        def forward(self, grid, ego):
            hidden = self.features(grid).flatten(1)
            return self.head(
                _torch().cat((hidden, ego.reshape(-1, 1)), dim=1)
            ).squeeze(-1)

    settings = default_model_kwargs()
    settings.update(kwargs)
    return TargetSpeedCNN(**settings)


def outputs_to_target_speed(outputs, ego_speed_mps):
    """Model output (normalised speed change) -> target speed in m/s."""

    speed = ego_speed_mps + outputs * SPEED_SCALE_MPS
    if hasattr(speed, "clamp"):
        return speed.clamp(min=0.0)
    return np.maximum(speed, 0.0)


def save_checkpoint(path, model, model_kwargs):
    torch = _torch()
    torch.save(
        {
            "model_state": model.state_dict(),
            "model_kwargs": model_kwargs,
            "model_type": MODEL_TYPE,
            "grid": {
                "range_bins": RANGE_BINS,
                "azimuth_bins": AZIMUTH_BINS,
                "channels": GRID_CHANNELS,
                "age_slots": AGE_SLOTS,
            },
        },
        path,
    )


def load_model(model_dir, device="cpu"):
    torch = _torch()
    with open(os.path.join(model_dir, "model_config.json"), "r", encoding="utf-8") as fh:
        config = json.load(fh)
    if config.get("model_type") != MODEL_TYPE:
        raise RuntimeError(
            f"{model_dir} holds a {config.get('model_type')!r} model, not {MODEL_TYPE!r}"
        )
    checkpoint = torch.load(
        os.path.join(model_dir, config.get("checkpoint", CHECKPOINT_NAME)),
        map_location=device,
        weights_only=False,
    )
    grid = checkpoint.get("grid", {})
    if int(grid.get("channels", GRID_CHANNELS)) != GRID_CHANNELS:
        raise RuntimeError(
            "checkpoint grid does not match this build: "
            f"{grid} vs {GRID_CHANNELS} channels"
        )
    model = create_model(**checkpoint["model_kwargs"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, config


def predict_target_speed(model, window, device="cpu"):
    """Target speed in m/s for one window dict from ``build_window_grid``."""

    torch = _torch()
    grid = torch.from_numpy(window["grid"][None]).to(device)
    ego = torch.tensor([float(window["ego"])], device=device)
    with torch.no_grad():
        normalised = float(model(grid, ego)[0])
    return float(
        outputs_to_target_speed(normalised, float(window["ego"]) * SPEED_SCALE_MPS)
    )
