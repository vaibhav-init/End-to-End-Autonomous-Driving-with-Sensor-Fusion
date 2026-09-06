#!/usr/bin/env python3
"""Target-speed transformer over the raw radar detection list.

The scalar controller sees three numbers for one selected track. By then the
evidence that separates a multipath ghost from a real car has been thrown
away by target selection. This controller sees the detection list itself:
every point of the last N scans with its range, azimuth, Doppler, amplitude
and age, plus the ego state, and predicts the same target speed the scalar
model predicts. One model, no separate filter stage. Whether it learns to
ignore ghosts is what the counterfactual test measures.

Point features reuse the ghost detector's physical, domain-agnostic contract
(`radar.ghost_detection.features`) so the frame-relative statistics are
computed identically here, in the filter and in the fidelity study. Ghost
labels never enter the input; they ride along as ``sources`` for evaluation.

Everything except the model class works without torch so the window builder
can be unit tested on the authoring box.
"""

import json
import os

import numpy as np

from radar.detection_log import DETECTION_DTYPE
from radar.ghost_detection.features import (
    FEATURE_NAMES,
    frame_context_statistics,
    physical_features,
    snr_db_to_amplitude,
)


MODEL_TYPE = "transformer"
CHECKPOINT_NAME = "target_speed_transformer.pt"
POINT_FEATURE_DIM = len(FEATURE_NAMES)
# Ego token: speed, acceleration, and a flag that marks it as the ego token.
EGO_FEATURE_DIM = 3
TOKEN_DIM = POINT_FEATURE_DIM + EGO_FEATURE_DIM
SPEED_SCALE_MPS = 20.0
ACCEL_SCALE_MPS2 = 5.0
# Same corridor as the radar selector, used for the cruise floor only.
PATH_HALF_WIDTH_M = 1.8
PATH_WIDTH_GROWTH_PER_M = 0.004
# Share of the point budget held for older scans so the age channel stays
# informative when one dense scan could fill the budget alone.
CONTEXT_RESERVE_FRACTION = 0.25

SOURCE_CODES = {"ego": 0, "direct": 1, "ghost": 2, "clutter": 3, "other": 4}


def _scan_columns(scan):
    """(range, azimuth, velocity, snr, source) arrays from a scan.

    Accepts a ``DETECTION_DTYPE`` structured array, any structured array with
    those field names, or an iterable of objects exposing the attributes
    (``RadarDetection``).
    """

    if isinstance(scan, np.ndarray) and scan.dtype.names:
        source = scan["source"]
        if source.dtype.kind == "S":
            source = np.char.decode(source, "ascii", errors="replace")
        return (
            np.asarray(scan["distance_m"], dtype=np.float64),
            np.asarray(scan["azimuth_rad"], dtype=np.float64),
            np.asarray(scan["relative_velocity_mps"], dtype=np.float64),
            np.asarray(scan["snr_db"], dtype=np.float64),
            np.asarray(source, dtype=str),
        )
    items = list(scan)
    if not items:
        empty = np.zeros(0, dtype=np.float64)
        return empty, empty.copy(), empty.copy(), empty.copy(), np.zeros(0, dtype=str)
    return (
        np.asarray([float(d.distance_m) for d in items]),
        np.asarray([float(d.azimuth_rad) for d in items]),
        np.asarray([float(d.relative_velocity_mps) for d in items]),
        np.asarray([float(d.snr_db) for d in items]),
        np.asarray([str(d.source) for d in items], dtype=str),
    )


def _evenly(indices, count):
    if len(indices) <= count:
        return indices
    positions = np.linspace(0, len(indices) - 1, count, dtype=np.int64)
    return indices[positions]


def build_window_tokens(scans, ego_speed_mps, ego_accel_mps2, max_points):
    """Assemble the model input for one decision.

    ``scans`` is a list of ``(age_s, scan)`` pairs, oldest first, with the
    current scan last at age 0. Returns a dict with ``tokens``
    (``max_points + 1`` x ``TOKEN_DIM`` float32; row 0 is the ego token),
    ``mask`` (True where a token is valid) and ``sources`` (int codes per
    token row, ``SOURCE_CODES``), which evaluation uses to switch ghosts off.
    """

    ranges, azimuths, velocities, snrs, ages, sources = [], [], [], [], [], []
    for age_s, scan in scans:
        r, az, vr, snr, src = _scan_columns(scan)
        if r.size == 0:
            continue
        ranges.append(r)
        azimuths.append(az)
        velocities.append(vr)
        snrs.append(snr)
        ages.append(np.full(r.size, max(0.0, float(age_s))))
        sources.append(src)
    tokens = np.zeros((max_points + 1, TOKEN_DIM), dtype=np.float32)
    mask = np.zeros(max_points + 1, dtype=np.bool_)
    source_codes = np.full(max_points + 1, -1, dtype=np.int8)
    tokens[0, POINT_FEATURE_DIM:] = (
        float(ego_speed_mps) / SPEED_SCALE_MPS,
        float(np.clip(ego_accel_mps2, -20.0, 20.0)) / ACCEL_SCALE_MPS2,
        1.0,
    )
    mask[0] = True
    source_codes[0] = SOURCE_CODES["ego"]
    if not ranges:
        return {"tokens": tokens, "mask": mask, "sources": source_codes, "point_count": 0}

    r = np.concatenate(ranges)
    az = np.concatenate(azimuths)
    vr = np.concatenate(velocities)
    snr = np.concatenate(snrs)
    age = np.concatenate(ages)
    src = np.concatenate(sources)

    # Budget: current scan first, but never let it starve the context.
    indices = np.arange(r.size)
    current = indices[age <= 1.0e-6]
    context = indices[age > 1.0e-6]
    if r.size > max_points:
        reserve = min(len(context), int(round(max_points * CONTEXT_RESERVE_FRACTION)))
        current_budget = max(1, max_points - reserve)
        current = _evenly(current, current_budget)
        context = _evenly(context, max_points - len(current))
        indices = np.concatenate((context, current))
        indices.sort()
    r, az, vr, snr, age, src = (
        r[indices], az[indices], vr[indices], snr[indices], age[indices], src[indices]
    )

    amplitude = snr_db_to_amplitude(snr)
    rel_amp, doppler_residual, density = frame_context_statistics(r, az, vr, amplitude)
    features = physical_features(
        r, az, vr, amplitude, age,
        relative_log_amplitude=rel_amp,
        doppler_cluster_residual=doppler_residual,
        local_density_ratio=density,
    )
    count = features.shape[0]
    tokens[1 : 1 + count, :POINT_FEATURE_DIM] = features
    mask[1 : 1 + count] = True
    for row, source in enumerate(src, start=1):
        source_codes[row] = SOURCE_CODES.get(str(source), SOURCE_CODES["other"])
    return {"tokens": tokens, "mask": mask, "sources": source_codes, "point_count": count}


def obstacle_in_corridor(scan, max_range_m):
    """Whether any point of the current scan sits in the ego corridor.

    Drives the cruise floor. Ghost points count as detected, so the floor can
    never mask a reaction to a ghost, exactly as in the scalar driver.
    """

    r, az, _vr, _snr, _src = _scan_columns(scan)
    if r.size == 0:
        return False
    forward = r * np.cos(az)
    lateral = np.abs(r * np.sin(az))
    half_width = PATH_HALF_WIDTH_M + PATH_WIDTH_GROWTH_PER_M * np.maximum(forward, 0.0)
    return bool(
        np.any((forward > 1.0) & (r < max_range_m * 0.95) & (lateral <= half_width))
    )


def default_model_kwargs():
    return {
        "token_dim": TOKEN_DIM,
        "d_model": 64,
        "heads": 4,
        "layers": 2,
        "ff_dim": 128,
        "dropout": 0.1,
    }


def _torch():
    import torch

    return torch


def create_model(**kwargs):
    """Build ``TargetSpeedTransformer``; imports torch on first use."""

    torch = _torch()
    nn = torch.nn

    class TargetSpeedTransformer(nn.Module):
        """Set attention over detection tokens, read out at the ego token."""

        def __init__(self, token_dim, d_model, heads, layers, ff_dim, dropout):
            super().__init__()
            self.input = nn.Sequential(
                nn.Linear(token_dim, d_model),
                nn.LayerNorm(d_model),
            )
            layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=heads,
                dim_feedforward=ff_dim,
                dropout=dropout,
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(
                layer, num_layers=layers, enable_nested_tensor=False
            )
            self.head = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, d_model),
                nn.ReLU(),
                nn.Linear(d_model, 1),
            )

        def forward(self, tokens, mask):
            """Normalised target speed (divide-by-SPEED_SCALE units)."""

            hidden = self.input(tokens)
            hidden = self.encoder(hidden, src_key_padding_mask=~mask)
            return self.head(hidden[:, 0]).squeeze(-1)

    settings = default_model_kwargs()
    settings.update(kwargs)
    return TargetSpeedTransformer(**settings)


def save_checkpoint(path, model, model_kwargs):
    torch = _torch()
    torch.save(
        {
            "model_type": MODEL_TYPE,
            "model_kwargs": dict(model_kwargs),
            "model_state": model.state_dict(),
            "token_dim": TOKEN_DIM,
            "speed_scale_mps": SPEED_SCALE_MPS,
        },
        path,
    )


def load_model(model_dir, device="cpu"):
    """Load the trained transformer and its ``model_config.json``."""

    torch = _torch()
    with open(os.path.join(model_dir, "model_config.json"), "r", encoding="utf-8") as fh:
        config = json.load(fh)
    if config.get("model_type") != MODEL_TYPE:
        raise ValueError(
            f"{model_dir} holds a {config.get('model_type', 'mlp')!r} model, "
            "not a transformer"
        )
    checkpoint = torch.load(
        os.path.join(model_dir, CHECKPOINT_NAME), map_location=device, weights_only=True
    )
    if int(checkpoint.get("token_dim", TOKEN_DIM)) != TOKEN_DIM:
        raise ValueError("Checkpoint token layout does not match this code")
    model = create_model(**checkpoint.get("model_kwargs", {}))
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return model, config


def predict_target_speed(model, window, device="cpu"):
    """Target speed in m/s for one window dict from ``build_window_tokens``."""

    torch = _torch()
    tokens = torch.from_numpy(window["tokens"][None]).to(device)
    mask = torch.from_numpy(window["mask"][None]).to(device)
    with torch.no_grad():
        normalised = float(model(tokens, mask)[0])
    return max(0.0, normalised * SPEED_SCALE_MPS)


def predict_batch(model, tokens, mask):
    """Normalised predictions for a batch; callers rescale."""

    return model(tokens, mask)


class ScanHistory:
    """Rolling window of the last N distinct scans for online inference."""

    def __init__(self, window_frames, fps):
        self.window_frames = int(window_frames)
        self.fps = float(fps)
        self._scans = []

    def reset(self):
        self._scans = []

    def push(self, scan):
        """Add a ``get_detections`` dict; repeated scan indices are ignored."""

        if self._scans and self._scans[-1]["scan_index"] == scan.get("scan_index"):
            return False
        self._scans.append(
            {
                "scan_index": scan.get("scan_index"),
                "timestamp": scan.get("timestamp"),
                "detections": tuple(scan.get("detections", ())),
            }
        )
        if len(self._scans) > self.window_frames:
            del self._scans[: len(self._scans) - self.window_frames]
        return True

    def windows(self):
        """``(age_s, detections)`` pairs, oldest first."""

        if not self._scans:
            return []
        latest = self._scans[-1]
        result = []
        for offset, scan in enumerate(self._scans):
            if latest["timestamp"] is not None and scan["timestamp"] is not None:
                age = max(0.0, float(latest["timestamp"]) - float(scan["timestamp"]))
            else:
                age = (len(self._scans) - 1 - offset) / self.fps
            result.append((age, scan["detections"]))
        return result

    def current(self):
        return self._scans[-1]["detections"] if self._scans else ()


def dataset_frames_to_scans(frames_available, detections_by_frame, frame_to_scan,
                            frame, window_frames, fps, min_frame=None):
    """Offline counterpart of ``ScanHistory`` for logged collections.

    Walks back ``window_frames`` frames from ``frame``, drops frames that
    repeat a scan index (the callback race), and returns ``(age_s, records)``
    pairs oldest first. ``frames_available`` is the set of logged frames.
    """

    scans = []
    seen_scans = set()
    for lag in range(window_frames):
        f = int(frame) - lag
        if min_frame is not None and f < min_frame:
            break
        if f not in frames_available:
            continue
        scan_index = frame_to_scan.get(f)
        if scan_index is not None:
            if scan_index in seen_scans:
                continue
            seen_scans.add(scan_index)
        records = detections_by_frame.get(f)
        if records is None:
            records = np.zeros(0, dtype=DETECTION_DTYPE)
        scans.append((lag / float(fps), records))
    scans.reverse()
    return scans
