#!/usr/bin/env python3
"""Closed-loop cost metrics shared by the analysis scripts.

The harness always measured the hazard side: collisions, minimum gap, TTC,
reaction time. This module measures the other side of the trade-off, the one
a ghost filter or a ghost-robust controller is supposed to improve:

* **phantom braking** -- brake commands with no ground-truth reason. A brake
  frame is legitimate only when an in-path actor is present and either inside
  the dynamic stopping envelope while closing, or inside the hard margin.
  Everything else is phantom. Reported as frames, rising-edge events, and
  events per kilometre driven.
* **jerk** -- RMS of the time derivative of a lightly smoothed longitudinal
  acceleration while moving. Comfort proxy; a controller that twitches on
  every ghost shows up here even when it never fully brakes.

Definitions are deliberately simple and stated once, here, so every driver
and every arm is scored the same way.
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from driving_contract import (  # noqa: E402
    COMFORT_DECEL_MPS2,
    HARD_MARGIN_M,
    REACTION_TIME_S,
    stopping_envelope_m,
)


FPS = 20
BRAKE_THRESHOLD = 0.3            # brake command counted as braking
OBSTACLE_PRESENT_MAX_M = 100.0   # GT distance below this => an actor is logged
MOVING_SPEED_MPS = 0.5           # jerk is measured only while moving
ACCEL_SMOOTH_FRAMES = 5
# CARLA vehicles cannot exceed this; larger logged values are spawn/teleport
# artefacts and collision spikes, not controller output.
ACCEL_CLIP_MPS2 = 12.0
# A brake episode: pulses closer than EPISODE_MERGE_S apart are one event
# (a PID chattering around zero is one decision, not fifty), and episodes
# shorter than EPISODE_MIN_S are ignored.
EPISODE_MERGE_S = 0.5
EPISODE_MIN_S = 0.15


def _column(df, name, default):
    if name in df.columns:
        return pd.to_numeric(df[name], errors="coerce").fillna(default).to_numpy()
    return np.full(len(df), default, dtype=np.float64)


def legitimate_brake_mask(df):
    """Frames where braking has a ground-truth justification."""

    distance = _column(df, "gt_distance_to_npc_m", 999.0)
    closing = _column(df, "gt_relative_velocity", 0.0)
    speed = _column(df, "gt_ego_speed_kmh", 0.0) / 3.6
    present = distance < OBSTACLE_PRESENT_MAX_M
    if "gt_npc_in_path" in df.columns:
        in_path = _column(df, "gt_npc_in_path", 1.0) > 0.5
    else:
        # Older logs predate the column; the staged scenarios only ever put
        # the actor in the ego lane, so presence implies path relevance.
        in_path = np.ones(len(df), dtype=bool)
    envelope = stopping_envelope_m(speed, closing)
    justified = (closing > 0.0) & (distance < envelope)
    justified |= distance < HARD_MARGIN_M
    return present & in_path & justified


def phantom_brake_mask(df, threshold=BRAKE_THRESHOLD):
    brake = _column(df, "brake", 0.0)
    return (brake > threshold) & ~legitimate_brake_mask(df)


def brake_episodes(mask, fps=FPS, merge_s=EPISODE_MERGE_S, min_s=EPISODE_MIN_S):
    """(start, end) frame pairs of merged braking episodes."""

    mask = np.asarray(mask, dtype=bool)
    if mask.size == 0:
        return []
    starts = np.flatnonzero(mask & ~np.concatenate(([False], mask[:-1])))
    ends = np.flatnonzero(mask & ~np.concatenate((mask[1:], [False]))) + 1
    merged = []
    gap = int(round(merge_s * fps))
    for start, end in zip(starts, ends):
        if merged and start - merged[-1][1] <= gap:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))
    minimum = max(1, int(round(min_s * fps)))
    return [(a, b) for a, b in merged if b - a >= minimum]


def peak_deceleration_mps2(df):
    accel = np.clip(_column(df, "ego_accel_mps2", 0.0), -ACCEL_CLIP_MPS2, ACCEL_CLIP_MPS2)
    return float(max(0.0, -accel.min())) if accel.size else 0.0


def rising_edges(mask):
    """Count events: runs of consecutive True frames."""

    mask = np.asarray(mask, dtype=bool)
    if mask.size == 0:
        return 0
    starts = mask & ~np.concatenate(([False], mask[:-1]))
    return int(starts.sum())


def distance_travelled_km(df, fps=FPS):
    speed = _column(df, "gt_ego_speed_kmh", 0.0) / 3.6
    return float(speed.sum() / float(fps) / 1000.0)


def jerk_rms_mps3(df, fps=FPS, smooth_frames=ACCEL_SMOOTH_FRAMES):
    accel = np.clip(_column(df, "ego_accel_mps2", 0.0), -ACCEL_CLIP_MPS2, ACCEL_CLIP_MPS2)
    speed = _column(df, "gt_ego_speed_kmh", 0.0) / 3.6
    if accel.size < smooth_frames + 2:
        return float("nan")
    kernel = np.ones(smooth_frames) / smooth_frames
    # Edge-pad before smoothing: zero padding would fake a ramp at both ends
    # of the run and charge the controller for jerk it never produced.
    pad_left = smooth_frames // 2
    pad_right = smooth_frames - 1 - pad_left
    padded = np.pad(accel, (pad_left, pad_right), mode="edge")
    smoothed = np.convolve(padded, kernel, mode="valid")
    jerk = np.diff(smoothed) * float(fps)
    moving = speed[1:] > MOVING_SPEED_MPS
    if not np.any(moving):
        return float("nan")
    return float(np.sqrt(np.mean(np.square(jerk[moving]))))


def longitudinal_cost_metrics(df, fps=FPS):
    """Phantom-braking and comfort metrics for one run CSV."""

    phantom = phantom_brake_mask(df)
    brake = _column(df, "brake", 0.0) > BRAKE_THRESHOLD
    distance_km = distance_travelled_km(df, fps)
    events = len(brake_episodes(phantom, fps))
    return {
        "brake_frames": int(brake.sum()),
        "brake_episodes": len(brake_episodes(brake, fps)),
        "phantom_brake_frames": int(phantom.sum()),
        "phantom_brake_events": events,
        "phantom_brake_per_km": (
            float(events / distance_km) if distance_km > 1.0e-6 else float("nan")
        ),
        "distance_km": distance_km,
        "jerk_rms_mps3": jerk_rms_mps3(df, fps),
    }
