"""Shared longitudinal data and deployment limits."""

RADAR_RANGE_M = 100.0
NATIVE_RADAR_POINTS_PER_SECOND = 3000

# General-purpose MLP operating envelope. Scenario evaluation can stage at a
# lower speed, but collection labels and runtime predictions must never exceed
# this ceiling.
MAX_TARGET_SPEED_KMH = 60.0

# The paired S2 comparison uses a separate lower speed configured in
# scenarios/config.py; this interval isolates handover from the brake event.
S2_HANDOVER_SETTLE_S = 1.0

# Dataset quality defaults.
MAX_STOPPED_FRACTION = 0.15
WEATHER_SEGMENT_S = 60


# Stopping-distance envelope, stated once for the drivers and the analysis.
# Braking for an in-path actor is justified inside it; beyond RELEVANCE_FACTOR
# times it a target cannot justify braking at the current speed, so every
# driver treats such a target like an empty road (cruise floor) instead of
# letting a model extrapolate on a stationary guardrail 90 m ahead or a car
# pulling away. This is speed logic, not a ghost filter: ghosts inside the
# window reach the model unchanged.
COMFORT_DECEL_MPS2 = 3.0
REACTION_TIME_S = 1.0
HARD_MARGIN_M = 8.0
RELEVANCE_FACTOR = 1.5


def stopping_envelope_m(speed_mps, closing_mps):
    """Distance inside which braking for a closing in-path actor is justified."""
    import numpy as np

    speed = np.maximum(np.asarray(speed_mps, dtype=np.float64), 0.0)
    closing = np.maximum(np.asarray(closing_mps, dtype=np.float64), 0.0)
    return speed * REACTION_TIME_S + closing * closing / (2.0 * COMFORT_DECEL_MPS2) + HARD_MARGIN_M


def relevance_window_m(speed_mps, closing_mps):
    return RELEVANCE_FACTOR * stopping_envelope_m(speed_mps, closing_mps)


def obstacle_relevant(distance_m, speed_mps, closing_mps, max_range_m=RADAR_RANGE_M):
    """Whether a selected target is close enough to matter at this speed."""
    window = min(float(max_range_m) * 0.95, float(relevance_window_m(speed_mps, closing_mps)))
    return float(distance_m) < window


def future_speed_label(df, horizon_frames, speed_col="ego_speed_now", group_col="episode_id"):
    """Mean ego speed over the next ``horizon_frames`` frames, per episode.

    The collector stores this at a 10-frame (0.5 s) horizon, which makes the
    label almost equal to the current speed (at most ~2.5 m/s below it under
    full braking) and trains controllers that only ever nudge the speed. A
    longer horizon labels the approach to a stopped car with the speed the
    teacher will actually be at, so the model learns to command the drop.
    Same arithmetic as collect_throttle_brake_data.compute_segmented_speed_label.
    """
    import numpy as np
    import pandas as pd

    horizon = int(horizon_frames)

    def one(segment):
        future = [segment[speed_col].shift(-step) for step in range(1, horizon + 1)]
        return pd.concat(future, axis=1).mean(axis=1)

    if group_col not in df.columns:
        return one(df)
    labels = pd.Series(np.nan, index=df.index, dtype=np.float64)
    for _, indices in df.groupby(group_col, sort=False).groups.items():
        labels.loc[indices] = one(df.loc[indices])
    return labels

