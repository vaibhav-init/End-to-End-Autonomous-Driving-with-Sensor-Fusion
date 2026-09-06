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

