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
