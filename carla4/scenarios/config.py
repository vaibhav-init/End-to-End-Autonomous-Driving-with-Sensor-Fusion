#!/usr/bin/env python3
"""
Shared configuration for NHTSA-aligned evaluation scenarios.
"""

# ============================================================================
# CARLA connection
# ============================================================================
CARLA_HOST = "127.0.0.1"
CARLA_PORT = 2000
DEFAULT_TOWN = "Town04"
FPS = 20

# ============================================================================
# Scenario geometry
# ============================================================================
S1_OBSTACLE_DISTANCE = 35.0       # metres ahead to place stopped vehicle (S1)
S2_NPC_INITIAL_GAP = 25.0         # initial gap between ego and NPC (S2)
S2_NPC_SPEED_KMH = 60.0           # NPC cruising speed (S2)
S2_BRAKE_TRIGGER_STEP = 300       # step at which NPC slams brakes (S2)
S3_NPC_CONSTANT_SPEED_KMH = 20.0  # NPC constant speed (S3)
S4_NPC_SPEED_KMH = 60.0           # NPC cruising speed in adjacent lane (S4)
S4_NPC_AHEAD_M = 40.0             # NPC starts 40m ahead in adjacent lane (S4)
S4_CUT_IN_TRIGGER_STEP = 80       # step at which NPC begins lane change (S4)

# ============================================================================
# Fog ladder
# ============================================================================
FOG_LADDER = [0]                    # clear weather only for now
RANDOM_SEEDS = [42]                 # one seed only

# ============================================================================
# Run configuration
# ============================================================================
RANDOM_SEEDS = [42, 123, 256]     # 3 seeds for statistical robustness
SCENARIO_DURATION_S = {
    1: 30,     # S1: accelerate from 0, detect static obstacle, stop
    2: 35,     # S2: follow NPC, react to sudden brake
    3: 40,     # S3: follow NPC at constant speed, observe gap
    4: 35,     # S4: NPC cuts in from adjacent lane, react to close gap
}
FOG_SETTLE_STEPS = 60             # ticks after applying fog before spawning

# ============================================================================
# NPC traffic
# ============================================================================
BACKGROUND_VEHICLES = 0  # no NPC traffic for controlled tests
BACKGROUND_PEDESTRIANS = 0
