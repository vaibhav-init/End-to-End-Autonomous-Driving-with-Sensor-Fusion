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
S1_OBSTACLE_DISTANCE = 25.0       # metres ahead to place stopped vehicle (S1) — tight!
S1_SPAWN_SPEED_KMH = 60.0         # spawn S1 obstacle only once ego exceeds this speed
S2_NPC_INITIAL_GAP = 25.0         # controlled comparison gap for S2
S2_NPC_SPEED_KMH = 30.0           # fair shared operating point for both models
S2_BRAKE_TRIGGER_STEP = 200       # fallback trigger when S2 staging is disabled
S3_NPC_CONSTANT_SPEED_KMH = 20.0  # NPC constant speed (S3)
S4_NPC_SPEED_KMH = 60.0           # NPC cruising speed in adjacent lane (S4)
S4_NPC_AHEAD_M = 25.0             # NPC starts 25m ahead in adjacent lane (S4) — tighter
S4_CUT_IN_TRIGGER_STEP = 60       # step at which NPC begins lane change (S4) — earlier

# ============================================================================
# Weather presets (IDs matching scenario_weather.py):
#   1 = Dark Night (headlights only, cameras blind)
#   2 = Dense Fog (0 visibility, cameras blind)
#   3 = Clear Day (both drivers should work)
#   4 = Night + Fog + Rain (worst-case for cameras)
# ============================================================================
FOG_LADDER = [1, 2, 3, 4]  # dark night → dense fog → clear → night+fog+rain

# ============================================================================
# Run configuration
# ============================================================================
RANDOM_SEEDS = [42]               # 1 seed
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
