#!/usr/bin/env python3
"""
Crash Probability Data Collector for CARLA 0.9.16
===================================================

Diverse accident scenario generation aligned with NHTSA pre-crash typology
and SafeBench benchmark. 13 scenario types with NHTSA-weighted scheduling.

Scenario Types:
  - Rear-end (ego throttle burst, NPC ram from behind)
  - Intersection T-bone (NPC runs red light perpendicular to ego)
  - Lane-change cut-in (NPC swerves into ego's lane)
  - Head-on / wrong-way (NPC drifts into oncoming lane)
  - Pedestrian dart-out (hidden behind parked car)
  - NPC red-light runner (NPC blows through red at ego's green)
  - Unprotected left turn (oncoming NPC blocks ego's left turn)
  - Highway high-speed (fast ego + slow vehicle / highway cut-in)
  - Multi-phase cascade (lead brakes → rear NPC rams ego)
  - Group pedestrian (3-6 pedestrians crossing, some delayed)
  - Jaywalker (simple pedestrian crossing)
  - Stopped vehicle (stationary obstacle in lane)
  - Mixed (original random triggers)

Features:
  - Multi-map rotation (Town01, Town03, Town04, Town05)
  - NHTSA-weighted scenario scheduling (13 types)
  - Ego vehicle diversity (6 vehicle types rotated per scenario)
  - Weather-scenario coupling (rain→rear-end, fog→head-on, etc.)
  - Scenario-specific ego driving profiles (highway=fast, cascade=tailgate)
  - Scenario type + town tagging in CSV for per-type model evaluation
  - 7 weather presets, halved traffic light durations
"""

import carla
import numpy as np
import os
import csv
import math
import time
import random
import argparse
import traceback

# ============================================================================
# Configuration
# ============================================================================
CARLA_HOST = '127.0.0.1'
CARLA_PORT = 2000
DEFAULT_TOWN = 'Town01'

# Multi-map rotation for geometric diversity
MAP_ROTATION = ['Town01', 'Town03', 'Town04', 'Town05']

FPS = 20
SAVE_DIR = 'dataset_crash'

MAX_SEARCH_DISTANCE = 50.0
LOOKAHEAD_SECONDS = 2.0
LOOKAHEAD_FRAMES = int(LOOKAHEAD_SECONDS * FPS)

SCENARIO_SECONDS = 180         # 3 minutes per scenario
NPC_VEHICLES = 40              # Moderate traffic (less gridlock)
NPC_PEDESTRIANS = 40           # Moderate pedestrians

# Probabilities (checked once per second)
P_FULL_THROTTLE = 0.08         # 8% chance of random full throttle
P_CRASH_WHEN_STOPPED = 0.05    # 5% chance to crash when stopped at traffic
P_REAR_END_CRASH = 0.02        # 2% chance NPC behind rams into ego
THROTTLE_BURST_SECONDS = 3     # Duration of full throttle burst

STUCK_TELEPORT_SECONDS = 10    # Faster unstick (was 25)

# Scenario types with NHTSA-weighted probabilities
# Weights reflect real-world crash frequency distribution
SCENARIO_TYPES = [
    ('rear_end',         0.14),   # Original throttle-burst + rear NPC ram
    ('intersection',     0.14),   # T-bone: NPC runs red perpendicular to ego
    ('cut_in',           0.10),   # NPC lane-change into ego's path
    ('head_on',          0.08),   # NPC drifts from oncoming lane
    ('npc_red_light',    0.08),   # NPC blows red light at ego's intersection
    ('pedestrian_dart',  0.08),   # Pedestrian hidden behind parked car
    ('left_turn',        0.08),   # Unprotected left turn across path (SafeBench #6)
    ('highway',          0.08),   # High-speed highway rear-end / cut-in (Town04)
    ('cascade',          0.06),   # Multi-phase chain-reaction crash
    ('group_pedestrian', 0.06),   # Group of pedestrians forcing swerve
    ('jaywalker',        0.04),   # Original simple jaywalker
    ('stopped_vehicle',  0.03),   # Stationary obstacle in lane
    ('mixed',            0.03),   # Original mixed random triggers
]

# Ego vehicle diversity — rotate between different vehicle types
# Different vehicles have different dimensions, mass, braking characteristics
EGO_VEHICLES = [
    'vehicle.tesla.model3',          # Standard sedan
    'vehicle.audi.a2',               # Small car
    'vehicle.dodge.charger_2020',    # Muscle car (heavy, fast)
    'vehicle.mercedes.coupe_2020',   # Luxury coupe
    'vehicle.volkswagen.t2',         # Van (tall, heavy, slow braking)
    'vehicle.mini.cooper_s_2021',    # Small hatchback
]

# Weather-scenario coupling: certain scenarios pair with certain weather
# for realism (rain → longer braking, fog → late detection, etc.)
WEATHER_SCENARIO_MAP = {
    'rear_end':       ['Rainy', 'Heavy Rain', 'Fog'],       # Reduced braking
    'head_on':        ['Night', 'Night Storm', 'Fog'],      # Low visibility
    'intersection':   ['Clear', 'Cloudy', 'Rainy'],         # Common conditions
    'cut_in':         ['Clear', 'Rainy', 'Cloudy'],         # Highway-like
    'highway':        ['Clear', 'Rainy', 'Fog'],            # Highway mix
    'pedestrian_dart':['Night', 'Fog', 'Heavy Rain'],       # Hard to see peds
    'group_pedestrian':['Rainy', 'Fog', 'Night'],           # Distracted walking
    'left_turn':      ['Clear', 'Cloudy', 'Night'],         # Judgment errors
    'cascade':        ['Heavy Rain', 'Night Storm', 'Fog'], # Chain reactions
}

# Weather presets
WEATHER_PRESETS = [
    {'name': 'Clear',       'cloudiness': 10, 'precipitation': 0,   'fog_density': 0,   'wetness': 0,   'sun_altitude': 60},
    {'name': 'Cloudy',      'cloudiness': 60, 'precipitation': 0,   'fog_density': 10,  'wetness': 0,   'sun_altitude': 40},
    {'name': 'Rainy',       'cloudiness': 80, 'precipitation': 60,  'fog_density': 30,  'wetness': 60,  'sun_altitude': 35},
    {'name': 'Heavy Rain',  'cloudiness': 95, 'precipitation': 100, 'fog_density': 50,  'wetness': 100, 'sun_altitude': 15},
    {'name': 'Fog',         'cloudiness': 70, 'precipitation': 0,   'fog_density': 80,  'wetness': 30,  'sun_altitude': 40},
    {'name': 'Night',       'cloudiness': 20, 'precipitation': 0,   'fog_density': 10,  'wetness': 0,   'sun_altitude': -30},
    {'name': 'Night Storm', 'cloudiness': 100,'precipitation': 100, 'fog_density': 70,  'wetness': 100, 'sun_altitude': -20},
]


# ============================================================================
# Highway detection (walkers can't navigate highway nav mesh → segfault)
# ============================================================================
def is_highway_waypoint(wp, carla_map):
    """
    Detect if a waypoint is on a highway section.
    CARLA's AI walker controller has no nav mesh on highways, causing segfaults.
    Uses lane width + speed limit heuristics.
    """
    if wp is None:
        return True  # Err on side of caution
    try:
        # Highway lanes tend to be wider and have higher speed limits
        # Also check if there's no sidewalk nearby (highway indicator)
        lane_width = wp.lane_width
        # Get waypoint at same location to check for sidewalk
        sidewalk_wp = carla_map.get_waypoint(
            wp.transform.location, project_to_road=False,
            lane_type=carla.LaneType.Sidewalk)

        # If no sidewalk within reasonable distance, likely highway
        if sidewalk_wp is None:
            return True

        sidewalk_dist = wp.transform.location.distance(sidewalk_wp.transform.location)
        if sidewalk_dist > 15:  # Very far from any sidewalk
            return True

        # Wide lanes (>4.5m) with no nearby junctions = highway
        if lane_width > 4.5:
            # Check if any junction nearby
            check_wp = wp
            for _ in range(10):
                nxt = check_wp.next(5.0)
                if not nxt:
                    break
                check_wp = nxt[0]
                if check_wp.is_junction:
                    return False  # Has junction nearby, probably not highway
            return True  # Wide lane, no junctions = highway

    except Exception:
        pass
    return False


# ============================================================================
# Waypoint-following steering (keeps ego on road during override)
# ============================================================================
def get_waypoint_steer(ego, carla_map):
    """Calculate steering to follow the road."""
    ego_tf = ego.get_transform()
    ego_wp = carla_map.get_waypoint(
        ego.get_location(), project_to_road=True,
        lane_type=carla.LaneType.Driving)
    if ego_wp is None:
        return 0.0

    targets = ego_wp.next(8.0)
    if not targets:
        return 0.0
    target = targets[0]

    dx = target.transform.location.x - ego_tf.location.x
    dy = target.transform.location.y - ego_tf.location.y
    fwd = ego_tf.get_forward_vector()
    cross = fwd.x * dy - fwd.y * dx
    return max(-1.0, min(1.0, cross * 0.5))


# ============================================================================
# Find nearest NPC vehicle behind ego
# ============================================================================
def find_nearest_vehicle_behind(ego, world, max_dist=40.0):
    """Find the nearest NPC vehicle behind the ego within max_dist meters."""
    ego_tf = ego.get_transform()
    ego_loc = ego_tf.location
    ego_fwd = ego_tf.get_forward_vector()
    ego_right = ego_tf.get_right_vector()

    best_actor = None
    best_dist = max_dist

    for v in world.get_actors().filter('*vehicle*'):
        if v.id == ego.id:
            continue
        v_loc = v.get_location()
        dx = v_loc.x - ego_loc.x
        dy = v_loc.y - ego_loc.y

        # Project onto ego's forward axis (negative = behind)
        forward_dist = dx * ego_fwd.x + dy * ego_fwd.y
        lateral_dist = abs(dx * ego_right.x + dy * ego_right.y)

        # Must be behind ego (forward_dist < -1) and in same lane area
        if forward_dist < -1.0 and forward_dist > -max_dist and lateral_dist < 4.0:
            dist = abs(forward_dist)
            if dist < best_dist:
                best_dist = dist
                best_actor = v

    return best_actor, best_dist


# ============================================================================
# Find NPC in adjacent lane (for cut-in scenario)
# ============================================================================
def find_adjacent_lane_npc(ego, world, carla_map, max_dist=30.0):
    """Find an NPC vehicle in an adjacent lane ahead of ego."""
    ego_tf = ego.get_transform()
    ego_loc = ego_tf.location
    ego_fwd = ego_tf.get_forward_vector()
    ego_right = ego_tf.get_right_vector()

    best_actor = None
    best_dist = max_dist

    for v in world.get_actors().filter('*vehicle*'):
        if v.id == ego.id:
            continue
        v_loc = v.get_location()
        dx = v_loc.x - ego_loc.x
        dy = v_loc.y - ego_loc.y

        forward_dist = dx * ego_fwd.x + dy * ego_fwd.y
        lateral_dist = dx * ego_right.x + dy * ego_right.y

        # Must be ahead (5-25m) and in adjacent lane (lateral 2.5-6m)
        if 5.0 < forward_dist < max_dist and 2.5 < abs(lateral_dist) < 6.0:
            if forward_dist < best_dist:
                best_dist = forward_dist
                best_actor = v

    return best_actor, best_dist


# ============================================================================
# Scenario Generators — New Diverse Accident Types
# ============================================================================
def select_scenario_type():
    """Weighted random selection of scenario type based on NHTSA distribution."""
    types, weights = zip(*SCENARIO_TYPES)
    return random.choices(types, weights=weights, k=1)[0]


def setup_intersection_tbone(ego, world, carla_map, tm):
    """
    Intersection T-bone: Find a junction ahead of ego, spawn NPC on a
    perpendicular approach, and prepare to force it through when ego arrives.
    Returns list of managed NPC dicts.
    """
    actors = []
    ego_wp = carla_map.get_waypoint(ego.get_location(), project_to_road=True,
                                     lane_type=carla.LaneType.Driving)
    if ego_wp is None:
        return actors

    # Walk forward to find a junction
    wp = ego_wp
    for _ in range(30):  # Search up to ~90m ahead
        nxt = wp.next(3.0)
        if not nxt:
            break
        wp = nxt[0]
        if wp.is_junction:
            break

    if not wp.is_junction:
        return actors  # No junction found

    junction = wp.get_junction()
    if junction is None:
        return actors

    # Get junction waypoints — find a perpendicular entry
    junction_wps = junction.get_waypoints(carla.LaneType.Driving)
    ego_yaw = ego.get_transform().rotation.yaw

    for entry_wp, exit_wp in junction_wps:
        entry_yaw = entry_wp.transform.rotation.yaw
        angle_diff = abs((entry_yaw - ego_yaw + 180) % 360 - 180)
        # Perpendicular = ~90 degree difference
        if 60 < angle_diff < 120:
            # Go back from junction entry to find a spawn point
            prev_wps = entry_wp.previous(25.0)
            if not prev_wps:
                continue
            spawn_wp = prev_wps[0]
            spawn_tf = spawn_wp.transform
            spawn_tf.location.z += 0.5

            bp_lib = world.get_blueprint_library()
            veh_bps = [bp for bp in bp_lib.filter('vehicle.*')
                       if int(bp.get_attribute('number_of_wheels')) >= 4]
            npc = world.try_spawn_actor(random.choice(veh_bps), spawn_tf)
            if npc:
                # Drive toward junction at moderate speed
                fwd = spawn_tf.get_forward_vector()
                speed = random.uniform(6.0, 10.0)
                npc.enable_constant_velocity(
                    carla.Vector3D(fwd.x * speed, fwd.y * speed, 0))
                actors.append({
                    'actor': npc,
                    'type': 'tbone_runner',
                    'junction_loc': wp.transform.location,
                    'triggered': False,
                })
                print(f"  🚦 T-BONE: NPC spawned on perpendicular approach "
                      f"(angle_diff={angle_diff:.0f}°, speed={speed:.0f}m/s)")
                break
    return actors


def setup_cut_in(ego, world, carla_map, tm):
    """
    Lane-change cut-in: Find an NPC in adjacent lane ahead, override it
    to swerve into ego's lane.
    Returns list of managed NPC dicts.
    """
    actors = []
    port = tm.get_port()
    npc, dist = find_adjacent_lane_npc(ego, world, carla_map)
    if npc is None:
        # Fallback: spawn one in adjacent lane
        ego_wp = carla_map.get_waypoint(ego.get_location(), project_to_road=True,
                                         lane_type=carla.LaneType.Driving)
        if ego_wp is None:
            return actors
        # Try left then right lane
        adj_wp = ego_wp.get_left_lane() or ego_wp.get_right_lane()
        if adj_wp is None or adj_wp.lane_type != carla.LaneType.Driving:
            return actors
        # Go 15-20m ahead in adjacent lane
        for _ in range(6):
            nxt = adj_wp.next(3.0)
            if not nxt:
                break
            adj_wp = nxt[0]
        spawn_tf = adj_wp.transform
        spawn_tf.location.z += 0.5
        bp_lib = world.get_blueprint_library()
        veh_bps = [bp for bp in bp_lib.filter('vehicle.*')
                   if int(bp.get_attribute('number_of_wheels')) >= 4]
        npc = world.try_spawn_actor(random.choice(veh_bps), spawn_tf)
        if npc is None:
            return actors
        # Match ego speed roughly
        ego_vel = ego.get_velocity()
        ego_spd = math.sqrt(ego_vel.x**2 + ego_vel.y**2 + ego_vel.z**2)
        fwd = spawn_tf.get_forward_vector()
        spd = max(3.0, ego_spd * 0.8)
        npc.enable_constant_velocity(carla.Vector3D(fwd.x * spd, fwd.y * spd, 0))
        dist = 18.0

    actors.append({
        'actor': npc,
        'type': 'cut_in',
        'trigger_dist': random.uniform(12.0, 20.0),
        'steer_dir': None,  # Will be determined dynamically
        'triggered': False,
        'trigger_frame': None,
    })
    print(f"  🔀 CUT-IN: NPC ready at ~{dist:.0f}m in adjacent lane")
    return actors


def setup_head_on(ego, world, carla_map, tm):
    """
    Head-on collision: Spawn NPC driving toward ego. Uses 3 strategies:
      1. Adjacent oncoming lane (get_left/right_lane with opposite yaw)
      2. Road topology search for opposing-direction roads nearby
      3. Fallback: spawn NPC ahead in ego's lane facing BACKWARD
    Returns list of managed NPC dicts.
    """
    actors = []
    ego_wp = carla_map.get_waypoint(ego.get_location(), project_to_road=True,
                                     lane_type=carla.LaneType.Driving)
    if ego_wp is None:
        print(f"  💀 HEAD-ON: No ego waypoint — skipped")
        return actors

    bp_lib = world.get_blueprint_library()
    veh_bps = [bp for bp in bp_lib.filter('vehicle.*')
               if int(bp.get_attribute('number_of_wheels')) >= 4]

    ego_yaw = ego_wp.transform.rotation.yaw

    # --- Strategy 1: Adjacent oncoming lane ---
    oncoming_wp = None
    for candidate in [ego_wp.get_left_lane(), ego_wp.get_right_lane()]:
        if candidate and candidate.lane_type == carla.LaneType.Driving:
            cand_yaw = candidate.transform.rotation.yaw
            angle_diff = abs((cand_yaw - ego_yaw + 180) % 360 - 180)
            if angle_diff > 120:
                oncoming_wp = candidate
                break

    if oncoming_wp is not None:
        # Walk forward in ego's direction, then get oncoming lane there
        spawn_dist = random.uniform(50.0, 80.0)
        fwd_wp = ego_wp
        for _ in range(int(spawn_dist / 3)):
            nxt = fwd_wp.next(3.0)
            if not nxt:
                break
            fwd_wp = nxt[0]

        oncoming_far = None
        for candidate in [fwd_wp.get_left_lane(), fwd_wp.get_right_lane()]:
            if candidate and candidate.lane_type == carla.LaneType.Driving:
                cand_yaw = candidate.transform.rotation.yaw
                angle_diff = abs((cand_yaw - fwd_wp.transform.rotation.yaw + 180) % 360 - 180)
                if angle_diff > 120:
                    oncoming_far = candidate
                    break

        if oncoming_far is not None:
            spawn_tf = oncoming_far.transform
            spawn_tf.location.z += 0.5
            npc = world.try_spawn_actor(random.choice(veh_bps), spawn_tf)
            if npc:
                fwd = spawn_tf.get_forward_vector()
                speed = random.uniform(8.0, 14.0)
                npc.enable_constant_velocity(carla.Vector3D(fwd.x * speed, fwd.y * speed, 0))
                actors.append({
                    'actor': npc, 'type': 'head_on',
                    'drift_dist': random.uniform(25.0, 40.0),
                    'triggered': False,
                })
                print(f"  💀 HEAD-ON (adjacent lane): NPC ~{spawn_dist:.0f}m ahead, speed={speed:.0f}m/s")
                return actors

    # --- Strategy 2: Topology search for nearby opposing road ---
    ego_loc = ego.get_location()
    topology = carla_map.get_topology()
    best_wp = None
    best_dist = 999

    for wp_start, wp_end in topology:
        # Check if this road segment runs opposite to ego
        seg_yaw = wp_start.transform.rotation.yaw
        angle_diff = abs((seg_yaw - ego_yaw + 180) % 360 - 180)
        if angle_diff > 130:  # Roughly opposite
            dist = ego_loc.distance(wp_start.transform.location)
            if 10 < dist < 100 and dist < best_dist:
                best_dist = dist
                best_wp = wp_start

    if best_wp is not None:
        # Walk forward along this opposing road to find a good spawn
        spawn_wp = best_wp
        for _ in range(random.randint(5, 15)):
            nxt = spawn_wp.next(3.0)
            if not nxt:
                break
            spawn_wp = nxt[0]

        spawn_tf = spawn_wp.transform
        spawn_tf.location.z += 0.5
        npc = world.try_spawn_actor(random.choice(veh_bps), spawn_tf)
        if npc:
            fwd = spawn_tf.get_forward_vector()
            speed = random.uniform(8.0, 14.0)
            npc.enable_constant_velocity(carla.Vector3D(fwd.x * speed, fwd.y * speed, 0))
            actors.append({
                'actor': npc, 'type': 'head_on',
                'drift_dist': random.uniform(25.0, 40.0),
                'triggered': False,
            })
            print(f"  💀 HEAD-ON (topology): NPC on opposing road ~{best_dist:.0f}m away, speed={speed:.0f}m/s")
            return actors

    # --- Strategy 3: Fallback — spawn NPC ahead facing BACKWARD in ego's lane ---
    spawn_dist = random.uniform(40.0, 65.0)
    fwd_wp = ego_wp
    for _ in range(int(spawn_dist / 3)):
        nxt = fwd_wp.next(3.0)
        if not nxt:
            break
        fwd_wp = nxt[0]

    spawn_tf = fwd_wp.transform
    # Rotate 180 degrees — NPC faces ego
    spawn_tf.rotation.yaw += 180
    spawn_tf.location.z += 0.5

    npc = world.try_spawn_actor(random.choice(veh_bps), spawn_tf)
    if npc:
        fwd = spawn_tf.get_forward_vector()
        speed = random.uniform(6.0, 10.0)
        npc.enable_constant_velocity(carla.Vector3D(fwd.x * speed, fwd.y * speed, 0))
        actors.append({
            'actor': npc, 'type': 'head_on',
            'drift_dist': random.uniform(20.0, 35.0),
            'triggered': False,
        })
        print(f"  💀 HEAD-ON (wrong-way fallback): NPC facing ego ~{spawn_dist:.0f}m ahead, speed={speed:.0f}m/s")
        return actors

    print(f"  ⚠️  HEAD-ON: All 3 strategies failed — no NPC spawned")
    return actors


def setup_pedestrian_dart(ego, world, carla_map):
    """
    Pedestrian dart-out: Spawn a parked car on roadside, hide pedestrian
    behind it. When ego gets close, pedestrian darts across road.
    Returns list of managed actor dicts.
    """
    actors = []
    ego_wp = carla_map.get_waypoint(ego.get_location(), project_to_road=True,
                                     lane_type=carla.LaneType.Driving)
    if ego_wp is None:
        return actors

    # Highway guard — walker controllers segfault on highway nav mesh
    if is_highway_waypoint(ego_wp, carla_map):
        print(f"  ⚠️  DART-OUT: Skipped — ego on highway (no walker nav mesh)")
        return actors

    # Go 25-35m ahead
    wp = ego_wp
    for _ in range(random.randint(8, 12)):
        nxt = wp.next(3.0)
        if not nxt:
            break
        wp = nxt[0]

    bp_lib = world.get_blueprint_library()

    # Spawn "parked" vehicle on roadside
    road_tf = wp.transform
    right_vec = road_tf.get_right_vector()
    side = random.choice([-1, 1])  # Left or right side
    parked_loc = carla.Location(
        x=road_tf.location.x + right_vec.x * 4.5 * side,
        y=road_tf.location.y + right_vec.y * 4.5 * side,
        z=road_tf.location.z + 0.3
    )
    parked_tf = carla.Transform(parked_loc, road_tf.rotation)

    veh_bps = [bp for bp in bp_lib.filter('vehicle.*')
               if int(bp.get_attribute('number_of_wheels')) >= 4]
    parked_car = world.try_spawn_actor(random.choice(veh_bps), parked_tf)
    if parked_car:
        parked_car.set_target_velocity(carla.Vector3D(0, 0, 0))
        parked_car.apply_control(carla.VehicleControl(brake=1.0, hand_brake=True))
        actors.append({'actor': parked_car, 'type': 'parked_prop'})

    # Spawn pedestrian BEHIND the parked car (further from road)
    ped_loc = carla.Location(
        x=parked_loc.x + right_vec.x * 2.0 * side,
        y=parked_loc.y + right_vec.y * 2.0 * side,
        z=parked_loc.z + 0.3
    )
    walker_bp = random.choice(bp_lib.filter('walker.pedestrian.*'))
    if walker_bp.has_attribute('is_invincible'):
        walker_bp.set_attribute('is_invincible', 'false')
    ped = world.try_spawn_actor(walker_bp, carla.Transform(ped_loc))
    if ped:
        try:
            ctrl_bp = bp_lib.find('controller.ai.walker')
            ctrl = world.spawn_actor(ctrl_bp, carla.Transform(), attach_to=ped)
            world.tick()

            # Target: cross to opposite side of road
            cross_loc = carla.Location(
                x=road_tf.location.x - right_vec.x * 6.0 * side,
                y=road_tf.location.y - right_vec.y * 6.0 * side,
                z=road_tf.location.z
            )

            actors.append({
                'actor': ped,
                'controller': ctrl,
                'type': 'dart_pedestrian',
                'cross_loc': cross_loc,
                'road_loc': road_tf.location,
                'triggered': False,
                'trigger_dist': random.uniform(12.0, 18.0),
            })
            print(f"  🏃 DART-OUT: Pedestrian hidden behind parked car, "
                  f"trigger at {actors[-1]['trigger_dist']:.0f}m")
        except Exception as e:
            print(f"  ⚠️  DART-OUT: Walker controller failed ({e}) — destroying ped")
            try:
                ped.destroy()
            except Exception:
                pass
    return actors


def setup_npc_red_light(ego, world, carla_map, tm):
    """
    NPC red-light runner: When ego approaches a green light, find an NPC
    approaching the perpendicular red light and force it through.
    Returns list of managed NPC dicts.
    """
    actors = []
    # This scenario is triggered dynamically during the loop,
    # so we just mark the intent and return empty
    # The main loop checks for this scenario type and acts
    return actors


def setup_jaywalker(ego, world, carla_map):
    """Spawn jaywalking pedestrian crossing the road ahead of ego."""
    actors = []
    ego_wp = carla_map.get_waypoint(ego.get_location(), project_to_road=True,
                                     lane_type=carla.LaneType.Driving)
    if ego_wp is None:
        return actors

    # Highway guard — walker controllers segfault on highway nav mesh
    if is_highway_waypoint(ego_wp, carla_map):
        print(f"  ⚠️  JAYWALKER: Skipped — ego on highway (no walker nav mesh)")
        return actors

    ahead_dist = random.uniform(12, 22)
    wp = ego_wp
    for _ in range(int(ahead_dist / 3)):
        nxt = wp.next(3.0)
        if not nxt:
            break
        wp = nxt[0]

    bp_lib = world.get_blueprint_library()
    walker_bp = random.choice(bp_lib.filter('walker.pedestrian.*'))
    if walker_bp.has_attribute('is_invincible'):
        walker_bp.set_attribute('is_invincible', 'false')

    right = wp.transform.get_right_vector()
    side = random.choice([-1, 1])
    spawn_tf = carla.Transform(
        carla.Location(
            x=wp.transform.location.x + right.x * 4.0 * side,
            y=wp.transform.location.y + right.y * 4.0 * side,
            z=wp.transform.location.z + 0.5
        )
    )
    walker = world.try_spawn_actor(walker_bp, spawn_tf)
    if walker:
        try:
            ctrl_bp = bp_lib.find('controller.ai.walker')
            ctrl = world.spawn_actor(ctrl_bp, carla.Transform(), attach_to=walker)
            world.tick()
            ctrl.start()
            cross_loc = carla.Location(
                x=wp.transform.location.x - right.x * 5.0 * side,
                y=wp.transform.location.y - right.y * 5.0 * side,
                z=wp.transform.location.z
            )
            ctrl.go_to_location(cross_loc)
            ctrl.set_max_speed(random.uniform(1.8, 3.5))
            actors.append({'actor': walker, 'controller': ctrl, 'type': 'jaywalker'})
            print(f"  🚶 JAYWALKER: crossing at ~{ahead_dist:.0f}m")
        except Exception as e:
            print(f"  ⚠️  JAYWALKER: Walker controller failed ({e}) — destroying walker")
            try:
                walker.destroy()
            except Exception:
                pass
    return actors


def setup_stopped_vehicle(ego, world, carla_map):
    """Spawn stopped vehicle in ego's lane ahead."""
    actors = []
    ego_wp = carla_map.get_waypoint(ego.get_location(), project_to_road=True,
                                     lane_type=carla.LaneType.Driving)
    if ego_wp is None:
        return actors

    ahead_dist = random.uniform(20, 35)
    wp = ego_wp
    for _ in range(int(ahead_dist / 3)):
        nxt = wp.next(3.0)
        if not nxt:
            break
        wp = nxt[0]

    bp_lib = world.get_blueprint_library()
    veh_bps = [bp for bp in bp_lib.filter('vehicle.*')
               if int(bp.get_attribute('number_of_wheels')) >= 4]
    spawn_tf = wp.transform
    spawn_tf.location.z += 0.5
    npc = world.try_spawn_actor(random.choice(veh_bps), spawn_tf)
    if npc:
        npc.set_target_velocity(carla.Vector3D(0, 0, 0))
        npc.apply_control(carla.VehicleControl(brake=1.0))
        actors.append({'actor': npc, 'type': 'stopped_vehicle'})
        print(f"  🅿️  STOPPED VEHICLE: in lane at ~{ahead_dist:.0f}m")
    return actors


def setup_left_turn(ego, world, carla_map, tm):
    """
    Unprotected left turn across path (SafeBench scenario #6):
    Ego approaches a junction. NPC is coming from opposite direction.
    Ego turns left, NPC maintains speed — creating a left-turn-across-path crash.
    We force the ego to attempt a left turn at the next junction.
    """
    actors = []
    ego_wp = carla_map.get_waypoint(ego.get_location(), project_to_road=True,
                                     lane_type=carla.LaneType.Driving)
    if ego_wp is None:
        return actors

    # Walk forward to find a junction
    wp = ego_wp
    for _ in range(30):
        nxt = wp.next(3.0)
        if not nxt:
            break
        wp = nxt[0]
        if wp.is_junction:
            break

    if not wp.is_junction:
        return actors

    junction = wp.get_junction()
    if junction is None:
        return actors

    # Find an oncoming entry to the junction (roughly opposite direction)
    junction_wps = junction.get_waypoints(carla.LaneType.Driving)
    ego_yaw = ego.get_transform().rotation.yaw

    for entry_wp, exit_wp in junction_wps:
        entry_yaw = entry_wp.transform.rotation.yaw
        angle_diff = abs((entry_yaw - ego_yaw + 180) % 360 - 180)
        # Opposite direction = ~180 degree difference
        if 150 < angle_diff < 210 or angle_diff < 30:
            # This is roughly oncoming — spawn NPC coming toward junction
            prev_wps = entry_wp.previous(35.0)
            if not prev_wps:
                continue
            spawn_wp = prev_wps[0]
            spawn_tf = spawn_wp.transform
            spawn_tf.location.z += 0.5

            bp_lib = world.get_blueprint_library()
            veh_bps = [bp for bp in bp_lib.filter('vehicle.*')
                       if int(bp.get_attribute('number_of_wheels')) >= 4]
            npc = world.try_spawn_actor(random.choice(veh_bps), spawn_tf)
            if npc:
                fwd = spawn_tf.get_forward_vector()
                speed = random.uniform(8.0, 13.0)
                npc.enable_constant_velocity(
                    carla.Vector3D(fwd.x * speed, fwd.y * speed, 0))
                actors.append({
                    'actor': npc,
                    'type': 'oncoming_for_left_turn',
                    'junction_loc': wp.transform.location,
                    'triggered': False,
                })
                print(f"  ↰ LEFT TURN: Oncoming NPC at speed={speed:.0f}m/s "
                      f"(ego must turn left across path)")
                break
    return actors


def setup_highway(ego, world, carla_map, tm):
    """
    Highway high-speed scenario: Force ego to high speed, spawn slow/stopped
    vehicle ahead or NPC that cuts in at highway speed.
    Best on Town04 (has highway sections) but works on any map.
    """
    actors = []
    port = tm.get_port()

    # Make ego drive fast
    try:
        tm.vehicle_percentage_speed_difference(ego, -40)  # 40% over speed limit
        tm.distance_to_leading_vehicle(ego, 1.5)  # Tailgating
    except Exception:
        pass

    ego_wp = carla_map.get_waypoint(ego.get_location(), project_to_road=True,
                                     lane_type=carla.LaneType.Driving)
    if ego_wp is None:
        return actors

    # Spawn a SLOW vehicle far ahead in ego's lane
    ahead_dist = random.uniform(50, 80)
    wp = ego_wp
    for _ in range(int(ahead_dist / 3)):
        nxt = wp.next(3.0)
        if not nxt:
            break
        wp = nxt[0]

    bp_lib = world.get_blueprint_library()
    veh_bps = [bp for bp in bp_lib.filter('vehicle.*')
               if int(bp.get_attribute('number_of_wheels')) >= 4]

    spawn_tf = wp.transform
    spawn_tf.location.z += 0.5
    slow_npc = world.try_spawn_actor(random.choice(veh_bps), spawn_tf)
    if slow_npc:
        # Moving very slowly — ego will close distance fast
        fwd = spawn_tf.get_forward_vector()
        slow_speed = random.uniform(2.0, 5.0)  # Very slow on highway
        slow_npc.enable_constant_velocity(
            carla.Vector3D(fwd.x * slow_speed, fwd.y * slow_speed, 0))
        actors.append({
            'actor': slow_npc,
            'type': 'highway_slow',
        })
        print(f"  🛣️  HIGHWAY: Slow NPC ({slow_speed:.0f}m/s) at ~{ahead_dist:.0f}m "
              f"(ego at +40% speed)")

    # Also try to set up a highway cut-in from adjacent lane
    adj_wp = wp.get_left_lane() or wp.get_right_lane()
    if adj_wp and adj_wp.lane_type == carla.LaneType.Driving:
        # Go a bit further ahead in adjacent lane
        for _ in range(3):
            nxt = adj_wp.next(3.0)
            if not nxt:
                break
            adj_wp = nxt[0]
        adj_tf = adj_wp.transform
        adj_tf.location.z += 0.5
        cut_npc = world.try_spawn_actor(random.choice(veh_bps), adj_tf)
        if cut_npc:
            fwd = adj_tf.get_forward_vector()
            cut_npc.enable_constant_velocity(
                carla.Vector3D(fwd.x * 10.0, fwd.y * 10.0, 0))
            actors.append({
                'actor': cut_npc,
                'type': 'cut_in',
                'trigger_dist': random.uniform(15.0, 25.0),
                'steer_dir': None,
                'triggered': False,
                'trigger_frame': None,
            })
            print(f"  🛣️  HIGHWAY CUT-IN: NPC in adjacent lane ready")

    return actors


def setup_cascade(ego, world, carla_map, tm):
    """
    Multi-phase cascade scenario: Chain reaction crash.
    Phase 1: Lead vehicle brakes suddenly
    Phase 2: NPC behind ego swerves into adjacent lane
    Phase 3: Another NPC rear-ends ego while ego is braking
    Creates a realistic multi-vehicle pileup scenario.
    """
    actors = []
    port = tm.get_port()
    ego_wp = carla_map.get_waypoint(ego.get_location(), project_to_road=True,
                                     lane_type=carla.LaneType.Driving)
    if ego_wp is None:
        return actors

    bp_lib = world.get_blueprint_library()
    veh_bps = [bp for bp in bp_lib.filter('vehicle.*')
               if int(bp.get_attribute('number_of_wheels')) >= 4]

    # Phase 1: Lead vehicle 20-30m ahead (will brake suddenly)
    lead_dist = random.uniform(20, 30)
    wp = ego_wp
    for _ in range(int(lead_dist / 3)):
        nxt = wp.next(3.0)
        if not nxt:
            break
        wp = nxt[0]

    lead_tf = wp.transform
    lead_tf.location.z += 0.5
    lead = world.try_spawn_actor(random.choice(veh_bps), lead_tf)
    if lead:
        fwd = lead_tf.get_forward_vector()
        lead.enable_constant_velocity(carla.Vector3D(fwd.x * 8.0, fwd.y * 8.0, 0))
        actors.append({
            'actor': lead,
            'type': 'cascade_lead',
            'brake_time': time.time() + random.uniform(5, 10),
            'braked': False,
        })
        print(f"  ⛓️  CASCADE Phase 1: Lead vehicle at ~{lead_dist:.0f}m")

    # Phase 2: Find NPC behind ego to rear-end after ego brakes
    rear_npc, rear_dist = find_nearest_vehicle_behind(ego, world)
    if rear_npc is not None:
        actors.append({
            'actor': rear_npc,
            'type': 'cascade_rear',
            'triggered': False,
        })
        print(f"  ⛓️  CASCADE Phase 2: Rear NPC at ~{rear_dist:.0f}m (will ram after lead brakes)")
    else:
        # Spawn one behind ego
        prev_wps = ego_wp.previous(15.0)
        if prev_wps:
            rear_tf = prev_wps[0].transform
            rear_tf.location.z += 0.5
            rear_npc_spawned = world.try_spawn_actor(random.choice(veh_bps), rear_tf)
            if rear_npc_spawned:
                fwd = rear_tf.get_forward_vector()
                rear_npc_spawned.enable_constant_velocity(
                    carla.Vector3D(fwd.x * 12.0, fwd.y * 12.0, 0))
                actors.append({
                    'actor': rear_npc_spawned,
                    'type': 'cascade_rear',
                    'triggered': False,
                })
                print(f"  ⛓️  CASCADE Phase 2: Spawned rear NPC ~15m behind")

    return actors


def setup_group_pedestrian(ego, world, carla_map):
    """
    Group pedestrian crossing: Multiple pedestrians cross the road together,
    creating a hazard that forces ego to brake or swerve.
    Includes variation: some pedestrians stop mid-crossing then resume.
    """
    actors = []
    ego_wp = carla_map.get_waypoint(ego.get_location(), project_to_road=True,
                                     lane_type=carla.LaneType.Driving)
    if ego_wp is None:
        return actors

    # Highway guard — walker controllers segfault on highway nav mesh
    if is_highway_waypoint(ego_wp, carla_map):
        print(f"  ⚠️  GROUP CROSSING: Skipped — ego on highway (no walker nav mesh)")
        return actors

    ahead_dist = random.uniform(18, 30)
    wp = ego_wp
    for _ in range(int(ahead_dist / 3)):
        nxt = wp.next(3.0)
        if not nxt:
            break
        wp = nxt[0]

    bp_lib = world.get_blueprint_library()
    right = wp.transform.get_right_vector()
    road_loc = wp.transform.location
    side = random.choice([-1, 1])

    num_peds = random.randint(3, 6)
    ctrl_bp = bp_lib.find('controller.ai.walker')

    for i in range(num_peds):
        walker_bp = random.choice(bp_lib.filter('walker.pedestrian.*'))
        if walker_bp.has_attribute('is_invincible'):
            walker_bp.set_attribute('is_invincible', 'false')

        # Stagger pedestrians slightly along and across the road edge
        offset_along = random.uniform(-2.0, 2.0)
        offset_across = random.uniform(0.0, 2.0)
        fwd = wp.transform.get_forward_vector()

        ped_loc = carla.Location(
            x=road_loc.x + right.x * (4.0 + offset_across) * side + fwd.x * offset_along,
            y=road_loc.y + right.y * (4.0 + offset_across) * side + fwd.y * offset_along,
            z=road_loc.z + 0.5
        )
        ped = world.try_spawn_actor(walker_bp, carla.Transform(ped_loc))
        if ped:
            try:
                ctrl = world.spawn_actor(ctrl_bp, carla.Transform(), attach_to=ped)
                world.tick()

                cross_loc = carla.Location(
                    x=road_loc.x - right.x * (5.0 + offset_across) * side + fwd.x * offset_along,
                    y=road_loc.y - right.y * (5.0 + offset_across) * side + fwd.y * offset_along,
                    z=road_loc.z
                )

                # Some pedestrians start immediately, others wait (staggered crossing)
                is_delayed = (i > 1 and random.random() < 0.4)

                actors.append({
                    'actor': ped,
                    'controller': ctrl,
                    'type': 'group_ped',
                    'cross_loc': cross_loc,
                    'road_loc': road_loc,
                    'triggered': not is_delayed,  # Non-delayed start immediately
                    'delayed': is_delayed,
                    'trigger_dist': random.uniform(15.0, 22.0),
                    'speed': random.uniform(1.5, 3.0),
                })

                if not is_delayed:
                    ctrl.start()
                    ctrl.go_to_location(cross_loc)
                    ctrl.set_max_speed(actors[-1]['speed'])
            except Exception as e:
                print(f"  ⚠️  GROUP PED: Walker controller failed ({e})")
                try:
                    ped.destroy()
                except Exception:
                    pass

    if actors:
        started = sum(1 for a in actors if a.get('type') == 'group_ped' and a.get('triggered'))
        delayed = sum(1 for a in actors if a.get('type') == 'group_ped' and a.get('delayed'))
        print(f"  👥 GROUP CROSSING: {len([a for a in actors if a.get('type') == 'group_ped'])} "
              f"pedestrians ({started} walking, {delayed} delayed)")
    return actors


# ============================================================================
# Halve traffic light durations
# ============================================================================
def halve_traffic_lights(world):
    """Set all traffic lights to very short durations and reset them."""
    count = 0
    for tl in world.get_actors().filter('traffic.traffic_light'):
        tl.set_green_time(3.0)    # 3 seconds green
        tl.set_red_time(3.0)      # 3 seconds red
        tl.set_yellow_time(1.0)   # 1 second yellow
        tl.set_state(carla.TrafficLightState.Green)  # Reset to green so new timing kicks in
        count += 1
    print(f"  🚦 Set {count} traffic lights to 3s green / 3s red / 1s yellow")


# ============================================================================
# Radar Sensor
# ============================================================================
class RadarRecorder:
    def __init__(self, vehicle, world, range_m=50.0):
        self.latest_data = {
            'distance': range_m,
            'relative_velocity': 0.0,
            'obstacle_speed': 0.0,
            'obstacle_type': 2,
            'lateral_offset': 0.0,
        }
        self._ego_speed = 0.0
        self._range = range_m

        bp = world.get_blueprint_library().find('sensor.other.radar')
        bp.set_attribute('horizontal_fov', '30')
        bp.set_attribute('vertical_fov', '10')
        bp.set_attribute('range', str(range_m))
        bp.set_attribute('points_per_second', '1500')

        tf = carla.Transform(carla.Location(x=2.5, z=0.7), carla.Rotation(pitch=0))
        self.sensor = world.spawn_actor(bp, tf, attach_to=vehicle)
        self.sensor.listen(self._on_radar)

    def update_ego_speed(self, speed):
        self._ego_speed = speed

    def _on_radar(self, data):
        nearest_dist = self._range
        nearest_vel = 0.0
        nearest_azimuth = 0.0

        for det in data:
            if abs(det.azimuth) > 0.3:
                continue
            if det.depth < 1.0:
                continue
            if det.depth < nearest_dist:
                nearest_dist = det.depth
                nearest_vel = det.velocity
                nearest_azimuth = det.azimuth

        rel_vel = -nearest_vel
        obs_speed = max(0, self._ego_speed - rel_vel)
        lat_off = nearest_dist * math.sin(nearest_azimuth) if nearest_dist < self._range else 0.0

        self.latest_data = {
            'distance': nearest_dist,
            'relative_velocity': rel_vel,
            'obstacle_speed': obs_speed,
            'obstacle_type': 0 if nearest_dist < self._range else 2,
            'lateral_offset': lat_off,
        }

    def get_nearest(self):
        return self.latest_data.copy()

    def cleanup(self):
        if self.sensor and self.sensor.is_alive:
            self.sensor.destroy()


# ============================================================================
# Rear Radar Sensor (facing backward)
# ============================================================================
class RearRadarRecorder:
    """Radar sensor pointing BACKWARD to detect rear-approaching vehicles."""

    def __init__(self, vehicle, world, range_m=50.0):
        self.latest_data = {
            'rear_distance': range_m,
            'rear_relative_velocity': 0.0,
            'rear_obstacle_speed': 0.0,
            'rear_obstacle_type': 2,
        }
        self._ego_speed = 0.0
        self._range = range_m

        bp = world.get_blueprint_library().find('sensor.other.radar')
        bp.set_attribute('horizontal_fov', '30')
        bp.set_attribute('vertical_fov', '10')
        bp.set_attribute('range', str(range_m))
        bp.set_attribute('points_per_second', '1500')

        # Mount at REAR, facing BACKWARD (yaw=180)
        tf = carla.Transform(
            carla.Location(x=-2.5, z=0.7),
            carla.Rotation(pitch=0, yaw=180)
        )
        self.sensor = world.spawn_actor(bp, tf, attach_to=vehicle)
        self.sensor.listen(self._on_radar)

    def update_ego_speed(self, speed):
        self._ego_speed = speed

    def _on_radar(self, data):
        nearest_dist = self._range
        nearest_vel = 0.0

        for det in data:
            if abs(det.azimuth) > 0.3:
                continue
            if det.depth < 1.0:
                continue
            if det.depth < nearest_dist:
                nearest_dist = det.depth
                nearest_vel = det.velocity

        # For rear radar: approaching vehicle = negative det.velocity
        rel_vel = -nearest_vel
        obs_speed = max(0, rel_vel + self._ego_speed)

        self.latest_data = {
            'rear_distance': nearest_dist,
            'rear_relative_velocity': rel_vel,
            'rear_obstacle_speed': obs_speed,
            'rear_obstacle_type': 0 if nearest_dist < self._range else 2,
        }

    def get_nearest(self):
        return self.latest_data.copy()

    def cleanup(self):
        if self.sensor and self.sensor.is_alive:
            self.sensor.destroy()


# ============================================================================
# Collision Sensor
# ============================================================================
class CollisionRecorder:
    COOLDOWN = 5.0
    MIN_IMPULSE = 300.0

    def __init__(self, vehicle, world):
        self.collision_frame_indices = []
        self.collision_details = []
        self.frame_counter = [0]
        self._last_hit_time = {}

        bp = world.get_blueprint_library().find('sensor.other.collision')
        self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=vehicle)
        self.sensor.listen(self._on_collision)

    def _on_collision(self, event):
        now = time.time()
        actor_type = event.other_actor.type_id
        impulse = event.normal_impulse.length()

        if not (actor_type.startswith('vehicle.') or actor_type.startswith('walker.')):
            return
        if impulse < self.MIN_IMPULSE:
            return

        aid = event.other_actor.id
        if aid in self._last_hit_time and now - self._last_hit_time[aid] < self.COOLDOWN:
            return
        self._last_hit_time[aid] = now

        frame = self.frame_counter[0]
        self.collision_frame_indices.append(frame)
        self.collision_details.append({
            'frame_idx': frame,
            'other_actor': actor_type,
            'impulse': impulse,
        })
        print(f"\n  💥 COLLISION at frame {frame}! Hit {actor_type} (impulse={impulse:.0f}N·s)")

    def cleanup(self):
        if self.sensor and self.sensor.is_alive:
            self.sensor.destroy()


# ============================================================================
# Label Application
# ============================================================================
def apply_collision_labels(data, collision_frames, lookahead):
    for row in data:
        row['collision_within_2s'] = 0
    for cf in collision_frames:
        start = max(0, cf - lookahead)
        end = min(len(data), cf + 1)
        for i in range(start, end):
            data[i]['collision_within_2s'] = 1
    pos = sum(1 for r in data if r['collision_within_2s'] == 1)
    neg = len(data) - pos
    print(f"  📊 Labels: {pos} positive ({pos/max(1,len(data))*100:.1f}%), {neg} negative")
    return data


# ============================================================================
# Traffic Spawning — HEAVY
# ============================================================================
def spawn_npc_vehicles(world, client, tm, count):
    bp_lib = world.get_blueprint_library()
    veh_bps = [bp for bp in bp_lib.filter('vehicle.*')
               if int(bp.get_attribute('number_of_wheels')) >= 4]
    spawns = world.get_map().get_spawn_points()
    random.shuffle(spawns)

    port = tm.get_port()
    batch = []
    for i in range(min(count, len(spawns))):
        bp = random.choice(veh_bps)
        if bp.has_attribute('color'):
            bp.set_attribute('color', random.choice(
                bp.get_attribute('color').recommended_values))
        bp.set_attribute('role_name', 'autopilot')
        # Spawn AND enable autopilot in one batch — no halted NPCs
        batch.append(
            carla.command.SpawnActor(bp, spawns[i])
            .then(carla.command.SetAutopilot(carla.command.FutureActor, True, port))
        )

    ids = []
    for r in client.apply_batch_sync(batch, True):
        if not r.error:
            ids.append(r.actor_id)

    for vid in ids:
        v = world.get_actor(vid)
        if v:
            tm.distance_to_leading_vehicle(v, 2.5)
            tm.auto_lane_change(v, True)

    print(f"  🚗 Spawned {len(ids)}/{count} vehicles")
    return ids


def spawn_pedestrians(world, client, count):
    walker_bps = world.get_blueprint_library().filter('walker.pedestrian.*')
    ctrl_bp = world.get_blueprint_library().find('controller.ai.walker')

    walkers, ctrls = [], []
    for _ in range(count):
        bp = random.choice(walker_bps)
        if bp.has_attribute('is_invincible'):
            bp.set_attribute('is_invincible', 'false')
        loc = world.get_random_location_from_navigation()
        if loc:
            w = world.try_spawn_actor(bp, carla.Transform(loc))
            if w:
                walkers.append(w)

    for w in walkers:
        c = world.spawn_actor(ctrl_bp, carla.Transform(), attach_to=w)
        ctrls.append(c)

    world.tick()
    for c in ctrls:
        dest = world.get_random_location_from_navigation()
        if dest:
            c.start()
            c.go_to_location(dest)
            c.set_max_speed(1.0 + random.random() * 1.5)

    print(f"  🚶 Spawned {len(walkers)}/{count} pedestrians (heavy)")
    return [w.id for w in walkers], [c.id for c in ctrls]


# ============================================================================
# Cleanup
# ============================================================================
def cleanup_traffic(world, client, veh_ids, wal_ids, ctrl_ids):
    for cid in ctrl_ids:
        try:
            a = world.get_actor(cid)
            if a:
                a.stop()
        except Exception:
            pass
    destroy = veh_ids + ctrl_ids + wal_ids
    if destroy:
        client.apply_batch([carla.command.DestroyActor(x) for x in destroy])


# ============================================================================
# CSV
# ============================================================================
CSV_COLUMNS = [
    'frame_id', 'scenario_id', 'timestamp', 'scenario_type', 'town',
    'ego_speed', 'ego_acceleration', 'nearest_distance',
    'relative_velocity', 'ttc', 'obstacle_speed', 'obstacle_type',
    'lateral_offset', 'ego_steering',
    'rear_distance', 'rear_relative_velocity', 'rear_ttc',
    'rear_obstacle_speed', 'rear_obstacle_type',
    'collision_within_2s'
]

def write_csv(path, data, header=False):
    mode = 'w' if header else 'a'
    with open(path, mode=mode, newline='') as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if header:
            w.writeheader()
        for row in data:
            w.writerow(row)


# ============================================================================
# Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--town', default=DEFAULT_TOWN)
    parser.add_argument('--scenarios', type=int, default=80)
    parser.add_argument('--host', default=CARLA_HOST)
    parser.add_argument('--port', type=int, default=CARLA_PORT)
    parser.add_argument('--rotate-maps', action='store_true', default=True,
                        help='Rotate maps between scenarios')
    args = parser.parse_args()

    print("=" * 70)
    print("CRASH DATA COLLECTOR — DIVERSE SCENARIOS")
    print("=" * 70)
    print(f"  Maps:          {', '.join(MAP_ROTATION) if args.rotate_maps else args.town}")
    print(f"  Scenarios:     {args.scenarios} × {SCENARIO_SECONDS}s each")
    print(f"  Traffic:       {NPC_VEHICLES} vehicles, {NPC_PEDESTRIANS} pedestrians")
    print(f"  Scenario types: {len(SCENARIO_TYPES)} NHTSA-weighted")
    print(f"  Traffic lights: halved duration")
    print("=" * 70)

    os.makedirs(SAVE_DIR, exist_ok=True)
    csv_path = os.path.join(SAVE_DIR, 'data.csv')

    client = carla.Client(args.host, args.port)
    client.set_timeout(30.0)

    # Initial world load
    current_town = args.town
    world = client.get_world()
    cur_map = world.get_map().name.split('/')[-1]
    if current_town not in cur_map:
        world = client.load_world(current_town)

    carla_map = world.get_map()
    original_settings = world.get_settings()

    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 1.0 / FPS
    world.apply_settings(settings)

    tm = client.get_trafficmanager(8000)
    tm.set_synchronous_mode(True)
    tm.set_global_distance_to_leading_vehicle(2.5)
    tm.set_random_device_seed(random.randint(0, 10000))

    halve_traffic_lights(world)

    g_frames = 0
    g_collisions = 0
    g_positive = 0

    ego = None
    col_rec = None
    rad_rec = None
    rear_rad_rec = None
    npc_ids, wal_ids, ctrl_ids = [], [], []
    scenario_actors = []  # Managed actors for diverse scenarios
    sc = 0

    try:
        for sc in range(args.scenarios):
            # --- Select scenario type (NHTSA-weighted) ---
            scenario_type = select_scenario_type()

            # --- Map rotation ---
            if args.rotate_maps:
                target_town = MAP_ROTATION[sc % len(MAP_ROTATION)]
            else:
                target_town = args.town

            if target_town not in world.get_map().name:
                print(f"\n  🗺️  Loading {target_town}...")
                world = client.load_world(target_town)
                carla_map = world.get_map()
                settings = world.get_settings()
                settings.synchronous_mode = True
                settings.fixed_delta_seconds = 1.0 / FPS
                world.apply_settings(settings)
                tm = client.get_trafficmanager(8000)
                tm.set_synchronous_mode(True)
                tm.set_global_distance_to_leading_vehicle(2.5)
                halve_traffic_lights(world)
                current_town = target_town
            else:
                current_town = target_town

            print(f"\n{'=' * 70}")
            print(f"SCENARIO {sc+1}/{args.scenarios} — {scenario_type.upper()} ({current_town})")
            print(f"{'=' * 70}")

            # Destroy ALL existing vehicles/walkers (clear leftovers)
            for a in world.get_actors().filter('vehicle.*'):
                try:
                    a.destroy()
                except Exception:
                    pass
            for a in world.get_actors().filter('walker.*'):
                try:
                    a.destroy()
                except Exception:
                    pass
            for a in world.get_actors().filter('controller.*'):
                try:
                    a.destroy()
                except Exception:
                    pass
            world.tick()

            # Weather — coupled to scenario type for realism
            preferred_weathers = WEATHER_SCENARIO_MAP.get(scenario_type, None)
            if preferred_weathers and random.random() < 0.7:  # 70% use coupled weather
                wp_name = random.choice(preferred_weathers)
                wp = next((w for w in WEATHER_PRESETS if w['name'] == wp_name),
                          random.choice(WEATHER_PRESETS))
            else:
                wp = random.choice(WEATHER_PRESETS)
            weather = carla.WeatherParameters(
                cloudiness=wp['cloudiness'], precipitation=wp['precipitation'],
                precipitation_deposits=0.0, wind_intensity=random.uniform(0, 30),
                sun_altitude_angle=wp['sun_altitude'], fog_density=wp['fog_density'],
                fog_distance=0.0, fog_falloff=0.2, wetness=wp['wetness'])
            world.set_weather(weather)
            print(f"  🌤️  {wp['name']} (coupled={scenario_type in WEATHER_SCENARIO_MAP})")

            # Spawn ego — rotate vehicle types for diversity
            ego_vehicle_name = EGO_VEHICLES[sc % len(EGO_VEHICLES)]
            ego_bp = world.get_blueprint_library().find(ego_vehicle_name)
            if ego_bp is None:
                # Fallback to Tesla
                ego_bp = world.get_blueprint_library().find('vehicle.tesla.model3')
                ego_vehicle_name = 'vehicle.tesla.model3'
            spawns = carla_map.get_spawn_points()
            random.shuffle(spawns)
            ego = None
            for sp in spawns:
                ego = world.try_spawn_actor(ego_bp, sp)
                if ego is not None:
                    break
            if ego is None:
                print(f"  ⚠️  Could not spawn ego — skipping scenario")
                continue
            print(f"  🚗 Ego: {ego_vehicle_name.split('.')[-1]} at ({sp.location.x:.0f}, {sp.location.y:.0f})")

            # TM autopilot — scenario-specific driving profile
            port = tm.get_port()
            ego.set_autopilot(True, port)
            tm.auto_lane_change(ego, True)

            # Vary ego aggressiveness by scenario type
            if scenario_type == 'highway':
                tm.distance_to_leading_vehicle(ego, 1.5)
                tm.vehicle_percentage_speed_difference(ego, -40)  # 40% over limit
                try:
                    tm.ignore_lights_percentage(ego, 100)  # No stops on highway
                except AttributeError:
                    pass
            elif scenario_type == 'left_turn':
                tm.distance_to_leading_vehicle(ego, 3.0)
                try:
                    tm.ignore_lights_percentage(ego, 80)  # Often runs lights to force turn
                    tm.vehicle_percentage_speed_difference(ego, 10)  # Slightly slower
                except AttributeError:
                    pass
            elif scenario_type in ('cascade', 'rear_end'):
                tm.distance_to_leading_vehicle(ego, 2.0)  # Tailgating
                try:
                    tm.ignore_lights_percentage(ego, 50)
                except AttributeError:
                    pass
            else:
                tm.distance_to_leading_vehicle(ego, 3.0)
                try:
                    tm.ignore_lights_percentage(ego, 50)
                except AttributeError:
                    pass

            try:
                tm.ignore_vehicles_percentage(ego, 0)
            except AttributeError:
                pass

            # Sensors
            col_rec = CollisionRecorder(ego, world)
            rad_rec = RadarRecorder(ego, world, range_m=MAX_SEARCH_DISTANCE)
            rear_rad_rec = RearRadarRecorder(ego, world, range_m=MAX_SEARCH_DISTANCE)
            print(f"  📡 Front + Rear radar attached")

            for _ in range(20):
                world.tick()

            # Heavy traffic
            npc_ids = spawn_npc_vehicles(world, client, tm, NPC_VEHICLES)
            wal_ids, ctrl_ids = spawn_pedestrians(world, client, NPC_PEDESTRIANS)

            for _ in range(30):
                world.tick()

            # --- Setup scenario-specific actors ---
            scenario_actors = []
            if scenario_type == 'intersection':
                scenario_actors = setup_intersection_tbone(ego, world, carla_map, tm)
            elif scenario_type == 'cut_in':
                scenario_actors = setup_cut_in(ego, world, carla_map, tm)
            elif scenario_type == 'head_on':
                scenario_actors = setup_head_on(ego, world, carla_map, tm)
            elif scenario_type == 'pedestrian_dart':
                scenario_actors = setup_pedestrian_dart(ego, world, carla_map)
            elif scenario_type == 'npc_red_light':
                scenario_actors = setup_npc_red_light(ego, world, carla_map, tm)
            elif scenario_type == 'jaywalker':
                scenario_actors = setup_jaywalker(ego, world, carla_map)
            elif scenario_type == 'stopped_vehicle':
                scenario_actors = setup_stopped_vehicle(ego, world, carla_map)
            elif scenario_type == 'left_turn':
                scenario_actors = setup_left_turn(ego, world, carla_map, tm)
            elif scenario_type == 'highway':
                scenario_actors = setup_highway(ego, world, carla_map, tm)
            elif scenario_type == 'cascade':
                scenario_actors = setup_cascade(ego, world, carla_map, tm)
            elif scenario_type == 'group_pedestrian':
                scenario_actors = setup_group_pedestrian(ego, world, carla_map)
            # 'rear_end' and 'mixed' use existing random triggers

            if scenario_actors:
                print(f"  🎬 Set up {len(scenario_actors)} scenario actors")

            for _ in range(10):
                world.tick()

            # ----------------------------------------------------------
            # DATA COLLECTION LOOP
            # ----------------------------------------------------------
            data = []
            prev_speed = 0.0
            stuck_count = 0
            t_start = time.time()
            post_col = -1
            POST_COL_FRAMES = FPS * 3
            frame = 0

            # Override state
            overriding = False
            override_end_frame = 0
            rear_override_npcs = []  # NPCs forced to ram ego from behind

            print(f"  🏁 Recording {SCENARIO_SECONDS}s ({scenario_type})...")

            while True:
                col_rec.frame_counter[0] = frame

                # ---- Post-collision countdown ----
                if post_col >= 0:
                    post_col += 1
                    if post_col >= POST_COL_FRAMES:
                        if overriding:
                            ego.set_autopilot(True, port)
                            overriding = False
                        ego.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))
                        print(f"  ✅ Post-crash data collected — ending scenario")
                        break

                # ---- Detect collision ----
                if col_rec.collision_frame_indices and post_col < 0:
                    post_col = 0

                # ---- Override expired? Restore autopilot ----
                if overriding and frame >= override_end_frame and post_col < 0:
                    ego.set_autopilot(True, port)
                    overriding = False

                # ---- Skip red lights 50% of the time (direct, not TM) ----
                if not overriding and post_col < 0:
                    try:
                        tl = ego.get_traffic_light()
                        if tl is not None and tl.get_state() == carla.TrafficLightState.Red:
                            if random.random() < 0.5:
                                tl.set_state(carla.TrafficLightState.Green)
                    except Exception:
                        pass

                # ---- Every second: roll the dice ----
                if frame % FPS == 0 and frame > 0 and not overriding and post_col < 0:
                    try:
                        v = ego.get_velocity()
                        cur_speed = math.sqrt(v.x**2 + v.y**2 + v.z**2)
                    except Exception:
                        cur_speed = 0

                    near = rad_rec.get_nearest()

                    # ---- Crash-forcing triggers (ALL scenario types) ----
                    # These ensure ego actually produces collisions.
                    # Scenario-specific actors add diversity to WHAT ego crashes into,
                    # but these triggers ensure the ego doesn't just sit behind traffic.
                    if cur_speed < 0.5 and near['distance'] < 15:
                        # STOPPED with car ahead — chance to floor it
                        if random.random() < P_CRASH_WHEN_STOPPED:
                            ego.set_autopilot(False, port)
                            overriding = True
                            override_end_frame = frame + FPS * THROTTLE_BURST_SECONDS
                            print(f"  🔴 STOPPED → FLOOR IT! (distance={near['distance']:.1f}m)")

                    elif cur_speed > 3.0:
                        # MOVING — random full throttle burst
                        if random.random() < P_FULL_THROTTLE:
                            ego.set_autopilot(False, port)
                            overriding = True
                            override_end_frame = frame + FPS * THROTTLE_BURST_SECONDS
                            print(f"  ⚡ THROTTLE BURST! (speed={cur_speed:.1f}m/s)")

                    # NPC rear-end crash (all scenario types)
                    if random.random() < P_REAR_END_CRASH and post_col < 0:
                        rear_npc, rear_dist = find_nearest_vehicle_behind(ego, world)
                        if rear_npc is not None:
                            try:
                                rear_npc.set_autopilot(False, port)
                                rear_npc.apply_control(carla.VehicleControl(
                                    throttle=1.0, steer=0.0, brake=0.0))
                                rear_override_npcs.append({
                                    'actor': rear_npc,
                                    'restore_frame': frame + FPS * 3
                                })
                                print(f"  🚨 REAR NPC RAMMING! dist={rear_dist:.1f}m")
                            except Exception:
                                pass

                    # ---- NPC red-light runner (dynamic trigger) ----
                    if scenario_type == 'npc_red_light' and post_col < 0:
                        try:
                            ego_tl = ego.get_traffic_light()
                            if ego_tl and ego_tl.get_state() == carla.TrafficLightState.Green:
                                # Find an NPC approaching a perpendicular red light
                                ego_loc = ego.get_location()
                                for npc_v in world.get_actors().filter('*vehicle*'):
                                    if npc_v.id == ego.id:
                                        continue
                                    npc_tl = npc_v.get_traffic_light()
                                    if npc_tl and npc_tl.get_state() == carla.TrafficLightState.Red:
                                        npc_dist = ego_loc.distance(npc_v.get_location())
                                        if npc_dist < 40:
                                            npc_v.set_autopilot(False, port)
                                            npc_v.apply_control(carla.VehicleControl(
                                                throttle=1.0, steer=0.0, brake=0.0))
                                            rear_override_npcs.append({
                                                'actor': npc_v,
                                                'restore_frame': frame + FPS * 4
                                            })
                                            print(f"  🚨 NPC RED-LIGHT RUNNER! dist={npc_dist:.0f}m")
                                            break
                        except Exception:
                            pass

                # ---- Apply override control ----
                if overriding:
                    steer = get_waypoint_steer(ego, carla_map)
                    ego.apply_control(carla.VehicleControl(
                        throttle=1.0, steer=steer, brake=0.0))

                # ---- Restore expired rear NPC overrides ----
                still_active = []
                for rnpc in rear_override_npcs:
                    if frame >= rnpc['restore_frame']:
                        try:
                            rnpc['actor'].set_autopilot(True, port)
                        except Exception:
                            pass
                    else:
                        # Keep ramming
                        try:
                            rnpc['actor'].apply_control(carla.VehicleControl(
                                throttle=1.0, steer=0.0, brake=0.0))
                        except Exception:
                            pass
                        still_active.append(rnpc)
                rear_override_npcs = still_active

                # ---- Manage diverse scenario actors ----
                ego_loc_now = ego.get_location()
                for sa in scenario_actors:
                    try:
                        sa_type = sa.get('type', '')

                        # T-bone: NPC runs through junction when ego is near
                        if sa_type == 'tbone_runner' and not sa.get('triggered'):
                            jloc = sa['junction_loc']
                            if ego_loc_now.distance(jloc) < 25:
                                # Force NPC to keep going (no braking)
                                sa['actor'].apply_control(carla.VehicleControl(
                                    throttle=1.0, steer=0.0, brake=0.0))
                                sa['triggered'] = True
                                print(f"  🚦 T-BONE TRIGGERED! NPC charging through junction")

                        # Cut-in: NPC swerves into ego's lane
                        elif sa_type == 'cut_in' and not sa.get('triggered'):
                            npc_loc = sa['actor'].get_location()
                            dist_to_ego = ego_loc_now.distance(npc_loc)
                            if dist_to_ego < sa['trigger_dist']:
                                # Determine steer direction toward ego's lane
                                ego_right = ego.get_transform().get_right_vector()
                                dx = ego_loc_now.x - npc_loc.x
                                dy = ego_loc_now.y - npc_loc.y
                                side = dx * ego_right.x + dy * ego_right.y
                                steer = 0.4 if side > 0 else -0.4
                                sa['actor'].apply_control(carla.VehicleControl(
                                    throttle=0.7, steer=steer, brake=0.0))
                                sa['triggered'] = True
                                sa['trigger_frame'] = frame
                                print(f"  🔀 CUT-IN TRIGGERED! NPC swerving into ego lane")
                        elif sa_type == 'cut_in' and sa.get('triggered'):
                            # Maintain swerve for 1.5 seconds, then straighten
                            if frame - sa.get('trigger_frame', frame) < FPS * 1.5:
                                pass  # Control already set
                            else:
                                sa['actor'].apply_control(carla.VehicleControl(
                                    throttle=0.5, steer=0.0, brake=0.0))

                        # Head-on: NPC drifts into ego's lane at trigger distance
                        elif sa_type == 'head_on' and not sa.get('triggered'):
                            npc_loc = sa['actor'].get_location()
                            dist_to_ego = ego_loc_now.distance(npc_loc)
                            if dist_to_ego < sa['drift_dist']:
                                # Steer slightly toward ego's lane
                                ego_fwd = ego.get_transform().get_forward_vector()
                                npc_fwd = sa['actor'].get_transform().get_forward_vector()
                                # Cross product determines steer direction
                                cross = npc_fwd.x * (ego_loc_now.y - npc_loc.y) - \
                                        npc_fwd.y * (ego_loc_now.x - npc_loc.x)
                                steer = 0.15 if cross > 0 else -0.15
                                sa['actor'].apply_control(carla.VehicleControl(
                                    throttle=0.8, steer=steer, brake=0.0))
                                sa['triggered'] = True
                                print(f"  💀 HEAD-ON TRIGGERED! NPC drifting into ego lane")

                        # Pedestrian dart-out: trigger when ego gets close
                        elif sa_type == 'dart_pedestrian' and not sa.get('triggered'):
                            road_loc = sa['road_loc']
                            if ego_loc_now.distance(road_loc) < sa['trigger_dist']:
                                sa['controller'].start()
                                sa['controller'].go_to_location(sa['cross_loc'])
                                sa['controller'].set_max_speed(random.uniform(3.0, 5.0))
                                sa['triggered'] = True
                                print(f"  🏃 DART-OUT TRIGGERED! Pedestrian sprinting across road")

                        # Cascade lead: brake suddenly after timer expires
                        elif sa_type == 'cascade_lead' and not sa.get('braked'):
                            if time.time() >= sa['brake_time']:
                                sa['actor'].disable_constant_velocity()
                                sa['actor'].apply_control(carla.VehicleControl(
                                    throttle=0.0, steer=0.0, brake=1.0))
                                sa['braked'] = True
                                print(f"  ⛓️  CASCADE: Lead vehicle EMERGENCY BRAKE!")
                                # Also trigger cascade rear NPC to ram
                                for sa2 in scenario_actors:
                                    if sa2.get('type') == 'cascade_rear' and not sa2.get('triggered'):
                                        sa2['actor'].apply_control(carla.VehicleControl(
                                            throttle=1.0, steer=0.0, brake=0.0))
                                        sa2['triggered'] = True
                                        print(f"  ⛓️  CASCADE: Rear NPC RAMMING!")

                        # Cascade rear: keep ramming once triggered
                        elif sa_type == 'cascade_rear' and sa.get('triggered'):
                            sa['actor'].apply_control(carla.VehicleControl(
                                throttle=1.0, steer=0.0, brake=0.0))

                        # Left turn: oncoming NPC keeps speed through junction
                        elif sa_type == 'oncoming_for_left_turn' and not sa.get('triggered'):
                            jloc = sa['junction_loc']
                            if ego_loc_now.distance(jloc) < 30:
                                # Force oncoming NPC to maintain speed (no braking)
                                sa['actor'].apply_control(carla.VehicleControl(
                                    throttle=0.9, steer=0.0, brake=0.0))
                                sa['triggered'] = True
                                print(f"  ↰ LEFT TURN: Oncoming NPC charging through junction!")

                        # Group pedestrian: trigger delayed walkers when ego is close
                        elif sa_type == 'group_ped' and sa.get('delayed') and not sa.get('triggered'):
                            road_loc = sa.get('road_loc')
                            if road_loc and ego_loc_now.distance(road_loc) < sa['trigger_dist']:
                                sa['controller'].start()
                                sa['controller'].go_to_location(sa['cross_loc'])
                                sa['controller'].set_max_speed(sa['speed'])
                                sa['triggered'] = True
                                sa['delayed'] = False
                                print(f"  👥 Delayed pedestrian NOW CROSSING!")

                    except Exception:
                        pass

                # ---- Tick world ----
                try:
                    world.tick()
                except RuntimeError as e:
                    print(f"  ⚠️  tick failed: {e}")
                    break

                # ---- Read ego state ----
                try:
                    vel = ego.get_velocity()
                    speed = math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)
                    accel = (speed - prev_speed) * FPS if frame > 0 else 0.0
                    prev_speed = speed
                    ctrl = ego.get_control()
                    loc = ego.get_location()
                except RuntimeError:
                    break

                # ---- Stuck recovery ----
                if not overriding and speed < 0.5:
                    stuck_count += 1
                    if stuck_count >= FPS * STUCK_TELEPORT_SECONDS:
                        print(f"  ⚠️  Stuck → teleporting + re-setup scenario")
                        random.shuffle(spawns)
                        new = next((s for s in spawns if loc.distance(s.location) > 50),
                                   random.choice(spawns))
                        ego.set_transform(new)
                        for _ in range(5):
                            world.tick()
                        stuck_count = 0

                        # Re-setup scenario actors near new position
                        for sa in scenario_actors:
                            try:
                                if 'controller' in sa:
                                    sa['controller'].stop()
                                    sa['controller'].destroy()
                                sa['actor'].destroy()
                            except Exception:
                                pass
                        scenario_actors = []
                        try:
                            if scenario_type == 'intersection':
                                scenario_actors = setup_intersection_tbone(ego, world, carla_map, tm)
                            elif scenario_type == 'cut_in':
                                scenario_actors = setup_cut_in(ego, world, carla_map, tm)
                            elif scenario_type == 'head_on':
                                scenario_actors = setup_head_on(ego, world, carla_map, tm)
                            elif scenario_type == 'pedestrian_dart':
                                scenario_actors = setup_pedestrian_dart(ego, world, carla_map)
                            elif scenario_type == 'jaywalker':
                                scenario_actors = setup_jaywalker(ego, world, carla_map)
                            elif scenario_type == 'stopped_vehicle':
                                scenario_actors = setup_stopped_vehicle(ego, world, carla_map)
                            elif scenario_type == 'left_turn':
                                scenario_actors = setup_left_turn(ego, world, carla_map, tm)
                            elif scenario_type == 'highway':
                                scenario_actors = setup_highway(ego, world, carla_map, tm)
                            elif scenario_type == 'cascade':
                                scenario_actors = setup_cascade(ego, world, carla_map, tm)
                            elif scenario_type == 'group_pedestrian':
                                scenario_actors = setup_group_pedestrian(ego, world, carla_map)
                            if scenario_actors:
                                print(f"  🎬 Re-setup {len(scenario_actors)} actors near new pos")
                        except Exception:
                            pass
                else:
                    stuck_count = 0

                # ---- Front Radar ----
                rad_rec.update_ego_speed(speed)
                near = rad_rec.get_nearest()

                ttc = near['distance'] / near['relative_velocity'] \
                    if near['relative_velocity'] > 0.1 else 10.0
                ttc = min(ttc, 10.0)

                # ---- Rear Radar ----
                rear_rad_rec.update_ego_speed(speed)
                rear = rear_rad_rec.get_nearest()

                rear_ttc = rear['rear_distance'] / rear['rear_relative_velocity'] \
                    if rear['rear_relative_velocity'] > 0.1 else 10.0
                rear_ttc = min(rear_ttc, 10.0)

                # ---- Record frame ----
                data.append({
                    'frame_id': g_frames + frame,
                    'scenario_id': sc,
                    'timestamp': round(frame / FPS, 3),
                    'scenario_type': scenario_type,
                    'town': current_town,
                    'ego_speed': round(speed, 3),
                    'ego_acceleration': round(accel, 3),
                    'nearest_distance': round(near['distance'], 3),
                    'relative_velocity': round(near['relative_velocity'], 3),
                    'ttc': round(ttc, 3),
                    'obstacle_speed': round(near['obstacle_speed'], 3),
                    'obstacle_type': near['obstacle_type'],
                    'lateral_offset': round(near['lateral_offset'], 3),
                    'ego_steering': round(ctrl.steer, 4),
                    'rear_distance': round(rear['rear_distance'], 3),
                    'rear_relative_velocity': round(rear['rear_relative_velocity'], 3),
                    'rear_ttc': round(rear_ttc, 3),
                    'rear_obstacle_speed': round(rear['rear_obstacle_speed'], 3),
                    'rear_obstacle_type': rear['rear_obstacle_type'],
                    'collision_within_2s': 0,
                })

                # ---- Spectator ----
                spectator = world.get_spectator()
                tf = ego.get_transform()
                spectator.set_transform(carla.Transform(
                    tf.location - tf.get_forward_vector() * 12 + carla.Location(z=6),
                    carla.Rotation(pitch=-20, yaw=tf.rotation.yaw)))

                # ---- Status every 2 seconds ----
                if frame % (FPS * 2) == 0 and frame > 0:
                    elapsed = time.time() - t_start
                    obs = ['VEH', 'PED', '---'][near['obstacle_type']]
                    mode = 'OVERRIDE' if overriding else 'AUTOPILOT'
                    cols = len(col_rec.collision_frame_indices)
                    print(f"  [{frame/FPS:5.0f}s] "
                          f"SPD:{speed:5.1f}  "
                          f"DIST:{near['distance']:5.1f}  "
                          f"TTC:{ttc:5.1f}  "
                          f"{obs}  "
                          f"{mode:<10s}  "
                          f"COL:{cols}")

                # ---- 60 second timeout ----
                if frame >= FPS * SCENARIO_SECONDS:
                    print(f"  ⏱️  60s done")
                    break

                frame += 1

            # ----------------------------------------------------------
            # POST-PROCESS
            # ----------------------------------------------------------
            # Restore autopilot if still overriding
            if overriding:
                try:
                    ego.set_autopilot(True, port)
                except Exception:
                    pass

            n_col = len(col_rec.collision_frame_indices)
            print(f"\n  Scenario {sc+1}: {len(data)} frames, {n_col} collisions")
            if n_col > 0:
                for d in col_rec.collision_details:
                    print(f"     → frame {d['frame_idx']}: {d['other_actor']} ({d['impulse']:.0f}N·s)")

            data = apply_collision_labels(data, col_rec.collision_frame_indices, LOOKAHEAD_FRAMES)

            hdr = not os.path.exists(csv_path)
            write_csv(csv_path, data, header=hdr)
            print(f"  💾 Wrote {len(data)} rows")

            pos = sum(1 for r in data if r['collision_within_2s'] == 1)
            g_frames += len(data)
            g_collisions += n_col
            g_positive += pos

            print(f"  📊 Total: {g_frames:,} frames, {g_collisions} collisions, "
                  f"{g_positive} positive ({g_positive/max(1,g_frames)*100:.1f}%)")

            # CLEANUP
            col_rec.cleanup()
            rad_rec.cleanup()
            rear_rad_rec.cleanup()
            # Cleanup scenario actors
            for sa in scenario_actors:
                try:
                    if 'controller' in sa:
                        sa['controller'].stop()
                        sa['controller'].destroy()
                    sa['actor'].destroy()
                except Exception:
                    pass
            scenario_actors = []
            cleanup_traffic(world, client, npc_ids, wal_ids, ctrl_ids)
            npc_ids, wal_ids, ctrl_ids = [], [], []
            if ego:
                try:
                    ego.set_autopilot(False, port)
                except Exception:
                    pass
                try:
                    ego.destroy()
                except Exception:
                    pass
                ego = None
            time.sleep(1)

    except KeyboardInterrupt:
        print(f"\n  ⚠️  Interrupted after scenario {sc+1}")

    except Exception as e:
        print(f"\n  ❌ Error: {e}")
        traceback.print_exc()

    finally:
        print(f"\n{'=' * 70}")
        print("DONE")
        print(f"{'=' * 70}")
        for obj in [col_rec, rad_rec, rear_rad_rec]:
            if obj:
                try:
                    obj.cleanup()
                except Exception:
                    pass
        for sa in scenario_actors:
            try:
                if 'controller' in sa:
                    sa['controller'].stop()
                    sa['controller'].destroy()
                sa['actor'].destroy()
            except Exception:
                pass
        cleanup_traffic(world, client, npc_ids, wal_ids, ctrl_ids)
        if ego:
            try:
                ego.destroy()
            except Exception:
                pass
        try:
            world.apply_settings(original_settings)
        except Exception:
            pass
        print(f"  Scenarios:  {sc+1}")
        print(f"  Frames:     {g_frames:,}")
        print(f"  Collisions: {g_collisions}")
        print(f"  Positive:   {g_positive} ({g_positive/max(1,g_frames)*100:.1f}%)")
        print(f"  Data:       {csv_path}")


if __name__ == '__main__':
    main()
