#!/usr/bin/env python3
"""
ScenarioRunner-based Crash Data Collector for CARLA 0.9.16
===========================================================

Runs ScenarioRunner IN-PROCESS (not as subprocess) so there's a single
tick loop controlling both the scenario behavior tree AND the data
recording. Ego is driven by CARLA Traffic Manager autopilot.

Usage:
    pip install -r scenario_runner/requirements.txt
    python3 data_collector_scenariorunner.py
    python3 data_collector_scenariorunner.py --scenarios CutIn_Left HighwayCutIn
    python3 data_collector_scenariorunner.py --repetitions 3
"""

import os
import sys
import math
import time
import random
import argparse
import traceback
import pandas as pd

# Add scenario_runner to path BEFORE importing anything from it
SR_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'scenario_runner')
sys.path.insert(0, SR_ROOT)
os.environ['SCENARIO_RUNNER_ROOT'] = SR_ROOT

import carla
import py_trees

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.timer import GameTime
from srunner.scenariomanager.watchdog import Watchdog
from srunner.tools.scenario_parser import ScenarioConfigurationParser

# ============================================================================
# Configuration
# ============================================================================
CARLA_HOST = '127.0.0.1'
CARLA_PORT = 2000
FPS = 20
MAX_RADAR_RANGE = 50.0
LOOKAHEAD_SECONDS = 2.0
LOOKAHEAD_FRAMES = int(LOOKAHEAD_SECONDS * FPS)
SAVE_DIR = 'dataset_crash_sr'

# Valid CARLA maps (skip any scenario on maps not in this list)
VALID_MAPS = {'Town01', 'Town02', 'Town03', 'Town04', 'Town05', 'Town06', 'Town07', 'Town10HD'}

# Scenario definitions: name → (xml file, NHTSA type)
# Curated from actual XML configs — only configs with known-working spawn points
SCENARIO_DEFS = [
    # --- FollowLeadingVehicle (PROVEN — reliable collisions) ---
    # Town01
    ('FollowLeadingVehicle_1',             'FollowLeadingVehicle.xml',  'rear_end'),
    ('FollowLeadingVehicleWithObstacle_1', 'FollowLeadingVehicle.xml',  'rear_end'),
    ('FollowLeadingVehicle_2',             'FollowLeadingVehicle.xml',  'rear_end'),
    ('FollowLeadingVehicleWithObstacle_2', 'FollowLeadingVehicle.xml',  'rear_end'),
    # Town02
    ('FollowLeadingVehicle_11',            'FollowLeadingVehicle.xml',  'rear_end'),
    ('FollowLeadingVehicleWithObstacle_11','FollowLeadingVehicle.xml',  'rear_end'),
    # Town03
    ('FollowLeadingVehicle_4',             'FollowLeadingVehicle.xml',  'rear_end'),
    ('FollowLeadingVehicleWithObstacle_4', 'FollowLeadingVehicle.xml',  'rear_end'),
    # Town05
    ('FollowLeadingVehicle_8',             'FollowLeadingVehicle.xml',  'rear_end'),
    ('FollowLeadingVehicleWithObstacle_8', 'FollowLeadingVehicle.xml',  'rear_end'),
    ('FollowLeadingVehicle_9',             'FollowLeadingVehicle.xml',  'rear_end'),
    ('FollowLeadingVehicle_10',            'FollowLeadingVehicle.xml',  'rear_end'),
    ('FollowLeadingVehicleWithObstacle_10','FollowLeadingVehicle.xml',  'rear_end'),

    # --- NoSignalJunctionCrossing (T-bone with SyncArrival — should work) ---
    ('NoSignalJunctionCrossing',           'NoSignalJunction.xml',      'intersection'),

    # --- DynamicObjectCrossing (moving pedestrian/cyclist — better than stationary) ---
    ('DynamicObjectCrossing_3',            'ObjectCrossing.xml',        'pedestrian'),
    ('DynamicObjectCrossing_4',            'ObjectCrossing.xml',        'pedestrian'),
    ('DynamicObjectCrossing_9',            'ObjectCrossing.xml',        'pedestrian'),
    ('DynamicObjectCrossing_6',            'ObjectCrossing.xml',        'pedestrian'),

    # --- ManeuverOppositeDirection (head-on — Town05 has better road topology) ---
    ('ManeuverOppositeDirection_4',         'OppositeDirection.xml',    'head_on'),

    # --- SignalizedJunction (uses ActorFlow — works if ego reaches junction) ---
    ('SignalizedJunctionLeftTurn_4',        'SignalizedJunctionLeftTurn.xml',  'left_turn'),
    ('SignalizedJunctionLeftTurn_5',        'SignalizedJunctionLeftTurn.xml',  'left_turn'),
    ('SignalizedJunctionRightTurn_4',       'SignalizedJunctionRightTurn.xml', 'right_turn'),
    ('SignalizedJunctionRightTurn_5',       'SignalizedJunctionRightTurn.xml', 'right_turn'),

    # --- ControlLoss (debris on road) ---
    ('ControlLoss_3',                      'ControlLoss.xml',           'control_loss'),
    ('ControlLoss_7',                      'ControlLoss.xml',           'control_loss'),
    ('ControlLoss_13',                     'ControlLoss.xml',           'control_loss'),
]


# ============================================================================
# Radar Sensor (identical to data_collector_crash.py)
# ============================================================================
class RadarRecorder:
    def __init__(self, vehicle, world, range_m=50.0):
        self.latest_data = {
            'distance': range_m, 'relative_velocity': 0.0,
            'obstacle_speed': 0.0, 'obstacle_type': 2, 'lateral_offset': 0.0,
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
            if abs(det.azimuth) > 0.3 or det.depth < 1.0:
                continue
            if det.depth < nearest_dist:
                nearest_dist = det.depth
                nearest_vel = det.velocity
                nearest_azimuth = det.azimuth
        rel_vel = -nearest_vel
        obs_speed = max(0, self._ego_speed - rel_vel)
        lat_off = nearest_dist * math.sin(nearest_azimuth) if nearest_dist < self._range else 0.0
        self.latest_data = {
            'distance': nearest_dist, 'relative_velocity': rel_vel,
            'obstacle_speed': obs_speed,
            'obstacle_type': 0 if nearest_dist < self._range else 2,
            'lateral_offset': lat_off,
        }

    def get_nearest(self):
        return self.latest_data.copy()

    def cleanup(self):
        if self.sensor and self.sensor.is_alive:
            self.sensor.destroy()


class RearRadarRecorder:
    def __init__(self, vehicle, world, range_m=50.0):
        self.latest_data = {
            'rear_distance': range_m, 'rear_relative_velocity': 0.0,
            'rear_obstacle_speed': 0.0, 'rear_obstacle_type': 2,
        }
        self._ego_speed = 0.0
        self._range = range_m

        bp = world.get_blueprint_library().find('sensor.other.radar')
        bp.set_attribute('horizontal_fov', '30')
        bp.set_attribute('vertical_fov', '10')
        bp.set_attribute('range', str(range_m))
        bp.set_attribute('points_per_second', '1500')
        tf = carla.Transform(carla.Location(x=-2.5, z=0.7), carla.Rotation(pitch=0, yaw=180))
        self.sensor = world.spawn_actor(bp, tf, attach_to=vehicle)
        self.sensor.listen(self._on_radar)

    def update_ego_speed(self, speed):
        self._ego_speed = speed

    def _on_radar(self, data):
        nearest_dist = self._range
        nearest_vel = 0.0
        for det in data:
            if abs(det.azimuth) > 0.3 or det.depth < 1.0:
                continue
            if det.depth < nearest_dist:
                nearest_dist = det.depth
                nearest_vel = det.velocity
        rel_vel = -nearest_vel
        obs_speed = max(0, rel_vel + self._ego_speed)
        self.latest_data = {
            'rear_distance': nearest_dist, 'rear_relative_velocity': rel_vel,
            'rear_obstacle_speed': obs_speed,
            'rear_obstacle_type': 0 if nearest_dist < self._range else 2,
        }

    def get_nearest(self):
        return self.latest_data.copy()

    def cleanup(self):
        if self.sensor and self.sensor.is_alive:
            self.sensor.destroy()


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
            'frame_idx': frame, 'other_actor': actor_type, 'impulse': impulse,
        })
        print(f"\n  💥 COLLISION at frame {frame}! Hit {actor_type} (impulse={impulse:.0f}N·s)")

    def cleanup(self):
        if self.sensor and self.sensor.is_alive:
            self.sensor.destroy()


def apply_collision_labels(data, collision_frames, lookahead, frame_offset=0):
    """Mark frames within lookahead of a collision as positive.
    collision_frames are LOCAL frame indices; frame_offset is added to match frame_id."""
    for row in data:
        row['collision_within_2s'] = 0
    for cf in collision_frames:
        global_cf = cf + frame_offset
        start = max(frame_offset, global_cf - lookahead)
        for row in data:
            if start <= row['frame_id'] <= global_cf:
                row['collision_within_2s'] = 1


# ============================================================================
# Load and run ONE scenario using ScenarioRunner in-process
# ============================================================================
def run_single_scenario(client, scenario_name, config_file, nhtsa_type,
                        scenario_id, global_frame_offset, max_seconds=120):
    """
    Load a scenario from XML, run it with autopilot on ego, record data.
    Returns (data_rows, n_collisions).
    """
    import importlib
    import inspect
    import glob

    # Parse scenario config from XML
    configs = ScenarioConfigurationParser.parse_scenario_configuration(
        scenario_name, config_file)
    if not configs:
        print(f"  ❌ Config for '{scenario_name}' not found in {config_file}")
        return [], 0

    config = configs[0]
    town = config.town

    # Skip Town04 (segfaults) and invalid map names (e.g. 'Highway')
    if town == 'Town04':
        print(f"  ⚠️  Skipping — Town04 causes CARLA segfaults")
        return [], 0
    if town not in VALID_MAPS:
        print(f"  ⚠️  Skipping — '{town}' is not a valid CARLA map")
        return [], 0

    # Load the correct map — only if different from current
    try:
        world = client.get_world()
        current_map = world.get_map().name.split('/')[-1]
    except Exception:
        current_map = ''

    if current_map == town:
        print(f"  🗺️  Already on {town} — reusing")
        # Reset async mode first, then re-apply sync
        try:
            settings = world.get_settings()
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = None
            world.apply_settings(settings)
        except Exception:
            pass
        time.sleep(1)
    else:
        print(f"  🗺️  Loading {town}...")
        try:
            world = client.load_world(town)
            time.sleep(3)
        except Exception as e:
            print(f"  ❌ Failed to load {town}: {e}")
            return [], 0

    # Setup sync mode
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 1.0 / FPS
    world.apply_settings(settings)

    tm = client.get_trafficmanager(8000)
    tm.set_synchronous_mode(True)

    # Setup CarlaDataProvider
    CarlaDataProvider.set_client(client)
    CarlaDataProvider.set_world(world)
    world.tick()

    # Spawn ego vehicle
    ego_vehicles = []
    for vehicle_cfg in config.ego_vehicles:
        ego_actor = CarlaDataProvider.request_new_actor(
            vehicle_cfg.model, vehicle_cfg.transform,
            vehicle_cfg.rolename, random_location=vehicle_cfg.random_location,
            color=vehicle_cfg.color, actor_category=vehicle_cfg.category)
        ego_vehicles.append(ego_actor)

    if not ego_vehicles:
        print(f"  ❌ Failed to spawn ego")
        CarlaDataProvider.cleanup()
        return [], 0

    ego = ego_vehicles[0]
    world.tick()

    print(f"  🚗 Ego: {ego.type_id} at ({ego.get_location().x:.0f}, {ego.get_location().y:.0f})")

    # Enable autopilot — but make ego AGGRESSIVE so it actually crashes
    # (SR scenarios test avoidance — we WANT the ego to fail and crash)
    ego.set_autopilot(True, 8000)
    tm.auto_lane_change(ego, False)
    tm.distance_to_leading_vehicle(ego, 0.5)  # Tailgate dangerously
    tm.ignore_vehicles_percentage(ego, 80)     # Ignore 80% of vehicles
    tm.ignore_walkers_percentage(ego, 80)      # Ignore 80% of walkers
    tm.vehicle_percentage_speed_difference(ego, -30)  # Drive 30% FASTER than limit

    # Find and instantiate the scenario class
    scenario_class = None
    scenarios_path = os.path.join(SR_ROOT, 'srunner', 'scenarios')
    for py_file in glob.glob(os.path.join(scenarios_path, '*.py')):
        module_name = os.path.basename(py_file).split('.')[0]
        sys.path.insert(0, os.path.dirname(py_file))
        try:
            mod = importlib.import_module(module_name)
            for name, cls in inspect.getmembers(mod, inspect.isclass):
                if config.type in name:
                    scenario_class = cls
                    break
        except Exception:
            pass
        sys.path.pop(0)
        if scenario_class:
            break

    if scenario_class is None:
        print(f"  ❌ Scenario class for type '{config.type}' not found")
        ego.destroy()
        CarlaDataProvider.cleanup()
        return [], 0

    # Create the scenario
    try:
        scenario = scenario_class(world=world, ego_vehicles=ego_vehicles,
                                  config=config, randomize=True, debug_mode=False)
    except Exception as e:
        print(f"  ❌ Failed to create scenario: {e}")
        traceback.print_exc()
        ego.destroy()
        CarlaDataProvider.cleanup()
        return [], 0

    print(f"  ✅ Scenario '{config.type}' loaded — {len(scenario.other_actors)} NPC actors")

    # Setup the behavior tree
    scenario_tree = scenario.scenario_tree

    # Attach sensors
    rad_rec = RadarRecorder(ego, world, range_m=MAX_RADAR_RANGE)
    rear_rad_rec = RearRadarRecorder(ego, world, range_m=MAX_RADAR_RANGE)
    col_rec = CollisionRecorder(ego, world)
    print(f"  📡 Front + Rear radar + Collision sensor attached")

    # Let sensors warm up
    for _ in range(5):
        world.tick()

    # === MAIN TICK LOOP ===
    # We run the scenario behavior tree AND record data in the SAME loop
    data = []
    frame = 0
    prev_speed = 0.0
    t_start = time.time()
    stuck_frames = 0
    stuck_retries = 0
    MAX_STUCK_RETRIES = 3
    GameTime.restart()

    print(f"  🏁 Recording (max {max_seconds}s)...")

    while True:
        col_rec.frame_counter[0] = frame

        # Tick the world
        snapshot = world.tick()
        timestamp = snapshot if hasattr(snapshot, 'timestamp') else world.get_snapshot().timestamp

        # Update ScenarioRunner internals
        GameTime.on_carla_tick(timestamp)
        CarlaDataProvider.on_carla_tick()

        # Tick the scenario behavior tree (this makes NPCs do their thing)
        scenario_tree.tick_once()

        # Check if scenario is done
        if scenario_tree.status != py_trees.common.Status.RUNNING:
            print(f"  🏁 Scenario behavior tree finished (status={scenario_tree.status})")
            # Record a few more seconds of post-scenario data
            for _ in range(FPS * 3):
                world.tick()
                # Quick data record for post-scenario
                try:
                    vel = ego.get_velocity()
                    speed = math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)
                except Exception:
                    break
                frame += 1
            break

        # Timeout
        if time.time() - t_start > max_seconds:
            print(f"  ⏱️  Timeout ({max_seconds}s)")
            break

        # Read ego state
        try:
            vel = ego.get_velocity()
            speed = math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)
            accel = (speed - prev_speed) * FPS if frame > 0 else 0.0
            prev_speed = speed
            ctrl = ego.get_control()
        except RuntimeError:
            print(f"  ⚠️  Ego lost")
            break

        # Stuck detection — if ego stopped for 6s, force throttle
        if speed < 0.3:
            stuck_frames += 1
        else:
            stuck_frames = 0

        if stuck_frames > FPS * 6:  # 6 seconds stuck
            stuck_retries += 1
            if stuck_retries >= MAX_STUCK_RETRIES:
                print(f"  ❌ Stuck {MAX_STUCK_RETRIES} times — skipping scenario")
                break
            ego.set_autopilot(False, 8000)
            ego.apply_control(carla.VehicleControl(throttle=1.0, brake=0.0, steer=random.uniform(-0.3, 0.3)))
            for _ in range(FPS * 2):  # Push for 2 seconds
                world.tick()
                GameTime.on_carla_tick(world.get_snapshot().timestamp)
                CarlaDataProvider.on_carla_tick()
                scenario_tree.tick_once()
                frame += 1
            ego.set_autopilot(True, 8000)
            tm.ignore_vehicles_percentage(ego, 80)
            tm.ignore_walkers_percentage(ego, 80)
            tm.distance_to_leading_vehicle(ego, 0.5)
            tm.vehicle_percentage_speed_difference(ego, -30)
            stuck_frames = 0
            print(f"  ⚠️  Stuck → forced throttle burst (retry {stuck_retries}/{MAX_STUCK_RETRIES})")

        # Front radar
        rad_rec.update_ego_speed(speed)
        near = rad_rec.get_nearest()
        ttc = near['distance'] / near['relative_velocity'] \
            if near['relative_velocity'] > 0.1 else 10.0
        ttc = min(ttc, 10.0)

        # Rear radar
        rear_rad_rec.update_ego_speed(speed)
        rear = rear_rad_rec.get_nearest()
        rear_ttc = rear['rear_distance'] / rear['rear_relative_velocity'] \
            if rear['rear_relative_velocity'] > 0.1 else 10.0
        rear_ttc = min(rear_ttc, 10.0)

        # Record frame
        data.append({
            'frame_id': global_frame_offset + frame,
            'scenario_id': scenario_id,
            'timestamp': round(frame / FPS, 3),
            'scenario_type': nhtsa_type,
            'town': town,
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

        # Spectator follow
        try:
            spectator = world.get_spectator()
            tf = ego.get_transform()
            spectator.set_transform(carla.Transform(
                tf.location - tf.get_forward_vector() * 12 + carla.Location(z=6),
                carla.Rotation(pitch=-20, yaw=tf.rotation.yaw)))
        except Exception:
            pass

        # Status print every 2s
        if frame % (FPS * 2) == 0 and frame > 0:
            obs = ['VEH', 'PED', '---'][near['obstacle_type']]
            cols = len(col_rec.collision_frame_indices)
            print(f"  [{frame/FPS:5.0f}s] SPD:{speed:5.1f}  DIST:{near['distance']:5.1f}"
                  f"  TTC:{ttc:5.1f}  {obs}  COL:{cols}")

        frame += 1

    # Apply collision labels (pass frame_offset so global frame_ids match)
    apply_collision_labels(data, col_rec.collision_frame_indices,
                           LOOKAHEAD_FRAMES, frame_offset=global_frame_offset)

    # === CLEANUP (order matters to prevent segfault!) ===
    # 1. Destroy sensors first (stops callbacks)
    for rec in [rad_rec, rear_rad_rec, col_rec]:
        try:
            rec.cleanup()
        except Exception:
            pass

    # 2. Disable autopilot before destruction
    try:
        ego.set_autopilot(False, 8000)
    except Exception:
        pass

    # 3. Flush a few ticks to let server process destructions
    for _ in range(5):
        try:
            world.tick()
        except Exception:
            break

    # 4. Let CarlaDataProvider handle ALL actor destruction
    #    (do NOT manually destroy ego — CDP registered it via request_new_actor)
    try:
        scenario.remove_all_actors()
    except Exception:
        pass

    CarlaDataProvider.cleanup()

    # 5. Reset world to async mode
    try:
        settings = world.get_settings()
        settings.synchronous_mode = False
        settings.fixed_delta_seconds = None
        world.apply_settings(settings)
        tm.set_synchronous_mode(False)
    except Exception:
        pass

    # 6. Extra settle time before next scenario (especially before map changes)
    time.sleep(3)

    n_collisions = len(col_rec.collision_frame_indices)
    n_positive = sum(1 for r in data if r['collision_within_2s'] == 1)

    print(f"\n  Scenario: {len(data)} frames, {n_collisions} collisions")
    if col_rec.collision_details:
        for d in col_rec.collision_details:
            print(f"     → frame {d['frame_idx']}: {d['other_actor']} ({d['impulse']:.0f}N·s)")
    print(f"  📊 Labels: {n_positive} positive ({100*n_positive/max(1,len(data)):.1f}%), "
          f"{len(data)-n_positive} negative")

    return data, n_collisions


# ============================================================================
# Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description='ScenarioRunner-based Crash Data Collector')
    parser.add_argument('--host', default=CARLA_HOST)
    parser.add_argument('--port', type=int, default=CARLA_PORT)
    parser.add_argument('--scenarios', nargs='+', default=None,
                        help='Scenario names to run (default: all)')
    parser.add_argument('--repetitions', type=int, default=1,
                        help='How many times to repeat each scenario')
    parser.add_argument('--timeout', type=int, default=120,
                        help='Max seconds per scenario')
    parser.add_argument('--output', default=SAVE_DIR,
                        help='Output directory')
    args = parser.parse_args()

    # Build scenario list
    examples_dir = os.path.join(SR_ROOT, 'srunner', 'examples')
    if args.scenarios:
        scenario_list = [(name, xml, nhtsa) for name, xml, nhtsa in SCENARIO_DEFS
                         if name in args.scenarios]
    else:
        scenario_list = SCENARIO_DEFS

    total_runs = len(scenario_list) * args.repetitions

    print("=" * 70)
    print("SCENARIORUNNER CRASH DATA COLLECTOR (in-process)")
    print("=" * 70)
    print(f"  Scenarios:    {len(scenario_list)} configs × {args.repetitions} reps = {total_runs} runs")
    print(f"  Timeout:      {args.timeout}s per scenario")
    print(f"  Output:       {args.output}/data.csv")
    print("=" * 70)

    os.makedirs(args.output, exist_ok=True)
    csv_path = os.path.join(args.output, 'data.csv')

    client = carla.Client(args.host, args.port)
    client.set_timeout(30.0)

    all_data = []
    g_frames = 0
    g_collisions = 0
    g_positive = 0
    run_id = 0

    try:
        for rep in range(args.repetitions):
            for scenario_name, xml_file, nhtsa_type in scenario_list:
                run_id += 1
                config_file = os.path.join(examples_dir, xml_file)

                print(f"\n{'=' * 70}")
                print(f"RUN {run_id}/{total_runs} — {scenario_name} ({nhtsa_type})"
                      f"  [rep {rep+1}/{args.repetitions}]")
                print(f"{'=' * 70}")

                try:
                    data, n_col = run_single_scenario(
                        client, scenario_name, config_file, nhtsa_type,
                        run_id, g_frames, max_seconds=args.timeout)
                except Exception as e:
                    print(f"  ❌ SCENARIO FAILED: {e}")
                    traceback.print_exc()
                    data, n_col = [], 0

                n_pos = sum(1 for r in data if r.get('collision_within_2s') == 1)
                g_frames += len(data)
                g_collisions += n_col
                g_positive += n_pos
                all_data.extend(data)

                # Incremental save
                if all_data:
                    df = pd.DataFrame(all_data)
                    df.to_csv(csv_path, index=False)

                print(f"  💾 Saved {len(all_data)} rows → {csv_path}")
                print(f"  📊 Total: {g_frames:,} frames, {g_collisions} collisions, "
                      f"{g_positive} positive ({100*g_positive/max(1,g_frames):.1f}%)")

                time.sleep(2)

    except KeyboardInterrupt:
        print(f"\n  ⚠️  Interrupted after run {run_id}")

    # Final summary
    print(f"\n{'=' * 70}")
    print("DONE")
    print(f"{'=' * 70}")
    print(f"  Runs:       {run_id}")
    print(f"  Frames:     {g_frames:,}")
    print(f"  Collisions: {g_collisions}")
    print(f"  Positive:   {g_positive} ({100*g_positive/max(1,g_frames):.1f}%)")
    print(f"  Data:       {csv_path}")


if __name__ == '__main__':
    main()
