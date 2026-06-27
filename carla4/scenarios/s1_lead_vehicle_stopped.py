#!/usr/bin/env python3
"""
Scenario S1 — Lead Vehicle Stopped
===================================

NHTSA Reference: Scenario #25 — Lead Vehicle Stopped (highest frequency crash, 16.4%)

Setup:
  - Straight road, Town04 highway
  - Ego on Traffic Manager autopilot (drives naturally)
  - Static NPC vehicle placed 35m ahead, hand brake on
  - Ego starts at 0 km/h, accelerates naturally
  - Fog density: configurable (0, 40, 70, 100)

Measures:
  - Collision rate (primary safety metric)
  - Minimum distance reached before stopping
  - Ego speed at moment of collision (if any)
  - Steps from obstacle detection to full stop (if no collision)

Expected finding:
  - Ego (autopilot) stops safely at all fog densities
  - This becomes the baseline that MLP models must match/exceed
"""

import argparse
import math
import os
import random
import sys
import time

import carla

# Add carla4 to path for shared utilities
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# CARLA agents path for BasicAgent
CARLA_ROOT = os.environ.get("CARLA_ROOT", "/opt/carla-simulator")
_agents_path = os.path.join(CARLA_ROOT, "PythonAPI", "carla")
if _agents_path not in sys.path:
    sys.path.insert(0, _agents_path)

try:
    from agents.navigation.basic_agent import BasicAgent
except ImportError:
    print("WARNING: BasicAgent not available; using fallback")
    BasicAgent = None

from ground_truth_logger import GroundTruthLogger, compute_vehicle_speed, distance_between
from spawn_utils import get_highway_spawns, spawn_obstacle_in_ego_direction
from config import (
    CARLA_HOST, CARLA_PORT, DEFAULT_TOWN, FPS,
    S1_OBSTACLE_DISTANCE, FOG_LADDER, RANDOM_SEEDS,
    SCENARIO_DURATION_S, FOG_SETTLE_STEPS,
    BACKGROUND_VEHICLES, BACKGROUND_PEDESTRIANS,
)
def set_fog_density(world, density):
    """Set a specific fog density on the current weather."""
    weather = world.get_weather()
    weather.fog_density = density
    if density > 0:
        weather.fog_distance = max(5.0, 50.0 - density * 0.4)
        weather.fog_falloff = min(3.0, 0.5 + density * 0.025)
    else:
        weather.fog_distance = 100.0
        weather.fog_falloff = 0.0
    world.set_weather(weather)


def spawn_stopped_obstacle(world, carla_map, ego_location, ahead_m):
    """Spawn a stopped vehicle ahead of ego on the same lane."""
    wp = carla_map.get_waypoint(ego_location, project_to_road=True)
    if wp is None:
        return None
    travelled = 0.0
    step = 3.0
    while travelled < ahead_m:
        next_wps = wp.next(step)
        if not next_wps:
            return None
        wp = next_wps[0]
        travelled += step

    bp_lib = world.get_blueprint_library()
    vehicle_bps = [b for b in bp_lib.filter("vehicle.*")
                   if int(b.get_attribute("number_of_wheels")) >= 4]
    bp = random.choice(vehicle_bps)
    transform = wp.transform
    transform.location.z += 0.5
    vehicle = world.try_spawn_actor(bp, transform)
    if vehicle:
        vehicle.apply_control(carla.VehicleControl(brake=1.0, hand_brake=True))
        vehicle.set_target_velocity(carla.Vector3D(0.0, 0.0, 0.0))
    return vehicle


def cleanup_actor(actor):
    if actor and actor.is_alive:
        try:
            actor.destroy()
        except RuntimeError:
            pass


def run_scenario(client, world, settings, fog_density, seed, output_dir, scenario_id=1):
    """Run S1: Lead Vehicle Stopped at a given fog density."""
    carla_map = world.get_map()
    rng = random.Random(seed)

    # Get highway-only spawn points (multi-lane straight roads)
    highway_spawns = get_highway_spawns(carla_map)
    if not highway_spawns:
        raise RuntimeError("No highway spawn points found in this map")
    print(f"  Found {len(highway_spawns)} highway spawn points")

    # Set fog
    set_fog_density(world, fog_density)
    for _ in range(FOG_SETTLE_STEPS):
        world.tick()

    # Spawn ego on highway
    ego_bp = world.get_blueprint_library().find("vehicle.tesla.model3")
    ego = None
    rng.shuffle(highway_spawns)
    for sp_tf in highway_spawns[:20]:
        ego = world.try_spawn_actor(ego_bp, sp_tf)
        if ego:
            break
    if ego is None:
        raise RuntimeError("Failed to spawn ego vehicle on highway")

    # Let the ego settle in physics before computing route
    for _ in range(5):
        world.tick()

    # Use BasicAgent for waypoint-based driving (no traffic lights, no stops)
    agent = BasicAgent(ego, target_speed=60)
    agent.ignore_traffic_lights(True)
    agent.ignore_stop_signs(True)

    # Set destination AFTER ego has settled in physics
    settled_wp = carla_map.get_waypoint(ego.get_location(), project_to_road=True,
                                         lane_type=carla.LaneType.Driving)
    dest_wps = settled_wp.next(500.0)
    if not dest_wps:
        dest_wps = settled_wp.next(200.0)  # fallback to shorter distance
    agent.set_destination(dest_wps[0].transform.location)

    for _ in range(30):
        control = agent.run_step()
        ego.apply_control(control)
        world.tick()

    # Position spectator behind and above ego for wide view (third-person chase cam)
    spectator = world.get_spectator()
    ego_t = ego.get_transform()
    spectator.set_transform(carla.Transform(
        ego_t.location - ego_t.get_forward_vector() * 15 + carla.Location(z=8),
        carla.Rotation(pitch=-20, yaw=ego_t.rotation.yaw)
    ))

    # Spawn collision sensor
    collision_bp = world.get_blueprint_library().find("sensor.other.collision")
    collision_sensor = world.spawn_actor(collision_bp, carla.Transform(), attach_to=ego)
    collision_occurred = [False]

    def on_collision(event):
        collision_occurred[0] = True

    collision_sensor.listen(on_collision)

    # Logger
    logger = GroundTruthLogger(output_dir, scenario_id, fog_density, seed)

    # Phase 1: Drive for a bit, then spawn obstacle (after ~5s)
    obstacle = None
    obstacle_spawned = False
    max_steps = SCENARIO_DURATION_S[scenario_id] * FPS

    print(f"  S1 | fog={fog_density} | seed={seed} | "
          f"spawned at ({ego.get_location().x:.0f}, {ego.get_location().y:.0f})")

    prev_speed = 0.0
    obstacle_spawn_step = 5 * FPS  # spawn after 5 seconds

    try:
        for step in range(max_steps):
            # BasicAgent drives the ego
            control = agent.run_step()
            ego.apply_control(control)

            _tick_start = time.perf_counter()
            world.tick()
            # Maintain real-time playback speed
            _elapsed = time.perf_counter() - _tick_start
            if _elapsed < 1.0 / FPS:
                time.sleep(1.0 / FPS - _elapsed)

            # Spawn obstacle after N ticks
            if not obstacle_spawned and step >= obstacle_spawn_step:
                obstacle = spawn_obstacle_in_ego_direction(
                    world, carla_map, ego, S1_OBSTACLE_DISTANCE)
                if obstacle:
                    obstacle_spawned = True
                    obs_dist = ego.get_location().distance(obstacle.get_location())
                    print(f"    🚧 Obstacle spawned {obs_dist:.0f}m ahead " +
                          f"at step {step}")

            # Measure
            control = ego.get_control() if ego.is_alive else None
            ego_speed = compute_vehicle_speed(ego)
            accel = (ego_speed / 3.6 - prev_speed) * FPS
            prev_speed = ego_speed / 3.6

            npc_speed = None
            dist = None
            rel_vel = None
            if obstacle and obstacle.is_alive:
                npc_speed = compute_vehicle_speed(obstacle)
                dist = distance_between(ego, obstacle)
                # Approximate relative velocity from speed difference
                rel_vel = (ego_speed - npc_speed) / 3.6

            throttle = control.throttle if control else 0.0
            brake = control.brake if control else 0.0
            steer = control.steer if control else 0.0

            logger.log(
                step=step,
                ego_speed_kmh=ego_speed,
                npc_speed_kmh=npc_speed,
                distance_to_npc=dist,
                relative_velocity=rel_vel,
                throttle=throttle,
                brake=brake,
                steer=steer,
                collision=collision_occurred[0],
                ego_accel=accel,
            )

            # Stop early if collision
            if collision_occurred[0]:
                print(f"    💥 COLLISION at step {step} (speed={ego_speed:.1f} km/h)")
                break

            # Stop early if ego stopped near obstacle
            if obstacle_spawned and obstacle and obstacle.is_alive:
                if dist is not None and dist < 0.5:
                    break
                if ego_speed < 0.1 and dist is not None and dist < 5.0:
                    break

            # Log progress
            if step % (FPS * 2) == 0 and obstacle_spawned:
                dist_str = f"{dist:.1f}m" if dist else "N/A"
                print(f"    step={step:4d}  spd={ego_speed:5.1f}km/h  dist={dist_str}")

            # Update spectator every tick (smooth chase cam)
            if ego.is_alive:
                ego_t = ego.get_transform()
                spectator.set_transform(carla.Transform(
                    ego_t.location - ego_t.get_forward_vector() * 15 + carla.Location(z=8),
                    carla.Rotation(pitch=-20, yaw=ego_t.rotation.yaw)
                ))

    finally:
        logger.close()
        collision_sensor.destroy()
        cleanup_actor(obstacle)
        cleanup_actor(ego)

    return logger


def main():
    parser = argparse.ArgumentParser(description="S1: Lead Vehicle Stopped (NHTSA #25)")
    parser.add_argument("--host", default=CARLA_HOST)
    parser.add_argument("--port", type=int, default=CARLA_PORT)
    parser.add_argument("--town", default=DEFAULT_TOWN)
    parser.add_argument("--fog", type=int, nargs="+", default=FOG_LADDER,
                        help="Fog densities to test")
    parser.add_argument("--seeds", type=int, nargs="+", default=RANDOM_SEEDS,
                        help="Random seeds")
    parser.add_argument("--output", default="results_s1")
    parser.add_argument("--headless", action="store_true", help="No spectator camera")
    args = parser.parse_args()

    client = carla.Client(args.host, args.port)
    client.set_timeout(30.0)
    world = client.get_world()
    if world.get_map().name.split("/")[-1] != args.town:
        print(f"Loading {args.town} ...")
        world = client.load_world(args.town)
        time.sleep(3)

    original_settings = world.get_settings()
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 1.0 / FPS
    world.apply_settings(settings)

    tm = client.get_trafficmanager(8000)
    tm.set_synchronous_mode(True)
    world.tick()

    print("=" * 64)
    print("SCENARIO S1 — LEAD VEHICLE STOPPED (NHTSA #25)")
    print("=" * 64)
    print(f"  Fog levels:      {args.fog}")
    print(f"  Seeds:           {args.seeds}")
    print(f"  Output dir:      {args.output}")
    print(f"  Obstacle at:     {S1_OBSTACLE_DISTANCE:.0f}m")
    print(f"  Runs:            {len(args.fog)} × {len(args.seeds)} = {len(args.fog) * len(args.seeds)}")
    print("=" * 64)

    results = []
    for fog in args.fog:
        for seed in args.seeds:
            try:
                logger = run_scenario(client, world, settings, fog, seed,
                                      args.output, scenario_id=1)
                results.append({
                    "fog": fog,
                    "seed": seed,
                    "collision": logger.has_collision,
                    "min_dist": logger.min_distance,
                    "rows": logger.row_count,
                })
                status = "💥" if logger.has_collision else "✅"
                print(f"    {status} fog={fog} seed={seed} — "
                      f"collision={logger.has_collision} min_dist={logger.min_distance:.1f}m")
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"    ❌ fog={fog} seed={seed} failed: {e}")

    print("\n" + "=" * 64)
    print("S1 SUMMARY")
    print("=" * 64)
    for r in results:
        icon = "💥" if r["collision"] else "✅"
        print(f"  {icon} fog={r['fog']:3d}  seed={r['seed']:3d}  "
              f"collision={r['collision']}  min_dist={r['min_dist']:.1f}m")
    collisions = sum(1 for r in results if r["collision"])
    print(f"\n  Total: {len(results)} runs, {collisions} collisions "
          f"({100 * collisions / max(1, len(results)):.0f}%)")
    print("=" * 64)

    # Let CARLA settle after actor cleanup
    try:
        for _ in range(5):
            world.tick()
    except RuntimeError:
        pass
    try:
        world.apply_settings(original_settings)
    except RuntimeError:
        pass
    try:
        tm.set_synchronous_mode(False)
    except RuntimeError:
        pass


if __name__ == "__main__":
    main()
