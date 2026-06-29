#!/usr/bin/env python3
"""
Scenario S3 — Lead Vehicle at Lower Constant Speed
===================================================

NHTSA Reference: Scenario #12 — Lead Vehicle Moving at Lower Constant Speed (3.5%)

Setup:
  - NPC moves at constant 20 km/h
  - Ego on Traffic Manager autopilot, targets ~40 km/h naturally
  - Gap closes over time — does ego detect and adjust?
  - Fog density: configurable (0, 40, 70, 100)

Measures:
  - Following distance over time (should stabilise, not keep closing)
  - Speed RMSE of ego vs NPC speed (ideally ego matches NPC)
  - Time to collision (TTC) — if gap keeps closing, TTC drops below 2s
  - Minimum distance reached

Expected finding:
  - Autopilot should maintain safe following distance at all fog levels
  - This establishes the baseline for MLP comparison
"""

import argparse
import math
import os
import random
import sys
import time

import carla

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
from drivers import make_driver
from spawn_utils import get_highway_spawns, spawn_npc_in_ego_direction
from config import (
    CARLA_HOST, CARLA_PORT, DEFAULT_TOWN, FPS,
    S3_NPC_CONSTANT_SPEED_KMH,
    FOG_LADDER, RANDOM_SEEDS, SCENARIO_DURATION_S, FOG_SETTLE_STEPS,
)
def set_fog_density(world, density):
    weather = world.get_weather()
    weather.fog_density = density
    if density > 0:
        weather.fog_distance = max(5.0, 50.0 - density * 0.4)
        weather.fog_falloff = min(3.0, 0.5 + density * 0.025)
    else:
        weather.fog_distance = 100.0
        weather.fog_falloff = 0.0
    world.set_weather(weather)


def spawn_npc_ahead(world, carla_map, ego_location, ahead_m=35.0):
    """Spawn a vehicle ahead of ego on the same lane."""
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
    return vehicle


def cleanup_actor(actor):
    if actor and actor.is_alive:
        try:
            actor.destroy()
        except RuntimeError:
            pass


def run_scenario(client, world, settings, fog_density, seed, output_dir,
                 driver_name="mlp", model_dir=None, pcla_agent="tfv6_visiononly",
                 scenario_id=3):
    """Run S3: Lead Vehicle at Lower Constant Speed at a given fog density."""
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

    # Pluggable longitudinal driver; steering is delegated to BasicAgent inside it
    driver = make_driver(driver_name, model_dir=model_dir, pcla_agent=pcla_agent)
    driver.setup(world, ego, carla_map, client)

    for _ in range(20):
        control = driver.get_control(ego, world)
        ego.apply_control(control)
        world.tick()

    # Spawn NPC ahead at slower speed (direction-aware)
    npc = spawn_npc_in_ego_direction(world, carla_map, ego, ahead_m=35.0)
    if npc is None:
        cleanup_actor(ego)
        raise RuntimeError("Failed to spawn NPC vehicle")

    # NPC on Traffic Manager at constant slower speed
    tm_port = client.get_trafficmanager(8000).get_port()
    npc.set_autopilot(True, tm_port)
    tm = client.get_trafficmanager(8000)
    tm.set_desired_speed(npc, S3_NPC_CONSTANT_SPEED_KMH)
    tm.ignore_lights_percentage(npc, 100)
    tm.ignore_signs_percentage(npc, 100)
    tm.auto_lane_change(npc, False)
    tm.distance_to_leading_vehicle(npc, 100.0)  # don't slow down for anyone

    for _ in range(30):
        control = driver.get_control(ego, world)
        ego.apply_control(control)
        world.tick()

    # Position spectator behind and above ego for wide view (third-person chase cam)
    spectator = world.get_spectator()
    ego_t = ego.get_transform()
    spectator.set_transform(carla.Transform(
        ego_t.location - ego_t.get_forward_vector() * 15 + carla.Location(z=8),
        carla.Rotation(pitch=-20, yaw=ego_t.rotation.yaw)
    ))

    # Collision sensor
    collision_bp = world.get_blueprint_library().find("sensor.other.collision")
    collision_sensor = world.spawn_actor(collision_bp, carla.Transform(), attach_to=ego)
    collision_occurred = [False]

    def on_collision(event):
        collision_occurred[0] = True

    collision_sensor.listen(on_collision)

    prev_speed_mps = 0.0
    logger = GroundTruthLogger(output_dir, scenario_id, fog_density, seed)
    max_steps = SCENARIO_DURATION_S[scenario_id] * FPS

    print(f"  S3 | fog={fog_density} | seed={seed} | "
          f"NPC at constant {S3_NPC_CONSTANT_SPEED_KMH} km/h")

    try:
        for step in range(max_steps):
            # BasicAgent drives the ego
            control = driver.get_control(ego, world)
            ego.apply_control(control)

            _tick_start = time.perf_counter()
            world.tick()
            # Maintain real-time playback speed
            _elapsed = time.perf_counter() - _tick_start
            if _elapsed < 1.0 / FPS:
                time.sleep(1.0 / FPS - _elapsed)

            # Also get the applied control for logging
            control = ego.get_control() if ego.is_alive else None
            ego_speed = compute_vehicle_speed(ego)
            ego_speed_mps = ego_speed / 3.6
            accel = (ego_speed_mps - prev_speed_mps) * FPS if step > 0 else 0.0
            prev_speed_mps = ego_speed_mps

            npc_speed = compute_vehicle_speed(npc) if npc and npc.is_alive else 0.0
            dist = distance_between(ego, npc) if npc and npc.is_alive else None
            rel_vel = (ego_speed - npc_speed) / 3.6 if npc and npc.is_alive else None

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

            if collision_occurred[0]:
                print(f"    💥 COLLISION at step {step} "
                      f"(speed={ego_speed:.1f} km/h)")
                break

            if step % (FPS * 2) == 0:
                dist_str = f"{dist:.1f}m" if dist else "N/A"
                print(f"    step={step:4d}  ego={ego_speed:5.1f}km/h  "
                      f"npc={npc_speed:5.1f}km/h  dist={dist_str}")

            # Update spectator every tick (smooth chase cam)
            if ego.is_alive:
                ego_t = ego.get_transform()
                spectator.set_transform(carla.Transform(
                    ego_t.location - ego_t.get_forward_vector() * 15 + carla.Location(z=8),
                    carla.Rotation(pitch=-20, yaw=ego_t.rotation.yaw)
                ))

    finally:
        logger.close()
        driver.cleanup()
        try:
            collision_sensor.destroy()
        except RuntimeError:
            pass
        cleanup_actor(npc)
        cleanup_actor(ego)

    return logger


def main():
    parser = argparse.ArgumentParser(
        description="S3: Lead Vehicle at Lower Constant Speed (NHTSA #12)")
    parser.add_argument("--host", default=CARLA_HOST)
    parser.add_argument("--port", type=int, default=CARLA_PORT)
    parser.add_argument("--town", default=DEFAULT_TOWN)
    parser.add_argument("--fog", type=int, nargs="+", default=FOG_LADDER)
    parser.add_argument("--seeds", type=int, nargs="+", default=RANDOM_SEEDS)
    parser.add_argument("--output", default="results_s3")
    parser.add_argument("--driver", choices=["pcla", "mlp"], default="mlp",
                        help="Longitudinal control source")
    parser.add_argument("--model-dir", default="../model_throttle_brake",
                        help="MLP model directory (for --driver mlp)")
    parser.add_argument("--pcla-agent", default="tfv6_visiononly",
                        help="PCLA agent name (for --driver pcla)")
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
    print("SCENARIO S3 — LEAD VEHICLE CONSTANT SPEED (NHTSA #12)")
    print("=" * 64)
    print(f"  Fog levels:      {args.fog}")
    print(f"  Seeds:           {args.seeds}")
    print(f"  Output dir:      {args.output}")
    print(f"  NPC constant:    {S3_NPC_CONSTANT_SPEED_KMH} km/h")
    print(f"  Ego target:      40 km/h (autopilot)")
    print(f"  Runs:            {len(args.fog)} × {len(args.seeds)} = "
          f"{len(args.fog) * len(args.seeds)}")
    print("=" * 64)

    results = []
    for fog in args.fog:
        for seed in args.seeds:
            random.seed(seed)
            try:
                logger = run_scenario(client, world, settings, fog, seed,
                                      args.output, driver_name=args.driver,
                                      model_dir=args.model_dir,
                                      pcla_agent=args.pcla_agent, scenario_id=3)
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
    print("S3 SUMMARY")
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
