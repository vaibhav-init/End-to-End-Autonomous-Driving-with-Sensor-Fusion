#!/usr/bin/env python3
"""
Scenario S1 — Lead Vehicle Stopped
===================================

NHTSA Reference: Scenario #25 — Lead Vehicle Stopped (highest frequency crash, 16.4%)

Setup:
  - Straight road, Town04 highway
  - Staging: SpeedController pushes ego to 60 km/h regardless of model
  - Obstacle spawns ahead once ego reaches target speed
  - Model takes longitudinal control → must detect obstacle and brake
  - Fog density: configurable (0, 50, 100, 150)

Measures:
  - Collision rate (primary safety metric)
  - Minimum distance reached before stopping
  - Ego speed at moment of collision (if any)
  - Steps from obstacle detection to full stop (if no collision)

Expected finding:
  - Both MLP and PCLA start at identical speeds (fair comparison)
  - Fog degrades detection → affects braking distance
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
from drivers import make_driver
from spawn_utils import get_highway_spawns, spawn_obstacle_in_ego_direction
from staging import SpeedController
from scenario_weather import set_weather_condition
from config import (
    CARLA_HOST, CARLA_PORT, DEFAULT_TOWN, FPS,
    S1_OBSTACLE_DISTANCE, S1_SPAWN_SPEED_KMH, FOG_LADDER, RANDOM_SEEDS,
    SCENARIO_DURATION_S, FOG_SETTLE_STEPS,
    BACKGROUND_VEHICLES, BACKGROUND_PEDESTRIANS,
)
from driving_contract import MAX_TARGET_SPEED_KMH
from radar import add_radar_arguments

# ---------------------------------------------------------------------------
# Staging constants
# ---------------------------------------------------------------------------
STAGE_TARGET_SPEED_KMH = S1_SPAWN_SPEED_KMH
STAGE_MIN_STEPS = 60             # minimum staging steps (let model warm up)
STAGE_STABLE_S = 1.0             # require a stable speed before the event
STAGE_SPEED_TOLERANCE_KMH = 2.0
TL_CLEARANCE_M = 100.0           # min distance from traffic lights


# Weather is now handled by the shared set_weather_condition() from scenario_weather.py


def cleanup_actor(actor):
    if actor and actor.is_alive:
        try:
            actor.destroy()
        except RuntimeError:
            pass


def run_scenario(client, world, settings, fog_density, seed, output_dir,
                 driver_name="mlp", model_dir=None, pcla_agent="tfv6_visiononly",
                 radar_backend=None, radar_profile=None,
                 radar_config_path=None, radar_seed=None,
                 radar_ghost_detector=None, radar_ghost_threshold=None,
                 radar_ghost_device="cpu",
                 target_speed_kmh=STAGE_TARGET_SPEED_KMH,
                 obstacle_distance_m=S1_OBSTACLE_DISTANCE,
                 stage_stable_s=STAGE_STABLE_S,
                 stage_speed_tolerance_kmh=STAGE_SPEED_TOLERANCE_KMH,
                 scenario_id=1, safety_rules=False):
    """Run S1: Lead Vehicle Stopped at a given fog density."""
    carla_map = world.get_map()
    rng = random.Random(seed)

    # SpeedController for staging — drives ego to target speed
    speed_ctrl = SpeedController(
        target_speed_mps=target_speed_kmh / 3.6, dt=1.0 / FPS)

    # Get highway-only spawn points (multi-lane straight roads)
    highway_spawns = get_highway_spawns(carla_map)

    # Filter out spawns near traffic lights
    traffic_lights = world.get_actors().filter("traffic.traffic_light")
    tl_locations = [tl.get_location() for tl in traffic_lights]
    if tl_locations:
        clean_spawns = []
        for sp_tf in highway_spawns:
            sp_loc = sp_tf.location
            too_close = any(
                sp_loc.distance(tl_loc) < TL_CLEARANCE_M
                for tl_loc in tl_locations
            )
            if not too_close:
                clean_spawns.append(sp_tf)
        print(f"  Found {len(highway_spawns)} highway spawns, "
              f"{len(clean_spawns)} away from traffic lights")
        highway_spawns = clean_spawns
    else:
        print(f"  Found {len(highway_spawns)} highway spawn points")

    if not highway_spawns:
        raise RuntimeError("No highway spawn points found away from traffic lights")

    # Set weather (rain-to-clear gradient)
    set_weather_condition(world, fog_density)
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
    driver = make_driver(
        driver_name,
        model_dir=model_dir,
        pcla_agent=pcla_agent,
        radar_backend=radar_backend,
        radar_profile=radar_profile,
        radar_config_path=radar_config_path,
        radar_seed=seed if radar_seed is None else radar_seed,
        radar_ghost_detector=radar_ghost_detector,
        radar_ghost_threshold=radar_ghost_threshold,
        radar_ghost_device=radar_ghost_device,
        safety_rules=safety_rules,
    )
    driver.setup(world, ego, carla_map, client)

    # Warm up with speed controller (staging)
    for _ in range(60):
        control = driver.get_control(ego, world)
        ego_spd = compute_vehicle_speed(ego) / 3.6  # m/s
        thr, brk = speed_ctrl.run_step(ego_spd)
        control = carla.VehicleControl(throttle=thr, brake=brk, steer=control.steer)
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
    logger = GroundTruthLogger(
        output_dir,
        scenario_id,
        fog_density,
        seed,
        target_speed_kmh=target_speed_kmh,
        event_distance_m=obstacle_distance_m,
    )

    obstacle = None
    obstacle_spawned = False
    max_steps = SCENARIO_DURATION_S[scenario_id] * FPS
    stable_speed_frames = 0
    required_stable_frames = max(1, int(stage_stable_s * FPS))

    print(f"  S1 | fog={fog_density} | seed={seed} | "
          f"spawned at ({ego.get_location().x:.0f}, {ego.get_location().y:.0f})")

    prev_speed = 0.0

    try:
        for step in range(max_steps):
            # Driver runs every tick (keeps model warm + supplies steer)
            control = driver.get_control(ego, world)

            # ── Staging: speed controller holds ego at target speed ──
            # Until the obstacle spawns, the speed controller overrides
            # the model's longitudinal output to ensure both MLP and PCLA
            # start the critical phase at identical speeds.
            ego_spd_kmh = compute_vehicle_speed(ego)

            if not obstacle_spawned:
                # Still staging — force speed
                ego_spd_mps = ego_spd_kmh / 3.6
                thr, brk = speed_ctrl.run_step(ego_spd_mps)
                control = carla.VehicleControl(
                    throttle=thr, brake=brk, steer=control.steer)

                speed_is_stable = (
                    abs(ego_spd_kmh - target_speed_kmh)
                    <= stage_speed_tolerance_kmh
                )
                stable_speed_frames = (
                    stable_speed_frames + 1 if speed_is_stable else 0
                )

                # Spawn only after the requested speed has actually stabilized.
                if (step >= STAGE_MIN_STEPS
                        and stable_speed_frames >= required_stable_frames):
                    obstacle = spawn_obstacle_in_ego_direction(
                        world, carla_map, ego, obstacle_distance_m)
                    if obstacle:
                        obstacle_spawned = True
                        obs_dist = ego.get_location().distance(
                            obstacle.get_location())
                        print(f"    🚧 Obstacle spawned {obs_dist:.0f}m ahead "
                              f"at {ego_spd_kmh:.0f} km/h (step {step})")
                        print(f"    🤝 Handover: model takes longitudinal "
                              f"control at {ego_spd_kmh:.1f} km/h")
            # else: model has full control (staging done)

            ego.apply_control(control)

            _tick_start = time.perf_counter()
            world.tick()
            # Maintain real-time playback speed
            _elapsed = time.perf_counter() - _tick_start
            if _elapsed < 1.0 / FPS:
                time.sleep(1.0 / FPS - _elapsed)

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
                critical_event=obstacle_spawned,
                collision=collision_occurred[0],
                ego_accel=accel,
                radar_diagnostics=driver.diagnostics(),
            )

            # Stop early if collision
            if collision_occurred[0]:
                print(f"    💥 COLLISION at step {step} "
                      f"(speed={ego_speed:.1f} km/h)")
                break

            # Stop early if ego stopped near obstacle
            if obstacle_spawned and obstacle and obstacle.is_alive:
                if dist is not None and dist < 1.0:
                    print(f"    ✅ Stopped at step {step} (dist={dist:.1f}m)")
                    break
                if ego_speed < 1.0 and dist is not None and dist < 10.0:
                    print(f"    ✅ Stopped safely at step {step} "
                          f"(dist={dist:.1f}m)")
                    break

            # Log progress
            if step % (FPS * 2) == 0:
                if obstacle_spawned:
                    dist_str = f"{dist:.1f}m" if dist else "N/A"
                    print(f"    step={step:4d}  spd={ego_speed:5.1f}km/h  "
                          f"dist={dist_str}")
                else:
                    print(f"    step={step:4d}  spd={ego_speed:5.1f}km/h  "
                          f"[staging]")

            # Update spectator every tick (smooth chase cam)
            if ego.is_alive:
                ego_t = ego.get_transform()
                spectator.set_transform(carla.Transform(
                    ego_t.location - ego_t.get_forward_vector() * 15
                    + carla.Location(z=8),
                    carla.Rotation(pitch=-20, yaw=ego_t.rotation.yaw)
                ))

    finally:
        logger.close()
        driver.cleanup()
        try:
            collision_sensor.destroy()
        except RuntimeError:
            pass
        cleanup_actor(obstacle)
        cleanup_actor(ego)

    if not obstacle_spawned:
        try:
            os.remove(logger.filepath)
        except OSError:
            pass
        raise RuntimeError(
            "S1 never reached a stable staged speed or could not spawn the "
            "obstacle; invalid CSV removed"
        )

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
    parser.add_argument(
        "--safety-rules",
        action="store_true",
        help=(
            "re-enable the hardcoded emergency-brake overrides in the mlp "
            "driver (ablation arm). Off by default so the model decides."
        ),
    )
    parser.add_argument("--driver", choices=["pcla", "mlp", "idm"], default="mlp",
                        help="Longitudinal control source")
    parser.add_argument("--model-dir", default="../model_throttle_brake",
                        help="MLP model directory (for --driver mlp)")
    parser.add_argument("--pcla-agent", default="tfv6_visiononly",
                        help="PCLA agent name (for --driver pcla)")
    add_radar_arguments(parser)
    parser.add_argument("--headless", action="store_true", help="No spectator camera")
    parser.add_argument(
        "--target-speed-kmh",
        type=float,
        default=STAGE_TARGET_SPEED_KMH,
        help="Stable ego speed required before spawning the stopped obstacle",
    )
    parser.add_argument(
        "--obstacle-distance-m",
        type=float,
        default=S1_OBSTACLE_DISTANCE,
        help="Stopped-obstacle spawn distance",
    )
    parser.add_argument(
        "--stage-stable-s",
        type=float,
        default=STAGE_STABLE_S,
        help="Time the ego must remain near target speed before the event",
    )
    parser.add_argument(
        "--stage-speed-tolerance-kmh",
        type=float,
        default=STAGE_SPEED_TOLERANCE_KMH,
        help="Allowed speed error while declaring the staged state stable",
    )
    args = parser.parse_args()
    if not 0.0 < args.target_speed_kmh <= MAX_TARGET_SPEED_KMH:
        parser.error(
            f"--target-speed-kmh must be in (0, {MAX_TARGET_SPEED_KMH:g}]"
        )
    if min(
        args.obstacle_distance_m,
        args.stage_stable_s,
        args.stage_speed_tolerance_kmh,
    ) <= 0.0:
        parser.error("S1 distance and staging values must be positive")

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
    print(f"  Obstacle at:     {args.obstacle_distance_m:.1f}m")
    print(f"  Staging target:  {args.target_speed_kmh:.1f}km/h")
    print(
        f"  Stable state:    {args.stage_stable_s:.1f}s within "
        f"±{args.stage_speed_tolerance_kmh:.1f}km/h"
    )
    print(f"  Runs:            {len(args.fog)} × {len(args.seeds)} = "
          f"{len(args.fog) * len(args.seeds)}")
    print("=" * 64)

    results = []
    for fog in args.fog:
        for seed in args.seeds:
            try:
                logger = run_scenario(client, world, settings, fog, seed,
                                      args.output, driver_name=args.driver,
                                      model_dir=args.model_dir,
                                      pcla_agent=args.pcla_agent,
                                      radar_backend=args.radar_backend,
                                      radar_profile=args.radar_profile,
                                      radar_config_path=args.radar_config,
                                      radar_seed=args.radar_seed,
                                      radar_ghost_detector=args.radar_ghost_detector,
                                      radar_ghost_threshold=args.radar_ghost_threshold,
                                      radar_ghost_device=args.radar_ghost_device,
                                      safety_rules=args.safety_rules,
                                      target_speed_kmh=args.target_speed_kmh,
                                      obstacle_distance_m=args.obstacle_distance_m,
                                      stage_stable_s=args.stage_stable_s,
                                      stage_speed_tolerance_kmh=(
                                          args.stage_speed_tolerance_kmh
                                      ),
                                      scenario_id=1)
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
