#!/usr/bin/env python3
"""
Scenario S2 — Lead Vehicle Decelerating
========================================

NHTSA Reference: Scenario #4 — Lead Vehicle Decelerating (7.2%, 4th most frequent)

Setup:
  - NPC starts at 30 km/h, 25m ahead of ego
  - Ego is staged to a stable matched-speed/matched-gap state
  - The evaluated driver controls for a settling interval before the NPC brakes
  - Fog density: configurable (0, 40, 70, 100)

Measures:
  - Reaction distance (distance at which ego begins braking after NPC brakes)
  - Impact speed if collision occurs
  - Speed error profile during deceleration event
  - Minimum distance reached

Expected finding:
  - Autopilot baseline handles this well in all fog
  - Sets the baseline for MLP comparison
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
from staging import GapKeepController
from spawn_utils import get_highway_spawns, spawn_npc_in_ego_direction
from scenario_weather import set_weather_condition
from config import (
    CARLA_HOST, CARLA_PORT, DEFAULT_TOWN, FPS,
    S2_NPC_INITIAL_GAP, S2_NPC_SPEED_KMH, S2_BRAKE_TRIGGER_STEP,
    FOG_LADDER, RANDOM_SEEDS, SCENARIO_DURATION_S, FOG_SETTLE_STEPS,
)
from driving_contract import MAX_TARGET_SPEED_KMH, S2_HANDOVER_SETTLE_S
from radar import add_radar_arguments

# Weather is now handled by the shared set_weather_condition() from scenario_weather.py



def spawn_npc_ahead(world, carla_map, ego_location, ahead_m):
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
                 radar_backend=None, radar_profile=None,
                 radar_config_path=None, radar_seed=None,
                 radar_ghost_detector=None, radar_ghost_threshold=None,
                 radar_ghost_device="cpu",
                 stage_approach=True,
                 stage_gap=S2_NPC_INITIAL_GAP,
                 initial_gap=S2_NPC_INITIAL_GAP,
                 target_speed_kmh=S2_NPC_SPEED_KMH,
                 handover_settle_s=S2_HANDOVER_SETTLE_S,
                 stage_timeout_s=12.0,
                 scenario_id=2, safety_rules=False):
    """Run S2: Lead Vehicle Decelerating at a given fog density."""
    carla_map = world.get_map()
    rng = random.Random(seed)

    # Get highway-only spawn points (multi-lane straight roads)
    highway_spawns = get_highway_spawns(carla_map)
    if not highway_spawns:
        raise RuntimeError("No highway spawn points found in this map")
    print(f"  Found {len(highway_spawns)} highway spawn points")

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

    for _ in range(20):
        control = driver.get_control(ego, world)
        ego.apply_control(control)
        world.tick()

    # Spawn NPC ahead (direction-aware)
    npc = spawn_npc_in_ego_direction(world, carla_map, ego, initial_gap)
    if npc is None:
        cleanup_actor(ego)
        raise RuntimeError("Failed to spawn NPC vehicle")

    # NPC on Traffic Manager at constant speed
    tm_port = client.get_trafficmanager(8000).get_port()
    npc.set_autopilot(True, tm_port)
    tm = client.get_trafficmanager(8000)
    tm.set_desired_speed(npc, target_speed_kmh)
    tm.ignore_lights_percentage(npc, 100)
    tm.ignore_signs_percentage(npc, 100)
    tm.auto_lane_change(npc, False)

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
    npc_braked = False
    # Optional staging: a gap-keep controller tailgates the NPC until it brakes,
    # then control is handed to the driver/model for the deceleration response.
    gapkeep = (
        GapKeepController(
            stage_gap,
            dt=1.0 / FPS,
            max_speed_mps=(target_speed_kmh + 5.0) / 3.6,
        )
        if stage_approach
        else None
    )
    stable_frames = 0
    required_stable_frames = FPS
    handover_step = None
    brake_trigger_step = (
        None if stage_approach else S2_BRAKE_TRIGGER_STEP
    )
    settle_frames = max(1, int(handover_settle_s * FPS))
    stage_timeout_steps = max(1, int(stage_timeout_s * FPS))
    if gapkeep is not None:
        print(
            f"  Staging ON: stabilize at {target_speed_kmh:.0f}km/h and "
            f"{stage_gap:.0f}m, then give the driver {handover_settle_s:.1f}s "
            "before NPC braking"
        )

    print(f"  S2 | fog={fog_density} | seed={seed} | "
          f"NPC at {target_speed_kmh} km/h, {initial_gap}m ahead")

    try:
        for step in range(max_steps):
            # Driver runs every tick (keeps an end-to-end model warm + supplies steer)
            control = driver.get_control(ego, world)
            # Establish the same pre-event state for both drivers. Handover is
            # deliberately separated from the NPC brake, so reaction latency
            # measures the braking event rather than a controller transition.
            if (
                gapkeep is not None
                and handover_step is None
                and npc
                and npc.is_alive
            ):
                gap = distance_between(ego, npc)
                ego_kmh = compute_vehicle_speed(ego)
                npc_kmh = compute_vehicle_speed(npc)
                thr, brk = gapkeep.run_step(
                    gap,
                    ego_kmh / 3.6,
                    npc_kmh / 3.6,
                )
                control = carla.VehicleControl(
                    throttle=thr,
                    brake=brk,
                    steer=control.steer,
                )
                stable = (
                    abs(gap - stage_gap) <= 2.5
                    and abs(ego_kmh - target_speed_kmh) <= 1.5
                    and abs(npc_kmh - target_speed_kmh) <= 1.5
                )
                stable_frames = stable_frames + 1 if stable else 0
                if (
                    stable_frames >= required_stable_frames
                    or step >= stage_timeout_steps
                ):
                    handover_step = step
                    brake_trigger_step = step + settle_frames
                    reason = (
                        "stable state reached"
                        if stable_frames >= required_stable_frames
                        else "staging timeout"
                    )
                    print(
                        f"    Handover at step {step}: {reason}; "
                        f"ego={ego_kmh:.1f}km/h gap={gap:.1f}m. "
                        f"NPC brakes at step {brake_trigger_step}."
                    )
            ego.apply_control(control)

            if (
                brake_trigger_step is not None
                and step >= brake_trigger_step
                and not npc_braked
                and npc
                and npc.is_alive
            ):
                npc.set_autopilot(False)
                npc.apply_control(
                    carla.VehicleControl(throttle=0.0, brake=1.0)
                )
                npc_braked = True
                print(f"    NPC slammed brakes at step {step}")

            _tick_start = time.perf_counter()
            world.tick()
            # Maintain real-time playback speed
            _elapsed = time.perf_counter() - _tick_start
            if _elapsed < 1.0 / FPS:
                time.sleep(1.0 / FPS - _elapsed)

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
                critical_event=npc_braked,
                collision=collision_occurred[0],
                ego_accel=accel,
                radar_diagnostics=driver.diagnostics(),
            )

            if collision_occurred[0]:
                print(f"    💥 COLLISION at step {step} "
                      f"(speed={ego_speed:.1f} km/h, dist={dist:.1f}m)")
                break

            # Stop early if ego stopped near NPC after it braked
            if npc_braked and ego_speed < 1.0 and dist is not None and dist < 10.0:
                print(f"    ✅ Stopped safely at step {step} "
                      f"(dist={dist:.1f}m)")
                break

            if step % (FPS * 2) == 0:
                npc_status = "braking" if npc_braked else "cruising"
                dist_str = f"{dist:.1f}m" if dist else "N/A"
                print(f"    step={step:4d}  ego={ego_speed:5.1f}km/h  "
                      f"npc={npc_speed:5.1f}km/h  dist={dist_str}  {npc_status}")

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
        if npc and npc.is_alive:
            try:
                npc.set_autopilot(False)
            except RuntimeError:
                pass
        cleanup_actor(npc)
        cleanup_actor(ego)

    return logger


def main():
    parser = argparse.ArgumentParser(description="S2: Lead Vehicle Decelerating (NHTSA #4)")
    parser.add_argument("--host", default=CARLA_HOST)
    parser.add_argument("--port", type=int, default=CARLA_PORT)
    parser.add_argument("--town", default=DEFAULT_TOWN)
    parser.add_argument("--fog", type=int, nargs="+", default=FOG_LADDER)
    parser.add_argument("--seeds", type=int, nargs="+", default=RANDOM_SEEDS)
    parser.add_argument("--output", default="results_s2")
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
    parser.add_argument("--stage-approach", action="store_true", default=True,
                        help="Tailgate the NPC with a gap-keeper, hand to the model when it brakes (default: on)")
    parser.add_argument("--no-stage-approach", dest="stage_approach",
                        action="store_false",
                        help="Disable staging; model controls from the start")
    parser.add_argument("--stage-gap", type=float, default=S2_NPC_INITIAL_GAP,
                        help="Target following gap in metres during staging")
    parser.add_argument(
        "--initial-gap",
        type=float,
        default=S2_NPC_INITIAL_GAP,
        help="NPC spawn gap in metres",
    )
    parser.add_argument(
        "--target-speed-kmh",
        type=float,
        default=S2_NPC_SPEED_KMH,
        help="Shared ego/NPC pre-event target speed",
    )
    parser.add_argument(
        "--handover-settle-s",
        type=float,
        default=S2_HANDOVER_SETTLE_S,
        help="Driver-only control time between staging and NPC braking",
    )
    parser.add_argument(
        "--stage-timeout-s",
        type=float,
        default=12.0,
        help="Maximum time to wait for a stable staged state",
    )
    args = parser.parse_args()
    if min(
        args.stage_gap,
        args.initial_gap,
        args.target_speed_kmh,
        args.handover_settle_s,
        args.stage_timeout_s,
    ) <= 0.0:
        parser.error("S2 speed, gaps, and timing values must be positive")
    if args.target_speed_kmh > MAX_TARGET_SPEED_KMH:
        parser.error(
            f"--target-speed-kmh cannot exceed {MAX_TARGET_SPEED_KMH:g}"
        )

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
    print("SCENARIO S2 — LEAD VEHICLE DECELERATING (NHTSA #4)")
    print("=" * 64)
    print(f"  Fog levels:      {args.fog}")
    print(f"  Seeds:           {args.seeds}")
    print(f"  Output dir:      {args.output}")
    print(f"  NPC speed:       {args.target_speed_kmh} km/h")
    print(f"  NPC initial gap: {args.initial_gap:.0f}m")
    print(f"  Staging gap:     {args.stage_gap:.0f}m")
    print(f"  Settle interval: {args.handover_settle_s:.1f}s")
    print(f"  Runs:            {len(args.fog)} × {len(args.seeds)} = {len(args.fog) * len(args.seeds)}")
    print("=" * 64)

    results = []
    for fog in args.fog:
        for seed in args.seeds:
            random.seed(seed)
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
                                      stage_approach=args.stage_approach,
                                      stage_gap=args.stage_gap,
                                      initial_gap=args.initial_gap,
                                      target_speed_kmh=args.target_speed_kmh,
                                      handover_settle_s=args.handover_settle_s,
                                      stage_timeout_s=args.stage_timeout_s,
                                      scenario_id=2)
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
    print("S2 SUMMARY")
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
