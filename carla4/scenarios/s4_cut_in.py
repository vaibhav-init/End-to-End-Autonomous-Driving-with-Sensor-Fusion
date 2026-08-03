#!/usr/bin/env python3
"""
Scenario S4 — Cut-In from Adjacent Lane
=========================================

NHTSA Reference: Pre-crash scenario — Vehicle cuts into ego's lane (~10% of
rear-end crashes involve a lane-changing vehicle).

Setup:
  - Ego drives on highway at ~60 km/h (BasicAgent)
  - NPC drives in ADJACENT lane, 25m ahead at 50 km/h
  - At step 200 (~10s), NPC changes lane into ego's lane
  - Ego must detect the now-close vehicle and brake to avoid collision

Why this matters for camera vs radar:
  - Radar detects the NPC instantly when it enters the ego's lane
  - Camera/YOLO needs 1-2 frames to detect and estimate distance
  - This latency difference is the core thesis comparison

Measures:
  - Collision rate (primary safety metric)
  - Reaction time: steps from cut-in completion to ego braking
  - Minimum distance after cut-in
  - Ego speed at moment of collision (if any)
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
from spawn_utils import get_highway_spawns
from scenario_weather import set_weather_condition
from config import (
    CARLA_HOST, CARLA_PORT, DEFAULT_TOWN, FPS,
    S4_NPC_SPEED_KMH, S4_NPC_AHEAD_M, S4_CUT_IN_TRIGGER_STEP,
    FOG_LADDER, RANDOM_SEEDS, SCENARIO_DURATION_S, FOG_SETTLE_STEPS,
)
from radar import add_radar_arguments



# Weather is now handled by the shared set_weather_condition() from scenario_weather.py



# Staging constants: NPC spawns just barely ahead in adjacent lane so when
# it cuts in it's RIGHT IN FRONT of the ego — a dangerous, realistic cut-in.
STAGE_NPC_AHEAD_M = 5.0    # very close ahead; cut-in lands right in front — tight!
STAGE_CUT_IN_STEP = 80     # earliest step the cut-in can fire — earlier
STAGE_MIN_SPEED_KMH = 45.0 # both vehicles must exceed this before cut-in fires
TL_CLEARANCE_M = 100.0     # min distance from traffic lights for spawn points


def spawn_npc_adjacent_lane(world, carla_map, ego, ahead_m):
    """
    Spawn a vehicle in an adjacent lane, ahead_m metres ahead of ego.

    Finds the ego's current lane waypoint, gets the adjacent lane (left or right),
    then walks ahead_m along that adjacent lane.

    Returns (vehicle, lane_direction) where lane_direction is True for right, False for left.
    """
    ego_tf = ego.get_transform()
    ego_loc = ego_tf.location
    ego_fwd = ego_tf.get_forward_vector()

    wp = carla_map.get_waypoint(ego_loc, project_to_road=True,
                                lane_type=carla.LaneType.Driving)
    if wp is None:
        return None, None

    # Check waypoint direction vs ego heading
    wp_fwd = wp.transform.get_forward_vector()
    dot = ego_fwd.x * wp_fwd.x + ego_fwd.y * wp_fwd.y
    advance_fn = (lambda w, d: w.next(d)) if dot >= 0 else (lambda w, d: w.previous(d))

    # Find adjacent lane (prefer right, fallback to left)
    adj_wp = None
    cut_in_direction = None  # True = NPC is to the right, will cut left into ego lane

    right = wp.get_right_lane()
    if right and right.lane_type == carla.LaneType.Driving and right.lane_id * wp.lane_id > 0:
        adj_wp = right
        cut_in_direction = True  # NPC to the right

    if adj_wp is None:
        left = wp.get_left_lane()
        if left and left.lane_type == carla.LaneType.Driving and left.lane_id * wp.lane_id > 0:
            adj_wp = left
            cut_in_direction = False  # NPC to the left

    if adj_wp is None:
        return None, None

    # Walk ahead_m along the adjacent lane
    travelled = 0.0
    step = 3.0
    while travelled < ahead_m:
        next_wps = advance_fn(adj_wp, step)
        if not next_wps:
            return None, None
        adj_wp = next_wps[0]
        travelled += step

    # Verify it's ahead of ego
    target_loc = adj_wp.transform.location
    to_target_x = target_loc.x - ego_loc.x
    to_target_y = target_loc.y - ego_loc.y
    dot_check = ego_fwd.x * to_target_x + ego_fwd.y * to_target_y
    if dot_check < 0:
        return None, None

    bp_lib = world.get_blueprint_library()
    # Use a sedan-sized car — trucks/buses accelerate too slowly and
    # behave unrealistically in a cut-in scenario.
    bp = bp_lib.find("vehicle.audi.a2")
    transform = adj_wp.transform
    transform.location.z += 0.5
    vehicle = world.try_spawn_actor(bp, transform)
    return vehicle, cut_in_direction


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
                 stage_approach=True, stage_gap=8.0,
                 cutin_stop=True,
                 scenario_id=4):
    """Run S4: Cut-In from Adjacent Lane at a given fog density."""
    carla_map = world.get_map()
    rng = random.Random(seed)

    # Staging: gap-keeper holds ego at a fixed gap behind the NPC (in the
    # adjacent lane) until both reach highway speed, then the NPC cuts in.
    npc_ahead_m = STAGE_NPC_AHEAD_M if stage_approach else S4_NPC_AHEAD_M
    cut_in_step = STAGE_CUT_IN_STEP if stage_approach else S4_CUT_IN_TRIGGER_STEP
    gapkeep = GapKeepController(stage_gap, dt=1.0 / FPS,
                                max_speed_mps=25.0) if stage_approach else None
    handover_announced = False

    # Get highway-only spawn points (multi-lane straight roads)
    highway_spawns = get_highway_spawns(carla_map)

    # Filter out spawns near traffic lights so the ego gets a clean highway run
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
    )
    driver.setup(world, ego, carla_map, client)

    # Spawn NPC in adjacent lane BEFORE any warmup so both vehicles
    # start from rest together and the gap-keeper can keep them matched.
    npc, cut_in_direction = spawn_npc_adjacent_lane(
        world, carla_map, ego, npc_ahead_m)
    if npc is None:
        cleanup_actor(ego)
        raise RuntimeError("Failed to spawn NPC in adjacent lane (no parallel lane?)")

    direction_str = "right" if cut_in_direction else "left"
    print(f"  NPC spawned in {direction_str} lane, {npc_ahead_m:.0f}m ahead")

    # NPC on Traffic Manager at constant speed in adjacent lane
    tm = client.get_trafficmanager(8000)
    tm_port = tm.get_port()
    npc.set_autopilot(True, tm_port)
    tm.set_desired_speed(npc, S4_NPC_SPEED_KMH)
    tm.ignore_lights_percentage(npc, 100)
    tm.ignore_signs_percentage(npc, 100)
    tm.auto_lane_change(npc, False)  # don't change lane on its own

    # Warm up BOTH vehicles together. The gap-keeper (if staging) keeps the
    # ego matched to the NPC so they accelerate in formation.
    for _ in range(60):
        control = driver.get_control(ego, world)
        if gapkeep is not None and npc and npc.is_alive:
            gap = distance_between(ego, npc)
            ego_spd = compute_vehicle_speed(ego) / 3.6
            npc_spd = compute_vehicle_speed(npc) / 3.6
            thr, brk = gapkeep.run_step(gap, ego_spd, npc_spd)
            control = carla.VehicleControl(throttle=thr, brake=brk,
                                           steer=control.steer)
        ego.apply_control(control)
        world.tick()

    # Position spectator
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
    cut_in_triggered = False
    cut_in_complete = False
    cut_in_complete_step = None

    if gapkeep is not None:
        print(f"  Staging ON: hold {stage_gap:.0f}m gap, hand over when "
              f"both vehicles >{STAGE_MIN_SPEED_KMH:.0f} km/h (earliest step {cut_in_step})")
    if cutin_stop:
        print("  Cut-in mode: NPC brakes to a full stop after cutting in")
    print(f"  S4 | fog={fog_density} | seed={seed} | "
          f"NPC at {S4_NPC_SPEED_KMH} km/h in adjacent lane, cut-in at step {cut_in_step}")

    try:
        for step in range(max_steps):
            # Driver runs every tick (keeps an end-to-end model warm + supplies steer)
            control = driver.get_control(ego, world)
            # Staging: hold a fixed gap behind the NPC until the cut-in fires,
            # then hand longitudinal control to the model under test.
            # Staging: gap-keeper holds until BOTH the minimum step has passed
            # AND both vehicles are above the speed threshold.
            if gapkeep is not None and not cut_in_triggered and npc and npc.is_alive:
                ego_spd_kmh = compute_vehicle_speed(ego)
                npc_spd_kmh = compute_vehicle_speed(npc)
                speeds_ok = (ego_spd_kmh >= STAGE_MIN_SPEED_KMH
                             and npc_spd_kmh >= STAGE_MIN_SPEED_KMH)
                if step < cut_in_step or not speeds_ok:
                    # Still staging — hold the gap
                    gap = distance_between(ego, npc)
                    thr, brk = gapkeep.run_step(gap, ego_spd_kmh / 3.6,
                                                npc_spd_kmh / 3.6)
                    control = carla.VehicleControl(throttle=thr, brake=brk, steer=control.steer)
                elif not handover_announced:
                    handover_announced = True
                    print(f"    🤝 Handover: model takes longitudinal control at step {step} "
                          f"(ego={ego_spd_kmh:.1f} npc={npc_spd_kmh:.1f} km/h)")
            ego.apply_control(control)

            _tick_start = time.perf_counter()
            world.tick()
            _elapsed = time.perf_counter() - _tick_start
            if _elapsed < 1.0 / FPS:
                time.sleep(1.0 / FPS - _elapsed)

            # Trigger cut-in: must pass minimum step AND both vehicles above speed gate
            if step >= cut_in_step and not cut_in_triggered and npc and npc.is_alive:
                ego_spd_now = compute_vehicle_speed(ego)
                npc_spd_now = compute_vehicle_speed(npc)
                if ego_spd_now >= STAGE_MIN_SPEED_KMH and npc_spd_now >= STAGE_MIN_SPEED_KMH:
                    npc.set_autopilot(False)
                    cut_in_triggered = True
                    print(f"    🔀 NPC cut-in triggered at step {step} "
                          f"(from {direction_str} lane) — manual steering "
                          f"(ego={ego_spd_now:.1f} npc={npc_spd_now:.1f} km/h)")

            # Manually steer NPC into ego's lane
            if cut_in_triggered and not cut_in_complete and npc and npc.is_alive:
                npc_loc = npc.get_location()
                npc_wp = carla_map.get_waypoint(npc_loc, project_to_road=True,
                                                 lane_type=carla.LaneType.Driving)
                ego_wp = carla_map.get_waypoint(ego.get_location(), project_to_road=True,
                                                 lane_type=carla.LaneType.Driving)

                if ego_wp and npc_wp:
                    if npc_wp.lane_id != ego_wp.lane_id:
                        # Get the ADJACENT lane (one lane toward ego), not ego's lane directly
                        # This prevents overshooting across multiple lanes
                        if cut_in_direction:
                            # NPC is to the right of ego — move left
                            adj_lane = npc_wp.get_left_lane()
                        else:
                            # NPC is to the left of ego — move right
                            adj_lane = npc_wp.get_right_lane()

                        if adj_lane and adj_lane.lane_type == carla.LaneType.Driving:
                            # Target: 10m ahead on the adjacent lane
                            npc_fwd = npc.get_transform().get_forward_vector()
                            wp_fwd = adj_lane.transform.get_forward_vector()
                            dot = npc_fwd.x * wp_fwd.x + npc_fwd.y * wp_fwd.y
                            adv = (lambda w, d: w.next(d)) if dot >= 0 else (lambda w, d: w.previous(d))
                            ahead = adv(adj_lane, 10.0)
                            target_loc = ahead[0].transform.location if ahead else adj_lane.transform.location

                            # Compute steering toward target
                            npc_tf = npc.get_transform()
                            dx = target_loc.x - npc_tf.location.x
                            dy = target_loc.y - npc_tf.location.y
                            target_yaw = math.atan2(dy, dx)
                            npc_yaw = math.radians(npc_tf.rotation.yaw)
                            yaw_err = (target_yaw - npc_yaw + math.pi) % (2 * math.pi) - math.pi
                            steer_npc = max(-0.4, min(0.4, yaw_err / math.radians(30)))

                            # Maintain speed
                            npc_speed_now = compute_vehicle_speed(npc) / 3.6
                            target_speed_mps = S4_NPC_SPEED_KMH / 3.6
                            if npc_speed_now < target_speed_mps - 0.5:
                                throttle_npc = 0.6
                            elif npc_speed_now < target_speed_mps:
                                throttle_npc = 0.3
                            else:
                                throttle_npc = 0.1

                            npc.apply_control(carla.VehicleControl(
                                throttle=throttle_npc, steer=steer_npc, brake=0.0))
                    else:
                        # NPC is now in ego's lane — cut-in complete!
                        cut_in_complete = True
                        cut_in_complete_step = step
                        dist_at_cutin = distance_between(ego, npc)
                        print(f"    ✅ Cut-in complete at step {step} "
                              f"— NPC now in ego's lane, dist={dist_at_cutin:.1f}m")

            # After cut-in is complete: NPC holds speed, or brakes to a stop (cutin_stop)
            if cut_in_complete and npc and npc.is_alive:
                npc_wp = carla_map.get_waypoint(npc.get_location(), project_to_road=True,
                                                 lane_type=carla.LaneType.Driving)
                if npc_wp:
                    npc_tf = npc.get_transform()
                    npc_fwd = npc_tf.get_forward_vector()
                    wp_fwd = npc_wp.transform.get_forward_vector()
                    dot = npc_fwd.x * wp_fwd.x + npc_fwd.y * wp_fwd.y
                    adv = (lambda w, d: w.next(d)) if dot >= 0 else (lambda w, d: w.previous(d))
                    ahead = adv(npc_wp, 10.0)
                    if ahead:
                        target_loc = ahead[0].transform.location
                        dx = target_loc.x - npc_tf.location.x
                        dy = target_loc.y - npc_tf.location.y
                        target_yaw = math.atan2(dy, dx)
                        npc_yaw = math.radians(npc_tf.rotation.yaw)
                        yaw_err = (target_yaw - npc_yaw + math.pi) % (2 * math.pi) - math.pi
                        steer_npc = max(-0.3, min(0.3, yaw_err / math.radians(45)))

                        npc_speed_now = compute_vehicle_speed(npc) / 3.6
                        if cutin_stop:
                            # Full emergency brake after cut-in — the model must
                            # react immediately or it will rear-end the NPC.
                            throttle_npc = 0.0
                            brake_npc = 1.0
                        else:
                            target_speed_mps = S4_NPC_SPEED_KMH / 3.6 * 0.8  # slow a bit after cut-in
                            brake_npc = 0.0
                            if npc_speed_now < target_speed_mps - 0.5:
                                throttle_npc = 0.5
                            elif npc_speed_now < target_speed_mps:
                                throttle_npc = 0.2
                            else:
                                throttle_npc = 0.0

                        npc.apply_control(carla.VehicleControl(
                            throttle=throttle_npc, steer=steer_npc, brake=brake_npc))

            # Measurements
            applied_control = ego.get_control() if ego.is_alive else None
            ego_speed = compute_vehicle_speed(ego)
            ego_speed_mps = ego_speed / 3.6
            accel = (ego_speed_mps - prev_speed_mps) * FPS if step > 0 else 0.0
            prev_speed_mps = ego_speed_mps

            npc_speed = compute_vehicle_speed(npc) if npc and npc.is_alive else 0.0
            dist = distance_between(ego, npc) if npc and npc.is_alive else None
            rel_vel = (ego_speed - npc_speed) / 3.6 if npc and npc.is_alive else None

            throttle = applied_control.throttle if applied_control else 0.0
            brake = applied_control.brake if applied_control else 0.0
            steer = applied_control.steer if applied_control else 0.0

            logger.log(
                step=step,
                ego_speed_kmh=ego_speed,
                npc_speed_kmh=npc_speed,
                distance_to_npc=dist,
                relative_velocity=rel_vel,
                throttle=throttle,
                brake=brake,
                steer=steer,
                critical_event=cut_in_triggered,
                collision=collision_occurred[0],
                ego_accel=accel,
                radar_diagnostics=driver.diagnostics(),
            )

            if collision_occurred[0]:
                print(f"    💥 COLLISION at step {step} "
                      f"(speed={ego_speed:.1f} km/h, dist={dist:.1f}m)")
                break

            # Stop early if ego stopped near NPC after cut-in completed
            if cut_in_complete and ego_speed < 1.0 and dist is not None and dist < 10.0:
                print(f"    ✅ Stopped safely at step {step} "
                      f"(dist={dist:.1f}m)")
                break

            # Log progress every 2 seconds
            if step % (FPS * 2) == 0:
                phase = "adjacent" if not cut_in_triggered else (
                    "cutting-in" if not cut_in_complete else "in-lane")
                dist_str = f"{dist:.1f}m" if dist else "N/A"
                print(f"    step={step:4d}  ego={ego_speed:5.1f}km/h  "
                      f"npc={npc_speed:5.1f}km/h  dist={dist_str}  {phase}")

            # Update spectator
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
    parser = argparse.ArgumentParser(
        description="S4: Cut-In from Adjacent Lane")
    parser.add_argument("--host", default=CARLA_HOST)
    parser.add_argument("--port", type=int, default=CARLA_PORT)
    parser.add_argument("--town", default=DEFAULT_TOWN)
    parser.add_argument("--fog", type=int, nargs="+", default=FOG_LADDER)
    parser.add_argument("--seeds", type=int, nargs="+", default=RANDOM_SEEDS)
    parser.add_argument("--output", default="results_s4")
    parser.add_argument("--driver", choices=["pcla", "mlp"], default="mlp",
                        help="Longitudinal control source")
    parser.add_argument("--model-dir", default="../model_throttle_brake",
                        help="MLP model directory (for --driver mlp)")
    parser.add_argument("--pcla-agent", default="tfv6_visiononly",
                        help="PCLA agent name (for --driver pcla)")
    add_radar_arguments(parser)
    parser.add_argument("--stage-approach", action="store_true", default=True,
                        help="Stage the scenario: gap-keeper holds close follow, "
                             "then hand over on cut-in (default: on)")
    parser.add_argument("--no-stage-approach", dest="stage_approach",
                        action="store_false",
                        help="Disable staging; cut-in fires at a fixed step")
    parser.add_argument("--stage-gap", type=float, default=8.0,
                        help="Target following gap in metres during staging (default: 8)")
    parser.add_argument("--cutin-stop", action="store_true", default=True,
                        help="NPC brakes to a full stop after cutting in (default: on)")
    parser.add_argument("--no-cutin-stop", dest="cutin_stop", action="store_false",
                        help="NPC holds speed after cutting in instead of braking")
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

    npc_ahead = STAGE_NPC_AHEAD_M if args.stage_approach else S4_NPC_AHEAD_M
    cutin_step = STAGE_CUT_IN_STEP if args.stage_approach else S4_CUT_IN_TRIGGER_STEP

    print("=" * 64)
    print("SCENARIO S4 — CUT-IN FROM ADJACENT LANE")
    print("=" * 64)
    print(f"  Fog levels:      {args.fog}")
    print(f"  Seeds:           {args.seeds}")
    print(f"  Output dir:      {args.output}")
    print(f"  NPC speed:       {S4_NPC_SPEED_KMH} km/h (adjacent lane)")
    print(f"  NPC ahead:       {npc_ahead:.0f}m")
    print(f"  Staging:         {'ON (gap={:.0f}m, min speed={:.0f}km/h)'.format(args.stage_gap, STAGE_MIN_SPEED_KMH) if args.stage_approach else 'OFF'}")
    print(f"  Cut-in stop:     {'YES (full emergency brake)' if args.cutin_stop else 'NO (hold speed)'}")
    print(f"  Cut-in trigger:  step {cutin_step} (earliest)")
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
                                      pcla_agent=args.pcla_agent,
                                      radar_backend=args.radar_backend,
                                      radar_profile=args.radar_profile,
                                      radar_config_path=args.radar_config,
                                      radar_seed=args.radar_seed,
                                      radar_ghost_detector=args.radar_ghost_detector,
                                      radar_ghost_threshold=args.radar_ghost_threshold,
                                      radar_ghost_device=args.radar_ghost_device,
                                      stage_approach=args.stage_approach,
                                      stage_gap=args.stage_gap,
                                      cutin_stop=args.cutin_stop, scenario_id=4)
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
    print("S4 SUMMARY")
    print("=" * 64)
    for r in results:
        icon = "💥" if r["collision"] else "✅"
        print(f"  {icon} fog={r['fog']:3d}  seed={r['seed']:3d}  "
              f"collision={r['collision']}  min_dist={r['min_dist']:.1f}m")
    collisions = sum(1 for r in results if r["collision"])
    print(f"\n  Total: {len(results)} runs, {collisions} collisions "
          f"({100 * collisions / max(1, len(results)):.0f}%)")
    print("=" * 64)

    # Let CARLA settle
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
