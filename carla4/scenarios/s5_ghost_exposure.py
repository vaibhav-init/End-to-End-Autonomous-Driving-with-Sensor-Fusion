#!/usr/bin/env python3
"""
Scenario S5 — Ghost-Exposure Drive
==================================

The false-positive side of the study. S1, S2 and S4 ask whether the controller
brakes when it must; this one asks whether it brakes when it must not.

Setup:
  - Town04 highway. The evaluated driver has longitudinal control from the
    first tick; there is no staged hazard and nothing is ever placed in the
    ego lane on purpose.
  - Several NPCs cruise in the neighbouring lanes at about the ego's speed.
    They are the parents that geometry multipath mirrors off the guardrails
    and barriers; background traffic supplies more reflectors elsewhere.
  - Every tick the privileged nearest in-path actor (if any) is logged, so a
    brake command can be scored as legitimate or phantom afterwards with
    `scenarios/metrics.py`. Natural traffic that does enter the ego lane is
    therefore handled correctly rather than assumed away.

Measures (via analyze_results.py / compare_drivers.py):
  - phantom brake events, phantom brakes per km
  - jerk RMS while moving
  - how often the radar's selected target was a ghost (from the diagnostics)

Run it paired: same seeds with `--radar-multipath-mode off` and `geometry`.
The difference between the two is the closed-loop cost of ghosts for that
controller. Whether an in-corridor ghost actually occurs on a given seed is
measured by the logged ghost-selection columns, not assumed.
"""

import argparse
import os
import random
import sys
import time

import carla

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ground_truth_logger import (
    GroundTruthLogger,
    compute_vehicle_speed,
    nearest_in_path_actor,
)
from drivers import DRIVER_NAMES, make_driver
from spawn_utils import get_highway_spawns
from s4_cut_in import spawn_npc_adjacent_lane
from scenario_weather import set_weather_condition
from config import (
    CARLA_HOST, CARLA_PORT, DEFAULT_TOWN, FPS,
    FOG_LADDER, RANDOM_SEEDS, SCENARIO_DURATION_S, FOG_SETTLE_STEPS,
    S5_TARGET_SPEED_KMH, S5_ADJACENT_NPCS, S5_BACKGROUND_VEHICLES,
)
from radar import add_radar_arguments, radar_kwargs_from_args


TL_CLEARANCE_M = 100.0
ADJACENT_SPAWN_OFFSETS_M = (18.0, 40.0, 65.0, 90.0)


def cleanup_actor(actor):
    if actor and actor.is_alive:
        try:
            actor.destroy()
        except RuntimeError:
            pass


def spawn_background(world, client, tm, count, rng):
    """Autopilot traffic elsewhere on the map: reflector and parent supply."""

    bp_lib = world.get_blueprint_library()
    vehicle_bps = [
        bp for bp in bp_lib.filter("vehicle.*")
        if int(bp.get_attribute("number_of_wheels")) >= 4
    ]
    spawn_points = world.get_map().get_spawn_points()
    rng.shuffle(spawn_points)
    port = tm.get_port()
    batch = []
    for index in range(min(count, len(spawn_points) - 1)):
        bp = rng.choice(vehicle_bps)
        batch.append(
            carla.command.SpawnActor(bp, spawn_points[index + 1]).then(
                carla.command.SetAutopilot(carla.command.FutureActor, True, port)
            )
        )
    ids = [r.actor_id for r in client.apply_batch_sync(batch, True) if not r.error]
    for actor_id in ids:
        actor = world.get_actor(actor_id)
        if actor:
            tm.ignore_lights_percentage(actor, 100)
            tm.ignore_signs_percentage(actor, 100)
    return ids


def run_scenario(client, world, settings, fog_density, seed, output_dir,
                 driver_name="mlp", model_dir=None, pcla_agent="tfv6_visiononly",
                 radar_kwargs=None, duration_s=None,
                 target_speed_kmh=S5_TARGET_SPEED_KMH,
                 adjacent_npcs=S5_ADJACENT_NPCS,
                 background_vehicles=S5_BACKGROUND_VEHICLES,
                 scenario_id=5, safety_rules=False):
    carla_map = world.get_map()
    rng = random.Random(seed)
    duration_s = duration_s or SCENARIO_DURATION_S[scenario_id]

    highway_spawns = get_highway_spawns(carla_map)
    traffic_lights = world.get_actors().filter("traffic.traffic_light")
    tl_locations = [tl.get_location() for tl in traffic_lights]
    if tl_locations:
        highway_spawns = [
            sp for sp in highway_spawns
            if all(sp.location.distance(loc) >= TL_CLEARANCE_M for loc in tl_locations)
        ]
    if not highway_spawns:
        raise RuntimeError("No highway spawn points found away from traffic lights")
    print(f"  Found {len(highway_spawns)} usable highway spawn points")

    set_weather_condition(world, fog_density)
    for _ in range(FOG_SETTLE_STEPS):
        world.tick()

    tm = client.get_trafficmanager(8000)
    tm.set_random_device_seed(seed)
    ego_bp = world.get_blueprint_library().find("vehicle.tesla.model3")
    ego = None
    rng.shuffle(highway_spawns)
    for sp_tf in highway_spawns[:20]:
        ego = world.try_spawn_actor(ego_bp, sp_tf)
        if ego:
            break
    if ego is None:
        raise RuntimeError("Failed to spawn ego vehicle on highway")
    for _ in range(5):
        world.tick()

    radar_kwargs = dict(radar_kwargs or {})
    radar_kwargs.setdefault("radar_seed", seed)
    driver = make_driver(
        driver_name,
        model_dir=model_dir,
        pcla_agent=pcla_agent,
        safety_rules=safety_rules,
        **radar_kwargs,
    )
    driver.setup(world, ego, carla_map, client)

    # Parents in the neighbouring lanes. spawn_npc_adjacent_lane prefers the
    # right lane and falls back to the left; failures are skipped, the count
    # actually spawned is logged so a thin run is visible.
    npcs = []
    for offset in ADJACENT_SPAWN_OFFSETS_M[:max(0, int(adjacent_npcs))]:
        npc, _direction = spawn_npc_adjacent_lane(world, carla_map, ego, offset)
        if npc is None:
            continue
        npc.set_autopilot(True, tm.get_port())
        tm.set_desired_speed(npc, target_speed_kmh + rng.uniform(-5.0, 5.0))
        tm.ignore_lights_percentage(npc, 100)
        tm.ignore_signs_percentage(npc, 100)
        tm.auto_lane_change(npc, False)
        tm.distance_to_leading_vehicle(npc, 8.0)
        npcs.append(npc)
    background_ids = spawn_background(world, client, tm, background_vehicles, rng)
    print(f"  Adjacent NPCs: {len(npcs)}/{adjacent_npcs}  background: {len(background_ids)}")

    for _ in range(40):
        control = driver.get_control(ego, world)
        ego.apply_control(control)
        world.tick()

    spectator = world.get_spectator()
    collision_bp = world.get_blueprint_library().find("sensor.other.collision")
    collision_sensor = world.spawn_actor(collision_bp, carla.Transform(), attach_to=ego)
    collision_occurred = [False]
    collision_sensor.listen(lambda _event: collision_occurred.__setitem__(0, True))

    logger = GroundTruthLogger(
        output_dir, scenario_id, fog_density, seed,
        target_speed_kmh=target_speed_kmh,
    )
    max_steps = int(duration_s * FPS)
    prev_speed_mps = 0.0
    ghost_selected_frames = 0
    print(f"  S5 | fog={fog_density} | seed={seed} | {duration_s:.0f}s exposure drive")

    try:
        for step in range(max_steps):
            control = driver.get_control(ego, world)
            ego.apply_control(control)

            _tick_start = time.perf_counter()
            world.tick()
            _elapsed = time.perf_counter() - _tick_start
            if _elapsed < 1.0 / FPS:
                time.sleep(1.0 / FPS - _elapsed)

            applied = ego.get_control() if ego.is_alive else None
            ego_speed = compute_vehicle_speed(ego)
            ego_speed_mps = ego_speed / 3.6
            accel = (ego_speed_mps - prev_speed_mps) * FPS if step > 0 else 0.0
            prev_speed_mps = ego_speed_mps

            lead, gap = nearest_in_path_actor(world, ego)
            lead_speed = compute_vehicle_speed(lead) if lead is not None else None
            rel_vel = (ego_speed - lead_speed) / 3.6 if lead is not None else None

            diagnostics = driver.diagnostics()
            if diagnostics.get("selected_source") == "ghost":
                ghost_selected_frames += 1

            logger.log(
                step=step,
                ego_speed_kmh=ego_speed,
                npc_speed_kmh=lead_speed,
                distance_to_npc=gap,
                relative_velocity=rel_vel,
                throttle=applied.throttle if applied else 0.0,
                brake=applied.brake if applied else 0.0,
                steer=applied.steer if applied else 0.0,
                critical_event=False,
                collision=collision_occurred[0],
                ego_accel=accel,
                radar_diagnostics=diagnostics,
                npc_in_path=lead is not None,
                detections=driver.latest_detections(),
            )

            if collision_occurred[0]:
                print(f"    💥 COLLISION at step {step} (speed={ego_speed:.1f} km/h)")
                break

            if step % (FPS * 5) == 0:
                gap_text = f"{gap:.1f}m" if gap is not None else "none"
                print(f"    step={step:4d}  spd={ego_speed:5.1f}km/h  in-path={gap_text}  "
                      f"brk={applied.brake if applied else 0.0:.2f}  "
                      f"ghost-selected={ghost_selected_frames}")

            if ego.is_alive:
                ego_t = ego.get_transform()
                spectator.set_transform(carla.Transform(
                    ego_t.location - ego_t.get_forward_vector() * 15 + carla.Location(z=8),
                    carla.Rotation(pitch=-20, yaw=ego_t.rotation.yaw),
                ))
    finally:
        logger.close()
        driver.cleanup()
        try:
            collision_sensor.destroy()
        except RuntimeError:
            pass
        for npc in npcs:
            try:
                npc.set_autopilot(False)
            except RuntimeError:
                pass
            cleanup_actor(npc)
        if background_ids:
            client.apply_batch([carla.command.DestroyActor(i) for i in background_ids])
        cleanup_actor(ego)

    print(f"    ghost-selected frames: {ghost_selected_frames}/{logger.row_count}")
    return logger


def main():
    parser = argparse.ArgumentParser(description="S5: Ghost-exposure drive")
    parser.add_argument("--host", default=CARLA_HOST)
    parser.add_argument("--port", type=int, default=CARLA_PORT)
    parser.add_argument("--town", default=DEFAULT_TOWN)
    parser.add_argument("--fog", type=int, nargs="+", default=FOG_LADDER)
    parser.add_argument("--seeds", type=int, nargs="+", default=RANDOM_SEEDS)
    parser.add_argument("--output", default="results_s5")
    parser.add_argument("--safety-rules", action="store_true",
                        help="re-enable the mlp driver's hardcoded brake rules (ablation)")
    parser.add_argument("--driver", choices=list(DRIVER_NAMES), default="mlp")
    parser.add_argument("--model-dir", default="../model_throttle_brake")
    parser.add_argument("--pcla-agent", default="tfv6_visiononly")
    parser.add_argument("--duration-s", type=float, default=SCENARIO_DURATION_S[5])
    parser.add_argument("--target-speed-kmh", type=float, default=S5_TARGET_SPEED_KMH,
                        help="speed the adjacent NPCs are asked to hold")
    parser.add_argument("--adjacent-npcs", type=int, default=S5_ADJACENT_NPCS)
    parser.add_argument("--background-vehicles", type=int, default=S5_BACKGROUND_VEHICLES)
    add_radar_arguments(parser)
    args = parser.parse_args()
    if args.duration_s <= 0.0:
        parser.error("--duration-s must be positive")

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
    print("SCENARIO S5 — GHOST-EXPOSURE DRIVE")
    print("=" * 64)
    print(f"  Fog levels:      {args.fog}")
    print(f"  Seeds:           {args.seeds}")
    print(f"  Duration:        {args.duration_s:.0f}s")
    print(f"  Adjacent NPCs:   {args.adjacent_npcs} at ~{args.target_speed_kmh:.0f} km/h")
    print(f"  Output dir:      {args.output}")
    print("=" * 64)

    results = []
    for fog in args.fog:
        for seed in args.seeds:
            random.seed(seed)
            try:
                logger = run_scenario(
                    client, world, settings, fog, seed, args.output,
                    driver_name=args.driver, model_dir=args.model_dir,
                    pcla_agent=args.pcla_agent,
                    radar_kwargs=radar_kwargs_from_args(args, seed=seed),
                    duration_s=args.duration_s,
                    target_speed_kmh=args.target_speed_kmh,
                    adjacent_npcs=args.adjacent_npcs,
                    background_vehicles=args.background_vehicles,
                    safety_rules=args.safety_rules,
                )
                results.append({
                    "fog": fog, "seed": seed,
                    "collision": logger.has_collision,
                    "min_dist": logger.min_distance,
                    "rows": logger.row_count,
                })
                status = "💥" if logger.has_collision else "✅"
                print(f"    {status} fog={fog} seed={seed} — "
                      f"collision={logger.has_collision} min_dist={logger.min_distance:.1f}m")
            except Exception as exc:  # noqa: BLE001
                import traceback
                traceback.print_exc()
                print(f"    ❌ fog={fog} seed={seed} failed: {exc}")

    print("\n" + "=" * 64)
    print("S5 SUMMARY")
    print("=" * 64)
    for r in results:
        icon = "💥" if r["collision"] else "✅"
        print(f"  {icon} fog={r['fog']:3d}  seed={r['seed']:3d}  "
              f"collision={r['collision']}  min_dist={r['min_dist']:.1f}m  rows={r['rows']}")
    collisions = sum(1 for r in results if r["collision"])
    print(f"\n  Total: {len(results)} runs, {collisions} collisions "
          f"({100 * collisions / max(1, len(results)):.0f}%)")
    print("  Phantom braking and jerk: python3 analyze_results.py --runs ...")
    print("=" * 64)

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
