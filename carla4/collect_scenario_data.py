#!/usr/bin/env python3
"""
Staged-scenario data collector for the radar target-speed MLP.

Mirrors the NHTSA evaluation scenarios (lead stopped / decelerating / constant /
cut-in) but drives the ego with a PRIVILEGED ACC teacher and RECORDS the same
radar + traffic-light feature schema as collect_throttle_brake_data.py:

  - Teacher (privileged): cruise at a set speed on open road; when a lead is
    within range, follow/brake with GapKeepController using GROUND-TRUTH lead
    distance/speed. This is a clean, collision-avoiding expert (like an ACC).
  - Student features (what we save): RADAR distance/rel-vel/obstacle-speed +
    radar features — exactly the columns the MLP trains on.

So the MLP imitates good staged-scenario braking/following from radar alone.

Output: one CSV per run saved into the SAME folder as the base dataset
(`dataset_throttle_brake/`). `train_throttle_brake.py` globs every *.csv in the
folder, so this trains together with the base data.

Usage:
    python3 collect_scenario_data.py --episodes 6 --scenarios stopped decelerating constant cut_in
"""

import argparse
from collections import deque
import json
import math
import os
import random
import sys
import time

import carla
import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
if os.path.join(_HERE, "scenarios") not in sys.path:
    sys.path.insert(0, os.path.join(_HERE, "scenarios"))

from speed_model import BASE_FEATURE_COLS, flatten_history
from collect_throttle_brake_data import (
    compute_future_speed_label,
    stacked_feature_names,
    set_tm_target_speed,
    spawn_npc_adjacent_lane,
)
from radar import (
    add_radar_arguments,
    create_front_radar,
    describe_radar_configuration,
    radar_diagnostics_row,
    radar_overrides_from_args,
)
from radar.detection_log import DetectionLog, sidecar_path
from staging import GapKeepController, SpeedController
from driving_contract import (
    MAX_TARGET_SPEED_KMH,
    NATIVE_RADAR_POINTS_PER_SECOND,
    RADAR_RANGE_M,
)
from spawn_utils import (
    get_highway_spawns,
    spawn_obstacle_in_ego_direction,
    spawn_npc_in_ego_direction,
)
from drivers.steering import BasicAgentSteering
from weather_utils import apply_random_fog


CARLA_HOST = "127.0.0.1"
CARLA_PORT = 2000
DEFAULT_TOWN = "Town04"
FPS = 20
RADAR_RANGE = RADAR_RANGE_M
HISTORY_FRAMES = 10
LABEL_HORIZON = 10
SAVE_DIR = "dataset_throttle_brake"

DETECT_RANGE_M = 70.0           # start following once a lead is within this range
FOLLOW_GAP_M = 12.0            # ACC target gap
WARMUP_S = 4                   # seconds to reach cruise before the event
EPISODE_S = 20                 # seconds per episode
DECEL_BRAKE_S = 7             # when the decelerating lead slams its brakes
CUT_IN_S = 6                  # when the adjacent NPC is forced to cut in

SCENARIO_CHOICES = ("stopped", "decelerating", "constant", "cut_in")


def compute_speed(actor):
    v = actor.get_velocity()
    return math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2)


def distance_between(a, b):
    return a.get_location().distance(b.get_location())


def cleanup_actor(actor):
    if actor and actor.is_alive:
        try:
            actor.destroy()
        except RuntimeError:
            pass


def build_base_features(ego_speed, accel, radar_state):
    if radar_state["relative_velocity"] > 0.1:
        ttc = min(radar_state["distance"] / radar_state["relative_velocity"], 10.0)
    else:
        ttc = 10.0
    return {
        "ego_speed": round(ego_speed, 4),
        "ego_acceleration": round(max(-20.0, min(20.0, accel)), 4),
        "distance": round(radar_state["distance"], 4),
        "relative_velocity": round(radar_state["relative_velocity"], 4),
        "ttc": round(ttc, 4),
        "obstacle_speed": round(radar_state["obstacle_speed"], 4),
    }


def spawn_scenario_npc(
    world,
    carla_map,
    tm,
    ego,
    scenario,
    rng,
    cruise_kmh,
    event_gap_m,
):
    """Spawn the lead/obstacle for a scenario. Returns (npc, cut_in_to_right)."""
    if scenario == "stopped":
        npc = spawn_obstacle_in_ego_direction(
            world, carla_map, ego, event_gap_m
        )
        return npc, None
    if scenario == "decelerating":
        npc = spawn_npc_in_ego_direction(
            world, carla_map, ego, event_gap_m
        )
        if npc:
            npc.set_autopilot(True, tm.get_port())
            tm.auto_lane_change(npc, False)
            set_tm_target_speed(npc, tm, cruise_kmh)
        return npc, None
    if scenario == "constant":
        npc = spawn_npc_in_ego_direction(
            world, carla_map, ego, event_gap_m
        )
        if npc:
            npc.set_autopilot(True, tm.get_port())
            tm.auto_lane_change(npc, False)
            set_tm_target_speed(
                npc,
                tm,
                min(cruise_kmh, rng.choice((15.0, 20.0, 25.0))),
            )
        return npc, None
    if scenario == "cut_in":
        return spawn_npc_adjacent_lane(
            world, ego, carla_map, tm, ahead_m=event_gap_m
        )
    return None, None


def run_episode(world, carla_map, tm, scenario, seed, frame_counter, args,
                detection_log=None):
    """Drive one staged episode with the ACC teacher; return (rows, frame_counter)."""
    rng = random.Random(seed)
    dt = 1.0 / FPS
    speed_choices = [
        speed
        for speed in (30.0, 40.0, 50.0, 60.0)
        if speed <= args.max_speed_kmh
    ]
    cruise_kmh = rng.choice(speed_choices) if speed_choices else args.max_speed_kmh
    cruise_mps = cruise_kmh / 3.6
    gap_choices = {
        "stopped": (25.0, 35.0, 50.0),
        "decelerating": (15.0, 20.0, 25.0, 30.0, 40.0),
        "constant": (20.0, 30.0, 40.0),
        "cut_in": (15.0, 20.0, 25.0),
    }
    event_gap_m = rng.choice(gap_choices[scenario])
    decel_brake = rng.choice((0.5, 0.75, 1.0))
    decel_duration_s = rng.choice((1.5, 2.5, 3.5))
    weather_name = apply_random_fog(world, rng=rng)
    event_details = (
        f", brake={decel_brake:.2f} for {decel_duration_s:.1f}s"
        if scenario == "decelerating"
        else ""
    )
    print(
        f"    weather={weather_name}, cruise={cruise_kmh:.0f}km/h, "
        f"gap={event_gap_m:.0f}m{event_details}"
    )

    highway_spawns = get_highway_spawns(carla_map)
    rng.shuffle(highway_spawns)
    ego_bp = world.get_blueprint_library().find("vehicle.tesla.model3")
    ego = None
    for sp in highway_spawns[:20]:
        ego = world.try_spawn_actor(ego_bp, sp)
        if ego:
            break
    if ego is None:
        return [], frame_counter

    for _ in range(5):
        world.tick()

    radar = create_front_radar(
        ego,
        world,
        RADAR_RANGE,
        backend=args.radar_backend,
        fps=FPS,
        points_per_second=args.radar_points_per_second,
        profile_name=args.radar_profile,
        config_path=args.radar_config,
        seed=args.radar_seed if args.radar_seed is not None else seed,
        ghost_detector_path=args.radar_ghost_detector,
        ghost_threshold=args.radar_ghost_threshold,
        ghost_device=args.radar_ghost_device,
        ghost_oracle=args.radar_ghost_oracle,
        overrides=args.radar_overrides,
    )
    steering = BasicAgentSteering(ego, carla_map)
    cruise = SpeedController(cruise_mps, dt)
    gapkeep = GapKeepController(
        FOLLOW_GAP_M,
        dt,
        max_speed_mps=args.max_speed_kmh / 3.6,
    )

    collision = [False]
    coll_bp = world.get_blueprint_library().find("sensor.other.collision")
    coll_sensor = world.spawn_actor(coll_bp, carla.Transform(), attach_to=ego)
    coll_sensor.listen(lambda e: collision.__setitem__(0, True))

    history = deque(maxlen=args.history)
    rows = []
    prev_speed = 0.0

    # Warm up to cruise on open road
    for _ in range(WARMUP_S * FPS):
        thr, brk = cruise.run_step(compute_speed(ego))
        ego.apply_control(carla.VehicleControl(throttle=thr, steer=steering.get_steer(), brake=brk))
        world.tick()

    npc, cut_in_to_right = spawn_scenario_npc(
        world,
        carla_map,
        tm,
        ego,
        scenario,
        rng,
        cruise_kmh,
        event_gap_m,
    )
    if npc is None:
        cleanup_actor(coll_sensor)
        radar.cleanup()
        cleanup_actor(ego)
        return [], frame_counter

    brake_step = DECEL_BRAKE_S * FPS
    cut_step = CUT_IN_S * FPS
    lead_braked = False
    lead_recovered = False
    cut_in_triggered = False

    for step in range(EPISODE_S * FPS):
        ego_speed = compute_speed(ego)

        # Privileged lead state (ground truth). For cut-in, ignore the NPC until
        # it has actually been forced into the ego's lane.
        is_lead = npc is not None and npc.is_alive
        if scenario == "cut_in" and not cut_in_triggered:
            is_lead = False
        gap = distance_between(ego, npc) if (npc and npc.is_alive) else 999.0
        lead_v = compute_speed(npc) if (npc and npc.is_alive) else 0.0

        if is_lead and gap < DETECT_RANGE_M:
            thr, brk = gapkeep.run_step(gap, ego_speed, lead_v)
        else:
            thr, brk = cruise.run_step(ego_speed)
        ego.apply_control(carla.VehicleControl(throttle=thr, steer=steering.get_steer(), brake=brk))

        # Inject the scenario event
        if scenario == "decelerating" and step >= brake_step and not lead_braked and npc and npc.is_alive:
            npc.set_autopilot(False)
            npc.apply_control(
                carla.VehicleControl(throttle=0.0, brake=decel_brake)
            )
            lead_braked = True
        if (
            scenario == "decelerating"
            and lead_braked
            and not lead_recovered
            and step >= brake_step + int(decel_duration_s * FPS)
            and npc
            and npc.is_alive
        ):
            npc.set_autopilot(True, tm.get_port())
            set_tm_target_speed(npc, tm, cruise_kmh)
            lead_recovered = True
        if scenario == "cut_in" and step >= cut_step and not cut_in_triggered and npc and npc.is_alive:
            try:
                tm.force_lane_change(npc, bool(cut_in_to_right))
            except RuntimeError:
                pass
            cut_in_triggered = True

        world.tick()

        # Record AFTER the tick (radar = student sensor, not the GT used by teacher)
        ego_speed = compute_speed(ego)
        accel = (ego_speed - prev_speed) * FPS if step > 0 else 0.0
        prev_speed = ego_speed
        radar.update_ego_speed(ego_speed)
        radar_state = radar.get()
        radar_diagnostics = radar_diagnostics_row(radar)
        if detection_log is not None:
            detection_log.append_radar(radar, frame_counter)
        history.append(build_base_features(ego_speed, accel, radar_state))
        if len(history) == args.history:
            row = flatten_history(history, BASE_FEATURE_COLS)
            row.update({
                "frame": frame_counter,
                "timestamp": round(frame_counter / FPS, 3),
                "scenario": scenario,
                "weather": weather_name,
                "episode_id": f"staged_{scenario}_{seed:06d}",
                "event_gap_m": event_gap_m,
                "event_brake": decel_brake if scenario == "decelerating" else 0.0,
                "cruise_speed_kmh": cruise_kmh,
                "ego_speed_now": round(ego_speed, 4),
                "autopilot_throttle": round(thr, 4),
                "autopilot_brake": round(brk, 4),
            })
            row.update(radar_diagnostics)
            rows.append(row)
        frame_counter += 1

        if collision[0]:
            break
        # Early stop once stopped close behind a halted lead (S1-style)
        if scenario == "stopped" and ego_speed < 0.3 and gap < 8.0:
            break

    cleanup_actor(coll_sensor)
    cleanup_actor(npc)
    radar.cleanup()
    cleanup_actor(ego)
    return rows, frame_counter


def main():
    parser = argparse.ArgumentParser(description="Collect staged-scenario imitation data")
    parser.add_argument("--host", default=CARLA_HOST)
    parser.add_argument("--port", type=int, default=CARLA_PORT)
    parser.add_argument("--town", default=DEFAULT_TOWN)
    parser.add_argument("--episodes", type=int, default=6, help="episodes per scenario")
    parser.add_argument("--scenarios", nargs="+", default=list(SCENARIO_CHOICES),
                        choices=SCENARIO_CHOICES)
    parser.add_argument("--history", type=int, default=HISTORY_FRAMES)
    parser.add_argument("--label-horizon", type=int, default=LABEL_HORIZON)
    parser.add_argument(
        "--max-speed-kmh",
        type=float,
        default=MAX_TARGET_SPEED_KMH,
        help="Hard ceiling for teacher cruise speed and target labels",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default=SAVE_DIR, help="dataset folder (same as base data)")
    parser.add_argument("--out-name", default="data_staged.csv")
    add_radar_arguments(parser)
    args = parser.parse_args()
    if not 0.0 < args.max_speed_kmh <= MAX_TARGET_SPEED_KMH:
        parser.error(
            f"--max-speed-kmh must be in (0, {MAX_TARGET_SPEED_KMH:g}]"
        )
    random.seed(args.seed)
    np.random.seed(args.seed)

    args.radar_points_per_second = (
        NATIVE_RADAR_POINTS_PER_SECOND
        if args.radar_backend == "native"
        else 240000
    )
    args.radar_overrides = radar_overrides_from_args(args)
    radar_metadata = describe_radar_configuration(
        backend=args.radar_backend,
        range_m=RADAR_RANGE,
        fps=FPS,
        points_per_second=args.radar_points_per_second,
        profile_name=args.radar_profile,
        config_path=args.radar_config,
        ghost_detector_path=args.radar_ghost_detector,
        ghost_threshold=args.radar_ghost_threshold,
        overrides=args.radar_overrides,
        ghost_oracle=args.radar_ghost_oracle,
    )
    os.makedirs(args.output, exist_ok=True)
    csv_path = os.path.join(args.output, args.out_name)
    detections_path = sidecar_path(csv_path)
    detection_log = DetectionLog()
    config_path = os.path.join(args.output, "dataset_config.json")
    existing_config = {}
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as fh:
            existing_config = json.load(fh)
        existing_backend = existing_config.get("radar_backend", "native")
        if existing_backend != args.radar_backend:
            raise RuntimeError(
                f"Dataset uses radar backend '{existing_backend}', but this run "
                f"requested '{args.radar_backend}'. Use a separate --output "
                "directory; do not mix sensor distributions."
            )
        existing_range = float(existing_config.get("radar_range_m", RADAR_RANGE))
        if abs(existing_range - RADAR_RANGE) > 1e-6:
            raise RuntimeError(
                f"Dataset uses radar range {existing_range}m, but this collector "
                f"uses {RADAR_RANGE}m. Use a separate --output directory."
            )
        requested_points = args.radar_points_per_second
        existing_points = int(
            existing_config.get("radar_points_per_second", requested_points)
        )
        if existing_points != requested_points:
            raise RuntimeError(
                f"Dataset uses {existing_points} radar points/s, but this run "
                f"uses {requested_points}."
            )
        if args.radar_backend == "realistic":
            existing_signature = existing_config.get("radar_config_signature")
            requested_signature = radar_metadata["radar_config_signature"]
            if existing_signature != requested_signature:
                raise RuntimeError(
                    "Dataset uses realistic radar config "
                    f"{existing_signature!r}, but this run requested "
                    f"{requested_signature!r}. Use a separate --output directory."
                )
            for key in (
                "radar_ghost_detector_signature",
                "radar_ghost_threshold",
                "radar_ghost_injection",
                "radar_ghost_oracle",
            ):
                existing_value = existing_config.get(key)
                requested_value = radar_metadata.get(key)
                if existing_value != requested_value:
                    raise RuntimeError(
                        f"Dataset uses {key}={existing_value!r}, but this run "
                        f"requested {requested_value!r}. Use a separate "
                        "--output directory."
                    )
        existing_max_speed = float(
            existing_config.get("max_target_speed_kmh", args.max_speed_kmh)
        )
        if abs(existing_max_speed - args.max_speed_kmh) > 1e-6:
            raise RuntimeError(
                f"Dataset uses max speed {existing_max_speed} km/h, but this run "
                f"requested {args.max_speed_kmh} km/h."
            )
        if existing_config.get("town", args.town) != args.town:
            raise RuntimeError(
                f"Dataset uses town {existing_config.get('town')}, but this run "
                f"requested {args.town}."
            )
        if int(existing_config.get("history_frames", args.history)) != args.history:
            raise RuntimeError(
                "Dataset history length differs from --history; use a separate "
                "--output directory."
            )
        if (
            int(existing_config.get("label_horizon", args.label_horizon))
            != args.label_horizon
        ):
            raise RuntimeError(
                "Dataset label horizon differs from --label-horizon; use a "
                "separate --output directory."
            )

    print("=" * 72)
    print("STAGED-SCENARIO DATA COLLECTOR")
    print("=" * 72)
    print(f"  Town:        {args.town}")
    print(f"  Radar range: {RADAR_RANGE:.0f}m")
    print(f"  Radar:       {args.radar_backend}")
    if args.radar_backend == "realistic":
        print(f"  Profile:     {radar_metadata['radar_profile']}")
        print(f"  Config ID:   {radar_metadata['radar_config_signature']}")
    print(f"  Speed cap:   {args.max_speed_kmh:.1f} km/h")
    print(f"  Seed:        {args.seed}")
    print(f"  Scenarios:   {args.scenarios}")
    print(f"  Episodes:    {args.episodes} each")
    print(f"  Output:      {csv_path}")
    print("=" * 72)

    client = carla.Client(args.host, args.port)
    client.set_timeout(30.0)
    world = client.get_world()
    if world.get_map().name.split("/")[-1] != args.town:
        print(f"  Loading {args.town} ...")
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

    carla_map = world.get_map()
    episode_frames = []  # per-episode DataFrames (label computed per episode)
    frame_counter = 0
    counts = {s: 0 for s in args.scenarios}

    try:
        for scenario in args.scenarios:
            for ep in range(args.episodes):
                seed = (
                    args.seed
                    + 1000 * SCENARIO_CHOICES.index(scenario)
                    + ep
                )
                print(f"\n  [{scenario}] episode {ep + 1}/{args.episodes}")
                rows, frame_counter = run_episode(
                    world, carla_map, tm, scenario, seed, frame_counter, args,
                    detection_log=detection_log)
                if rows:
                    ep_df = pd.DataFrame(rows)
                    # Label per episode so future-speed never leaks across episodes
                    ep_df["teacher_target_speed"] = compute_future_speed_label(
                        ep_df, args.label_horizon
                    ).clip(lower=0.0, upper=args.max_speed_kmh / 3.6)
                    ep_df = ep_df.dropna(subset=["teacher_target_speed"]).reset_index(drop=True)
                    episode_frames.append(ep_df)
                    counts[scenario] += len(ep_df)
                    print(f"    +{len(ep_df):,} samples")
    except KeyboardInterrupt:
        print("\n  Interrupted")
    finally:
        if episode_frames:
            df = pd.concat(episode_frames, ignore_index=True)
            df.to_csv(csv_path, index=False)
            if detection_log.frame_count:
                detection_log.save(detections_path)
                print(f"  Detections: {detections_path} ({detection_log.point_count:,} points)")
            dataset_config = existing_config
            dataset_config.update({
                "town": dataset_config.get("town", args.town),
                "fps": FPS,
                "max_target_speed_kmh": args.max_speed_kmh,
                "history_frames": args.history,
                "label_horizon": args.label_horizon,
                "base_feature_cols": BASE_FEATURE_COLS,
                "stacked_feature_cols": stacked_feature_names(
                    BASE_FEATURE_COLS, args.history
                ),
                "label_col": "teacher_target_speed",
                "episode_col": "episode_id",
            })
            dataset_config.update(radar_metadata)
            with open(config_path, "w", encoding="utf-8") as fh:
                json.dump(dataset_config, fh, indent=2)
            print("\n" + "=" * 72)
            print("STAGED COLLECTION COMPLETE")
            print("=" * 72)
            print(f"  Samples saved:     {len(df):,}")
            print(f"  Per-scenario rows: {counts}")
            print(f"  Dataset:           {csv_path}")
            print(f"  Config:            {config_path}")
            print(f"  (train picks this up alongside data.csv via folder glob)")
            print("=" * 72)
        else:
            print("  No samples collected.")
        try:
            world.apply_settings(original_settings)
            tm.set_synchronous_mode(False)
        except RuntimeError:
            pass


if __name__ == "__main__":
    main()
