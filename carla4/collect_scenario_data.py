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
    YOLO traffic-light features — exactly the columns the MLP trains on.

So the MLP imitates good staged-scenario braking/following from radar alone.

Output: one CSV per run saved into the SAME folder as the base dataset
(`dataset_throttle_brake/`). `train_throttle_brake.py` globs every *.csv in the
folder, so this trains together with the base data.

Usage:
    python3 collect_scenario_data.py --episodes 6 --scenarios stopped decelerating constant cut_in
"""

import argparse
from collections import deque
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

from yolo_perception import (
    CameraManager,
    YOLOPerception,
    YOLO_AVAILABLE,
    empty_visual_features,
)
from speed_model import BASE_FEATURE_COLS, flatten_history
from collect_throttle_brake_data import (
    FrontRadar,
    compute_future_speed_label,
    stacked_feature_names,
    set_tm_target_speed,
    spawn_npc_adjacent_lane,
)
from staging import GapKeepController, SpeedController
from spawn_utils import (
    get_highway_spawns,
    spawn_obstacle_in_ego_direction,
    spawn_npc_in_ego_direction,
)
from drivers.steering import BasicAgentSteering


CARLA_HOST = "127.0.0.1"
CARLA_PORT = 2000
DEFAULT_TOWN = "Town04"
FPS = 20
RADAR_RANGE = 100.0
HISTORY_FRAMES = 10
LABEL_HORIZON = 10
SAVE_DIR = "dataset_throttle_brake"

CRUISE_KMH = 60.0
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


def build_base_features(ego_speed, accel, radar_state, visual):
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
        "traffic_light_state": float(visual["traffic_light_state"]),
        "tl_confidence": round(visual["tl_confidence"], 4),
        "tl_bbox_area": round(visual["tl_bbox_area"], 6),
        "tl_center_x": round(visual["tl_center_x"], 4),
    }


def spawn_scenario_npc(world, carla_map, tm, ego, scenario):
    """Spawn the lead/obstacle for a scenario. Returns (npc, cut_in_to_right)."""
    if scenario == "stopped":
        npc = spawn_obstacle_in_ego_direction(world, carla_map, ego, 50.0)
        return npc, None
    if scenario == "decelerating":
        npc = spawn_npc_in_ego_direction(world, carla_map, ego, 25.0)
        if npc:
            npc.set_autopilot(True, tm.get_port())
            tm.auto_lane_change(npc, False)
            set_tm_target_speed(npc, tm, CRUISE_KMH)
        return npc, None
    if scenario == "constant":
        npc = spawn_npc_in_ego_direction(world, carla_map, ego, 35.0)
        if npc:
            npc.set_autopilot(True, tm.get_port())
            tm.auto_lane_change(npc, False)
            set_tm_target_speed(npc, tm, 20.0)
        return npc, None
    if scenario == "cut_in":
        return spawn_npc_adjacent_lane(world, ego, carla_map, tm, ahead_m=20.0)
    return None, None


def run_episode(world, carla_map, tm, scenario, seed, frame_counter, args):
    """Drive one staged episode with the ACC teacher; return (rows, frame_counter)."""
    rng = random.Random(seed)
    dt = 1.0 / FPS
    cruise_mps = CRUISE_KMH / 3.6

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

    radar = FrontRadar(ego, world, RADAR_RANGE)
    camera = CameraManager(ego, world)
    yolo = YOLOPerception() if YOLO_AVAILABLE else None
    steering = BasicAgentSteering(ego, carla_map)
    cruise = SpeedController(cruise_mps, dt)
    gapkeep = GapKeepController(FOLLOW_GAP_M, dt)

    collision = [False]
    coll_bp = world.get_blueprint_library().find("sensor.other.collision")
    coll_sensor = world.spawn_actor(coll_bp, carla.Transform(), attach_to=ego)
    coll_sensor.listen(lambda e: collision.__setitem__(0, True))

    history = deque(maxlen=HISTORY_FRAMES)
    rows = []
    prev_speed = 0.0

    # Warm up to cruise on open road
    for _ in range(WARMUP_S * FPS):
        thr, brk = cruise.run_step(compute_speed(ego))
        ego.apply_control(carla.VehicleControl(throttle=thr, steer=steering.get_steer(), brake=brk))
        world.tick()

    npc, cut_in_to_right = spawn_scenario_npc(world, carla_map, tm, ego, scenario)
    if npc is None:
        cleanup_actor(coll_sensor)
        radar.cleanup()
        camera.cleanup()
        cleanup_actor(ego)
        return [], frame_counter

    brake_step = DECEL_BRAKE_S * FPS
    cut_step = CUT_IN_S * FPS
    lead_braked = False
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
            npc.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))
            lead_braked = True
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
        visual = empty_visual_features()
        cam_frame = camera.get_frame()
        if yolo is not None and cam_frame is not None:
            visual = yolo.extract_scene_features(cam_frame)["visual"]

        history.append(build_base_features(ego_speed, accel, radar_state, visual))
        if len(history) == HISTORY_FRAMES:
            row = flatten_history(history, BASE_FEATURE_COLS)
            row.update({
                "frame": frame_counter,
                "timestamp": round(frame_counter / FPS, 3),
                "scenario": scenario,
                "ego_speed_now": round(ego_speed, 4),
                "autopilot_throttle": round(thr, 4),
                "autopilot_brake": round(brk, 4),
            })
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
    camera.cleanup()
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
    parser.add_argument("--output", default=SAVE_DIR, help="dataset folder (same as base data)")
    parser.add_argument("--out-name", default="data_staged.csv")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    csv_path = os.path.join(args.output, args.out_name)

    print("=" * 72)
    print("STAGED-SCENARIO DATA COLLECTOR")
    print("=" * 72)
    print(f"  Town:        {args.town}")
    print(f"  Radar range: {RADAR_RANGE:.0f}m")
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
                seed = 1000 * args.scenarios.index(scenario) + ep
                print(f"\n  [{scenario}] episode {ep + 1}/{args.episodes}")
                rows, frame_counter = run_episode(
                    world, carla_map, tm, scenario, seed, frame_counter, args)
                if rows:
                    ep_df = pd.DataFrame(rows)
                    # Label per episode so future-speed never leaks across episodes
                    ep_df["teacher_target_speed"] = compute_future_speed_label(
                        ep_df, args.label_horizon).clip(lower=0.0)
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
            print("\n" + "=" * 72)
            print("STAGED COLLECTION COMPLETE")
            print("=" * 72)
            print(f"  Samples saved:     {len(df):,}")
            print(f"  Per-scenario rows: {counts}")
            print(f"  Dataset:           {csv_path}")
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
