#!/usr/bin/env python3
"""
Vision-Only Data Collector for Target-Speed Imitation
=====================================================

Mirrors collect_throttle_brake_data.py but replaces the front radar sensor
with YOLO-based obstacle detection + monocular distance estimation.

The 10 base feature columns are IDENTICAL to the radar version so the same
train_throttle_brake.py script can be used without modification:

  Kept as-is (vehicle physics / camera):
    ego_speed, ego_acceleration,
    traffic_light_state, tl_confidence, tl_bbox_area, tl_center_x

  Replaced (radar → YOLO vision):
    distance           ← pinhole-camera distance from YOLO bbox height
    relative_velocity  ← frame-to-frame distance change (EMA smoothed)
    ttc                ← distance / relative_velocity
    obstacle_speed     ← ego_speed − relative_velocity

Usage:
    python collect_vision_only_data.py --duration 900
    python train_throttle_brake.py --data dataset_vision_only/data.csv \\
                                   --config dataset_vision_only/dataset_config.json
"""

import argparse
from collections import deque
import json
import math
import os
import random
import threading
import time

import carla
import numpy as np
import pandas as pd

from yolo_perception import (
    CameraManager,
    TL_STATE_NAMES,
    VisionDistanceTracker,
    YOLO_AVAILABLE,
    YOLOPerception,
    empty_visual_features,
    empty_obstacle_features,
)
from speed_model import BASE_FEATURE_COLS, flatten_history
from weather_utils import apply_random_fog

try:
    import cv2

    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("WARNING: OpenCV not available; camera preview disabled")


# ============================================================================
# Configuration
# ============================================================================
CARLA_HOST = "127.0.0.1"
CARLA_PORT = 2000
DEFAULT_TOWN = "Town01"
FPS = 20
MAX_VISION_RANGE = 50.0

NPC_VEHICLES = 45
NPC_PEDESTRIANS = 25
HISTORY_FRAMES = 10
LABEL_HORIZON = 10
SAVE_DIR = "dataset_vision_only"

SCENARIOS = ("traffic_light", "car_following", "emergency")
STALL_SPEED_MPS = 0.3
STALL_FRAMES = FPS * 8
STALL_MIN_THROTTLE = 0.2
STALL_MAX_BRAKE = 0.1
STALL_MIN_CLEAR_DISTANCE_M = 12.0
RESPAWN_Z_OFFSET = 0.5

# EMA smoothing factor for vision-derived distance (lower = smoother)
DISTANCE_EMA_ALPHA = 0.4


# ============================================================================
# Helpers (reused from collect_throttle_brake_data.py)
# ============================================================================
def stacked_feature_names(base_cols, history_frames):
    cols = []
    for lag in range(history_frames):
        for name in base_cols:
            cols.append(f"{name}_t-{lag}")
    return cols


def spawn_vehicles(world, client, tm, count):
    bp_lib = world.get_blueprint_library()
    vehicle_bps = [
        bp
        for bp in bp_lib.filter("vehicle.*")
        if int(bp.get_attribute("number_of_wheels")) >= 4
    ]
    spawn_points = world.get_map().get_spawn_points()
    random.shuffle(spawn_points)

    port = tm.get_port()
    batch = []
    for index in range(min(count, len(spawn_points) - 1)):
        bp = random.choice(vehicle_bps)
        if bp.has_attribute("color"):
            bp.set_attribute(
                "color", random.choice(bp.get_attribute("color").recommended_values)
            )
        batch.append(
            carla.command.SpawnActor(bp, spawn_points[index + 1]).then(
                carla.command.SetAutopilot(carla.command.FutureActor, True, port)
            )
        )

    ids = [r.actor_id for r in client.apply_batch_sync(batch, True) if not r.error]
    for vehicle_id in ids:
        actor = world.get_actor(vehicle_id)
        if actor:
            tm.vehicle_percentage_speed_difference(actor, random.randint(-10, 20))
            tm.distance_to_leading_vehicle(actor, random.uniform(3.0, 7.0))
            tm.ignore_lights_percentage(actor, 0)
            tm.ignore_signs_percentage(actor, 0)
            tm.auto_lane_change(actor, True)

    print(f"  Spawned {len(ids)}/{count} NPC vehicles")
    return ids


def spawn_pedestrians(world, count):
    bp_lib = world.get_blueprint_library()
    walker_bps = bp_lib.filter("walker.pedestrian.*")
    ctrl_bp = bp_lib.find("controller.ai.walker")

    walkers = []
    for _ in range(count):
        bp = random.choice(walker_bps)
        if bp.has_attribute("is_invincible"):
            bp.set_attribute("is_invincible", "false")
        loc = world.get_random_location_from_navigation()
        if loc:
            walker = world.try_spawn_actor(bp, carla.Transform(loc))
            if walker:
                walkers.append(walker)

    controllers = []
    for walker in walkers:
        controller = world.spawn_actor(ctrl_bp, carla.Transform(), attach_to=walker)
        controllers.append(controller)

    world.tick()
    for controller in controllers:
        dest = world.get_random_location_from_navigation()
        if dest:
            controller.start()
            controller.go_to_location(dest)
            controller.set_max_speed(1.0 + random.random() * 2.0)

    print(f"  Spawned {len(walkers)}/{count} pedestrians")
    return [w.id for w in walkers], [c.id for c in controllers]


def configure_ego_autopilot(ego, tm, port):
    ego.set_autopilot(True, port)
    tm.vehicle_percentage_speed_difference(ego, -5)
    tm.distance_to_leading_vehicle(ego, 10.0)
    tm.ignore_lights_percentage(ego, 0)
    tm.ignore_signs_percentage(ego, 0)
    tm.ignore_walkers_percentage(ego, 0)
    tm.auto_lane_change(ego, True)


def set_tm_target_speed(actor, tm, target_kmh):
    speed_limit = max(20.0, actor.get_speed_limit())
    pct_diff = 100.0 * (1.0 - target_kmh / speed_limit)
    pct_diff = max(-50.0, min(90.0, pct_diff))
    tm.vehicle_percentage_speed_difference(actor, pct_diff)


def waypoint_ahead(carla_map, location, distance_m):
    waypoint = carla_map.get_waypoint(location, project_to_road=True)
    if waypoint is None:
        return None
    traveled = 0.0
    step = 3.0
    while traveled < distance_m:
        next_wps = waypoint.next(step)
        if not next_wps:
            return None
        waypoint = next_wps[0]
        traveled += step
    return waypoint


def spawn_lead_vehicle(world, ego, carla_map, tm, ahead_options=(22.0, 28.0, 34.0)):
    bp_lib = world.get_blueprint_library()
    vehicle_bps = [
        bp
        for bp in bp_lib.filter("vehicle.*")
        if int(bp.get_attribute("number_of_wheels")) >= 4
    ]
    port = tm.get_port()
    for distance in ahead_options:
        waypoint = waypoint_ahead(carla_map, ego.get_location(), distance)
        if waypoint is None:
            continue
        bp = random.choice(vehicle_bps)
        transform = waypoint.transform
        transform.location.z += 0.5
        lead = world.try_spawn_actor(bp, transform)
        if lead:
            lead.set_autopilot(True, port)
            tm.auto_lane_change(lead, False)
            tm.distance_to_leading_vehicle(lead, 2.5)
            set_tm_target_speed(lead, tm, random.uniform(10.0, 40.0))
            return lead
    return None


def cleanup_actor(actor):
    if actor and actor.is_alive:
        try:
            actor.destroy()
        except RuntimeError:
            pass


def compute_future_speed_label(df, horizon):
    future_speeds = [df["ego_speed_now"].shift(-step) for step in range(1, horizon + 1)]
    return pd.concat(future_speeds, axis=1).mean(axis=1)


def safe_respawn_ego(world, ego, spawn_transform, tm, port):
    respawn_transform = carla.Transform(
        carla.Location(
            x=spawn_transform.location.x,
            y=spawn_transform.location.y,
            z=spawn_transform.location.z + RESPAWN_Z_OFFSET,
        ),
        spawn_transform.rotation,
    )

    ego.set_autopilot(False)
    ego.set_target_velocity(carla.Vector3D(0.0, 0.0, 0.0))
    ego.set_target_angular_velocity(carla.Vector3D(0.0, 0.0, 0.0))
    ego.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0, hand_brake=True))
    ego.set_simulate_physics(False)
    ego.set_transform(respawn_transform)
    world.tick()
    ego.set_simulate_physics(True)
    ego.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))
    world.tick()
    configure_ego_autopilot(ego, tm, port)
    world.tick()


# ============================================================================
# Camera overlay
# ============================================================================
def draw_camera_overlay(frame, visual, obstacle, speed, scenario, vision_state, ttc, weather_name):
    if not CV2_AVAILABLE or frame is None:
        return

    display = frame.copy()
    bbox = visual["tl_bbox"]
    tl_state = visual["traffic_light_state"]
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        color_map = {
            0: (180, 180, 180),
            1: (0, 255, 0),
            2: (0, 255, 255),
            3: (0, 0, 255),
        }
        box_color = color_map.get(tl_state, (180, 180, 180))
        cv2.rectangle(display, (x1, y1), (x2, y2), box_color, 2)
        label = (
            f"{TL_STATE_NAMES.get(tl_state, 'none')} "
            f"conf={visual['tl_confidence']:.2f} "
            f"area={visual['tl_bbox_area']:.4f}"
        )
        cv2.putText(
            display,
            label,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            box_color,
            1,
        )

    for entry in obstacle.get("obstacle_boxes", []):
        x1, y1, x2, y2 = entry["bbox"]
        box_color = (0, 140, 255) if entry["is_primary"] else (255, 180, 0)
        thickness = 2 if entry["is_primary"] else 1
        cv2.rectangle(display, (x1, y1), (x2, y2), box_color, thickness)
        obj_label = (
            f"{entry['label']} "
            f"{entry['distance']:.1f}m "
            f"{entry['confidence']:.2f}"
        )
        cv2.putText(
            display,
            obj_label,
            (x1, min(display.shape[0] - 10, y2 + 16)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            box_color,
            1,
        )

    hud_lines = [
        f"VISION-ONLY | Scenario: {scenario}",
        f"Weather: {weather_name}",
        f"Speed: {speed * 3.6:.1f} km/h",
        f"TL: {TL_STATE_NAMES.get(tl_state, 'none')}",
        f"Vis Dist: {vision_state['distance']:.1f}m  TTC: {ttc:.1f}s",
        f"Rel Vel: {vision_state['relative_velocity']:.1f}m/s",
    ]
    for line_index, line in enumerate(hud_lines):
        cv2.putText(
            display,
            line,
            (10, 25 + line_index * 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 200, 255),
            2,
        )

    cv2.imshow("CARLA Vision-Only Collector", display)
    cv2.waitKey(1)


# ============================================================================
# Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="Collect vision-only target-speed data")
    parser.add_argument("--host", default=CARLA_HOST)
    parser.add_argument("--port", type=int, default=CARLA_PORT)
    parser.add_argument("--town", default=DEFAULT_TOWN)
    parser.add_argument("--duration", type=int, default=900)
    parser.add_argument("--vehicles", type=int, default=NPC_VEHICLES)
    parser.add_argument("--pedestrians", type=int, default=NPC_PEDESTRIANS)
    parser.add_argument("--history", type=int, default=HISTORY_FRAMES)
    parser.add_argument("--label-horizon", type=int, default=LABEL_HORIZON)
    parser.add_argument("--output", default=SAVE_DIR)
    args = parser.parse_args()

    total_frames = args.duration * FPS
    os.makedirs(args.output, exist_ok=True)
    csv_path = os.path.join(args.output, "data.csv")
    config_path = os.path.join(args.output, "dataset_config.json")

    print("=" * 72)
    print("VISION-ONLY TARGET-SPEED DATA COLLECTOR")
    print("=" * 72)
    print(f"  Town:            {args.town}")
    print(f"  Duration:        {args.duration}s")
    print(f"  History frames:  {args.history}")
    print(f"  Label horizon:   {args.label_horizon}")
    print(f"  Vehicles:        {args.vehicles}")
    print(f"  Pedestrians:     {args.pedestrians}")
    print(f"  Output:          {csv_path}")
    print(f"  Scenarios:       {', '.join(SCENARIOS)}")
    print(f"  Distance source: YOLO bbox (NO RADAR)")
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
    spawn_points = carla_map.get_spawn_points()
    safe_spawns = []
    for sp in spawn_points:
        waypoint = carla_map.get_waypoint(sp.location, project_to_road=True)
        if waypoint and not waypoint.is_junction:
            safe_spawns.append(sp)
    if not safe_spawns:
        safe_spawns = spawn_points
    random.shuffle(safe_spawns)

    ego_bp = world.get_blueprint_library().find("vehicle.tesla.model3")
    ego = None
    for sp in safe_spawns:
        ego = world.try_spawn_actor(ego_bp, sp)
        if ego:
            break
    if ego is None:
        raise RuntimeError("Failed to spawn ego vehicle")

    port = tm.get_port()
    configure_ego_autopilot(ego, tm, port)

    npc_ids = spawn_vehicles(world, client, tm, args.vehicles)
    walker_ids, ctrl_ids = spawn_pedestrians(world, args.pedestrians)

    # Camera + YOLO — the ONLY perception sensors (no radar!)
    camera = CameraManager(ego, world)
    yolo = YOLOPerception() if YOLO_AVAILABLE else None
    if yolo is None:
        raise RuntimeError(
            "Vision-only collection requires YOLO. Install ultralytics on the remote machine first."
        )
    vision_tracker = VisionDistanceTracker(alpha=DISTANCE_EMA_ALPHA, max_range=MAX_VISION_RANGE, fps=FPS)
    current_weather_name = apply_random_fog(world)
    print(f"  Weather:         {current_weather_name}")

    for _ in range(40):
        world.tick()

    feature_history = deque(maxlen=args.history)
    samples = []
    scenario_counts = {name: 0 for name in SCENARIOS}

    prev_speed = 0.0
    active_scenario = None
    lead_actor = None
    lead_last_change = 0.0
    emergency_actor = None
    emergency_stopped_frames = 0
    last_emergency_time = time.time()
    emergency_count = 0
    respawn_count = 0
    stuck_frames = 0

    spawn_requested = threading.Event()

    def key_listener():
        while True:
            try:
                input()
                spawn_requested.set()
            except EOFError:
                break

    listener_thread = threading.Thread(target=key_listener, daemon=True)
    listener_thread.start()
    print("  Press ENTER during the emergency phase to force an obstacle spawn")

    try:
        for frame in range(total_frames):
            world.tick()

            velocity = ego.get_velocity()
            speed = math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)
            accel = (speed - prev_speed) * FPS if frame > 0 else 0.0
            prev_speed = speed
            control = ego.get_control()

            # --- Scenario management (identical to radar version) ---
            progress = frame / max(1, total_frames)
            scenario_index = min(len(SCENARIOS) - 1, int(progress * len(SCENARIOS)))
            scenario = SCENARIOS[scenario_index]
            if scenario != active_scenario:
                active_scenario = scenario
                current_weather_name = apply_random_fog(world)
                print(f"\n  Switching scenario -> {scenario}")
                print(f"  Fog preset -> {current_weather_name}")
                if scenario == "car_following":
                    tm.auto_lane_change(ego, False)
                else:
                    tm.auto_lane_change(ego, True)
                    cleanup_actor(lead_actor)
                    lead_actor = None
                if scenario != "emergency":
                    cleanup_actor(emergency_actor)
                    emergency_actor = None
                    emergency_stopped_frames = 0

            if scenario == "car_following":
                scenario_counts["car_following"] += 1
                if lead_actor is None or not lead_actor.is_alive:
                    lead_actor = spawn_lead_vehicle(world, ego, carla_map, tm)
                    lead_last_change = time.time()
                elif ego.get_location().distance(lead_actor.get_location()) > 55.0:
                    cleanup_actor(lead_actor)
                    lead_actor = spawn_lead_vehicle(world, ego, carla_map, tm)
                    lead_last_change = time.time()

                if lead_actor and time.time() - lead_last_change > 4.0:
                    set_tm_target_speed(lead_actor, tm, random.uniform(10.0, 40.0))
                    lead_last_change = time.time()
            else:
                scenario_counts[scenario] += 1

            if scenario == "emergency":
                now = time.time()
                if (
                    emergency_actor is None
                    and speed > 5.0
                    and now - last_emergency_time > 18.0
                ):
                    spawn_requested.set()
                    last_emergency_time = now

                if spawn_requested.is_set() and emergency_actor is None and speed > 5.0:
                    spawn_requested.clear()
                    waypoint = waypoint_ahead(
                        carla_map, ego.get_location(), random.uniform(55.0, 75.0)
                    )
                    if waypoint is not None:
                        vehicle_bps = [
                            bp
                            for bp in world.get_blueprint_library().filter("vehicle.*")
                            if int(bp.get_attribute("number_of_wheels")) == 4
                        ]
                        bp = random.choice(vehicle_bps)
                        transform = waypoint.transform
                        transform.location.z += 0.5
                        emergency_actor = world.try_spawn_actor(bp, transform)
                        if emergency_actor:
                            emergency_actor.apply_control(carla.VehicleControl(brake=1.0))
                            emergency_count += 1
                            emergency_stopped_frames = 0
                            tm.auto_lane_change(ego, False)
                            print(
                                f"  Emergency obstacle #{emergency_count} spawned at "
                                f"{speed * 3.6:.1f} km/h"
                            )
                elif spawn_requested.is_set():
                    spawn_requested.clear()

                if emergency_actor is not None:
                    try:
                        emergency_actor.apply_control(carla.VehicleControl(brake=1.0))
                    except RuntimeError:
                        emergency_actor = None

                    if speed < 0.3:
                        emergency_stopped_frames += 1
                    else:
                        emergency_stopped_frames = 0

                    if emergency_stopped_frames >= FPS:
                        cleanup_actor(emergency_actor)
                        emergency_actor = None
                        emergency_stopped_frames = 0
                        tm.auto_lane_change(ego, True)
                        print("  Emergency obstacle removed")

            # --- Feature extraction (VISION ONLY, no radar) ---
            cam_frame = camera.get_frame()
            scene_features = {
                "visual": empty_visual_features(),
                "obstacle": empty_obstacle_features(),
            }
            if cam_frame is not None:
                scene_features = yolo.extract_scene_features(cam_frame)
            visual = scene_features["visual"]
            obstacle = scene_features["obstacle"]

            vision_tracker.update_ego_speed(speed)
            vision_state = vision_tracker.update(obstacle)

            if vision_state["relative_velocity"] > 0.1:
                ttc = min(
                    vision_state["distance"] / vision_state["relative_velocity"], 10.0
                )
            else:
                ttc = 10.0

            blocked_by_obstacle = vision_state["distance"] < STALL_MIN_CLEAR_DISTANCE_M
            trying_to_move = (
                control.throttle > STALL_MIN_THROTTLE and control.brake < STALL_MAX_BRAKE
            )
            if speed < STALL_SPEED_MPS and emergency_actor is None and trying_to_move and not blocked_by_obstacle:
                stuck_frames += 1
            else:
                stuck_frames = 0

            if stuck_frames >= STALL_FRAMES:
                stuck_frames = 0
                respawn_count += 1
                new_spawn = random.choice(safe_spawns)
                safe_respawn_ego(world, ego, new_spawn, tm, port)
                feature_history.clear()
                vision_tracker.reset()
                prev_speed = 0.0
                lead_last_change = time.time()
                print(f"  Ego safely respawned after confirmed stall ({respawn_count})")
                continue

            draw_camera_overlay(
                cam_frame,
                visual,
                obstacle,
                speed,
                scenario,
                vision_state,
                ttc,
                current_weather_name,
            )

            # Build feature dict — SAME column names as radar version
            base_features = {
                "ego_speed": round(speed, 4),
                "ego_acceleration": round(max(-20.0, min(20.0, accel)), 4),
                "distance": vision_state["distance"],
                "relative_velocity": vision_state["relative_velocity"],
                "ttc": round(ttc, 4),
                "obstacle_speed": vision_state["obstacle_speed"],
                "traffic_light_state": float(visual["traffic_light_state"]),
                "tl_confidence": round(visual["tl_confidence"], 4),
                "tl_bbox_area": round(visual["tl_bbox_area"], 6),
                "tl_center_x": round(visual["tl_center_x"], 4),
            }
            feature_history.append(base_features)

            if len(feature_history) == args.history:
                row = flatten_history(feature_history, BASE_FEATURE_COLS)
                row.update(
                    {
                        "frame": frame,
                        "timestamp": round(frame / FPS, 3),
                        "scenario": scenario,
                        "ego_speed_now": round(speed, 4),
                        "autopilot_throttle": round(control.throttle, 4),
                        "autopilot_brake": round(control.brake, 4),
                    }
                )
                samples.append(row)

            if frame % (FPS * 5) == 0 and frame > 0:
                tl_name = TL_STATE_NAMES.get(int(visual["traffic_light_state"]), "none")
                print(
                    f"  [{frame:>6,}/{total_frames:,}] "
                    f"scene={scenario:>13s} "
                    f"weather={current_weather_name:>11s} "
                    f"spd={speed * 3.6:5.1f}km/h "
                    f"vdist={vision_state['distance']:5.1f}m "
                    f"ttc={ttc:4.1f}s "
                    f"tl={tl_name:>6s} "
                    f"area={visual['tl_bbox_area']:.4f}"
                )

            try:
                spectator = world.get_spectator()
                transform = ego.get_transform()
                spectator.set_transform(
                    carla.Transform(
                        transform.location - transform.get_forward_vector() * 12
                        + carla.Location(z=6),
                        carla.Rotation(pitch=-20, yaw=transform.rotation.yaw),
                    )
                )
            except RuntimeError:
                pass

    except KeyboardInterrupt:
        print("\n  Collection interrupted")

    if samples:
        df = pd.DataFrame(samples)
        df["teacher_target_speed"] = compute_future_speed_label(df, args.label_horizon)
        df["teacher_target_speed"] = df["teacher_target_speed"].clip(lower=0.0)
        df = df.dropna(subset=["teacher_target_speed"]).reset_index(drop=True)
        df.to_csv(csv_path, index=False)

        config = {
            "town": args.town,
            "fps": FPS,
            "history_frames": args.history,
            "label_horizon": args.label_horizon,
            "base_feature_cols": BASE_FEATURE_COLS,
            "stacked_feature_cols": stacked_feature_names(BASE_FEATURE_COLS, args.history),
            "label_col": "teacher_target_speed",
            "distance_source": "yolo_vision",
        }
        with open(config_path, "w", encoding="utf-8") as fh:
            json.dump(config, fh, indent=2)

        print("\n" + "=" * 72)
        print("COLLECTION COMPLETE (VISION-ONLY)")
        print("=" * 72)
        print(f"  Samples saved:      {len(df):,}")
        print(f"  Emergency events:   {emergency_count}")
        print(f"  Respawns:           {respawn_count}")
        print(f"  Scenario coverage:  {scenario_counts}")
        print(f"  Dataset:            {csv_path}")
        print(f"  Config:             {config_path}")
        print("=" * 72)

    cleanup_actor(lead_actor)
    cleanup_actor(emergency_actor)
    camera.cleanup()
    if CV2_AVAILABLE:
        cv2.destroyAllWindows()

    for controller_id in ctrl_ids:
        actor = world.get_actor(controller_id)
        if actor:
            try:
                actor.stop()
            except RuntimeError:
                pass

    destroy_ids = npc_ids + ctrl_ids + walker_ids
    if destroy_ids:
        client.apply_batch([carla.command.DestroyActor(actor_id) for actor_id in destroy_ids])

    cleanup_actor(ego)
    world.apply_settings(original_settings)
    tm.set_synchronous_mode(False)


if __name__ == "__main__":
    main()
