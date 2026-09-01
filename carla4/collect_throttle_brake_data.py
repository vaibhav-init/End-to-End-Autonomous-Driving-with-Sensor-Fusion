#!/usr/bin/env python3
"""
Collect non-GT longitudinal driving data for target-speed imitation.

The collector now saves stacked temporal history instead of single-frame rows.
Each sample contains the latest N frames of radar + vision features, and the
training label is built later as a smoothed future speed target.
"""

import argparse
from collections import deque
import faulthandler
import json
import math
import os
import random
import sys
import threading
import time

import carla
import numpy as np
import pandas as pd

from radar import (
    FrontRadar,
    add_radar_arguments,
    create_front_radar,
    describe_radar_configuration,
    radar_diagnostics_row,
)
from driving_contract import (
    MAX_TARGET_SPEED_KMH,
    NATIVE_RADAR_POINTS_PER_SECOND,
    RADAR_RANGE_M,
    WEATHER_SEGMENT_S,
)
from yolo_perception import (
    CameraManager,
    TL_STATE_NAMES,
    YOLO_AVAILABLE,
    YOLOPerception,
    empty_obstacle_features,
    empty_visual_features,
)
from speed_model import BASE_FEATURE_COLS, feature_cols_for, flatten_history
from weather_utils import apply_random_fog

try:
    import cv2

    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("WARNING: OpenCV not available; camera preview disabled")


CARLA_HOST = "127.0.0.1"
CARLA_PORT = 2000
DEFAULT_TOWN = "Town04"
FPS = 20
MAX_RADAR_RANGE = RADAR_RANGE_M

NPC_VEHICLES = 45
NPC_PEDESTRIANS = 25
HISTORY_FRAMES = 10
LABEL_HORIZON = 10
SAVE_DIR = "dataset_throttle_brake"

# Scenario phases the autopilot teacher is driven through; the MLP imitates it.
#   traffic_light     - urban driving, obey lights
#   car_following     - follow a lead at a lower constant speed (NHTSA S3)
#   lead_decelerating - follow a lead that periodically slams its brakes (NHTSA S2)
#   emergency         - stopped vehicle appears ahead, must brake hard (NHTSA S1)
#   cut_in            - adjacent-lane vehicle forced into the ego's lane (NHTSA S4)
SCENARIOS = ("traffic_light", "car_following", "lead_decelerating", "emergency", "cut_in")
LEAD_BRAKE_PERIOD_S = 10.0    # how often the decelerating lead slams its brakes
CUT_IN_PERIOD_S = 12.0        # how often a fresh cut-in is forced
STALL_SPEED_MPS = 0.3
STALL_FRAMES = FPS * 8
STALL_MIN_THROTTLE = 0.2
STALL_MAX_BRAKE = 0.1
STALL_MIN_CLEAR_DISTANCE_M = 12.0
RESPAWN_Z_OFFSET = 0.5

_SCENARIOS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "scenarios"
)
if _SCENARIOS_DIR not in sys.path:
    sys.path.insert(0, _SCENARIOS_DIR)


def _load_idm_teacher(desired_speed_kmh, fps):
    """Build the IDM teacher, reusing the components the scenarios already use.

    The autopilot teacher almost never brakes hard, so a model imitating it
    never learns the response that matters -- measured directly: the shipped
    model asks for 53 km/h with a stopped car 8 m ahead. IDM brakes properly,
    is deterministic, and takes exactly the radar contract as input, so the
    labels it produces cover the regime the controller actually needs.
    """

    from drivers.longitudinal import (
        HybridStateMachineController,
        IntelligentDriverModel,
        PIDSpeedController,
    )
    from drivers.steering import BasicAgentSteering

    model = IntelligentDriverModel(
        desired_speed_mps=min(float(desired_speed_kmh), MAX_TARGET_SPEED_KMH) / 3.6
    )
    controller = HybridStateMachineController(PIDSpeedController(dt=1.0 / fps))
    return model, controller, BasicAgentSteering


def follow_ego_with_spectator(world, ego):
    """Point the CARLA spectator at the ego (third-person chase view).

    The spectator is a free camera and follows nothing on its own, so without
    this the window stays wherever it was left and the run appears to do
    nothing. Called from the settle loop as well as the main loop so the ego
    is visible from the moment it spawns rather than only once collection
    starts.
    """

    try:
        if not ego.is_alive:
            return
        transform = ego.get_transform()
        world.get_spectator().set_transform(
            carla.Transform(
                transform.location
                - transform.get_forward_vector() * 15
                + carla.Location(z=8),
                carla.Rotation(pitch=-20, yaw=transform.rotation.yaw),
            )
        )
    except RuntimeError:
        pass


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


def configure_ego_autopilot(ego, tm, port, max_speed_kmh=MAX_TARGET_SPEED_KMH):
    ego.set_autopilot(True, port)
    set_tm_target_speed(ego, tm, max_speed_kmh)
    tm.distance_to_leading_vehicle(ego, 10.0)
    tm.ignore_lights_percentage(ego, 0)
    tm.ignore_signs_percentage(ego, 0)
    tm.ignore_walkers_percentage(ego, 0)
    tm.auto_lane_change(ego, True)


def set_tm_target_speed(actor, tm, target_kmh):
    target_kmh = max(0.0, min(float(target_kmh), MAX_TARGET_SPEED_KMH))
    try:
        tm.set_desired_speed(actor, target_kmh)
        return
    except (AttributeError, RuntimeError):
        pass
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


def spawn_lead_vehicle(
    world,
    ego,
    carla_map,
    tm,
    ahead_options=(15.0, 20.0, 25.0, 30.0, 40.0),
):
    bp_lib = world.get_blueprint_library()
    vehicle_bps = [
        bp
        for bp in bp_lib.filter("vehicle.*")
        if int(bp.get_attribute("number_of_wheels")) >= 4
    ]
    port = tm.get_port()
    ahead_options = list(ahead_options)
    random.shuffle(ahead_options)
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


def spawn_npc_adjacent_lane(world, ego, carla_map, tm, ahead_m=25.0):
    """Spawn an NPC in a same-direction lane next to the ego, ahead_m ahead.

    Returns (vehicle, to_right) where to_right is the force_lane_change direction
    that moves the NPC INTO the ego's lane. Returns (None, True) on failure.
    """
    wp = carla_map.get_waypoint(ego.get_location(), project_to_road=True,
                                lane_type=carla.LaneType.Driving)
    if wp is None:
        return None, True

    right = wp.get_right_lane()
    left = wp.get_left_lane()
    if right and right.lane_type == carla.LaneType.Driving and right.lane_id * wp.lane_id > 0:
        adj, to_right = right, False   # NPC on the right -> cuts left into ego's lane
    elif left and left.lane_type == carla.LaneType.Driving and left.lane_id * wp.lane_id > 0:
        adj, to_right = left, True     # NPC on the left -> cuts right into ego's lane
    else:
        return None, True

    ahead = adj.next(ahead_m)
    if not ahead:
        return None, True

    bp_lib = world.get_blueprint_library()
    vehicle_bps = [
        bp for bp in bp_lib.filter("vehicle.*")
        if int(bp.get_attribute("number_of_wheels")) >= 4
    ]
    transform = ahead[0].transform
    transform.location.z += 0.5
    npc = world.try_spawn_actor(random.choice(vehicle_bps), transform)
    if npc:
        npc.set_autopilot(True, tm.get_port())
        tm.auto_lane_change(npc, False)
        set_tm_target_speed(npc, tm, 55.0)
        return npc, to_right
    return None, True


def cleanup_actor(actor):
    if actor and actor.is_alive:
        try:
            actor.destroy()
        except RuntimeError:
            pass


def compute_future_speed_label(df, horizon):
    future_speeds = [df["ego_speed_now"].shift(-step) for step in range(1, horizon + 1)]
    return pd.concat(future_speeds, axis=1).mean(axis=1)


def compute_segmented_speed_label(df, horizon, group_col="episode_id"):
    """Compute future-speed labels without crossing episode boundaries."""
    if group_col not in df.columns:
        return compute_future_speed_label(df, horizon)

    labels = pd.Series(np.nan, index=df.index, dtype=np.float64)
    for _, indices in df.groupby(group_col, sort=False).groups.items():
        segment = df.loc[indices]
        labels.loc[indices] = compute_future_speed_label(segment, horizon)
    return labels


def safe_respawn_ego(
    world,
    ego,
    spawn_transform,
    tm,
    port,
    max_speed_kmh=MAX_TARGET_SPEED_KMH,
):
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
    configure_ego_autopilot(ego, tm, port, max_speed_kmh=max_speed_kmh)
    world.tick()


def draw_camera_overlay(frame, visual, obstacle, speed, scenario, radar_state, ttc, weather_name, teacher="", target_speed=0.0):
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
        f"Scenario: {scenario}",
        f"Weather: {weather_name}",
        f"Speed: {speed * 3.6:.1f} km/h",
        f"TL: {TL_STATE_NAMES.get(tl_state, 'none')}",
        f"TL center_x: {visual['tl_center_x']:.2f}",
        f"Dist: {radar_state['distance']:.1f}m TTC: {ttc:.1f}s",
    ]
    for line_index, line in enumerate(hud_lines):
        cv2.putText(
            display,
            line,
            (10, 25 + line_index * 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

    if teacher:
        cv2.putText(
            display,
            f"teacher={teacher} target={target_speed * 3.6:5.1f}km/h "
            f"gap={radar_state['distance']:5.1f}m",
            (10, display.shape[0] - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            1,
        )
    cv2.imshow("CARLA Collector Camera", display)
    cv2.waitKey(1)


def main():
    parser = argparse.ArgumentParser(description="Collect target-speed imitation data")
    parser.add_argument("--host", default=CARLA_HOST)
    parser.add_argument("--port", type=int, default=CARLA_PORT)
    parser.add_argument("--town", default=DEFAULT_TOWN)
    parser.add_argument("--duration", type=int, default=900)
    parser.add_argument("--vehicles", type=int, default=NPC_VEHICLES)
    parser.add_argument("--pedestrians", type=int, default=NPC_PEDESTRIANS)
    parser.add_argument("--history", type=int, default=HISTORY_FRAMES)
    parser.add_argument("--label-horizon", type=int, default=LABEL_HORIZON)
    parser.add_argument(
        "--max-speed-kmh",
        type=float,
        default=MAX_TARGET_SPEED_KMH,
        help="Hard ceiling for teacher speed and target-speed labels",
    )
    parser.add_argument(
        "--weather-interval-s",
        type=int,
        default=WEATHER_SEGMENT_S,
        help="Change weather within each scenario at this interval",
    )
    parser.add_argument(
        "--teacher",
        choices=("autopilot", "idm"),
        default="idm",
        help=(
            "policy that drives the ego and therefore defines the labels. "
            "autopilot rarely brakes hard, so a model imitating it does not "
            "learn to brake; idm does and is deterministic (default: idm)"
        ),
    )
    parser.add_argument(
        "--client-timeout",
        type=float,
        default=120.0,
        help=(
            "CARLA client timeout. The first ticks after attaching semantic "
            "LiDAR, camera and YOLO are far slower than steady state, and the "
            "old 30 s was not enough to get through them"
        ),
    )
    parser.add_argument(
        "--watchdog-s",
        type=float,
        default=0.0,
        help=(
            "if a tick blocks this long, dump every thread's stack and exit. "
            "A CARLA sensor callback that stalls shows up only as a client "
            "timeout otherwise, with no indication of where"
        ),
    )
    parser.add_argument(
        "--no-vision",
        dest="vision",
        action="store_false",
        help=(
            "collect from radar alone: no RGB camera, no YOLO. The traffic-"
            "light features stay in the schema but are held at zero, so the "
            "column contract is unchanged and vision can be reinstated later "
            "without recollecting. Also removes the camera sensor and YOLO's "
            "CUDA context, which contend with CARLA's renderer"
        ),
    )
    parser.add_argument(
        "--radar-points-per-second",
        type=int,
        default=None,
        help=(
            "semantic-LiDAR density for the non-native backends. Defaults to "
            "240000, which is 80x what the native radar uses and appears to be "
            "what stalls synchronous ticks; lower it if collection hangs"
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default=SAVE_DIR)
    add_radar_arguments(parser)
    args = parser.parse_args()
    if not 0.0 < args.max_speed_kmh <= MAX_TARGET_SPEED_KMH:
        parser.error(
            f"--max-speed-kmh must be in (0, {MAX_TARGET_SPEED_KMH:g}]"
        )
    if args.weather_interval_s <= 0:
        parser.error("--weather-interval-s must be positive")
    random.seed(args.seed)
    np.random.seed(args.seed)

    total_frames = args.duration * FPS
    radar_points_per_second = (
        int(args.radar_points_per_second)
        if args.radar_points_per_second
        else (
            NATIVE_RADAR_POINTS_PER_SECOND
            if args.radar_backend == "native"
            else 240000
        )
    )
    # Radar-only runs drop the traffic-light columns entirely rather than
    # pinning them at zero: ten frames of four constants would spend 40 of the
    # model's 100 inputs saying nothing.
    active_feature_cols = feature_cols_for(args.vision)
    print(
        f"  Features:        {len(active_feature_cols)} base x {args.history} "
        f"frames = {len(active_feature_cols) * args.history} inputs"
        f"{'' if args.vision else '  (radar only)'}"
    )

    radar_metadata = describe_radar_configuration(
        backend=args.radar_backend,
        range_m=MAX_RADAR_RANGE,
        fps=FPS,
        points_per_second=radar_points_per_second,
        profile_name=args.radar_profile,
        config_path=args.radar_config,
        ghost_detector_path=args.radar_ghost_detector,
        ghost_threshold=args.radar_ghost_threshold,
    )
    os.makedirs(args.output, exist_ok=True)
    csv_path = os.path.join(args.output, "data.csv")
    config_path = os.path.join(args.output, "dataset_config.json")
    # Recorded so the deployed driver can match how the data was collected:
    # a model trained with the traffic-light columns pinned at zero must not
    # be fed live ones at runtime.
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as fh:
            existing_config = json.load(fh)
        expected = {
            "town": args.town,
            "max_target_speed_kmh": args.max_speed_kmh,
            "vision_enabled": bool(args.vision),
            "teacher": args.teacher,
            "history_frames": args.history,
            "label_horizon": args.label_horizon,
        }
        expected.update(
            {
                key: value
                for key, value in radar_metadata.items()
                if key != "radar_config"
            }
        )
        mismatches = []
        for key, requested in expected.items():
            existing = existing_config.get(key, requested)
            if isinstance(requested, float):
                matches = abs(float(existing) - requested) <= 1e-6
            else:
                matches = existing == requested
            if not matches:
                mismatches.append(
                    f"{key}: existing={existing!r}, requested={requested!r}"
                )
        if mismatches:
            raise RuntimeError(
                "Refusing to mix incompatible data in the same output folder: "
                + "; ".join(mismatches)
                + ". Use a new --output directory or remove the old dataset."
            )

    print("=" * 72)
    print("TARGET-SPEED DATA COLLECTOR")
    print("=" * 72)
    print(f"  Town:            {args.town}")
    print(f"  Radar range:     {MAX_RADAR_RANGE:.0f}m")
    print(f"  Radar backend:   {args.radar_backend}")
    if args.radar_backend == "realistic":
        print(f"  Radar profile:   {radar_metadata['radar_profile']}")
        print(f"  Radar config ID: {radar_metadata['radar_config_signature']}")
    print(f"  Duration:        {args.duration}s")
    print(f"  History frames:  {args.history}")
    print(f"  Label horizon:   {args.label_horizon}")
    print(f"  Speed ceiling:   {args.max_speed_kmh:.1f} km/h")
    print(f"  Weather segment: {args.weather_interval_s}s")
    print(f"  Seed:            {args.seed}")
    print(f"  Vehicles:        {args.vehicles}")
    print(f"  Pedestrians:     {args.pedestrians}")
    print(f"  Output:          {csv_path}")
    print(f"  Scenarios:       {', '.join(SCENARIOS)}")
    print("=" * 72)

    client = carla.Client(args.host, args.port)
    client.set_timeout(float(args.client_timeout))

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
    configure_ego_autopilot(ego, tm, port, max_speed_kmh=args.max_speed_kmh)

    # Teacher selection. The autopilot path is unchanged; the IDM path takes
    # longitudinal control away from the traffic manager and leaves it only
    # the route, with BasicAgent supplying steer exactly as the scenario
    # drivers do.
    idm_model = idm_controller = idm_steering = None
    if args.teacher == "idm":
        idm_model, idm_controller, steering_class = _load_idm_teacher(
            args.max_speed_kmh, FPS
        )
        ego.set_autopilot(False)
        idm_steering = steering_class(ego, world.get_map())
        print(
            f"  Teacher: IDM (v0={idm_model.desired_speed_mps * 3.6:.0f} km/h, "
            f"T={idm_model.time_headway_s}s, s0={idm_model.minimum_gap_m}m); "
            "autopilot disabled, BasicAgent steers"
        )
    else:
        print("  Teacher: CARLA autopilot")

    npc_ids = spawn_vehicles(world, client, tm, args.vehicles)
    walker_ids, ctrl_ids = spawn_pedestrians(world, args.pedestrians)

    radar = create_front_radar(
        ego,
        world,
        MAX_RADAR_RANGE,
        backend=args.radar_backend,
        fps=FPS,
        points_per_second=radar_points_per_second,
        profile_name=args.radar_profile,
        config_path=args.radar_config,
        seed=args.radar_seed if args.radar_seed is not None else args.seed,
        ghost_detector_path=args.radar_ghost_detector,
        ghost_threshold=args.radar_ghost_threshold,
        ghost_device=args.radar_ghost_device,
    )
    if args.vision:
        camera = CameraManager(ego, world)
        yolo = YOLOPerception() if YOLO_AVAILABLE else None
        if yolo is None:
            print("  YOLO unavailable; traffic-light features will be zeroed")
    else:
        camera = None
        yolo = None
        print("  Vision disabled: radar only, traffic-light features zeroed")
    current_weather_name = apply_random_fog(world)
    print(f"  Weather:         {current_weather_name}")

    # First ticks after the sensors attach are much slower than steady state;
    # report them so a stall here is visible instead of just timing out.
    if args.watchdog_s > 0:
        faulthandler.enable(all_threads=True)
        faulthandler.dump_traceback_later(
            float(args.watchdog_s), repeat=False, exit=True
        )
        print(f"  Watchdog armed at {args.watchdog_s:.0f}s")

    settle_start = time.time()
    for settle_index in range(40):
        tick_start = time.time()
        world.tick()
        follow_ego_with_spectator(world, ego)
        tick_elapsed = time.time() - tick_start
        if args.watchdog_s > 0:
            faulthandler.cancel_dump_traceback_later()
            faulthandler.dump_traceback_later(
                float(args.watchdog_s), repeat=False, exit=True
            )
        if settle_index < 3 or tick_elapsed > 1.0:
            print(
                f"  settle tick {settle_index:2d}: {tick_elapsed:6.2f}s",
                flush=True,
            )
    if args.watchdog_s > 0:
        # The settle loop re-arms the watchdog each tick; leaving the last one
        # pending would kill the collection itself a few seconds in.
        faulthandler.cancel_dump_traceback_later()
    print(f"  Sensors settled in {time.time() - settle_start:.1f}s")

    feature_history = deque(maxlen=args.history)
    samples = []
    scenario_counts = {name: 0 for name in SCENARIOS}

    prev_speed = 0.0
    active_scenario = None
    segment_counter = -1
    active_episode_id = None
    last_weather_change_frame = -args.weather_interval_s * FPS
    lead_actor = None
    lead_last_change_frame = 0
    emergency_actor = None
    emergency_stopped_frames = 0
    last_emergency_frame = 0
    emergency_count = 0
    respawn_count = 0
    stuck_frames = 0
    absolute_stuck_frames = 0
    lead_brake_until = 0
    lead_brake_force = 1.0
    last_lead_brake_frame = 0
    cut_in_npc = None
    cut_in_to_right = True
    last_cut_in_frame = 0

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

            progress = frame / max(1, total_frames)
            scenario_index = min(len(SCENARIOS) - 1, int(progress * len(SCENARIOS)))
            scenario = SCENARIOS[scenario_index]
            if scenario != active_scenario:
                active_scenario = scenario
                current_weather_name = apply_random_fog(world)
                last_weather_change_frame = frame
                segment_counter += 1
                active_episode_id = f"base_{segment_counter:04d}_{scenario}"
                feature_history.clear()
                prev_speed = speed
                last_emergency_frame = frame
                last_lead_brake_frame = frame
                last_cut_in_frame = frame
                print(f"\n  Switching scenario -> {scenario}")
                print(f"  Fog preset -> {current_weather_name}")
                if scenario in ("car_following", "lead_decelerating"):
                    tm.auto_lane_change(ego, False)
                else:
                    tm.auto_lane_change(ego, True)
                    cleanup_actor(lead_actor)
                    lead_actor = None
                if scenario != "emergency":
                    cleanup_actor(emergency_actor)
                    emergency_actor = None
                    emergency_stopped_frames = 0
                if scenario != "cut_in":
                    cleanup_actor(cut_in_npc)
                    cut_in_npc = None
            elif frame - last_weather_change_frame >= args.weather_interval_s * FPS:
                current_weather_name = apply_random_fog(world)
                last_weather_change_frame = frame
                segment_counter += 1
                active_episode_id = f"base_{segment_counter:04d}_{scenario}"
                feature_history.clear()
                prev_speed = speed
                print(f"\n  Weather segment -> {current_weather_name} ({scenario})")

            scenario_counts[scenario] += 1

            if scenario in ("car_following", "lead_decelerating"):
                if lead_actor is None or not lead_actor.is_alive:
                    lead_actor = spawn_lead_vehicle(world, ego, carla_map, tm)
                    lead_last_change_frame = frame
                    lead_brake_until = 0
                elif ego.get_location().distance(lead_actor.get_location()) > 55.0:
                    cleanup_actor(lead_actor)
                    lead_actor = spawn_lead_vehicle(world, ego, carla_map, tm)
                    lead_last_change_frame = frame
                    lead_brake_until = 0

                if lead_actor and frame - lead_last_change_frame > 4 * FPS:
                    set_tm_target_speed(lead_actor, tm, random.uniform(10.0, 40.0))
                    lead_last_change_frame = frame

                # S2: the decelerating lead periodically slams its brakes
                if scenario == "lead_decelerating" and lead_actor and lead_actor.is_alive:
                    if frame < lead_brake_until:
                        lead_actor.apply_control(
                            carla.VehicleControl(
                                throttle=0.0,
                                brake=lead_brake_force,
                            )
                        )
                        if frame == lead_brake_until - 1:
                            lead_actor.set_autopilot(True, port)
                            set_tm_target_speed(lead_actor, tm, random.uniform(20.0, 40.0))
                    elif frame - last_lead_brake_frame > LEAD_BRAKE_PERIOD_S * FPS:
                        lead_actor.set_autopilot(False)
                        lead_brake_force = random.choice((0.5, 0.75, 1.0))
                        brake_duration_s = random.uniform(1.5, 3.5)
                        lead_brake_until = frame + int(brake_duration_s * FPS)
                        last_lead_brake_frame = frame
                        print(
                            "  Lead vehicle decelerating (S2): "
                            f"brake={lead_brake_force:.2f}, "
                            f"duration={brake_duration_s:.1f}s"
                        )

            if scenario == "emergency":
                if (
                    emergency_actor is None
                    and speed > 5.0
                    and frame - last_emergency_frame > 18 * FPS
                ):
                    spawn_requested.set()
                    last_emergency_frame = frame

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

            if scenario == "cut_in":
                if cut_in_npc is None or not cut_in_npc.is_alive:
                    if speed > 5.0:
                        cut_in_npc, cut_in_to_right = spawn_npc_adjacent_lane(
                            world, ego, carla_map, tm)
                        last_cut_in_frame = frame
                elif ego.get_location().distance(cut_in_npc.get_location()) > 60.0:
                    cleanup_actor(cut_in_npc)
                    cut_in_npc = None
                elif frame - last_cut_in_frame > CUT_IN_PERIOD_S * FPS:
                    try:
                        tm.force_lane_change(cut_in_npc, cut_in_to_right)
                        print("  NPC forced cut-in (S4)")
                    except RuntimeError:
                        pass
                    last_cut_in_frame = frame

            radar.update_ego_speed(speed)
            radar_state = radar.get()
            radar_diagnostics = radar_diagnostics_row(radar)
            if radar_state["relative_velocity"] > 0.1:
                ttc = min(radar_state["distance"] / radar_state["relative_velocity"], 10.0)
            else:
                ttc = 10.0

            teacher_target_speed = 0.0
            if idm_model is not None:
                teacher_target_speed = idm_model.target_speed(
                    speed_mps=speed,
                    gap_m=radar_state["distance"],
                    closing_speed_mps=radar_state["relative_velocity"],
                    dt=1.0 / FPS,
                )
                throttle_cmd, brake_cmd = idm_controller.run_step(
                    teacher_target_speed, speed
                )
                control = carla.VehicleControl(
                    throttle=float(throttle_cmd),
                    steer=float(idm_steering.get_steer()),
                    brake=float(brake_cmd),
                )
                ego.apply_control(control)

            cam_frame = camera.get_frame() if camera is not None else None
            visual = empty_visual_features()
            obstacle = empty_obstacle_features()
            if yolo is not None and cam_frame is not None:
                scene_features = yolo.extract_scene_features(cam_frame)
                visual = scene_features["visual"]
                obstacle = scene_features["obstacle"]

            blocked_by_obstacle = radar_state["distance"] < STALL_MIN_CLEAR_DISTANCE_M
            trying_to_move = (
                control.throttle > STALL_MIN_THROTTLE and control.brake < STALL_MAX_BRAKE
            )
            if speed < STALL_SPEED_MPS:
                absolute_stuck_frames += 1
                if emergency_actor is None and trying_to_move and not blocked_by_obstacle:
                    stuck_frames += 1
                else:
                    stuck_frames = 0 
            else:
                stuck_frames = 0
                absolute_stuck_frames = 0

            if stuck_frames >= STALL_FRAMES or absolute_stuck_frames >= FPS * 30:
                stuck_frames = 0
                absolute_stuck_frames = 0
                respawn_count += 1
                new_spawn = random.choice(safe_spawns)
                safe_respawn_ego(
                    world,
                    ego,
                    new_spawn,
                    tm,
                    port,
                    max_speed_kmh=args.max_speed_kmh,
                )
                if idm_model is not None:
                    # safe_respawn_ego re-enables autopilot; the IDM teacher
                    # must take longitudinal control back and re-aim BasicAgent
                    # at the new location, or the ego drives itself from here.
                    ego.set_autopilot(False)
                    idm_steering = steering_class(ego, world.get_map())
                    idm_controller.pid.reset()
                feature_history.clear()
                prev_speed = 0.0
                lead_last_change_frame = frame
                segment_counter += 1
                active_episode_id = f"base_{segment_counter:04d}_{scenario}"
                print(f"  Ego safely respawned after confirmed stall ({respawn_count})")
                continue

            if camera is not None:
                draw_camera_overlay(
                    cam_frame,
                    visual,
                    obstacle,
                    speed,
                    scenario,
                    radar_state,
                    ttc,
                    current_weather_name,
                    teacher=args.teacher,
                    target_speed=teacher_target_speed,
                )

            base_features = {
                "ego_speed": round(speed, 4),
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
            feature_history.append(base_features)

            if len(feature_history) == args.history:
                row = flatten_history(feature_history, active_feature_cols)
                row.update(
                    {
                        "frame": frame,
                        "timestamp": round(frame / FPS, 3),
                        "scenario": scenario,
                        "weather": current_weather_name,
                        "episode_id": active_episode_id,
                        "ego_speed_now": round(speed, 4),
                        "teacher": args.teacher,
                        # Distinct from the label column of the same idea:
                        # this is what the teacher asked for on this frame,
                        # while teacher_target_speed is the smoothed future
                        # speed computed at save time.
                        "teacher_command_speed": round(teacher_target_speed, 4),
                        "autopilot_throttle": round(control.throttle, 4),
                        "autopilot_brake": round(control.brake, 4),
                    }
                )
                row.update(radar_diagnostics)
                samples.append(row)

            if frame % (FPS * 5) == 0 and frame > 0:
                tl_name = TL_STATE_NAMES.get(int(visual["traffic_light_state"]), "none")
                print(
                    f"  [{frame:>6,}/{total_frames:,}] "
                    f"scene={scenario:>13s} "
                    f"weather={current_weather_name:>11s} "
                    f"spd={speed * 3.6:5.1f}km/h "
                    f"dist={radar_state['distance']:5.1f}m "
                    f"ttc={ttc:4.1f}s "
                    f"tl={tl_name:>6s} "
                    f"area={visual['tl_bbox_area']:.4f}"
                )

            follow_ego_with_spectator(world, ego)

    except KeyboardInterrupt:
        print("\n  Collection interrupted")
    finally:
        # Save whatever was collected (best-effort), then ALWAYS clean the map \u2014
        # even on Ctrl+C or an error \u2014 so the next run starts from a clean world.
        try:
            if samples:
                df = pd.DataFrame(samples)
                df["teacher_target_speed"] = compute_segmented_speed_label(
                    df,
                    args.label_horizon,
                )
                df["teacher_target_speed"] = df["teacher_target_speed"].clip(
                    lower=0.0,
                    upper=args.max_speed_kmh / 3.6,
                )
                df = df.dropna(subset=["teacher_target_speed"]).reset_index(drop=True)
                df.to_csv(csv_path, index=False)

                config = {
                    "town": args.town,
                    "fps": FPS,
                    "max_target_speed_kmh": args.max_speed_kmh,
                    "weather_interval_s": args.weather_interval_s,
                    "collection_seed": args.seed,
                    "history_frames": args.history,
                    "label_horizon": args.label_horizon,
                    "vision_enabled": bool(args.vision),
                    "teacher": args.teacher,
                    "base_feature_cols": active_feature_cols,
                    "stacked_feature_cols": stacked_feature_names(
                        active_feature_cols, args.history
                    ),
                    "label_col": "teacher_target_speed",
                }
                config.update(radar_metadata)
                with open(config_path, "w", encoding="utf-8") as fh:
                    json.dump(config, fh, indent=2)

                print("\n" + "=" * 72)
                print("COLLECTION COMPLETE")
                print("=" * 72)
                print(f"  Samples saved:      {len(df):,}")
                print(f"  Emergency events:   {emergency_count}")
                print(f"  Respawns:           {respawn_count}")
                print(f"  Scenario coverage:  {scenario_counts}")
                print(f"  Dataset:            {csv_path}")
                print(f"  Config:             {config_path}")
                print("=" * 72)
        except KeyboardInterrupt:
            print("  Save interrupted; cleaning up anyway")
        except Exception as exc:  # noqa: BLE001
            print(f"  Save failed: {exc}")

        print("\n  Cleaning up the map ...")
        cleanup_actor(lead_actor)
        cleanup_actor(emergency_actor)
        cleanup_actor(cut_in_npc)
        radar.cleanup()
        if camera is not None:
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
        try:
            world.apply_settings(original_settings)
            tm.set_synchronous_mode(False)
        except RuntimeError:
            pass
        print("  Cleanup done.")


if __name__ == "__main__":
    main()
