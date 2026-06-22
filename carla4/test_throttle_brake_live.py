#!/usr/bin/env python3
"""
Live test for the target-speed sequence model.
"""

import argparse
from collections import deque
import json
import math
import os
import pickle
import random
import sys
import threading
import time

import carla
import numpy as np
import torch

from yolo_perception import (
    CameraManager,
    TL_STATE_NAMES,
    VisionDistanceTracker,
    YOLO_AVAILABLE,
    YOLOPerception,
    empty_obstacle_features,
    empty_visual_features,
)
from speed_model import BASE_FEATURE_COLS as DEFAULT_BASE_FEATURE_COLS
from speed_model import TargetSpeedMLP, flatten_history
from weather_utils import apply_random_fog


CARLA_ROOT = os.environ.get("CARLA_ROOT", "/opt/carla-simulator")
AGENTS_PATH = os.path.join(CARLA_ROOT, "PythonAPI", "carla")
if AGENTS_PATH not in sys.path:
    sys.path.insert(0, AGENTS_PATH)

try:
    from agents.navigation.basic_agent import BasicAgent
except ImportError:
    print("ERROR: CARLA agents not found. Please ensure CARLA PythonAPI is in your PYTHONPATH.")
    print(f"       Tried: {AGENTS_PATH}")
    sys.exit(1)


CARLA_HOST = "127.0.0.1"
CARLA_PORT = 2000
DEFAULT_TOWN = "Town01"
FPS = 20
MAX_RADAR_RANGE = 50.0
BOOTSTRAP_TARGET_SPEED_MPS = 12.0 / 3.6
LAUNCH_HOLD_SPEED_MPS = 0.5
LAUNCH_CLEAR_DISTANCE_M = 15.0
LAUNCH_ASSIST_FRAMES = FPS * 5

MODEL_DIRS = {
    "cameraplusradar": "model_throttle_brake",
    "cameraonly": "model_vision_only",
}

class FrontRadar:
    def __init__(self, vehicle, world, range_m=50.0):
        self.latest = {
            "distance": range_m,
            "relative_velocity": 0.0,
            "obstacle_speed": 0.0,
        }
        self._ego_speed = 0.0
        self._range = range_m

        bp = world.get_blueprint_library().find("sensor.other.radar")
        bp.set_attribute("horizontal_fov", "10")
        bp.set_attribute("vertical_fov", "2")
        bp.set_attribute("range", str(range_m))
        bp.set_attribute("points_per_second", "1500")
        transform = carla.Transform(
            carla.Location(x=2.5, z=1.0),
            carla.Rotation(pitch=2.0),
        )
        self.sensor = world.spawn_actor(bp, transform, attach_to=vehicle)
        self.sensor.listen(self._on_radar)

    def _on_radar(self, data):
        nearest_dist = self._range
        nearest_vel = 0.0
        for det in data:
            if abs(det.azimuth) > 0.3 or det.depth < 1.0 or det.altitude < -0.02:
                continue
            if det.depth < nearest_dist:
                nearest_dist = det.depth
                nearest_vel = det.velocity

        rel_vel = -nearest_vel
        obs_speed = max(0.0, self._ego_speed - rel_vel)
        self.latest = {
            "distance": nearest_dist,
            "relative_velocity": rel_vel,
            "obstacle_speed": obs_speed,
        }

    def update_ego_speed(self, speed):
        self._ego_speed = speed

    def get(self):
        return self.latest.copy()

    def cleanup(self):
        if self.sensor and self.sensor.is_alive:
            self.sensor.destroy()


class CollisionRecorder:
    COOLDOWN = 3.0
    MIN_IMPULSE = 200.0

    def __init__(self, vehicle, world):
        self.collisions = []
        self._last = {}
        bp = world.get_blueprint_library().find("sensor.other.collision")
        self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=vehicle)
        self.sensor.listen(self._on_collision)

    def _on_collision(self, event):
        now = time.time()
        actor_type = event.other_actor.type_id
        impulse = event.normal_impulse.length()
        if not (actor_type.startswith("vehicle.") or actor_type.startswith("walker.")):
            return
        if impulse < self.MIN_IMPULSE:
            return
        actor_id = event.other_actor.id
        if actor_id in self._last and now - self._last[actor_id] < self.COOLDOWN:
            return
        self._last[actor_id] = now
        self.collisions.append({"time": now, "actor": actor_type, "impulse": impulse})
        print(f"\n  COLLISION with {actor_type} ({impulse:.0f} N*s)")

    def cleanup(self):
        if self.sensor and self.sensor.is_alive:
            self.sensor.destroy()


class PIDSpeedController:
    """Convert desired speed into throttle/brake."""

    def __init__(self, dt, kp=0.75, ki=0.08, kd=0.12):
        self.dt = dt
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral = 0.0
        self.prev_error = 0.0

    def run_step(self, target_speed, current_speed):
        error = target_speed - current_speed
        self.integral += error * self.dt
        self.integral = max(-5.0, min(5.0, self.integral))
        derivative = (error - self.prev_error) / max(self.dt, 1e-6)
        self.prev_error = error

        accel_cmd = self.kp * error + self.ki * self.integral + self.kd * derivative
        accel_cmd = max(-1.0, min(1.0, accel_cmd))

        if accel_cmd >= 0.0:
            throttle = accel_cmd
            brake = 0.0
        else:
            throttle = 0.0
            brake = -accel_cmd

        return throttle, brake


class HybridStateMachineController:
    """
    Manages transitions between continuous dynamic control (PID) 
    and discrete static states (Brake Hold) for safe boundary conditions.
    """
    def __init__(self, pid_controller, hold_speed_threshold=0.3, stop_current_speed_threshold=0.8, hold_brake_force=0.35):
        self.pid = pid_controller
        self.hold_speed_threshold = hold_speed_threshold
        self.stop_current_speed_threshold = stop_current_speed_threshold
        self.hold_brake_force = hold_brake_force
        self.state = "CONTINUOUS"

    def run_step(self, target_speed, current_speed):
        throttle, brake = self.pid.run_step(target_speed, current_speed)
        
        if target_speed < self.hold_speed_threshold and current_speed < self.stop_current_speed_threshold:
            self.state = "HOLD"
            return 0.0, max(brake, self.hold_brake_force)
        
        self.state = "CONTINUOUS"
        return throttle, brake


def cleanup_actor(actor):
    if actor and actor.is_alive:
        try:
            actor.destroy()
        except RuntimeError:
            pass


def choose_new_destination(ego, spawn_points, min_distance=35.0):
    current_location = ego.get_location()
    candidates = [sp.location for sp in spawn_points if sp.location.distance(current_location) >= min_distance]
    if not candidates:
        candidates = [sp.location for sp in spawn_points]
    return random.choice(candidates)


def waypoint_ahead(carla_map, location, distance_m):
    waypoint = carla_map.get_waypoint(location, project_to_road=True)
    if waypoint is None:
        return None
    travelled = 0.0
    step = 3.0
    while travelled < distance_m:
        next_wps = waypoint.next(step)
        if not next_wps:
            return None
        waypoint = next_wps[0]
        travelled += step
    return waypoint


def spawn_stopped_vehicle(world, ego, carla_map, ahead_m=35.0):
    waypoint = waypoint_ahead(carla_map, ego.get_location(), ahead_m)
    if waypoint is None:
        return None

    bp_lib = world.get_blueprint_library()
    vehicle_bps = [
        bp
        for bp in bp_lib.filter("vehicle.*")
        if int(bp.get_attribute("number_of_wheels")) >= 4
    ]
    bp = random.choice(vehicle_bps)
    transform = waypoint.transform
    transform.location.z += 0.5
    vehicle = world.try_spawn_actor(bp, transform)
    if vehicle:
        vehicle.apply_control(carla.VehicleControl(brake=1.0))
        vehicle.set_target_velocity(carla.Vector3D(0.0, 0.0, 0.0))
        print(f"  Spawned stopped vehicle at {ahead_m:.0f}m")
    return vehicle


def spawn_sudden_braker(world, ego, carla_map, ahead_m=28.0):
    waypoint = waypoint_ahead(carla_map, ego.get_location(), ahead_m)
    if waypoint is None:
        return None

    bp_lib = world.get_blueprint_library()
    vehicle_bps = [
        bp
        for bp in bp_lib.filter("vehicle.*")
        if int(bp.get_attribute("number_of_wheels")) >= 4
    ]
    bp = random.choice(vehicle_bps)
    transform = waypoint.transform
    transform.location.z += 0.5
    vehicle = world.try_spawn_actor(bp, transform)
    if vehicle:
        fwd = transform.get_forward_vector()
        vehicle.enable_constant_velocity(carla.Vector3D(fwd.x * 8.0, fwd.y * 8.0, 0.0))
        print(f"  Spawned sudden braker at {ahead_m:.0f}m")
    return vehicle


def spawn_pedestrian_crossing(world, ego, carla_map, ahead_m=20.0):
    waypoint = waypoint_ahead(carla_map, ego.get_location(), ahead_m)
    if waypoint is None:
        return None, None

    bp_lib = world.get_blueprint_library()
    walker_bp = random.choice(bp_lib.filter("walker.pedestrian.*"))
    if walker_bp.has_attribute("is_invincible"):
        walker_bp.set_attribute("is_invincible", "false")

    transform = waypoint.transform
    right = transform.get_right_vector()
    transform.location.x -= right.x * 4.0
    transform.location.y -= right.y * 4.0
    transform.location.z += 0.5

    walker = world.try_spawn_actor(walker_bp, transform)
    controller = None
    if walker:
        ctrl_bp = bp_lib.find("controller.ai.walker")
        controller = world.spawn_actor(ctrl_bp, carla.Transform(), attach_to=walker)
        world.tick()
        dest = waypoint.transform.location + carla.Location(x=right.x * 8.0, y=right.y * 8.0)
        controller.start()
        controller.go_to_location(dest)
        controller.set_max_speed(1.8)
        print(f"  Spawned pedestrian crossing at {ahead_m:.0f}m")
    return walker, controller


def spawn_background_traffic(world, client, tm, count):
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
            tm.ignore_lights_percentage(actor, 0)
            tm.ignore_signs_percentage(actor, 0)
    print(f"  Spawned {len(ids)} background vehicles")
    return ids


def spawn_background_pedestrians(world, count):
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

    print(f"  Spawned {len(walkers)} background pedestrians")
    return [w.id for w in walkers], [c.id for c in controllers]


def init_scenario_state():
    return {
        "name": None,
        "vehicle": None,
        "walker": None,
        "controller": None,
        "spawn_time": 0.0,
        "brake_after": None,
        "done": False,
    }


def destroy_scenario_state(state):
    if state["controller"]:
        try:
            state["controller"].stop()
        except RuntimeError:
            pass
    cleanup_actor(state["controller"])
    cleanup_actor(state["walker"])
    if state["vehicle"]:
        try:
            state["vehicle"].disable_constant_velocity()
        except RuntimeError:
            pass
    cleanup_actor(state["vehicle"])


def maybe_spawn_scenario(world, ego, carla_map, state):
    choice = random.choice(["stopped_vehicle", "sudden_braker", "pedestrian"])
    state["name"] = choice
    state["spawn_time"] = time.time()
    state["brake_after"] = None
    state["done"] = False

    if choice == "stopped_vehicle":
        state["vehicle"] = spawn_stopped_vehicle(world, ego, carla_map)
    elif choice == "sudden_braker":
        state["vehicle"] = spawn_sudden_braker(world, ego, carla_map)
        state["brake_after"] = state["spawn_time"] + 3.5
    else:
        state["walker"], state["controller"] = spawn_pedestrian_crossing(world, ego, carla_map)

    if state["vehicle"] or state["walker"]:
        return True

    state["name"] = None
    return False


def update_scenario_state(state):
    if state["name"] == "sudden_braker" and state["vehicle"] and state["brake_after"]:
        if time.time() >= state["brake_after"] and not state["done"]:
            try:
                state["vehicle"].disable_constant_velocity()
            except RuntimeError:
                pass
            try:
                state["vehicle"].apply_control(carla.VehicleControl(brake=1.0))
            except RuntimeError:
                pass
            state["done"] = True


def main():
    parser = argparse.ArgumentParser(description="Live test target-speed sequence model")
    parser.add_argument("--host", default=CARLA_HOST)
    parser.add_argument("--port", type=int, default=CARLA_PORT)
    parser.add_argument("--town", default=DEFAULT_TOWN)
    parser.add_argument(
        "--mode",
        choices=["cameraonly", "cameraplusradar"],
        default="cameraplusradar",
        help="cameraonly = YOLO-based distance (no radar); "
             "cameraplusradar = radar + camera (default)",
    )
    parser.add_argument("--model", default=None)
    parser.add_argument("--scaler", default=None)
    parser.add_argument("--config", default=None)
    parser.add_argument("--duration", type=int, default=180)
    parser.add_argument("--vehicles", type=int, default=20)
    parser.add_argument("--pedestrians", type=int, default=10)
    args = parser.parse_args()

    # Resolve model paths from --mode if not explicitly given
    model_dir = MODEL_DIRS[args.mode]
    if args.model is None:
        args.model = os.path.join(model_dir, "target_speed_mlp.pt")
    if args.scaler is None:
        args.scaler = os.path.join(model_dir, "scaler.pkl")
    if args.config is None:
        args.config = os.path.join(model_dir, "model_config.json")

    use_vision_only = args.mode == "cameraonly"

    total_frames = args.duration * FPS

    with open(args.config, "r", encoding="utf-8") as fh:
        model_config = json.load(fh)
    with open(args.scaler, "rb") as fh:
        scaler = pickle.load(fh)

    feature_cols = model_config["feature_cols"]
    history_frames = int(model_config.get("history_frames") or 10)
    base_feature_cols = model_config.get("base_feature_cols") or DEFAULT_BASE_FEATURE_COLS
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TargetSpeedMLP(input_dim=len(feature_cols)).to(device)
    model.load_state_dict(torch.load(args.model, map_location=device, weights_only=True))
    model.eval()

    print("=" * 76)
    print("TARGET-SPEED LIVE TEST")
    print("=" * 76)
    print(f"  Mode:           {args.mode.upper()}")
    print(f"  Distance src:   {'YOLO vision' if use_vision_only else 'Front radar'}")
    print(f"  Model:          {args.model}")
    print(f"  Scaler:         {args.scaler}")
    print(f"  Config:         {args.config}")
    print(f"  Town:           {args.town}")
    print(f"  Duration:       {args.duration}s")
    print(f"  History frames: {history_frames}")
    print(f"  Feature count:  {len(feature_cols)}")
    print("=" * 76)

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

    world.tick()
    print(
        f"  Ego spawned at ({ego.get_location().x:.1f}, {ego.get_location().y:.1f})"
    )

    route_agent = BasicAgent(ego, target_speed=30)
    initial_destination = choose_new_destination(ego, spawn_points)
    route_agent.set_destination(initial_destination)
    print(
        "  BasicAgent route follower initialized "
        f"-> ({initial_destination.x:.1f}, {initial_destination.y:.1f})"
    )
    speed_controller = PIDSpeedController(dt=1.0 / FPS)
    hybrid_controller = HybridStateMachineController(speed_controller)

    collision = CollisionRecorder(ego, world)
    camera = CameraManager(ego, world)
    yolo = YOLOPerception() if YOLO_AVAILABLE else None
    if use_vision_only and yolo is None:
        raise RuntimeError(
            "Camera-only live testing requires YOLO. Install ultralytics on the remote machine first."
        )
    if yolo is None:
        print("  YOLO unavailable; traffic-light features will be zeroed")

    # Distance source: radar or vision
    radar = None
    vision_tracker = None
    if use_vision_only:
        vision_tracker = VisionDistanceTracker(fps=FPS, max_range=MAX_RADAR_RANGE)
        print("  Distance source: YOLO vision (camera-only mode)")
    else:
        radar = FrontRadar(ego, world, MAX_RADAR_RANGE)
        print("  Distance source: Front radar")

    npc_ids = spawn_background_traffic(world, client, tm, args.vehicles)
    walker_ids, ctrl_ids = spawn_background_pedestrians(world, args.pedestrians)
    current_weather_name = apply_random_fog(world)
    print(f"  Weather:        {current_weather_name}")

    for _ in range(40):
        world.tick()

    feature_history = deque(maxlen=history_frames)
    prev_speed = 0.0
    min_distance_seen = MAX_RADAR_RANGE
    max_target_seen = 0.0
    total_brake_frames = 0
    total_throttle_frames = 0
    near_miss_count = 0

    scenario_state = init_scenario_state()
    next_auto_scenario_time = time.time() + 25.0
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
    print("  Press ENTER to spawn a random test scenario ahead of the ego vehicle")

    try:
        for frame in range(total_frames):
            world.tick()

            velocity = ego.get_velocity()
            speed = math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)
            accel = (speed - prev_speed) * FPS if frame > 0 else 0.0
            prev_speed = speed

            # Get distance features from radar or vision depending on mode
            if use_vision_only:
                vision_tracker.update_ego_speed(speed)
                dist_state = None
            else:
                radar.update_ego_speed(speed)
                dist_state = radar.get()

            cam_frame = camera.get_frame()
            visual = empty_visual_features()
            obstacle = empty_obstacle_features()
            if yolo is not None and cam_frame is not None:
                scene_features = yolo.extract_scene_features(cam_frame)
                visual = scene_features["visual"]
                obstacle = scene_features["obstacle"]

            if use_vision_only:
                dist_state = vision_tracker.update(obstacle)

            if dist_state["relative_velocity"] > 0.1:
                ttc = min(dist_state["distance"] / dist_state["relative_velocity"], 10.0)
            else:
                ttc = 10.0
            min_distance_seen = min(min_distance_seen, dist_state["distance"])

            current_features = {
                "ego_speed": round(speed, 4),
                "ego_acceleration": round(max(-20.0, min(20.0, accel)), 4),
                "distance": round(dist_state["distance"], 4),
                "relative_velocity": round(dist_state["relative_velocity"], 4),
                "ttc": round(ttc, 4),
                "obstacle_speed": round(dist_state["obstacle_speed"], 4),
                "traffic_light_state": float(visual["traffic_light_state"]),
                "tl_confidence": round(visual["tl_confidence"], 4),
                "tl_bbox_area": round(visual["tl_bbox_area"], 6),
                "tl_center_x": round(visual["tl_center_x"], 4),
            }
            feature_history.append(current_features)

            if len(feature_history) == history_frames:
                row = flatten_history(feature_history, base_feature_cols)
                feature_vec = np.array([[row[name] for name in feature_cols]], dtype=np.float32)
                scaled = scaler.transform(feature_vec)
                with torch.no_grad():
                    pred = model(torch.tensor(scaled, device=device))
                    target_speed_pred = float(pred.item())
                target_speed_pred = max(0.0, target_speed_pred)
            else:
                target_speed_pred = BOOTSTRAP_TARGET_SPEED_MPS

            if (
                use_vision_only
                and frame < LAUNCH_ASSIST_FRAMES
                and speed < LAUNCH_HOLD_SPEED_MPS
                and dist_state["distance"] > LAUNCH_CLEAR_DISTANCE_M
            ):
                target_speed_pred = max(target_speed_pred, BOOTSTRAP_TARGET_SPEED_MPS)

            max_target_seen = max(max_target_seen, target_speed_pred)

            throttle, brake = hybrid_controller.run_step(target_speed_pred, speed)
            if brake > 0.05:
                total_brake_frames += 1
            if throttle > 0.05:
                total_throttle_frames += 1
            if dist_state["distance"] < 5.0 and speed > 0.5:
                near_miss_count += 1

            if route_agent.done():
                destination = choose_new_destination(ego, spawn_points)
                route_agent.set_destination(destination)
                print(
                    "  New destination -> "
                    f"({destination.x:.1f}, {destination.y:.1f})"
                )

            agent_control = route_agent.run_step()
            steer = max(-0.7, min(0.7, agent_control.steer))

            ego.apply_control(
                carla.VehicleControl(
                    throttle=throttle,
                    steer=steer,
                    brake=brake,
                )
            )

            if (
                scenario_state["name"] is None
                and speed > 5.0
                and (spawn_requested.is_set() or time.time() >= next_auto_scenario_time)
            ):
                spawn_requested.clear()
                if maybe_spawn_scenario(world, ego, carla_map, scenario_state):
                    current_weather_name = apply_random_fog(world)
                    print(f"  Fog preset -> {current_weather_name}")
                    next_auto_scenario_time = time.time() + 25.0
            elif spawn_requested.is_set():
                spawn_requested.clear()

            if scenario_state["name"] is not None:
                update_scenario_state(scenario_state)
                elapsed = time.time() - scenario_state["spawn_time"]
                scenario_vehicle = scenario_state["vehicle"]
                scenario_walker = scenario_state["walker"]

                if (
                    elapsed > 12.0
                    or (scenario_vehicle and ego.get_location().distance(scenario_vehicle.get_location()) > 70.0)
                    or (scenario_walker and ego.get_location().distance(scenario_walker.get_location()) > 40.0)
                ):
                    destroy_scenario_state(scenario_state)
                    scenario_state = init_scenario_state()

            if frame % FPS == 0:
                tl_name = TL_STATE_NAMES.get(int(visual["traffic_light_state"]), "none")
                scenario_name = scenario_state["name"] or "none"
                print(
                    f"  {frame:>6,} "
                    f"spd={speed * 3.6:5.1f}km/h "
                    f"target={target_speed_pred * 3.6:5.1f}km/h "
                    f"dist={dist_state['distance']:5.1f}m "
                    f"ttc={ttc:4.1f}s "
                    f"thr={throttle:4.2f} "
                    f"brk={brake:4.2f} "
                    f"fog={current_weather_name:>11s} "
                    f"tl={tl_name:>6s} "
                    f"area={visual['tl_bbox_area']:.4f} "
                    f"scene={scenario_name} "
                    f"state={hybrid_controller.state}"
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
        print("\n  Live test interrupted")

    total = max(1, frame + 1)
    print("\n" + "=" * 76)
    print("TEST RESULTS")
    print("=" * 76)
    print(f"  Duration:          {total / FPS:.0f}s ({total:,} frames)")
    print(f"  Collisions:        {len(collision.collisions)}")
    print(f"  Near misses <5m:   {near_miss_count}")
    print(f"  Min distance:      {min_distance_seen:.2f}m")
    print(f"  Max target speed:  {max_target_seen * 3.6:.1f}km/h")
    print(f"  Brake frames:      {total_brake_frames} ({100 * total_brake_frames / total:.1f}%)")
    print(f"  Throttle frames:   {total_throttle_frames} ({100 * total_throttle_frames / total:.1f}%)")
    if collision.collisions:
        for event in collision.collisions:
            print(f"  - Hit {event['actor']} ({event['impulse']:.0f} N*s)")
    print("=" * 76)

    destroy_scenario_state(scenario_state)
    if radar is not None:
        radar.cleanup()
    collision.cleanup()
    camera.cleanup()

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
