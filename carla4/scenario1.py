#!/usr/bin/env python3
"""
Scenario 1: Heavy Fog Obstacle Test
====================================

Tests the camera-only (vision) model in extreme fog.
- Spawns ego at a random road location
- Sets heaviest possible fog (YOLO can barely see anything)
- Drives at a constant ~30 km/h using PID
- Spawns a stopped vehicle ~40m ahead on the same lane
- Measures: does the model brake in time, or does it collide?

Run:
    python3 scenario1.py

Then (after training radar model):
    python3 scenario1.py --mode radar

To compare both models on the same test.
"""

import argparse
from collections import deque
import json
import math
import os
import pickle
import random
import sys
import time

import carla
import numpy as np
import torch

from yolo_perception import (
    CameraManager,
    TL_RED,
    TL_STATE_NAMES,
    VisionDistanceTracker,
    YOLO_AVAILABLE,
    YOLOPerception,
    empty_obstacle_features,
    empty_visual_features,
)
from speed_model import BASE_FEATURE_COLS as DEFAULT_BASE_FEATURE_COLS
from speed_model import TargetSpeedMLP, flatten_history

CARLA_ROOT = os.environ.get("CARLA_ROOT", "/opt/carla-simulator")
AGENTS_PATH = os.path.join(CARLA_ROOT, "PythonAPI", "carla")
if AGENTS_PATH not in sys.path:
    sys.path.insert(0, AGENTS_PATH)

try:
    from agents.navigation.basic_agent import BasicAgent
except ImportError:
    print("ERROR: CARLA agents not found.")
    sys.exit(1)

# BasicAgent is imported but unused in scenario1 (we use follow_lane instead).
# Keep the import in case we want to switch back to routing-based steering.

# ============================================================================
# Config
# ============================================================================
CARLA_HOST = "127.0.0.1"
CARLA_PORT = 2000
DEFAULT_TOWN = "Town01"
FPS = 20
MAX_RANGE = 50.0
CRUISE_SPEED_MPS = 30.0 / 3.6  # ~30 km/h
OBSTACLE_SPAWN_DISTANCE = 45.0  # metres ahead to spawn the stopped car
TEST_DURATION_S = 25  # how long to run after obstacle spawn

# Paths for the two models
VISION_MODEL_DIR = "model_vision_only"
RADAR_MODEL_DIR = "model_throttle_brake"

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("WARNING: OpenCV not available; camera overlay disabled")


# ============================================================================
# Steering
# ============================================================================
def follow_lane(ego, carla_map):
    """Simple lane-following steer."""
    wp = carla_map.get_waypoint(ego.get_location(), project_to_road=True)
    if wp is None:
        return 0.0
    target_wp = wp
    remaining = 10.0
    while remaining > 0:
        step = min(2.0, remaining)
        nexts = target_wp.next(step)
        if not nexts:
            break
        target_wp = nexts[0]
        remaining -= step
    ego_tf = ego.get_transform()
    target_yaw = math.radians(target_wp.transform.rotation.yaw)
    ego_yaw = math.radians(ego_tf.rotation.yaw)
    err = (target_yaw - ego_yaw + math.pi) % (2 * math.pi) - math.pi
    return max(-0.7, min(0.7, err / math.radians(60)))


# ============================================================================
# PIDSpeedController (same as live test)
# ============================================================================
class PIDSpeedController:
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
            return accel_cmd, 0.0
        else:
            return 0.0, -accel_cmd


# ============================================================================
# FrontRadar (for radar model mode)
# ============================================================================
class FrontRadar:
    def __init__(self, vehicle, world, range_m=50.0):
        self.latest = {"distance": range_m, "relative_velocity": 0.0, "obstacle_speed": 0.0}
        self._ego_speed = 0.0
        self._range = range_m
        bp = world.get_blueprint_library().find("sensor.other.radar")
        bp.set_attribute("horizontal_fov", "10")
        bp.set_attribute("vertical_fov", "2")
        bp.set_attribute("range", str(range_m))
        bp.set_attribute("points_per_second", "1500")
        transform = carla.Transform(carla.Location(x=2.5, z=1.0), carla.Rotation(pitch=2.0))
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
        self.latest = {"distance": nearest_dist, "relative_velocity": rel_vel,
                       "obstacle_speed": max(0.0, self._ego_speed - rel_vel)}

    def update_ego_speed(self, speed):
        self._ego_speed = speed

    def get(self):
        return self.latest.copy()

    def cleanup(self):
        if self.sensor and self.sensor.is_alive:
            self.sensor.destroy()


# ============================================================================
# Collision Recorder
# ============================================================================
class CollisionRecorder:
    def __init__(self, vehicle, world):
        self.collided = False
        self.collision_time = None
        self.collision_impulse = 0.0
        self.collision_actor = ""
        bp = world.get_blueprint_library().find("sensor.other.collision")
        self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=vehicle)
        self.sensor.listen(self._on_collision)

    def _on_collision(self, event):
        if not self.collided:
            self.collided = True
            self.collision_time = time.time()
            self.collision_impulse = event.normal_impulse.length()
            self.collision_actor = event.other_actor.type_id

    def cleanup(self):
        if self.sensor and self.sensor.is_alive:
            self.sensor.destroy()


def set_fog(world):
    """Heavy fog that challenges YOLO but still lets the user see."""
    weather = carla.WeatherParameters(
        cloudiness=80.0,
        precipitation=0.0,
        precipitation_deposits=0.0,
        wind_intensity=0.0,
        sun_azimuth_angle=0.0,
        sun_altitude_angle=15.0,
        fog_density=80.0,        # very thick but not max
        fog_distance=20.0,       # can see ~20m ahead
        fog_falloff=1.5,
        wetness=0.0,
    )
    world.set_weather(weather)


def waypoint_ahead(carla_map, location, distance_m):
    """Get the waypoint N metres ahead on the same lane."""
    wp = carla_map.get_waypoint(location, project_to_road=True)
    if wp is None:
        return None
    travelled = 0.0
    step = 3.0
    while travelled < distance_m:
        nexts = wp.next(step)
        if not nexts:
            return None
        wp = nexts[0]
        travelled += step
    return wp


def spawn_stopped_obstacle(world, carla_map, ego_location, ahead_m):
    """Spawn a stopped vehicle ahead of ego."""
    wp = waypoint_ahead(carla_map, ego_location, ahead_m)
    if wp is None:
        return None
    bp_lib = world.get_blueprint_library()
    bp = random.choice([b for b in bp_lib.filter("vehicle.*") if int(b.get_attribute("number_of_wheels")) >= 4])
    transform = wp.transform
    transform.location.z += 0.5
    vehicle = world.try_spawn_actor(bp, transform)
    if vehicle:
        vehicle.apply_control(carla.VehicleControl(brake=1.0, throttle=0.0))
        vehicle.set_target_velocity(carla.Vector3D(0.0, 0.0, 0.0))
    return vehicle


def draw_camera_overlay(frame, visual, obstacle, speed, dist_state, ttc):
    """Show YOLO detections as overlay on the camera feed."""
    if not CV2_AVAILABLE or frame is None:
        return

    display = frame.copy()

    # Traffic light boxes
    bbox = visual["tl_bbox"]
    tl_state = visual["traffic_light_state"]
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        color_map = {0: (180, 180, 180), 1: (0, 255, 0), 2: (0, 255, 255), 3: (0, 0, 255)}
        box_color = color_map.get(tl_state, (180, 180, 180))
        cv2.rectangle(display, (x1, y1), (x2, y2), box_color, 2)
        label = f"{TL_STATE_NAMES.get(tl_state, 'none')} conf={visual['tl_confidence']:.2f}"
        cv2.putText(display, label, (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 1)

    # Obstacle boxes
    for entry in obstacle.get("obstacle_boxes", []):
        x1, y1, x2, y2 = entry["bbox"]
        box_color = (0, 140, 255) if entry["is_primary"] else (255, 180, 0)
        thickness = 2 if entry["is_primary"] else 1
        cv2.rectangle(display, (x1, y1), (x2, y2), box_color, thickness)
        obj_label = f"{entry['label']} {entry['distance']:.1f}m {entry['confidence']:.2f}"
        cv2.putText(display, obj_label, (x1, min(display.shape[0] - 10, y2 + 16)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, box_color, 1)

    # HUD
    sensor_label = f"dist={dist_state['distance']:.1f}m rel_vel={dist_state['relative_velocity']:.1f}m/s"
    cv2.putText(display, f"Speed: {speed*3.6:.1f} km/h", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
    cv2.putText(display, sensor_label, (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)
    cv2.putText(display, f"TTC: {ttc:.1f}s | YOLO dets: {len(obstacle.get('obstacle_boxes', []))}",
                (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)

    cv2.imshow("Scenario 1 - Camera View", display)
    cv2.waitKey(1)


def cleanup_actor(actor):
    if actor and actor.is_alive:
        try:
            actor.destroy()
        except RuntimeError:
            pass


def load_vision_model(model_dir):
    """Load the vision-only TargetSpeedMLP and its scaler/config."""
    model_path = os.path.join(model_dir, "target_speed_mlp.pt")
    scaler_path = os.path.join(model_dir, "scaler.pkl")
    config_path = os.path.join(model_dir, "model_config.json")

    with open(config_path, "r", encoding="utf-8") as fh:
        model_config = json.load(fh)
    with open(scaler_path, "rb") as fh:
        scaler = pickle.load(fh)

    feature_cols = model_config["feature_cols"]
    history_frames = int(model_config.get("history_frames") or 10)
    base_feature_cols = model_config.get("base_feature_cols") or DEFAULT_BASE_FEATURE_COLS
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TargetSpeedMLP(input_dim=len(feature_cols)).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    return model, scaler, feature_cols, history_frames, base_feature_cols, device


def main():
    parser = argparse.ArgumentParser(description="Scenario 1: Heavy Fog Obstacle Test")
    parser.add_argument("--host", default=CARLA_HOST)
    parser.add_argument("--port", type=int, default=CARLA_PORT)
    parser.add_argument("--town", default=DEFAULT_TOWN)
    parser.add_argument(
        "--mode", choices=["vision", "radar"], default="vision",
        help="Which model to test: vision (camera-only) or radar (camera+radar)"
    )
    args = parser.parse_args()

    use_vision = args.mode == "vision"
    model_dir = VISION_MODEL_DIR if use_vision else RADAR_MODEL_DIR

    print("=" * 76)
    print(f"  SCENARIO 1: HEAVY FOG OBSTACLE TEST  [{args.mode.upper()}]")
    print("=" * 76)
    print(f"  Mode:       {'Camera-only (YOLO)' if use_vision else 'Camera + Radar'}")
    print(f"  Model dir:  {model_dir}")
    print(f"  Fog:        Heavy (density=80, distance=20m)")
    print(f"  Speed:      {CRUISE_SPEED_MPS * 3.6:.0f} km/h (constant PID target)")
    print(f"  Obstacle:   Stopped car {OBSTACLE_SPAWN_DISTANCE:.0f}m ahead")
    print("=" * 76)

    # Connect to CARLA
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

    carla_map = world.get_map()
    spawn_points = carla_map.get_spawn_points()

    # Find a spawn point on a straight road (heading change < 5° over 50m)
    # so the ego drives directly toward the obstacle without curves.
    def is_straight_road(sp):
        wp = carla_map.get_waypoint(sp.location, project_to_road=True)
        if wp is None or wp.is_junction:
            return False
        ahead = wp.next(50.0)
        if not ahead:
            return False
        start_yaw = wp.transform.rotation.yaw
        end_yaw = ahead[0].transform.rotation.yaw
        diff = abs((end_yaw - start_yaw + 180) % 360 - 180)
        return diff < 5.0

    straight_spawns = [sp for sp in spawn_points if is_straight_road(sp)]
    if not straight_spawns:
        print("  No straight road found, falling back to any safe spawn")
        straight_spawns = [sp for sp in spawn_points
                          if carla_map.get_waypoint(sp.location, project_to_road=True) and
                          not carla_map.get_waypoint(sp.location, project_to_road=True).is_junction]
    random.shuffle(straight_spawns)

    # Set heavy fog BEFORE spawning
    set_fog(world)
    print("  Heavy fog set: density=80, distance=20m")

    # Spawn ego on straight road
    ego_bp = world.get_blueprint_library().find("vehicle.tesla.model3")
    ego = None
    for sp in straight_spawns:
        ego = world.try_spawn_actor(ego_bp, sp)
        if ego:
            break
    if ego is None:
        raise RuntimeError("Failed to spawn ego")

    world.tick()
    print(f"  Ego at ({ego.get_location().x:.1f}, {ego.get_location().y:.1f})")

    # Load model
    model, scaler, feature_cols, history_frames, base_feature_cols, device = load_vision_model(model_dir)

    # Sensors
    collision = CollisionRecorder(ego, world)
    camera = CameraManager(ego, world)

    # Distance source
    radar = None
    vision_tracker = None
    if use_vision:
        vision_tracker = VisionDistanceTracker(fps=FPS, max_range=MAX_RANGE)
    else:
        radar = FrontRadar(ego, world, MAX_RANGE)

    yolo = YOLOPerception() if YOLO_AVAILABLE else None
    if use_vision and yolo is None:
        raise RuntimeError("Vision mode requires YOLO.")

    speed_controller = PIDSpeedController(dt=1.0 / FPS)

    feature_history = deque(maxlen=history_frames)
    prev_speed = 0.0
    min_dist = MAX_RANGE
    obstacle = None
    obstacle_alive = False
    obstacle_spawned = False
    stopped_after_obstacle = False
    obstacle_distance_initial = 0.0

    # Warmup ticks
    for _ in range(20):
        world.tick()

    spawn_time = time.time()  # spawn obstacle after 3s of driving

    print("\n  --- Starting drive ---\n")
    print(f"  {'Frame':>6} │ {'Speed':>8} │ {'Dist':>6} │ {'Target':>7} │ {'Thr/Brk':>9} │ Status")
    print(f"  {'─'*6}─┼─{'─'*8}─┼─{'─'*6}─┼─{'─'*7}─┼─{'─'*9}─┼─{'─'*24}")

    try:
        for frame in range(FPS * TEST_DURATION_S):
            world.tick()

            velocity = ego.get_velocity()
            speed = math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)
            accel = (speed - prev_speed) * FPS if frame > 0 else 0.0
            prev_speed = speed

            # Perception
            cam_frame = camera.get_frame()
            visual = empty_visual_features()
            obstacle_feat = empty_obstacle_features()
            if yolo is not None and cam_frame is not None:
                scene_features = yolo.extract_scene_features(cam_frame)
                visual = scene_features["visual"]
                obstacle_feat = scene_features["obstacle"]

            if use_vision:
                vision_tracker.update_ego_speed(speed)
                dist_state = vision_tracker.update(obstacle_feat)
            else:
                radar.update_ego_speed(speed)
                dist_state = radar.get()

            # TTC
            if dist_state["relative_velocity"] > 0.1:
                ttc = min(dist_state["distance"] / dist_state["relative_velocity"], 10.0)
            else:
                ttc = 10.0

            min_dist = min(min_dist, dist_state["distance"])

            # Model features (same as live test)
            obstacle_detected = float(dist_state["distance"] < MAX_RANGE * 0.95)
            current_features = {
                "ego_speed": round(speed, 4),
                "ego_acceleration": round(max(-20.0, min(20.0, accel)), 4),
                "distance": round(dist_state["distance"], 4),
                "relative_velocity": round(dist_state["relative_velocity"], 4),
                "ttc": round(ttc, 4),
                "obstacle_speed": round(dist_state["obstacle_speed"], 4),
                "obstacle_detected": obstacle_detected,
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
                target_speed_pred = CRUISE_SPEED_MPS

            # Apply the same hybrid override as the live test
            if obstacle_detected < 0.5 and int(visual["traffic_light_state"]) != TL_RED:
                target_speed_pred = max(target_speed_pred, CRUISE_SPEED_MPS)

            # PID control
            throttle, brake = speed_controller.run_step(target_speed_pred, speed)

            # Steering (lane follow — keeps ego on same road toward obstacle)
            steer = follow_lane(ego, carla_map)
            ego.apply_control(carla.VehicleControl(throttle=throttle, steer=steer, brake=brake))

            # Spawn obstacle after 3 seconds
            if not obstacle_spawned and time.time() - spawn_time > 3.0:
                obstacle = spawn_stopped_obstacle(world, carla_map, ego.get_location(), OBSTACLE_SPAWN_DISTANCE)
                if obstacle:
                    obstacle_alive = True
                    obstacle_spawned = True
                    obstacle_distance_initial = ego.get_location().distance(obstacle.get_location())
                    print(f"\n  🚧 OBSTACLE SPAWNED at {obstacle_distance_initial:.0f}m ahead\n")
                else:
                    print("  ⚠️  Failed to spawn obstacle (no road ahead)")

            # Track if obstacle is alive and distance to it
            if obstacle_alive and obstacle and obstacle.is_alive:
                current_obs_dist = ego.get_location().distance(obstacle.get_location())
                if current_obs_dist < 0.5 or collision.collided:
                    obstacle_alive = False
                    if collision.collided:
                        stopped_after_obstacle = False
            elif obstacle_alive:
                obstacle_alive = False

            # Check if ego stopped near obstacle
            if obstacle_spawned and speed < 0.3 and obstacle_alive:
                stopped_after_obstacle = True

            # Camera overlay (YOLO detections)
            draw_camera_overlay(cam_frame, visual, obstacle_feat, speed, dist_state, ttc)

            # Spectator camera
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

            # Log every 20 frames
            if frame % 20 == 0:
                thr_brk = f"{throttle:.2f}/{brake:.2f}"
                status = "driving"
                if collision.collided:
                    status = "💥 COLLISION"
                elif stopped_after_obstacle:
                    status = "🛑 STOPPED"
                elif obstacle_spawned and obstacle_alive:
                    status = "approaching obst"
                print(f"  {frame:6d} │ {speed*3.6:6.1f}km/h │ {dist_state['distance']:5.1f}m │ "
                      f"{target_speed_pred*3.6:5.1f}km/h │ {thr_brk:>9s} │ {status}")

    except KeyboardInterrupt:
        print("\n  Interrupted")

    # --- Results ---
    print("\n" + "=" * 76)
    print("  RESULTS")
    print("=" * 76)

    if collision.collided:
        print(f"  💥 COLLISION with {collision.collision_actor}")
        print(f"     Impulse: {collision.collision_impulse:.0f} N*s")
    else:
        print(f"  ✅ NO COLLISION")

    if stopped_after_obstacle:
        print(f"  🛑  Ego stopped before hitting the obstacle")
    elif not collision.collided:
        print(f"  ⚠️  Ego did NOT stop — distance was {min_dist:.1f}m at closest")

    print(f"  Min distance to anything:  {min_dist:.2f}m")
    if obstacle_spawned:
        print(f"  Obstacle initial distance: {obstacle_distance_initial:.0f}m")
    print(f"  Test duration:             {frame+1} frames ({(frame+1)/FPS:.0f}s)")

    result = "FAIL" if collision.collided else "PASS"
    if collision.collided:
        print(f"\n  ❌ {args.mode.upper()} MODEL: FAILED — collided in fog")
    else:
        print(f"\n  ✅ {args.mode.upper()} MODEL: PASSED — avoided collision in fog")

    # Cleanup
    cleanup_actor(obstacle)
    if radar is not None:
        radar.cleanup()
    collision.cleanup()
    camera.cleanup()

    cleanup_actor(ego)
    world.apply_settings(original_settings)
    if CV2_AVAILABLE:
        cv2.destroyAllWindows()

    print("=" * 76)


if __name__ == "__main__":
    main()
