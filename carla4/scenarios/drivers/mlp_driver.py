#!/usr/bin/env python3
"""
MLP driver — custom vision-only target-speed model for longitudinal control.

Ports the "as-deployed" longitudinal pipeline from
carla4/test_throttle_brake_live.py:
  YOLO perception -> VisionDistanceTracker -> stacked feature history ->
  TargetSpeedMLP -> target speed -> PID (+ hold-state machine) -> throttle/brake.

Steering is delegated to BasicAgent (lateral-only). Runs in the `carla4` conda
env (needs ultralytics + scikit-learn).

Lazy-imported by drivers/__init__.py so the PCLA env never has to import it.
"""

from collections import deque
import json
import math
import os
import pickle  # trusted: loads our own scaler.pkl produced by train_throttle_brake.py
import sys

import carla
import numpy as np
import torch

from .base import Driver
from .steering import BasicAgentSteering

# carla4/ (two levels up) holds the shared perception + model modules
_CARLA4_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _CARLA4_DIR not in sys.path:
    sys.path.insert(0, _CARLA4_DIR)

from yolo_perception import (  # noqa: E402
    CameraManager,
    TL_RED,
    VisionDistanceTracker,
    YOLO_AVAILABLE,
    YOLOPerception,
    empty_obstacle_features,
    empty_visual_features,
)
from speed_model import BASE_FEATURE_COLS as DEFAULT_BASE_FEATURE_COLS  # noqa: E402
from speed_model import TargetSpeedMLP, flatten_history  # noqa: E402
from radar import create_front_radar, normalize_radar_backend  # noqa: E402


# Same constants as the live deployment (test_throttle_brake_live.py)
FPS = 20
MAX_RANGE = 50.0          # vision (YOLO monocular) max range
RADAR_RANGE = 100.0       # radar max range (matches collect_throttle_brake_data.py)
BOOTSTRAP_TARGET_SPEED_MPS = 12.0 / 3.6
CRUISE_SPEED_MPS = 30.0 / 3.6
LAUNCH_HOLD_SPEED_MPS = 0.5
LAUNCH_CLEAR_DISTANCE_M = 15.0
LAUNCH_ASSIST_FRAMES = FPS * 5


class PIDSpeedController:
    """Convert desired speed into throttle/brake (same gains as live test)."""

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
        return 0.0, -accel_cmd


class HybridStateMachineController:
    """PID for dynamics + a brake-hold state at very low speed (live test)."""

    def __init__(self, pid_controller, hold_speed_threshold=0.3,
                 stop_current_speed_threshold=0.8, hold_brake_force=0.35):
        self.pid = pid_controller
        self.hold_speed_threshold = hold_speed_threshold
        self.stop_current_speed_threshold = stop_current_speed_threshold
        self.hold_brake_force = hold_brake_force
        self.state = "CONTINUOUS"

    def run_step(self, target_speed, current_speed):
        throttle, brake = self.pid.run_step(target_speed, current_speed)
        if (target_speed < self.hold_speed_threshold
                and current_speed < self.stop_current_speed_threshold):
            self.state = "HOLD"
            return 0.0, max(brake, self.hold_brake_force)
        self.state = "CONTINUOUS"
        return throttle, brake


class MLPDriver(Driver):
    name = "mlp"

    def __init__(
        self,
        model_dir,
        fps=FPS,
        debug_every=20,
        radar_backend=None,
        **_ignored,
    ):
        self.model_dir = model_dir
        self.fps = fps
        self.debug_every = debug_every
        self.radar_backend = normalize_radar_backend(radar_backend)
        self.device = None
        self.model = None
        self.scaler = None
        self.feature_cols = None
        self.base_feature_cols = None
        self.history_frames = None
        self.camera = None
        self.yolo = None
        self.vision_tracker = None
        self.radar = None
        self.use_radar = False
        self.max_range = MAX_RANGE
        self.steering = None
        self.controller = None
        self.feature_history = None
        self._frame = 0
        self._prev_speed = 0.0

    def setup(self, world, ego, carla_map, client):
        model_path = os.path.join(self.model_dir, "target_speed_mlp.pt")
        scaler_path = os.path.join(self.model_dir, "scaler.pkl")
        config_path = os.path.join(self.model_dir, "model_config.json")

        with open(config_path, "r", encoding="utf-8") as fh:
            model_config = json.load(fh)
        with open(scaler_path, "rb") as fh:
            self.scaler = pickle.load(fh)

        self.feature_cols = model_config["feature_cols"]
        self.history_frames = int(model_config.get("history_frames") or 10)
        self.base_feature_cols = (
            model_config.get("base_feature_cols") or DEFAULT_BASE_FEATURE_COLS
        )
        trained_radar_backend = model_config.get("radar_backend", "native")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = TargetSpeedMLP(input_dim=len(self.feature_cols)).to(self.device)
        self.model.load_state_dict(
            torch.load(model_path, map_location=self.device, weights_only=True)
        )
        self.model.eval()

        # Distance source is chosen from the model's feature schema: the radar
        # model (model_throttle_brake) has 10 base cols; the vision model adds
        # 'obstacle_detected'. Radar models need radar at inference, not YOLO depth.
        self.use_radar = "obstacle_detected" not in self.base_feature_cols
        self.max_range = RADAR_RANGE if self.use_radar else MAX_RANGE
        if (
            self.use_radar
            and self.radar_backend != trained_radar_backend
        ):
            print(
                "  [mlp] WARNING: model data used radar backend "
                f"'{trained_radar_backend}', runtime uses '{self.radar_backend}'. "
                "Recollect and retrain for final results."
            )

        # Camera + YOLO give traffic-light features in both modes (and obstacle
        # detection in vision mode). YOLO is required for vision, optional for radar.
        self.camera = CameraManager(ego, world)
        self.yolo = YOLOPerception() if YOLO_AVAILABLE else None
        if self.use_radar:
            self.radar = create_front_radar(
                ego,
                world,
                self.max_range,
                backend=self.radar_backend,
                fps=self.fps,
            )
            self.vision_tracker = None
            if self.yolo is None:
                print("  [mlp] YOLO unavailable; traffic-light features zeroed")
        else:
            if self.yolo is None:
                raise RuntimeError(
                    "Vision MLP driver requires YOLO (ultralytics) in the carla4 env."
                )
            self.radar = None
            self.vision_tracker = VisionDistanceTracker(fps=self.fps, max_range=self.max_range)

        self.steering = BasicAgentSteering(ego, carla_map)
        self.controller = HybridStateMachineController(
            PIDSpeedController(dt=1.0 / self.fps))
        self.feature_history = deque(maxlen=self.history_frames)
        self._frame = 0
        self._prev_speed = 0.0

        print("  [mlp] driver ready")
        print(f"  [mlp]   model:          {model_path}")
        sensor_name = (
            f"radar ({self.radar_backend})"
            if self.use_radar
            else "vision (YOLO)"
        )
        print(f"  [mlp]   sensor:         {sensor_name} "
              f"(max {self.max_range:.0f}m)")
        print(f"  [mlp]   device:         {self.device}")
        print(f"  [mlp]   history_frames: {self.history_frames}")
        print(f"  [mlp]   feature_count:  {len(self.feature_cols)} "
              f"(base={len(self.base_feature_cols)})")

    def get_control(self, ego, world):
        velocity = ego.get_velocity()
        speed = math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)
        accel = (speed - self._prev_speed) * self.fps if self._frame > 0 else 0.0
        self._prev_speed = speed

        cam_frame = self.camera.get_frame()
        visual = empty_visual_features()
        obstacle = empty_obstacle_features()
        if self.yolo is not None and cam_frame is not None:
            scene_features = self.yolo.extract_scene_features(cam_frame)
            visual = scene_features["visual"]
            obstacle = scene_features["obstacle"]

        if self.use_radar:
            self.radar.update_ego_speed(speed)
            dist_state = self.radar.get()
        else:
            self.vision_tracker.update_ego_speed(speed)
            dist_state = self.vision_tracker.update(obstacle)

        if dist_state["relative_velocity"] > 0.1:
            ttc = min(dist_state["distance"] / dist_state["relative_velocity"], 10.0)
        else:
            ttc = 10.0

        obstacle_detected = float(dist_state["distance"] < self.max_range * 0.95)
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
        self.feature_history.append(current_features)

        if len(self.feature_history) == self.history_frames:
            row = flatten_history(self.feature_history, self.base_feature_cols)
            feature_vec = np.array(
                [[row[name] for name in self.feature_cols]], dtype=np.float32)
            scaled = self.scaler.transform(feature_vec)
            with torch.no_grad():
                pred = self.model(torch.tensor(scaled, device=self.device))
                target_speed_pred = max(0.0, float(pred.item()))
        else:
            target_speed_pred = BOOTSTRAP_TARGET_SPEED_MPS

        # Hybrid override (same as deployment): cruise on open road, but never
        # override a red light.
        if obstacle_detected < 0.5 and int(visual["traffic_light_state"]) != TL_RED:
            target_speed_pred = max(target_speed_pred, CRUISE_SPEED_MPS)

        # Launch assist: help the car pull away from a standstill on clear road.
        if (self._frame < LAUNCH_ASSIST_FRAMES
                and speed < LAUNCH_HOLD_SPEED_MPS
                and dist_state["distance"] > LAUNCH_CLEAR_DISTANCE_M):
            target_speed_pred = max(target_speed_pred, BOOTSTRAP_TARGET_SPEED_MPS)

        # ── Hardcoded safety overrides ──────────────────────────────────
        # Rule 1: Model predicts < 1 km/h → full emergency brake.
        if target_speed_pred < (1.0 / 3.6):
            throttle, brake = 0.0, 1.0

        # Rule 2: Obstacle within 30m AND actively closing → full brake.
        #         (requires relv > 0.5 so it doesn't fire during stable
        #          following — fixes over-conservative S2 behavior)
        elif (obstacle_detected > 0.5
              and dist_state["distance"] < 30.0
              and dist_state["relative_velocity"] > 0.5):
            throttle, brake = 0.0, 1.0

        # Rule 3: TTC < 3.0s → full emergency brake.
        #         Catches fast-closing scenarios like cut-ins where distance
        #         alone doesn't trigger in time.
        elif obstacle_detected > 0.5 and ttc < 3.0:
            throttle, brake = 0.0, 1.0

        else:
            throttle, brake = self.controller.run_step(target_speed_pred, speed)
        # ────────────────────────────────────────────────────────────────
        steer = self.steering.get_steer()

        if self.debug_every and self._frame % self.debug_every == 0:
            print(f"  [mlp] f={self._frame:4d} spd={speed * 3.6:5.1f}km/h "
                  f"tgt={target_speed_pred * 3.6:5.1f}km/h "
                  f"dist={dist_state['distance']:5.1f}m "
                  f"relv={dist_state['relative_velocity']:+5.1f} "
                  f"det={int(obstacle_detected)} thr/brk={throttle:.2f}/{brake:.2f} "
                  f"steer={steer:+.2f} [{self.controller.state}]")

        self._frame += 1
        return carla.VehicleControl(throttle=throttle, steer=steer, brake=brake)

    def cleanup(self):
        if self.camera is not None:
            self.camera.cleanup()
            self.camera = None
        if self.radar is not None:
            self.radar.cleanup()
            self.radar = None
