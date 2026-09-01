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
from .longitudinal import HybridStateMachineController, PIDSpeedController
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
from radar import (  # noqa: E402
    create_front_radar,
    describe_radar_configuration,
    normalize_radar_backend,
    realistic_radar_config_signature,
    resolve_realistic_radar_config,
)
from driving_contract import (  # noqa: E402
    MAX_TARGET_SPEED_KMH,
    NATIVE_RADAR_POINTS_PER_SECOND,
    RADAR_RANGE_M,
)


# Same constants as the live deployment (test_throttle_brake_live.py)
FPS = 20
MAX_RANGE = 50.0          # vision (YOLO monocular) max range
RADAR_RANGE = RADAR_RANGE_M
BOOTSTRAP_TARGET_SPEED_MPS = 12.0 / 3.6
CRUISE_SPEED_MPS = 30.0 / 3.6
LAUNCH_HOLD_SPEED_MPS = 0.5
LAUNCH_CLEAR_DISTANCE_M = 15.0
LAUNCH_ASSIST_FRAMES = FPS * 5


class MLPDriver(Driver):
    name = "mlp"

    def __init__(
        self,
        model_dir,
        fps=FPS,
        debug_every=20,
        radar_backend=None,
        radar_profile=None,
        radar_config_path=None,
        radar_seed=42,
        radar_ghost_detector=None,
        radar_ghost_threshold=None,
        radar_ghost_device="cpu",
        safety_rules=False,
        cruise_floor=True,
        **_ignored,
    ):
        self.model_dir = model_dir
        # The hardcoded emergency-brake overrides are OFF by default: they
        # fire on <30 m closing or TTC<3 s, which is every hazard in S1/S2/S4,
        # so with them on the rules drive and the model does not. Enable them
        # only as an explicit ablation arm.
        self.safety_rules = bool(safety_rules)
        # The cruise floor only applies when nothing is detected, so it cannot
        # mask a reaction to a ghost; it stops the model under-driving on an
        # empty road. Separately switchable all the same.
        self.cruise_floor = bool(cruise_floor)
        self.fps = fps
        self.debug_every = debug_every
        self.radar_backend = normalize_radar_backend(radar_backend)
        self.radar_profile = radar_profile
        self.radar_config_path = radar_config_path
        self.radar_seed = int(radar_seed)
        self.radar_ghost_detector = radar_ghost_detector
        self.radar_ghost_threshold = radar_ghost_threshold
        self.radar_ghost_device = radar_ghost_device
        self.realistic_radar_config = None
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
        self.max_target_speed_mps = MAX_TARGET_SPEED_KMH / 3.6
        self.radar_points_per_second = NATIVE_RADAR_POINTS_PER_SECOND
        self.steering = None
        self.controller = None
        self.feature_history = None
        self._frame = 0
        self._prev_speed = 0.0
        self._last_distance_state = None
        self._rule_fired = None

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
        # A radar-only model has no traffic-light columns, so there is nothing
        # for the camera or YOLO to feed; attaching them would only add a
        # sensor and a CUDA context that contend with CARLA's renderer.
        self.vision_enabled = bool(
            model_config.get(
                "vision_enabled",
                any(
                    name.startswith("tl_") or name == "traffic_light_state"
                    for name in self.base_feature_cols
                ),
            )
        )
        trained_radar_backend = model_config.get("radar_backend", "native")
        self.radar_points_per_second = int(
            model_config.get(
                "radar_points_per_second",
                (
                    NATIVE_RADAR_POINTS_PER_SECOND
                    if trained_radar_backend == "native"
                    else 240000
                ),
            )
        )
        self.max_target_speed_mps = (
            min(
                float(
                    model_config.get(
                        "max_target_speed_kmh",
                        MAX_TARGET_SPEED_KMH,
                    )
                ),
                MAX_TARGET_SPEED_KMH,
            )
            / 3.6
        )
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
        self.max_range = (
            float(model_config.get("radar_range_m", RADAR_RANGE))
            if self.use_radar
            else MAX_RANGE
        )
        if (
            self.use_radar
            and self.radar_backend != trained_radar_backend
        ):
            raise RuntimeError(
                "Sensor distribution mismatch: model data used radar backend "
                f"'{trained_radar_backend}', runtime requested "
                f"'{self.radar_backend}'."
            )
        if self.use_radar and self.radar_backend == "realistic":
            embedded_config = (
                None
                if self.radar_profile or self.radar_config_path
                else model_config.get("radar_config")
            )
            self.realistic_radar_config = resolve_realistic_radar_config(
                range_m=self.max_range,
                fps=self.fps,
                profile_name=self.radar_profile,
                config_path=self.radar_config_path,
                config=embedded_config,
            )
            requested_signature = realistic_radar_config_signature(
                self.realistic_radar_config
            )
            trained_signature = model_config.get("radar_config_signature")
            if trained_signature != requested_signature:
                raise RuntimeError(
                    "Realistic radar configuration mismatch: model data used "
                    f"{trained_signature!r}, runtime requested "
                    f"{requested_signature!r}."
                )
            runtime_metadata = describe_radar_configuration(
                backend="realistic",
                range_m=self.max_range,
                fps=self.fps,
                points_per_second=self.radar_points_per_second,
                config=self.realistic_radar_config,
                ghost_detector_path=self.radar_ghost_detector,
                ghost_threshold=self.radar_ghost_threshold,
            )
            trained_ghost_signature = model_config.get(
                "radar_ghost_detector_signature"
            )
            requested_ghost_signature = runtime_metadata.get(
                "radar_ghost_detector_signature"
            )
            if trained_ghost_signature != requested_ghost_signature:
                raise RuntimeError(
                    "Ghost-detector mismatch: model data used "
                    f"{trained_ghost_signature!r}, runtime requested "
                    f"{requested_ghost_signature!r}."
                )
            if model_config.get("radar_ghost_threshold") != runtime_metadata.get(
                "radar_ghost_threshold"
            ):
                raise RuntimeError(
                    "Ghost rejection threshold differs from training data."
                )

        # Camera + YOLO give traffic-light features (and obstacle detection in
        # vision mode). A radar-only model has no columns for them, so skip
        # both entirely rather than paying for a sensor and a CUDA context
        # whose output is discarded.
        if self.vision_enabled:
            self.camera = CameraManager(ego, world)
            self.yolo = YOLOPerception() if YOLO_AVAILABLE else None
        else:
            self.camera = None
            self.yolo = None
            print("  [mlp] radar-only model; camera and YOLO not attached")
        if self.use_radar:
            self.radar = create_front_radar(
                ego,
                world,
                self.max_range,
                backend=self.radar_backend,
                fps=self.fps,
                points_per_second=self.radar_points_per_second,
                config=self.realistic_radar_config,
                seed=self.radar_seed,
                ghost_detector_path=self.radar_ghost_detector,
                ghost_threshold=self.radar_ghost_threshold,
                ghost_device=self.radar_ghost_device,
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
        self._last_distance_state = None

        print("  [mlp] driver ready")
        print(f"  [mlp]   model:          {model_path}")
        sensor_name = (
            f"radar ({self.radar_backend})"
            if self.use_radar
            else "vision (YOLO)"
        )
        print(f"  [mlp]   sensor:         {sensor_name} "
              f"(max {self.max_range:.0f}m)")
        if self.use_radar:
            print(
                f"  [mlp]   radar sampling: "
                f"{self.radar_points_per_second} points/s"
            )
            if self.realistic_radar_config is not None:
                print(
                    "  [mlp]   radar profile:  "
                    f"{self.realistic_radar_config.profile_name}"
                )
                print(
                    "  [mlp]   radar config:   "
                    f"{realistic_radar_config_signature(self.realistic_radar_config)}"
                )
        print(f"  [mlp]   device:         {self.device}")
        print(f"  [mlp]   history_frames: {self.history_frames}")
        print(
            f"  [mlp]   speed ceiling:  "
            f"{self.max_target_speed_mps * 3.6:.1f}km/h"
        )
        print(f"  [mlp]   feature_count:  {len(self.feature_cols)} "
              f"(base={len(self.base_feature_cols)})")

    def get_control(self, ego, world):
        velocity = ego.get_velocity()
        speed = math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)
        accel = (speed - self._prev_speed) * self.fps if self._frame > 0 else 0.0
        self._prev_speed = speed

        cam_frame = self.camera.get_frame() if self.camera is not None else None
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
        self._last_distance_state = dist_state.copy()

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
        # override a red light. Only fires with nothing detected, so a ghost
        # still reaches the model unmodified.
        if (self.cruise_floor
                and obstacle_detected < 0.5
                and int(visual["traffic_light_state"]) != TL_RED):
            target_speed_pred = max(target_speed_pred, CRUISE_SPEED_MPS)

        # Launch assist: help the car pull away from a standstill on clear road.
        if (self._frame < LAUNCH_ASSIST_FRAMES
                and speed < LAUNCH_HOLD_SPEED_MPS
                and dist_state["distance"] > LAUNCH_CLEAR_DISTANCE_M):
            target_speed_pred = max(target_speed_pred, BOOTSTRAP_TARGET_SPEED_MPS)
        target_speed_pred = min(
            max(0.0, target_speed_pred),
            self.max_target_speed_mps,
        )

        # ── Optional hardcoded overrides (ablation arm; OFF by default) ──
        # These three rules fire on <30 m closing or TTC<3 s, which describes
        # every hazard in S1/S2/S4. With them enabled the rules brake before
        # the model has any influence, so a measurement of the model -- or of
        # anything upstream of it, such as a ghost filter -- means nothing.
        # The model's target speed drives by default.
        self._rule_fired = None
        if self.safety_rules and target_speed_pred < (1.0 / 3.6):
            self._rule_fired = "model_stop"
            throttle, brake = 0.0, 1.0
        elif (self.safety_rules
              and obstacle_detected > 0.5
              and dist_state["distance"] < 30.0
              and dist_state["relative_velocity"] > 0.5):
            self._rule_fired = "close_and_closing"
            throttle, brake = 0.0, 1.0
        elif (self.safety_rules
              and obstacle_detected > 0.5
              and ttc < 3.0):
            self._rule_fired = "low_ttc"
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

    def diagnostics(self):
        diagnostics = {
            "safety_rules_enabled": int(self.safety_rules),
            "safety_rule_fired": getattr(self, "_rule_fired", None) or "",
        }
        if self.radar is not None:
            diagnostics.update(self.radar.diagnostics())
        if self._last_distance_state is not None:
            diagnostics.update(
                {
                    "controller_distance_m": self._last_distance_state[
                        "distance"
                    ],
                    "controller_relative_velocity_mps": (
                        self._last_distance_state["relative_velocity"]
                    ),
                    "controller_obstacle_speed_mps": self._last_distance_state[
                        "obstacle_speed"
                    ],
                }
            )
        return diagnostics

    def cleanup(self):
        if self.camera is not None:
            self.camera.cleanup()
            self.camera = None
        if self.radar is not None:
            self.radar.cleanup()
            self.radar = None
