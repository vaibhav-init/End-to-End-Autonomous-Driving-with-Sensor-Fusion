#!/usr/bin/env python3
"""
MLP driver — the learned radar target-speed model for longitudinal control.

Ports the "as-deployed" longitudinal pipeline from
carla4/test_throttle_brake_live.py:
  radar -> stacked feature history -> TargetSpeedMLP -> target speed ->
  PID (+ hold-state machine) -> throttle/brake.

Steering is delegated to BasicAgent (lateral-only). Runs in the `carla4` conda
env (needs scikit-learn).

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
from .longitudinal import obstacle_relevant, HybridStateMachineController, PIDSpeedController
from .steering import BasicAgentSteering

# carla4/ (two levels up) holds the shared perception + model modules
_CARLA4_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _CARLA4_DIR not in sys.path:
    sys.path.insert(0, _CARLA4_DIR)

from speed_model import BASE_FEATURE_COLS as DEFAULT_BASE_FEATURE_COLS  # noqa: E402
from speed_model import TargetSpeedMLP, flatten_history  # noqa: E402
from radar import (  # noqa: E402
    create_front_radar,
    describe_radar_configuration,
    normalize_radar_backend,
    realistic_radar_config_signature,
    resolve_realistic_radar_config,
)
from radar_provenance import (  # noqa: E402
    check_radar_provenance,
    print_provenance_warnings,
)
from driving_contract import (  # noqa: E402
    MAX_TARGET_SPEED_KMH,
    NATIVE_RADAR_POINTS_PER_SECOND,
    RADAR_RANGE_M,
)


# Same constants as the live deployment (test_throttle_brake_live.py)
FPS = 20
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
        radar_ghost_oracle=False,
        radar_overrides=None,
        safety_rules=False,
        cruise_floor=True,
        **_ignored,
    ):
        self.model_dir = model_dir
        self.radar_ghost_oracle = bool(radar_ghost_oracle)
        self.radar_overrides = dict(radar_overrides or {})
        self._last_target_speed = None
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
        self.radar = None
        self.max_range = RADAR_RANGE
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

        self.max_range = float(model_config.get("radar_range_m", RADAR_RANGE))
        if self.radar_backend == "realistic":
            # The trained sensor is the default; a profile or config path on
            # the command line replaces it, and the runtime ghost knobs apply
            # on top of either. Ghost knobs sit outside the signature, so a
            # clean-trained model can be deployed with ghosts on.
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
                overrides=self.radar_overrides,
            )
            runtime_metadata = describe_radar_configuration(
                backend="realistic",
                range_m=self.max_range,
                fps=self.fps,
                points_per_second=self.radar_points_per_second,
                config=self.realistic_radar_config,
                ghost_detector_path=self.radar_ghost_detector,
                ghost_threshold=self.radar_ghost_threshold,
                ghost_oracle=self.radar_ghost_oracle,
            )
        else:
            runtime_metadata = describe_radar_configuration(
                backend=self.radar_backend,
                range_m=self.max_range,
                fps=self.fps,
                points_per_second=self.radar_points_per_second,
            )
        # Sensor mismatch raises; ghost-injection and filter differences are
        # the experiment and are printed, never hidden.
        print_provenance_warnings(
            check_radar_provenance(model_config, runtime_metadata),
            prefix="  [mlp] ",
        )

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
            ghost_oracle=self.radar_ghost_oracle,
        )

        self.steering = BasicAgentSteering(ego, carla_map)
        self.controller = HybridStateMachineController(
            PIDSpeedController(dt=1.0 / self.fps))
        self.feature_history = deque(maxlen=self.history_frames)
        self._frame = 0
        self._prev_speed = 0.0
        self._last_distance_state = None

        print("  [mlp] driver ready")
        print(f"  [mlp]   model:          {model_path}")
        print(f"  [mlp]   sensor:         radar ({self.radar_backend}) "
              f"(max {self.max_range:.0f}m)")
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
            print(
                "  [mlp]   multipath:      "
                f"{self.realistic_radar_config.multipath_mode} "
                f"(rate x{self.realistic_radar_config.ghost_rate_scale:g}, "
                f"snr {self.realistic_radar_config.ghost_snr_offset_db:+g} dB)"
            )
            if self.radar_ghost_oracle:
                print("  [mlp]   ghost filter:   ORACLE (ground-truth ceiling)")
            elif self.radar_ghost_detector:
                print(f"  [mlp]   ghost filter:   {self.radar_ghost_detector}")
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

        self.radar.update_ego_speed(speed)
        dist_state = self.radar.get()
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

        # Hybrid override (same as deployment): cruise on open road, where
        # "open" means no target inside the stopping-distance relevance
        # window (driving_contract.obstacle_relevant). A ghost inside the
        # window still reaches the model unmodified.
        if self.cruise_floor and not obstacle_relevant(
            dist_state["distance"], speed, dist_state["relative_velocity"], RADAR_RANGE
        ):
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
        self._last_target_speed = target_speed_pred

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
        if self._last_target_speed is not None:
            diagnostics["controller_target_speed_mps"] = self._last_target_speed
        return diagnostics

    def latest_detections(self):
        getter = getattr(self.radar, "get_detections", None)
        return getter() if getter is not None else None

    def cleanup(self):
        if self.radar is not None:
            self.radar.cleanup()
            self.radar = None
