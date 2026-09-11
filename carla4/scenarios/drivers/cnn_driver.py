#!/usr/bin/env python3
"""
CNN driver — one learned model over a range-azimuth grid of the radar scans.

  realistic radar -> point-level scans (last N) -> range-azimuth grid
  -> TargetSpeedCNN -> target speed -> the same PID tail as every other driver.

Same contract as the transformer driver and no filter stage anywhere; the
only difference is that the detection list is rendered into a grid instead of
being subsampled into 256 tokens, so every point survives and a ghost's parent
can never be dropped while the ghost is kept.

Needs the realistic backend: the native radar exposes no detection list.
"""

import json
import math
import os
import sys

import carla

from .base import Driver
from .longitudinal import HybridStateMachineController, PIDSpeedController
from .steering import BasicAgentSteering

_CARLA4_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _CARLA4_DIR not in sys.path:
    sys.path.insert(0, _CARLA4_DIR)

from radar import (wait_for_radar_frame,   # noqa: E402
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
from driving_contract import MAX_TARGET_SPEED_KMH, RADAR_RANGE_M  # noqa: E402
from transformer_controller import ScanHistory, obstacle_in_corridor  # noqa: E402
from cnn_controller import (  # noqa: E402
    AZIMUTH_BINS,
    GRID_CHANNELS,
    RANGE_BINS,
    build_window_grid,
    load_model,
    predict_target_speed,
)


FPS = 20
BOOTSTRAP_TARGET_SPEED_MPS = 12.0 / 3.6
CRUISE_SPEED_MPS = 30.0 / 3.6
LAUNCH_HOLD_SPEED_MPS = 0.5
LAUNCH_CLEAR_DISTANCE_M = 15.0
LAUNCH_ASSIST_FRAMES = FPS * 5


class CNNDriver(Driver):
    name = "cnn"

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
        cruise_floor=True,
        **_ignored,
    ):
        self.model_dir = model_dir
        self.fps = fps
        self.debug_every = debug_every
        self.cruise_floor = bool(cruise_floor)
        self.radar_sync_timeout_s = 5.0
        self._radar_sync_timeouts = 0
        self.radar_backend = normalize_radar_backend(radar_backend or "realistic")
        self.radar_profile = radar_profile
        self.radar_config_path = radar_config_path
        self.radar_seed = int(radar_seed)
        self.radar_ghost_detector = radar_ghost_detector
        self.radar_ghost_threshold = radar_ghost_threshold
        self.radar_ghost_device = radar_ghost_device
        self.radar_ghost_oracle = bool(radar_ghost_oracle)
        self.radar_overrides = dict(radar_overrides or {})
        self.realistic_radar_config = None
        self.model = None
        self.model_config = None
        self.device = "cpu"
        self.radar = None
        self.max_range = RADAR_RANGE_M
        self.max_target_speed_mps = MAX_TARGET_SPEED_KMH / 3.6
        self.window_frames = 10
        self.history = None
        self.steering = None
        self.controller = None
        self._frame = 0
        self._prev_speed = 0.0
        self._last_state = None
        self._last_target_speed = None
        self._last_point_count = 0

    def setup(self, world, ego, carla_map, client):
        import torch

        if self.radar_backend != "realistic":
            raise RuntimeError(
                "The transformer driver needs the realistic backend: only it "
                "produces the point-level detection list the model consumes."
            )
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.model_config = load_model(self.model_dir, device=self.device)
        self.window_frames = int(self.model_config.get("window_frames", 10))
        self.max_range = float(self.model_config.get("radar_range_m", RADAR_RANGE_M))
        self.max_target_speed_mps = (
            min(float(self.model_config.get("max_target_speed_kmh", MAX_TARGET_SPEED_KMH)),
                MAX_TARGET_SPEED_KMH) / 3.6
        )
        points_per_second = int(self.model_config.get("radar_points_per_second", 240000))

        embedded_config = (
            None if self.radar_profile or self.radar_config_path
            else self.model_config.get("radar_config")
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
            points_per_second=points_per_second,
            config=self.realistic_radar_config,
            ghost_detector_path=self.radar_ghost_detector,
            ghost_threshold=self.radar_ghost_threshold,
            ghost_oracle=self.radar_ghost_oracle,
        )
        print_provenance_warnings(
            check_radar_provenance(self.model_config, runtime_metadata),
            prefix="  [cnn] ",
        )

        self.radar = create_front_radar(
            ego,
            world,
            self.max_range,
            backend="realistic",
            fps=self.fps,
            points_per_second=points_per_second,
            config=self.realistic_radar_config,
            seed=self.radar_seed,
            ghost_detector_path=self.radar_ghost_detector,
            ghost_threshold=self.radar_ghost_threshold,
            ghost_device=self.radar_ghost_device,
            ghost_oracle=self.radar_ghost_oracle,
        )
        self.history = ScanHistory(self.window_frames, self.fps)
        self.steering = BasicAgentSteering(ego, carla_map)
        self.controller = HybridStateMachineController(PIDSpeedController(dt=1.0 / self.fps))
        self._frame = 0
        self._prev_speed = 0.0

        config = self.realistic_radar_config
        print("  [cnn] driver ready")
        print(f"  [cnn]   model:          {os.path.join(self.model_dir, self.model_config.get('checkpoint', ''))}")
        print(f"  [cnn]   window:         {self.window_frames} frames -> {GRID_CHANNELS}x{RANGE_BINS}x{AZIMUTH_BINS} grid")
        print(f"  [cnn]   radar profile:  {config.profile_name}")
        print(f"  [cnn]   radar config:   {realistic_radar_config_signature(config)}")
        print(f"  [cnn]   multipath:      {config.multipath_mode} "
              f"(rate x{config.ghost_rate_scale:g}, snr {config.ghost_snr_offset_db:+g} dB)")
        if self.radar_ghost_oracle:
            print("  [cnn]   ghost filter:   ORACLE (ground-truth ceiling)")
        print(f"  [cnn]   device:         {self.device}")
        print(f"  [cnn]   speed ceiling:  {self.max_target_speed_mps * 3.6:.1f} km/h")

    def get_control(self, ego, world):

        # The sensor thread lags the simulation under a dense LiDAR; without
        # this the model reads a target list over a second stale (a stopped
        # car 9 m ahead reported at 36 m). Same wait in every driver, so arms
        # stay comparable.
        if getattr(self, "radar", None) is not None:
            frame = world.get_snapshot().frame
            if not wait_for_radar_frame(self.radar, frame, self.radar_sync_timeout_s):
                self._radar_sync_timeouts += 1
        velocity = ego.get_velocity()
        speed = math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)
        accel = (speed - self._prev_speed) * self.fps if self._frame > 0 else 0.0
        self._prev_speed = speed

        self.radar.update_ego_speed(speed)
        self._last_state = self.radar.get()
        scan = self.radar.get_detections()
        self.history.push(scan)
        current = self.history.current()
        self._last_point_count = len(current)

        if len(self.history.windows()) >= 1 and self._frame >= 1:
            window = build_window_grid(self.history.windows(), speed)
            target_speed = predict_target_speed(self.model, window, device=self.device)
        else:
            target_speed = BOOTSTRAP_TARGET_SPEED_MPS

        detected = obstacle_in_corridor(current, self.max_range, speed)
        if self.cruise_floor and not detected:
            target_speed = max(target_speed, CRUISE_SPEED_MPS)
        if (self._frame < LAUNCH_ASSIST_FRAMES and speed < LAUNCH_HOLD_SPEED_MPS
                and self._last_state["distance"] > LAUNCH_CLEAR_DISTANCE_M):
            target_speed = max(target_speed, BOOTSTRAP_TARGET_SPEED_MPS)
        target_speed = min(max(0.0, target_speed), self.max_target_speed_mps)
        self._last_target_speed = target_speed

        throttle, brake = self.controller.run_step(target_speed, speed)
        steer = self.steering.get_steer()

        if self.debug_every and self._frame % self.debug_every == 0:
            print(f"  [cnn] f={self._frame:4d} spd={speed * 3.6:5.1f}km/h "
                  f"tgt={target_speed * 3.6:5.1f}km/h pts={self._last_point_count:3d} "
                  f"det={int(detected)} sel={self._last_state['distance']:5.1f}m "
                  f"thr/brk={throttle:.2f}/{brake:.2f} steer={steer:+.2f} "
                  f"[{self.controller.state}]")
        self._frame += 1
        return carla.VehicleControl(throttle=throttle, steer=steer, brake=brake)

    def diagnostics(self):
        diagnostics = {"safety_rules_enabled": 0, "safety_rule_fired": ""}
        if self.radar is not None:
            diagnostics.update(self.radar.diagnostics())
        if self._last_state is not None:
            diagnostics.update({
                "controller_distance_m": self._last_state["distance"],
                "controller_relative_velocity_mps": self._last_state["relative_velocity"],
                "controller_obstacle_speed_mps": self._last_state["obstacle_speed"],
            })
        if self._last_target_speed is not None:
            diagnostics["controller_target_speed_mps"] = self._last_target_speed
        diagnostics["controller_point_count"] = self._last_point_count
        return diagnostics

    def latest_detections(self):
        return self.radar.get_detections() if self.radar is not None else None

    def cleanup(self):
        if self.radar is not None:
            self.radar.cleanup()
            self.radar = None
