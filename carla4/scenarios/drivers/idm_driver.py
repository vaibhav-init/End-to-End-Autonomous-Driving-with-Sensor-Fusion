#!/usr/bin/env python3
"""IDM driver — the reference longitudinal policy for the ghost study.

Radar gap and closing speed feed Treiber's Intelligent Driver Model, which
produces a target speed for the same PID tail the learned driver uses.
Steering is delegated to BasicAgent, exactly as for every other driver, so a
run differs from the MLP arm only in how the target speed was chosen.

Why this exists: the learned driver imitates a weak autopilot teacher and its
behaviour is not independently characterised, so odd results cannot be
attributed to ghosts rather than to the model. IDM is deterministic, its
parameters are physical, and it has twenty-five years of published behaviour
behind it. It is the measurement instrument; the MLP is the subject.

Needs no camera and no learned weights, so it runs anywhere the radar runs.
"""

import math
import os
import sys

import carla

from .base import Driver
from .longitudinal import (
    HybridStateMachineController,
    IntelligentDriverModel,
    PIDSpeedController,
)
from .steering import BasicAgentSteering

_CARLA4_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _CARLA4_DIR not in sys.path:
    sys.path.insert(0, _CARLA4_DIR)

from radar import (  # noqa: E402
    create_front_radar,
    normalize_radar_backend,
    resolve_realistic_radar_config,
)
from driving_contract import (  # noqa: E402
    MAX_TARGET_SPEED_KMH,
    NATIVE_RADAR_POINTS_PER_SECOND,
    RADAR_RANGE_M,
)

FPS = 20


class IDMDriver(Driver):
    name = "idm"

    def __init__(
        self,
        fps=FPS,
        debug_every=20,
        radar_backend=None,
        radar_profile=None,
        radar_config_path=None,
        radar_seed=42,
        radar_ghost_detector=None,
        radar_ghost_threshold=None,
        radar_ghost_device="cpu",
        radar_range_m=RADAR_RANGE_M,
        radar_points_per_second=None,
        desired_speed_kmh=MAX_TARGET_SPEED_KMH,
        time_headway_s=1.5,
        minimum_gap_m=2.0,
        max_acceleration_mps2=1.5,
        comfortable_deceleration_mps2=2.0,
        **_ignored,
    ):
        self.fps = int(fps)
        self.debug_every = debug_every
        self.radar_backend = normalize_radar_backend(radar_backend)
        self.radar_profile = radar_profile
        self.radar_config_path = radar_config_path
        self.radar_seed = int(radar_seed)
        self.radar_ghost_detector = radar_ghost_detector
        self.radar_ghost_threshold = radar_ghost_threshold
        self.radar_ghost_device = radar_ghost_device
        self.max_range = float(radar_range_m)
        self.radar_points_per_second = int(
            radar_points_per_second
            if radar_points_per_second is not None
            else (
                NATIVE_RADAR_POINTS_PER_SECOND
                if self.radar_backend == "native"
                else 240000
            )
        )
        self.desired_speed_mps = (
            min(float(desired_speed_kmh), MAX_TARGET_SPEED_KMH) / 3.6
        )
        self.idm = IntelligentDriverModel(
            desired_speed_mps=self.desired_speed_mps,
            time_headway_s=time_headway_s,
            minimum_gap_m=minimum_gap_m,
            max_acceleration_mps2=max_acceleration_mps2,
            comfortable_deceleration_mps2=comfortable_deceleration_mps2,
        )
        self.realistic_radar_config = None
        self.radar = None
        self.steering = None
        self.controller = None
        self._frame = 0
        self._last_distance_state = None
        self._last_target_speed = 0.0

    def setup(self, world, ego, carla_map, client):
        if self.radar_backend == "realistic":
            self.realistic_radar_config = resolve_realistic_radar_config(
                range_m=self.max_range,
                fps=self.fps,
                profile_name=self.radar_profile,
                config_path=self.radar_config_path,
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
        )
        self.steering = BasicAgentSteering(ego, carla_map)
        self.controller = HybridStateMachineController(
            PIDSpeedController(dt=1.0 / self.fps)
        )
        print(
            f"  [idm] v0={self.desired_speed_mps * 3.6:.0f} km/h "
            f"T={self.idm.time_headway_s}s s0={self.idm.minimum_gap_m}m "
            f"a={self.idm.max_acceleration_mps2} b="
            f"{self.idm.comfortable_deceleration_mps2} "
            f"| radar={self.radar_backend}"
        )

    def get_control(self, ego, world):
        velocity = ego.get_velocity()
        speed = math.sqrt(
            velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2
        )
        self.radar.update_ego_speed(speed)
        distance_state = self.radar.get()
        self._last_distance_state = distance_state.copy()

        target_speed = self.idm.target_speed(
            speed_mps=speed,
            gap_m=distance_state["distance"],
            closing_speed_mps=distance_state["relative_velocity"],
            dt=1.0 / self.fps,
        )
        self._last_target_speed = target_speed
        throttle, brake = self.controller.run_step(target_speed, speed)
        steer = self.steering.get_steer()

        if self.debug_every and self._frame % self.debug_every == 0:
            print(
                f"  [idm] f={self._frame:4d} spd={speed * 3.6:5.1f}km/h "
                f"tgt={target_speed * 3.6:5.1f}km/h "
                f"gap={distance_state['distance']:5.1f}m "
                f"relv={distance_state['relative_velocity']:+5.1f} "
                f"thr/brk={throttle:.2f}/{brake:.2f} "
                f"steer={steer:+.2f} [{self.controller.state}]"
            )
        self._frame += 1
        return carla.VehicleControl(
            throttle=throttle,
            steer=steer,
            brake=brake,
        )

    def diagnostics(self):
        diagnostics = {}
        if self.radar is not None:
            diagnostics.update(self.radar.diagnostics())
        if self._last_distance_state is not None:
            diagnostics["radar_distance_m"] = self._last_distance_state[
                "distance"
            ]
            diagnostics["radar_relative_velocity_mps"] = (
                self._last_distance_state["relative_velocity"]
            )
        diagnostics["target_speed_mps"] = self._last_target_speed
        return diagnostics

    def cleanup(self):
        if self.radar is not None:
            self.radar.cleanup()
            self.radar = None
