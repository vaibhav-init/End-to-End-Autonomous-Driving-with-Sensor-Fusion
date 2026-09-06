#!/usr/bin/env python3
"""Shared longitudinal control primitives for every driver.

Both the learned and the reference driver end in the same place: a target
speed handed to the same PID and the same brake-hold state machine. Keeping
that tail identical is what makes the comparison between them meaningful --
the only thing that differs is how the target speed was chosen.
"""

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from driving_contract import obstacle_relevant, relevance_window_m  # noqa: E402,F401


class PIDSpeedController:
    """Convert a desired speed into throttle/brake."""

    def __init__(self, dt, kp=0.75, ki=0.08, kd=0.12):
        self.dt = dt
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral = 0.0
        self.previous_error = 0.0

    def reset(self):
        self.integral = 0.0
        self.previous_error = 0.0

    def run_step(self, target_speed, current_speed):
        error = target_speed - current_speed
        self.integral += error * self.dt
        self.integral = max(-5.0, min(5.0, self.integral))
        derivative = (error - self.previous_error) / self.dt
        self.previous_error = error
        command = (
            self.kp * error + self.ki * self.integral + self.kd * derivative
        )
        command = max(-1.0, min(1.0, command))
        if command >= 0.0:
            return command, 0.0
        return 0.0, -command


class HybridStateMachineController:
    """PID for dynamics plus a brake-hold state at very low speed."""

    def __init__(
        self,
        pid_controller,
        hold_speed_threshold=0.3,
        stop_current_speed_threshold=0.8,
        hold_brake_force=0.35,
    ):
        self.pid = pid_controller
        self.hold_speed_threshold = hold_speed_threshold
        self.stop_current_speed_threshold = stop_current_speed_threshold
        self.hold_brake_force = hold_brake_force
        self.state = "DRIVE"

    def run_step(self, target_speed, current_speed):
        if (
            target_speed < self.hold_speed_threshold
            and current_speed < self.stop_current_speed_threshold
        ):
            self.state = "HOLD"
            self.pid.reset()
            return 0.0, self.hold_brake_force
        self.state = "DRIVE"
        return self.pid.run_step(target_speed, current_speed)
