#!/usr/bin/env python3
"""
Scenario staging helpers.

GapKeepController drives the ego to a fixed following distance behind a lead
vehicle using ground-truth positions/speeds. It lets a dynamic scenario stage a
controlled tailgating state (so every run starts the critical phase identically)
before handing longitudinal control to the model under test.

Throttle/brake only — steering is handled elsewhere. Env-safe (no heavy deps).
"""


class GapKeepController:
    """Hold the ego `target_gap_m` behind a lead vehicle (returns throttle, brake)."""

    def __init__(self, target_gap_m, dt, kp=0.6, ki=0.06, kd=0.10,
                 gap_gain=0.4, max_speed_mps=20.0):
        self.target_gap = target_gap_m
        self.dt = dt
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.gap_gain = gap_gain            # m/s of speed bias per metre of gap error
        self.max_speed = max_speed_mps      # cap so the ego can catch up but not run away
        self._integral = 0.0
        self._prev_error = 0.0

    def desired_speed(self, current_gap_m, lead_speed_mps):
        """Match the lead's speed, biased to drive the gap toward the target."""
        target = lead_speed_mps + self.gap_gain * (current_gap_m - self.target_gap)
        return max(0.0, min(self.max_speed, target))

    def run_step(self, current_gap_m, ego_speed_mps, lead_speed_mps):
        target_speed = self.desired_speed(current_gap_m, lead_speed_mps)
        error = target_speed - ego_speed_mps
        self._integral = max(-5.0, min(5.0, self._integral + error * self.dt))
        derivative = (error - self._prev_error) / max(self.dt, 1e-6)
        self._prev_error = error
        accel_cmd = self.kp * error + self.ki * self._integral + self.kd * derivative
        accel_cmd = max(-1.0, min(1.0, accel_cmd))
        if accel_cmd >= 0.0:
            return accel_cmd, 0.0
        return 0.0, -accel_cmd
