#!/usr/bin/env python3
"""Shared longitudinal control primitives for every driver.

Both the learned and the reference driver end in the same place: a target
speed handed to the same PID and the same brake-hold state machine. Keeping
that tail identical is what makes the comparison between them meaningful --
the only thing that differs is how the target speed was chosen.
"""

import math


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


class IntelligentDriverModel:
    """Treiber's IDM, used as the reference longitudinal policy.

    IDM is the natural comparison for this study: its inputs are exactly the
    radar contract -- gap, ego speed and closing speed -- its parameters are
    physically meaningful rather than fitted, and its behaviour is documented
    well enough that a reviewer can predict it. That makes it an instrument
    for isolating what false alarms do to a controller, where a learned model
    of unknown character cannot separate "the ghosts did this" from "my model
    is quirky".

    Returns an acceleration, which the caller integrates into a target speed
    so the PID tail stays shared with the learned driver.
    """

    def __init__(
        self,
        desired_speed_mps,
        time_headway_s=1.5,
        minimum_gap_m=2.0,
        max_acceleration_mps2=1.5,
        comfortable_deceleration_mps2=2.0,
        acceleration_exponent=4.0,
        command_lead_margin_mps=2.0,
    ):
        self.desired_speed_mps = float(desired_speed_mps)
        self.time_headway_s = float(time_headway_s)
        self.minimum_gap_m = float(minimum_gap_m)
        self.max_acceleration_mps2 = float(max_acceleration_mps2)
        self.comfortable_deceleration_mps2 = float(
            comfortable_deceleration_mps2
        )
        self.acceleration_exponent = float(acceleration_exponent)
        self.command_lead_margin_mps = float(command_lead_margin_mps)
        self._commanded_speed = None

    def reset(self):
        """Forget the integrated command; call after a respawn or handover."""

        self._commanded_speed = None

    def acceleration(self, speed_mps, gap_m, closing_speed_mps):
        """IDM acceleration.

        ``closing_speed_mps`` follows the radar contract: positive means the
        gap is shrinking.
        """

        speed = max(0.0, float(speed_mps))
        free_road = 1.0 - (speed / max(self.desired_speed_mps, 1.0e-6)) ** (
            self.acceleration_exponent
        )
        gap = float(gap_m)
        if not math.isfinite(gap) or gap <= 0.0:
            return -self.comfortable_deceleration_mps2
        denominator = 2.0 * math.sqrt(
            self.max_acceleration_mps2 * self.comfortable_deceleration_mps2
        )
        desired_gap = self.minimum_gap_m + max(
            0.0,
            speed * self.time_headway_s
            + speed * float(closing_speed_mps) / max(denominator, 1.0e-6),
        )
        interaction = (desired_gap / max(gap, 1.0e-3)) ** 2
        return self.max_acceleration_mps2 * (free_road - interaction)

    def target_speed(self, speed_mps, gap_m, closing_speed_mps, dt):
        """Integrate one step of IDM acceleration into a target speed.

        The command is integrated from its own previous value, not from the
        measured speed. Integrating from the measured speed returns
        ``speed + a*dt``, so the PID downstream sees an error of exactly
        ``a*dt`` on every tick no matter how far from the desired speed the
        vehicle is -- at 20 Hz that is 0.075 m/s for a full 1.5 m/s^2 demand,
        which is not enough throttle to accelerate at all. From a standstill
        it also sits under the brake-hold threshold, so the state machine
        holds the brake, the speed stays zero, and the ego never moves.

        ``command_lead_margin_mps`` is the anti-windup: the setpoint may not
        run more than that far ahead of what the vehicle actually achieved,
        so a blocked ego does not accumulate an unreachable command.
        """

        acceleration = self.acceleration(speed_mps, gap_m, closing_speed_mps)
        measured = max(0.0, float(speed_mps))
        base = (
            measured if self._commanded_speed is None else self._commanded_speed
        )
        command = base + acceleration * float(dt)
        # Anti-windup, applied after integrating so the bound is exact.
        command = min(command, measured + self.command_lead_margin_mps)
        self._commanded_speed = max(
            0.0, min(self.desired_speed_mps, command)
        )
        return self._commanded_speed
