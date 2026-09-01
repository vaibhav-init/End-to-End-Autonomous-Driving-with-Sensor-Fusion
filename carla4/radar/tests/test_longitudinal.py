"""Phase 2: the reference longitudinal policy and the rules switch."""

import os
import sys
import unittest

_SCENARIOS = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "scenarios")
)
if _SCENARIOS not in sys.path:
    sys.path.insert(0, _SCENARIOS)

from drivers.longitudinal import (  # noqa: E402
    HybridStateMachineController,
    IntelligentDriverModel,
    PIDSpeedController,
)


def idm(**overrides):
    params = dict(
        desired_speed_mps=60.0 / 3.6,
        time_headway_s=1.5,
        minimum_gap_m=2.0,
        max_acceleration_mps2=1.5,
        comfortable_deceleration_mps2=2.0,
    )
    params.update(overrides)
    return IntelligentDriverModel(**params)


class IntelligentDriverModelTest(unittest.TestCase):
    def test_accelerates_on_a_clear_road(self):
        self.assertGreater(idm().acceleration(10.0, 1.0e6, 0.0), 0.0)

    def test_stops_accelerating_at_the_desired_speed(self):
        model = idm()
        self.assertAlmostEqual(
            model.acceleration(model.desired_speed_mps, 1.0e6, 0.0),
            0.0,
            places=6,
        )

    def test_brakes_harder_as_the_gap_closes(self):
        model = idm()
        far = model.acceleration(15.0, 40.0, 5.0)
        near = model.acceleration(15.0, 15.0, 5.0)
        very_near = model.acceleration(15.0, 6.0, 5.0)
        self.assertGreater(far, near)
        self.assertGreater(near, very_near)
        self.assertLess(very_near, 0.0)

    def test_closing_speed_increases_braking_demand(self):
        model = idm()
        steady = model.acceleration(15.0, 25.0, 0.0)
        closing = model.acceleration(15.0, 25.0, 8.0)
        self.assertLess(closing, steady)

    def test_zero_or_negative_gap_brakes(self):
        model = idm()
        self.assertLess(model.acceleration(10.0, 0.0, 0.0), 0.0)
        self.assertLess(model.acceleration(10.0, -1.0, 0.0), 0.0)

    def test_target_speed_is_bounded(self):
        model = idm()
        # Never negative, even under an emergency deceleration demand.
        self.assertGreaterEqual(
            model.target_speed(12.0, 3.0, 12.0, dt=0.05), 0.0
        )
        # Never above the desired speed on a clear road.
        self.assertLessEqual(
            model.target_speed(model.desired_speed_mps, 1.0e6, 0.0, dt=0.05),
            model.desired_speed_mps + 1.0e-9,
        )

    def test_accelerates_away_from_a_standstill(self):
        """Regression: the ego sat still forever at the start of collection.

        ``target_speed`` used to integrate from the *measured* speed, so at
        rest it returned 0 + 1.5*0.05 = 0.075 m/s. That is below the hold
        threshold of the controller it feeds, so the brake was held, the
        speed stayed zero, and the command stayed at 0.075 -- a deadlock the
        run could never leave.
        """

        model = idm()
        controller = HybridStateMachineController(PIDSpeedController(dt=0.05))
        speed = 0.0
        for _ in range(40):  # two seconds at 20 Hz
            target = model.target_speed(
                speed, gap_m=100.0, closing_speed_mps=0.0, dt=0.05
            )
            throttle, brake = controller.run_step(target, speed)
            # Crude plant: enough to tell "moving" from "held at zero".
            speed = max(0.0, speed + (throttle * 1.5 - brake * 4.0) * 0.05)
        self.assertEqual(controller.state, "DRIVE")
        self.assertGreater(speed, 1.0)

    def test_command_does_not_wind_up_while_blocked(self):
        """A stuck ego must not accumulate an unreachable setpoint."""

        model = idm()
        target = 0.0
        for _ in range(200):
            target = model.target_speed(
                0.0, gap_m=100.0, closing_speed_mps=0.0, dt=0.05
            )
        self.assertLessEqual(target, model.command_lead_margin_mps + 1.0e-6)

    def test_reset_forgets_the_integrated_command(self):
        model = idm()
        for _ in range(40):
            model.target_speed(8.0, gap_m=100.0, closing_speed_mps=0.0, dt=0.05)
        model.reset()
        # After a reset the command restarts from the measured speed.
        self.assertAlmostEqual(
            model.target_speed(0.0, gap_m=100.0, closing_speed_mps=0.0, dt=0.05),
            1.5 * 0.05,
            places=3,
        )

    def test_larger_headway_keeps_a_larger_gap(self):
        timid = idm(time_headway_s=2.5).acceleration(15.0, 30.0, 0.0)
        eager = idm(time_headway_s=0.8).acceleration(15.0, 30.0, 0.0)
        self.assertLess(timid, eager)


class ControllerTailTest(unittest.TestCase):
    def test_pid_returns_throttle_or_brake_never_both(self):
        pid = PIDSpeedController(dt=0.05)
        throttle, brake = pid.run_step(20.0, 5.0)
        self.assertGreater(throttle, 0.0)
        self.assertEqual(brake, 0.0)
        pid.reset()
        throttle, brake = pid.run_step(0.0, 20.0)
        self.assertEqual(throttle, 0.0)
        self.assertGreater(brake, 0.0)

    def test_hold_state_engages_only_when_stopping(self):
        controller = HybridStateMachineController(PIDSpeedController(dt=0.05))
        controller.run_step(0.0, 0.2)
        self.assertEqual(controller.state, "HOLD")
        controller.run_step(10.0, 5.0)
        self.assertEqual(controller.state, "DRIVE")
