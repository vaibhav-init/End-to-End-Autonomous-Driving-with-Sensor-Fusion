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
