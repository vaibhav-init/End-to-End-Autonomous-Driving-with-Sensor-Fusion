"""Phase 2: the shared longitudinal control tail and the rules switch."""

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
    PIDSpeedController,
)


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
