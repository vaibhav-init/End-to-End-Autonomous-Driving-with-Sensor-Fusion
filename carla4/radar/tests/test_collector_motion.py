"""Unit tests for the collector's radial triangular motion profile (no CARLA)."""

import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from collect_carla_radar_ghosts import _triangular_offset_speed  # noqa: E402


class TriangularMotionTests(unittest.TestCase):
    def test_speed_magnitude_is_constant(self):
        amplitude = 1.5
        speed = 1.4
        period = max(4.0, 4.0 * amplitude / speed)
        for step in range(0, 100):
            elapsed = step * 0.1
            offset, velocity = _triangular_offset_speed(elapsed, amplitude, period)
            self.assertAlmostEqual(
                abs(velocity),
                4.0 * amplitude / period,
                places=6,
            )
            self.assertLessEqual(abs(offset), amplitude + 1.0e-9)

    def test_mean_radial_speed_reaches_walking_speed(self):
        # Task 1 requirement: mean |v_r| >= 1.3 m/s for a 1.4 m/s pedestrian.
        amplitude = 1.5
        speed = 1.4
        period = max(4.0, 4.0 * amplitude / speed)
        velocities = []
        for step in range(400):
            _offset, velocity = _triangular_offset_speed(
                step / 10.0,
                amplitude,
                period,
            )
            velocities.append(abs(velocity))
        mean = sum(velocities) / len(velocities)
        self.assertGreaterEqual(mean, 1.3)
        self.assertAlmostEqual(mean, speed, delta=0.05)

    def test_position_is_continuous(self):
        amplitude = 2.0
        period = 8.0
        previous = None
        for step in range(0, 800):
            elapsed = step / 100.0
            offset, _velocity = _triangular_offset_speed(elapsed, amplitude, period)
            if previous is not None:
                self.assertLess(abs(offset - previous), 0.5)
            previous = offset

    def test_wraps_periodically(self):
        amplitude = 1.5
        period = max(4.0, 4.0 * amplitude / 1.4)
        offset_start, _v = _triangular_offset_speed(0.0, amplitude, period)
        offset_after, _v = _triangular_offset_speed(period, amplitude, period)
        self.assertAlmostEqual(offset_start, offset_after)


if __name__ == "__main__":
    unittest.main()
