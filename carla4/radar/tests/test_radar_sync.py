"""The sensor-synchronisation helper the closed-loop study depends on.

Sensor callbacks run on their own thread and the C-Shenron pipeline needs
longer than one 20 Hz tick, so a loop that ticks and reads immediately
consumes target lists that fall further and further behind. Measured on a
straight approach at 16 m/s: a stopped car 9 m ahead was reported at 36 m.
"""

import sys
import threading
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from radar import wait_for_radar_frame  # noqa: E402


class FakeRadar:
    def __init__(self, frame=-1, error=None, has_frame=True):
        self._frame = frame
        self._error = error
        self._has_frame = has_frame

    def diagnostics(self):
        info = {"last_error": self._error}
        if self._has_frame:
            info["frame"] = self._frame
        return info

    def advance(self, frame):
        self._frame = frame


class RadarSyncTest(unittest.TestCase):
    def test_returns_immediately_when_already_current(self):
        started = time.monotonic()
        self.assertTrue(wait_for_radar_frame(FakeRadar(frame=120), 100, timeout_s=2.0))
        self.assertLess(time.monotonic() - started, 0.5)

    def test_waits_for_a_lagging_sensor(self):
        radar = FakeRadar(frame=90)
        threading.Timer(0.05, radar.advance, (100,)).start()
        self.assertTrue(wait_for_radar_frame(radar, 100, timeout_s=2.0))

    def test_times_out_instead_of_blocking_forever(self):
        started = time.monotonic()
        self.assertFalse(wait_for_radar_frame(FakeRadar(frame=10), 100, timeout_s=0.2))
        self.assertLess(time.monotonic() - started, 1.0)

    def test_callback_error_ends_the_wait(self):
        self.assertFalse(
            wait_for_radar_frame(FakeRadar(frame=10, error="boom"), 100, timeout_s=5.0)
        )

    def test_backend_without_a_frame_counter_never_blocks(self):
        self.assertTrue(
            wait_for_radar_frame(FakeRadar(has_frame=False), 100, timeout_s=5.0)
        )
        self.assertTrue(wait_for_radar_frame(object(), 100, timeout_s=5.0))


if __name__ == "__main__":
    unittest.main()
