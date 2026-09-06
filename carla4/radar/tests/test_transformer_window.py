"""Window builder and scan history of the transformer controller (no torch)."""

import sys
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from radar.detection_log import DETECTION_DTYPE  # noqa: E402
from transformer_controller import (  # noqa: E402
    SOURCE_CODES,
    ScanHistory,
    build_window_tokens,
    dataset_frames_to_scans,
)


def _point(x, y, source, vr=-3.0, snr=20.0):
    r = float(np.hypot(x, y))
    return SimpleNamespace(
        distance_m=r,
        azimuth_rad=float(np.arctan2(y, x)),
        relative_velocity_mps=vr,
        snr_db=snr,
        source=source,
    )


def _scan(rng, ego_lane=10, adjacent=10, far=300):
    points = []
    for _ in range(ego_lane):
        points.append(_point(rng.uniform(20, 40), rng.uniform(-0.9, 0.9), "direct"))
    for _ in range(adjacent):
        points.append(_point(rng.uniform(15, 45), 3.5 * rng.choice((-1, 1)), "ghost"))
    for _ in range(far):
        points.append(_point(rng.uniform(5, 90), 12.0 * rng.choice((-1, 1)), "other", vr=0.0, snr=8.0))
    rng.shuffle(points)
    return points


class WindowBudgetTest(unittest.TestCase):
    def test_ego_lane_and_neighbours_fill_the_budget_before_far_statics(self):
        rng = np.random.default_rng(3)
        scans = [(0.1, _scan(rng)), (0.0, _scan(rng))]
        window = build_window_tokens(scans, 12.0, 0.0, max_points=50)
        self.assertEqual(window["point_count"], 50)
        codes = window["sources"][1:51]
        # All 20 ego-lane and all 20 adjacent-lane points of both scans
        # survive; the remaining 10 slots go to the far statics.
        self.assertEqual(int((codes == SOURCE_CODES["direct"]).sum()), 20)
        self.assertEqual(int((codes == SOURCE_CODES["ghost"]).sum()), 20)
        self.assertEqual(int((codes == SOURCE_CODES["other"]).sum()), 10)
        self.assertTrue(window["mask"][0])
        self.assertEqual(window["sources"][0], SOURCE_CODES["ego"])
        self.assertTrue(window["mask"][1:51].all())
        self.assertFalse(window["mask"][51:].any())

    def test_within_a_band_the_newest_scan_wins(self):
        rng = np.random.default_rng(5)
        old = [_point(30.0 + i * 0.1, 0.2, "direct") for i in range(30)]
        new = [_point(30.0 + i * 0.1, -0.2, "ghost") for i in range(30)]
        window = build_window_tokens([(0.2, old), (0.0, new)], 10.0, 0.0, max_points=40)
        codes = window["sources"][1:41]
        self.assertEqual(int((codes == SOURCE_CODES["ghost"]).sum()), 30)
        self.assertEqual(int((codes == SOURCE_CODES["direct"]).sum()), 10)
        del rng

    def test_small_scans_are_kept_whole(self):
        scans = [(0.0, [_point(20.0, 0.0, "direct"), _point(25.0, 4.0, "ghost")])]
        window = build_window_tokens(scans, 5.0, 0.0, max_points=16)
        self.assertEqual(window["point_count"], 2)
        self.assertEqual(window["tokens"].shape, (17, window["tokens"].shape[1]))


class ScanHistoryTest(unittest.TestCase):
    def test_online_window_spans_the_same_time_as_the_offline_builder(self):
        fps = 20
        window_frames = 10
        history = ScanHistory(window_frames, fps)
        # A 10 Hz radar inside the 20 Hz loop: one new scan every 0.1 s.
        for index in range(12):
            history.push({"scan_index": index, "timestamp": 0.1 * index, "detections": ()})
            history.push({"scan_index": index, "timestamp": 0.1 * index, "detections": ()})
        online = history.windows()

        frames = set(range(24))
        frame_to_scan = {frame: frame // 2 for frame in frames}
        empty = {frame: np.zeros(0, dtype=DETECTION_DTYPE) for frame in frames}
        offline = dataset_frames_to_scans(frames, empty, frame_to_scan, 23, window_frames, fps)

        self.assertEqual(len(online), len(offline))
        self.assertEqual(len(online), 5)
        self.assertAlmostEqual(max(age for age, _ in online), 0.4, places=6)
        self.assertLessEqual(max(age for age, _ in offline), 0.45)


if __name__ == "__main__":
    unittest.main()
