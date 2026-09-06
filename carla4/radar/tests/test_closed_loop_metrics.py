"""Phantom-brake and jerk metrics, and the detection sidecar round trip."""

import os
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

_SCENARIOS = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "scenarios")
)
if _SCENARIOS not in sys.path:
    sys.path.insert(0, _SCENARIOS)

from metrics import (  # noqa: E402
    longitudinal_cost_metrics,
    phantom_brake_mask,
    rising_edges,
    stopping_envelope_m,
)
from radar.detection_log import (  # noqa: E402
    DetectionLog,
    detections_by_frame,
    load_detection_log,
    sidecar_path,
)
from radar.realistic_core import RadarDetection  # noqa: E402


def _run(frames, speed_kmh=50.0):
    return pd.DataFrame(
        {
            "gt_ego_speed_kmh": np.full(frames, speed_kmh),
            "gt_distance_to_npc_m": np.full(frames, 999.0),
            "gt_relative_velocity": np.zeros(frames),
            "gt_npc_in_path": np.zeros(frames),
            "brake": np.zeros(frames),
            "ego_accel_mps2": np.zeros(frames),
        }
    )


class PhantomBrakeTest(unittest.TestCase):
    def test_brake_with_nothing_ahead_is_phantom(self):
        df = _run(100)
        df.loc[40:49, "brake"] = 1.0
        df.loc[70:72, "brake"] = 0.6
        mask = phantom_brake_mask(df)
        self.assertEqual(int(mask.sum()), 13)
        self.assertEqual(rising_edges(mask), 2)
        metrics = longitudinal_cost_metrics(df)
        self.assertEqual(metrics["phantom_brake_events"], 2)
        self.assertAlmostEqual(metrics["distance_km"], 100 * (50 / 3.6) / 20 / 1000)

    def test_brake_for_closing_in_path_actor_is_legitimate(self):
        df = _run(50, speed_kmh=60.0)
        df["gt_distance_to_npc_m"] = 20.0
        df["gt_relative_velocity"] = 16.0
        df["gt_npc_in_path"] = 1.0
        df["brake"] = 1.0
        self.assertEqual(int(phantom_brake_mask(df).sum()), 0)

    def test_brake_for_adjacent_lane_actor_is_phantom(self):
        df = _run(50, speed_kmh=60.0)
        df["gt_distance_to_npc_m"] = 20.0
        df["gt_relative_velocity"] = 16.0
        df["gt_npc_in_path"] = 0.0
        df["brake"] = 1.0
        self.assertEqual(int(phantom_brake_mask(df).sum()), 50)

    def test_far_slow_closing_actor_does_not_justify_braking(self):
        df = _run(10, speed_kmh=50.0)
        df["gt_distance_to_npc_m"] = 90.0
        df["gt_relative_velocity"] = 1.0
        df["gt_npc_in_path"] = 1.0
        df["brake"] = 0.8
        envelope = stopping_envelope_m(50.0 / 3.6, 1.0)
        self.assertLess(envelope, 90.0)
        self.assertEqual(int(phantom_brake_mask(df).sum()), 10)

    def test_jerk_is_zero_for_constant_acceleration_and_positive_for_twitching(self):
        smooth = _run(200)
        smooth["ego_accel_mps2"] = 1.0
        twitchy = _run(200)
        twitchy["ego_accel_mps2"] = np.where(np.arange(200) % 10 < 5, 3.0, -3.0)
        self.assertAlmostEqual(longitudinal_cost_metrics(smooth)["jerk_rms_mps3"], 0.0)
        self.assertGreater(longitudinal_cost_metrics(twitchy)["jerk_rms_mps3"], 1.0)


class DetectionLogTest(unittest.TestCase):
    def test_round_trip_and_frame_grouping(self):
        log = DetectionLog()
        point = RadarDetection(
            distance_m=20.0,
            azimuth_rad=0.1,
            relative_velocity_mps=3.0,
            snr_db=25.0,
            source="direct",
            truth_object_id=7,
            semantic_tag=14,
        )
        ghost = RadarDetection(
            distance_m=30.0,
            azimuth_rad=-0.2,
            relative_velocity_mps=2.5,
            snr_db=15.0,
            source="ghost",
            truth_object_id=-5,
            semantic_tag=14,
            truth_parent_object_id=7,
            bounce_type="type2",
            bounce_order=2,
        )
        log.append(100, 5.0, 1, (point, ghost))
        log.append(101, 5.05, 1, (point, ghost))   # same scan seen twice
        log.append(102, 5.10, 2, ())                # empty scan
        with tempfile.TemporaryDirectory() as tmp:
            path = sidecar_path(os.path.join(tmp, "data.csv"))
            self.assertTrue(path.endswith("data.detections.npz"))
            log.save(path)
            loaded = load_detection_log(path)
        self.assertEqual(len(loaded["detections"]), 4)
        self.assertEqual(len(loaded["frames"]), 3)
        self.assertEqual(int(loaded["frames"]["count"][2]), 0)
        by_frame = detections_by_frame(loaded["detections"])
        self.assertEqual(sorted(by_frame), [100, 101])
        self.assertEqual(by_frame[100]["source"].tolist(), [b"direct", b"ghost"])
        self.assertEqual(int(by_frame[100]["truth_parent_object_id"][1]), 7)


if __name__ == "__main__":
    unittest.main()
