"""Calibration statistics on a synthetic prepared sequence with known answers."""

import json
import math
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from radar.ghost_detection.statistics import (  # noqa: E402
    amplitude_to_db,
    bounce_family,
    merge_statistics,
    sequence_statistics,
    wasserstein_1d,
)
from calibrate_ghost_profile import derive_overrides  # noqa: E402


def _sequence(frames=6, real_points=4, ghost_points=2, ghost_offset_m=3.0,
              ghost_amp_ratio=0.5, instance=True, doppler_std=0.3, seed=0):
    """One pedestrian (instance 5) with type1 second-order ghosts each frame."""

    rng = np.random.default_rng(seed)
    rows = []
    for frame in range(frames):
        r_parent = 10.0 + 0.1 * frame
        for _ in range(real_points):
            rows.append((frame, 0, r_parent + rng.uniform(-0.2, 0.2), 0.01 * rng.standard_normal(),
                         -1.4 + doppler_std * rng.standard_normal(), 200.0 * math.exp(0.05 * rng.standard_normal()),
                         1011, 0, 1, 1, 1, 5))
        for _ in range(ghost_points):
            rows.append((frame, 0, r_parent + ghost_offset_m + rng.uniform(-0.1, 0.1), 0.2,
                         -1.4 * 0.8, 200.0 * ghost_amp_ratio, 1012, 1, 1, 1, 2, 5))
        # unlabeled background
        rows.append((frame, 0, 40.0, 0.5, 0.0, 30.0, 0, -1, -1, -1, -1, -1))
    arr = np.array(rows, dtype=[
        ("frame", np.int64), ("sensor", np.int8), ("r_sc", np.float32), ("phi_sc", np.float32),
        ("vr_sc", np.float32), ("amp", np.float32), ("label_id", np.int32), ("target", np.int8),
        ("class_id", np.int8), ("bounce_type", np.int8), ("bounce_order", np.int8), ("instance_id", np.int64),
    ])
    sequence = {name: np.asarray(arr[name]) for name in arr.dtype.names}
    if not instance:
        del sequence["instance_id"]
    return sequence


class StatisticsTest(unittest.TestCase):
    def test_bounce_family_codes(self):
        families = bounce_family([1, 2, 2, 0], [2, 2, 4, 0])
        self.assertEqual(families.tolist(), ["type1_second", "type2_second", "type2_third", "other_multipath"])

    def test_amplitude_unit_detection(self):
        _, unit = amplitude_to_db(np.array([10.0, 500.0, 2000.0]))
        self.assertEqual(unit, "linear")
        values, unit = amplitude_to_db(np.array([-5.0, 12.0, 40.0]))
        self.assertEqual(unit, "db")
        self.assertEqual(values.tolist(), [-5.0, 12.0, 40.0])

    def test_ghost_parent_relations_recover_known_offsets(self):
        stats = sequence_statistics(_sequence())
        meta = stats["_meta"]
        self.assertEqual(meta["frames"], 6)
        self.assertEqual(meta["real_points"], 24)
        self.assertEqual(meta["ghost_points"], 12)
        self.assertEqual(meta["instance_pairs"], 12)
        self.assertEqual(meta["class_pairs"], 0)
        self.assertAlmostEqual(float(np.median(stats["type1_second_delta_range_m"])), 3.0, delta=0.3)
        self.assertAlmostEqual(float(np.median(stats["type1_second_delta_amp_db"])), 20 * math.log10(0.5), delta=0.6)
        self.assertAlmostEqual(float(np.median(stats["type1_second_doppler_ratio"])), 0.8, delta=0.05)
        self.assertEqual(int(np.median(stats["object_points"])), 4)
        self.assertEqual(stats["ghost_lifetime_frames"].tolist(), [6.0])
        self.assertEqual(int(np.median(stats["points_per_frame"])), 7)
        self.assertEqual(stats["ghost_cluster_points"].tolist(), [2.0] * 6)
        self.assertEqual(stats["ghost_clusters_per_object"].tolist(), [1.0] * 6)
        # One unlabelled background point per frame; the frame median is in
        # 20log10 dB of the linear amplitude (200 -> 46 dB).
        self.assertEqual(stats["background_per_frame"].tolist(), [1] * 6)
        self.assertEqual(len(stats["frame_median_amp_db"]), 6)
        self.assertAlmostEqual(float(np.median(stats["frame_median_amp_db"])), 20 * math.log10(200.0), delta=1.0)

    def test_class_centroid_fallback_without_instance_ids(self):
        stats = sequence_statistics(_sequence(instance=False))
        self.assertEqual(stats["_meta"]["instance_pairs"], 0)
        self.assertEqual(stats["_meta"]["class_pairs"], 12)
        self.assertAlmostEqual(float(np.median(stats["type1_second_delta_range_m"])), 3.0, delta=0.3)

    def test_merge_and_wasserstein(self):
        a = sequence_statistics(_sequence(seed=1))
        b = sequence_statistics(_sequence(seed=2))
        merged = merge_statistics([a, b])
        self.assertEqual(merged["_meta"]["ghost_points"], 24)
        self.assertEqual(len(merged["type1_second_delta_range_m"]), 24)
        self.assertAlmostEqual(wasserstein_1d([1, 2, 3], [1, 2, 3]), 0.0)
        self.assertAlmostEqual(wasserstein_1d([0, 0, 0], [2, 2, 2]), 2.0)
        self.assertTrue(math.isnan(wasserstein_1d([], [1.0])))

    def test_derived_overrides_follow_the_measurements(self):
        parts = [sequence_statistics(_sequence(frames=8, real_points=6, ghost_offset_m=3.0,
                                               ghost_amp_ratio=0.25, doppler_std=0.9, seed=s))
                 for s in range(4)]
        stats = merge_statistics(parts)
        overrides, notes = derive_overrides(stats, "rgd_regime_v1")
        self.assertEqual(overrides["points_per_object_mean"], 6.0)
        # 0.25 amplitude ratio = -12 dB; spreading 40log10(13/10) ~ 4.6 dB; so
        # bounce loss ~ 12 - 4.6 - 2 = 5.4 dB.
        self.assertAlmostEqual(overrides["multipath_second_order_loss_db"], 5.4, delta=0.8)
        self.assertIn("multipath_second_order_loss_db", notes)
        # Pedestrian Doppler std 0.9 against an implied ~0.44 -> scale ~2.
        self.assertGreater(overrides["micro_doppler_scale"], 1.5)
        # Two ghost points per cluster against six per real object.
        self.assertAlmostEqual(overrides["ghost_points_scale"], 2.0 / 6.0, places=2)
        self.assertNotIn("ghost_rate_scale", overrides)
        for key, value in overrides.items():
            self.assertTrue(isinstance(value, (int, float)), key)

    def test_relative_fits_correct_the_base_overrides(self):
        real = merge_statistics([sequence_statistics(_sequence(frames=8, seed=s)) for s in range(3)])
        # Synthetic reference with three ghost clusters per object (real has
        # one), road users 20 dB brighter than in the real data and ghosts
        # 4 dB weaker relative to their parents.
        synthetic = sequence_statistics(_sequence(frames=8, seed=9))
        synthetic["ghost_clusters_per_object"] = np.full(24, 3.0)
        synthetic["real_rel_amp_db"] = real["real_rel_amp_db"] + 20.0
        synthetic["ghost_rel_amp_db"] = real["ghost_rel_amp_db"] + 16.0
        # Pad to the minimum sample counts the fits require.
        synthetic["real_rel_amp_db"] = np.tile(synthetic["real_rel_amp_db"], 3)
        synthetic["ghost_rel_amp_db"] = np.tile(synthetic["ghost_rel_amp_db"], 6)
        real["real_rel_amp_db"] = np.tile(real["real_rel_amp_db"], 3)
        real["ghost_rel_amp_db"] = np.tile(real["ghost_rel_amp_db"], 6)
        # Synthetic frames sit 100 dB below the real ones, carry half the
        # background points, half the object range spread, twice the
        # within-object Doppler std and the same fading.
        synthetic["frame_median_amp_db"] = real["frame_median_amp_db"] - 100.0
        synthetic["background_per_frame"] = real["background_per_frame"] * 0.5
        synthetic["object_range_spread_m"] = real["object_range_spread_m"] * 0.5
        synthetic["object_doppler_std_mps"] = real["object_doppler_std_mps"] * 2.0
        real["ghost_fading_std_db"] = np.full(12, 2.0)
        synthetic["ghost_fading_std_db"] = np.full(12, 4.0)
        base = {
            "ghost_rate_scale": 0.5, "road_user_snr_offset_db": -10.0, "ghost_snr_offset_db": 1.0,
            "amplitude_gain_db": 5.0, "static_points_per_cluster_mean": 1.5,
            "point_footprint_scale": 0.6, "micro_doppler_scale": 0.5, "multipath_fading_std_db": 2.0,
        }
        overrides, notes = derive_overrides(real, "rgd_regime_v1", synthetic, base)
        self.assertAlmostEqual(overrides["ghost_rate_scale"], 0.5 / 3.0, places=3)
        self.assertAlmostEqual(overrides["road_user_snr_offset_db"], -30.0, delta=0.05)
        # real gap - synthetic gap = (g - r) - ((g+16) - (r+20)) = +4 dB
        self.assertAlmostEqual(overrides["ghost_snr_offset_db"], 5.0, delta=0.05)
        self.assertAlmostEqual(overrides["amplitude_gain_db"], 105.0, delta=0.05)
        self.assertAlmostEqual(overrides["static_points_per_cluster_mean"], 3.0, places=3)
        self.assertTrue(overrides["expand_static_points"])
        self.assertAlmostEqual(overrides["point_footprint_scale"], 1.2, places=3)
        self.assertAlmostEqual(overrides["micro_doppler_scale"], 0.25, places=3)
        self.assertAlmostEqual(overrides["multipath_fading_std_db"], 1.0, places=3)
        for key in (
            "ghost_rate_scale", "road_user_snr_offset_db", "ghost_snr_offset_db", "amplitude_gain_db",
            "static_points_per_cluster_mean", "point_footprint_scale", "micro_doppler_scale",
        ):
            self.assertIn(key, notes)

    def test_without_reference_relative_knobs_are_absent(self):
        real = merge_statistics([sequence_statistics(_sequence(frames=8, seed=s)) for s in range(3)])
        overrides, _ = derive_overrides(real, "rgd_regime_v1")
        for key in (
            "ghost_rate_scale", "road_user_snr_offset_db", "ghost_snr_offset_db",
            "amplitude_gain_db", "static_points_per_cluster_mean", "expand_static_points",
        ):
            self.assertNotIn(key, overrides)


if __name__ == "__main__":
    unittest.main()
