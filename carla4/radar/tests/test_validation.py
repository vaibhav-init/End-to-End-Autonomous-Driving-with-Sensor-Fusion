import unittest

from radar.validation import BackendAccuracy, error_statistics
from validate_radar_accuracy import (
    _assess_carla_versions,
    _is_path_relevant,
)


class RadarValidationMetricsTest(unittest.TestCase):
    def test_version_assessment_accepts_semver_and_matching_build_hash(self):
        semantic = _assess_carla_versions("0.9.16", "0.9.16")
        self.assertTrue(semantic["accepted"])
        self.assertEqual(
            semantic["mode"],
            "verified_semantic_version",
        )

        source_build = _assess_carla_versions("9c62014", "9c62014")
        self.assertTrue(source_build["accepted"])
        self.assertEqual(
            source_build["mode"],
            "matching_source_build_id",
        )

        mismatch = _assess_carla_versions("9c62014", "294096e")
        self.assertFalse(mismatch["accepted"])

    def test_error_statistics(self):
        result = error_statistics((-2.0, 0.0, 2.0))
        self.assertEqual(result["count"], 3)
        self.assertAlmostEqual(result["bias"], 0.0)
        self.assertAlmostEqual(result["mae"], 4.0 / 3.0)
        self.assertAlmostEqual(result["rmse"], (8.0 / 3.0) ** 0.5)
        self.assertEqual(result["p95_abs"], 2.0)

    def test_backend_accuracy_separates_misses_and_wrong_targets(self):
        metrics = BackendAccuracy("realistic", identity_available=True)
        metrics.update(
            observable=True,
            reported=False,
            target_id=None,
            lead_id=7,
            synchronized=True,
            callback_error=False,
            frame_lag=0,
        )
        correct = metrics.update(
            observable=True,
            reported=True,
            target_id=-3,
            lead_id=7,
            synchronized=False,
            callback_error=False,
            frame_lag=1,
            range_error_current=4.0,
            velocity_error_current=1.0,
            range_error_aligned=3.0,
            velocity_error_aligned=0.5,
        )
        self.assertFalse(correct)
        summary = metrics.summary()
        self.assertEqual(summary["miss_rate_when_observable"], 0.5)
        self.assertEqual(summary["correct_target_rate"], 0.0)
        self.assertEqual(summary["unsynchronized_frames"], 1)

    def test_identity_rate_excludes_outputs_when_lead_is_not_relevant(self):
        metrics = BackendAccuracy("realistic", identity_available=True)
        metrics.update(
            observable=False,
            reported=True,
            target_id=99,
            lead_id=7,
            synchronized=True,
            callback_error=False,
            frame_lag=0,
        )
        metrics.update(
            observable=True,
            reported=True,
            target_id=7,
            lead_id=7,
            synchronized=True,
            callback_error=False,
            frame_lag=0,
        )
        summary = metrics.summary()
        self.assertEqual(summary["correct_target_rate"], 1.0)
        self.assertEqual(
            summary["correct_target_rate_all_reported_outputs"],
            0.5,
        )
        self.assertEqual(summary["lead_detection_rate_when_observable"], 1.0)

    def test_path_relevance_is_stricter_than_wide_sensor_visibility(self):
        ground_truth = {
            "longitudinal_m": 20.0,
            "surface_range_m": 21.0,
            "bbox_azimuth_min_deg": 20.0,
            "bbox_azimuth_max_deg": 24.0,
            "bbox_elevation_min_deg": -1.0,
            "bbox_elevation_max_deg": 1.0,
            "bbox_longitudinal_min_m": 18.0,
            "bbox_lateral_min_m": 7.0,
            "bbox_lateral_max_m": 8.5,
        }
        envelope = {
            "horizontal_fov_deg": 120.0,
            "min_elevation_deg": -8.0,
            "max_elevation_deg": 8.0,
            "max_range_m": 100.0,
        }
        self.assertFalse(
            _is_path_relevant(
                ground_truth,
                envelope,
                path_half_width_m=1.8,
                path_width_growth_per_m=0.004,
            )
        )

    def test_path_relevance_follows_yaw_rate_curvature(self):
        ground_truth = {
            "longitudinal_m": 18.0,
            "surface_range_m": 18.2,
            "bbox_azimuth_min_deg": 4.0,
            "bbox_azimuth_max_deg": 9.0,
            "bbox_elevation_min_deg": -1.0,
            "bbox_elevation_max_deg": 1.0,
            "bbox_longitudinal_min_m": 17.0,
            "bbox_lateral_min_m": 1.9,
            "bbox_lateral_max_m": 2.8,
        }
        envelope = {
            "horizontal_fov_deg": 120.0,
            "min_elevation_deg": -8.0,
            "max_elevation_deg": 8.0,
            "max_range_m": 100.0,
        }
        self.assertTrue(
            _is_path_relevant(
                ground_truth,
                envelope,
                path_half_width_m=1.8,
                path_width_growth_per_m=0.004,
                path_curvature_per_m=0.014,
            )
        )


if __name__ == "__main__":
    unittest.main()
