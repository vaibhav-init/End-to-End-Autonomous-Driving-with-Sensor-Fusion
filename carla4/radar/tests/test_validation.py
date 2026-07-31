import unittest

from radar.validation import BackendAccuracy, error_statistics
from validate_radar_accuracy import _assess_carla_versions


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


if __name__ == "__main__":
    unittest.main()
