"""Tests for the v2 zero-shot feature contract and CFAR-style export."""

import unittest

import numpy as np

from radar.ghost_detection.export_expansion import expand_detection_points
from radar.ghost_detection.features import (
    FEATURE_NAMES,
    FEATURE_SCHEMA_VERSION,
    frame_context_statistics,
    physical_features,
)


def _expand(detection, rng, mean_points):
    return expand_detection_points(
        detection,
        rng,
        mean_points,
        snr_to_amplitude=lambda snr_db: float(10.0 ** (snr_db / 20.0)),
    )


class FrameContextStatisticsTest(unittest.TestCase):
    def test_relative_log_amplitude_is_frame_centred(self):
        amplitudes = np.array((1.0, 10.0, 100.0, 1000.0))
        rel, _, _ = frame_context_statistics(
            np.array((5.0, 5.0, 5.0, 5.0)),
            np.zeros(4),
            np.zeros(4),
            amplitudes,
        )
        # The median-centred statistic must be scale-invariant: multiplying
        # every amplitude by a constant gain shifts nothing.
        rel_scaled, _, _ = frame_context_statistics(
            np.array((5.0, 5.0, 5.0, 5.0)),
            np.zeros(4),
            np.zeros(4),
            amplitudes * 500.0,
        )
        np.testing.assert_allclose(rel, rel_scaled, atol=1e-5)
        self.assertAlmostEqual(float(np.median(rel)), 0.0, places=4)

    def test_doppler_residual_flags_velocity_outliers(self):
        ranges = np.array((10.0, 10.2, 10.4, 60.0))
        azimuth = np.zeros(4)
        # Three clustered points share one velocity core; the isolated
        # single-point cell falls back to the frame median, so a point whose
        # velocity differs from the scene norm gets a large residual.
        velocity = np.array((-1.4, -1.45, -1.35, 2.5))
        _, residual, _ = frame_context_statistics(
            ranges,
            azimuth,
            velocity,
            np.ones(4),
        )
        self.assertGreater(float(residual[3]), 5.0 * float(residual[1]))

    def test_density_higher_inside_cluster_than_isolated_point(self):
        ranges = np.asarray(
            [10.0 + 0.3 * i for i in range(20)] + [80.0]
        )
        azimuth = np.zeros(ranges.size)
        stats = frame_context_statistics(
            ranges,
            azimuth,
            np.zeros(ranges.size),
            np.ones(ranges.size),
        )
        density = stats[2]
        self.assertGreater(float(density[:20].mean()), float(density[-1]) + 0.1)

    def test_physical_features_v2_shape_and_range_compensation(self):
        features = physical_features(
            np.array((10.0, 40.0)),
            np.zeros(2),
            np.zeros(2),
            np.array((10.0, 10.0)),
            np.zeros(2),
            relative_log_amplitude=np.array((0.5, -0.5)),
            doppler_cluster_residual=np.array((0.2, 0.4)),
            local_density_ratio=np.array((1.0, 0.25)),
        )
        self.assertEqual(features.shape, (2, len(FEATURE_NAMES)))
        # Same amplitude at longer range must yield a LARGER
        # range-compensated log amplitude (log1p(a r^2)).
        self.assertGreater(float(features[1, 8]), float(features[0, 8]))
        self.assertEqual(
            FEATURE_SCHEMA_VERSION,
            "radar_ghost_physical_v2",
        )

    def test_legacy_call_signature_still_works(self):
        features = physical_features(
            np.array((10.0,)),
            np.array((0.0,)),
            np.array((2.0,)),
            np.array((10.0,)),
            np.array((0.0,)),
        )
        self.assertEqual(features.shape, (1, len(FEATURE_NAMES)))


class ExpandDetectionPointsTest(unittest.TestCase):
    def _detection(self, semantic_tag=12, snr_db=30.0):
        return {
            "distance_m": 20.0,
            "azimuth_rad": 0.05,
            "relative_velocity_mps": -1.4,
            "snr_db": snr_db,
            "semantic_tag": semantic_tag,
            "source": "direct",
            "label_id": 1000 + 11,
        }

    def test_expansion_inherits_labels_and_respects_grids(self):
        rng = np.random.default_rng(7)
        detection = self._detection()
        points = _expand(detection, rng, 16.0)
        self.assertGreaterEqual(len(points), 1)
        for point in points:
            self.assertEqual(point["label_id"], detection["label_id"])
            self.assertEqual(point["source"], "direct")
            self.assertAlmostEqual(
                point["distance_m"] / 0.15,
                round(point["distance_m"] / 0.15),
                places=6,
            )
            self.assertAlmostEqual(
                point["relative_velocity_mps"] / 0.087,
                round(point["relative_velocity_mps"] / 0.087),
                places=6,
            )
            # Points stay within one footprint of the parent target.
            self.assertLess(abs(point["distance_m"] - 20.0), 3.0)

    def test_pedestrian_micro_doppler_exceeds_vehicle(self):
        ped_rng = np.random.default_rng(11)
        car_rng = np.random.default_rng(11)
        ped = _expand(self._detection(12), ped_rng, 32.0)
        car = _expand(self._detection(14), car_rng, 32.0)
        ped_spread = float(np.std([p["relative_velocity_mps"] for p in ped]))
        car_spread = float(np.std([p["relative_velocity_mps"] for p in car]))
        self.assertGreater(ped_spread, car_spread * 2.0)

    def test_mean_count_tracks_poisson_parameter(self):
        rng = np.random.default_rng(3)
        counts = [
            len(_expand(self._detection(), rng, 12.0))
            for _ in range(200)
        ]
        mean = float(np.mean(counts))
        self.assertGreater(mean, 9.0)
        self.assertLess(mean, 15.0)


if __name__ == "__main__":
    unittest.main()
