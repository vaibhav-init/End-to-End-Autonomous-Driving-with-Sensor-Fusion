import unittest

import numpy as np
import torch

from radar.ghost_detection.features import FEATURE_NAMES, physical_features
from radar.ghost_detection.metrics import BinaryHistogramMetrics
from radar.ghost_detection.model import PointMLP, TemporalPointNet


class RadarGhostModelTest(unittest.TestCase):
    def test_physical_feature_shape_and_coordinate_contract(self):
        features = physical_features(
            np.array((10.0, 20.0)),
            np.array((0.0, np.pi / 2.0)),
            np.array((2.0, -2.0)),
            np.array((10.0, 100.0)),
            np.array((0.0, 0.1)),
        )
        self.assertEqual(features.shape, (2, len(FEATURE_NAMES)))
        self.assertAlmostEqual(float(features[0, 0]), 0.1)
        self.assertAlmostEqual(float(features[1, 1]), 0.2)
        self.assertGreater(float(features[0, 5]), 0.0)
        self.assertLess(float(features[1, 5]), 0.0)

    def test_model_output_shapes(self):
        features = torch.zeros((2, 12, len(FEATURE_NAMES)))
        mask = torch.ones((2, 12), dtype=torch.bool)
        self.assertEqual(PointMLP()(features, mask).shape, (2, 12))
        self.assertEqual(TemporalPointNet()(features, mask).shape, (2, 12))

    def test_temporal_context_ignores_padding(self):
        torch.manual_seed(5)
        model = TemporalPointNet(dropout=0.0).eval()
        first = torch.zeros((1, 4, len(FEATURE_NAMES)))
        second = first.clone()
        second[:, 2:, :] = 1000.0
        mask = torch.tensor(((True, True, False, False),))
        with torch.no_grad():
            first_logits = model(first, mask)
            second_logits = model(second, mask)
        self.assertTrue(torch.allclose(first_logits[:, :2], second_logits[:, :2]))

    def test_streaming_metrics_detect_perfect_separation(self):
        metrics = BinaryHistogramMetrics()
        metrics.update((0.01, 0.1, 0.9, 0.99), (0, 0, 1, 1))
        result = metrics.compute(fixed_threshold=0.5)
        self.assertAlmostEqual(result["auprc"], 1.0)
        self.assertAlmostEqual(result["auroc"], 1.0)
        self.assertEqual(result["false_positive"], 0)
        self.assertEqual(result["false_negative"], 0)


if __name__ == "__main__":
    unittest.main()
