"""Threshold-curve aggregation over synthetic score-only sidecars."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from calibrate_runtime_threshold import cluster_mean_probabilities


def sidecar(rows):
    """Build a sidecar-like structured array from (source, tag, prob, n)."""

    dtype = np.dtype(
        [
            ("frame", np.int64),
            ("truth_object_id", np.int64),
            ("semantic_tag", np.int16),
            ("bounce_type", "S16"),
            ("bounce_order", np.int8),
            ("source", "S8"),
            ("ghost_probability", np.float32),
        ]
    )
    records = []
    for frame, (source, tag, probability, count) in enumerate(rows):
        for _ in range(count):
            records.append(
                (
                    frame,
                    tag * 10,           # unique object per tag in this test
                    tag,
                    b"direct" if source == "direct" else b"ghost",
                    1,
                    source.encode(),
                    probability,
                )
            )
    return np.array(records, dtype=dtype)


class ClusterMeanTest(unittest.TestCase):
    def sidecar_rows(self):
        """(source, tag, probability, count) rows; the helper adds frames."""

        return [
            ("direct", 14, 0.2, 4),   # cluster A: mean 0.2
            ("direct", 14, 0.8, 4),   # cluster B: mean 0.8
            ("ghost", 14, 0.9, 3),
            ("direct", 0, 0.9, 2),    # static background
        ]

    def test_cluster_mean_is_per_cluster_not_per_point(self):
        detections = sidecar(self.sidecar_rows())
        means, counts = cluster_mean_probabilities(detections)
        self.assertEqual(len(means), 3)
        self.assertEqual(sorted(counts.tolist()), [2, 4, 4])
        self.assertAlmostEqual(float(np.sort(means)[0]), 0.2, places=6)
        self.assertAlmostEqual(float(np.sort(means)[1]), 0.8, places=6)

    def test_dynamic_tags_exclude_static_background(self):
        detections = sidecar(
            [
                ("direct", 14, 0.9, 2),   # dynamic
                ("direct", 0, 0.95, 5),   # static: must not count
            ]
        )
        means_dynamic, counts_dynamic = cluster_mean_probabilities(
            detections, (12, 14, 18, 19, 21)
        )
        self.assertEqual(len(means_dynamic), 1)
        self.assertEqual(int(counts_dynamic.sum()), 2)
        means_all, counts_all = cluster_mean_probabilities(detections)
        self.assertEqual(len(means_all), 2)
        self.assertEqual(int(counts_all.sum()), 7)

    def test_frames_keep_clusters_separate(self):
        detections = sidecar(
            [
                ("direct", 14, 0.1, 2),
                ("direct", 14, 0.9, 2),
            ]
        )
        means, _ = cluster_mean_probabilities(detections)
        self.assertEqual(len(means), 2)
        self.assertAlmostEqual(float(np.min(means)), 0.1, places=6)
        self.assertAlmostEqual(float(np.max(means)), 0.9, places=6)


if __name__ == "__main__":
    unittest.main()
