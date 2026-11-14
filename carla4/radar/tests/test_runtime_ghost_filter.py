"""RuntimeGhostFilter: amplitude gain and expanded-point scoring.

These tests need torch, so they import-fail on the authoring box and run on
the GPU box (same as test_ghost_model).

What they pin down:

* the deployed radar config's export amplitude gain is applied to the SNR
  proxy, so amplitude-dependent features stay inside the training range;
* score_points returns one probability per input point in input order and
  keeps a history of previous scans (the context the temporal model needs);
* filter_points partitions points at the threshold, always accepts clutter,
  and annotates every point with its probability;
* metadata records the gain and the scoring path for provenance.
"""

import math
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from radar.ghost_detection.features import (
    FEATURE_NAMES,
    FEATURE_SCHEMA_VERSION,
    snr_db_to_amplitude,
)
from radar.ghost_detection.model import create_ghost_model
from radar.ghost_detection.runtime import RuntimeGhostFilter
from radar.realistic_core import RadarDetection


def tiny_checkpoint(window_frames=3, max_points=64, threshold=0.5):
    model = create_ghost_model(
        "temporal_pointnet",
        hidden_dim=8,
        context_dim=8,
    )
    return {
        "feature_schema": FEATURE_SCHEMA_VERSION,
        "feature_names": FEATURE_NAMES,
        "model_name": "temporal_pointnet",
        "model_kwargs": {"hidden_dim": 8, "context_dim": 8},
        "model_state": model.state_dict(),
        "threshold": threshold,
        "window_frames": window_frames,
        "max_points": max_points,
        "cycle_time_s": 0.05,
    }


def point(distance_m, azimuth_rad=0.0, velocity=0.0, snr_db=20.0,
          source="direct", object_id=1):
    return RadarDetection(
        distance_m=distance_m,
        azimuth_rad=azimuth_rad,
        relative_velocity_mps=velocity,
        snr_db=snr_db,
        source=source,
        truth_object_id=object_id,
        semantic_tag=14,
    )


class RuntimeGhostFilterTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        path = Path(self.directory.name) / "best_detector.pt"
        import torch

        torch.save(tiny_checkpoint(), path)
        self.path = str(path)
        self.torch = torch

    def tearDown(self):
        self.directory.cleanup()

    def test_amplitude_gain_is_applied(self):
        filt = RuntimeGhostFilter(self.path, amplitude_gain_db=77.32)
        record = filt._point_record(point(snr_db=20.0), 0.0)
        expected = float(
            snr_db_to_amplitude(np.array([20.0]))[0]
        ) * 10.0 ** (77.32 / 20.0)
        self.assertAlmostEqual(record[3], expected, places=3)
        zero_gain = RuntimeGhostFilter(self.path, amplitude_gain_db=0.0)
        plain = zero_gain._point_record(point(snr_db=20.0), 0.0)
        self.assertAlmostEqual(
            plain[3],
            float(snr_db_to_amplitude(np.array([20.0]))[0]),
            places=3,
        )

    def test_score_points_returns_one_probability_per_point(self):
        filt = RuntimeGhostFilter(self.path, amplitude_gain_db=0.0)
        points = [
            point(20.0 + index, azimuth_rad=0.01 * index,
                  velocity=1.0 + 0.1 * index, snr_db=15.0 + index)
            for index in range(9)
        ]
        probabilities = filt.score_points(points, timestamp_s=0.0,
                                           scan_index=1)
        self.assertEqual(len(probabilities), len(points))
        self.assertTrue(np.all(np.isfinite(probabilities)))
        self.assertTrue(np.all((probabilities >= 0) & (probabilities <= 1)))
        self.assertEqual(len(filt._point_history), 1)

    def test_history_feeds_context_on_the_next_scan(self):
        filt = RuntimeGhostFilter(self.path, amplitude_gain_db=0.0)
        first = [point(20.0 + index, snr_db=15.0) for index in range(5)]
        filt.score_points(first, timestamp_s=0.0, scan_index=1)
        second = [point(60.0 + index, snr_db=15.0) for index in range(5)]
        probabilities = filt.score_points(second, timestamp_s=0.05,
                                          scan_index=2)
        self.assertEqual(len(filt._point_history), 2)
        self.assertEqual(len(probabilities), len(second))
        # The ages of the context points come from the timestamps.
        old_timestamp, old_scan, stored = filt._point_history[0]
        self.assertEqual(old_timestamp, 0.0)
        self.assertEqual(old_scan, 1)
        self.assertEqual(len(stored), len(first))

    def test_filter_points_partitions_and_exempts_clutter(self):
        filt = RuntimeGhostFilter(
            self.path,
            threshold=0.5,
            amplitude_gain_db=0.0,
        )
        points = [
            point(20.0, snr_db=15.0, source="direct"),
            point(21.0, snr_db=15.0, source="ghost", object_id=-7),
            point(22.0, snr_db=15.0, source="clutter", object_id=-9),
        ]
        accepted, rejected = filt.filter_points(points, timestamp_s=0.0)
        sources_accepted = [item.source for item in accepted]
        sources_rejected = [item.source for item in rejected]
        self.assertIn("clutter", sources_accepted)
        self.assertNotIn("clutter", sources_rejected)
        for item in accepted + rejected:
            self.assertTrue(0.0 <= item.ghost_probability <= 1.0)
        # Direct and ghost points are partitioned consistently with their
        # scores: nothing is lost, nothing duplicated.
        self.assertEqual(len(accepted) + len(rejected), len(points))
        self.assertEqual(
            sorted(sources_accepted + sources_rejected),
            ["clutter", "direct", "ghost"],
        )

    def test_metadata_records_gain_and_scoring_path(self):
        filt = RuntimeGhostFilter(self.path, amplitude_gain_db=12.5)
        metadata = filt.metadata()
        self.assertEqual(metadata["amplitude_gain_db"], 12.5)
        self.assertEqual(metadata["scoring"], "expanded-points")
        self.assertEqual(metadata["threshold"], 0.5)


if __name__ == "__main__":
    unittest.main()
