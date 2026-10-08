"""Unit tests for the realism metrics module."""

import json
import os
import tempfile
import unittest

import numpy as np

from carla4.radar.realism_metrics import (
    AZIMUTH_GATE_RAD,
    RADIAL_VELOCITY_GATE_MPS,
    RANGE_GATE_M,
    RealismSummary,
    Scan,
    _greedy_associate,
    compare_summaries,
    _l1_distance,
    summarize_scans,
)


def _scan(dets, tgts):
    dets = np.asarray(dets, dtype=float).reshape((-1, 3)) if len(dets) else np.zeros((0, 3))
    tgts = np.asarray(tgts, dtype=float).reshape((-1, 3)) if len(tgts) else np.zeros((0, 3))
    return Scan(detections=dets, targets=tgts)


class GreedyAssociationTest(unittest.TestCase):
    def test_perfect_detector_matches_everything(self):
        tgts = [[10.0, 0.1, 5.0], [30.0, -0.3, -2.0]]
        dets = [[10.0, 0.1, 5.0], [30.0, -0.3, -2.0]]
        stats = _greedy_associate(np.asarray(dets), np.asarray(tgts))
        self.assertEqual(stats.matched_delta.shape[0], 2)
        self.assertEqual(stats.unmatched_detections.shape[0], 0)
        self.assertTrue(np.allclose(stats.matched_delta, 0.0))

    def test_out_of_gate_detection_stays_unmatched(self):
        tgts = [[10.0, 0.0, 0.0]]
        # 2 m range error also exceeds the 1.5 m gate.
        dets = [[12.0, 0.0, 0.0]]
        stats = _greedy_associate(np.asarray(dets), np.asarray(tgts))
        self.assertEqual(stats.matched_delta.shape[0], 0)
        self.assertEqual(stats.unmatched_detections.shape[0], 1)

    def test_velocity_gate_prevents_cross_match(self):
        tgts = [[10.0, 0.0, 3.0]]
        dets = [[10.0, 0.0, -3.0]]  # 6 m/s apart, at the gate edge
        stats = _greedy_associate(np.asarray(dets), np.asarray(tgts))
        self.assertEqual(stats.matched_delta.shape[0], 0)

    def test_each_target_matched_once(self):
        tgts = [[10.0, 0.0, 0.0]]
        dets = [[10.0, 0.0, 0.0], [10.4, 0.1, 0.2]]
        stats = _greedy_associate(np.asarray(dets), np.asarray(tgts))
        self.assertEqual(stats.matched_delta.shape[0], 1)
        self.assertEqual(stats.unmatched_detections.shape[0], 1)


class SummarizeTest(unittest.TestCase):
    def test_summary_fields_and_json_roundtrip(self):
        scans = []
        rng = np.random.default_rng(7)
        for _ in range(20):
            tgts = np.column_stack([
                rng.uniform(5, 90, 3),
                rng.uniform(-0.5, 0.5, 3),
                rng.uniform(-20, 20, 3),
            ])
            dets = tgts + rng.normal(0, [0.05, 0.01, 0.08], size=tgts.shape)
            dets = np.vstack((dets, [[60.0, 0.9, 0.0]]))  # one ghost per scan
            scans.append(_scan(dets, tgts))
        summary = summarize_scans(scans)
        self.assertEqual(summary.n_scans, 20)
        self.assertAlmostEqual(summary.detections_per_scan_mean, 4.0, places=6)
        self.assertAlmostEqual(summary.ghosts_per_scan_mean, 1.0, places=6)
        self.assertEqual(summary.n_matched, 60)
        self.assertLess(summary.range_error_std_m, 0.2)
        # Ghost lands in the 50-75 m bin.
        self.assertGreater(summary.ghost_range_counts[2], 0)
        self.assertEqual(summary.velocity_misassignment_rate, 0.0)

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "summary.json")
            summary.to_json(path)
            loaded = RealismSummary.from_json(path)
            self.assertEqual(loaded.n_scans, 20)
            self.assertEqual(loaded.pd_counts, summary.pd_counts)

    def test_pd_counts_distinguish_matched_targets(self):
        scans = [
            _scan([[10.0, 0.0, 0.0]], [[10.0, 0.0, 0.0]]),          # seen and detected
            _scan([[10.0, 0.0, 0.0]], [[40.0, 0.0, 0.0]]),          # target unanswered
        ]
        summary = summarize_scans(scans)
        detected = [row[1] for row in summary.pd_counts]
        total = sum(sum(row) for row in summary.pd_counts)
        self.assertEqual(total, 3)  # 2 seen + 1 detected
        self.assertEqual(sum(detected), 1)

    def test_empty_sequence_is_safe(self):
        summary = summarize_scans([])
        self.assertEqual(summary.n_scans, 0)
        self.assertEqual(sum(sum(row) for row in summary.pd_counts), 0)


class CompareTest(unittest.TestCase):
    def test_identical_summaries_distance_is_zero(self):
        d_1 = [3, 3, 4, 3, 3, 3, 3, 3, 3, 3]
        d_2 = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
        summary_a = summarize_scans([_scan([[10.0, 0.0, 0.0]] * 3, [[10.0, 0.0, 0.0]] * 3)] * 5
                                    + [_scan([[20.0, 0.0, 0.0]] * 2, [[20.0, 0.0, 0.0]] * 2)] * 5)
        report = compare_summaries(summary_a, summary_a)
        self.assertEqual(report["detections_per_scan"]["hist_l1"], 0.0)
        self.assertEqual(report["statistics"]["range_error_ks"], 0.0)
        self.assertEqual(report["detections_per_scan"]["candidate_mean"],
                         report["detections_per_scan"]["reference_mean"])

    def test_ks_on_raw_samples(self):
        rng = np.random.default_rng(1)
        scans_a = []
        scans_b = []
        for _ in range(50):
            tgt = [10.0, 0.0, 0.0]
            det_a = [10.0 + rng.normal(0, 0.05), 0.0, 0.0]
            det_b = [10.0 + rng.normal(0, 0.5), 0.0, 0.0]
            scans_a.append(_scan([det_a], [tgt]))
            scans_b.append(_scan([det_b], [tgt]))
        report = compare_summaries(summarize_scans(scans_a), summarize_scans(scans_b))
        ks = report["statistics"].get("range_error_ks")
        self.assertIsInstance(ks, float)
        self.assertGreater(abs(ks), 0.3)  # clearly different spreads

    def test_hist_distance_helper(self):
        self.assertEqual(_l1_distance([1, 1], [1, 1]), 0.0)
        self.assertEqual(_l1_distance([0, 2], [2, 0]), 1.0)

    def test_gate_constants_are_documented_values(self):
        self.assertEqual(RANGE_GATE_M, 1.5)
        self.assertAlmostEqual(AZIMUTH_GATE_RAD, 0.1396263, places=4)
        self.assertEqual(RADIAL_VELOCITY_GATE_MPS, 6.0)


if __name__ == "__main__":
    unittest.main()
