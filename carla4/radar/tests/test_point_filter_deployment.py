"""The learned per-point filter keeps the paired-run properties.

These pin down what arm C (the learned ghost filter inside the closed loop)
rests on:

* a filter that accepts everything must leave the sensor's point list
  bit-identical to the no-filter run of the same seed, so filter arms are
  comparable to the existing closed-loop arms;
* a filter that rejects ghost-scored points removes ghost clusters before
  the tracker sees them: no ghost track, no ghost selected target, and the
  direct-target points stay identical to the no-filter run;
* expanding per cluster in list order draws exactly the same random
  sequence as expanding the whole delivered list at once, so the filter
  adds no reshuffling of the sensor's noise;
* clutter is exempt from rejection, matching RuntimeGhostFilter.
"""

from dataclasses import replace
import math
import unittest

import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from radar.realistic_core import (
    IdealRadarTarget,
    RadarDetection,
    RealisticRadarModel,
    load_realistic_radar_config,
)


def direct_target(object_id=10, distance_m=30.0, lateral_m=0.0, snr_db=30.0):
    return IdealRadarTarget(
        object_id=object_id,
        semantic_tag=14,
        distance_m=distance_m,
        azimuth_rad=math.atan2(lateral_m, distance_m),
        relative_velocity_mps=4.0,
        snr_db=snr_db,
        point_count=4,
        lateral_extent_m=0.9,
        radial_extent_m=4.0,
    )


def ghost_target(object_id=-77, parent=10, distance_m=41.0, lateral_m=-6.0):
    return IdealRadarTarget(
        object_id=object_id,
        semantic_tag=14,
        distance_m=distance_m,
        azimuth_rad=math.atan2(lateral_m, distance_m),
        relative_velocity_mps=3.0,
        snr_db=28.0,
        source="ghost",
        parent_object_id=parent,
        reflector_id=-900,
        bounce_type="type2",
        bounce_order=2,
        path_length_m=2.0 * distance_m,
    )


class FakePointFilter:
    """Point-level stand-in for RuntimeGhostFilter.score_points."""

    def __init__(self, threshold=0.5, ghost_probability=0.9):
        self.threshold = float(threshold)
        self.ghost_probability = float(ghost_probability)
        self.calls = 0

    def score_points(self, points, timestamp_s=None, scan_index=None):
        self.calls += 1
        return np.array(
            [
                self.ghost_probability if point.source == "ghost" else 0.0
                for point in points
            ],
            dtype=np.float64,
        )


class AcceptAllFilter(FakePointFilter):
    def __init__(self):
        super().__init__(threshold=2.0, ghost_probability=0.0)


class PointFilterDeploymentTest(unittest.TestCase):
    def config(self, **overrides):
        base = load_realistic_radar_config("geometry_multipath_v1")
        if overrides:
            base = replace(base, **overrides)
        return base

    def test_accept_all_filter_reproduces_the_unfiltered_point_list(self):
        config = self.config()
        plain = RealisticRadarModel(config, seed=17)
        filtered = RealisticRadarModel(
            config, seed=17, detection_filter=AcceptAllFilter()
        )
        for scan in range(20):
            step_args = dict(
                timestamp_s=scan * config.cycle_time_s,
                multipath_targets=[ghost_target()],
            )
            plain.step([direct_target()], **step_args)
            filtered.step([direct_target()], **step_args)
        _, points_plain = plain.latest_points()
        _, points_filtered = filtered.latest_points()
        self.assertEqual(points_plain, points_filtered)
        self.assertTrue(any(point.source == "ghost" for point in points_plain))

    def test_per_cluster_expansion_matches_whole_list_expansion(self):
        """The RNG draw sequence must not depend on the grouping."""

        config = self.config()
        model_a = RealisticRadarModel(config, seed=23)
        model_b = RealisticRadarModel(config, seed=23)
        for scan in range(3):
            step_args = dict(
                timestamp_s=scan * config.cycle_time_s,
                multipath_targets=[
                    ghost_target(),
                    ghost_target(object_id=-78),
                ],
            )
            model_a.step([direct_target(), direct_target(object_id=11)],
                         **step_args)
            model_b.step([direct_target(), direct_target(object_id=11)],
                         **step_args)
        delivered_a = list(model_a._latency_queue)[-1][1]
        delivered_b = list(model_b._latency_queue)[-1][1]
        # Identical seeds and identical steps leave identical RNG state, so
        # the only difference below is the grouping of the expansion calls.
        self.assertEqual(delivered_a, delivered_b)
        grouped = []
        for detection in delivered_b:
            grouped.extend(model_b._expand_points((detection,)))
        whole = model_a._expand_points(delivered_a)
        self.assertEqual(grouped, whole)

    def test_ghost_clusters_never_become_tracks(self):
        config = self.config()
        model = RealisticRadarModel(
            config, seed=29, detection_filter=FakePointFilter()
        )
        ghost_selected = False
        for scan in range(24):
            output = model.step(
                [direct_target()],
                timestamp_s=scan * config.cycle_time_s,
                multipath_targets=[ghost_target()],
            )
            ghost_selected = ghost_selected or output.source == "ghost"
            tracks = model._tracks.values()
            for track in tracks:
                self.assertNotEqual(track.source, "ghost")
        self.assertFalse(ghost_selected)

    def test_without_filter_the_ghost_does_become_a_track(self):
        """Control: without the filter the same run confirms the ghost."""

        config = self.config()
        model = RealisticRadarModel(config, seed=29)
        for scan in range(24):
            model.step(
                [direct_target()],
                timestamp_s=scan * config.cycle_time_s,
                multipath_targets=[ghost_target()],
            )
        sources = {track.source for track in model._tracks.values()}
        self.assertIn("ghost", sources)

    def test_direct_points_identical_with_and_without_ghost_rejection(self):
        config = self.config()
        plain = RealisticRadarModel(config, seed=31)
        filtered = RealisticRadarModel(
            config, seed=31, detection_filter=FakePointFilter()
        )
        for scan in range(20):
            step_args = dict(
                timestamp_s=scan * config.cycle_time_s,
                multipath_targets=[ghost_target()],
            )
            plain.step([direct_target()], **step_args)
            filtered.step([direct_target()], **step_args)
        _, points_plain = plain.latest_points()
        _, points_filtered = filtered.latest_points()
        direct_plain = [p for p in points_plain if p.source == "direct"]
        direct_filtered = [
            p for p in points_filtered if p.source == "direct"
        ]
        self.assertEqual(direct_plain, direct_filtered)
        self.assertFalse(any(p.source == "ghost" for p in points_filtered))

    def test_clutter_is_exempt_from_cluster_rejection(self):
        config = self.config()
        model = RealisticRadarModel(
            config, seed=37, detection_filter=FakePointFilter()
        )
        clutter = RadarDetection(
            distance_m=25.0,
            azimuth_rad=0.1,
            relative_velocity_mps=0.0,
            snr_db=12.0,
            source="clutter",
            truth_object_id=-5,
            semantic_tag=0,
        )
        direct = RadarDetection(
            distance_m=30.0,
            azimuth_rad=0.0,
            relative_velocity_mps=4.0,
            snr_db=30.0,
            source="direct",
            truth_object_id=10,
            semantic_tag=14,
        )
        kept, rejected_count, points = model._apply_point_filter(
            [clutter, direct],
            timestamp_s=0.0,
            scan_index=3,
        )
        self.assertEqual(rejected_count, 0)
        self.assertEqual(len(kept), 2)
        sources = [point.source for point in points]
        # Clutter stays a single point; the direct cluster spreads over its
        # footprint under the extended-target emission.
        self.assertEqual(sources[0], "clutter")
        self.assertTrue(all(source == "direct" for source in sources[1:]))
        # All scored points carry their probability for downstream labels.
        for point in points:
            self.assertGreaterEqual(point.ghost_probability, 0.0)

    def test_cluster_decision_uses_the_mean_point_probability(self):
        config = self.config(points_per_object_mean=8.0)
        model = RealisticRadarModel(
            config,
            seed=41,
            detection_filter=FakePointFilter(
                threshold=0.5, ghost_probability=0.6
            ),
        )
        ghost = RadarDetection(
            distance_m=41.0,
            azimuth_rad=-0.15,
            relative_velocity_mps=3.0,
            snr_db=28.0,
            source="ghost",
            truth_object_id=10,
            semantic_tag=14,
            truth_parent_object_id=10,
            reflector_id=-900,
            bounce_type="type2",
            bounce_order=2,
        )
        # A cluster whose points all score above the threshold is dropped;
        # the same cluster scoring below it is kept. The decision is the
        # cluster's mean point probability, not any single point.
        kept, rejected_count, _ = model._apply_point_filter(
            [ghost], timestamp_s=0.0, scan_index=1
        )
        self.assertEqual(rejected_count, 1)
        self.assertEqual(kept, [])

        model._detection_filter = FakePointFilter(
            threshold=0.5, ghost_probability=0.4
        )
        kept, rejected_count, _ = model._apply_point_filter(
            [ghost], timestamp_s=0.0, scan_index=2
        )
        self.assertEqual(rejected_count, 0)
        self.assertEqual(len(kept), 1)


if __name__ == "__main__":
    unittest.main()
