"""Ghost injection is a knob outside the sensor identity, and it is paired.

These pin down the properties the closed-loop study rests on:

* the same sensor with ghosts on and off shares one config signature, so a
  clean-trained controller can be deployed against ghosts;
* the direct-target detections of a ghosts-on run are bit-identical to the
  ghosts-off run of the same seed, so every difference is attributable;
* the runtime knobs thin and strengthen ghosts and nothing else;
* the extended-target emission produces several points per object while the
  tracker still sees one cluster;
* the oracle filter removes exactly the labelled ghosts.
"""

from dataclasses import replace
import math
import unittest

import numpy as np

from radar.extended_target import expand_detection
from radar.oracle_filter import OracleGhostFilter
from radar.realistic_core import (
    GHOST_INJECTION_FIELDS,
    IdealRadarTarget,
    RadarDetection,
    RealisticRadarModel,
    ghost_injection_dict,
    load_realistic_radar_config,
    realistic_radar_config_signature,
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


def run_model(config, seed, ghosts, scans=40):
    """Return (direct detections per scan, total ghost detections)."""

    model = RealisticRadarModel(config, seed=seed, capture_debug=True)
    direct_by_scan = []
    ghost_total = 0
    for scan in range(scans):
        model.step(
            [direct_target(), direct_target(object_id=11, distance_m=55.0, lateral_m=1.0)],
            timestamp_s=scan * config.cycle_time_s,
            multipath_targets=[ghost_target()] if ghosts else None,
        )
        snapshot = model.debug_snapshot()
        direct_by_scan.append(
            tuple(
                (d["truth_object_id"], d["distance_m"], d["azimuth_rad"],
                 d["relative_velocity_mps"], round(d["snr_db"], 9))
                for d in snapshot["generated_detections"]
                if d["source"] != "ghost"
            )
        )
        ghost_total += model.diagnostics()["ghost_detection_count"]
    return direct_by_scan, ghost_total


class SignatureTest(unittest.TestCase):
    def test_clean_and_geometry_profiles_share_a_signature(self):
        clean = load_realistic_radar_config("realistic_clean_v1")
        geometry = load_realistic_radar_config("geometry_multipath_v1")
        self.assertEqual(
            realistic_radar_config_signature(clean),
            realistic_radar_config_signature(geometry),
        )
        self.assertNotEqual(
            ghost_injection_dict(clean)["multipath_mode"],
            ghost_injection_dict(geometry)["multipath_mode"],
        )

    def test_ghost_knobs_do_not_change_the_signature_but_noise_does(self):
        base = load_realistic_radar_config("geometry_multipath_v1")
        knobbed = replace(base, ghost_rate_scale=0.25, ghost_snr_offset_db=9.0)
        self.assertEqual(
            realistic_radar_config_signature(base),
            realistic_radar_config_signature(knobbed),
        )
        noisier = replace(base, range_noise_floor_m=0.5)
        self.assertNotEqual(
            realistic_radar_config_signature(base),
            realistic_radar_config_signature(noisier),
        )
        for name in ("multipath_mode", "ghost_rate_scale", "profile_name"):
            self.assertIn(name, GHOST_INJECTION_FIELDS)
        self.assertNotIn("range_noise_floor_m", GHOST_INJECTION_FIELDS)
        self.assertNotIn("points_per_object_mean", GHOST_INJECTION_FIELDS)

    def test_overrides_apply_on_top_of_a_profile(self):
        config = load_realistic_radar_config(
            "realistic_clean_v1",
            overrides={"multipath_mode": "geometry", "ghost_snr_offset_db": 6.0},
        )
        self.assertEqual(config.multipath_mode, "geometry")
        self.assertEqual(config.ghost_snr_offset_db, 6.0)
        with self.assertRaises(ValueError):
            load_realistic_radar_config(overrides={"ghost_rate_scale": 1.5})


class PairedRunTest(unittest.TestCase):
    def test_direct_detections_identical_with_and_without_ghosts(self):
        config = replace(
            load_realistic_radar_config("geometry_multipath_v1"),
            emit_extended_points=False,
        )
        without, ghosts_off = run_model(config, seed=11, ghosts=False)
        with_ghosts, ghosts_on = run_model(config, seed=11, ghosts=True)
        self.assertEqual(without, with_ghosts)
        # And the ghost run really did contain ghosts, or the test is vacuous.
        self.assertEqual(ghosts_off, 0)
        self.assertGreater(ghosts_on, 0)

    def test_direct_points_identical_with_extended_emission(self):
        config = load_realistic_radar_config("geometry_multipath_v1")
        model_a = RealisticRadarModel(config, seed=5)
        model_b = RealisticRadarModel(config, seed=5)
        for scan in range(30):
            model_a.step([direct_target()], timestamp_s=scan * 0.05)
            model_b.step(
                [direct_target()],
                timestamp_s=scan * 0.05,
                multipath_targets=[ghost_target()],
            )
        _, points_a = model_a.latest_points()
        _, points_b = model_b.latest_points()
        direct_a = [p for p in points_a if p.source == "direct"]
        direct_b = [p for p in points_b if p.source == "direct"]
        self.assertEqual(direct_a, direct_b)
        self.assertTrue(any(p.source == "ghost" for p in points_b))


class GhostKnobTest(unittest.TestCase):
    def _ghost_counts(self, **knobs):
        config = replace(
            load_realistic_radar_config("geometry_multipath_v1"),
            emit_extended_points=False,
            **knobs,
        )
        model = RealisticRadarModel(config, seed=3)
        total = 0
        for scan in range(300):
            model.step(
                [direct_target()],
                timestamp_s=scan * 0.05,
                multipath_targets=[ghost_target(), ghost_target(object_id=-78, distance_m=44.0)],
            )
            total += model.diagnostics()["ghost_detection_count"]
        return total

    def test_rate_scale_thins_and_zero_removes(self):
        full = self._ghost_counts()
        thinned = self._ghost_counts(ghost_rate_scale=0.3)
        none = self._ghost_counts(ghost_rate_scale=0.0)
        self.assertGreater(full, thinned)
        self.assertGreater(thinned, 0)
        self.assertEqual(none, 0)

    def test_snr_offset_changes_visibility(self):
        weak = self._ghost_counts(ghost_snr_offset_db=-40.0)
        strong = self._ghost_counts(ghost_snr_offset_db=30.0)
        self.assertLess(weak, strong)


class ExtendedPointsTest(unittest.TestCase):
    def test_tracker_sees_one_cluster_while_points_spread(self):
        config = load_realistic_radar_config("ideal_target_list_v1")
        config = replace(config, points_per_object_mean=12.0)
        model = RealisticRadarModel(config, seed=9)
        model.step([direct_target()])
        diagnostics = model.diagnostics()
        self.assertEqual(diagnostics["delivered_detection_count"], 1)
        self.assertEqual(diagnostics["active_track_count"], 1)
        self.assertGreater(diagnostics["point_detection_count"], 3)
        _, points = model.latest_points()
        ranges = [p.distance_m for p in points]
        self.assertGreater(max(ranges) - min(ranges), 0.5)
        self.assertTrue(all(p.truth_object_id == 10 for p in points))
        self.assertTrue(all(p.source == "direct" for p in points))

    def test_static_clusters_stay_single_and_ghost_points_scale(self):
        config = replace(
            load_realistic_radar_config("geometry_multipath_v1"),
            points_per_object_mean=20.0,
            ghost_points_scale=0.1,
            ghost_snr_offset_db=30.0,
            min_detection_probability=1.0,
            max_detection_probability=1.0,
            dropout_enter_probability=0.0,
            interference_enter_probability=0.0,
            false_alarms_per_scan=0.0,
            latency_scans=0,
        )
        model = RealisticRadarModel(config, seed=12)
        wall = replace(direct_target(object_id=-500, distance_m=12.0), semantic_tag=4)
        direct_points = ghost_points = wall_points = 0
        for scan in range(40):
            model.step(
                [direct_target(), wall],
                timestamp_s=scan * 0.05,
                multipath_targets=[ghost_target()],
            )
            _, points = model.latest_points()
            direct_points += sum(p.source == "direct" and p.semantic_tag == 14 for p in points)
            ghost_points += sum(p.source == "ghost" for p in points)
            wall_points += sum(p.semantic_tag == 4 and p.source == "direct" for p in points)
        self.assertEqual(wall_points, 40)
        self.assertGreater(direct_points, 40 * 10)
        self.assertGreater(ghost_points, 0)
        self.assertLess(ghost_points, direct_points / 4)

    def test_emission_can_be_switched_off(self):
        config = replace(
            load_realistic_radar_config("ideal_target_list_v1"),
            emit_extended_points=False,
        )
        model = RealisticRadarModel(config, seed=9)
        model.step([direct_target()])
        self.assertEqual(model.diagnostics()["point_detection_count"], 1)

    def test_pedestrian_points_carry_more_micro_doppler_than_car(self):
        rng_ped = np.random.default_rng(2)
        rng_car = np.random.default_rng(2)
        base = RadarDetection(
            distance_m=20.0,
            azimuth_rad=0.05,
            relative_velocity_mps=-1.4,
            snr_db=30.0,
            source="direct",
            truth_object_id=1,
            semantic_tag=12,
        )
        common = dict(
            mean_points=40.0,
            range_resolution_m=0.0,
            doppler_resolution_mps=0.0,
            azimuth_resolution_rad=lambda _az: 0.0,
            minimum_range_m=1.0,
            maximum_range_m=100.0,
        )
        pedestrian = expand_detection(base, rng_ped, **common)
        car = expand_detection(replace(base, semantic_tag=14), rng_car, **common)
        spread = lambda points: float(np.std([p.relative_velocity_mps for p in points]))
        self.assertGreater(spread(pedestrian), 2.0 * spread(car))

    def test_zero_micro_doppler_scale_keeps_bulk_velocity(self):
        rng = np.random.default_rng(4)
        base = RadarDetection(
            distance_m=20.0,
            azimuth_rad=0.0,
            relative_velocity_mps=3.0,
            snr_db=30.0,
            source="direct",
            truth_object_id=1,
            semantic_tag=12,
        )
        points = expand_detection(
            base,
            rng,
            mean_points=20.0,
            range_resolution_m=0.0,
            doppler_resolution_mps=0.0,
            azimuth_resolution_rad=lambda _az: 0.0,
            minimum_range_m=1.0,
            maximum_range_m=100.0,
            micro_doppler_scale=0.0,
        )
        self.assertTrue(all(abs(p.relative_velocity_mps - 3.0) < 1e-9 for p in points))


class OracleFilterTest(unittest.TestCase):
    def test_oracle_removes_exactly_the_ghosts(self):
        config = replace(
            load_realistic_radar_config("geometry_multipath_v1"),
            emit_extended_points=False,
        )
        model = RealisticRadarModel(
            config,
            seed=8,
            capture_debug=True,
            detection_filter=OracleGhostFilter(),
        )
        saw_ghost = False
        for scan in range(60):
            model.step(
                [direct_target()],
                timestamp_s=scan * 0.05,
                multipath_targets=[ghost_target()],
            )
            snapshot = model.debug_snapshot()
            generated_ghosts = [
                d for d in snapshot["generated_detections"] if d["source"] == "ghost"
            ]
            saw_ghost = saw_ghost or bool(generated_ghosts)
            self.assertEqual(
                len(snapshot["rejected_detections"]), len(generated_ghosts)
            )
            self.assertTrue(
                all(d["source"] != "ghost" for d in snapshot["accepted_detections"])
            )
        self.assertTrue(saw_ghost)


if __name__ == "__main__":
    unittest.main()
