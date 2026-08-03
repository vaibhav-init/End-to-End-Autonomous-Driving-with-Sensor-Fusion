from dataclasses import replace
import math
import unittest

from radar.multipath import (
    ReflectorSegment,
    generate_multipath_targets,
)
from radar.realistic_core import (
    IdealRadarTarget,
    RealisticRadarModel,
    load_realistic_radar_config,
)


def geometry_config():
    return replace(
        load_realistic_radar_config("ideal_target_list_v1"),
        profile_name="geometry_test",
        multipath_mode="geometry",
        multipath_min_range_separation_m=0.01,
        multipath_max_ghosts_per_target=6,
        multipath_enable_third_order=True,
    )


def target(y_m=2.0):
    x_m = 20.0
    return IdealRadarTarget(
        object_id=77,
        semantic_tag=14,
        distance_m=math.hypot(x_m, y_m),
        azimuth_rad=math.atan2(y_m, x_m),
        relative_velocity_mps=4.0,
        snr_db=45.0,
        lateral_extent_m=0.8,
        parent_object_id=77,
    )


def wall():
    return ReflectorSegment(
        reflector_id=-900,
        semantic_tag=4,
        point_xy_m=(12.5, 5.0),
        tangent_xy=(1.0, 0.0),
        normal_xy=(0.0, -1.0),
        length_m=40.0,
        rms_residual_m=0.02,
        point_count=100,
        reflection_loss_db=3.0,
    )


class MultipathGeometryTest(unittest.TestCase):
    def test_builtin_geometry_profile_is_available(self):
        config = load_realistic_radar_config("geometry_multipath_v1")
        self.assertEqual(config.multipath_mode, "geometry")
        self.assertEqual(config.ghost_start_probability, 0.0)

    def test_image_method_generates_supported_path_families(self):
        paths = generate_multipath_targets(
            [target()],
            [wall()],
            geometry_config(),
        )
        families = {(path.bounce_type, path.bounce_order) for path in paths}
        self.assertEqual(
            families,
            {("type1", 2), ("type2", 2), ("type2", 3)},
        )
        type1 = next(path for path in paths if path.bounce_type == "type1")
        type2 = next(
            path
            for path in paths
            if path.bounce_type == "type2" and path.bounce_order == 2
        )
        third = next(path for path in paths if path.bounce_order == 3)
        self.assertAlmostEqual(type1.azimuth_rad, target().azimuth_rad)
        self.assertAlmostEqual(type1.distance_m, type2.distance_m)
        self.assertGreater(third.distance_m, type2.distance_m)
        self.assertEqual(type1.parent_object_id, 77)
        self.assertEqual(type1.reflector_id, -900)

    def test_ids_and_geometry_are_deterministic(self):
        first = generate_multipath_targets(
            [target()],
            [wall()],
            geometry_config(),
        )
        second = generate_multipath_targets(
            [target()],
            [wall()],
            geometry_config(),
        )
        self.assertEqual(first, second)
        self.assertTrue(all(path.object_id < 0 for path in first))

    def test_reflector_does_not_transmit_through_surface(self):
        paths = generate_multipath_targets(
            [target(y_m=8.0)],
            [wall()],
            geometry_config(),
        )
        self.assertEqual(paths, [])

    def test_model_preserves_path_truth_but_selects_direct_target(self):
        config = geometry_config()
        direct = target()
        paths = generate_multipath_targets([direct], [wall()], config)
        ideal_paths = [
            IdealRadarTarget(
                object_id=path.object_id,
                semantic_tag=path.semantic_tag,
                distance_m=path.distance_m,
                azimuth_rad=path.azimuth_rad,
                relative_velocity_mps=path.relative_velocity_mps,
                snr_db=path.snr_db,
                source="ghost",
                parent_object_id=path.parent_object_id,
                reflector_id=path.reflector_id,
                bounce_type=path.bounce_type,
                bounce_order=path.bounce_order,
                path_length_m=path.path_length_m,
            )
            for path in paths
        ]
        model = RealisticRadarModel(config, seed=3, capture_debug=True)
        selected = model.step([direct], multipath_targets=ideal_paths)
        self.assertEqual(selected.truth_object_id, 77)
        self.assertEqual(model.diagnostics()["ghost_detection_count"], 3)
        snapshot = model.debug_snapshot()
        self.assertEqual(len(snapshot["multipath_ideal_targets"]), 3)
        self.assertTrue(
            all(
                item["truth_parent_object_id"] == 77
                for item in snapshot["generated_detections"]
                if item["source"] == "ghost"
            )
        )

    def test_detection_filter_runs_before_tracking(self):
        class RejectKnownGhosts:
            @staticmethod
            def filter_detections(detections, timestamp_s=None, scan_index=None):
                accepted = [item for item in detections if item.source != "ghost"]
                rejected = [item for item in detections if item.source == "ghost"]
                return accepted, rejected

        config = geometry_config()
        direct = target()
        paths = generate_multipath_targets([direct], [wall()], config)
        ideal_paths = [
            IdealRadarTarget(
                object_id=path.object_id,
                semantic_tag=path.semantic_tag,
                distance_m=path.distance_m,
                azimuth_rad=path.azimuth_rad,
                relative_velocity_mps=path.relative_velocity_mps,
                snr_db=path.snr_db,
                source="ghost",
                parent_object_id=path.parent_object_id,
                reflector_id=path.reflector_id,
                bounce_type=path.bounce_type,
                bounce_order=path.bounce_order,
                path_length_m=path.path_length_m,
            )
            for path in paths
        ]
        model = RealisticRadarModel(
            config,
            seed=4,
            capture_debug=True,
            detection_filter=RejectKnownGhosts(),
        )
        model.step([direct], multipath_targets=ideal_paths)
        diagnostics = model.diagnostics()
        self.assertEqual(diagnostics["generated_detection_count"], 4)
        self.assertEqual(diagnostics["accepted_detection_count"], 1)
        self.assertEqual(diagnostics["rejected_detection_count"], 3)
        self.assertEqual(len(model.debug_snapshot()["rejected_detections"]), 3)


if __name__ == "__main__":
    unittest.main()
