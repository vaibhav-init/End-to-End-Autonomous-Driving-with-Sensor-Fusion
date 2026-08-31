from dataclasses import replace
import math
import unittest

from radar.multipath import (
    ReflectorSegment,
    generate_multipath_targets,
    incidence_reflection_loss_db,
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


class MultipathPhysicsTest(unittest.TestCase):
    """Phase 1 realism fixes: Fresnel loss, true-velocity Doppler, fading."""

    def test_reflection_loss_falls_towards_grazing_incidence(self):
        # A dielectric reflects weakly head-on and almost perfectly at
        # grazing, so loss must decrease as the incidence cosine shrinks.
        normal_incidence = incidence_reflection_loss_db(4, 1.0, 3.0)
        oblique = incidence_reflection_loss_db(4, 0.5, 3.0)
        grazing = incidence_reflection_loss_db(4, 0.05, 3.0)
        self.assertGreater(normal_incidence, oblique)
        self.assertGreater(oblique, grazing)
        self.assertGreaterEqual(grazing, 0.0)

    def test_conductor_reflects_better_than_concrete(self):
        guard_rail = incidence_reflection_loss_db(28, 0.5, 3.0)
        concrete = incidence_reflection_loss_db(4, 0.5, 3.0)
        self.assertLess(guard_rail, concrete)

    def test_unknown_material_falls_back_to_stored_loss(self):
        self.assertAlmostEqual(
            incidence_reflection_loss_db(99, 0.5, 7.25),
            7.25,
        )

    def test_tangential_velocity_changes_ghost_doppler(self):
        # A target moving along the wall is almost purely tangential: its
        # radial component is near zero, so a radial-only reconstruction
        # cannot recover the mirrored path's Doppler.
        config = geometry_config()
        along_wall = replace(
            target(),
            relative_velocity_mps=0.1,
            velocity_xy_mps=(6.0, 0.0),
        )
        radial_only = replace(along_wall, velocity_xy_mps=(0.0, 0.0))
        with_vector = generate_multipath_targets([along_wall], [wall()], config)
        without_vector = generate_multipath_targets(
            [radial_only], [wall()], config
        )
        self.assertTrue(with_vector and without_vector)
        paired = {
            (path.bounce_type, path.bounce_order): path.relative_velocity_mps
            for path in with_vector
        }
        baseline = {
            (path.bounce_type, path.bounce_order): path.relative_velocity_mps
            for path in without_vector
        }
        shared = set(paired) & set(baseline)
        self.assertTrue(shared)
        self.assertTrue(
            any(
                abs(paired[key] - baseline[key]) > 0.5
                for key in shared
            ),
            "tangential motion must change the mirrored-path Doppler",
        )

    def test_ghost_fading_is_stochastic_and_correlated(self):
        config = replace(
            geometry_config(),
            multipath_fading_std_db=4.0,
            multipath_fading_correlation=0.85,
        )
        model = RealisticRadarModel(config, seed=7)
        samples = [model._update_ghost_fading(-1234) for _ in range(400)]
        self.assertGreater(max(samples) - min(samples), 1.0)
        mean = sum(samples) / len(samples)
        variance = sum((value - mean) ** 2 for value in samples) / len(samples)
        self.assertGreater(variance, 0.5)
        # Correlated, not white: successive draws should track each other.
        lagged = sum(
            (a - mean) * (b - mean) for a, b in zip(samples, samples[1:])
        ) / (len(samples) - 1)
        self.assertGreater(lagged / max(variance, 1e-9), 0.4)

    def test_fading_is_disabled_when_std_is_zero(self):
        config = replace(geometry_config(), multipath_fading_std_db=0.0)
        model = RealisticRadarModel(config, seed=7)
        samples = [model._update_ghost_fading(-99) for _ in range(20)]
        self.assertTrue(all(value == 0.0 for value in samples))

    def test_reflection_loss_stays_physically_plausible(self):
        # Roughness is measured against a 3.9 mm wavelength: millimetre-scale
        # RMS heights drive the coherent term to hundreds of dB and delete the
        # specular path. Every material must stay within a usable budget at
        # every incidence angle.
        for tag in (3, 4, 5, 20, 26, 28):
            for cosine in (1.0, 0.7, 0.3, 0.05, 0.001):
                loss = incidence_reflection_loss_db(tag, cosine, 3.0)
                self.assertGreaterEqual(loss, 0.0)
                self.assertLess(
                    loss,
                    35.0,
                    f"tag {tag} at cos={cosine} lost {loss:.1f} dB per bounce",
                )
