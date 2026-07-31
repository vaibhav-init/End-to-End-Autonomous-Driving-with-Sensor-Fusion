from dataclasses import replace
import math
import unittest

from radar.realistic_core import (
    IdealRadarTarget,
    RadarEnvironment,
    RealisticRadarModel,
    load_realistic_radar_config,
    realistic_radar_config_signature,
)


def ideal_target(
    object_id=10,
    distance_m=30.0,
    lateral_m=0.0,
    relative_velocity_mps=4.0,
    snr_db=40.0,
    semantic_tag=14,
):
    return IdealRadarTarget(
        object_id=object_id,
        semantic_tag=semantic_tag,
        distance_m=distance_m,
        azimuth_rad=math.atan2(lateral_m, distance_m),
        relative_velocity_mps=relative_velocity_mps,
        snr_db=snr_db,
        point_count=4,
    )


class RealisticRadarCoreTest(unittest.TestCase):
    def test_ideal_profile_is_exact_and_filters_adjacent_lane(self):
        config = load_realistic_radar_config("ideal_target_list_v1")
        model = RealisticRadarModel(config, seed=2)
        output = model.step(
            [
                ideal_target(object_id=1, distance_m=32.0),
                ideal_target(object_id=2, distance_m=18.0, lateral_m=3.6),
            ]
        )
        self.assertEqual(output.truth_object_id, 1)
        self.assertEqual(output.distance_m, 32.0)
        self.assertEqual(output.relative_velocity_mps, 4.0)

    def test_default_tracker_applies_latency_and_m_of_n_confirmation(self):
        base = load_realistic_radar_config("generic_lrr_v1")
        config = replace(
            base,
            min_detection_probability=1.0,
            max_detection_probability=1.0,
            dropout_enter_probability=0.0,
            false_alarms_per_scan=0.0,
            interference_enter_probability=0.0,
            ghost_start_probability=0.0,
            range_noise_floor_m=0.0,
            range_noise_snr_scale_m=0.0,
            doppler_noise_floor_mps=0.0,
            doppler_noise_snr_scale_mps=0.0,
            azimuth_noise_floor_deg=0.0,
            azimuth_noise_snr_scale_deg=0.0,
            snr_fluctuation_std_db=0.0,
            error_correlation=0.0,
        )
        model = RealisticRadarModel(config, seed=3)
        target = ideal_target()

        self.assertEqual(model.step([target]).track_id, 0)
        self.assertEqual(model.step([target]).track_id, 0)
        self.assertNotEqual(model.step([target]).track_id, 0)
        self.assertEqual(model.diagnostics()["confirmed_track_count"], 1)

    def test_correlated_dropout_can_remove_an_entire_burst(self):
        base = load_realistic_radar_config("ideal_target_list_v1")
        config = replace(
            base,
            dropout_enter_probability=1.0,
            dropout_exit_probability=0.0,
            dropout_detection_scale=0.0,
            deletion_misses=3,
        )
        model = RealisticRadarModel(config, seed=4)
        outputs = [model.step([ideal_target()]) for _ in range(5)]
        self.assertTrue(all(output.track_id == 0 for output in outputs))
        self.assertEqual(model.diagnostics()["dropped_direct_count"], 1)

    def test_seed_reproduces_complete_temporal_sequence(self):
        config = load_realistic_radar_config("generic_lrr_v1")
        first = RealisticRadarModel(config, seed=19)
        second = RealisticRadarModel(config, seed=19)
        target = ideal_target(snr_db=12.0)
        environment = RadarEnvironment(
            precipitation=0.8,
            wetness=1.0,
            fog=0.7,
        )

        first_outputs = []
        second_outputs = []
        for frame in range(30):
            timestamp = frame * config.cycle_time_s
            first_outputs.append(
                first.step([target], timestamp, environment)
            )
            second_outputs.append(
                second.step([target], timestamp, environment)
            )
        self.assertEqual(first_outputs, second_outputs)
        self.assertEqual(first.diagnostics(), second.diagnostics())

    def test_persistent_ghost_is_generated_and_reported(self):
        base = load_realistic_radar_config("ideal_target_list_v1")
        config = replace(
            base,
            ghost_start_probability=1.0,
            ghost_survival_probability=1.0,
            max_active_ghosts=1,
            ghost_min_range_bias_m=5.0,
            ghost_max_range_bias_m=5.0,
            ghost_snr_loss_db=0.0,
            deletion_misses=3,
        )
        model = RealisticRadarModel(config, seed=5)
        model.step([ideal_target()])
        first_diagnostics = model.diagnostics()
        self.assertEqual(first_diagnostics["active_ghost_count"], 1)
        self.assertEqual(first_diagnostics["ghost_detection_count"], 1)

        model.step([])
        second_diagnostics = model.diagnostics()
        self.assertEqual(second_diagnostics["active_ghost_count"], 1)
        self.assertEqual(second_diagnostics["ghost_detection_count"], 1)

    def test_signature_changes_with_sensor_distribution(self):
        config = load_realistic_radar_config("generic_lrr_v1")
        changed = replace(config, false_alarms_per_scan=0.5)
        self.assertNotEqual(
            realistic_radar_config_signature(config),
            realistic_radar_config_signature(changed),
        )

    def test_invalid_fractional_scan_count_fails_during_config_load(self):
        with self.assertRaisesRegex(ValueError, "latency_scans must be an integer"):
            load_realistic_radar_config(
                config={"latency_scans": 1.5},
            )

    def test_debug_capture_contains_targets_detections_and_tracks(self):
        config = load_realistic_radar_config("ideal_target_list_v1")
        model = RealisticRadarModel(
            config,
            seed=6,
            capture_debug=True,
        )
        model.step([ideal_target(object_id=88)])
        snapshot = model.debug_snapshot()
        self.assertEqual(snapshot["ideal_targets"][0]["object_id"], 88)
        self.assertEqual(
            snapshot["delivered_detections"][0]["truth_object_id"],
            88,
        )
        self.assertEqual(snapshot["tracks"][0]["truth_object_id"], 88)
        self.assertEqual(snapshot["selected"]["semantic_tag"], 14)


if __name__ == "__main__":
    unittest.main()
