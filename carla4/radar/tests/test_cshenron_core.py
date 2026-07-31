import math
import unittest

import numpy as np

from radar.cshenron_core import (
    CARLA_0916_SEMANTIC_TAGS,
    CShenronConfig,
    Material,
    SEMANTIC_LIDAR_DTYPE,
    cshenron_return_power,
    extract_targets,
    map_semantic_materials,
    semantic_material_name,
    semantic_tag_name,
)


def make_returns(rows):
    result = np.zeros(len(rows), dtype=SEMANTIC_LIDAR_DTYPE)
    for index, (x, y, z, cosine, object_id, tag) in enumerate(rows):
        result[index] = (x, y, z, cosine, object_id, tag)
    return result


class CShenronCoreTest(unittest.TestCase):
    def test_carla_0916_semantic_contract(self):
        self.assertEqual(SEMANTIC_LIDAR_DTYPE.itemsize, 24)
        self.assertEqual(CARLA_0916_SEMANTIC_TAGS[0], "Unlabeled")
        self.assertEqual(CARLA_0916_SEMANTIC_TAGS[14], "Car")
        self.assertEqual(CARLA_0916_SEMANTIC_TAGS[28], "GuardRail")
        self.assertEqual(semantic_tag_name(99), "Unknown(99)")
        self.assertEqual(semantic_material_name(12), "HUMAN")
        self.assertEqual(semantic_material_name(14), "METAL")
        self.assertEqual(semantic_material_name(25), "CONCRETE")

    def test_carla_0916_material_mapping(self):
        tags = np.array((12, 1, 9, 14, 255))
        mapped = map_semantic_materials(tags)
        self.assertEqual(
            mapped.tolist(),
            [
                Material.HUMAN,
                Material.CONCRETE,
                Material.WOOD,
                Material.METAL,
                Material.UNLABELLED,
            ],
        )

    def test_forward_vehicle_becomes_target_and_road_is_ignored(self):
        rows = [
            (25.0, -0.4, 0.0, 1.0, 77, 14),
            (25.1, 0.0, 0.1, 1.0, 77, 14),
            (25.0, 0.4, -0.1, 1.0, 77, 14),
            (8.0, 0.0, -1.0, 1.0, 0, 1),
        ]
        targets = extract_targets(make_returns(rows))
        self.assertEqual(len(targets), 1)
        self.assertEqual(targets[0].object_id, 77)
        self.assertEqual(targets[0].semantic_tag, 14)
        self.assertAlmostEqual(targets[0].distance_m, 25.0, places=1)

    def test_out_of_cone_vehicle_is_rejected(self):
        rows = [
            (20.0, 5.0, 0.0, 1.0, 8, 14),
            (20.1, 5.0, 0.0, 1.0, 8, 14),
        ]
        targets = extract_targets(make_returns(rows))
        self.assertEqual(targets, [])

    def test_nearest_signal_qualified_target_sorts_first(self):
        rows = [
            (40.0, -0.2, 0.0, 1.0, 4, 14),
            (40.0, 0.2, 0.0, 1.0, 4, 14),
            (18.0, -0.2, 0.0, 1.0, 9, 14),
            (18.0, 0.2, 0.0, 1.0, 9, 14),
        ]
        config = CShenronConfig(min_snr_db=-20.0)
        targets = extract_targets(make_returns(rows), config)
        self.assertEqual([target.object_id for target in targets], [9, 4])

    def test_extended_target_angle_uses_robust_cluster_centroid(self):
        rows = [
            (10.0, 3.0, 0.0, 1.0, 77, 14),
            (12.0, -0.1, 0.0, 1.0, 77, 14),
            (12.0, 0.0, 0.0, 1.0, 77, 14),
            (12.0, 0.1, 0.0, 1.0, 77, 14),
        ]
        config = CShenronConfig(
            horizontal_fov_deg=40.0,
            min_snr_db=-40.0,
        )
        target = extract_targets(make_returns(rows), config)[0]
        angle_deg = math.degrees(
            math.atan2(target.direction[1], target.direction[0])
        )
        self.assertLess(abs(angle_deg), 1.0)

    def test_static_surfaces_are_split_into_stable_range_angle_cells(self):
        rows = [
            (10.0, 0.0, 0.0, 1.0, 500, 4),
            (10.1, 0.0, 0.0, 1.0, 500, 4),
            (30.0, 0.0, 0.0, 1.0, 500, 4),
            (30.1, 0.0, 0.0, 1.0, 500, 4),
        ]
        config = CShenronConfig(
            horizontal_fov_deg=20.0,
            min_snr_db=-40.0,
        )
        first = extract_targets(make_returns(rows), config)
        second = extract_targets(make_returns(rows), config)
        self.assertEqual(len(first), 2)
        self.assertTrue(all(target.object_id < 0 for target in first))
        self.assertEqual(
            [target.object_id for target in first],
            [target.object_id for target in second],
        )

    def test_return_power_decays_with_range(self):
        config = CShenronConfig()
        powers = cshenron_return_power(
            np.array((10.0, 20.0)),
            np.array((1.0, 1.0)),
            np.array((Material.METAL, Material.METAL)),
            config,
        )
        self.assertGreater(powers[0], powers[1])
        self.assertAlmostEqual(powers[0] / powers[1], 4.0, places=5)


if __name__ == "__main__":
    unittest.main()
