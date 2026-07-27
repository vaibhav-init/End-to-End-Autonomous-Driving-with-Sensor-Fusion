import unittest

import numpy as np

from radar.cshenron_core import (
    CShenronConfig,
    Material,
    SEMANTIC_LIDAR_DTYPE,
    cshenron_return_power,
    extract_targets,
    map_semantic_materials,
)


def make_returns(rows):
    result = np.zeros(len(rows), dtype=SEMANTIC_LIDAR_DTYPE)
    for index, (x, y, z, cosine, object_id, tag) in enumerate(rows):
        result[index] = (x, y, z, cosine, object_id, tag)
    return result


class CShenronCoreTest(unittest.TestCase):
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
