"""The range-azimuth rasteriser the CNN controller consumes.

The design claim is that a ghost sits at the same bearing as its parent at a
longer range, so in this grid the pair is a pure offset along one axis, which
is what a convolution kernel can learn. These pin that down, plus the point
budget being gone and the counterfactual's point removal.
"""

import math
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from cnn_controller import (  # noqa: E402
    AGE_SLOTS,
    AZIMUTH_BINS,
    CHANNELS_PER_SLOT,
    GRID_CHANNELS,
    RANGE_BINS,
    build_window_grid,
    rasterize,
    window_points,
)
from transformer_controller import SOURCE_CODES  # noqa: E402


def point(range_m, azimuth_deg, vr=-3.0, snr=30.0, source="direct"):
    return SimpleNamespace(
        distance_m=float(range_m),
        azimuth_rad=math.radians(azimuth_deg),
        relative_velocity_mps=float(vr),
        snr_db=float(snr),
        source=source,
    )


class RasterTest(unittest.TestCase):
    def test_shape_and_empty_window(self):
        grid = rasterize([], [], [], [], [])
        self.assertEqual(grid.shape, (GRID_CHANNELS, RANGE_BINS, AZIMUTH_BINS))
        self.assertEqual(float(np.abs(grid).sum()), 0.0)

    def test_a_point_lands_in_its_own_cell(self):
        grid = rasterize([30.5], [0.0], [-5.0], [0.0], [1.0])
        occupancy = grid[0]
        self.assertEqual(int((occupancy != 0).sum()), 1)
        # 30.5 m -> range bin 30; azimuth 0 deg -> the middle bin.
        self.assertGreater(occupancy[30, AZIMUTH_BINS // 2], 0.0)

    def test_ghost_and_parent_differ_only_along_the_range_axis(self):
        """The relation a convolution is meant to pick up."""
        parent = point(30.0, 4.0)
        ghost = point(31.7, 4.0, source="ghost")   # real second-order offset ~1.7 m
        r, az, vr, age, rel, _src = window_points([(0.0, [parent, ghost])])
        occupancy = rasterize(r, az, vr, age, rel)[0]
        cells = np.argwhere(occupancy != 0)
        self.assertEqual(len(cells), 2)
        (r1, a1), (r2, a2) = sorted(map(tuple, cells))
        self.assertEqual(a1, a2)              # same bearing column
        self.assertEqual(r2 - r1, 1)          # one 1 m range bin apart

    def test_age_goes_into_separate_channel_groups(self):
        now = rasterize([20.0], [0.0], [-3.0], [0.0], [1.0])
        old = rasterize([20.0], [0.0], [-3.0], [0.3], [1.0])
        self.assertGreater(now[0].sum(), 0.0)
        self.assertEqual(float(old[0].sum()), 0.0)
        self.assertGreater(old[3 * CHANNELS_PER_SLOT].sum(), 0.0)

    def test_points_outside_the_grid_are_dropped(self):
        grid = rasterize(
            [150.0, 40.0, 40.0], [0.0, math.radians(85.0), math.radians(-85.0)],
            [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0],
        )
        self.assertEqual(float(grid[0].sum()), 0.0)

    def test_keep_mask_removes_only_those_points(self):
        pts = [point(30.0, 0.0), point(45.0, -6.0, source="ghost")]
        r, az, vr, age, rel, src = window_points([(0.0, pts)])
        full = rasterize(r, az, vr, age, rel)
        without = rasterize(r, az, vr, age, rel, keep=src != SOURCE_CODES["ghost"])
        self.assertEqual(int((full[0] != 0).sum()), 2)
        self.assertEqual(int((without[0] != 0).sum()), 1)
        # The surviving cell is untouched, so a prediction change is attributable.
        self.assertTrue(np.array_equal(full[0][full[0] != 0][:1], without[0][without[0] != 0][:1]))

    def test_every_point_survives_there_is_no_budget(self):
        rng = np.random.default_rng(0)
        crowd = [point(rng.uniform(5, 95), rng.uniform(-60, 60)) for _ in range(1200)]
        built = build_window_grid([(0.0, crowd)], 12.0)
        self.assertEqual(built["point_count"], 1200)

    def test_drop_sources_reaches_the_grid(self):
        pts = [point(30.0, 0.0), point(31.7, 0.0, source="ghost")]
        full = build_window_grid([(0.0, pts)], 10.0)
        clean = build_window_grid([(0.0, pts)], 10.0, drop_sources=(SOURCE_CODES["ghost"],))
        self.assertEqual(full["point_count"], 2)
        self.assertEqual(clean["point_count"], 1)
        self.assertGreater(float(np.abs(full["grid"] - clean["grid"]).sum()), 0.0)

    def test_ego_speed_is_normalised_onto_the_window(self):
        built = build_window_grid([(0.0, [point(30.0, 0.0)])], 20.0)
        self.assertAlmostEqual(float(built["ego"]), 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
