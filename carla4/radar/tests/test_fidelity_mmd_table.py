"""MMD fidelity estimator sanity: same distribution -> ~0, shifted -> >0."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from fidelity_mmd_table import _joint_mmd2, _mmd2, _median_bandwidth, _rbf_gram


class MmdEstimatorTest(unittest.TestCase):
    def test_same_distribution_gives_near_zero_mmd(self):
        rng = np.random.default_rng(7)
        x = rng.normal(0.0, 1.0, size=800)
        y = rng.normal(0.0, 1.0, size=800)
        bandwidth = _median_bandwidth(np.concatenate((x, y)))
        self.assertGreater(bandwidth, 0.0)
        self.assertLess(abs(_mmd2(x, y, bandwidth)), 0.01)

    def test_rbf_gram_is_a_full_pairwise_matrix(self):
        """Regression: b must broadcast as a ROW, not a column.

        With b.reshape(-1, 1) the 'gram' collapsed to an elementwise
        column, which silently turned every per-feature MMD^2 into a
        1/n artifact (all statistics read ~1/n regardless of the domain
        gap). The joint 11-d estimator had its own correct broadcasting,
        so only the per-feature rows were affected.
        """

        x = np.array([0.0, 3.0])
        gram = _rbf_gram(x, x, bandwidth=1.0)
        self.assertEqual(gram.shape, (2, 2))
        self.assertAlmostEqual(float(gram[0, 1]), np.exp(-4.5), places=9)
        self.assertAlmostEqual(float(gram[1, 0]), np.exp(-4.5), places=9)
        self.assertAlmostEqual(float(gram[0, 0]), 1.0, places=9)

    def test_shifted_distribution_gives_positive_mmd(self):
        rng = np.random.default_rng(7)
        x = rng.normal(0.0, 1.0, size=800)
        y = rng.normal(3.0, 1.0, size=800)
        bandwidth = _median_bandwidth(np.concatenate((x, y)))
        mmd_shifted = _mmd2(x, y, bandwidth)
        self.assertGreater(mmd_shifted, 0.05)

    def test_mmd2_is_symmetric(self):
        rng = np.random.default_rng(3)
        x = rng.normal(0.0, 1.0, size=300)
        y = rng.normal(0.5, 1.0, size=300)
        bandwidth = _median_bandwidth(np.concatenate((x, y)))
        self.assertAlmostEqual(
            _mmd2(x, y, bandwidth),
            _mmd2(y, x, bandwidth),
            places=10,
        )

    def test_joint_mmd2_detects_a_shifted_feature(self):
        rng = np.random.default_rng(11)
        x = rng.normal(0.0, 1.0, size=(500, 4))
        y = np.concatenate(
            (rng.normal(0.0, 1.0, size=(500, 3)),
             rng.normal(4.0, 1.0, size=(500, 1))),
            axis=1,
        )
        self.assertGreater(_joint_mmd2(x, y), 0.05)
        y_same = rng.normal(0.0, 1.0, size=(500, 4))
        self.assertLess(_joint_mmd2(x, y_same), 0.05)


if __name__ == "__main__":
    unittest.main()
