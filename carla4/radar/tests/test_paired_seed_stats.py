"""Exact Wilcoxon signed-rank and sign tests against known values."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from paired_seed_stats import (
    bootstrap_ci,
    exact_sign_test,
    exact_wilcoxon_signed_rank,
)


class ExactWilcoxonTest(unittest.TestCase):
    def test_all_positive_n5_gives_the_enumeration_floor(self):
        # 2^5 = 32 assignments; all-positive W+ is the extreme tail.
        p, n = exact_wilcoxon_signed_rank([1.0, 2.0, 3.0, 4.0, 5.0])
        self.assertEqual(n, 5)
        self.assertAlmostEqual(p, 2.0 / 32.0, places=9)

    def test_symmetric_diffs_give_one(self):
        # diffs +1, -1 with equal magnitudes: W+ = 1 either way -> p = 1.
        p, _ = exact_wilcoxon_signed_rank([1.0, -1.0, 2.0, -2.0])
        self.assertAlmostEqual(p, 1.0, places=9)

    def test_known_small_case_n3(self):
        # n=3, all positive: 2/8 = 0.25.
        p, n = exact_wilcoxon_signed_rank([0.5, 1.0, 2.0])
        self.assertEqual(n, 3)
        self.assertAlmostEqual(p, 0.25, places=9)

    def test_zero_diffs_are_dropped(self):
        p_all, n_all = exact_wilcoxon_signed_rank([1.0, 2.0, 0.0, 3.0])
        p_drop, n_drop = exact_wilcoxon_signed_rank([1.0, 2.0, 3.0])
        self.assertEqual(n_all, n_drop)
        self.assertAlmostEqual(p_all, p_drop, places=9)

    def test_ties_get_average_ranks(self):
        # diffs 1, 1, 1: ranks 2,2,2 (all tie); all positive -> floor 2/8.
        p, n = exact_wilcoxon_signed_rank([1.0, 1.0, 1.0])
        self.assertEqual(n, 3)
        self.assertAlmostEqual(p, 0.25, places=9)


class ExactSignTest(unittest.TestCase):
    def test_all_positive_n5(self):
        p, n = exact_sign_test([1.0, 2.0, -0.0 + 3.0, 4.0, 5.0])
        self.assertEqual(n, 5)
        self.assertAlmostEqual(p, 2.0 / 32.0, places=9)

    def test_balanced_signs_give_one(self):
        p, _ = exact_sign_test([1.0, -1.0, 1.0, -1.0])
        self.assertAlmostEqual(p, 1.0, places=9)


class BootstrapTest(unittest.TestCase):
    def test_ci_brackets_the_mean(self):
        diffs = [0.02, 0.03, 0.01, 0.04, 0.02, 0.05, 0.03, 0.02, 0.01, 0.04]
        low, high = bootstrap_ci(diffs, draws=2000, seed=1)
        mean = sum(diffs) / len(diffs)
        self.assertLessEqual(low, mean)
        self.assertLessEqual(mean, high)
        self.assertLess(low, high)


if __name__ == "__main__":
    unittest.main()
