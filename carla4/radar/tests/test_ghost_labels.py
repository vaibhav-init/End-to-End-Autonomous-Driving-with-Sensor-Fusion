import unittest

import numpy as np

from radar.ghost_detection.labels import (
    decode_cmto_label,
    label_id_to_binary_target,
)


class RadarGhostLabelsTest(unittest.TestCase):
    def test_official_real_and_multipath_examples(self):
        self.assertEqual(decode_cmto_label(1111).binary_target, 0)
        self.assertEqual(decode_cmto_label(1011).binary_target, 0)
        self.assertEqual(decode_cmto_label(1112).binary_target, 1)
        self.assertEqual(decode_cmto_label(1124).binary_target, 1)
        self.assertEqual(decode_cmto_label(2126).binary_target, 1)
        self.assertEqual(decode_cmto_label(2000).binary_target, 1)

    def test_background_noise_ignore_and_sketchy_are_not_clean_negatives(self):
        values = np.array((0, -1, -2, -1112), dtype=np.int32)
        np.testing.assert_array_equal(
            label_id_to_binary_target(values),
            np.full(4, -1, dtype=np.int8),
        )

    def test_sketchy_and_undecided_policies_are_explicit(self):
        self.assertEqual(
            decode_cmto_label(-1112, include_sketchy=True).binary_target,
            1,
        )
        self.assertEqual(
            decode_cmto_label(2000, include_undecided=False).binary_target,
            -1,
        )

    def test_invalid_cmto_is_ignored(self):
        self.assertEqual(decode_cmto_label(9999).binary_target, -1)
        self.assertEqual(decode_cmto_label(123).binary_target, -1)


if __name__ == "__main__":
    unittest.main()
