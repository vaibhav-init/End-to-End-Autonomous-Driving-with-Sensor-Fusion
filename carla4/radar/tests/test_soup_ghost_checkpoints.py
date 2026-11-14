"""State averaging for checkpoint soups (torch required: GPU box only)."""

import copy
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from soup_ghost_checkpoints import _average_state_dicts


def tiny_state(scale=1.0, seed=0):
    generator = torch.Generator().manual_seed(seed)
    return {
        "weight": torch.randn(4, 4, generator=generator) * scale,
        "bias": torch.randn(4, generator=generator) * scale,
        "num_batches_tracked": torch.tensor(7),
    }


class AverageStateDictsTest(unittest.TestCase):
    def test_uniform_average_of_two_states(self):
        a = tiny_state(seed=1)
        b = tiny_state(seed=2)
        averaged = _average_state_dicts([a, b])
        expected = 0.5 * (a["weight"] + b["weight"])
        self.assertTrue(torch.allclose(averaged["weight"], expected))

    def test_weighted_average(self):
        a = tiny_state(seed=1)
        b = tiny_state(seed=2)
        averaged = _average_state_dicts([a, b], weights=[0.75, 0.25])
        expected = 0.75 * a["weight"] + 0.25 * b["weight"]
        self.assertTrue(torch.allclose(averaged["weight"], expected))
        # The 2-model soup of anchor + finetuned is exactly a WiSE-FT step.
        wise = _average_state_dicts([b, a], weights=[0.5, 0.5])
        self.assertTrue(torch.allclose(wise["weight"], expected * 0 + 0.5 * (a["weight"] + b["weight"])))

    def test_integer_buffers_are_copied_not_averaged(self):
        a = tiny_state(seed=1)
        b = tiny_state(seed=2)
        b["num_batches_tracked"] = torch.tensor(11)
        averaged = _average_state_dicts([a, b])
        self.assertEqual(int(averaged["num_batches_tracked"]), 7)

    def test_original_states_are_not_mutated(self):
        a = tiny_state(seed=1)
        b = tiny_state(seed=2)
        weight_before = a["weight"].clone()
        _average_state_dicts([a, b])
        self.assertTrue(torch.equal(a["weight"], weight_before))


if __name__ == "__main__":
    unittest.main()
