"""Flip direction: did removing points break correct brake calls or repair them?

A flip rate alone only says the controller is sensitive to ghosts. These cases
pin the part that says whether that sensitivity helps or hurts, because that is
what separates "ghosts are noise the filter should remove" from "ghosts are
evidence of a parent object and filtering them costs information".

Needs torch only because the module it lives in imports it; the function under
test is plain numpy.
"""

import unittest

try:
    import torch  # noqa: F401
    HAVE_TORCH = True
except ImportError:  # authoring box has no torch; the GPU box does
    HAVE_TORCH = False

if HAVE_TORCH:
    from counterfactual_ghost_test import _flip_directions


@unittest.skipUnless(HAVE_TORCH, "counterfactual_ghost_test imports torch")
class FlipDirectionTest(unittest.TestCase):
    def test_removing_ghosts_destroys_correct_decisions(self):
        """Ghosts were load-bearing: without them the model stops matching the teacher."""

        result = _flip_directions(
            brake_full=[True, True, True, True],
            brake_variant=[False, False, False, True],
            brake_truth=[True, True, True, True],
        )
        self.assertEqual(result["harmful_flips"], 3)
        self.assertEqual(result["helpful_flips"], 0)
        self.assertEqual(result["net_harmful_flips"], 3)
        self.assertEqual(result["accuracy_before"], 1.0)
        self.assertEqual(result["accuracy_after"], 0.25)

    def test_removing_ghosts_repairs_false_brakes(self):
        """The opposite reading: the ghosts were causing the unnecessary braking."""

        result = _flip_directions(
            brake_full=[True, True, True, False],
            brake_variant=[False, False, False, False],
            brake_truth=[False, False, False, False],
        )
        self.assertEqual(result["harmful_flips"], 0)
        self.assertEqual(result["helpful_flips"], 3)
        self.assertEqual(result["net_harmful_flips"], -3)

    def test_no_flips_leaves_accuracy_unchanged(self):
        result = _flip_directions([True, False], [True, False], [True, False])
        self.assertEqual(result["harmful_flips"], 0)
        self.assertEqual(result["helpful_flips"], 0)
        self.assertEqual(result["accuracy_change"], 0.0)

    def test_accuracy_change_reconciles_with_flip_counts(self):
        """Bookkeeping: accuracy moves by exactly (helpful - harmful) / windows."""

        result = _flip_directions(
            brake_full=[True, False, True, False, True],
            brake_variant=[False, True, True, False, False],
            brake_truth=[True, True, False, False, False],
        )
        expected = result["accuracy_before"] + (
            result["helpful_flips"] - result["harmful_flips"]
        ) / result["windows"]
        self.assertAlmostEqual(result["accuracy_after"], expected, places=12)

    def test_no_windows_is_not_an_error(self):
        """A collection with no ghosts reports nothing rather than dividing by zero."""

        self.assertEqual(_flip_directions([], [], []), {"windows": 0})


if __name__ == "__main__":
    unittest.main()
