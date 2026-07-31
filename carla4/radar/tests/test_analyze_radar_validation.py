import math
import unittest

from analyze_radar_validation import _wrong_selection_reason


CONFIG = {
    "minimum_track_confidence": 0.0,
    "minimum_forward_distance_m": 1.0,
    "path_half_width_m": 1.8,
    "path_width_growth_per_m": 0.004,
}


def _lead_debug(lead_azimuth=0.0):
    lead = {
        "truth_object_id": 7,
        "semantic_tag": 14,
        "distance_m": 20.0,
        "azimuth_rad": lead_azimuth,
    }
    return {
        "ideal_targets": [
            {
                "object_id": 7,
                "semantic_tag": 14,
                "distance_m": 20.0,
                "azimuth_rad": lead_azimuth,
            }
        ],
        "generated_detections": [lead.copy()],
        "delivered_detections": [lead.copy()],
        "tracks": [
            {
                **lead,
                "confirmed": True,
                "confidence": 1.0,
            }
        ],
        "selected": {
            "track_id": 9,
            "truth_object_id": -10,
            "semantic_tag": 4,
            "distance_m": 12.0,
            "azimuth_rad": 0.0,
            "source": "direct",
        },
    }


class RadarValidationForensicsTest(unittest.TestCase):
    def test_classifies_closer_static_competitor(self):
        reason, _, _ = _wrong_selection_reason(
            _lead_debug(),
            lead_id=7,
            config=CONFIG,
        )
        self.assertEqual(reason, "closer_direct_Wall")

    def test_classifies_lead_track_outside_path(self):
        reason, _, _ = _wrong_selection_reason(
            _lead_debug(lead_azimuth=math.radians(10.0)),
            lead_id=7,
            config=CONFIG,
        )
        self.assertEqual(reason, "lead_track_outside_path_corridor")


if __name__ == "__main__":
    unittest.main()
