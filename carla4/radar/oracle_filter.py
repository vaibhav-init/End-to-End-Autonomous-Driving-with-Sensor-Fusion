"""Ground-truth ghost rejection: the ceiling every learned filter is measured against.

The oracle reads the simulator's ``source`` label and drops every multipath
detection before the tracker sees it. It is not a method, and it could never
run on a real radar. It exists so the closed-loop study has an upper bound:
the best any filter could possibly do at the controller. A learned filter, or
a controller that absorbs ghosts on its own, is judged by how close it lands
to this line, and the scalar controller on unfiltered ghosts sets the floor.
"""

from dataclasses import replace


class OracleGhostFilter:
    """Reject detections whose simulator source is ``ghost``."""

    signature = "oracle"
    threshold = None
    model_name = "oracle"

    def filter_detections(self, detections, timestamp_s=None, scan_index=None):
        accepted = []
        rejected = []
        for detection in detections:
            if detection.source == "ghost":
                rejected.append(replace(detection, ghost_probability=1.0))
            else:
                accepted.append(detection)
        return accepted, rejected

    def metadata(self):
        return {
            "path": None,
            "signature": self.signature,
            "threshold": self.threshold,
            "model_name": self.model_name,
            "window_frames": 0,
            "max_points": 0,
            "feature_schema": None,
        }
