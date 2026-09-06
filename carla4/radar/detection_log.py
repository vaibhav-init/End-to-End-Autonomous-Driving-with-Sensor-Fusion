"""Per-scan detection-list logging shared by the collectors and the harness.

The CSV rows the controller pipeline has always written hold one selected
scalar per frame. A point-set controller, the counterfactual ghost test and
the fidelity study all need the full detection list per scan, with the
simulator's provenance labels. That list is variable-length, so it goes in a
compressed NumPy sidecar next to the CSV rather than into it:

    data.csv            ->  data.detections.npz
    s1_fog3_seed42.csv  ->  s1_fog3_seed42.detections.npz

The archive holds ``detections`` (one row per point, ``DETECTION_DTYPE``) and
``frames`` (one row per logged frame, so an empty scan is distinguishable
from a frame that was never logged). ``scan_index`` is the sensor's own
counter; two consecutive frames that carry the same scan index saw the same
sweep, and window builders drop the duplicate.

Labels (``source``, truth ids, bounce family) are for evaluation only. No
consumer that pretends to be a controller may read them.
"""

import numpy as np


DETECTION_DTYPE = np.dtype(
    [
        ("frame", np.int64),
        ("timestamp", np.float64),
        ("scan_index", np.int64),
        ("distance_m", np.float32),
        ("azimuth_rad", np.float32),
        ("relative_velocity_mps", np.float32),
        ("snr_db", np.float32),
        ("source", "S8"),
        ("truth_object_id", np.int64),
        ("truth_parent_object_id", np.int64),
        ("semantic_tag", np.int16),
        ("bounce_type", "S16"),
        ("bounce_order", np.int8),
        ("ghost_probability", np.float32),
    ]
)

FRAME_DTYPE = np.dtype(
    [
        ("frame", np.int64),
        ("timestamp", np.float64),
        ("scan_index", np.int64),
        ("count", np.int32),
    ]
)


def sidecar_path(csv_path):
    """``foo.csv`` -> ``foo.detections.npz`` in the same directory."""

    text = str(csv_path)
    if text.endswith(".csv"):
        text = text[: -len(".csv")]
    return text + ".detections.npz"


def detection_record(frame, timestamp, scan_index, detection):
    return (
        int(frame),
        float("nan") if timestamp is None else float(timestamp),
        int(scan_index),
        float(detection.distance_m),
        float(detection.azimuth_rad),
        float(detection.relative_velocity_mps),
        float(detection.snr_db),
        str(detection.source).encode("ascii", errors="replace"),
        int(detection.truth_object_id),
        int(getattr(detection, "truth_parent_object_id", 0)),
        int(detection.semantic_tag),
        str(getattr(detection, "bounce_type", "direct")).encode(
            "ascii", errors="replace"
        ),
        int(getattr(detection, "bounce_order", 1)),
        float(getattr(detection, "ghost_probability", 0.0)),
    )


class DetectionLog:
    """Accumulate per-frame detection lists and write the sidecar once."""

    def __init__(self):
        self._rows = []
        self._frames = []

    def append(self, frame, timestamp, scan_index, detections):
        detections = tuple(detections)
        self._frames.append(
            (
                int(frame),
                float("nan") if timestamp is None else float(timestamp),
                int(scan_index),
                len(detections),
            )
        )
        for detection in detections:
            self._rows.append(
                detection_record(frame, timestamp, scan_index, detection)
            )

    def append_radar(self, radar, frame):
        """Log the radar's latest scan; a no-op for backends without lists."""

        getter = getattr(radar, "get_detections", None)
        if getter is None:
            return False
        scan = getter()
        self.append(
            frame,
            scan.get("timestamp"),
            scan.get("scan_index", 0),
            scan.get("detections", ()),
        )
        return True

    @property
    def frame_count(self):
        return len(self._frames)

    @property
    def point_count(self):
        return len(self._rows)

    def arrays(self):
        detections = np.asarray(self._rows, dtype=DETECTION_DTYPE)
        frames = np.asarray(self._frames, dtype=FRAME_DTYPE)
        return detections, frames

    def save(self, path):
        detections, frames = self.arrays()
        np.savez_compressed(path, detections=detections, frames=frames)
        return path


def load_detection_log(path):
    """Return ``{"detections": array, "frames": array}`` from a sidecar."""

    with np.load(path, allow_pickle=False) as archive:
        detections = np.asarray(archive["detections"], dtype=DETECTION_DTYPE)
        frames = (
            np.asarray(archive["frames"], dtype=FRAME_DTYPE)
            if "frames" in archive.files
            else np.zeros(0, dtype=FRAME_DTYPE)
        )
    return {"detections": detections, "frames": frames}


def detections_by_frame(detections):
    """Map frame -> structured sub-array, in frame order."""

    if len(detections) == 0:
        return {}
    order = np.argsort(detections["frame"], kind="stable")
    ordered = detections[order]
    frames = ordered["frame"]
    boundaries = np.flatnonzero(frames[1:] != frames[:-1]) + 1
    starts = np.concatenate((np.array((0,), dtype=np.int64), boundaries))
    ends = np.concatenate((boundaries, np.array((len(ordered),), dtype=np.int64)))
    return {
        int(frames[start]): ordered[start:end]
        for start, end in zip(starts, ends)
    }
