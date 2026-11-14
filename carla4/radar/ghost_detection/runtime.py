"""Online checkpoint adapter for RealisticRadarModel detections."""

from collections import deque
from dataclasses import replace
import hashlib
import math

import numpy as np

from .features import (
    FEATURE_NAMES,
    FEATURE_SCHEMA_VERSION,
    frame_context_statistics,
    physical_features,
    snr_db_to_amplitude,
)
from .model import create_ghost_model


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_metadata(path, threshold=None, device="cpu"):
    """Read deployment-relevant metadata without constructing the network."""

    import torch

    try:
        checkpoint = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    if checkpoint.get("feature_schema") != FEATURE_SCHEMA_VERSION:
        raise ValueError("Ghost detector feature schema is incompatible")
    effective_threshold = (
        float(checkpoint.get("threshold", 0.5))
        if threshold is None
        else float(threshold)
    )
    if not 0.0 <= effective_threshold <= 1.0:
        raise ValueError("Ghost rejection threshold must be in [0, 1]")
    return {
        "signature": file_sha256(path)[:16],
        "threshold": effective_threshold,
        "model_name": checkpoint.get("model_name"),
        "window_frames": int(checkpoint.get("window_frames", 1)),
        "max_points": int(checkpoint.get("max_points", 1024)),
        "feature_schema": checkpoint.get("feature_schema"),
    }


class RuntimeGhostFilter:
    """Classify current target-list points using prior scans as context.

    Two scoring paths exist:

    - :meth:`filter_detections` scores *clusters* (one detection per object).
      This is the legacy path and only makes sense for filters that were
      trained on cluster-level data; the released detectors are per-point.
    - :meth:`score_points` / :meth:`filter_points` score the *expanded*
      per-point scan, which is the distribution the exporter writes and the
      detector was trained on. :class:`~radar.realistic_core
      .RealisticRadarModel` routes through this path when it is available.

    ``amplitude_gain_db`` must be the *deployed* radar config's export gain:
    exporters write ``10**(snr_db/20) * 10**(amplitude_gain_db/20)``, so the
    runtime has to reproduce that scale or every amplitude-dependent feature
    is out of the training range (the calibrated profile carries a ~77 dB
    gain; omitting it was the reason the first cross-domain scores sat at
    chance).
    """

    def __init__(
        self,
        checkpoint_path,
        threshold=None,
        device="cpu",
        amplitude_gain_db=0.0,
    ):
        import torch

        self.torch = torch
        self.checkpoint_path = str(checkpoint_path)
        try:
            checkpoint = torch.load(
                self.checkpoint_path,
                map_location=device,
                weights_only=True,
            )
        except TypeError:
            checkpoint = torch.load(self.checkpoint_path, map_location=device)
        if checkpoint.get("feature_schema") != FEATURE_SCHEMA_VERSION:
            raise ValueError(
                "Ghost detector feature schema mismatch: "
                f"{checkpoint.get('feature_schema')!r}"
            )
        if tuple(checkpoint.get("feature_names", ())) != FEATURE_NAMES:
            raise ValueError("Ghost detector feature ordering is incompatible")
        self.model_name = checkpoint["model_name"]
        self.model_kwargs = dict(checkpoint.get("model_kwargs", {}))
        self.model = create_ghost_model(self.model_name, **self.model_kwargs)
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.to(device)
        self.model.eval()
        self.device = device
        self.window_frames = int(checkpoint.get("window_frames", 1))
        self.max_points = int(checkpoint.get("max_points", 1024))
        checkpoint_threshold = float(checkpoint.get("threshold", 0.5))
        self.threshold = (
            checkpoint_threshold if threshold is None else float(threshold)
        )
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError("Ghost rejection threshold must be in [0, 1]")
        self.cycle_time_s = float(checkpoint.get("cycle_time_s", 0.05))
        self.amplitude_gain_db = float(amplitude_gain_db)
        self._amplitude_scale = 10.0 ** (self.amplitude_gain_db / 20.0)
        self.signature = file_sha256(self.checkpoint_path)[:16]
        # window_frames includes the current scan, matching PreparedGhostDataset.
        self._history = deque(maxlen=max(0, self.window_frames - 1))
        self._point_history = deque(maxlen=max(0, self.window_frames - 1))

    @staticmethod
    def _select_evenly(values, count):
        if len(values) <= count:
            return list(values)
        indices = np.linspace(0, len(values) - 1, count, dtype=np.int64)
        return [values[int(index)] for index in indices]

    def _point_record(self, detection, age_s):
        return (
            float(detection.distance_m),
            # CARLA's positive sensor-y points right; the real dataset's
            # automotive convention points left.
            -float(detection.azimuth_rad),
            -float(detection.relative_velocity_mps),
            float(snr_db_to_amplitude(detection.snr_db)) * self._amplitude_scale,
            max(0.0, float(age_s)),
        )

    def filter_detections(self, detections, timestamp_s=None, scan_index=None):
        detections = list(detections)
        timestamp = (
            float(timestamp_s)
            if timestamp_s is not None and math.isfinite(float(timestamp_s))
            else None
        )
        scan = int(scan_index) if scan_index is not None else 0
        context = []
        for old_timestamp, old_scan, old_detections in self._history:
            if timestamp is not None and old_timestamp is not None:
                age_s = max(0.0, timestamp - old_timestamp)
            else:
                age_s = max(0, scan - old_scan) * self.cycle_time_s
            context.extend(
                self._point_record(detection, age_s)
                for detection in old_detections
            )

        current_count = min(len(detections), self.max_points)
        context_capacity = self.max_points - current_count
        context = self._select_evenly(context, context_capacity)
        records = context + [
            self._point_record(detection, 0.0)
            for detection in detections[:current_count]
        ]
        self._history.append((timestamp, scan, tuple(detections)))
        if not records or current_count == 0:
            return detections, []

        values = np.asarray(records, dtype=np.float32)
        rel_log_amp, doppler_residual, density_ratio = frame_context_statistics(
            values[:, 0],
            values[:, 1],
            values[:, 2],
            values[:, 3],
        )
        features = physical_features(
            values[:, 0],
            values[:, 1],
            values[:, 2],
            values[:, 3],
            values[:, 4],
            relative_log_amplitude=rel_log_amp,
            doppler_cluster_residual=doppler_residual,
            local_density_ratio=density_ratio,
        )
        padded = np.zeros(
            (1, self.max_points, len(FEATURE_NAMES)),
            dtype=np.float32,
        )
        point_mask = np.zeros((1, self.max_points), dtype=np.bool_)
        padded[0, : len(features)] = features
        point_mask[0, : len(features)] = True
        with self.torch.no_grad():
            logits = self.model(
                self.torch.from_numpy(padded).to(self.device),
                self.torch.from_numpy(point_mask).to(self.device),
            )
            probabilities = self.torch.sigmoid(logits)[0].cpu().numpy()
        current_start = len(context)
        current_probabilities = probabilities[
            current_start : current_start + current_count
        ]

        accepted = []
        rejected = []
        for index, detection in enumerate(detections):
            probability = (
                float(current_probabilities[index])
                if index < current_count
                else 0.0
            )
            annotated = replace(detection, ghost_probability=probability)
            # The released real labels supervise real-vs-multipath, not the
            # synthetic unstructured-clutter process. Keep clutter for the
            # existing tracker rather than applying an out-of-domain label.
            if detection.source != "clutter" and probability >= self.threshold:
                rejected.append(annotated)
            else:
                accepted.append(annotated)
        return accepted, rejected

    def score_points(self, points, timestamp_s=None, scan_index=None):
        """Score one *expanded* point scan; return probabilities in input order.

        This mirrors the training/export distribution: the detector was
        trained on per-point features whose frame-relative statistics are
        computed over the full expanded frame (``--points-source sensor``),
        so the runtime must score expanded points, not one cluster per
        object. History keeps the previous scans' point lists; context
        capacity is shared with the current scan exactly as in
        :meth:`filter_detections`.
        """

        points = list(points)
        timestamp = (
            float(timestamp_s)
            if timestamp_s is not None and math.isfinite(float(timestamp_s))
            else None
        )
        scan = int(scan_index) if scan_index is not None else 0
        context = []
        for old_timestamp, old_scan, old_points in self._point_history:
            if timestamp is not None and old_timestamp is not None:
                age_s = max(0.0, timestamp - old_timestamp)
            else:
                age_s = max(0, scan - old_scan) * self.cycle_time_s
            context.extend(
                self._point_record(point, age_s) for point in old_points
            )

        current_count = min(len(points), self.max_points)
        context_capacity = self.max_points - current_count
        context = self._select_evenly(context, context_capacity)
        records = context + [
            self._point_record(point, 0.0) for point in points[:current_count]
        ]
        self._point_history.append(
            (timestamp, scan, tuple(self._select_evenly(points, self.max_points)))
        )
        if not records or current_count == 0:
            return np.zeros(len(points), dtype=np.float64)

        values = np.asarray(records, dtype=np.float32)
        rel_log_amp, doppler_residual, density_ratio = frame_context_statistics(
            values[:, 0],
            values[:, 1],
            values[:, 2],
            values[:, 3],
        )
        features = physical_features(
            values[:, 0],
            values[:, 1],
            values[:, 2],
            values[:, 3],
            values[:, 4],
            relative_log_amplitude=rel_log_amp,
            doppler_cluster_residual=doppler_residual,
            local_density_ratio=density_ratio,
        )
        padded = np.zeros(
            (1, self.max_points, len(FEATURE_NAMES)),
            dtype=np.float32,
        )
        point_mask = np.zeros((1, self.max_points), dtype=np.bool_)
        padded[0, : len(features)] = features
        point_mask[0, : len(features)] = True
        with self.torch.no_grad():
            logits = self.model(
                self.torch.from_numpy(padded).to(self.device),
                self.torch.from_numpy(point_mask).to(self.device),
            )
            probabilities = self.torch.sigmoid(logits)[0].cpu().numpy()
        current_start = len(context)
        result = np.zeros(len(points), dtype=np.float64)
        result[:current_count] = probabilities[
            current_start : current_start + current_count
        ]
        return result

    def filter_points(self, points, timestamp_s=None, scan_index=None):
        """Score an expanded point scan and partition it.

        Clutter points are always accepted, matching
        :meth:`filter_detections`: the released labels supervise
        real-vs-multipath, not the synthetic unstructured-clutter process.
        Returns ``(accepted, rejected)`` with ``ghost_probability`` set.
        """

        points = list(points)
        probabilities = self.score_points(
            points,
            timestamp_s=timestamp_s,
            scan_index=scan_index,
        )
        accepted = []
        rejected = []
        for point, probability in zip(points, probabilities):
            annotated = replace(point, ghost_probability=float(probability))
            if point.source != "clutter" and probability >= self.threshold:
                rejected.append(annotated)
            else:
                accepted.append(annotated)
        return accepted, rejected

    def metadata(self):
        return {
            "path": self.checkpoint_path,
            "signature": self.signature,
            "threshold": self.threshold,
            "model_name": self.model_name,
            "window_frames": self.window_frames,
            "max_points": self.max_points,
            "feature_schema": FEATURE_SCHEMA_VERSION,
            "amplitude_gain_db": self.amplitude_gain_db,
            "scoring": "expanded-points",
        }
