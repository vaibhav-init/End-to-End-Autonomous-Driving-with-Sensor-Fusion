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


def _even_indices(count, select):
    """Deterministic evenly-spaced subset of ``range(count)``.

    Same convention as PreparedGhostDataset's current-scan selection and
    CrossDomainDataset's deterministic context sampling.
    """

    if select >= count:
        return np.arange(count, dtype=np.int64)
    if select <= 0:
        return np.zeros(0, dtype=np.int64)
    return np.linspace(0, count - 1, select, dtype=np.int64)


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
        arguments = checkpoint.get("arguments") or {}
        self.context_reserve_fraction = float(
            arguments.get("context_reserve", 0.25)
        )
        self.signature = file_sha256(self.checkpoint_path)[:16]
        # window_frames includes the current scan, matching PreparedGhostDataset.
        # One store serves both scoring paths: each entry is
        # (timestamp, scan_index, records (n x 5), per-scan stats).
        self._point_history = deque(maxlen=max(0, self.window_frames - 1))

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

    @staticmethod
    def _scan_features(values):
        """Feature rows for ONE scan's records with per-scan statistics.

        Training features are per-frame: prepare_radar_ghost_dataset.py
        computes the v2 frame-relative statistics over each point's own
        full frame, and PreparedGhostDataset indexes them per point. The
        runtime must therefore compute the statistics over the point's own
        scan only — pooling the context window into one statistic set
        (the previous behaviour) mixes the same object's returns across
        scans, shrinks the Doppler-residual cue and dilutes the amplitude
        median, which is a train/deploy distribution mismatch.

        Returns ``(features, stats)`` with ``stats`` cached so the history
        scan can be rebuilt with fresh ages later without recomputing.
        """

        if len(values) == 0:
            empty = np.zeros(0, dtype=np.float32)
            return (
                np.zeros((0, len(FEATURE_NAMES)), dtype=np.float32),
                (empty.copy(), empty.copy(), empty.copy()),
            )
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
        return features, (rel_log_amp, doppler_residual, density_ratio)

    def _context_rows(self, timestamp, scan):
        """Feature rows for every stored history scan, ages recomputed."""

        rows = []
        for old_timestamp, old_scan, old_values, old_stats in self._point_history:
            if timestamp is not None and old_timestamp is not None:
                age_s = max(0.0, timestamp - old_timestamp)
            else:
                age_s = max(0, scan - old_scan) * self.cycle_time_s
            ages = np.full(len(old_values), age_s, dtype=np.float32)
            rows.append(
                physical_features(
                    old_values[:, 0],
                    old_values[:, 1],
                    old_values[:, 2],
                    old_values[:, 3],
                    ages,
                    relative_log_amplitude=old_stats[0],
                    doppler_cluster_residual=old_stats[1],
                    local_density_ratio=old_stats[2],
                )
            )
        if not rows:
            return np.zeros((0, len(FEATURE_NAMES)), dtype=np.float32)
        return np.concatenate(rows, axis=0)

    def _score_records(self, current_values, timestamp, scan):
        """Score one scan's records; return probabilities in input order.

        Context comes from the stored history scans (each carrying its own
        per-scan statistics and a recomputed age). The point-budget split
        mirrors PreparedGhostDataset: context is first reserved a fixed
        share (``context_reserve_fraction``) of ``max_points`` and the
        current scan fills the remainder — current-first splitting starves
        the context on dense frames and silently disables the age feature.
        """

        current_features, current_stats = self._scan_features(current_values)
        context_all = self._context_rows(timestamp, scan)
        reserve = min(
            len(context_all),
            int(round(self.max_points * self.context_reserve_fraction)),
        )
        current_capacity = self.max_points - reserve

        # History is stored AFTER scoring so the current scan never appears
        # in its own context.
        self._point_history.append(
            (timestamp, scan, current_values, current_stats)
        )

        current_count = min(len(current_values), current_capacity)
        if current_count == 0:
            return np.zeros(len(current_values), dtype=np.float64)
        context_indices = _even_indices(len(context_all), reserve)
        current_indices = _even_indices(len(current_features), current_count)
        rows = np.concatenate(
            (
                context_all[context_indices],
                current_features[current_indices],
            ),
            axis=0,
        )
        padded = np.zeros(
            (1, self.max_points, len(FEATURE_NAMES)),
            dtype=np.float32,
        )
        point_mask = np.zeros((1, self.max_points), dtype=np.bool_)
        padded[0, : len(rows)] = rows
        point_mask[0, : len(rows)] = True
        with self.torch.no_grad():
            logits = self.model(
                self.torch.from_numpy(padded).to(self.device),
                self.torch.from_numpy(point_mask).to(self.device),
            )
            probabilities = self.torch.sigmoid(logits)[0].cpu().numpy()
        result = np.zeros(len(current_values), dtype=np.float64)
        current_start = len(context_indices)
        result[current_indices] = probabilities[
            current_start : current_start + len(current_indices)
        ]
        return result

    def filter_detections(self, detections, timestamp_s=None, scan_index=None):
        """Legacy cluster-level path (per-scan statistics, like training)."""

        detections = list(detections)
        timestamp = (
            float(timestamp_s)
            if timestamp_s is not None and math.isfinite(float(timestamp_s))
            else None
        )
        scan = int(scan_index) if scan_index is not None else 0
        records = [self._point_record(detection, 0.0) for detection in detections]
        values = np.asarray(records, dtype=np.float32) if records else np.zeros(
            (0, 5), dtype=np.float32
        )
        probabilities = self._score_records(values, timestamp, scan)

        accepted = []
        rejected = []
        # threshold >= 1.0 is the annotate-only mode used by
        # calibrate_runtime_threshold.py: every point is scored and
        # annotated, nothing is rejected — otherwise exactly-saturated
        # probabilities (p == 1.0) would be dropped even in score-only
        # runs and silently shift the deployment trajectory.
        reject_active = self.threshold < 1.0
        for index, detection in enumerate(detections):
            probability = float(probabilities[index])
            annotated = replace(detection, ghost_probability=probability)
            # The released real labels supervise real-vs-multipath, not the
            # synthetic unstructured-clutter process. Keep clutter for the
            # existing tracker rather than applying an out-of-domain label.
            if (
                reject_active
                and detection.source != "clutter"
                and probability >= self.threshold
            ):
                rejected.append(annotated)
            else:
                accepted.append(annotated)
        return accepted, rejected

    def score_points(self, points, timestamp_s=None, scan_index=None):
        """Score one *expanded* point scan; return probabilities in input order.

        This mirrors the training/export distribution: the detector was
        trained on per-point features whose frame-relative statistics are
        computed over the point's own full expanded frame
        (``--points-source sensor``), so the runtime scores expanded points
        with per-scan statistics — never one cluster per object, and never
        statistics pooled across the context window.
        """

        points = list(points)
        timestamp = (
            float(timestamp_s)
            if timestamp_s is not None and math.isfinite(float(timestamp_s))
            else None
        )
        scan = int(scan_index) if scan_index is not None else 0
        records = [self._point_record(point, 0.0) for point in points]
        values = np.asarray(records, dtype=np.float32) if records else np.zeros(
            (0, 5), dtype=np.float32
        )
        return self._score_records(values, timestamp, scan)

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
        # threshold >= 1.0 is annotate-only: score and annotate everything,
        # reject nothing (see filter_detections).
        reject_active = self.threshold < 1.0
        for point, probability in zip(points, probabilities):
            annotated = replace(point, ghost_probability=float(probability))
            if (
                reject_active
                and point.source != "clutter"
                and probability >= self.threshold
            ):
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
            "context_reserve_fraction": self.context_reserve_fraction,
            "scoring": "expanded-points",
            "statistics": "per-scan",
        }
