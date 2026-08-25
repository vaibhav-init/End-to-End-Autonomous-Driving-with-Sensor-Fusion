"""Windowed point-set dataset built by prepare_radar_ghost_dataset.py."""

from collections import OrderedDict
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .features import (
    FEATURE_SCHEMA_VERSION,
    frame_context_statistics,
    physical_features,
)


class PreparedGhostDataset(Dataset):
    """One sample per sensor cycle, with prior cycles supplied as context."""

    def __init__(
        self,
        root,
        split,
        window_frames=5,
        max_points=1024,
        augment=False,
        seed=42,
        cache_sequences=4,
    ):
        self.root = Path(root)
        with (self.root / "manifest.json").open("r", encoding="utf-8") as handle:
            self.manifest = json.load(handle)
        if self.manifest.get("feature_schema") != FEATURE_SCHEMA_VERSION:
            raise ValueError(
                "Prepared feature schema is incompatible: "
                f"{self.manifest.get('feature_schema')!r}"
            )
        self.window_frames = int(window_frames)
        self.max_points = int(max_points)
        self.augment = bool(augment)
        self.seed = int(seed)
        self.epoch = 0
        self.cache_sequences = max(1, int(cache_sequences))
        if self.window_frames < 1 or self.max_points < 1:
            raise ValueError("window_frames and max_points must be positive")

        self.sequences = [
            record
            for record in self.manifest.get("sequences", ())
            if record.get("split") == split
        ]
        if not self.sequences:
            raise ValueError(f"No prepared sequences found for split {split!r}")
        self.samples = []
        for sequence_index, record in enumerate(self.sequences):
            for sensor_text, frames in record.get("sensor_frames", {}).items():
                sensor = int(sensor_text)
                for frame_position, frame in enumerate(frames):
                    self.samples.append(
                        (sequence_index, sensor, frame_position, int(frame))
                    )
        if not self.samples:
            raise ValueError(f"Split {split!r} contains no radar frames")
        self._cache = OrderedDict()

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        return len(self.samples)

    def _load_sequence(self, sequence_index):
        if sequence_index in self._cache:
            value = self._cache.pop(sequence_index)
            self._cache[sequence_index] = value
            return value
        path = self.root / self.sequences[sequence_index]["path"]
        with np.load(path, allow_pickle=False) as archive:
            value = {name: np.copy(archive[name]) for name in archive.files}
        frame_index = {}
        if len(value["frame"]):
            changes = np.flatnonzero(
                (value["sensor"][1:] != value["sensor"][:-1])
                | (value["frame"][1:] != value["frame"][:-1])
            ) + 1
            starts = np.concatenate((np.array((0,), dtype=np.int64), changes))
            ends = np.concatenate(
                (changes, np.array((len(value["frame"]),), dtype=np.int64))
            )
            for start, end in zip(starts, ends):
                key = (
                    int(value["sensor"][start]),
                    int(value["frame"][start]),
                )
                frame_index[key] = (int(start), int(end))
        value["_frame_index"] = frame_index
        # v2 frame-relative statistics: use the values precomputed at
        # preparation time when present (the normal path); otherwise compute
        # them here once per sequence load (older prepared sets).
        if (
            "rel_log_amplitude" in value
            and "doppler_cluster_residual" in value
            and "local_density_ratio" in value
        ):
            value["_rel_log_amp"] = value["rel_log_amplitude"]
            value["_doppler_residual"] = value["doppler_cluster_residual"]
            value["_density_ratio"] = value["local_density_ratio"]
        else:
            context = {
                "_rel_log_amp": np.zeros(len(value["frame"]), dtype=np.float32),
                "_doppler_residual": np.zeros(
                    len(value["frame"]), dtype=np.float32
                ),
                "_density_ratio": np.zeros(len(value["frame"]), dtype=np.float32),
            }
            for (frame_sensor, _), (start, end) in frame_index.items():
                stats = frame_context_statistics(
                    value["r_sc"][start:end],
                    value["phi_sc"][start:end],
                    value["vr_sc"][start:end],
                    value["amp"][start:end],
                )
                context["_rel_log_amp"][start:end] = stats[0]
                context["_doppler_residual"][start:end] = stats[1]
                context["_density_ratio"][start:end] = stats[2]
            value.update(context)
        self._cache[sequence_index] = value
        while len(self._cache) > self.cache_sequences:
            self._cache.popitem(last=False)
        return value

    @staticmethod
    def _evenly_select(indices, count):
        if len(indices) <= count:
            return indices
        positions = np.linspace(0, len(indices) - 1, count, dtype=np.int64)
        return indices[positions]

    def _sample_indices(self, indices, current_mask, sample_index):
        current = indices[current_mask]
        context = indices[~current_mask]
        if len(current) >= self.max_points:
            return self._evenly_select(current, self.max_points)
        remaining = self.max_points - len(current)
        if len(context) > remaining:
            rng = np.random.default_rng(
                self.seed
                + self.epoch * 1_000_003
                + int(sample_index) * 9_176
            )
            context = np.sort(rng.choice(context, remaining, replace=False))
        return np.concatenate((context, current))

    def __getitem__(self, sample_index):
        sequence_index, sensor, frame_position, current_frame = self.samples[
            sample_index
        ]
        record = self.sequences[sequence_index]
        sensor_frames = record["sensor_frames"][str(sensor)]
        first_position = max(0, frame_position - self.window_frames + 1)
        window_frame_values = np.asarray(
            sensor_frames[first_position : frame_position + 1],
            dtype=np.int64,
        )
        data = self._load_sequence(sequence_index)
        index_chunks = []
        for frame in window_frame_values:
            start, end = data["_frame_index"][(sensor, int(frame))]
            index_chunks.append(np.arange(start, end, dtype=np.int64))
        indices = np.concatenate(index_chunks)
        if len(indices) == 0:
            raise RuntimeError(
                f"Prepared sample has no points: {record['path']} frame {current_frame}"
            )
        current_mask = data["frame"][indices] == current_frame
        indices = self._sample_indices(indices, current_mask, sample_index)
        current_mask = data["frame"][indices] == current_frame

        current_start, current_end = data["_frame_index"][
            (sensor, current_frame)
        ]
        current_timestamps = data["frame_timestamp"][
            current_start:current_end
        ]
        current_timestamp = float(np.median(current_timestamps))
        age_s = np.maximum(
            0.0,
            current_timestamp - data["frame_timestamp"][indices],
        )
        azimuth = data["phi_sc"][indices].astype(np.float32, copy=True)
        amplitude = data["amp"][indices].astype(np.float32, copy=True)
        velocity = data["vr_sc"][indices].astype(np.float32, copy=True)
        if self.augment:
            rng = np.random.default_rng(
                self.seed
                + self.epoch * 2_000_003
                + int(sample_index) * 37
            )
            if rng.random() < 0.5:
                azimuth *= -1.0
            amplitude *= float(np.exp(rng.normal(0.0, 0.08)))
            velocity += rng.normal(0.0, 0.03, size=velocity.shape).astype(
                np.float32
            )

        features = physical_features(
            data["r_sc"][indices],
            azimuth,
            velocity,
            amplitude,
            age_s,
            relative_log_amplitude=data["_rel_log_amp"][indices],
            doppler_cluster_residual=data["_doppler_residual"][indices],
            local_density_ratio=data["_density_ratio"][indices],
        )
        count = len(indices)
        feature_dim = features.shape[-1]
        padded_features = np.zeros(
            (self.max_points, feature_dim),
            dtype=np.float32,
        )
        point_mask = np.zeros(self.max_points, dtype=np.bool_)
        padded_current = np.zeros(self.max_points, dtype=np.bool_)
        target = np.full(self.max_points, -1.0, dtype=np.float32)
        class_id = np.full(self.max_points, -1, dtype=np.int8)
        bounce_type = np.full(self.max_points, -1, dtype=np.int8)
        bounce_order = np.full(self.max_points, -1, dtype=np.int8)
        is_main = np.full(self.max_points, -1, dtype=np.int8)
        label_id = np.zeros(self.max_points, dtype=np.int32)
        label_mask = np.zeros(self.max_points, dtype=np.bool_)
        padded_features[:count] = features
        point_mask[:count] = True
        padded_current[:count] = current_mask
        target[:count] = data["target"][indices]
        class_id[:count] = data["class_id"][indices]
        bounce_type[:count] = data["bounce_type"][indices]
        bounce_order[:count] = data["bounce_order"][indices]
        is_main[:count] = data["is_main"][indices]
        label_id[:count] = data["label_id"][indices]
        label_mask[:count] = current_mask & (data["target"][indices] >= 0)
        return {
            "features": torch.from_numpy(padded_features),
            "point_mask": torch.from_numpy(point_mask),
            "current_mask": torch.from_numpy(padded_current),
            "target": torch.from_numpy(target),
            "class_id": torch.from_numpy(class_id),
            "bounce_type": torch.from_numpy(bounce_type),
            "bounce_order": torch.from_numpy(bounce_order),
            "is_main": torch.from_numpy(is_main),
            "label_id": torch.from_numpy(label_id),
            "label_mask": torch.from_numpy(label_mask),
            "sequence_index": torch.tensor(sequence_index, dtype=torch.int64),
            "sensor": torch.tensor(sensor, dtype=torch.int64),
            "frame": torch.tensor(current_frame, dtype=torch.int64),
        }


def split_label_counts(manifest, split):
    """Return single-counted real/ghost annotations from a manifest."""

    real = 0
    ghost = 0
    for record in manifest.get("sequences", ()):
        if record.get("split") != split:
            continue
        real += int(record.get("real_points", 0))
        ghost += int(record.get("ghost_points", 0))
    return real, ghost
