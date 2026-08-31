#!/usr/bin/env python3
"""Cross-domain ghost-detector evaluation without the schema gate.

Evaluates ANY trained checkpoint against ANY prepared dataset split,
regardless of which feature schema either was built with. Features are
reconstructed according to the *checkpoint's* schema:

- ``radar_ghost_physical_v1``: the original 8 absolute-amplitude features.
- ``radar_ghost_physical_v2``: the 11 frame-relative features (uses stored
  statistics when the prepared npz provides them, otherwise computes them
  with the shared fast path).

This exists purely for transfer diagnostics ("how does the real-data model
rank old vs new synthetic points?"). It bypasses the production schema gate,
so its outputs are only comparable to gated runs when schemas match.

Usage:
  python3 evaluate_cross_domain.py \
    --checkpoint artifacts/ghost_temporal_official/best_detector.pt \
    --data artifacts/carla_ghost_rgd_densified_prepared \
           artifacts/ghost_carla_zeroshot_v2 \
    --split test \
    --output-dir artifacts/cross_domain_eval
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from radar.ghost_detection.features import (
    frame_context_statistics,
    physical_features,
)
from radar.ghost_detection.metrics import (
    BinaryHistogramMetrics,
    format_all_confusion_matrices,
)
from radar.ghost_detection.model import create_ghost_model


KNOWN_SCHEMAS = ("radar_ghost_physical_v1", "radar_ghost_physical_v2")

STAT_FIELDS = (
    "rel_log_amplitude",
    "doppler_cluster_residual",
    "local_density_ratio",
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--data",
        required=True,
        nargs="+",
        help="one or more prepared dataset directories",
    )
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--window-frames",
        type=int,
        default=None,
        help="override the checkpoint's window size",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=None,
        help="override the checkpoint's point budget",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument(
        "--max-real-fpr",
        type=float,
        default=None,
        help="default: value recorded in the checkpoint arguments",
    )
    parser.add_argument(
        "--limit-samples",
        type=int,
        default=None,
        help="optional cap on evaluated windows per dataset (debugging)",
    )
    parser.add_argument(
        "--cache-sequences",
        type=int,
        default=64,
        help="archives kept per worker; keep above the split's count",
    )
    parser.add_argument(
        "--context-reserve",
        type=float,
        default=0.25,
        help="share of --max-points held for older scans (match training)",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser.parse_args()


def load_checkpoint(path, device):
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    schema = checkpoint.get("feature_schema")
    if schema not in KNOWN_SCHEMAS:
        raise ValueError(f"Unsupported checkpoint schema {schema!r}")
    model_kwargs = dict(checkpoint.get("model_kwargs", {}))
    model_kwargs.setdefault("input_dim", len(checkpoint["feature_names"]))
    model = create_ghost_model(checkpoint["model_name"], **model_kwargs)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()
    return {
        "path": str(Path(path).resolve()),
        "schema": schema,
        "feature_names": tuple(checkpoint["feature_names"]),
        "window_frames": int(checkpoint.get("window_frames", 1)),
        "max_points": int(checkpoint.get("max_points", 1024)),
        "threshold": float(checkpoint.get("threshold", 0.5)),
        "arguments": checkpoint.get("arguments", {}),
        "model": model,
    }


def v1_features(ranges, azimuth, velocity, amplitude, age_s):
    """Reconstruct the original 8-feature v1 contract exactly."""

    amplitudes = np.asarray(amplitude, dtype=np.float32)
    signed_log_amplitude = np.sign(amplitudes) * np.log1p(np.abs(amplitudes))
    return np.stack(
        (
            ranges * np.cos(azimuth) / 100.0,
            ranges * np.sin(azimuth) / 100.0,
            ranges / 100.0,
            np.sin(azimuth),
            np.cos(azimuth),
            velocity / 40.0,
            signed_log_amplitude / 10.0,
            np.clip(age_s / 0.5, 0.0, 4.0),
        ),
        axis=-1,
    ).astype(np.float32, copy=False)


def features_for_schema(schema, ranges, azimuth, velocity, amplitude, age_s,
                        rel_log_amp, doppler_residual, density_ratio):
    if schema == "radar_ghost_physical_v1":
        return v1_features(ranges, azimuth, velocity, amplitude, age_s)
    return physical_features(
        ranges,
        azimuth,
        velocity,
        amplitude,
        age_s,
        relative_log_amplitude=rel_log_amp,
        doppler_cluster_residual=doppler_residual,
        local_density_ratio=density_ratio,
    )


class CrossDomainDataset(Dataset):
    """Windows from a prepared npz set, tolerant of v1/v2 manifests."""

    def __init__(self, root, split, window_frames, max_points,
                 cache_sequences=64, context_reserve_fraction=0.25):
        self.root = Path(root)
        with (self.root / "manifest.json").open("r", encoding="utf-8") as fh:
            self.manifest = json.load(fh)
        self.dataset_schema = self.manifest.get("feature_schema", "unknown")
        if self.dataset_schema not in KNOWN_SCHEMAS:
            print(
                f"[warn] {self.root}: unknown dataset schema "
                f"{self.dataset_schema!r}; raw fields treated as authoritative"
            )
        self.window_frames = int(window_frames)
        self.max_points = int(max_points)
        self.context_reserve_fraction = float(context_reserve_fraction)
        self.sequences = [
            record
            for record in self.manifest.get("sequences", ())
            if record.get("split") == split
        ]
        if not self.sequences:
            raise ValueError(f"No sequences for split {split!r} in {self.root}")
        self.samples = []
        for seq_index, record in enumerate(self.sequences):
            for sensor_text, frames in record.get("sensor_frames", {}).items():
                for position, frame in enumerate(frames):
                    self.samples.append(
                        (seq_index, int(sensor_text), position, int(frame))
                    )
        if not self.samples:
            raise ValueError(f"Split {split!r} has no radar frames")
        self._cache_limit = max(1, int(cache_sequences))
        self._cache = {}

    def __len__(self):
        return len(self.samples)

    def _load_archive(self, seq_index):
        cached = self._cache.get(seq_index)
        if cached is not None:
            return cached
        path = self.root / self.sequences[seq_index]["path"]
        with np.load(path, allow_pickle=False) as archive:
            value = {name: np.copy(archive[name]) for name in archive.files}
        frame_index = {}
        if len(value["frame"]):
            boundaries = np.flatnonzero(
                (value["sensor"][1:] != value["sensor"][:-1])
                | (value["frame"][1:] != value["frame"][:-1])
            ) + 1
            starts = np.concatenate((np.array((0,), dtype=np.int64), boundaries))
            ends = np.concatenate(
                (boundaries, np.array((len(value["frame"]),), dtype=np.int64))
            )
            for start, end in zip(starts, ends):
                key = (int(value["sensor"][start]), int(value["frame"][start]))
                frame_index[key] = (int(start), int(end))
        value["_frame_index"] = frame_index

        # Frame-relative v2 statistics: stored when preparation baked them
        # in; computed on load for older prepared sets.
        if not set(STAT_FIELDS).issubset(value.keys()):
            rel = np.zeros(len(value["frame"]), dtype=np.float32)
            resid = np.zeros(len(value["frame"]), dtype=np.float32)
            density = np.zeros(len(value["frame"]), dtype=np.float32)
            for (_, _), (start, end) in frame_index.items():
                computed = frame_context_statistics(
                    value["r_sc"][start:end],
                    value["phi_sc"][start:end],
                    value["vr_sc"][start:end],
                    value["amp"][start:end],
                )
                rel[start:end] = computed[0]
                resid[start:end] = computed[1]
                density[start:end] = computed[2]
            value["rel_log_amplitude"] = rel
            value["doppler_cluster_residual"] = resid
            value["local_density_ratio"] = density

        while len(self._cache) >= self._cache_limit:
            oldest = next(iter(self._cache))
            self._cache.pop(oldest)
        self._cache[seq_index] = value
        return value

    def __getitem__(self, sample_index):
        seq_index, sensor, position, current_frame = self.samples[sample_index]
        record = self.sequences[seq_index]
        frames_list = record["sensor_frames"][str(sensor)]
        first = max(0, position - self.window_frames + 1)
        window = np.asarray(frames_list[first : position + 1], dtype=np.int64)
        data = self._load_archive(seq_index)

        chunks = []
        for frame in window:
            start, end = data["_frame_index"][(sensor, int(frame))]
            chunks.append(np.arange(start, end, dtype=np.int64))
        indices = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.int64)
        if len(indices) == 0:
            raise RuntimeError(
                f"Empty window at {record['path']} frame {current_frame}"
            )
        current_mask = data["frame"][indices] == current_frame

        # Mirrors PreparedGhostDataset._sample_indices: hold part of the
        # budget for older scans. Without it one dense scan fills the budget
        # alone, `age` collapses to 0 everywhere, and the window stops
        # matching what the checkpoint was trained on.
        current_indices = indices[current_mask]
        context_indices = indices[~current_mask]
        if len(context_indices):
            reserve = min(
                len(context_indices),
                int(round(self.max_points * self.context_reserve_fraction)),
            )
            current_budget = max(1, self.max_points - reserve)
        else:
            current_budget = self.max_points
        if len(current_indices) > current_budget:
            sel = np.linspace(
                0, len(current_indices) - 1, current_budget, dtype=np.int64
            )
            current_indices = current_indices[sel]
        budget = max(self.max_points - len(current_indices), 0)
        if len(context_indices) > budget:
            sel = np.linspace(0, len(context_indices) - 1, budget, dtype=np.int64)
            context_indices = context_indices[sel]
        ordered = np.concatenate((context_indices, current_indices))

        start, end = data["_frame_index"][(sensor, current_frame)]
        current_ts = float(np.median(data["frame_timestamp"][start:end]))
        age_s = np.maximum(0.0, current_ts - data["frame_timestamp"][ordered])

        features = features_for_schema(
            self.checkpoint_schema,
            data["r_sc"][ordered],
            data["phi_sc"][ordered],
            data["vr_sc"][ordered],
            data["amp"][ordered],
            age_s,
            data["rel_log_amplitude"][ordered],
            data["doppler_cluster_residual"][ordered],
            data["local_density_ratio"][ordered],
        )

        count = len(ordered)
        padded = np.zeros((self.max_points, features.shape[-1]), dtype=np.float32)
        point_mask = np.zeros(self.max_points, dtype=np.bool_)
        label_mask = np.zeros(self.max_points, dtype=np.bool_)
        target = np.full(self.max_points, -1.0, dtype=np.float32)
        padded[:count] = features
        point_mask[:count] = True
        label_mask[:count] = (data["target"][ordered] >= 0) & (
            data["frame"][ordered] == current_frame
        )
        target[:count] = data["target"][ordered]
        return {
            "features": torch.from_numpy(padded),
            "point_mask": torch.from_numpy(point_mask),
            "label_mask": torch.from_numpy(label_mask),
            "target": torch.from_numpy(target),
        }


def evaluate_dataset(checkpoint, data_root, args):
    dataset = CrossDomainDataset(
        data_root,
        args.split,
        window_frames=args.window_frames or checkpoint["window_frames"],
        max_points=args.max_points or checkpoint["max_points"],
        cache_sequences=args.cache_sequences,
        context_reserve_fraction=args.context_reserve,
    )
    dataset.checkpoint_schema = checkpoint["schema"]
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    metrics = BinaryHistogramMetrics()
    evaluated = 0
    with torch.no_grad():
        for batch in loader:
            features = batch["features"].to(args.device, non_blocking=True)
            point_mask = batch["point_mask"].to(args.device, non_blocking=True)
            logits = checkpoint["model"](features, point_mask)
            probabilities = torch.sigmoid(logits).cpu().numpy()
            label_masks = batch["label_mask"].numpy()
            targets = batch["target"].numpy()
            for row in range(probabilities.shape[0]):
                selected = label_masks[row]
                if not np.any(selected):
                    continue
                metrics.update(
                    probabilities[row][selected],
                    targets[row][selected].astype(np.int64),
                )
                evaluated += 1
            if args.limit_samples and evaluated >= args.limit_samples:
                break
    max_fpr = (
        float(args.max_real_fpr)
        if args.max_real_fpr is not None
        else float(checkpoint["arguments"].get("max_real_fpr", 0.01))
    )
    result = metrics.compute(
        fixed_threshold=checkpoint["threshold"],
        max_false_positive_rate=max_fpr,
    )
    result.update(
        {
            "data": str(Path(data_root).resolve()),
            "dataset_schema": dataset.dataset_schema,
            "checkpoint": checkpoint["path"],
            "checkpoint_schema": checkpoint["schema"],
            "split": args.split,
            "windows_evaluated": evaluated,
        }
    )
    return result


def main():
    args = parse_args()
    checkpoint = load_checkpoint(args.checkpoint, args.device)
    print(
        f"checkpoint schema={checkpoint['schema']} "
        f"window={args.window_frames or checkpoint['window_frames']} "
        f"points={args.max_points or checkpoint['max_points']} "
        f"threshold={checkpoint['threshold']:.4f} device={args.device}"
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for data_root in args.data:
        try:
            result = evaluate_dataset(checkpoint, data_root, args)
        except ValueError as exc:
            print(f"[skip] {data_root}: {exc}")
            continue
        out_name = Path(data_root).resolve().name + ".json"
        with (output_dir / out_name).open("w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, sort_keys=True)
            handle.write("\n")
        print()
        print(f"--- {Path(data_root).name} ---")
        print(format_all_confusion_matrices(result))
        print()
        print(
            f"{Path(data_root).name}: AUPRC={result['auprc']:.4f} "
            f"AUROC={result['auroc']:.4f} "
            f"recall@thr={result['recall']:.4f} "
            f"FPR@thr={result['false_positive_rate']:.4f} "
            f"windows={result['windows_evaluated']} -> {out_name}"
        )


if __name__ == "__main__":
    main()