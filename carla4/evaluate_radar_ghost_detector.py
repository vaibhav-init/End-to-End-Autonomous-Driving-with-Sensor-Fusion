#!/usr/bin/env python3
"""Evaluate a radar ghost detector on a prepared sequence split."""

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from radar.ghost_detection.dataset import PreparedGhostDataset
from radar.ghost_detection.features import FEATURE_NAMES, FEATURE_SCHEMA_VERSION
from radar.ghost_detection.metrics import (
    BinaryHistogramMetrics,
    format_all_confusion_matrices,
)
from radar.ghost_detection.model import create_ghost_model


def _flip_features(features):
    """Mirror a point cloud about the sensor boresight.

    Same transform as evaluate_ghost_transfer.py: x (column 0) and
    sin(azimuth) (column 3) flip; everything else is mirror-invariant.
    """

    flipped = features.clone()
    flipped[..., 0] *= -1.0
    flipped[..., 3] *= -1.0
    return flipped


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--output", help="optional metrics JSON path")
    parser.add_argument(
        "--tta",
        action="store_true",
        help=(
            "azimuth-mirror test-time augmentation (average the two "
            "mirrored views). evaluate_ghost_transfer.py dump applies this "
            "by default; enable it here to compare numbers across the two "
            "evaluators on the same protocol"
        ),
    )
    return parser.parse_args()


def evaluate_checkpoint(
    data,
    checkpoint,
    split,
    batch_size=16,
    num_workers=4,
    device="cpu",
    threshold=None,
    tta=False,
):
    """Score one checkpoint on one split and return the full result dict.

    Shared by the command-line evaluator and soup_ghost_checkpoints.py so
    every reported number comes from the same evaluation path.
    """

    dataset = PreparedGhostDataset(
        data,
        split,
        window_frames=int(checkpoint["window_frames"]),
        max_points=int(checkpoint["max_points"]),
        augment=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )
    model = create_ghost_model(
        checkpoint["model_name"],
        **checkpoint.get("model_kwargs", {}),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    metrics = BinaryHistogramMetrics()
    threshold = (
        float(checkpoint.get("threshold", 0.5))
        if threshold is None
        else threshold
    )
    scenario_metrics = {}
    ghost_groups = {}
    real_class_groups = {}

    def update_rate(group, key, predicted_positive):
        counts = group.setdefault(key, [0, 0])
        counts[0] += 1
        counts[1] += int(predicted_positive)

    with torch.no_grad():
        for batch in loader:
            features = batch["features"].to(device, non_blocking=True)
            point_mask = batch["point_mask"].to(device, non_blocking=True)
            logits = model(features, point_mask).cpu()
            probability_matrix = torch.sigmoid(logits).numpy()
            if tta:
                mirrored = model(_flip_features(features), point_mask).cpu()
                probability_matrix = 0.5 * (
                    probability_matrix + torch.sigmoid(mirrored).numpy()
                )
            mask_matrix = batch["label_mask"].numpy()
            target_matrix = batch["target"].numpy()
            class_matrix = batch["class_id"].numpy()
            type_matrix = batch["bounce_type"].numpy()
            order_matrix = batch["bounce_order"].numpy()
            sequence_indices = batch["sequence_index"].numpy()
            for row in range(len(sequence_indices)):
                mask = mask_matrix[row]
                probabilities = probability_matrix[row][mask]
                targets = target_matrix[row][mask]
                metrics.update(probabilities, targets)
                sequence_record = dataset.sequences[int(sequence_indices[row])]
                scenario = sequence_record.get("scenario", sequence_record["name"])
                scenario_metrics.setdefault(
                    scenario,
                    BinaryHistogramMetrics(),
                ).update(probabilities, targets)
                classes = class_matrix[row][mask]
                bounce_types = type_matrix[row][mask]
                bounce_orders = order_matrix[row][mask]
                predicted = probabilities >= threshold
                for target, class_id, bounce_type, bounce_order, prediction in zip(
                    targets,
                    classes,
                    bounce_types,
                    bounce_orders,
                    predicted,
                ):
                    if int(target) == 0:
                        update_rate(
                            real_class_groups,
                            str(int(class_id)),
                            prediction,
                        )
                        continue
                    if bounce_type == 1 and bounce_order == 2:
                        group_name = "type1_second"
                    elif bounce_type == 2 and bounce_order == 2:
                        group_name = "type2_second"
                    elif bounce_type == 2 and bounce_order == 4:
                        group_name = "type2_third"
                    elif bounce_order in (3, 6):
                        group_name = "ambiguous_order"
                    elif bounce_type == 0 or bounce_order == 0:
                        group_name = "generic_multipath"
                    else:
                        group_name = "other_multipath"
                    update_rate(ghost_groups, group_name, prediction)
    result = metrics.compute(fixed_threshold=threshold)
    result.update(
        {
            "data": str(Path(data).resolve()),
            "checkpoint": str(checkpoint.get("_source_path", "in-memory")),
            "split": split,
            "model_name": checkpoint["model_name"],
            "window_frames": checkpoint["window_frames"],
            "tta": bool(tta),
            "protocol_note": (
                "only confusion_matrix.fixed uses the checkpoint's "
                "validation-selected threshold; the operating and best_f1 "
                "blocks are re-derived on this split and are descriptive "
                "only (test-fitted when split=test)"
            ),
            "ghost_recall_by_bounce": {
                name: {
                    "count": counts[0],
                    "recall": counts[1] / max(counts[0], 1),
                }
                for name, counts in sorted(ghost_groups.items())
            },
            "real_false_positive_rate_by_class": {
                name: {
                    "count": counts[0],
                    "false_positive_rate": counts[1] / max(counts[0], 1),
                }
                for name, counts in sorted(real_class_groups.items())
            },
            "per_scenario": {
                name: scenario_metric.compute(fixed_threshold=threshold)
                for name, scenario_metric in sorted(scenario_metrics.items())
            },
        }
    )
    return result


def load_checkpoint(path, device="cpu"):
    """Load and gate one detector checkpoint."""

    try:
        checkpoint = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    if checkpoint.get("feature_schema") != FEATURE_SCHEMA_VERSION:
        raise ValueError("Checkpoint feature schema is incompatible")
    if tuple(checkpoint.get("feature_names", ())) != FEATURE_NAMES:
        raise ValueError("Checkpoint feature ordering is incompatible")
    checkpoint["_source_path"] = str(Path(path).resolve())
    return checkpoint


def main():
    args = parse_args()
    checkpoint = load_checkpoint(args.checkpoint, args.device)
    result = evaluate_checkpoint(
        args.data,
        checkpoint,
        args.split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
        threshold=args.threshold,
        tta=args.tta,
    )
    print(format_all_confusion_matrices(result))
    print()
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, sort_keys=True)
            handle.write("\n")


if __name__ == "__main__":
    main()
