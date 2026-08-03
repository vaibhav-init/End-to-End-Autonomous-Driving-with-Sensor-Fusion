#!/usr/bin/env python3
"""Evaluate a radar ghost detector on a prepared sequence split."""

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from radar.ghost_detection.dataset import PreparedGhostDataset
from radar.ghost_detection.features import FEATURE_NAMES, FEATURE_SCHEMA_VERSION
from radar.ghost_detection.metrics import BinaryHistogramMetrics
from radar.ghost_detection.model import create_ghost_model


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
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        checkpoint = torch.load(
            args.checkpoint,
            map_location=args.device,
            weights_only=True,
        )
    except TypeError:
        checkpoint = torch.load(args.checkpoint, map_location=args.device)
    if checkpoint.get("feature_schema") != FEATURE_SCHEMA_VERSION:
        raise ValueError("Checkpoint feature schema is incompatible")
    if tuple(checkpoint.get("feature_names", ())) != FEATURE_NAMES:
        raise ValueError("Checkpoint feature ordering is incompatible")
    dataset = PreparedGhostDataset(
        args.data,
        args.split,
        window_frames=int(checkpoint["window_frames"]),
        max_points=int(checkpoint["max_points"]),
        augment=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
    )
    model = create_ghost_model(
        checkpoint["model_name"],
        **checkpoint.get("model_kwargs", {}),
    ).to(args.device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    metrics = BinaryHistogramMetrics()
    threshold = (
        float(checkpoint.get("threshold", 0.5))
        if args.threshold is None
        else args.threshold
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
            features = batch["features"].to(args.device, non_blocking=True)
            point_mask = batch["point_mask"].to(args.device, non_blocking=True)
            logits = model(features, point_mask).cpu()
            probability_matrix = torch.sigmoid(logits).numpy()
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
            "data": str(Path(args.data).resolve()),
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "split": args.split,
            "model_name": checkpoint["model_name"],
            "window_frames": checkpoint["window_frames"],
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
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, sort_keys=True)
            handle.write("\n")


if __name__ == "__main__":
    main()
