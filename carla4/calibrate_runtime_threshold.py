#!/usr/bin/env python3
"""Pick the runtime ghost-filter threshold from a score-only sidecar.

Deploying the benchmark-calibrated threshold (<=1% real FPR on RGD) inside
dense-town S5 traffic suppresses 50-75% of direct points: the operating
point that is safe on the benchmark is unsafe in deployment. This script
computes the deployment tradeoff directly from a score-only run (the
filter deployed with --radar-ghost-threshold 1.0, so every emitted point
carries its ghost_probability and the sim's exact source labels), where

  ghost recall(t)      = fraction of ghost points with probability >= t
  direct cluster loss  = fraction of direct clusters whose mean point
                         probability >= t  (clusters = one tracker
                         observation, keyed by truth id + reflector +
                         bounce family per frame)

The knee of that curve is the defensible deployment threshold, and the
resulting (ghost recall, direct loss) pair is the honest cost of the
learned filter in the loop.

Example:
  python3 calibrate_runtime_threshold.py \
    --sidecar scenarios/results_score_only/results_s5/s5_fog3_seed42.detections.npz \
    --output artifacts/final/runtime_threshold_calibration.json
"""

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sidecar",
        action="append",
        required=True,
        help="score-only .detections.npz path; repeat to pool runs",
    )
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="*",
        default=(0.5, 0.6, 0.7, 0.77, 0.8, 0.85, 0.9, 0.95, 0.99),
    )
    parser.add_argument(
        "--dynamic-tags",
        type=int,
        nargs="*",
        default=(12, 14, 18, 19, 21),
        help=(
            "semantic tags counted as dynamic road users; the controller "
            "only ever selects a dynamic track, so the dynamic-cluster "
            "loss is the deployment-relevant cost"
        ),
    )
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def cluster_mean_probabilities(detections, dynamic_tags=None):
    """Mean ghost probability per (frame, cluster) for direct points.

    Clusters are the tracker's observations: one per object per frame. The
    emitted points inherit their cluster's truth ids and bounce family, so
    (frame, truth_object_id, bounce_type, bounce_order) is the cluster key.
    ``dynamic_tags`` optionally restricts the clusters to dynamic road
    users (the controller only ever selects a dynamic track).
    """

    source = detections["source"].astype(str)
    direct = source == "direct"
    if dynamic_tags is not None:
        direct &= np.isin(detections["semantic_tag"], list(dynamic_tags))
    if not np.any(direct):
        return np.zeros(0), np.zeros(0)
    selected = detections[direct]
    keys = (
        selected["frame"].astype(np.int64) * 1_000_003
        + selected["truth_object_id"].astype(np.int64) * 1009
        + np.array(
            [hash((str(btype), int(order))) % 997
             for btype, order in zip(
                 selected["bounce_type"].astype(str),
                 selected["bounce_order"],
             )]
        )
    )
    probabilities = selected["ghost_probability"].astype(np.float64)
    unique_keys, inverse, counts = np.unique(keys, return_inverse=True,
                                             return_counts=True)
    sums = np.bincount(inverse, weights=probabilities)
    means = sums / counts
    return means, counts


def compute_curve(sidecars, thresholds, dynamic_tags):
    """Ghost-recall vs cluster-loss rows for a set of score-only sidecars."""

    ghost_probabilities = []
    all_cluster_means = []
    dynamic_cluster_means = []
    for path in sidecars:
        detections = np.load(path, allow_pickle=True)["detections"]
        source = detections["source"].astype(str)
        ghost_probabilities.append(
            detections["ghost_probability"].astype(np.float64)[
                source == "ghost"
            ]
        )
        all_cluster_means.append(cluster_mean_probabilities(detections))
        dynamic_cluster_means.append(
            cluster_mean_probabilities(detections, dynamic_tags)
        )
    ghost_points = np.concatenate(ghost_probabilities)
    pooled = [
        (np.concatenate([item[0] for item in groups]),
         np.concatenate([item[1] for item in groups]))
        for groups in (all_cluster_means, dynamic_cluster_means)
    ]
    ghost_points_total = int(len(ghost_points))

    rows = []
    for threshold in thresholds:
        row = {"threshold": float(threshold)}
        row["ghost_point_recall"] = (
            float(np.mean(ghost_points >= threshold))
            if ghost_points_total else float("nan")
        )
        for name, (means, counts) in zip(("all", "dynamic"), pooled):
            total = float(counts.sum())
            lost = float(np.sum(counts[means >= threshold]))
            row[f"{name}_cluster_loss"] = lost / total if total else float("nan")
            row[f"{name}_clusters_lost"] = int(lost)
            row[f"{name}_clusters_total"] = int(total)
        rows.append(row)
    return {
        "ghost_points": ghost_points_total,
        "curve": rows,
    }


def main():
    args = parse_args()
    curve = compute_curve(args.sidecars, args.thresholds, args.dynamic_tags)
    result = {
        "sidecars": [str(Path(path).resolve()) for path in args.sidecar],
        "ghost_points": curve["ghost_points"],
        "dynamic_tags": list(args.dynamic_tags),
        "curve": curve["curve"],
        "note": (
            "ghost_point_recall is over ghost points; cluster loss weights "
            "clusters equally (a lost cluster is a lost tracker "
            "observation). The deployment-relevant cost is the dynamic "
            "cluster loss: the controller only selects dynamic tracks. "
            "Deployment threshold = knee of the curve, chosen on the "
            "simulator's exact labels."
        ),
    }
    print(
        f"{'thr':>6s} {'ghost rec':>10s} "
        f"{'dyn loss':>9s} {'dyn lost':>9s} "
        f"{'all loss':>9s}"
    )
    for row in result["curve"]:
        print(
            f"{row['threshold']:6.2f} {row['ghost_point_recall']:10.4f} "
            f"{row['dynamic_cluster_loss']:9.4f} "
            f"{row['dynamic_clusters_lost']:9d} "
            f"{row['all_cluster_loss']:9.4f}"
        )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
