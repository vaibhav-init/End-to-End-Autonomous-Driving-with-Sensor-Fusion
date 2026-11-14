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
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def cluster_mean_probabilities(detections):
    """Mean ghost probability per (frame, cluster) for direct points.

    Clusters are the tracker's observations: one per object per frame. The
    emitted points inherit their cluster's truth ids and bounce family, so
    (frame, truth_object_id, reflector_id, bounce_type, bounce_order) is
    the cluster key.
    """

    source = detections["source"].astype(str)
    direct = source == "direct"
    if not np.any(direct):
        return np.zeros(0), np.zeros(0)
    selected = detections[direct]
    keys = (
        selected["frame"].astype(np.int64) * 1_000_003
        + selected["truth_object_id"].astype(np.int64) * 1009
        + np.array(
            [hash((str(reflector), str(btype), int(order))) % 997
             for reflector, btype, order in zip(
                 selected["reflector_id"],
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


def main():
    args = parse_args()
    ghost_probabilities = []
    cluster_means = []
    for path in args.sidecar:
        detections = np.load(path, allow_pickle=True)["detections"]
        source = detections["source"].astype(str)
        ghost_probabilities.append(
            detections["ghost_probability"].astype(np.float64)[
                source == "ghost"
            ]
        )
        means, counts = cluster_mean_probabilities(detections)
        cluster_means.append((means, counts))
    ghost_points = np.concatenate(ghost_probabilities)
    means = np.concatenate([item[0] for item in cluster_means])
    counts = np.concatenate([item[1] for item in cluster_means])
    weighted_clusters = float(counts.sum())

    rows = []
    for threshold in args.thresholds:
        ghost_recall = float(np.mean(ghost_points >= threshold)) \
            if len(ghost_points) else float("nan")
        lost = float(np.sum(counts[means >= threshold]))
        rows.append(
            {
                "threshold": float(threshold),
                "ghost_point_recall": ghost_recall,
                "direct_cluster_loss": lost / weighted_clusters,
                "direct_clusters_lost": int(lost),
                "direct_clusters_total": int(weighted_clusters),
            }
        )
    ghost_points_total = int(len(ghost_points))
    result = {
        "sidecars": [str(Path(path).resolve()) for path in args.sidecar],
        "ghost_points": ghost_points_total,
        "direct_clusters": int(weighted_clusters),
        "curve": rows,
        "note": (
            "ghost_point_recall is over ghost points; direct_cluster_loss "
            "weights clusters equally (a lost cluster is a lost tracker "
            "observation regardless of its point count). Deployment "
            "threshold = knee of the curve, chosen on the simulator's "
            "exact labels."
        ),
    }
    print(
        f"{'threshold':>10s} {'ghost recall':>13s} "
        f"{'direct loss':>12s} {'clusters lost':>14s}"
    )
    for row in rows:
        print(
            f"{row['threshold']:10.2f} {row['ghost_point_recall']:13.4f} "
            f"{row['direct_cluster_loss']:12.4f} "
            f"{row['direct_clusters_lost']:14d}"
        )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
