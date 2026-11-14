#!/usr/bin/env python3
"""Per-feature MMD fidelity table between two prepared ghost datasets.

For each of the 11 features of ``radar_ghost_physical_v2`` this script
computes the squared maximum mean discrepancy (MMD^2, RBF kernel with the
median heuristic) between the real and synthetic per-point distributions,
restricted to labeled current-scan points (age 0, matching the pooled
AUROC/AUPRC protocol in the matched study). It also reports the same
distances between two disjoint halves of the *real* data — the "floor": two
halves of the same real sensor already differ by that much, so a synthetic
distance near the floor means indistinguishable at this statistic
(same convention as evaluate_ghost_fidelity.py's Wasserstein floor).

This is the RadarGen-style tier-1 distributional check for the paper:
cheap, split-respecting, and preempts "is the synthetic data distributionally
realistic?" with numbers instead of assertions.

Example:
  python3 fidelity_mmd_table.py \
    --real artifacts/ghost_real_official_v2 \
    --synthetic artifacts/ghost_carla_matched_v1_prepared \
    --split test --output artifacts/final/mmd_fidelity.json
"""

import argparse
import json
from pathlib import Path

import numpy as np

from radar.ghost_detection.features import (
    FEATURE_NAMES,
    physical_features,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real", required=True, help="prepared real dataset")
    parser.add_argument(
        "--synthetic",
        required=True,
        help="prepared synthetic dataset",
    )
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--max-points",
        type=int,
        default=50000,
        help="per-domain subsample cap for the kernel estimates",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", help="output JSON path")
    return parser.parse_args()


def load_split_points(prepared_dir, split, rng, max_points):
    """Return per-point feature rows for labeled points of one split.

    Features follow physical_features() with age 0: the fidelity protocol
    covers current-scan labeled points, exactly like the pooled AUROC/AUPRC
    numbers in the matched study. Frame statistics are the precomputed v2
    fields in the npz (computed over the full frame at preparation time).
    """

    manifest = json.loads(
        (Path(prepared_dir) / "manifest.json").read_text()
    )
    records = [
        record
        for record in manifest["sequences"]
        if record.get("split") == split
    ]
    if not records:
        raise SystemExit(
            f"no sequences with split={split!r} in {prepared_dir}"
        )
    features = []
    for record in records:
        path = Path(prepared_dir) / record["path"]
        data = np.load(path, allow_pickle=False)
        target = data["target"]
        labeled = (target == 0) | (target == 1)
        if not np.any(labeled):
            continue
        features.append(
            physical_features(
                data["r_sc"][labeled],
                data["phi_sc"][labeled],
                data["vr_sc"][labeled],
                data["amp"][labeled],
                np.zeros(int(np.count_nonzero(labeled))),
                relative_log_amplitude=data["rel_log_amplitude"][labeled],
                doppler_cluster_residual=data["doppler_cluster_residual"][
                    labeled
                ],
                local_density_ratio=data["local_density_ratio"][labeled],
            )
        )
    if not features:
        raise SystemExit(f"no labeled points in {prepared_dir} split {split}")
    stacked = np.concatenate(features, axis=0)
    stacked = stacked[np.isfinite(stacked).all(axis=1)]
    if len(stacked) > max_points:
        picked = rng.choice(len(stacked), size=max_points, replace=False)
        stacked = stacked[np.sort(picked)]
    return stacked


def _rbf_gram(a, b, bandwidth):
    """RBF kernel gram matrix with a fixed bandwidth (1-D inputs)."""

    diff = (a.reshape(-1, 1) - b.reshape(-1, 1)) / bandwidth
    return np.exp(-0.5 * diff * diff)


def _median_bandwidth(pooled_1d):
    """Median pairwise distance heuristic for one feature dimension."""

    pooled = pooled_1d.reshape(-1)
    if pooled.size > 2000:
        pooled = pooled[
            np.sort(
                np.random.choice(pooled.size, size=2000, replace=False)
            )
        ]
    distances = np.abs(pooled.reshape(-1, 1) - pooled.reshape(1, -1))
    median = float(np.median(distances[distances > 0]))
    if not np.isfinite(median) or median <= 0:
        return 1.0
    return median


def _mmd2(x, y, bandwidth):
    """Unbiased MMD^2 with an RBF kernel on 1-D samples."""

    n, m = len(x), len(y)
    if n < 2 or m < 2:
        return float("nan")
    kxx = _rbf_gram(x, x, bandwidth)
    kyy = _rbf_gram(y, y, bandwidth)
    kxy = _rbf_gram(x, y, bandwidth)
    sum_x = kxx.sum() - np.trace(kxx)
    sum_y = kyy.sum() - np.trace(kyy)
    sum_xy = kxy.sum()
    return float(
        sum_x / (n * (n - 1)) + sum_y / (m * (m - 1)) - 2.0 * sum_xy / (n * m)
    )


def _joint_mmd2(x, y, bandwidth_scale=1.0):
    """MMD^2 on the full feature vector with a median-heuristic bandwidth."""

    pooled = np.concatenate((x, y), axis=0)
    if pooled.shape[0] > 2000:
        pooled = pooled[
            np.sort(
                np.random.choice(pooled.shape[0], size=2000, replace=False)
            )
        ]
    distances = np.linalg.norm(
        pooled.reshape(-1, 1, pooled.shape[1])
        - pooled.reshape(1, pooled.shape[0], pooled.shape[1]),
        axis=2,
    )
    finite = distances[np.isfinite(distances) & (distances > 0)]
    median = float(np.median(finite)) if finite.size else 1.0
    bandwidth = max(median * bandwidth_scale, 1.0e-6)

    def _gram(a, b):
        difference = (
            a.reshape(len(a), 1, a.shape[1]) - b.reshape(1, len(b), a.shape[1])
        )
        return np.exp(
            -0.5
            * np.sum(difference * difference, axis=2)
            / (bandwidth * bandwidth)
        )

    n, m = len(x), len(y)
    kxx = _gram(x, x)
    kyy = _gram(y, y)
    kxy = _gram(x, y)
    return float(
        (kxx.sum() - np.trace(kxx)) / (n * (n - 1))
        + (kyy.sum() - np.trace(kyy)) / (m * (m - 1))
        - 2.0 * kxy.sum() / (n * m)
    )


def _mmd_table(x, y, feature_names, max_points, rng, kernel_points=4000):
    """Per-feature MMD^2 plus the joint 11-d MMD^2.

    Kernels are built on at most ``kernel_points`` per domain: the unbiased
    MMD^2 estimator is O(n^2) in the kernel gram matrices, so a 30k x 30k
    matrix would need ~7 GB per feature. The estimator stays consistent —
    it is a subsample of the same distribution — and the cap is reported.
    """

    cap = min(kernel_points, len(x), len(y))
    if len(x) > cap:
        x = x[np.sort(rng.choice(len(x), size=cap, replace=False))]
    if len(y) > cap:
        y = y[np.sort(rng.choice(len(y), size=cap, replace=False))]
    result = {}
    for column, name in enumerate(feature_names):
        pooled = np.concatenate((x[:, column], y[:, column]))
        bandwidth = _median_bandwidth(pooled)
        result[name] = _mmd2(x[:, column], y[:, column], bandwidth)
    result["joint_11d"] = _joint_mmd2(x, y)
    return result


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    np.random.seed(args.seed)
    real = load_split_points(args.real, args.split, rng, args.max_points)
    synthetic = load_split_points(
        args.synthetic, args.split, rng, args.max_points
    )
    # Floor: two disjoint halves of the real split (even/odd sequence
    # ordering already interleaves scenarios, so a positional split is a
    # same-sensor between-halves comparison, not a domain comparison).
    half = len(real) // 2
    floor = _mmd_table(
        real[:half],
        real[half:],
        FEATURE_NAMES,
        args.max_points,
        rng,
    )
    domain = _mmd_table(
        real,
        synthetic,
        FEATURE_NAMES,
        args.max_points,
        rng,
    )
    result = {
        "real_dataset": str(Path(args.real).resolve()),
        "synthetic_dataset": str(Path(args.synthetic).resolve()),
        "split": args.split,
        "real_points": int(len(real)),
        "synthetic_points": int(len(synthetic)),
        "max_points": int(args.max_points),
        "seed": int(args.seed),
        "domain_mmd2": domain,
        "real_floor_mmd2": floor,
        "ratio": {
            name: (
                domain[name] / floor[name]
                if floor.get(name, 0.0) > 0.0
                else None
            )
            for name in domain
        },
        "note": (
            "MMD^2 with an RBF kernel and median-heuristic bandwidth on "
            "labeled current-scan points (age 0). ratio = domain/floor; "
            "ratio near 1 means the synthetic side is as close to the real "
            "split as two halves of the real split are to each other."
        ),
    }
    print(
        f"{'feature':38s} {'domain MMD^2':>14s} {'real floor':>12s} "
        f"{'ratio':>8s}"
    )
    for name in list(FEATURE_NAMES) + ["joint_11d"]:
        ratio = result["ratio"].get(name)
        ratio_text = f"{ratio:8.2f}" if ratio is not None else "     inf"
        print(
            f"{name:38s} {domain[name]:14.6f} {floor[name]:12.6f} "
            f"{ratio_text}"
        )
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print(f"wrote {output}")


if __name__ == "__main__":
    main()
