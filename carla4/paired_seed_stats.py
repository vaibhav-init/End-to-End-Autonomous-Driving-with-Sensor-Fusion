#!/usr/bin/env python3
"""Paired per-seed statistics for the matched-vs-real-only comparison.

Reads the per-seed ``test_metrics.json`` written by
``evaluate_radar_ghost_detector.py`` for two families of fine-tune runs
(e.g. ``s2r_matched/ft_twostage_s<seed>`` and ``s2r/real_only_s<seed>``),
pairs them by seed, and reports:

- per-seed AUROC/AUPRC deltas (mean, SD, min, max, all-positive count)
- paired t-test (scipy, when available)
- exact Wilcoxon signed-rank test (enumerated in numpy, no scipy needed)
- exact two-sided sign test
- percentile bootstrap 95% CI over seed pairs

With n seeds the Wilcoxon/sign floors are 2/2^n: at n=5 both bottom out at
0.0625, at n=10 at ~0.002 — which is why the study runs ten paired seeds.

Example:
  python3 paired_seed_stats.py \
    --matched /storage/vaibhav/s2r_results/s2r_matched \
    --real /storage/vaibhav/s2r_results/s2r \
    --seeds 7 42 123 256 2024 5 11 77 99 300 \
    --matched-pattern "ft_twostage_s{seed}/test_metrics.json" \
    --real-pattern "real_only_s{seed}/test_metrics.json" \
    --output /storage/vaibhav/s2r_results/paired_seed_stats.json
"""

import argparse
import json
import math
from itertools import product
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matched", required=True,
                        help="directory of matched fine-tune runs")
    parser.add_argument("--real", required=True,
                        help="directory of real-only fine-tune runs")
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument(
        "--matched-pattern",
        default="ft_twostage_s{seed}/test_metrics.json",
    )
    parser.add_argument(
        "--real-pattern",
        default="real_only_s{seed}/test_metrics.json",
    )
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def exact_wilcoxon_signed_rank(diffs):
    """Two-sided exact p for the Wilcoxon signed-rank test, numpy only.

    Enumerates all 2^n sign assignments of the ranked absolute
    differences (n <= 20 keeps this at a million subsets or fewer) and
    returns the probability of W+ at least as extreme as observed under
    the symmetric null.
    """

    diffs = np.asarray(diffs, dtype=np.float64)
    diffs = diffs[diffs != 0.0]
    n = len(diffs)
    if n == 0:
        return float("nan"), 0
    # Average ranks for absolute differences (ties get the mean rank).
    order = np.argsort(np.abs(diffs), kind="stable")
    ranks = np.empty(n, dtype=np.float64)
    sorted_abs = np.abs(diffs)[order]
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sorted_abs[j + 1] == sorted_abs[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    w_obs = float(np.sum(ranks[diffs > 0]))
    signs = np.array(
        list(product((-1.0, 1.0), repeat=n)),
        dtype=np.float64,
    )
    # Each sign assignment gives S = sum(+-rank); W+ = (S + rank_sum) / 2.
    rank_sum = float(ranks.sum())
    assignments = ranks.reshape(1, -1) * signs
    w_distribution = (assignments.sum(axis=1) + rank_sum) / 2.0
    center = rank_sum / 2.0
    extreme = np.abs(w_distribution - center) >= abs(w_obs - center) - 1e-9
    p_value = float(extreme.mean())
    return p_value, n


def exact_sign_test(diffs):
    """Two-sided exact sign-test p (numpy/math only)."""

    diffs = np.asarray(diffs, dtype=np.float64)
    diffs = diffs[diffs != 0.0]
    n = len(diffs)
    if n == 0:
        return float("nan"), 0
    positive = int(np.sum(diffs > 0))
    k = min(positive, n - positive)
    mass = sum(math.comb(n, i) for i in range(k + 1)) / float(2**n)
    return min(1.0, 2.0 * mass), n


def paired_t_test(diffs):
    """scipy paired t-test when scipy is importable, else None."""

    try:
        from scipy import stats
    except ImportError:
        return None
    diffs = np.asarray(diffs, dtype=np.float64)
    if len(diffs) < 2:
        return None
    return float(stats.ttest_rel(diffs, np.zeros_like(diffs)).pvalue)


def bootstrap_ci(diffs, draws, seed):
    rng = np.random.default_rng(seed)
    diffs = np.asarray(diffs, dtype=np.float64)
    if len(diffs) == 0:
        return (float("nan"), float("nan"))
    means = rng.choice(diffs, size=(draws, len(diffs)), replace=True).mean(
        axis=1
    )
    return (
        float(np.percentile(means, 2.5)),
        float(np.percentile(means, 97.5)),
    )


def metric_for(path, name):
    with open(path, encoding="utf-8") as handle:
        return float(json.load(handle)[name])


def main():
    args = parse_args()
    matched_root = Path(args.matched)
    real_root = Path(args.real)
    rows = []
    for seed in args.seeds:
        matched_path = matched_root / args.matched_pattern.format(seed=seed)
        real_path = real_root / args.real_pattern.format(seed=seed)
        if not matched_path.exists() or not real_path.exists():
            raise SystemExit(
                f"missing pair for seed {seed}: "
                f"{matched_path if not matched_path.exists() else ''}"
                f"{real_path if not real_path.exists() else ''}"
            )
        rows.append(
            {
                "seed": int(seed),
                "matched_auroc": metric_for(matched_path, "auroc"),
                "real_auroc": metric_for(real_path, "auroc"),
                "matched_auprc": metric_for(matched_path, "auprc"),
                "real_auprc": metric_for(real_path, "auprc"),
            }
        )
    auroc_diffs = np.array(
        [row["matched_auroc"] - row["real_auroc"] for row in rows]
    )
    auprc_diffs = np.array(
        [row["matched_auprc"] - row["real_auprc"] for row in rows]
    )

    def summarize(diffs):
        wilcoxon_p, wilcoxon_n = exact_wilcoxon_signed_rank(diffs)
        sign_p, sign_n = exact_sign_test(diffs)
        low, high = bootstrap_ci(diffs, args.bootstrap, args.seed)
        return {
            "n_pairs": int(len(diffs)),
            "mean": float(diffs.mean()),
            "std": float(diffs.std(ddof=1)) if len(diffs) > 1 else 0.0,
            "min": float(diffs.min()),
            "max": float(diffs.max()),
            "positive_pairs": int(np.sum(diffs > 0)),
            "negative_pairs": int(np.sum(diffs < 0)),
            "paired_t_p": paired_t_test(diffs),
            "wilcoxon_p": wilcoxon_p,
            "wilcoxon_n_nonzero": wilcoxon_n,
            "sign_test_p": sign_p,
            "bootstrap_ci95": [low, high],
            "bootstrap_draws": int(args.bootstrap),
        }

    result = {
        "matched_dir": str(matched_root.resolve()),
        "real_dir": str(real_root.resolve()),
        "matched_pattern": args.matched_pattern,
        "real_pattern": args.real_pattern,
        "seeds": [row["seed"] for row in rows],
        "per_seed": rows,
        "delta_auroc": summarize(auroc_diffs),
        "delta_auprc": summarize(auprc_diffs),
        "note": (
            "Paired by seed. Exact Wilcoxon signed-rank and sign tests are "
            "enumerated (no asymptotic approximation); bootstrap CI "
            "resamples seed pairs."
        ),
    }

    print(
        f"{'seed':>6s} {'matched':>8s} {'real':>8s} {'dAUROC':>8s} "
        f"{'mPRC':>8s} {'rPRC':>8s} {'dAUPRC':>8s}"
    )
    for row in rows:
        print(
            f"{row['seed']:6d} {row['matched_auroc']:8.4f} "
            f"{row['real_auroc']:8.4f} "
            f"{row['matched_auroc'] - row['real_auroc']:8.4f} "
            f"{row['matched_auprc']:8.4f} {row['real_auprc']:8.4f} "
            f"{row['matched_auprc'] - row['real_auprc']:8.4f}"
        )
    for name, summary in (("AUROC", result["delta_auroc"]),
                          ("AUPRC", result["delta_auprc"])):
        t_text = (
            f"{summary['paired_t_p']:.4f}"
            if summary["paired_t_p"] is not None else "n/a"
        )
        print(
            f"delta {name}: mean {summary['mean']:+.4f} "
            f"(SD {summary['std']:.4f}, {summary['positive_pairs']}/"
            f"{summary['n_pairs']} positive) | "
            f"t-p {t_text} | wilcoxon {summary['wilcoxon_p']:.4f} | "
            f"sign {summary['sign_test_p']:.4f} | "
            f"CI95 [{summary['bootstrap_ci95'][0]:+.4f}, "
            f"{summary['bootstrap_ci95'][1]:+.4f}]"
        )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
