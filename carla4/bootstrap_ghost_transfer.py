#!/usr/bin/env python3
"""Paired bootstrap for ghost-detector transfer scores.

The official test split has five scenarios, so a pooled point-level AUROC
treats thousands of correlated points as independent evidence. This script
resamples whole sequences (or whole scenarios) with replacement and reports
confidence intervals for each model plus the paired difference against a
reference. Histograms are accumulated per sequence first, which keeps the
bootstrap cheap even for ~200k labelled points.

Example:
  python3 bootstrap_ghost_transfer.py \
    --scores artifacts/transfer/ft_test_scores.npz \
    --reference real_only --output artifacts/transfer/bootstrap.json
"""

import argparse
import json
from pathlib import Path
import re

import numpy as np

from radar.ghost_detection.metrics import BinaryHistogramMetrics


SCENARIO_PATTERN = re.compile(r"scenario[-_](\d+)", re.IGNORECASE)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", required=True, help="score .npz from dump")
    parser.add_argument(
        "--reference",
        required=True,
        help="checkpoint name used for the paired difference",
    )
    parser.add_argument("--output", required=True, help="output .json path")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=12345)
    return parser.parse_args()


def scenario_of(sequence):
    match = SCENARIO_PATTERN.search(sequence)
    return f"scenario-{int(match.group(1)):02d}" if match else sequence


def per_group_histograms(groups, targets, probabilities):
    result = {}
    for name in sorted(set(groups.tolist())):
        selected = groups == name
        if not np.any(selected):
            continue
        metrics = BinaryHistogramMetrics()
        metrics.update(probabilities[selected], targets[selected])
        result[name] = (metrics.positive.copy(), metrics.negative.copy())
    return result


def aggregate(histograms, names):
    positive = np.sum([histograms[name][0] for name in names], axis=0)
    negative = np.sum([histograms[name][1] for name in names], axis=0)
    metrics = BinaryHistogramMetrics()
    metrics.positive = positive.astype(np.int64)
    metrics.negative = negative.astype(np.int64)
    return metrics.compute()


def bootstrap_samples(names, rng, count):
    """Draw one shared set of resamples for paired model comparisons."""

    return [
        rng.choice(names, size=len(names), replace=True) for _ in range(count)
    ]


def bootstrap_metrics(histograms, samples):
    auroc = []
    auprc = []
    for sample in samples:
        result = aggregate(histograms, sample)
        auroc.append(result["auroc"])
        auprc.append(result["auprc"])
    return np.asarray(auroc), np.asarray(auprc)


def summarize(values):
    return {
        "mean": float(np.mean(values)),
        "ci95_low": float(np.quantile(values, 0.025)),
        "ci95_high": float(np.quantile(values, 0.975)),
    }


def paired_summary(reference, candidate):
    difference = candidate - reference
    return {
        "mean": float(np.mean(difference)),
        "ci95_low": float(np.quantile(difference, 0.025)),
        "ci95_high": float(np.quantile(difference, 0.975)),
        "fraction_positive": float(np.mean(difference > 0.0)),
        "two_sided_p": float(
            min(1.0, 2.0 * min(np.mean(difference <= 0.0), np.mean(difference >= 0.0)))
        ),
    }


def main():
    args = parse_args()
    with np.load(args.scores, allow_pickle=True) as archive:
        names = sorted(
            key[len("prob__") :] for key in archive.files if key.startswith("prob__")
        )
        if args.reference not in names:
            raise ValueError(f"reference {args.reference!r} not in {names}")
        target = archive["target"].astype(np.int64)
        sequence = archive["sequence"]
        scenario = np.asarray(
            [scenario_of(text) for text in sequence.tolist()], dtype=object
        )
        probability = {name: archive[f"prob__{name}"] for name in names}

    report = {
        "scores": str(Path(args.scores).resolve()),
        "reference": args.reference,
        "checkpoints": names,
        "bootstrap": args.bootstrap,
        "sequence_level": {},
        "scenario_level": {},
    }

    rng = np.random.default_rng(args.seed)
    for level, groups in (("sequence_level", sequence), ("scenario_level", scenario)):
        histograms = {
            name: per_group_histograms(groups, target, probability[name])
            for name in names
        }
        samples = bootstrap_samples(
            sorted(histograms[names[0]]), rng, args.bootstrap
        )
        auroc, auprc = {}, {}
        for name in names:
            auroc[name], auprc[name] = bootstrap_metrics(
                histograms[name], samples
            )
        level_report = {
            "groups": sorted(histograms[names[0]]),
            "models": {},
        }
        for name in names:
            observed = aggregate(histograms[name], sorted(histograms[name]))
            level_report["models"][name] = {
                "observed_auroc": observed["auroc"],
                "observed_auprc": observed["auprc"],
                "auroc": summarize(auroc[name]),
                "auprc": summarize(auprc[name]),
            }
        level_report["paired_vs_reference"] = {
            name: {
                "auroc": paired_summary(auroc[args.reference], auroc[name]),
                "auprc": paired_summary(auprc[args.reference], auprc[name]),
            }
            for name in names
            if name != args.reference
        }
        report[level] = level_report

    for level in ("sequence_level", "scenario_level"):
        print(f"== {level} (groups={len(report[level]['groups'])})")
        for name in names:
            model = report[level]["models"][name]
            print(
                f"  {name:16s} AUROC {model['observed_auroc']:.4f} "
                f"[{model['auroc']['ci95_low']:.4f}, {model['auroc']['ci95_high']:.4f}] "
                f"AUPRC {model['observed_auprc']:.4f} "
                f"[{model['auprc']['ci95_low']:.4f}, {model['auprc']['ci95_high']:.4f}]"
            )
        for name, difference in report[level]["paired_vs_reference"].items():
            print(
                f"  {name:16s} - {args.reference}: dAUROC "
                f"{difference['auroc']['mean']:+.4f} "
                f"[{difference['auroc']['ci95_low']:+.4f}, "
                f"{difference['auroc']['ci95_high']:+.4f}] p="
                f"{difference['auroc']['two_sided_p']:.4f} | dAUPRC "
                f"{difference['auprc']['mean']:+.4f} "
                f"[{difference['auprc']['ci95_low']:+.4f}, "
                f"{difference['auprc']['ci95_high']:+.4f}] p="
                f"{difference['auprc']['two_sided_p']:.4f}"
            )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
