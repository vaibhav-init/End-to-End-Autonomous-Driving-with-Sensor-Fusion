#!/usr/bin/env python3
"""Score fixed seed ensembles of ghost detectors on a held-out split.

Two training recipes can have similar single-seed means while differing in
variance. Averaging a fixed set of seeds is a deployment choice made before
looking at test, so comparing the two ensembles is a fair recipe comparison.

Thresholds are selected on the validation scores and applied unchanged to the
test scores. The paired bootstrap resamples whole sequences, so correlated
points from one recording are not counted as independent evidence.

Example:
  python3 ensemble_ghost_scores.py \
    --val artifacts/transfer/all_val_scores.npz \
    --test artifacts/transfer/all_test_scores.npz \
    --group ft=ft42,ft7,ft123,ft2024,ft256 \
    --group real=ro42,ro7,ro123,ro2024,ro256 \
    --output artifacts/transfer/ensemble_report.json
"""

import argparse
import json
from pathlib import Path
import re

import numpy as np

from bootstrap_ghost_transfer import (
    aggregate,
    paired_summary,
    per_group_histograms,
)
from evaluate_ghost_transfer import (
    _calibrate,
    _class_breakdown,
    _metrics,
    _scenario_breakdown,
    _transform,
)


SCENARIO_PATTERN = re.compile(r"scenario[-_](\d+)", re.IGNORECASE)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--val", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument(
        "--group",
        action="append",
        required=True,
        help="name=member1,member2,...; repeat for each ensemble",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-real-fpr", type=float, default=0.01)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument(
        "--mode",
        choices=("mean", "logit"),
        default="mean",
        help="probability fusion rule",
    )
    return parser.parse_args()


def scenario_of(sequence):
    match = SCENARIO_PATTERN.search(sequence)
    return f"scenario-{int(match.group(1)):02d}" if match else sequence


def load_groups(path, groups):
    with np.load(path, allow_pickle=True) as archive:
        target = archive["target"].astype(np.int64)
        sequence = archive["sequence"]
        probability = {}
        for name, members in groups.items():
            missing = [member for member in members if f"prob__{member}" not in archive.files]
            if missing:
                raise ValueError(f"{path}: missing members {missing} in group {name!r}")
            probability[name] = np.stack(
                [archive[f"prob__{member}"] for member in members]
            )
    return target, sequence, probability


def main():
    args = parse_args()
    groups = {}
    for spec in args.group:
        name, members = spec.split("=", 1)
        groups[name] = [member.strip() for member in members.split(",") if member.strip()]
    if len(groups) < 2:
        raise ValueError("at least two groups are needed for a paired comparison")

    val_target, _, val_probability = load_groups(args.val, groups)
    test_target, test_sequence, test_probability = load_groups(args.test, groups)
    test_scenario = np.asarray(
        [scenario_of(text) for text in test_sequence.tolist()], dtype=object
    )

    report = {
        "val": str(Path(args.val).resolve()),
        "test": str(Path(args.test).resolve()),
        "mode": args.mode,
        "max_real_fpr": args.max_real_fpr,
        "groups": groups,
        "models": {},
    }

    transformed_val = {}
    transformed_test = {}
    for name in groups:
        transformed_val[name] = _transform(val_probability[name], args.mode)
        transformed_test[name] = _transform(test_probability[name], args.mode)
        threshold = _calibrate(
            val_target, transformed_val[name], args.max_real_fpr
        )
        val_result = _metrics(
            val_target, transformed_val[name], threshold, args.max_real_fpr
        )
        test_result = _metrics(
            test_target, transformed_test[name], threshold, args.max_real_fpr
        )
        fixed = test_result["confusion_matrix"]["fixed"]
        report["models"][name] = {
            "members": groups[name],
            "threshold": float(threshold),
            "val_auroc": val_result["auroc"],
            "val_auprc": val_result["auprc"],
            "test_auroc": test_result["auroc"],
            "test_auprc": test_result["auprc"],
            "test_fixed": fixed,
            "test_by_class": {},
            "test_by_scenario": {},
        }
        print(
            f"{name:8s} val AUROC {val_result['auroc']:.4f} "
            f"AUPRC {val_result['auprc']:.4f} | test AUROC "
            f"{test_result['auroc']:.4f} AUPRC {test_result['auprc']:.4f} "
            f"| thr {threshold:.4f} recall {fixed['recall']:.3f} "
            f"FPR {fixed['false_positive_rate']:.3f}"
        )

    # Per-class and per-scenario need the class ids; reload them once.
    with np.load(args.test, allow_pickle=True) as archive:
        test_class = archive["class_id"]
    for name in groups:
        fixed = report["models"][name]["test_fixed"]
        report["models"][name]["test_by_class"] = _class_breakdown(
            test_class,
            test_target,
            transformed_test[name],
            fixed["threshold"],
            args.max_real_fpr,
        )
        report["models"][name]["test_by_scenario"] = _scenario_breakdown(
            test_sequence,
            test_target,
            transformed_test[name],
            fixed["threshold"],
            args.max_real_fpr,
        )

    # Paired bootstrap over whole sequences for every group against the first.
    reference = next(iter(groups))
    names = sorted(set(test_sequence.tolist()))
    rng = np.random.default_rng(args.seed)
    samples = [rng.choice(names, size=len(names), replace=True) for _ in range(args.bootstrap)]
    histograms = {
        name: per_group_histograms(
            test_sequence, test_target, transformed_test[name]
        )
        for name in groups
    }
    boot = {}
    for name, values in histograms.items():
        auroc, auprc = [], []
        for sample in samples:
            result = aggregate(values, sample)
            auroc.append(result["auroc"])
            auprc.append(result["auprc"])
        boot[name] = (np.asarray(auroc), np.asarray(auprc))
    report["paired_vs_reference"] = {}
    for name in groups:
        if name == reference:
            continue
        report["paired_vs_reference"][name] = {
            "auroc": paired_summary(boot[reference][0], boot[name][0]),
            "auprc": paired_summary(boot[reference][1], boot[name][1]),
        }
        print(
            f"{name} - {reference}: dAUROC "
            f"{report['paired_vs_reference'][name]['auroc']['mean']:+.4f} "
            f"p={report['paired_vs_reference'][name]['auroc']['two_sided_p']:.4f} "
            f"| dAUPRC "
            f"{report['paired_vs_reference'][name]['auprc']['mean']:+.4f} "
            f"p={report['paired_vs_reference'][name]['auprc']['two_sided_p']:.4f}"
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
