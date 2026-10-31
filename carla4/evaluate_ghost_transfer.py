#!/usr/bin/env python3
"""Dump, calibrate and ensemble ghost-detector scores across domains.

Two subcommands exist because threshold selection needs a *validation* split
while the final number is reported on a disjoint *test* split:

  dump     run one or more checkpoints over a prepared split and save the
           per-point probabilities plus point identity
  analyze  calibrate thresholds on validation scores and report the fixed
           operating point on test, including per-class and per-scenario
           breakdowns and validation-selected ensembles

The dump uses ``evaluate_cross_domain.CrossDomainDataset`` so a checkpoint can
be scored against data built with a different feature schema. Point identity
(sequence, sensor, frame, raw index) is saved so several checkpoints can be
aligned exactly; the dataset's sampling is deterministic.

Example:
  python3 evaluate_ghost_transfer.py dump \
    --checkpoint zeroshot=artifacts/zeroshot_v3_carla_pretrain/best_detector.pt \
    --checkpoint ft=artifacts/ft_v7s_measloss_v1/best_detector.pt \
    --data artifacts/ghost_real_official --split test \
    --output artifacts/transfer/test_scores.npz
"""

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from evaluate_cross_domain import CrossDomainDataset, load_checkpoint
from radar.ghost_detection.labels import CLASS_NAMES
from radar.ghost_detection.metrics import BinaryHistogramMetrics


KEY_FIELDS = ("sequence", "sensor", "frame", "indices", "target")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    dump = subparsers.add_parser("dump", help="score one or more checkpoints")
    dump.add_argument(
        "--checkpoint",
        action="append",
        required=True,
        help="name=path; repeat for several checkpoints",
    )
    dump.add_argument("--data", required=True, help="prepared dataset directory")
    dump.add_argument("--split", default="test")
    dump.add_argument("--output", required=True, help="output .npz path")
    dump.add_argument("--batch-size", type=int, default=64)
    dump.add_argument("--num-workers", type=int, default=4)
    dump.add_argument("--cache-sequences", type=int, default=128)
    dump.add_argument(
        "--context-reserve",
        type=float,
        default=0.25,
        help="must match the checkpoint's training value",
    )
    dump.add_argument(
        "--no-tta",
        action="store_true",
        help="disable azimuth-mirror test-time augmentation",
    )
    dump.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )

    analyze = subparsers.add_parser("analyze", help="calibrate and ensemble")
    analyze.add_argument("--val", required=True, help="validation score .npz")
    analyze.add_argument("--test", required=True, help="test score .npz")
    analyze.add_argument("--output", required=True, help="output .json path")
    analyze.add_argument(
        "--max-real-fpr",
        type=float,
        default=0.01,
        help="target false-positive rate used to pick the threshold on validation",
    )
    analyze.add_argument(
        "--select-metric",
        choices=("auprc", "auroc"),
        default="auprc",
        help="validation metric used to choose among ensembles",
    )
    analyze.add_argument(
        "--max-ensemble-size",
        type=int,
        default=3,
        help="largest validation-selected ensemble to consider",
    )
    analyze.add_argument(
        "--ensemble-mode",
        choices=("mean", "logit", "rank"),
        action="append",
        help="repeat to evaluate several fusion rules",
    )
    return parser.parse_args()


def _parse_spec(text):
    if "=" not in text:
        raise ValueError(f"--checkpoint must be name=path, got {text!r}")
    name, path = text.split("=", 1)
    if not name or not path:
        raise ValueError(f"--checkpoint must be name=path, got {text!r}")
    return name, path


def _flip_features(features):
    """Mirror a point cloud about the sensor boresight.

    Both schemas put x at column 0 and sin(azimuth) at column 3; range,
    cos(azimuth), Doppler, amplitude statistics, age and density are invariant
    under a left/right mirror. The detector is trained with the same 50%
    azimuth flip, so averaging the two views is a valid TTA.
    """

    flipped = features.clone()
    flipped[..., 0] *= -1.0
    flipped[..., 3] *= -1.0
    return flipped


def _predict(module, features, point_mask, tta):
    probabilities = torch.sigmoid(module(features, point_mask))
    if tta:
        mirrored = torch.sigmoid(module(_flip_features(features), point_mask))
        probabilities = 0.5 * (probabilities + mirrored)
    return probabilities


def _flatten_records(records):
    return {
        field: np.concatenate([record[field] for record in records])
        for field in records[0]
    }


def _collate(batch):
    """Keep the variable-length raw point indices out of the stack."""

    result = {}
    for key in batch[0]:
        if key == "indices":
            result[key] = [sample[key] for sample in batch]
        else:
            result[key] = torch.stack([sample[key] for sample in batch])
    return result


def dump(args):
    specifications = [_parse_spec(text) for text in args.checkpoint]
    names = [name for name, _ in specifications]
    if len(set(names)) != len(names):
        raise ValueError("checkpoint names must be unique")

    reference = None
    probabilities = {}
    metadata = {}
    for name, path in specifications:
        checkpoint = load_checkpoint(path, args.device)
        dataset = CrossDomainDataset(
            args.data,
            args.split,
            window_frames=checkpoint["window_frames"],
            max_points=checkpoint["max_points"],
            cache_sequences=args.cache_sequences,
            context_reserve_fraction=args.context_reserve,
            return_indices=True,
        )
        dataset.checkpoint_schema = checkpoint["schema"]
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=_collate,
        )
        records = []
        with torch.no_grad():
            for batch in loader:
                features = batch["features"].to(args.device, non_blocking=True)
                point_mask = batch["point_mask"].to(args.device, non_blocking=True)
                module = checkpoint["model"]
                point_probabilities = (
                    _predict(module, features, point_mask, not args.no_tta)
                    .cpu()
                    .numpy()
                )
                label_masks = batch["label_mask"].numpy()
                for row in range(point_probabilities.shape[0]):
                    selected = label_masks[row]
                    if not np.any(selected):
                        continue
                    sequence_index = int(batch["sequence_index"][row])
                    records.append(
                        {
                            "sequence": np.full(
                                int(selected.sum()),
                                dataset.sequences[sequence_index]["name"],
                                dtype=object,
                            ),
                            "sensor": np.full(
                                int(selected.sum()),
                                int(batch["sensor"][row]),
                                dtype=np.int64,
                            ),
                            "frame": np.full(
                                int(selected.sum()),
                                int(batch["frame"][row]),
                                dtype=np.int64,
                            ),
                            "indices": batch["indices"][row].numpy()[selected],
                            "target": batch["target"][row].numpy()[selected],
                            "class_id": batch["class_id"][row].numpy()[selected],
                            "is_main": batch["is_main"][row].numpy()[selected],
                            "bounce_type": batch["bounce_type"][row]
                            .numpy()[selected],
                            "prob": point_probabilities[row][selected],
                        }
                    )
        if not records:
            raise RuntimeError(f"{path}: no labelled points on split {args.split!r}")
        flat = _flatten_records(records)
        key = {field: flat[field] for field in KEY_FIELDS}
        if reference is None:
            reference = key
            metadata = {
                field: flat[field]
                for field in ("class_id", "is_main", "bounce_type")
            }
            metadata["split"] = np.full(len(flat["target"]), args.split, dtype=object)
        else:
            for field in KEY_FIELDS:
                if len(key[field]) != len(reference[field]) or not np.array_equal(
                    key[field], reference[field]
                ):
                    raise RuntimeError(
                        f"{path}: point identity mismatch in {field!r}; "
                        "all checkpoints must share window size, point budget "
                        "and context reserve"
                    )
        probabilities[name] = flat["prob"].astype(np.float32)
        print(
            f"[dump] {name}: {len(flat['target'])} points, "
            f"{len(dataset.sequences)} sequences, {len(dataset)} windows"
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(reference)
    payload.update(metadata)
    for name, values in probabilities.items():
        payload[f"prob__{name}"] = values
    np.savez_compressed(output, **payload)
    print(f"[dump] wrote {output} ({len(reference['target'])} labelled points)")


def _checkpoint_names(archive):
    return sorted(key[len("prob__") :] for key in archive.files if key.startswith("prob__"))


def _metrics(target, probability, fixed_threshold, max_real_fpr):
    metrics = BinaryHistogramMetrics()
    metrics.update(probability, target)
    return metrics.compute(
        fixed_threshold=float(fixed_threshold),
        max_false_positive_rate=max_real_fpr,
    )


def _fixed_block(result):
    return result["confusion_matrix"]["fixed"]


def _calibrate(target, probability, max_real_fpr):
    metrics = BinaryHistogramMetrics()
    metrics.update(probability, target)
    return metrics.compute(
        fixed_threshold=0.5,
        max_false_positive_rate=max_real_fpr,
    )["operating_threshold"]


def _transform(probability, mode):
    probability = np.clip(np.asarray(probability, dtype=np.float64), 1e-6, 1 - 1e-6)
    if mode == "mean":
        return probability
    if mode == "logit":
        logits = np.log(probability) - np.log1p(-probability)
        return 1.0 / (1.0 + np.exp(-logits.mean(axis=0)))
    if mode == "rank":
        from scipy.stats import rankdata

        ranks = np.empty_like(probability)
        for row in range(probability.shape[0]):
            ranks[row] = rankdata(probability[row]) / probability.shape[1]
        return ranks.mean(axis=0)
    raise ValueError(f"unknown ensemble mode {mode!r}")


def _subset_metrics(
    val_target,
    val_probabilities,
    test_target,
    test_probabilities,
    mode,
    max_real_fpr,
):
    val = _transform(val_probabilities, mode)
    test = _transform(test_probabilities, mode)
    threshold = _calibrate(val, val_target, max_real_fpr)
    val_result = _metrics(val_target, val, threshold, max_real_fpr)
    test_result = _metrics(test_target, test, threshold, max_real_fpr)
    return val_result, test_result


def _scenario_breakdown(sequence, target, probability, threshold, max_real_fpr):
    result = {}
    for name in sorted(set(sequence.tolist())):
        selected = sequence == name
        if not np.any(selected):
            continue
        metrics = BinaryHistogramMetrics()
        metrics.update(probability[selected], target[selected])
        block = metrics.compute(
            fixed_threshold=float(threshold),
            max_false_positive_rate=max_real_fpr,
        )
        result[name] = {
            "auroc": block["auroc"],
            "auprc": block["auprc"],
            "ghost_count": block["ghost_count"],
            "real_count": block["real_count"],
            "fixed": _fixed_block(block),
        }
    return result


def _class_breakdown(class_id, target, probability, threshold, max_real_fpr):
    result = {}
    for value, label in sorted(CLASS_NAMES.items()):
        selected = class_id == value
        if not np.any(selected):
            continue
        metrics = BinaryHistogramMetrics()
        metrics.update(probability[selected], target[selected])
        block = metrics.compute(
            fixed_threshold=float(threshold),
            max_false_positive_rate=max_real_fpr,
        )
        result[label] = {
            "auroc": block["auroc"],
            "auprc": block["auprc"],
            "ghost_count": block["ghost_count"],
            "real_count": block["real_count"],
            "fixed": _fixed_block(block),
        }
    return result


def analyze(args):
    with np.load(args.val, allow_pickle=True) as val_archive, np.load(
        args.test, allow_pickle=True
    ) as test_archive:
        names = _checkpoint_names(val_archive)
        test_names = _checkpoint_names(test_archive)
        if set(names) != set(test_names):
            raise ValueError(
                f"validation/test checkpoint sets differ: {names} vs {test_names}"
            )
        val_target = val_archive["target"].astype(np.int64)
        test_target = test_archive["target"].astype(np.int64)
        val_probability = np.stack(
            [val_archive[f"prob__{name}"] for name in names]
        )
        test_probability = np.stack(
            [test_archive[f"prob__{name}"] for name in names]
        )
        test_sequence = test_archive["sequence"]
        test_class = test_archive["class_id"]

    modes = args.ensemble_mode or ["mean", "logit"]
    report = {
        "validation": str(Path(args.val).resolve()),
        "test": str(Path(args.test).resolve()),
        "max_real_fpr": float(args.max_real_fpr),
        "select_metric": args.select_metric,
        "checkpoints": names,
        "singles": {},
        "ensembles": {},
    }

    for index, name in enumerate(names):
        val_result, test_result = _subset_metrics(
            val_target,
            val_probability[index : index + 1],
            test_target,
            test_probability[index : index + 1],
            "mean",
            args.max_real_fpr,
        )
        fixed = _fixed_block(test_result)
        report["singles"][name] = {
            "val_auroc": val_result["auroc"],
            "val_auprc": val_result["auprc"],
            "val_threshold": val_result["fixed_threshold"],
            "test_auroc": test_result["auroc"],
            "test_auprc": test_result["auprc"],
            "test_fixed": fixed,
            "test_by_class": _class_breakdown(
                test_class,
                test_target,
                test_probability[index],
                fixed["threshold"],
                args.max_real_fpr,
            ),
            "test_by_scenario": _scenario_breakdown(
                test_sequence,
                test_target,
                test_probability[index],
                fixed["threshold"],
                args.max_real_fpr,
            ),
        }
        print(
            f"{name:24s} val AUROC {val_result['auroc']:.4f} "
            f"AUPRC {val_result['auprc']:.4f} | test AUROC "
            f"{test_result['auroc']:.4f} AUPRC {test_result['auprc']:.4f} "
            f"| thr {fixed['threshold']:.4f} recall {fixed['recall']:.3f} "
            f"FPR {fixed['false_positive_rate']:.3f}"
        )

    best = None
    for size in range(1, min(args.max_ensemble_size, len(names)) + 1):
        for subset in itertools.combinations(range(len(names)), size):
            for mode in modes:
                label = f"{mode}:" + "+".join(names[index] for index in subset)
                val_result, test_result = _subset_metrics(
                    val_target,
                    val_probability[list(subset)],
                    test_target,
                    test_probability[list(subset)],
                    mode,
                    args.max_real_fpr,
                )
                score = float(val_result[args.select_metric])
                entry = {
                    "members": [names[index] for index in subset],
                    "mode": mode,
                    "val_auroc": val_result["auroc"],
                    "val_auprc": val_result["auprc"],
                    "val_threshold": val_result["fixed_threshold"],
                    "test_auroc": test_result["auroc"],
                    "test_auprc": test_result["auprc"],
                    "test_fixed": _fixed_block(test_result),
                }
                report["ensembles"][label] = entry
                if best is None or score > best[0]:
                    best = (score, label, entry)

    if best is not None:
        score, label, entry = best
        report["best_by_validation"] = {
            "label": label,
            "selection_score": score,
            **entry,
        }
        print()
        print(
            f"best validation-selected ensemble: {label}\n"
            f"  val {args.select_metric} {score:.4f} | test AUROC "
            f"{entry['test_auroc']:.4f} AUPRC {entry['test_auprc']:.4f}"
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"[analyze] wrote {output}")


def main():
    args = parse_args()
    if args.command == "dump":
        dump(args)
    else:
        analyze(args)


if __name__ == "__main__":
    main()
