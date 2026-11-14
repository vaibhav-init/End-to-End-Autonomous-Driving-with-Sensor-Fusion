#!/usr/bin/env python3
"""Weight-average (soup) fine-tuned ghost-detector seeds, or blend with the
synthetic-pretrained anchor (WiSE-FT).

Three modes, all producing one deployable checkpoint with no inference
cost (Wortsman et al., Model Soups, ICML 2022; WiSE-FT, CVPR 2022; Model
Stock, ECCV 2024):

  uniform  average the state dicts of all --checkpoint members
  greedy   start from the individually best member on the validation split,
           add members one by one (best val metric first) while the soup's
           validation metric improves
  wise     average the members, then interpolate with the first --checkpoint
           (the synthetic-pretrained anchor) at --alpha

Every member must share the feature schema, architecture and window
settings; the recipe that produced the matched-study seeds (two-stage
fine-tune, identical hyperparameters apart from the seed) satisfies this.

Example:
  python3 soup_ghost_checkpoints.py \
    --checkpoint s7=artifacts/s2r_matched/ft_twostage_s7/best_detector.pt \
    --checkpoint s42=artifacts/s2r_matched/ft_twostage_s42/best_detector.pt \
    ... \
    --data artifacts/ghost_real_official_v2 \
    --mode greedy --output artifacts/soup_matched_greedy
"""

import argparse
import copy
import json
from pathlib import Path

import torch

from evaluate_radar_ghost_detector import evaluate_checkpoint, load_checkpoint
from radar.ghost_detection.features import FEATURE_SCHEMA_VERSION


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        action="append",
        required=True,
        help="name=path; repeat for each soup member. In wise mode the "
        "first member is the pretrained anchor",
    )
    parser.add_argument("--data", required=True, help="prepared dataset")
    parser.add_argument("--mode", choices=("uniform", "greedy", "wise"),
                        default="uniform")
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="wise mode: weight on the fine-tuned soup vs the anchor",
    )
    parser.add_argument("--val-split", default="val")
    parser.add_argument(
        "--test-split",
        default="test",
        help="evaluated for the report only; never used for soup decisions",
    )
    parser.add_argument(
        "--select-metric",
        choices=("auprc", "auroc"),
        default="auprc",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--output", required=True, help="output run directory")
    return parser.parse_args()


def _parse_members(specs):
    members = []
    for spec in specs:
        if "=" not in spec:
            raise SystemExit(f"--checkpoint must be name=path, got {spec!r}")
        name, path = spec.split("=", 1)
        if not name or not path:
            raise SystemExit(f"--checkpoint must be name=path, got {spec!r}")
        members.append((name, path))
    return members


def _average_state_dicts(state_dicts, weights=None):
    if weights is None:
        weights = [1.0 / len(state_dicts)] * len(state_dicts)
    keys = state_dicts[0].keys()
    averaged = {}
    for key in keys:
        first = state_dicts[0][key]
        if not torch.is_floating_point(first):
            # Buffers/counters are copied from the first member instead of
            # averaged: they are batch-norm running stats or int state, and
            # every member shares the architecture.
            averaged[key] = first.clone()
            continue
        accumulator = torch.zeros_like(first, dtype=torch.float64)
        for state, weight in zip(state_dicts, weights):
            accumulator += state[key].to(torch.float64) * weight
        averaged[key] = accumulator.to(first.dtype)
    return averaged


def _metric(result, name):
    return float(result[name]) if name in result else float("nan")


def main():
    args = parse_args()
    members = _parse_members(args.checkpoint)
    checkpoints = []
    for name, path in members:
        checkpoint = load_checkpoint(path, args.device)
        if checkpoint.get("feature_schema") != FEATURE_SCHEMA_VERSION:
            raise SystemExit(f"{path}: feature schema mismatch")
        checkpoints.append((name, path, checkpoint))
    reference = checkpoints[0][2]
    for name, path, checkpoint in checkpoints[1:]:
        if checkpoint.get("model_name") != reference.get("model_name"):
            raise SystemExit(f"{path}: model architecture differs")
        if checkpoint.get("model_kwargs") != reference.get("model_kwargs"):
            raise SystemExit(f"{path}: model kwargs differ")
        if checkpoint.get("window_frames") != reference.get("window_frames"):
            raise SystemExit(f"{path}: window_frames differ")
        if checkpoint.get("max_points") != reference.get("max_points"):
            raise SystemExit(f"{path}: max_points differ")

    def build(state_dict, threshold):
        souped = copy.deepcopy(reference)
        souped["model_state"] = state_dict
        if threshold is not None:
            souped["threshold"] = float(threshold)
        return souped

    if args.mode in ("uniform", "wise"):
        states = [entry[2]["model_state"] for entry in checkpoints]
        averaged = _average_state_dicts(states)
        if args.mode == "wise":
            anchor = checkpoints[0][2]["model_state"]
            averaged = _average_state_dicts(
                [averaged, anchor],
                weights=[float(args.alpha), 1.0 - float(args.alpha)],
            )
        souped = build(averaged, reference.get("threshold"))
        members_used = [entry[0] for entry in checkpoints]
    else:
        # Greedy soup: score each member on the validation split, then add
        # members while the soup's validation metric improves. Test is only
        # evaluated once at the end.
        scored = []
        for name, path, checkpoint in checkpoints:
            result = evaluate_checkpoint(
                args.data,
                checkpoint,
                args.val_split,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                device=args.device,
            )
            scored.append(
                (_metric(result, args.select_metric), name, checkpoint)
            )
            print(
                f"[member] {name:12s} val {args.select_metric}="
                f"{scored[-1][0]:.4f}"
            )
        scored.sort(key=lambda entry: entry[0], reverse=True)
        soup_states = [scored[0][2]["model_state"]]
        members_used = [scored[0][1]]
        best_score = scored[0][0]
        for score, name, checkpoint in scored[1:]:
            candidate = _average_state_dicts(
                soup_states + [checkpoint["model_state"]]
            )
            result = evaluate_checkpoint(
                args.data,
                build(candidate, reference.get("threshold")),
                args.val_split,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                device=args.device,
            )
            candidate_score = _metric(result, args.select_metric)
            if candidate_score > best_score:
                soup_states.append(checkpoint["model_state"])
                members_used.append(name)
                best_score = candidate_score
                print(
                    f"[soup  ] +{name:12s} val {args.select_metric}="
                    f"{candidate_score:.4f} (accepted)"
                )
            else:
                print(
                    f"[soup  ] +{name:12s} val {args.select_metric}="
                    f"{candidate_score:.4f} (rejected)"
                )
        souped = build(_average_state_dicts(soup_states),
                       reference.get("threshold"))

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output / "best_detector.pt"
    torch.save(souped, checkpoint_path)

    report = {
        "mode": args.mode,
        "alpha": args.alpha if args.mode == "wise" else None,
        "members": members_used,
        "members_considered": [entry[0] for entry in checkpoints],
        "select_metric": args.select_metric,
        "data": str(Path(args.data).resolve()),
        "checkpoint": str(checkpoint_path.resolve()),
    }
    for split_name, split in (("val", args.val_split),
                              ("test", args.test_split)):
        result = evaluate_checkpoint(
            args.data,
            souped,
            split,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=args.device,
        )
        report[f"{split_name}_metrics"] = result
        print(
            f"[{split_name}] AUROC {result['auroc']:.4f} "
            f"AUPRC {result['auprc']:.4f} "
            f"fixed recall {result['confusion_matrix']['fixed']['recall']:.4f} "
            f"FPR {result['confusion_matrix']['fixed']['false_positive_rate']:.4f}"
        )
    (output / "soup_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(f"wrote {checkpoint_path} and {output / 'soup_report.json'}")


if __name__ == "__main__":
    main()
