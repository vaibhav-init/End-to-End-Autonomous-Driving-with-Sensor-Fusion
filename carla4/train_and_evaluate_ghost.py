#!/usr/bin/env python3
"""Train a radar ghost detector on the official RGD split and evaluate on test.

Runs the full pipeline in one pass:
  1. Train TemporalPointNet on the train split (augmented)
  2. Validate each epoch on the val split (no augmentation)
  3. After training, load the best checkpoint and evaluate on the test split
  4. Print and save all metrics

Usage (run on remote machine):
  cd carla4
  python3 train_and_evaluate_ghost.py \
    --data artifacts/ghost_real_official \
    --output artifacts/ghost_temporal_official
"""

import argparse
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from radar.ghost_detection.dataset import (
    PreparedGhostDataset,
    split_label_counts,
)
from radar.ghost_detection.features import FEATURE_NAMES, FEATURE_SCHEMA_VERSION
from radar.ghost_detection.metrics import BinaryHistogramMetrics
from radar.ghost_detection.model import create_ghost_model


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # Data
    parser.add_argument(
        "--data", required=True,
        help="prepared dataset directory (must contain train/val/test splits)",
    )
    parser.add_argument(
        "--output", required=True,
        help="directory for checkpoints, history, and test metrics",
    )
    # Model
    parser.add_argument(
        "--model", choices=("point_mlp", "temporal_pointnet"),
        default="temporal_pointnet",
    )
    parser.add_argument("--window-frames", type=int, default=5)
    parser.add_argument("--max-points", type=int, default=1024)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--context-dim", type=int, default=192)
    parser.add_argument("--dropout", type=float, default=0.15)
    # Training
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--max-real-fpr", type=float, default=0.01,
        help="max validation false-positive rate for operating threshold",
    )
    # Evaluation
    parser.add_argument(
        "--eval-batch-size", type=int, default=16,
        help="batch size for final test evaluation",
    )
    parser.add_argument(
        "--threshold", type=float, default=None,
        help="override deployment threshold (default: from best checkpoint)",
    )
    # System
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--cycle-time", type=float, default=0.05)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _model_kwargs(args):
    common = {
        "input_dim": len(FEATURE_NAMES),
        "hidden_dim": args.hidden_dim,
        "dropout": args.dropout,
    }
    if args.model == "temporal_pointnet":
        common["context_dim"] = args.context_dim
    return common


def _make_loader(dataset, batch_size, num_workers, shuffle):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )


# ---------------------------------------------------------------------------
# One-epoch runner
# ---------------------------------------------------------------------------

def run_epoch(model, loader, device, static_pos_weight=1.0, optimizer=None,
              max_real_fpr=0.01):
    """Run one training or validation epoch.

    Training batches use per-batch class weighting (pos_weight = real/ghost
    clipped to [0.1, 20]) so a class-imbalanced batch does not dominate.
    Validation uses the dataset-level static weight for comparable loss.
    """
    training = optimizer is not None
    model.train(training)
    metrics = BinaryHistogramMetrics()
    total_loss = 0.0
    labeled_count = 0
    weight_sum = 0.0
    weight_count = 0

    for batch in loader:
        features = batch["features"].to(device, non_blocking=True)
        point_mask = batch["point_mask"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)
        label_mask = batch["label_mask"].to(device, non_blocking=True)

        if not torch.any(label_mask):
            continue

        with torch.set_grad_enabled(training):
            logits = model(features, point_mask)
            selected_logits = logits[label_mask]
            selected_targets = targets[label_mask]

            if training:
                real_count = int((selected_targets == 0).sum().item())
                ghost_count = int((selected_targets == 1).sum().item())
                pos_weight = float(
                    np.clip(real_count / max(ghost_count, 1), 0.1, 20.0)
                )
            else:
                pos_weight = float(static_pos_weight)

            loss = F.binary_cross_entropy_with_logits(
                selected_logits,
                selected_targets,
                pos_weight=torch.tensor(pos_weight, device=device),
            )

            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

        count = int(label_mask.sum().item())
        total_loss += float(loss.item()) * count
        labeled_count += count
        weight_sum += pos_weight
        weight_count += 1
        metrics.update(
            torch.sigmoid(selected_logits).detach().cpu().numpy(),
            selected_targets.detach().cpu().numpy(),
        )

    result = metrics.compute(max_false_positive_rate=max_real_fpr)
    result["loss"] = total_loss / max(labeled_count, 1)
    result["labeled_count"] = labeled_count
    result["mean_pos_weight"] = weight_sum / max(weight_count, 1)
    return result


# ---------------------------------------------------------------------------
# Test-set evaluation (richer than training validation)
# ---------------------------------------------------------------------------

def evaluate_test(model, dataset, loader, device, threshold, output_dir):
    """Full evaluation on the test split with per-scenario and per-family
    breakdowns.  Returns a serialisable metrics dict."""
    model.eval()
    metrics = BinaryHistogramMetrics()
    ghost_groups = {}
    real_class_groups = {}
    scenario_metrics = {}

    def _inc_rate(group, key, predicted_positive):
        counts = group.setdefault(key, [0, 0])
        counts[0] += 1
        counts[1] += int(predicted_positive)

    with torch.no_grad():
        for batch in loader:
            features = batch["features"].to(device, non_blocking=True)
            point_mask = batch["point_mask"].to(device, non_blocking=True)
            logits = model(features, point_mask).cpu()
            prob = torch.sigmoid(logits).numpy()
            mask = batch["label_mask"].numpy()
            tgt = batch["target"].numpy()
            cls = batch["class_id"].numpy()
            btype = batch["bounce_type"].numpy()
            border = batch["bounce_order"].numpy()
            seq_idx = batch["sequence_index"].numpy()

            for row in range(len(seq_idx)):
                m = mask[row]
                p = prob[row][m]
                t = tgt[row][m]
                metrics.update(p, t)

                # Per-scenario
                seq_record = dataset.sequences[int(seq_idx[row])]
                scenario = seq_record.get("scenario", seq_record["name"])
                sm = scenario_metrics.setdefault(
                    scenario, BinaryHistogramMetrics()
                )
                sm.update(p, t)

                # Per bounce-family and per real-class false-positive rate
                c = cls[row][m]
                bt = btype[row][m]
                bo = border[row][m]
                pred = p >= threshold
                for ti, ci, bti, boi, pi in zip(t, c, bt, bo, pred):
                    if int(ti) == 0:
                        _inc_rate(real_class_groups, str(int(ci)), pi)
                        continue
                    if bti == 1 and boi == 2:
                        fam = "type1_second"
                    elif bti == 2 and boi == 2:
                        fam = "type2_second"
                    elif bti == 2 and boi == 4:
                        fam = "type2_third"
                    elif boi in (3, 6):
                        fam = "ambiguous_order"
                    elif bti == 0 or boi == 0:
                        fam = "generic_multipath"
                    else:
                        fam = "other_multipath"
                    _inc_rate(ghost_groups, fam, pi)

    result = metrics.compute(fixed_threshold=threshold)
    result["threshold"] = threshold
    result["ghost_recall_by_bounce"] = {
        name: {"count": counts[0], "recall": counts[1] / max(counts[0], 1)}
        for name, counts in sorted(ghost_groups.items())
    }
    result["real_false_positive_rate_by_class"] = {
        name: {
            "count": counts[0],
            "false_positive_rate": counts[1] / max(counts[0], 1),
        }
        for name, counts in sorted(real_class_groups.items())
    }
    result["per_scenario"] = {
        name: sm.compute(fixed_threshold=threshold)
        for name, sm in sorted(scenario_metrics.items())
    }

    # Save
    metrics_path = output_dir / "test_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, sort_keys=True)
        fh.write("\n")
    return result


# ---------------------------------------------------------------------------
# Pretty-print helpers
# ---------------------------------------------------------------------------

def _print_epoch_summary(epoch, lr, train, val):
    print(
        f"  epoch {epoch:03d}  lr={lr:.2e}  "
        f"train_loss={train['loss']:.5f}  "
        f"train_pos_wt={train['mean_pos_weight']:.3f}  "
        f"val_loss={val['loss']:.5f}  "
        f"val_auprc={val['auprc']:.5f}  "
        f"val_f1={val['best_f1']:.5f}  "
        f"val_op_prec={val['operating_precision']:.4f}  "
        f"val_op_recall={val['operating_recall']:.4f}"
    )


def _print_test_report(result):
    print()
    print("=" * 70)
    print("  TEST SET RESULTS")
    print("=" * 70)
    print(f"  AUPRC:              {result['auprc']:.5f}")
    print(f"  AUROC:              {result['auroc']:.5f}")
    print(f"  Best F1:            {result['best_f1']:.5f}  "
          f"(threshold={result['best_threshold']:.4f})")
    print(f"  Operating:          P={result['operating_precision']:.4f}  "
          f"R={result['operating_recall']:.4f}  "
          f"FPR={result['operating_false_positive_rate']:.5f}  "
          f"(threshold={result['operating_threshold']:.4f})")
    print(f"  At threshold={result['threshold']:.4f}:  "
          f"P={result['precision']:.4f}  "
          f"R={result['recall']:.4f}  "
          f"FPR={result['false_positive_rate']:.5f}")
    print(f"  Real points:        {result['real_count']:,}")
    print(f"  Ghost points:       {result['ghost_count']:,}")
    print()
    print("  Ghost recall by bounce family:")
    for fam, info in result.get("ghost_recall_by_bounce", {}).items():
        print(f"    {fam:<24s}  n={info['count']:>5d}  recall={info['recall']:.4f}")
    print()
    print("  Real false-positive rate by class:")
    for cls, info in result.get("real_false_positive_rate_by_class", {}).items():
        print(f"    class={cls}  n={info['count']:>5d}  "
              f"FPR={info['false_positive_rate']:.5f}")
    print()
    print("  Per-scenario AUPRC:")
    for scenario, info in result.get("per_scenario", {}).items():
        print(f"    {scenario:<30s}  AUPRC={info['auprc']:.5f}  "
              f"F1={info['best_f1']:.5f}")
    print("=" * 70)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("epochs and batch-size must be positive")
    if not 0.0 <= args.max_real_fpr <= 1.0:
        raise ValueError("--max-real-fpr must be in [0, 1]")

    seed_everything(args.seed)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load datasets
    # ------------------------------------------------------------------
    print("=" * 70)
    print("  GHOST DETECTOR — TRAIN + EVALUATE")
    print("=" * 70)
    print(f"  Data:          {args.data}")
    print(f"  Output:        {args.output}")
    print(f"  Model:         {args.model}")
    print(f"  Window:        {args.window_frames} frames")
    print(f"  Max points:    {args.max_points}")
    print(f"  Epochs:        {args.epochs}")
    print(f"  Batch:         {args.batch_size}")
    print(f"  LR:            {args.learning_rate}")
    print(f"  Max real FPR:  {args.max_real_fpr}")
    print(f"  Device:        {args.device}")
    print(f"  Seed:          {args.seed}")
    print("=" * 70)

    model_kwargs = _model_kwargs(args)

    print("\n  Loading train split ...")
    train_dataset = PreparedGhostDataset(
        args.data, "train",
        window_frames=args.window_frames,
        max_points=args.max_points,
        augment=True, seed=args.seed,
    )
    print(f"    {len(train_dataset):,} samples")

    print("  Loading val split ...")
    val_dataset = PreparedGhostDataset(
        args.data, "val",
        window_frames=args.window_frames,
        max_points=args.max_points,
        augment=False, seed=args.seed,
    )
    print(f"    {len(val_dataset):,} samples")

    print("  Loading test split ...")
    test_dataset = PreparedGhostDataset(
        args.data, "test",
        window_frames=args.window_frames,
        max_points=args.max_points,
        augment=False, seed=args.seed,
    )
    print(f"    {len(test_dataset):,} samples")

    train_loader = _make_loader(
        train_dataset, args.batch_size, args.num_workers, shuffle=True,
    )
    val_loader = _make_loader(
        val_dataset, args.batch_size, args.num_workers, shuffle=False,
    )
    test_loader = _make_loader(
        test_dataset, args.eval_batch_size, args.num_workers, shuffle=False,
    )

    # Class balance report
    real_count, ghost_count = split_label_counts(
        train_dataset.manifest, "train",
    )
    if real_count == 0 or ghost_count == 0:
        raise ValueError(
            f"Training split needs both classes; "
            f"real={real_count}, ghost={ghost_count}"
        )
    positive_weight = float(np.clip(real_count / ghost_count, 0.1, 20.0))
    print(f"\n  Train class balance: real={real_count:,}  ghost={ghost_count:,}  "
          f"pos_weight={positive_weight:.3f}")

    # ------------------------------------------------------------------
    # 2. Build model
    # ------------------------------------------------------------------
    print(f"\n  Building {args.model} ...")
    model = create_ghost_model(args.model, **model_kwargs).to(args.device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"    Parameters: {param_count:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=6,
    )

    # ------------------------------------------------------------------
    # 3. Train
    # ------------------------------------------------------------------
    print("\n  Training ...")
    history = []
    best_auprc = -1.0
    best_threshold = 0.5
    best_epoch = 0
    best_checkpoint_path = output_dir / "best_detector.pt"
    started = time.time()

    for epoch in range(1, args.epochs + 1):
        train_dataset.set_epoch(epoch)
        train_result = run_epoch(
            model, train_loader, args.device,
            static_pos_weight=positive_weight,
            optimizer=optimizer, max_real_fpr=args.max_real_fpr,
        )
        with torch.no_grad():
            val_result = run_epoch(
                model, val_loader, args.device,
                static_pos_weight=positive_weight,
                max_real_fpr=args.max_real_fpr,
            )

        lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_result["auprc"])

        epoch_record = {
            "epoch": epoch,
            "learning_rate": lr,
            "train": train_result,
            "val": val_result,
        }
        history.append(epoch_record)
        _print_epoch_summary(epoch, lr, train_result, val_result)

        # Save best by val AUPRC
        if val_result["auprc"] > best_auprc:
            best_auprc = val_result["auprc"]
            best_threshold = val_result["operating_threshold"]
            best_epoch = epoch
            checkpoint = {
                "schema_version": 1,
                "model_name": args.model,
                "model_kwargs": model_kwargs,
                "model_state": model.state_dict(),
                "feature_schema": FEATURE_SCHEMA_VERSION,
                "feature_names": list(FEATURE_NAMES),
                "window_frames": args.window_frames,
                "max_points": args.max_points,
                "cycle_time_s": args.cycle_time,
                "threshold": val_result["operating_threshold"],
                "validation_metrics": val_result,
                "epoch": epoch,
                "training_data": str(Path(args.data).resolve()),
            }
            torch.save(checkpoint, best_checkpoint_path)
            print(f"    ★ new best (val_auprc={best_auprc:.5f}, "
                  f"threshold={best_threshold:.4f})")

        # Write history after every epoch (safe to resume from)
        with (output_dir / "history.json").open("w", encoding="utf-8") as fh:
            json.dump(history, fh, indent=2)
            fh.write("\n")

    elapsed = time.time() - started
    print(f"\n  Training complete: {args.epochs} epochs in {elapsed:.1f}s")
    print(f"  Best epoch: {best_epoch}  val_AUPRC={best_auprc:.5f}  "
          f"threshold={best_threshold:.4f}")
    print(f"  Checkpoint: {best_checkpoint_path}")

    # ------------------------------------------------------------------
    # 4. Evaluate on test split
    # ------------------------------------------------------------------
    print("\n  Loading best checkpoint for test evaluation ...")
    checkpoint = torch.load(
        best_checkpoint_path, map_location=args.device, weights_only=True,
    )
    model_eval = create_ghost_model(
        checkpoint["model_name"], **checkpoint.get("model_kwargs", {}),
    ).to(args.device)
    model_eval.load_state_dict(checkpoint["model_state"])
    model_eval.eval()

    eval_threshold = (
        args.threshold
        if args.threshold is not None
        else checkpoint.get("threshold", 0.5)
    )

    print(f"  Evaluating on test split (threshold={eval_threshold:.4f}) ...")
    test_result = evaluate_test(
        model_eval, test_dataset, test_loader, args.device,
        eval_threshold, output_dir,
    )

    _print_test_report(test_result)

    # ------------------------------------------------------------------
    # 5. Save training summary
    # ------------------------------------------------------------------
    summary = {
        "best_checkpoint": str(best_checkpoint_path),
        "best_epoch": best_epoch,
        "best_validation_auprc": best_auprc,
        "test_threshold": eval_threshold,
        "test_auprc": test_result["auprc"],
        "test_auroc": test_result["auroc"],
        "test_best_f1": test_result["best_f1"],
        "test_precision": test_result["precision"],
        "test_recall": test_result["recall"],
        "test_false_positive_rate": test_result["false_positive_rate"],
        "static_positive_weight": positive_weight,
        "batch_class_weighting": True,
        "elapsed_seconds": elapsed,
        "arguments": vars(args),
    }
    with (output_dir / "training_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, sort_keys=True)
        fh.write("\n")
    print(f"\n  All outputs saved to: {output_dir}/")
    print(f"    best_detector.pt      — model checkpoint")
    print(f"    test_metrics.json     — full test evaluation")
    print(f"    training_summary.json — run summary")
    print(f"    history.json          — per-epoch train/val metrics")


if __name__ == "__main__":
    main()
