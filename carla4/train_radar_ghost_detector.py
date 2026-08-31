#!/usr/bin/env python3
"""Train a real-vs-multipath radar point classifier."""

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
from radar.ghost_detection.metrics import (
    BinaryHistogramMetrics,
    format_all_confusion_matrices,
)
from radar.ghost_detection.model import create_ghost_model


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="prepared dataset directory")
    parser.add_argument("--output", required=True, help="checkpoint/output directory")
    parser.add_argument(
        "--model",
        choices=("point_mlp", "temporal_pointnet"),
        default="temporal_pointnet",
    )
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--val-split", default="val")
    parser.add_argument("--window-frames", type=int, default=5)
    parser.add_argument("--max-points", type=int, default=1024)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--context-dim", type=int, default=192)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument(
        "--max-real-fpr",
        type=float,
        default=0.01,
        help="maximum validation false-rejection rate for deployment threshold",
    )
    parser.add_argument(
        "--label-smoothing",
        type=float,
        default=0.02,
        help=(
            "training-target smoothing in [0, 0.49]; prevents probabilities "
            "from saturating at 1.0, which sets a hard floor on the "
            "achievable false-positive rate (0 disables)"
        ),
    )
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--cache-sequences",
        type=int,
        default=64,
        help=(
            "sequences kept in each dataloader worker's memory cache. Must "
            "exceed the split's sequence count: v2 frame statistics are "
            "recomputed on every cache miss, so a cache smaller than the "
            "shuffled sequence count thrashes and slows epochs by orders of "
            "magnitude. Roughly 100-200 MB RAM per cached sequence."
        ),
    )
    parser.add_argument(
        "--context-reserve",
        type=float,
        default=0.25,
        help=(
            "share of --max-points held for older scans so the age feature "
            "stays informative on dense exports (0 restores the old "
            "current-scan-only behaviour)"
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--pretrained",
        help="checkpoint used to initialize the model for sim-to-real fine-tuning",
    )
    parser.add_argument(
        "--cycle-time",
        type=float,
        default=0.05,
        help="runtime fallback cycle time stored with the checkpoint",
    )
    return parser.parse_args()


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


def _loader(dataset, batch_size, workers, shuffle, persistent=True):
    # Persistent workers keep their own copy of the dataset, so a later
    # set_epoch() on the parent object never reaches them. The training loader
    # therefore runs non-persistent: otherwise the augmentation RNG is frozen
    # at epoch 0 and every epoch replays identical augmentations.
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=persistent and workers > 0,
    )


def run_epoch(
    model,
    loader,
    device,
    static_pos_weight=1.0,
    optimizer=None,
    max_real_fpr=0.01,
    label_smoothing=0.0,
):
    """Run one epoch over a loader.

    Training batches apply class weighting computed from that batch's own
    real/ghost frequencies (``pos_weight = real_count / ghost_count``,
    clipped to 0.1-20), so a ghost-heavy synthetic prior (CARLA is ~3:1
    ghost:real) cannot carry into real-data fine-tuning or inflate the
    deployment false-positive rate. Validation uses the dataset-level static
    weight so the reported loss stays comparable across epochs.

    ``label_smoothing`` maps targets from {0, 1} to {smoothing, 1-smoothing}
    during training. Without it the network can drive probabilities to
    exactly 1.0 on confident mistakes; every such point sits in the top
    histogram bin and sets a hard floor on the achievable false-positive
    rate, which is what saturated the v1 run's operating threshold at 0.9995.
    Metrics are always computed against the *unsmoothed* targets.
    """

    training = optimizer is not None
    model.train(training)
    metrics = BinaryHistogramMetrics()
    total_loss = 0.0
    labeled_count = 0
    weight_sum = 0.0
    weight_count = 0
    smoothing = float(np.clip(label_smoothing, 0.0, 0.49))
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
                # Class frequencies measured from this training batch only.
                real_count = int((selected_targets == 0).sum().item())
                ghost_count = int((selected_targets == 1).sum().item())
                pos_weight = float(
                    np.clip(real_count / max(ghost_count, 1), 0.1, 20.0)
                )
            else:
                pos_weight = float(static_pos_weight)
            loss_targets = selected_targets
            if training and smoothing > 0.0:
                loss_targets = selected_targets * (1.0 - smoothing) + (
                    1.0 - selected_targets
                ) * smoothing
            loss = F.binary_cross_entropy_with_logits(
                selected_logits,
                loss_targets,
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


def main():
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("epochs and batch-size must be positive")
    if not 0.0 <= args.max_real_fpr <= 1.0:
        raise ValueError("--max-real-fpr must be in [0, 1]")
    if not 0.0 <= args.label_smoothing <= 0.49:
        raise ValueError("--label-smoothing must be in [0, 0.49]")
    seed_everything(args.seed)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    train_dataset = PreparedGhostDataset(
        args.data,
        args.train_split,
        window_frames=args.window_frames,
        max_points=args.max_points,
        augment=True,
        seed=args.seed,
        cache_sequences=args.cache_sequences,
        context_reserve_fraction=args.context_reserve,
    )
    val_dataset = PreparedGhostDataset(
        args.data,
        args.val_split,
        window_frames=args.window_frames,
        max_points=args.max_points,
        augment=False,
        seed=args.seed,
        cache_sequences=args.cache_sequences,
        context_reserve_fraction=args.context_reserve,
    )
    train_loader = _loader(
        train_dataset,
        args.batch_size,
        args.num_workers,
        shuffle=True,
        persistent=False,
    )
    val_loader = _loader(
        val_dataset,
        args.batch_size,
        args.num_workers,
        shuffle=False,
    )

    model_kwargs = _model_kwargs(args)
    model = create_ghost_model(args.model, **model_kwargs).to(args.device)
    device_label = str(args.device)
    if device_label.startswith("cuda"):
        device_label = (
            f"{args.device} ({torch.cuda.get_device_name(0)})"
        )
    print(
        f"training on: {device_label} | "
        f"model: {args.model} (input_dim={len(FEATURE_NAMES)}) | "
        f"train samples/epoch: {len(train_dataset)} | "
        f"label smoothing: {args.label_smoothing}"
    )
    if args.pretrained:
        try:
            pretrained = torch.load(
                args.pretrained,
                map_location=args.device,
                weights_only=True,
            )
        except TypeError:
            pretrained = torch.load(args.pretrained, map_location=args.device)
        if pretrained.get("feature_schema") != FEATURE_SCHEMA_VERSION:
            raise ValueError("Pretrained feature schema is incompatible")
        if pretrained.get("model_name") != args.model:
            raise ValueError("Pretrained model architecture does not match --model")
        model.load_state_dict(pretrained["model_state"])

    real_count, ghost_count = split_label_counts(
        train_dataset.manifest,
        args.train_split,
    )
    if real_count == 0 or ghost_count == 0:
        raise ValueError(
            f"Training split needs both classes; real={real_count}, ghost={ghost_count}"
        )
    # Static dataset-level weight: used for the validation loss and as the
    # default when a training batch is degenerate. During training each batch
    # re-derives pos_weight from its own class frequencies instead (see
    # run_epoch), so the prior follows the data actually seen.
    positive_weight = float(np.clip(real_count / ghost_count, 0.1, 20.0))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=6,
    )

    history = []
    best_auprc = -1.0
    best_val_metrics = {}
    best_path = output / "best_detector.pt"
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        train_dataset.set_epoch(epoch)
        train_metrics = run_epoch(
            model,
            train_loader,
            args.device,
            static_pos_weight=positive_weight,
            optimizer=optimizer,
            max_real_fpr=args.max_real_fpr,
            label_smoothing=args.label_smoothing,
        )
        with torch.no_grad():
            val_metrics = run_epoch(
                model,
                val_loader,
                args.device,
                static_pos_weight=positive_weight,
                max_real_fpr=args.max_real_fpr,
            )
        scheduler.step(val_metrics["auprc"])
        epoch_record = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train": train_metrics,
            "val": val_metrics,
        }
        history.append(epoch_record)
        print(
            f"epoch {epoch:03d} "
            f"train_loss={train_metrics['loss']:.5f} "
            f"train_pos_weight={train_metrics['mean_pos_weight']:.3f} "
            f"val_loss={val_metrics['loss']:.5f} "
            f"val_auprc={val_metrics['auprc']:.5f} "
            f"val_f1={val_metrics['best_f1']:.5f}"
        )
        if val_metrics["auprc"] > best_auprc:
            best_auprc = val_metrics["auprc"]
            best_val_metrics = val_metrics
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
                "threshold": val_metrics["operating_threshold"],
                "validation_metrics": val_metrics,
                "epoch": epoch,
                "training_data": str(Path(args.data).resolve()),
                "pretrained": args.pretrained,
            }
            torch.save(checkpoint, best_path)

        with (output / "history.json").open("w", encoding="utf-8") as handle:
            json.dump(history, handle, indent=2)
            handle.write("\n")

    print()
    print("=== best validation confusion matrices ===")
    print(format_all_confusion_matrices(best_val_metrics))
    print()

    summary = {
        "best_checkpoint": str(best_path),
        "best_validation_auprc": best_auprc,
        "best_validation_confusion_matrix": best_val_metrics.get(
            "confusion_matrix"
        ),
        "static_positive_weight": positive_weight,
        "batch_class_weighting": True,
        "elapsed_seconds": time.time() - started,
        "arguments": vars(args),
    }
    with (output / "training_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"Best checkpoint: {best_path}")


if __name__ == "__main__":
    main()
