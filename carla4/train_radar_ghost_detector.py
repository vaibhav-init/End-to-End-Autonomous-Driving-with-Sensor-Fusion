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
from torch.utils.data import DataLoader

from radar.ghost_detection.dataset import (
    PreparedGhostDataset,
    split_label_counts,
)
from radar.ghost_detection.features import FEATURE_NAMES, FEATURE_SCHEMA_VERSION
from radar.ghost_detection.metrics import BinaryHistogramMetrics
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
    parser.add_argument("--num-workers", type=int, default=4)
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


def _loader(dataset, batch_size, workers, shuffle):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
    )


def run_epoch(
    model,
    loader,
    device,
    criterion=None,
    optimizer=None,
    max_real_fpr=0.01,
):
    training = optimizer is not None
    model.train(training)
    metrics = BinaryHistogramMetrics()
    total_loss = 0.0
    labeled_count = 0
    for batch in loader:
        features = batch["features"].to(device, non_blocking=True)
        point_mask = batch["point_mask"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)
        label_mask = batch["label_mask"].to(device, non_blocking=True)
        if not torch.any(label_mask):
            continue
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            logits = model(features, point_mask)
            selected_logits = logits[label_mask]
            selected_targets = targets[label_mask]
            loss = criterion(selected_logits, selected_targets)
            if training:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
        count = int(label_mask.sum().item())
        total_loss += float(loss.item()) * count
        labeled_count += count
        metrics.update(
            torch.sigmoid(selected_logits).detach().cpu().numpy(),
            selected_targets.detach().cpu().numpy(),
        )
    result = metrics.compute(max_false_positive_rate=max_real_fpr)
    result["loss"] = total_loss / max(labeled_count, 1)
    result["labeled_count"] = labeled_count
    return result


def main():
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("epochs and batch-size must be positive")
    if not 0.0 <= args.max_real_fpr <= 1.0:
        raise ValueError("--max-real-fpr must be in [0, 1]")
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
    )
    val_dataset = PreparedGhostDataset(
        args.data,
        args.val_split,
        window_frames=args.window_frames,
        max_points=args.max_points,
        augment=False,
        seed=args.seed,
    )
    train_loader = _loader(
        train_dataset,
        args.batch_size,
        args.num_workers,
        shuffle=True,
    )
    val_loader = _loader(
        val_dataset,
        args.batch_size,
        args.num_workers,
        shuffle=False,
    )

    model_kwargs = _model_kwargs(args)
    model = create_ghost_model(args.model, **model_kwargs).to(args.device)
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
    positive_weight = float(np.clip(real_count / ghost_count, 0.1, 20.0))
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(positive_weight, device=args.device)
    )
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
    best_path = output / "best_detector.pt"
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        train_dataset.set_epoch(epoch)
        train_metrics = run_epoch(
            model,
            train_loader,
            args.device,
            criterion=criterion,
            optimizer=optimizer,
            max_real_fpr=args.max_real_fpr,
        )
        with torch.no_grad():
            val_metrics = run_epoch(
                model,
                val_loader,
                args.device,
                criterion=criterion,
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
            f"val_loss={val_metrics['loss']:.5f} "
            f"val_auprc={val_metrics['auprc']:.5f} "
            f"val_f1={val_metrics['best_f1']:.5f}"
        )
        if val_metrics["auprc"] > best_auprc:
            best_auprc = val_metrics["auprc"]
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

    summary = {
        "best_checkpoint": str(best_path),
        "best_validation_auprc": best_auprc,
        "positive_weight": positive_weight,
        "elapsed_seconds": time.time() - started,
        "arguments": vars(args),
    }
    with (output / "training_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"Best checkpoint: {best_path}")


if __name__ == "__main__":
    main()
