#!/usr/bin/env python3
"""Plot training curves from ghost detector training history.

Reads the history.json produced by train_and_evaluate_ghost.py (or
train_radar_ghost_detector.py) and generates publication-quality plots:
  - Loss curves (train vs val)
  - AUPRC / AUROC curves
  - Precision, Recall, F1 at operating threshold
  - Learning rate schedule
  - Class weighting trajectory

Usage:
  python3 plot_ghost_training.py artifacts/ghost_temporal_official/history.json
  python3 plot_ghost_training.py artifacts/ghost_temporal_official/history.json --output plots/
  python3 plot_ghost_training.py artifacts/ghost_temporal_official/history.json --no-show
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("history", help="path to history.json")
    parser.add_argument(
        "--output", "-o", default=None,
        help="output directory (default: same directory as history.json)",
    )
    parser.add_argument(
        "--no-show", action="store_true",
        help="save plots without displaying",
    )
    parser.add_argument(
        "--dpi", type=int, default=150,
        help="output DPI (default: 150)",
    )
    parser.add_argument(
        "--format", choices=("png", "pdf", "svg"), default="png",
        help="output format (default: png)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    history_path = Path(args.history)
    with history_path.open("r", encoding="utf-8") as fh:
        history = json.load(fh)

    output_dir = Path(args.output) if args.output else history_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    epochs = [r["epoch"] for r in history]
    lrs = [r["learning_rate"] for r in history]

    train_loss = [r["train"]["loss"] for r in history]
    val_loss = [r["val"]["loss"] for r in history]

    train_auprc = [r["train"]["auprc"] for r in history]
    val_auprc = [r["val"]["auprc"] for r in history]

    train_auroc = [r["train"]["auroc"] for r in history]
    val_auroc = [r["val"]["auroc"] for r in history]

    val_f1 = [r["val"]["best_f1"] for r in history]
    val_precision = [r["val"]["operating_precision"] for r in history]
    val_recall = [r["val"]["operating_recall"] for r in history]

    val_pos_weight = [r["train"]["mean_pos_weight"] for r in history]

    # Find best epoch by val AUPRC
    best_idx = val_auprc.index(max(val_auprc))
    best_epoch = epochs[best_idx]

    fmt = args.format
    dpi = args.dpi

    # ---- Figure 1: Loss + AUPRC ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs, train_loss, "b-", label="Train", linewidth=1.5)
    ax1.plot(epochs, val_loss, "r-", label="Val", linewidth=1.5)
    ax1.axvline(best_epoch, color="gray", linestyle="--", alpha=0.5,
                label=f"Best (ep {best_epoch})")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, train_auprc, "b-", label="Train AUPRC", linewidth=1.5)
    ax2.plot(epochs, val_auprc, "r-", label="Val AUPRC", linewidth=1.5)
    ax2.plot(epochs, train_auroc, "b--", label="Train AUROC", linewidth=1, alpha=0.6)
    ax2.plot(epochs, val_auroc, "r--", label="Val AUROC", linewidth=1, alpha=0.6)
    ax2.axvline(best_epoch, color="gray", linestyle="--", alpha=0.5)
    ax2.axhline(max(val_auprc), color="red", linestyle=":", alpha=0.3)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Score")
    ax2.set_title("AUPRC / AUROC")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1.05)

    fig.suptitle("Radar Ghost Detector — Training Curves", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(
        output_dir / f"loss_and_auprc.{fmt}", dpi=dpi,
        bbox_inches="tight",
    )
    print(f"  Saved: {output_dir / f'loss_and_auprc.{fmt}'}")

    # ---- Figure 2: Precision / Recall / F1 ----
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(epochs, val_precision, "g-", label="Precision", linewidth=1.5)
    ax.plot(epochs, val_recall, "m-", label="Recall", linewidth=1.5)
    ax.plot(epochs, val_f1, "k-", label="F1", linewidth=1.5)
    ax.axvline(best_epoch, color="gray", linestyle="--", alpha=0.5,
               label=f"Best (ep {best_epoch})")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Score")
    ax.set_title("Validation: Precision / Recall / F1")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(
        output_dir / f"precision_recall_f1.{fmt}", dpi=dpi,
        bbox_inches="tight",
    )
    print(f"  Saved: {output_dir / f'precision_recall_f1.{fmt}'}")

    # ---- Figure 3: Learning rate + class weight ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))

    ax1.plot(epochs, lrs, "k-", linewidth=1.5)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Learning Rate")
    ax1.set_title("Learning Rate Schedule")
    ax1.set_yscale("log")
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, val_pos_weight, "b-", linewidth=1.5)
    ax2.axhline(1.0, color="gray", linestyle=":", alpha=0.5)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Pos Weight (real/ghost)")
    ax2.set_title("Training Class Weight (per-batch)")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(
        output_dir / f"lr_and_weights.{fmt}", dpi=dpi,
        bbox_inches="tight",
    )
    print(f"  Saved: {output_dir / f'lr_and_weights.{fmt}'}")

    # ---- Summary ----
    print(f"\n  Best epoch: {best_epoch}")
    print(f"  Best val AUPRC: {max(val_auprc):.5f}")
    print(f"  Best val AUROC: {max(val_auroc):.5f}")
    print(f"  Best val F1:    {max(val_f1):.5f}")
    print(f"  Val P/R at best: {val_precision[best_idx]:.4f} / {val_recall[best_idx]:.4f}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
