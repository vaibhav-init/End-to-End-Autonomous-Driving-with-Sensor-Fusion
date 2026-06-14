#!/usr/bin/env python3
"""
Train MLP for Crash Probability Prediction
===========================================

Reads dataset_crash/data.csv, trains a simple MLP, saves model.

Usage:
    python train_mlp.py
    python train_mlp.py --epochs 200 --lr 0.001

Features used (9 inputs):
    ego_speed, ego_acceleration, nearest_distance, relative_velocity,
    ttc, obstacle_speed, obstacle_type, lateral_offset, ego_steering

Output (1):
    crash probability (0.0 - 1.0)
"""

import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
import time

DATA_PATH = 'dataset_crash/data.csv'
MODEL_DIR = 'model'

FEATURE_COLS = [
    'ego_speed', 'ego_acceleration', 'nearest_distance',
    'relative_velocity', 'ttc', 'obstacle_speed',
    'obstacle_type', 'lateral_offset', 'ego_steering',
    'rear_distance', 'rear_relative_velocity', 'rear_ttc',
    'rear_obstacle_speed', 'rear_obstacle_type'
]
LABEL_COL = 'collision_within_2s'


# ============================================================================
# MLP Model
# ============================================================================
class CrashMLP(nn.Module):
    """Simple MLP: 14 → 128 → 64 → 32 → 1"""
    def __init__(self, input_dim=14):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


# ============================================================================
# Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batch', type=int, default=256)
    parser.add_argument('--data', default=DATA_PATH)
    args = parser.parse_args()

    os.makedirs(MODEL_DIR, exist_ok=True)

    # ---- Load data ----
    print("=" * 60)
    print("CRASH MLP TRAINER")
    print("=" * 60)

    df = pd.read_csv(args.data)
    print(f"  Loaded {len(df):,} rows from {args.data}")

    # ---- Remove noise ----
    # Drop frames where ego is completely stationary AND no obstacle nearby
    # (these are uninformative — parked at spawn)
    initial_len = len(df)
    df = df[~((df['ego_speed'] < 0.1) & (df['nearest_distance'] > 40))].reset_index(drop=True)
    dropped = initial_len - len(df)
    if dropped > 0:
        print(f"  Dropped {dropped:,} uninformative idle frames")

    # Clip extreme acceleration spikes
    df['ego_acceleration'] = df['ego_acceleration'].clip(-20, 20)

    # ---- Features & labels ----
    X = df[FEATURE_COLS].values.astype(np.float32)
    y = df[LABEL_COL].values.astype(np.float32)

    pos = int(y.sum())
    neg = len(y) - pos
    print(f"  Positive: {pos:,} ({pos/len(y)*100:.2f}%)")
    print(f"  Negative: {neg:,} ({neg/len(y)*100:.2f}%)")

    if pos < 10:
        print(f"\n  ❌ Too few positive samples ({pos}). Need at least 10 crashes.")
        print(f"     Collect more data with crashes before training.")
        return

    # ---- Train/Val split ----
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    # ---- Normalize features ----
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    # Save scaler for inference
    with open(os.path.join(MODEL_DIR, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)
    print(f"  Saved scaler to {MODEL_DIR}/scaler.pkl")

    # ---- Weighted sampler for class imbalance ----
    pos_train = int(y_train.sum())
    neg_train = len(y_train) - pos_train
    class_weight = neg_train / max(1, pos_train)
    sample_weights = np.where(y_train == 1, class_weight, 1.0)
    sampler = WeightedRandomSampler(
        weights=sample_weights, num_samples=len(sample_weights), replacement=True)

    print(f"  Class weight for positive: {class_weight:.1f}x")

    # ---- DataLoaders ----
    train_ds = TensorDataset(
        torch.tensor(X_train), torch.tensor(y_train))
    val_ds = TensorDataset(
        torch.tensor(X_val), torch.tensor(y_val))

    train_dl = DataLoader(train_ds, batch_size=args.batch, sampler=sampler)
    val_dl = DataLoader(val_ds, batch_size=args.batch * 2, shuffle=False)

    # ---- Model ----
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}")

    model = CrashMLP(input_dim=len(FEATURE_COLS)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=15)
    criterion = nn.BCELoss()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Model params: {total_params:,}")
    print(f"  Epochs: {args.epochs}, LR: {args.lr}, Batch: {args.batch}")

    # ---- Training ----
    print(f"\n{'='*60}")
    print(f"TRAINING")
    print(f"{'='*60}")

    best_val_loss = float('inf')
    best_epoch = 0
    t_start = time.time()

    for epoch in range(1, args.epochs + 1):
        # Train
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        train_tp, train_fp, train_fn = 0, 0, 0

        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = criterion(pred, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(xb)
            preds_binary = (pred > 0.5).float()
            train_correct += (preds_binary == yb).sum().item()
            train_total += len(yb)
            train_tp += ((preds_binary == 1) & (yb == 1)).sum().item()
            train_fp += ((preds_binary == 1) & (yb == 0)).sum().item()
            train_fn += ((preds_binary == 0) & (yb == 1)).sum().item()

        train_loss /= train_total
        train_acc = train_correct / train_total
        train_prec = train_tp / max(1, train_tp + train_fp)
        train_rec = train_tp / max(1, train_tp + train_fn)

        # Validate
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        val_tp, val_fp, val_fn, val_tn = 0, 0, 0, 0

        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                loss = criterion(pred, yb)
                val_loss += loss.item() * len(xb)
                preds_binary = (pred > 0.5).float()
                val_correct += (preds_binary == yb).sum().item()
                val_total += len(yb)
                val_tp += ((preds_binary == 1) & (yb == 1)).sum().item()
                val_fp += ((preds_binary == 1) & (yb == 0)).sum().item()
                val_fn += ((preds_binary == 0) & (yb == 1)).sum().item()
                val_tn += ((preds_binary == 0) & (yb == 0)).sum().item()

        val_loss /= val_total
        val_acc = val_correct / val_total
        val_prec = val_tp / max(1, val_tp + val_fp)
        val_rec = val_tp / max(1, val_tp + val_fn)
        val_f1 = 2 * val_prec * val_rec / max(1e-8, val_prec + val_rec)

        scheduler.step(val_loss)

        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, 'crash_mlp.pt'))

        # Print every 10 epochs
        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{args.epochs}  "
                  f"Loss: {train_loss:.4f}/{val_loss:.4f}  "
                  f"Acc: {train_acc:.3f}/{val_acc:.3f}  "
                  f"Prec: {val_prec:.3f}  Rec: {val_rec:.3f}  F1: {val_f1:.3f}")

    elapsed = time.time() - t_start
    print(f"\n  Training done in {elapsed:.1f}s")
    print(f"  Best val loss: {best_val_loss:.4f} at epoch {best_epoch}")

    # ---- Final evaluation ----
    print(f"\n{'='*60}")
    print(f"FINAL EVALUATION (best model)")
    print(f"{'='*60}")

    model.load_state_dict(torch.load(os.path.join(MODEL_DIR, 'crash_mlp.pt'),
                                      weights_only=True))
    model.eval()

    all_preds = []
    all_labels = []
    with torch.no_grad():
        for xb, yb in val_dl:
            xb = xb.to(device)
            pred = model(xb)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(yb.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Metrics at different thresholds
    print(f"\n  Threshold Analysis:")
    print(f"  {'Thresh':>8s} {'Prec':>8s} {'Recall':>8s} {'F1':>8s} {'FP':>6s} {'FN':>6s}")
    print(f"  {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*6} {'-'*6}")
    for thresh in [0.3, 0.5, 0.7, 0.8, 0.9]:
        pred_bin = (all_preds > thresh).astype(float)
        tp = ((pred_bin == 1) & (all_labels == 1)).sum()
        fp = ((pred_bin == 1) & (all_labels == 0)).sum()
        fn = ((pred_bin == 0) & (all_labels == 1)).sum()
        prec = tp / max(1, tp + fp)
        rec = tp / max(1, tp + fn)
        f1 = 2 * prec * rec / max(1e-8, prec + rec)
        print(f"  {thresh:8.1f} {prec:8.3f} {rec:8.3f} {f1:8.3f} {int(fp):6d} {int(fn):6d}")

    # Distribution of predictions
    pos_preds = all_preds[all_labels == 1]
    neg_preds = all_preds[all_labels == 0]
    print(f"\n  Prediction distribution:")
    print(f"    Safe frames:  avg={neg_preds.mean():.3f}, median={np.median(neg_preds):.3f}")
    if len(pos_preds) > 0:
        print(f"    Crash frames: avg={pos_preds.mean():.3f}, median={np.median(pos_preds):.3f}")

    print(f"\n  Model saved: {MODEL_DIR}/crash_mlp.pt")
    print(f"  Scaler saved: {MODEL_DIR}/scaler.pkl")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
