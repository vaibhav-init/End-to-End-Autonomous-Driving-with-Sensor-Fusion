#!/usr/bin/env python3
"""
Train MLP for Throttle/Brake Prediction (Dual Output)
======================================================

Reads dataset_throttle_brake/data.csv, trains an MLP that predicts
separate throttle and brake values from radar features.

Output (2):
    throttle ∈ [0, 1]
    brake    ∈ [0, 1]

Optionally applies a time shift to make the model predict EARLIER
than the autopilot reacted (anticipatory braking).

Usage:
    python train_throttle_brake.py
    python train_throttle_brake.py --epochs 200 --shift 10
"""

import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
import time


DATA_PATH = 'dataset_throttle_brake/data.csv'
MODEL_DIR = 'model_throttle_brake'

FEATURE_COLS = [
    'ego_speed', 'target_speed', 'ego_acceleration', 'distance',
    'relative_velocity', 'ttc', 'obstacle_speed',
    'approaching_intersection', 'traffic_light_state',
]
LABEL_COLS = ['autopilot_throttle', 'autopilot_brake']


# ============================================================================
# MLP Model (Dual Output)
# ============================================================================
class ThrottleBrakeMLP(nn.Module):
    """MLP: 9 → 64 → 32 → 2 (sigmoid output for [0, 1] per pedal)"""

    def __init__(self, input_dim=9):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(32, 2),
            nn.Sigmoid(),  # output in [0, 1] for each pedal
        )

    def forward(self, x):
        return self.net(x)  # shape: (batch, 2) → [throttle, brake]


# ============================================================================
# Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description='Train throttle/brake MLP')
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batch', type=int, default=256)
    parser.add_argument('--data', default=DATA_PATH)
    parser.add_argument('--shift', type=int, default=10,
                        help='Shift targets N frames into the future (anticipatory braking). '
                             'Set to 0 to disable.')
    args = parser.parse_args()

    os.makedirs(MODEL_DIR, exist_ok=True)

    # ---- Load data ----
    print("=" * 60)
    print("THROTTLE/BRAKE MLP TRAINER (Dual Output)")
    print("=" * 60)

    df = pd.read_csv(args.data)
    print(f"  Loaded {len(df):,} rows from {args.data}")

    # ---- Time shift (anticipatory braking) ----
    if args.shift > 0:
        print(f"\n  ⏱️  Applying {args.shift}-frame target shift")
        print(f"     Model will learn to react {args.shift/20:.2f}s EARLIER than autopilot")
        df['target_throttle'] = df['autopilot_throttle'].shift(-args.shift)
        df['target_brake'] = df['autopilot_brake'].shift(-args.shift)
        df = df.dropna().reset_index(drop=True)
        label_cols = ['target_throttle', 'target_brake']
        print(f"     {len(df):,} rows after shift")
    else:
        label_cols = LABEL_COLS
        print(f"\n  No time shift applied")

    # ---- Clean data ----
    initial_len = len(df)

    # Drop fully uninformative frames (parked at spawn, no obstacle)
    df = df[~((df['ego_speed'] < 0.1) & (df['distance'] > 49))].reset_index(drop=True)
    dropped = initial_len - len(df)
    if dropped > 0:
        print(f"  Dropped {dropped:,} uninformative idle frames (speed=0, no obstacle)")

    # Cap stopped frames (speed < 0.5 m/s) to max 15% of dataset.
    # Without this, the ego getting stuck behind traffic creates thousands
    # of identical "speed=0, dist=4m, brake=1.0" rows that drown out
    # the critical braking TRANSITION data (speed going from 25→0).
    stopped_mask = df['ego_speed'] < 0.5 / 3.6  # 0.5 km/h in m/s
    moving_df = df[~stopped_mask]
    stopped_df = df[stopped_mask]
    max_stopped = int(len(moving_df) * 0.15 / 0.85)  # 15% of final total
    if len(stopped_df) > max_stopped:
        stopped_df = stopped_df.sample(n=max_stopped, random_state=42)
        print(f"  Downsampled stopped frames: {stopped_mask.sum():,} → {max_stopped:,} "
              f"(capped at 15% of dataset)")
    df = pd.concat([moving_df, stopped_df]).sort_index().reset_index(drop=True)
    print(f"  Final dataset: {len(df):,} rows")

    # Clip extreme values
    df['ego_acceleration'] = df['ego_acceleration'].clip(-20, 20)

    # ---- Features & labels ----
    X = df[FEATURE_COLS].values.astype(np.float32)
    y = df[label_cols].values.astype(np.float32)

    print(f"\n  Features: {FEATURE_COLS}")
    print(f"  Labels:   {label_cols}")
    print(f"  Samples:  {len(y):,}")

    # Distribution analysis
    throttle_vals = y[:, 0]
    brake_vals = y[:, 1]
    braking = (brake_vals > 0.05).sum()
    throttle = (throttle_vals > 0.05).sum()
    idle = len(y) - braking - throttle + ((brake_vals > 0.05) & (throttle_vals > 0.05)).sum()
    print(f"\n  Distribution:")
    print(f"    Braking  (brake > 0.05):    {braking:,} ({100*braking/len(y):.1f}%)")
    print(f"    Throttle (throttle > 0.05): {throttle:,} ({100*throttle/len(y):.1f}%)")
    print(f"    Throttle range: [{throttle_vals.min():.3f}, {throttle_vals.max():.3f}]")
    print(f"    Brake range:    [{brake_vals.min():.3f}, {brake_vals.max():.3f}]")

    # Sequential split — no shuffle! Time-series data with shift
    # would leak future info if randomly shuffled.
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, shuffle=False)

    # ---- Normalize features ----
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    # Save scaler
    scaler_path = os.path.join(MODEL_DIR, 'scaler.pkl')
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"\n  Saved scaler to {scaler_path}")

    # ---- DataLoaders ----
    train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    val_ds = TensorDataset(torch.tensor(X_val), torch.tensor(y_val))

    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch * 2, shuffle=False)

    # ---- Model ----
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}")

    model = ThrottleBrakeMLP(input_dim=len(FEATURE_COLS)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=15)
    criterion = nn.MSELoss()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Model params: {total_params:,}")
    print(f"  Architecture: {len(FEATURE_COLS)} → 64 → 32 → 2 (sigmoid)")
    print(f"  Epochs: {args.epochs}, LR: {args.lr}, Batch: {args.batch}")

    # ---- Training ----
    print(f"\n{'=' * 60}")
    print(f"TRAINING")
    print(f"{'=' * 60}")

    best_val_loss = float('inf')
    best_epoch = 0
    t_start = time.time()

    for epoch in range(1, args.epochs + 1):
        # Train
        model.train()
        train_loss = 0
        train_total = 0

        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = criterion(pred, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(xb)
            train_total += len(xb)

        train_loss /= train_total

        # Validate
        model.eval()
        val_loss = 0
        val_total = 0
        val_preds = []
        val_labels = []

        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                loss = criterion(pred, yb)
                val_loss += loss.item() * len(xb)
                val_total += len(xb)
                val_preds.append(pred.cpu().numpy())
                val_labels.append(yb.cpu().numpy())

        val_loss /= val_total
        scheduler.step(val_loss)

        val_preds_np = np.concatenate(val_preds)
        val_labels_np = np.concatenate(val_labels)

        # MAE per output
        throttle_mae = np.mean(np.abs(val_preds_np[:, 0] - val_labels_np[:, 0]))
        brake_mae = np.mean(np.abs(val_preds_np[:, 1] - val_labels_np[:, 1]))

        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            model_path = os.path.join(MODEL_DIR, 'throttle_brake_mlp.pt')
            torch.save(model.state_dict(), model_path)

        # Print every 10 epochs
        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{args.epochs}  "
                  f"Loss: {train_loss:.6f}/{val_loss:.6f}  "
                  f"THR_MAE: {throttle_mae:.4f}  BRK_MAE: {brake_mae:.4f}")

    elapsed = time.time() - t_start
    print(f"\n  Training done in {elapsed:.1f}s")
    print(f"  Best val loss: {best_val_loss:.6f} at epoch {best_epoch}")

    # ---- Final evaluation ----
    print(f"\n{'=' * 60}")
    print(f"FINAL EVALUATION (best model)")
    print(f"{'=' * 60}")

    model_path = os.path.join(MODEL_DIR, 'throttle_brake_mlp.pt')
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()

    all_preds = []
    all_labels = []
    with torch.no_grad():
        for xb, yb in val_dl:
            xb = xb.to(device)
            pred = model(xb)
            all_preds.append(pred.cpu().numpy())
            all_labels.append(yb.numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    # Overall metrics
    print(f"\n  Throttle:")
    t_preds, t_labels = all_preds[:, 0], all_labels[:, 0]
    print(f"    MSE: {np.mean((t_preds - t_labels) ** 2):.6f}")
    print(f"    MAE: {np.mean(np.abs(t_preds - t_labels)):.4f}")
    print(f"    Pred range: [{t_preds.min():.3f}, {t_preds.max():.3f}]")

    print(f"\n  Brake:")
    b_preds, b_labels = all_preds[:, 1], all_labels[:, 1]
    print(f"    MSE: {np.mean((b_preds - b_labels) ** 2):.6f}")
    print(f"    MAE: {np.mean(np.abs(b_preds - b_labels)):.4f}")
    print(f"    Pred range: [{b_preds.min():.3f}, {b_preds.max():.3f}]")

    # Braking accuracy
    brake_active = b_labels > 0.1
    if brake_active.sum() > 0:
        brake_preds_when_should = b_preds[brake_active]
        brake_correct = (brake_preds_when_should > 0.1).sum()
        print(f"\n  Braking detection:")
        print(f"    Frames where autopilot braked: {brake_active.sum():,}")
        print(f"    Model correctly predicted brake: {brake_correct:,} "
              f"({100*brake_correct/brake_active.sum():.1f}%)")

    print(f"\n  Model saved:  {model_path}")
    print(f"  Scaler saved: {scaler_path}")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
