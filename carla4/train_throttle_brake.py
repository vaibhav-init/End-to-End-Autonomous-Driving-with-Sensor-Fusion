#!/usr/bin/env python3
"""
Train a sequence-aware MLP that predicts desired target speed.
"""

import argparse
import glob
import json
import os
import pickle
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from speed_model import TargetSpeedMLP, feature_sort_key


DATA_PATH = "dataset_throttle_brake"
DATASET_CONFIG_PATH = "dataset_throttle_brake/dataset_config.json"
MODEL_DIR = "model_throttle_brake"


def infer_feature_columns(df, dataset_config):
    if dataset_config and "stacked_feature_cols" in dataset_config:
        return dataset_config["stacked_feature_cols"]
    return sorted([col for col in df.columns if "_t-" in col], key=feature_sort_key)


def main():
    parser = argparse.ArgumentParser(description="Train target-speed sequence model")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--data", default=DATA_PATH)
    parser.add_argument("--config", default=DATASET_CONFIG_PATH)
    parser.add_argument("--output", default=MODEL_DIR, help="Directory to save model artifacts")
    args = parser.parse_args()

    model_dir = args.output
    os.makedirs(model_dir, exist_ok=True)

    print("=" * 64)
    print("TARGET-SPEED MLP TRAINER")
    print("=" * 64)

    dataset_config = None
    if args.config and os.path.exists(args.config):
        with open(args.config, "r", encoding="utf-8") as fh:
            dataset_config = json.load(fh)
        print(f"  Loaded dataset config: {args.config}")

    # --data may be a single CSV (back-compat) or a folder: load every *.csv in it
    # and concatenate, so base data + staged-scenario data train together.
    if os.path.isdir(args.data):
        csv_paths = sorted(glob.glob(os.path.join(args.data, "*.csv")))
        if not csv_paths:
            raise RuntimeError(f"No CSV files found in {args.data}")
        frames = []
        for path in csv_paths:
            part = pd.read_csv(path)
            frames.append(part)
            print(f"  Loaded {len(part):,} rows from {path}")
        df = pd.concat(frames, ignore_index=True)
        print(f"  Combined {len(df):,} rows from {len(csv_paths)} CSV file(s)")
    else:
        df = pd.read_csv(args.data)
        print(f"  Loaded {len(df):,} rows from {args.data}")

    feature_cols = infer_feature_columns(df, dataset_config)
    label_col = (
        dataset_config.get("label_col", "teacher_target_speed")
        if dataset_config
        else "teacher_target_speed"
    )

    if label_col not in df.columns:
        raise RuntimeError(f"Expected label column '{label_col}' in dataset")

    ego_speed_col = "ego_speed_t-0"
    distance_col = "distance_t-0"
    if ego_speed_col not in df.columns or distance_col not in df.columns:
        raise RuntimeError("Dataset does not contain expected current-frame columns")

    # Guard against column mismatches when concatenating multiple CSVs.
    df = df.dropna(subset=feature_cols + [label_col]).reset_index(drop=True)

    initial_len = len(df)
    df = df[~((df[ego_speed_col] < 0.1) & (df[distance_col] > 49.0))].reset_index(drop=True)
    dropped = initial_len - len(df)
    if dropped > 0:
        print(f"  Dropped {dropped:,} idle frames with no nearby obstacle")

    df[label_col] = df[label_col].clip(lower=0.0)

    stopped_mask = df[ego_speed_col] < (0.5 / 3.6)
    moving_df = df[~stopped_mask]
    stopped_df = df[stopped_mask]
    max_stopped = int(len(moving_df) * 0.15 / 0.85) if len(moving_df) > 0 else len(stopped_df)
    if len(stopped_df) > max_stopped > 0:
        stopped_df = stopped_df.sample(n=max_stopped, random_state=42)
        print(
            f"  Downsampled stopped frames: {stopped_mask.sum():,} -> {max_stopped:,}"
        )
    df = pd.concat([moving_df, stopped_df]).sort_index().reset_index(drop=True)
    print(f"  Final dataset: {len(df):,} rows")

    X = df[feature_cols].values.astype(np.float32)
    y = df[label_col].values.astype(np.float32)

    print(f"  Features: {len(feature_cols)} columns")
    print(f"  Label:    {label_col}")
    print(f"  Target speed range: [{y.min():.3f}, {y.max():.3f}] m/s")
    print(f"  Target speed mean:  {y.mean():.3f} m/s")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    scaler_path = os.path.join(model_dir, "scaler.pkl")
    with open(scaler_path, "wb") as fh:
        pickle.dump(scaler, fh)
    print(f"  Saved scaler to {scaler_path}")

    train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    val_ds = TensorDataset(torch.tensor(X_val), torch.tensor(y_val))

    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch * 2, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TargetSpeedMLP(input_dim=len(feature_cols)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=12
    )
    criterion = nn.MSELoss()

    total_params = sum(param.numel() for param in model.parameters())
    print(f"  Device: {device}")
    print(f"  Model params: {total_params:,}")
    print(f"  Epochs: {args.epochs}, LR: {args.lr}, Batch: {args.batch}")

    print("\n" + "=" * 64)
    print("TRAINING")
    print("=" * 64)

    best_val_loss = float("inf")
    best_epoch = 0
    model_path = os.path.join(model_dir, "target_speed_mlp.pt")
    train_start = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
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

        train_loss /= max(1, train_total)

        model.eval()
        val_loss = 0.0
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

        val_loss /= max(1, val_total)
        scheduler.step(val_loss)

        val_preds_np = np.concatenate(val_preds)
        val_labels_np = np.concatenate(val_labels)
        val_mae = np.mean(np.abs(val_preds_np - val_labels_np))
        val_rmse = np.sqrt(np.mean((val_preds_np - val_labels_np) ** 2))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save(model.state_dict(), model_path)

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"  Epoch {epoch:3d}/{args.epochs} "
                f"Loss: {train_loss:.5f}/{val_loss:.5f} "
                f"MAE: {val_mae:.4f} RMSE: {val_rmse:.4f}"
            )

    elapsed = time.time() - train_start
    print(f"\n  Training done in {elapsed:.1f}s")
    print(f"  Best val loss: {best_val_loss:.6f} at epoch {best_epoch}")

    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
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

    mae = float(np.mean(np.abs(all_preds - all_labels)))
    rmse = float(np.sqrt(np.mean((all_preds - all_labels) ** 2)))
    bias = float(np.mean(all_preds - all_labels))

    print("\n" + "=" * 64)
    print("FINAL EVALUATION")
    print("=" * 64)
    print(f"  MAE:              {mae:.4f} m/s")
    print(f"  RMSE:             {rmse:.4f} m/s")
    print(f"  Mean bias:        {bias:+.4f} m/s")
    print(f"  Prediction range: [{all_preds.min():.3f}, {all_preds.max():.3f}] m/s")
    print(f"  Model saved:      {model_path}")
    print(f"  Scaler saved:     {scaler_path}")

    model_config = {
        "feature_cols": feature_cols,
        "label_col": label_col,
        "history_frames": dataset_config.get("history_frames") if dataset_config else None,
        "base_feature_cols": dataset_config.get("base_feature_cols") if dataset_config else None,
        "fps": dataset_config.get("fps") if dataset_config else None,
        "radar_backend": (
            dataset_config.get("radar_backend", "native")
            if dataset_config
            else "native"
        ),
    }
    model_config_path = os.path.join(model_dir, "model_config.json")
    with open(model_config_path, "w", encoding="utf-8") as fh:
        json.dump(model_config, fh, indent=2)
    print(f"  Model config:     {model_config_path}")
    print("=" * 64)


if __name__ == "__main__":
    main()
