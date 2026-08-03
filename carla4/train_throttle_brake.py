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
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from driving_contract import (
    MAX_STOPPED_FRACTION,
    MAX_TARGET_SPEED_KMH,
    NATIVE_RADAR_POINTS_PER_SECOND,
    RADAR_RANGE_M,
)
from speed_model import TargetSpeedMLP, feature_sort_key


DATA_PATH = "dataset_throttle_brake"
DATASET_CONFIG_PATH = "dataset_throttle_brake/dataset_config.json"
MODEL_DIR = "model_throttle_brake"


def infer_feature_columns(df, dataset_config):
    if dataset_config and "stacked_feature_cols" in dataset_config:
        return dataset_config["stacked_feature_cols"]
    return sorted([col for col in df.columns if "_t-" in col], key=feature_sort_key)


def ensure_episode_ids(part, source_name):
    """Backfill episode IDs for older CSVs without crossing discontinuities."""
    scenario = (
        part["scenario"].fillna("unknown").astype(str)
        if "scenario" in part.columns
        else pd.Series("unknown", index=part.index)
    )
    if "frame" in part.columns:
        frame = pd.to_numeric(part["frame"], errors="coerce")
        frame_diff = frame.diff()
    else:
        frame_diff = pd.Series(1.0, index=part.index)
    boundary = (
        scenario.ne(scenario.shift())
        | frame_diff.isna()
        | frame_diff.le(0)
        | frame_diff.gt(1)
    )
    inferred = (
        source_name
        + ":"
        + scenario
        + ":"
        + boundary.cumsum().astype(str)
    )
    if "episode_id" not in part.columns:
        part["episode_id"] = inferred
    else:
        supplied = part["episode_id"].astype("string")
        part["episode_id"] = supplied.fillna(inferred).astype(str)
        part["episode_id"] = source_name + ":" + part["episode_id"]
    return part


def episode_aware_split(df, validation_fraction, seed):
    """Split whole episodes, stratified by scenario where possible."""
    rng = np.random.default_rng(seed)
    validation_groups = set()
    for _, scenario_df in df.groupby("scenario", dropna=False, sort=False):
        groups = np.asarray(scenario_df["episode_id"].unique(), dtype=object)
        rng.shuffle(groups)
        if len(groups) < 2:
            continue
        count = max(1, int(round(len(groups) * validation_fraction)))
        count = min(count, len(groups) - 1)
        validation_groups.update(groups[:count].tolist())

    all_groups = np.asarray(df["episode_id"].unique(), dtype=object)
    if not validation_groups and len(all_groups) >= 2:
        rng.shuffle(all_groups)
        validation_groups.add(all_groups[0])

    validation_mask = df["episode_id"].isin(validation_groups)
    if not validation_mask.any() or validation_mask.all():
        raise RuntimeError(
            "Episode-aware split needs at least two usable episodes. "
            "Recollect with multiple episodes or weather segments."
        )
    return ~validation_mask, validation_mask


def main():
    parser = argparse.ArgumentParser(description="Train target-speed sequence model")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=30,
        help="Stop after this many epochs without validation improvement (0 disables)",
    )
    parser.add_argument("--data", default=DATA_PATH)
    parser.add_argument("--config", default=DATASET_CONFIG_PATH)
    parser.add_argument("--output", default=MODEL_DIR, help="Directory to save model artifacts")
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument(
        "--max-speed-kmh",
        type=float,
        default=None,
        help="Training speed ceiling (default: dataset config, otherwise 60)",
    )
    parser.add_argument(
        "--speed-tolerance-kmh",
        type=float,
        default=3.0,
        help="Drop rows whose observed ego speed exceeds the ceiling by this amount",
    )
    parser.add_argument(
        "--max-stopped-fraction",
        type=float,
        default=MAX_STOPPED_FRACTION,
    )
    args = parser.parse_args()
    if not 0.0 < args.validation_fraction < 1.0:
        parser.error("--validation-fraction must be between 0 and 1")
    if not 0.0 < args.max_stopped_fraction < 1.0:
        parser.error("--max-stopped-fraction must be between 0 and 1")
    if args.early_stop_patience < 0:
        parser.error("--early-stop-patience cannot be negative")
    np.random.seed(args.split_seed)
    torch.manual_seed(args.split_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.split_seed)

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
            part = ensure_episode_ids(part, os.path.basename(path))
            frames.append(part)
            print(f"  Loaded {len(part):,} rows from {path}")
        df = pd.concat(frames, ignore_index=True)
        print(f"  Combined {len(df):,} rows from {len(csv_paths)} CSV file(s)")
    else:
        df = pd.read_csv(args.data)
        df = ensure_episode_ids(df, os.path.basename(args.data))
        print(f"  Loaded {len(df):,} rows from {args.data}")

    feature_cols = infer_feature_columns(df, dataset_config)
    label_col = (
        dataset_config.get("label_col", "teacher_target_speed")
        if dataset_config
        else "teacher_target_speed"
    )

    if label_col not in df.columns:
        raise RuntimeError(f"Expected label column '{label_col}' in dataset")
    if "scenario" not in df.columns:
        df["scenario"] = "unknown"

    configured_max_speed = (
        float(dataset_config.get("max_target_speed_kmh", MAX_TARGET_SPEED_KMH))
        if dataset_config
        else MAX_TARGET_SPEED_KMH
    )
    max_speed_kmh = (
        float(args.max_speed_kmh)
        if args.max_speed_kmh is not None
        else configured_max_speed
    )
    if args.max_speed_kmh is not None and not (
        0.0 < max_speed_kmh <= MAX_TARGET_SPEED_KMH
    ):
        parser.error(
            f"--max-speed-kmh must be in (0, {MAX_TARGET_SPEED_KMH:g}]"
        )
    if max_speed_kmh <= 0.0:
        raise RuntimeError("Dataset max_target_speed_kmh must be positive")
    if configured_max_speed > MAX_TARGET_SPEED_KMH:
        print(
            f"  Dataset ceiling {configured_max_speed:.1f} km/h exceeds the "
            f"project limit; using {MAX_TARGET_SPEED_KMH:.1f} km/h"
        )
        max_speed_kmh = MAX_TARGET_SPEED_KMH

    ego_speed_col = "ego_speed_t-0"
    distance_col = "distance_t-0"
    if ego_speed_col not in df.columns or distance_col not in df.columns:
        raise RuntimeError("Dataset does not contain expected current-frame columns")

    # Guard against column mismatches when concatenating multiple CSVs.
    missing_features = [col for col in feature_cols if col not in df.columns]
    if missing_features:
        raise RuntimeError(
            "Dataset is missing configured feature columns; recollect all CSVs "
            f"with one schema. First missing columns: {missing_features[:5]}"
        )
    df = df.dropna(subset=feature_cols + [label_col]).reset_index(drop=True)

    ego_speed_cols = [col for col in feature_cols if col.startswith("ego_speed_t-")]
    state_speed_ceiling = (max_speed_kmh + args.speed_tolerance_kmh) / 3.6
    high_speed_mask = df[ego_speed_cols].max(axis=1) > state_speed_ceiling
    if high_speed_mask.any():
        print(
            f"  Dropped {int(high_speed_mask.sum()):,} rows above "
            f"{max_speed_kmh + args.speed_tolerance_kmh:.1f} km/h"
        )
        df = df[~high_speed_mask].reset_index(drop=True)

    initial_len = len(df)
    df = df[~((df[ego_speed_col] < 0.1) & (df[distance_col] > 49.0))].reset_index(drop=True)
    dropped = initial_len - len(df)
    if dropped > 0:
        print(f"  Dropped {dropped:,} idle frames with no nearby obstacle")

    df[label_col] = df[label_col].clip(
        lower=0.0,
        upper=max_speed_kmh / 3.6,
    )

    stopped_mask = df[ego_speed_col] < (0.5 / 3.6)
    moving_df = df[~stopped_mask]
    stopped_df = df[stopped_mask]
    max_stopped = (
        int(
            len(moving_df)
            * args.max_stopped_fraction
            / (1.0 - args.max_stopped_fraction)
        )
        if len(moving_df) > 0
        else len(stopped_df)
    )
    if len(stopped_df) > max_stopped > 0:
        stopped_df = stopped_df.sample(
            n=max_stopped,
            random_state=args.split_seed,
        )
        print(
            f"  Downsampled stopped frames: {stopped_mask.sum():,} -> {max_stopped:,}"
        )
    df = pd.concat([moving_df, stopped_df]).sort_index().reset_index(drop=True)
    if len(df) < 100:
        raise RuntimeError(
            f"Only {len(df)} usable rows remain; collect more data before training."
        )
    print(f"  Final dataset: {len(df):,} rows")
    print(f"  Episodes:      {df['episode_id'].nunique():,}")
    scenario_counts = df["scenario"].value_counts().sort_index()
    print(
        "  Scenario rows: "
        + ", ".join(
            f"{name}={count:,}"
            for name, count in scenario_counts.items()
        )
    )

    X = df[feature_cols].values.astype(np.float32)
    y = df[label_col].values.astype(np.float32)

    print(f"  Features: {len(feature_cols)} columns")
    print(f"  Label:    {label_col}")
    print(f"  Target speed range: [{y.min():.3f}, {y.max():.3f}] m/s")
    print(f"  Target speed mean:  {y.mean():.3f} m/s")

    train_mask, validation_mask = episode_aware_split(
        df,
        args.validation_fraction,
        args.split_seed,
    )
    X_train = X[train_mask.to_numpy()]
    X_val = X[validation_mask.to_numpy()]
    y_train = y[train_mask.to_numpy()]
    y_val = y[validation_mask.to_numpy()]
    train_groups = set(df.loc[train_mask, "episode_id"])
    validation_groups = set(df.loc[validation_mask, "episode_id"])
    overlap = train_groups & validation_groups
    if overlap:
        raise RuntimeError("Internal error: train/validation episode leakage")
    print(
        f"  Episode split: {len(train_groups)} train / "
        f"{len(validation_groups)} validation episodes"
    )
    print(
        f"  Rows:          {len(X_train):,} train / "
        f"{len(X_val):,} validation"
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
        if (
            args.early_stop_patience
            and epoch - best_epoch >= args.early_stop_patience
        ):
            print(
                f"  Early stopping at epoch {epoch}: no validation "
                f"improvement for {args.early_stop_patience} epochs"
            )
            break

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
        "town": dataset_config.get("town") if dataset_config else None,
        "history_frames": dataset_config.get("history_frames") if dataset_config else None,
        "base_feature_cols": dataset_config.get("base_feature_cols") if dataset_config else None,
        "fps": dataset_config.get("fps") if dataset_config else None,
        "radar_backend": (
            dataset_config.get("radar_backend", "native")
            if dataset_config
            else "native"
        ),
        "radar_range_m": (
            float(dataset_config.get("radar_range_m", RADAR_RANGE_M))
            if dataset_config
            else RADAR_RANGE_M
        ),
        "radar_points_per_second": (
            int(
                dataset_config.get(
                    "radar_points_per_second",
                    (
                        NATIVE_RADAR_POINTS_PER_SECOND
                        if dataset_config.get("radar_backend", "native")
                        == "native"
                        else 240000
                    ),
                )
            )
            if dataset_config
            else NATIVE_RADAR_POINTS_PER_SECOND
        ),
        "radar_profile": (
            dataset_config.get("radar_profile")
            if dataset_config
            else None
        ),
        "radar_config_signature": (
            dataset_config.get("radar_config_signature")
            if dataset_config
            else None
        ),
        "radar_config": (
            dataset_config.get("radar_config")
            if dataset_config
            else None
        ),
        "radar_ghost_detector": (
            dataset_config.get("radar_ghost_detector")
            if dataset_config
            else None
        ),
        "radar_ghost_detector_signature": (
            dataset_config.get("radar_ghost_detector_signature")
            if dataset_config
            else None
        ),
        "radar_ghost_threshold": (
            dataset_config.get("radar_ghost_threshold")
            if dataset_config
            else None
        ),
        "radar_ghost_model": (
            dataset_config.get("radar_ghost_model")
            if dataset_config
            else None
        ),
        "radar_ghost_feature_schema": (
            dataset_config.get("radar_ghost_feature_schema")
            if dataset_config
            else None
        ),
        "max_target_speed_kmh": max_speed_kmh,
        "validation": {
            "method": "episode_aware_stratified",
            "fraction": args.validation_fraction,
            "seed": args.split_seed,
            "early_stop_patience": args.early_stop_patience,
            "train_episodes": len(train_groups),
            "validation_episodes": len(validation_groups),
            "mae_mps": mae,
            "rmse_mps": rmse,
            "bias_mps": bias,
        },
    }
    model_config_path = os.path.join(model_dir, "model_config.json")
    with open(model_config_path, "w", encoding="utf-8") as fh:
        json.dump(model_config, fh, indent=2)
    print(f"  Model config:     {model_config_path}")
    print("=" * 64)


if __name__ == "__main__":
    main()
