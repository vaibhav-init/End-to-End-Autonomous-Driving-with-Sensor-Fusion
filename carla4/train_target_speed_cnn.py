#!/usr/bin/env python3
"""Train the target-speed CNN on the range-azimuth grid.

Same data, same labels, same episode-aware split, same brake-weighted loss and
same delta-speed output as the transformer trainer, so the only difference
between the two arms is how the detection list is presented to the network:
an unordered bag of 256 points against a grid that holds every point.

Needs the detection sidecars, so realistic-backend collections only.
"""

import argparse
import json
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from driving_contract import (
    MAX_STOPPED_FRACTION,
    MAX_TARGET_SPEED_KMH,
    NATIVE_RADAR_POINTS_PER_SECOND,
    RADAR_RANGE_M,
    future_speed_label,
)
from cnn_controller import (
    CHECKPOINT_NAME,
    GRID_CHANNELS,
    MODEL_TYPE,
    SPEED_SCALE_MPS,
    build_window_grid,
    create_model,
    default_model_kwargs,
    save_checkpoint,
)
from train_target_speed_transformer import (
    BRAKE_WEIGHT_SCALE_MPS,
    load_collection,
    select_rows,
)
from train_throttle_brake import episode_aware_split
from transformer_controller import dataset_frames_to_scans


class GriddedDetectionDataset(Dataset):
    """One sample per selected CSV row: the window rendered as a grid."""

    def __init__(self, rows, sidecars, window_frames, fps, label_col):
        self.rows = rows.reset_index(drop=True)
        self.window_frames = int(window_frames)
        self.fps = float(fps)
        self.label_col = label_col
        self.sidecars = sidecars
        episode_start = self.rows.groupby("episode_id")["frame"].transform("min")
        self.min_frames = (episode_start - self.window_frames).to_numpy()

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows.iloc[index]
        sidecar = self.sidecars[row["source_csv"]]
        scans = dataset_frames_to_scans(
            sidecar["frames"], sidecar["by_frame"], sidecar["frame_to_scan"],
            int(row["frame"]), self.window_frames, self.fps,
            min_frame=int(self.min_frames[index]),
        )
        ego_speed = float(row["ego_speed_now"])
        built = build_window_grid(scans, ego_speed)
        return {
            "grid": torch.from_numpy(built["grid"]),
            "ego": torch.tensor(float(built["ego"]), dtype=torch.float32),
            "target": torch.tensor(
                (float(row[self.label_col]) - ego_speed) / SPEED_SCALE_MPS,
                dtype=torch.float32,
            ),
            "ego_speed": torch.tensor(ego_speed, dtype=torch.float32),
            "point_count": torch.tensor(int(built["point_count"]), dtype=torch.int64),
        }


def run_epoch(model, loader, device, optimizer=None, brake_weight=4.0):
    training = optimizer is not None
    model.train(training)
    loss_fn = torch.nn.SmoothL1Loss(beta=0.1, reduction="none")
    total = count = 0.0
    abs_error = 0.0
    for batch in loader:
        grid = batch["grid"].to(device, non_blocking=True)
        ego = batch["ego"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        braking = (-target * SPEED_SCALE_MPS / BRAKE_WEIGHT_SCALE_MPS).clamp(0.0, 1.0)
        weight = 1.0 + float(brake_weight) * braking
        with torch.set_grad_enabled(training):
            prediction = model(grid, ego)
            loss = (loss_fn(prediction, target) * weight).sum() / weight.sum()
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
        total += float(loss.item()) * len(target)
        count += len(target)
        abs_error += float((prediction.detach() - target).abs().sum().item()) * SPEED_SCALE_MPS
    return total / max(count, 1), abs_error / max(count, 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="dataset_throttle_brake")
    parser.add_argument("--config", default=None)
    parser.add_argument("--output", default="model_cnn")
    parser.add_argument("--window-frames", type=int, default=10)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--label-horizon", type=int, default=None)
    parser.add_argument("--brake-weight", type=float, default=4.0)
    parser.add_argument("--early-stop-patience", type=int, default=15)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--max-speed-kmh", type=float, default=None)
    parser.add_argument("--speed-tolerance-kmh", type=float, default=3.0)
    parser.add_argument("--max-stopped-fraction", type=float, default=MAX_STOPPED_FRACTION)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    torch.manual_seed(args.split_seed)
    np.random.seed(args.split_seed)

    config_path = args.config or os.path.join(args.data, "dataset_config.json")
    dataset_config = {}
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as fh:
            dataset_config = json.load(fh)
    label_col = dataset_config.get("label_col", "teacher_target_speed")
    fps = float(dataset_config.get("fps") or 20)
    max_speed_kmh = min(
        float(args.max_speed_kmh) if args.max_speed_kmh is not None
        else float(dataset_config.get("max_target_speed_kmh", MAX_TARGET_SPEED_KMH)),
        MAX_TARGET_SPEED_KMH,
    )
    if dataset_config.get("radar_backend", "native") != "realistic":
        raise RuntimeError(
            "The CNN needs the point-level detection list, which only the "
            "realistic backend produces. Collect with --radar-backend realistic."
        )

    print("=" * 64)
    print("TARGET-SPEED CNN TRAINER")
    print("=" * 64)
    rows, sidecars = load_collection(args.data, label_col)
    if args.label_horizon:
        rows[label_col] = future_speed_label(rows, args.label_horizon)
        print(f"  Relabelled with a {args.label_horizon}-frame future-speed horizon")
    rows = select_rows(rows, label_col, max_speed_kmh, args.speed_tolerance_kmh,
                       args.max_stopped_fraction, args.split_seed)
    train_mask, val_mask = episode_aware_split(rows, args.validation_fraction, args.split_seed)
    train_rows, val_rows = rows[train_mask], rows[val_mask]
    print(f"  Episode split: {train_rows['episode_id'].nunique()} train / "
          f"{val_rows['episode_id'].nunique()} validation")
    print(f"  Rows:          {len(train_rows):,} train / {len(val_rows):,} validation")

    make = lambda part: GriddedDetectionDataset(part, sidecars, args.window_frames, fps, label_col)
    loader_kwargs = dict(num_workers=args.num_workers,
                         pin_memory=args.device.startswith("cuda"),
                         persistent_workers=args.num_workers > 0)
    train_dl = DataLoader(make(train_rows), batch_size=args.batch, shuffle=True, **loader_kwargs)
    val_dl = DataLoader(make(val_rows), batch_size=args.batch * 2, shuffle=False, **loader_kwargs)

    model_kwargs = default_model_kwargs()
    model_kwargs.update(width=args.width, dropout=args.dropout)
    model = create_model(**model_kwargs).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=6)
    print(f"  Device: {args.device}")
    print(f"  Model params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Grid: {GRID_CHANNELS} channels over the last {args.window_frames} frames")

    os.makedirs(args.output, exist_ok=True)
    checkpoint_path = os.path.join(args.output, CHECKPOINT_NAME)
    best_val, best_epoch, history = float("inf"), 0, []
    started = time.time()
    print("\n" + "=" * 64)
    print("TRAINING")
    for epoch in range(1, args.epochs + 1):
        train_loss, train_mae = run_epoch(model, train_dl, args.device, optimizer, args.brake_weight)
        val_loss, val_mae = run_epoch(model, val_dl, args.device, brake_weight=args.brake_weight)
        scheduler.step(val_loss)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                        "train_mae_mps": train_mae, "val_mae_mps": val_mae,
                        "lr": optimizer.param_groups[0]["lr"]})
        if val_loss < best_val:
            best_val, best_epoch = val_loss, epoch
            save_checkpoint(checkpoint_path, model, model_kwargs)
        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{args.epochs} loss {train_loss:.5f}/{val_loss:.5f} "
                  f"MAE {train_mae:.3f}/{val_mae:.3f} m/s")
        if args.early_stop_patience and epoch - best_epoch >= args.early_stop_patience:
            print(f"  Early stopping at epoch {epoch}")
            break
    with open(os.path.join(args.output, "history.json"), "w", encoding="utf-8") as fh:
        json.dump(history, fh, indent=2)
    print(f"\n  Training done in {time.time() - started:.1f}s; best val loss {best_val:.5f} at epoch {best_epoch}")

    checkpoint = torch.load(checkpoint_path, map_location=args.device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    _loss, val_mae = run_epoch(model, val_dl, args.device, brake_weight=args.brake_weight)
    print(f"  Validation MAE (best checkpoint): {val_mae:.4f} m/s")

    provenance_keys = (
        "town", "teacher", "fps", "radar_backend", "radar_range_m",
        "radar_points_per_second", "radar_profile", "radar_config_signature",
        "radar_config", "radar_ghost_injection", "radar_ghost_oracle",
        "radar_ghost_detector", "radar_ghost_detector_signature",
        "radar_ghost_threshold", "radar_ghost_model", "radar_ghost_feature_schema",
    )
    model_config = {key: dataset_config.get(key) for key in provenance_keys}
    model_config.update({
        "model_type": MODEL_TYPE,
        "checkpoint": CHECKPOINT_NAME,
        "label_col": label_col,
        "label_horizon_frames": args.label_horizon,
        "feature_cols": [],
        "base_feature_cols": dataset_config.get("base_feature_cols"),
        "history_frames": args.window_frames,
        "window_frames": args.window_frames,
        "model_kwargs": model_kwargs,
        "speed_scale_mps": SPEED_SCALE_MPS,
        "max_target_speed_kmh": max_speed_kmh,
        "radar_range_m": float(dataset_config.get("radar_range_m", RADAR_RANGE_M)),
        "radar_points_per_second": int(dataset_config.get("radar_points_per_second", NATIVE_RADAR_POINTS_PER_SECOND)),
        "validation": {
            "method": "episode_aware_stratified",
            "fraction": args.validation_fraction,
            "seed": args.split_seed,
            "train_episodes": int(train_rows["episode_id"].nunique()),
            "validation_episodes": int(val_rows["episode_id"].nunique()),
            "mae_mps": val_mae,
            "best_epoch": best_epoch,
        },
    })
    with open(os.path.join(args.output, "model_config.json"), "w", encoding="utf-8") as fh:
        json.dump(model_config, fh, indent=2)
    print(f"  Model saved:  {checkpoint_path}")
    print("  Next: python3 acceptance_test.py --model-dir", args.output)


if __name__ == "__main__":
    main()
