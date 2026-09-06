#!/usr/bin/env python3
"""Train the target-speed transformer on logged detection lists.

Reads a collection directory written by `collect_throttle_brake_data.py` (and
optionally `collect_scenario_data.py`): every `*.csv` supplies the per-frame
ego state, teacher label and episode ids; its `*.detections.npz` sidecar
supplies the point-level radar scans. Rows whose CSV has no sidecar are
skipped with a warning, because the transformer cannot be trained on the
scalar columns alone.

Selection and splitting mirror `train_throttle_brake.py` exactly (idle-frame
drop, stopped-frame downsampling, episode-aware validation split, label
clipping) so a transformer and an MLP trained on the same collection see the
same rows. Provenance is copied from `dataset_config.json` into
`model_config.json` with `model_type: transformer`, so the scenario driver's
sensor gate and the acceptance test work unchanged.
"""

import argparse
import glob
import json
import os
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from driving_contract import (
    MAX_STOPPED_FRACTION,
    MAX_TARGET_SPEED_KMH,
    NATIVE_RADAR_POINTS_PER_SECOND,
    RADAR_RANGE_M,
)
from radar.detection_log import detections_by_frame, load_detection_log, sidecar_path
from train_throttle_brake import ensure_episode_ids, episode_aware_split
from driving_contract import future_speed_label
from transformer_controller import (
    CHECKPOINT_NAME,
    MODEL_TYPE,
    SPEED_SCALE_MPS,
    build_window_tokens,
    create_model,
    dataset_frames_to_scans,
    default_model_kwargs,
    save_checkpoint,
)


class WindowedDetectionDataset(Dataset):
    """One sample per selected CSV row: the detection window ending at its frame."""

    def __init__(self, rows, sidecars, window_frames, max_points, fps, label_col):
        self.rows = rows.reset_index(drop=True)
        self.window_frames = int(window_frames)
        self.max_points = int(max_points)
        self.fps = float(fps)
        self.label_col = label_col
        # Per source CSV: frame -> records, frame -> scan index, frames set.
        self.sidecars = sidecars
        episode_start = self.rows.groupby("episode_id")["frame"].transform("min")
        self.min_frames = (episode_start - self.window_frames).to_numpy()

    def __len__(self):
        return len(self.rows)

    def window(self, index):
        row = self.rows.iloc[index]
        sidecar = self.sidecars[row["source_csv"]]
        return dataset_frames_to_scans(
            sidecar["frames"],
            sidecar["by_frame"],
            sidecar["frame_to_scan"],
            int(row["frame"]),
            self.window_frames,
            self.fps,
            min_frame=int(self.min_frames[index]),
        )

    def __getitem__(self, index):
        row = self.rows.iloc[index]
        scans = self.window(index)
        built = build_window_tokens(
            scans,
            float(row["ego_speed_now"]),
            float(row.get("ego_acceleration_t-0", 0.0)),
            self.max_points,
        )
        return {
            "tokens": torch.from_numpy(built["tokens"]),
            "mask": torch.from_numpy(built["mask"]),
            "sources": torch.from_numpy(built["sources"]),
            # Speed change from now to the label horizon, normalised.
            "target": torch.tensor(
                (float(row[self.label_col]) - float(row["ego_speed_now"])) / SPEED_SCALE_MPS,
                dtype=torch.float32,
            ),
            "ego_speed": torch.tensor(float(row["ego_speed_now"]), dtype=torch.float32),
            "point_count": torch.tensor(int(built["point_count"]), dtype=torch.int64),
        }


def load_collection(data_dir, label_col):
    """Return (rows DataFrame, sidecars dict) for every CSV with a sidecar."""

    csv_paths = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    if not csv_paths:
        raise RuntimeError(f"No CSV files found in {data_dir}")
    frames = []
    sidecars = {}
    for path in csv_paths:
        sidecar = sidecar_path(path)
        if not os.path.exists(sidecar):
            print(f"  skipping {os.path.basename(path)}: no detection sidecar {os.path.basename(sidecar)}")
            continue
        part = pd.read_csv(path)
        part = ensure_episode_ids(part, os.path.basename(path))
        part["source_csv"] = path
        frames.append(part)
        log = load_detection_log(sidecar)
        by_frame = detections_by_frame(log["detections"])
        frames_table = log["frames"]
        frame_to_scan = {
            int(frame): int(scan)
            for frame, scan in zip(frames_table["frame"], frames_table["scan_index"])
        }
        available = set(frame_to_scan) | set(by_frame)
        sidecars[path] = {
            "by_frame": by_frame,
            "frame_to_scan": frame_to_scan,
            "frames": available,
        }
        print(
            f"  Loaded {len(part):,} rows and {len(log['detections']):,} points "
            f"over {len(available):,} frames from {os.path.basename(path)}"
        )
    if not frames:
        raise RuntimeError(
            "No CSV in the collection has a detection sidecar. Collect with the "
            "realistic radar backend; the native backend has no detection list."
        )
    rows = pd.concat(frames, ignore_index=True)
    if label_col not in rows.columns:
        raise RuntimeError(f"Expected label column {label_col!r} in the collection")
    return rows, sidecars


def select_rows(rows, label_col, max_speed_kmh, speed_tolerance_kmh,
                max_stopped_fraction, split_seed):
    """The same row selection as train_throttle_brake.py."""

    ego_speed_col = "ego_speed_t-0"
    distance_col = "distance_t-0"
    rows = rows.dropna(subset=[label_col, ego_speed_col, distance_col, "frame"]).reset_index(drop=True)
    speed_cols = [col for col in rows.columns if col.startswith("ego_speed_t-")]
    ceiling = (max_speed_kmh + speed_tolerance_kmh) / 3.6
    high = rows[speed_cols].max(axis=1) > ceiling
    if high.any():
        print(f"  Dropped {int(high.sum()):,} rows above {max_speed_kmh + speed_tolerance_kmh:.1f} km/h")
        rows = rows[~high].reset_index(drop=True)
    idle = (rows[ego_speed_col] < 0.1) & (rows[distance_col] > 49.0)
    if idle.any():
        print(f"  Dropped {int(idle.sum()):,} idle frames with no nearby obstacle")
        rows = rows[~idle].reset_index(drop=True)
    rows[label_col] = rows[label_col].clip(lower=0.0, upper=max_speed_kmh / 3.6)
    stopped = rows[ego_speed_col] < (0.5 / 3.6)
    moving = rows[~stopped]
    stopped_rows = rows[stopped]
    max_stopped = (
        int(len(moving) * max_stopped_fraction / (1.0 - max_stopped_fraction))
        if len(moving) else len(stopped_rows)
    )
    if len(stopped_rows) > max_stopped > 0:
        stopped_rows = stopped_rows.sample(n=max_stopped, random_state=split_seed)
        print(f"  Downsampled stopped frames: {int(stopped.sum()):,} -> {max_stopped:,}")
    rows = pd.concat([moving, stopped_rows]).sort_index().reset_index(drop=True)
    if "scenario" not in rows.columns:
        rows["scenario"] = "unknown"
    return rows


# Braking windows are rare (a few percent of frames) and are the ones that
# matter; a window whose label sits BRAKE_WEIGHT_SCALE_MPS or more below the
# ego speed carries 1 + brake_weight times the loss of a cruise window.
BRAKE_WEIGHT_SCALE_MPS = 3.0


def run_epoch(model, loader, device, optimizer=None, brake_weight=4.0):
    training = optimizer is not None
    model.train(training)
    loss_fn = torch.nn.SmoothL1Loss(beta=0.1, reduction="none")
    total = 0.0
    count = 0
    abs_error = 0.0
    for batch in loader:
        tokens = batch["tokens"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        braking = (-target * SPEED_SCALE_MPS / BRAKE_WEIGHT_SCALE_MPS).clamp(0.0, 1.0)
        weight = 1.0 + float(brake_weight) * braking
        with torch.set_grad_enabled(training):
            prediction = model(tokens, mask)
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
    parser.add_argument("--config", default=None,
                        help="dataset_config.json (default: <data>/dataset_config.json)")
    parser.add_argument("--output", default="model_transformer")
    parser.add_argument("--window-frames", type=int, default=10)
    parser.add_argument("--max-points", type=int, default=256)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--ff-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--label-horizon", type=int, default=None,
                        help="recompute the label as the mean ego speed over this many future frames")
    parser.add_argument("--brake-weight", type=float, default=4.0,
                        help="extra loss weight for windows whose label is well below the ego speed")
    parser.add_argument("--early-stop-patience", type=int, default=15)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--max-speed-kmh", type=float, default=None)
    parser.add_argument("--speed-tolerance-kmh", type=float, default=3.0)
    parser.add_argument("--max-stopped-fraction", type=float, default=MAX_STOPPED_FRACTION)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if not 0.0 < args.validation_fraction < 1.0:
        parser.error("--validation-fraction must be between 0 and 1")
    if args.window_frames < 1 or args.max_points < 1:
        parser.error("--window-frames and --max-points must be positive")
    torch.manual_seed(args.split_seed)
    np.random.seed(args.split_seed)

    config_path = args.config or os.path.join(args.data, "dataset_config.json")
    dataset_config = {}
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as fh:
            dataset_config = json.load(fh)
        print(f"  Loaded dataset config: {config_path}")
    label_col = dataset_config.get("label_col", "teacher_target_speed")
    fps = float(dataset_config.get("fps") or 20)
    configured_max = float(dataset_config.get("max_target_speed_kmh", MAX_TARGET_SPEED_KMH))
    max_speed_kmh = min(
        float(args.max_speed_kmh) if args.max_speed_kmh is not None else configured_max,
        MAX_TARGET_SPEED_KMH,
    )
    if dataset_config.get("radar_backend", "native") != "realistic":
        raise RuntimeError(
            "The transformer needs the point-level detection list, which only the "
            "realistic backend produces. Collect with --radar-backend realistic."
        )

    print("=" * 64)
    print("TARGET-SPEED TRANSFORMER TRAINER")
    print("=" * 64)
    rows, sidecars = load_collection(args.data, label_col)
    if args.label_horizon:
        rows[label_col] = future_speed_label(rows, args.label_horizon)
        print(f"  Relabelled with a {args.label_horizon}-frame future-speed horizon")
    rows = select_rows(
        rows, label_col, max_speed_kmh, args.speed_tolerance_kmh,
        args.max_stopped_fraction, args.split_seed,
    )
    if len(rows) < 100:
        raise RuntimeError(f"Only {len(rows)} usable rows remain; collect more data before training.")
    print(f"  Final dataset: {len(rows):,} rows over {rows['episode_id'].nunique():,} episodes")
    ghost_col = "radar_selected_source"
    if ghost_col in rows.columns:
        print(f"  Ghost-selected share in rows: {(rows[ghost_col] == 'ghost').mean():.3f}")

    train_mask, val_mask = episode_aware_split(rows, args.validation_fraction, args.split_seed)
    train_rows = rows[train_mask]
    val_rows = rows[val_mask]
    print(f"  Episode split: {train_rows['episode_id'].nunique()} train / {val_rows['episode_id'].nunique()} validation")
    print(f"  Rows:          {len(train_rows):,} train / {len(val_rows):,} validation")

    train_ds = WindowedDetectionDataset(train_rows, sidecars, args.window_frames, args.max_points, fps, label_col)
    val_ds = WindowedDetectionDataset(val_rows, sidecars, args.window_frames, args.max_points, fps, label_col)
    loader_kwargs = dict(num_workers=args.num_workers, pin_memory=args.device.startswith("cuda"),
                         persistent_workers=args.num_workers > 0)
    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True, **loader_kwargs)
    val_dl = DataLoader(val_ds, batch_size=args.batch * 2, shuffle=False, **loader_kwargs)

    model_kwargs = default_model_kwargs()
    model_kwargs.update(d_model=args.d_model, heads=args.heads, layers=args.layers,
                        ff_dim=args.ff_dim, dropout=args.dropout)
    model = create_model(**model_kwargs).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=6)
    print(f"  Device: {args.device}")
    print(f"  Model params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Window: {args.window_frames} frames x {args.max_points} points, token dim {model_kwargs['token_dim']}")

    os.makedirs(args.output, exist_ok=True)
    checkpoint_path = os.path.join(args.output, CHECKPOINT_NAME)
    best_val = float("inf")
    best_epoch = 0
    history = []
    started = time.time()
    print("\n" + "=" * 64)
    print("TRAINING")
    print("=" * 64)
    for epoch in range(1, args.epochs + 1):
        train_loss, train_mae = run_epoch(model, train_dl, args.device, optimizer, args.brake_weight)
        val_loss, val_mae = run_epoch(model, val_dl, args.device, brake_weight=args.brake_weight)
        scheduler.step(val_loss)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                        "train_mae_mps": train_mae, "val_mae_mps": val_mae,
                        "lr": optimizer.param_groups[0]["lr"]})
        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            save_checkpoint(checkpoint_path, model, model_kwargs)
        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{args.epochs} loss {train_loss:.5f}/{val_loss:.5f} "
                  f"MAE {train_mae:.3f}/{val_mae:.3f} m/s")
        if args.early_stop_patience and epoch - best_epoch >= args.early_stop_patience:
            print(f"  Early stopping at epoch {epoch}: no validation improvement for {args.early_stop_patience} epochs")
            break
    with open(os.path.join(args.output, "history.json"), "w", encoding="utf-8") as fh:
        json.dump(history, fh, indent=2)
    print(f"\n  Training done in {time.time() - started:.1f}s; best val loss {best_val:.5f} at epoch {best_epoch}")

    checkpoint = torch.load(checkpoint_path, map_location=args.device, weights_only=True)
    model.load_state_dict(checkpoint["model_state"])
    _val_loss, val_mae = run_epoch(model, val_dl, args.device, brake_weight=args.brake_weight)
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
        "max_points": args.max_points,
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
    print(f"  Model config: {os.path.join(args.output, 'model_config.json')}")
    print("  Next: python3 acceptance_test.py --model-dir", args.output)
    print("        python3 counterfactual_ghost_test.py --model-dir", args.output, "--data", args.data)


if __name__ == "__main__":
    main()
