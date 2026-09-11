#!/usr/bin/env python3
"""Open-loop ghost counterfactual for the transformer controller.

The sharpest test of "does this controller react to ghosts", and it needs no
simulator: every logged window is scored three times.

  full      every point the sensor delivered
  no-ghost  the same window with the simulator-labelled ghost points removed
  dropout   the same window with an equal number of *direct* points removed
            at random, so generic sensitivity to fewer points is not mistaken
            for ghost sensitivity

A controller that is unaffected by ghosts shows full vs no-ghost differences
that are indistinguishable from full vs dropout. Reported: the distribution
of |delta target speed| in km/h, the fraction of windows whose brake decision
flips, both restricted to windows that contained ghosts, and broken down by
the closest ghost's range. Labels are read from the sidecar only; the model
never sees them.
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from cnn_controller import (  # noqa: E402
    MODEL_TYPE as CNN_MODEL_TYPE,
    SPEED_SCALE_MPS as CNN_SPEED_SCALE,
    load_model as load_cnn,
    outputs_to_target_speed as cnn_outputs_to_target_speed,
    rasterize,
    window_points,
)
from driving_contract import future_speed_label
from train_target_speed_transformer import (
    WindowedDetectionDataset,
    load_collection,
    select_rows,
)
from transformer_controller import (
    SOURCE_CODES,
    SPEED_SCALE_MPS,
    outputs_to_target_speed,
    load_model,
)
from driving_contract import MAX_TARGET_SPEED_KMH


BRAKE_DECISION_MARGIN_MPS = 0.5


def _summary(values):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {"count": 0}
    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p90": float(np.percentile(values, 90)),
        "p99": float(np.percentile(values, 99)),
        "max": float(values.max()),
    }


class GridVariantDataset(Dataset):
    """Three renderings of the same window: full, ghosts removed, control.

    The CNN has no token mask, so a variant means re-rasterising the same
    points with some of them left out. The control removes as many direct
    points as there were ghosts, which is what separates "reacts to ghosts"
    from "reacts to losing any points at all".
    """

    def __init__(self, rows, sidecars, window_frames, fps, label_col, seed=42):
        self.rows = rows.reset_index(drop=True)
        self.window_frames = int(window_frames)
        self.fps = float(fps)
        self.sidecars = sidecars
        self.label_col = label_col
        self.seed = int(seed)
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
        r, az, vr, age, rel, src = window_points(scans)
        ghost = src == SOURCE_CODES["ghost"]
        direct = src == SOURCE_CODES["direct"]
        ghost_count = int(ghost.sum())
        full = rasterize(r, az, vr, age, rel)
        if ghost_count:
            noghost = rasterize(r, az, vr, age, rel, keep=~ghost)
            rng = np.random.default_rng(self.seed + index)
            control = np.ones(src.shape, dtype=bool)
            direct_idx = np.flatnonzero(direct)
            take = min(ghost_count, len(direct_idx))
            if take:
                control[rng.choice(direct_idx, take, replace=False)] = False
            dropout = rasterize(r, az, vr, age, rel, keep=control)
            closest = float(r[ghost].min()) if ghost_count else 0.0
        else:
            noghost = full
            dropout = full
            closest = 0.0
        return {
            "full": torch.from_numpy(full),
            "noghost": torch.from_numpy(noghost),
            "dropout": torch.from_numpy(dropout),
            "ego": torch.tensor(ego_speed / CNN_SPEED_SCALE, dtype=torch.float32),
            "ego_speed": torch.tensor(ego_speed, dtype=torch.float32),
            "ghost_count": torch.tensor(ghost_count, dtype=torch.int64),
            "closest_ghost_m": torch.tensor(closest, dtype=torch.float32),
        }


def run_cnn(args, rows, sidecars, model_config, fps, label_col):
    """Counterfactual for the grid model; returns the same fields as the token path."""

    model, _ = load_cnn(args.model_dir, device=args.device)
    dataset = GridVariantDataset(
        rows, sidecars, int(model_config["window_frames"]), fps, label_col, args.seed
    )
    loader = DataLoader(dataset, batch_size=args.batch, shuffle=False, num_workers=0)
    deltas_ghost, deltas_dropout, flips_ghost, flips_dropout = [], [], [], []
    ghost_counts, closest_ghost_range, predictions_full = [], [], []
    windows_total = windows_with_ghosts = 0
    with torch.no_grad():
        for batch in loader:
            ego = batch["ego"].to(args.device)
            ego_speed = batch["ego_speed"].numpy()
            counts = batch["ghost_count"].numpy()
            full = cnn_outputs_to_target_speed(
                model(batch["full"].to(args.device), ego).cpu().numpy(), ego_speed)
            predictions_full.extend(full.tolist())
            windows_total += len(full)
            has = counts > 0
            if not has.any():
                continue
            noghost = cnn_outputs_to_target_speed(
                model(batch["noghost"].to(args.device), ego).cpu().numpy(), ego_speed)
            dropout = cnn_outputs_to_target_speed(
                model(batch["dropout"].to(args.device), ego).cpu().numpy(), ego_speed)
            for row in np.flatnonzero(has):
                windows_with_ghosts += 1
                deltas_ghost.append(abs(full[row] - noghost[row]) * 3.6)
                deltas_dropout.append(abs(full[row] - dropout[row]) * 3.6)
                margin = ego_speed[row] - BRAKE_DECISION_MARGIN_MPS
                brake_full = full[row] < margin
                flips_ghost.append(brake_full != (noghost[row] < margin))
                flips_dropout.append(brake_full != (dropout[row] < margin))
                ghost_counts.append(int(counts[row]))
                closest_ghost_range.append(float(batch["closest_ghost_m"][row]))
    return dict(
        deltas_ghost=np.asarray(deltas_ghost), deltas_dropout=np.asarray(deltas_dropout),
        flips_ghost=flips_ghost, flips_dropout=flips_dropout, ghost_counts=ghost_counts,
        closest_ghost_range=closest_ghost_range, windows_total=windows_total,
        windows_with_ghosts=windows_with_ghosts, predictions_full=predictions_full,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--data", required=True, help="collection directory with sidecars")
    parser.add_argument("--config", default=None)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--limit", type=int, default=None, help="cap on evaluated windows")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default=None, help="JSON report path")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    with open(os.path.join(args.model_dir, "model_config.json"), "r", encoding="utf-8") as fh:
        model_config = json.load(fh)
    model_type = model_config.get("model_type")
    model = None if model_type == CNN_MODEL_TYPE else load_model(args.model_dir, device=args.device)[0]
    config_path = args.config or os.path.join(args.data, "dataset_config.json")
    dataset_config = {}
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as fh:
            dataset_config = json.load(fh)
    label_col = dataset_config.get("label_col", "teacher_target_speed")
    fps = float(dataset_config.get("fps") or 20)
    max_speed_kmh = min(
        float(dataset_config.get("max_target_speed_kmh", MAX_TARGET_SPEED_KMH)),
        MAX_TARGET_SPEED_KMH,
    )
    rows, sidecars = load_collection(args.data, label_col)
    horizon = model_config.get("label_horizon_frames")
    if horizon:
        rows[label_col] = future_speed_label(rows, int(horizon))
    rows = select_rows(rows, label_col, max_speed_kmh, 3.0, 1.0 - 1e-9, args.seed)
    if args.limit and len(rows) > args.limit:
        rows = rows.sample(n=args.limit, random_state=args.seed).sort_index()
    if model_type == CNN_MODEL_TYPE:
        result = run_cnn(args, rows, sidecars, model_config, fps, label_col)
        deltas_ghost = result["deltas_ghost"]
        deltas_dropout = result["deltas_dropout"]
        flips_ghost = result["flips_ghost"]
        flips_dropout = result["flips_dropout"]
        ghost_counts = result["ghost_counts"]
        closest_ghost_range = result["closest_ghost_range"]
        windows_total = result["windows_total"]
        windows_with_ghosts = result["windows_with_ghosts"]
        predictions_full = result["predictions_full"]
    else:
        dataset = WindowedDetectionDataset(
            rows, sidecars, int(model_config["window_frames"]),
            int(model_config["max_points"]), fps, label_col,
        )
        loader = DataLoader(dataset, batch_size=args.batch, shuffle=False, num_workers=0)
        rng = np.random.default_rng(args.seed)

        deltas_ghost, deltas_dropout = [], []
        flips_ghost, flips_dropout = [], []
        ghost_counts, closest_ghost_range = [], []
        windows_total = 0
        windows_with_ghosts = 0
        predictions_full = []

        ghost_code = SOURCE_CODES["ghost"]
        direct_code = SOURCE_CODES["direct"]
        with torch.no_grad():
            for batch in loader:
                tokens = batch["tokens"].to(args.device)
                mask = batch["mask"].to(args.device)
                sources = batch["sources"].numpy()
                ego_speed = batch["ego_speed"].numpy()
                full = outputs_to_target_speed(model(tokens, mask).cpu().numpy(), ego_speed)
                predictions_full.extend(full.tolist())
                windows_total += len(full)

                is_ghost = sources == ghost_code
                has_ghost = is_ghost.any(axis=1)
                if not has_ghost.any():
                    continue
                # No-ghost variant: mask the ghost tokens out.
                noghost_mask = mask.clone()
                noghost_mask[torch.from_numpy(is_ghost).to(args.device)] = False
                noghost = outputs_to_target_speed(model(tokens, noghost_mask).cpu().numpy(), ego_speed)
                # Dropout variant: remove as many direct points as there were ghosts.
                dropout_mask = mask.clone().cpu().numpy()
                for row in np.flatnonzero(has_ghost):
                    direct_idx = np.flatnonzero(sources[row] == direct_code)
                    take = min(int(is_ghost[row].sum()), len(direct_idx))
                    if take > 0:
                        drop = rng.choice(direct_idx, take, replace=False)
                        dropout_mask[row, drop] = False
                dropout = outputs_to_target_speed(
                    model(tokens, torch.from_numpy(dropout_mask).to(args.device)).cpu().numpy(), ego_speed
                )

                # Closest ghost range from the range feature (range/100 in column 2).
                range_feature = tokens[..., 2].cpu().numpy() * 100.0
                for row in np.flatnonzero(has_ghost):
                    windows_with_ghosts += 1
                    deltas_ghost.append(abs(full[row] - noghost[row]) * 3.6)
                    deltas_dropout.append(abs(full[row] - dropout[row]) * 3.6)
                    brake_full = full[row] < ego_speed[row] - BRAKE_DECISION_MARGIN_MPS
                    brake_noghost = noghost[row] < ego_speed[row] - BRAKE_DECISION_MARGIN_MPS
                    brake_dropout = dropout[row] < ego_speed[row] - BRAKE_DECISION_MARGIN_MPS
                    flips_ghost.append(brake_full != brake_noghost)
                    flips_dropout.append(brake_full != brake_dropout)
                    ghost_counts.append(int(is_ghost[row].sum()))
                    closest_ghost_range.append(float(range_feature[row][is_ghost[row]].min()))

    deltas_ghost = np.asarray(deltas_ghost)
    deltas_dropout = np.asarray(deltas_dropout)
    closest = np.asarray(closest_ghost_range)
    by_range = {}
    for low, high in ((0, 15), (15, 30), (30, 60), (60, 200)):
        sel = (closest >= low) & (closest < high)
        by_range[f"{low}-{high}m"] = {
            "windows": int(sel.sum()),
            "delta_kmh_no_ghost": _summary(deltas_ghost[sel]),
            "delta_kmh_dropout": _summary(deltas_dropout[sel]),
            "brake_flip_rate_no_ghost": float(np.mean(np.asarray(flips_ghost)[sel])) if sel.any() else None,
        }
    report = {
        "model_dir": os.path.abspath(args.model_dir),
        "data": os.path.abspath(args.data),
        "windows_total": windows_total,
        "windows_with_ghosts": windows_with_ghosts,
        "mean_prediction_kmh": float(np.mean(predictions_full) * 3.6) if predictions_full else None,
        "delta_kmh_when_ghosts_removed": _summary(deltas_ghost),
        "delta_kmh_when_equal_direct_points_removed": _summary(deltas_dropout),
        "brake_decision_flip_rate_no_ghost": float(np.mean(flips_ghost)) if flips_ghost else None,
        "brake_decision_flip_rate_dropout": float(np.mean(flips_dropout)) if flips_dropout else None,
        "ghost_points_per_window": _summary(ghost_counts),
        "by_closest_ghost_range": by_range,
        "brake_decision_margin_mps": BRAKE_DECISION_MARGIN_MPS,
    }

    print("=" * 70)
    print("COUNTERFACTUAL GHOST TEST")
    print("=" * 70)
    print(f"  model               {args.model_dir}")
    print(f"  windows             {windows_total:,}  with ghosts: {windows_with_ghosts:,}")
    if windows_with_ghosts:
        g = report["delta_kmh_when_ghosts_removed"]
        d = report["delta_kmh_when_equal_direct_points_removed"]
        print(f"  |Δ target| km/h     remove ghosts: median {g['median']:.2f}  p90 {g['p90']:.2f}  max {g['max']:.2f}")
        print(f"                      remove direct: median {d['median']:.2f}  p90 {d['p90']:.2f}  max {d['max']:.2f}")
        print(f"  brake flips         remove ghosts: {report['brake_decision_flip_rate_no_ghost']:.3%}   "
              f"remove direct: {report['brake_decision_flip_rate_dropout']:.3%}")
        print("  by closest ghost range:")
        for name, block in by_range.items():
            if block["windows"]:
                print(f"    {name:>8}  n={block['windows']:5d}  "
                      f"median Δ {block['delta_kmh_no_ghost']['median']:.2f} km/h  "
                      f"flip {block['brake_flip_rate_no_ghost']:.3%}")
        verdict = (
            "ghost sensitivity is no larger than generic point-dropout sensitivity"
            if g["median"] <= 1.25 * max(d["median"], 0.1)
            else "the model reacts to ghosts specifically"
        )
        print(f"  verdict             {verdict}")
    else:
        print("  no windows contained ghosts; collect with multipath on to run this test")
    print("=" * 70)
    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)
        print(f"  report: {args.output}")


if __name__ == "__main__":
    main()
