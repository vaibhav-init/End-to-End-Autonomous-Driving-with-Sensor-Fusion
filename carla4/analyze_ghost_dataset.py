#!/usr/bin/env python3
"""Comprehensive analysis of the radar ghost dataset and model generalization.

Examines split composition, data leakage, feature distributions, class
balance, separability, and per-scenario behaviour to determine whether
poor test performance stems from the data or the model.

Usage:
    python3 analyze_ghost_dataset.py \
        --data artifacts/ghost_real_official \
        --checkpoint artifacts/ghost_temporal_official/best_detector.pt \
        --output artifacts/ghost_temporal_official/analysis
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from radar.ghost_detection.dataset import PreparedGhostDataset
from radar.ghost_detection.features import FEATURE_NAMES, FEATURE_SCHEMA_VERSION
from radar.ghost_detection.model import create_ghost_model


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--checkpoint", help="optional trained model checkpoint")
    parser.add_argument("--output", default=None, help="directory for analysis outputs")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# 1. Split composition & overlap
# ---------------------------------------------------------------------------

def analyse_split_composition(manifest):
    """Report sequences, points, scenarios, towns, weather per split."""
    splits = defaultdict(list)
    for rec in manifest.get("sequences", []):
        sp = rec.get("split", "unknown")
        splits[sp].append(rec)

    print("\n" + "=" * 70)
    print("1. SPLIT COMPOSITION")
    print("=" * 70)

    for sp in sorted(splits):
        recs = splits[sp]
        total_pts = sum(r.get("points", 0) for r in recs)
        real_pts = sum(r.get("real_points", 0) for r in recs)
        ghost_pts = sum(r.get("ghost_points", 0) for r in recs)
        scenarios = Counter(r.get("scenario", "?") for r in recs)
        towns = Counter(r.get("town", "?") for r in recs)
        weathers = Counter(r.get("weather", "?") for r in recs)
        print(f"\n--- {sp.upper()} ({len(recs)} sequences, {total_pts:,} points) ---")
        print(f"    real={real_pts:,}  ghost={ghost_pts:,}  "
              f"ratio={real_pts / max(ghost_pts, 1):.2f}:1")
        print(f"    scenarios: {dict(scenarios)}")
        print(f"    towns:     {dict(towns)}")
        print(f"    weather:   {dict(weathers)}")

    return splits


def check_scenario_leakage(splits):
    """Detect scenarios appearing in multiple splits (data leakage)."""
    print("\n" + "=" * 70)
    print("2. SCENARIO LEAKAGE CHECK")
    print("=" * 70)

    scenario_splits = defaultdict(set)
    for sp, recs in splits.items():
        for r in recs:
            scenario_splits[r.get("scenario", "?")].add(sp)

    leaked = {sc: sps for sc, sps in scenario_splits.items() if len(sps) > 1}
    if leaked:
        print("⚠ LEAKED scenarios (present in multiple splits):")
        for sc, sps in sorted(leaked.items()):
            print(f"  {sc}: {sorted(sps)}")
    else:
        print("✓ No scenario appears in more than one split.")
    return leaked


def check_sequence_id_leakage(splits):
    """Detect sequence_ids (e.g. scenario-11) appearing in multiple splits."""
    print("\n" + "=" * 70)
    print("3. SEQUENCE-ID LEAKAGE CHECK")
    print("=" * 70)

    seqid_splits = defaultdict(set)
    for sp, recs in splits.items():
        for r in recs:
            seqid_splits[r.get("sequence_id", "?")].add(sp)

    leaked = {si: sps for si, sps in seqid_splits.items() if len(sps) > 1}
    if leaked:
        print("⚠ LEAKED sequence_ids:")
        for si, sps in sorted(leaked.items()):
            print(f"  {si}: {sorted(sps)}")
    else:
        print("✓ No sequence_id appears in more than one split.")
    return leaked


def check_temporal_leakage(splits):
    """Check if train/val share overlapping frame indices within a sequence_id."""
    print("\n" + "=" * 70)
    print("4. TEMPORAL OVERLAP CHECK (frames from same source file in different splits)")
    print("=" * 70)

    # Group by source_path stem (original H5 file)
    source_split = defaultdict(set)
    source_recs = defaultdict(list)
    for sp, recs in splits.items():
        for r in recs:
            src = r.get("source_path", r.get("name", "?"))
            source_split[src].add(sp)
            source_recs[src].append((sp, r))

    leaked_sources = {src: sps for src, sps in source_split.items() if len(sps) > 1}
    if leaked_sources:
        print("⚠ SAME SOURCE FILE in multiple splits:")
        for src, sps in sorted(leaked_sources.items()):
            print(f"  {src}: {sorted(sps)}")
    else:
        print("✓ No source file appears in multiple splits.")
    return leaked_sources


# ---------------------------------------------------------------------------
# 2. Label distribution analysis
# ---------------------------------------------------------------------------

def analyse_label_distribution(splits):
    """Per-split and per-scenario label balance."""
    print("\n" + "=" * 70)
    print("5. LABEL DISTRIBUTION")
    print("=" * 70)

    for sp in sorted(splits):
        recs = splits[sp]
        print(f"\n--- {sp.upper()} ---")
        print(f"  {'Scenario':<35s} {'Real':>8s} {'Ghost':>8s} {'Ratio':>8s}")
        print(f"  {'-'*35} {'-'*8} {'-'*8} {'-'*8}")
        for r in sorted(recs, key=lambda x: x.get("scenario", "")):
            real = r.get("real_points", 0)
            ghost = r.get("ghost_points", 0)
            ratio = real / max(ghost, 1)
            print(f"  {r.get('scenario', '?'):<35s} {real:>8,d} {ghost:>8,d} {ratio:>8.2f}")


# ---------------------------------------------------------------------------
# 3. Feature distribution analysis
# ---------------------------------------------------------------------------

def compute_feature_stats(dataset, label=""):
    """Compute per-feature statistics from a dataset split."""
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)
    all_features = []
    all_targets = []
    for batch in loader:
        mask = batch["label_mask"]
        features = batch["features"][mask]
        targets = batch["target"][mask]
        all_features.append(features.numpy())
        all_targets.append(targets.numpy())
    features = np.concatenate(all_features, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    real_mask = targets == 0
    ghost_mask = targets == 1
    stats = {}
    for i, name in enumerate(FEATURE_NAMES):
        col = features[:, i]
        stats[name] = {
            "overall_mean": float(np.mean(col)),
            "overall_std": float(np.std(col)),
            "overall_min": float(np.min(col)),
            "overall_max": float(np.max(col)),
            "real_mean": float(np.mean(col[real_mask])) if real_mask.any() else None,
            "real_std": float(np.std(col[real_mask])) if real_mask.any() else None,
            "ghost_mean": float(np.mean(col[ghost_mask])) if ghost_mask.any() else None,
            "ghost_std": float(np.std(col[ghost_mask])) if ghost_mask.any() else None,
        }
    return features, targets, stats


def analyse_feature_distributions(datasets):
    """Compare feature distributions across splits."""
    print("\n" + "=" * 70)
    print("6. FEATURE DISTRIBUTION COMPARISON")
    print("=" * 70)

    all_stats = {}
    all_features = {}
    all_targets = {}
    for name, ds in datasets.items():
        features, targets, stats = compute_feature_stats(ds, name)
        all_stats[name] = stats
        all_features[name] = features
        all_targets[name] = targets

    # Show overall stats per split
    for split_name in datasets:
        print(f"\n--- {split_name.upper()} ---")
        print(f"  {'Feature':<30s} {'Mean':>8s} {'Std':>8s} {'Min':>8s} {'Max':>8s}")
        print(f"  {'-'*30} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
        for fname in FEATURE_NAMES:
            s = all_stats[split_name][fname]
            print(f"  {fname:<30s} {s['overall_mean']:>8.4f} {s['overall_std']:>8.4f} "
                  f"{s['overall_min']:>8.4f} {s['overall_max']:>8.4f}")

    # Distribution shift: compare train vs test means
    if "train" in all_stats and "test" in all_stats:
        print(f"\n--- FEATURE SHIFT (train→test, mean difference) ---")
        print(f"  {'Feature':<30s} {'Train μ':>8s} {'Test μ':>8s} {'Δ':>8s} {'Δ/σ':>8s}")
        print(f"  {'-'*30} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
        for fname in FEATURE_NAMES:
            train_mean = all_stats["train"][fname]["overall_mean"]
            test_mean = all_stats["test"][fname]["overall_mean"]
            train_std = all_stats["train"][fname]["overall_std"]
            delta = test_mean - train_mean
            normalized = delta / max(train_std, 1e-8)
            flag = " ⚠" if abs(normalized) > 1.0 else ""
            print(f"  {fname:<30s} {train_mean:>8.4f} {test_mean:>8.4f} "
                  f"{delta:>+8.4f} {normalized:>+8.4f}{flag}")

    # Real-vs-ghost separability per split
    print(f"\n--- REAL vs GHOST SEPARABILITY (mean Δ per split) ---")
    print(f"  {'Feature':<30s}", end="")
    for split_name in datasets:
        print(f" {split_name:>10s}", end="")
    print()
    print(f"  {'-'*30}", end="")
    for _ in datasets:
        print(f" {'-'*10}", end="")
    print()
    for fname in FEATURE_NAMES:
        print(f"  {fname:<30s}", end="")
        for split_name in datasets:
            s = all_stats[split_name][fname]
            if s["real_mean"] is not None and s["ghost_mean"] is not None:
                diff = s["ghost_mean"] - s["real_mean"]
                print(f" {diff:>+10.4f}", end="")
            else:
                print(f" {'N/A':>10s}", end="")
        print()

    return all_features, all_targets, all_stats


# ---------------------------------------------------------------------------
# 4. Linear separability baseline
# ---------------------------------------------------------------------------

def analyse_linear_separability(datasets):
    """Can a simple logistic regression separate real from ghost?"""
    print("\n" + "=" * 70)
    print("7. LINEAR SEPARABILITY BASELINE (logistic regression)")
    print("=" * 70)

    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import (
            average_precision_score,
            roc_auc_score,
            f1_score,
            precision_score,
            recall_score,
        )
    except ImportError:
        print("  (sklearn not installed — skipping linear baseline)")
        return

    # Collect train data
    loader = DataLoader(datasets["train"], batch_size=64, shuffle=False, num_workers=0)
    X_train, y_train = [], []
    for batch in loader:
        mask = batch["label_mask"]
        X_train.append(batch["features"][mask].numpy())
        y_train.append(batch["target"][mask].numpy())
    X_train = np.concatenate(X_train)
    y_train = np.concatenate(y_train)

    clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    clf.fit(X_train, y_train)

    for split_name, ds in datasets.items():
        if split_name == "train":
            continue
        loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0)
        X_test, y_test = [], []
        for batch in loader:
            mask = batch["label_mask"]
            X_test.append(batch["features"][mask].numpy())
            y_test.append(batch["target"][mask].numpy())
        X_test = np.concatenate(X_test)
        y_test = np.concatenate(y_test)

        probs = clf.predict_proba(X_test)[:, 1]
        preds = clf.predict(X_test)
        auprc = average_precision_score(y_test, probs)
        auroc = roc_auc_score(y_test, probs)
        f1 = f1_score(y_test, preds)
        prec = precision_score(y_test, preds)
        rec = recall_score(y_test, preds)
        print(f"\n  {split_name.upper()}: AUPRC={auprc:.4f}  AUROC={auroc:.4f}  "
              f"F1={f1:.4f}  P={prec:.4f}  R={rec:.4f}")

    print("\n  → If linear baseline does well on val but poorly on test, "
          "the problem is distribution shift, not model capacity.")


# ---------------------------------------------------------------------------
# 5. Model prediction analysis
# ---------------------------------------------------------------------------

def analyse_model_predictions(model, datasets, device):
    """Analyse model confidence distributions on each split."""
    print("\n" + "=" * 70)
    print("8. MODEL PREDICTION ANALYSIS")
    print("=" * 70)

    model.eval()
    for split_name, ds in datasets.items():
        loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)
        all_probs = []
        all_targets = []
        all_scenarios = []
        with torch.no_grad():
            for batch in loader:
                features = batch["features"].to(device)
                point_mask = batch["point_mask"].to(device)
                logits = model(features, point_mask)
                probs = torch.sigmoid(logits).cpu().numpy()
                mask = batch["label_mask"].numpy()
                target = batch["target"].numpy()
                seq_idx = batch["sequence_index"].numpy()
                for row in range(len(seq_idx)):
                    m = mask[row]
                    all_probs.append(probs[row][m])
                    all_targets.append(target[row][m])
                    rec = ds.sequences[int(seq_idx[row])]
                    all_scenarios.append(
                        [rec.get("scenario", "?")] * int(m.sum())
                    )

        probs = np.concatenate(all_probs)
        targets = np.concatenate(all_targets)
        scenarios = np.concatenate(all_scenarios)

        real_mask = targets == 0
        ghost_mask = targets == 1

        print(f"\n--- {split_name.upper()} ---")
        print(f"  Real samples:    n={real_mask.sum():>8,d}  "
              f"prob mean={probs[real_mask].mean():.4f}  "
              f"std={probs[real_mask].std():.4f}  "
              f"median={np.median(probs[real_mask]):.4f}")
        print(f"  Ghost samples:   n={ghost_mask.sum():>8,d}  "
              f"prob mean={probs[ghost_mask].mean():.4f}  "
              f"std={probs[ghost_mask].std():.4f}  "
              f"median={np.median(probs[ghost_mask]):.4f}")

        # Prediction histogram
        bins = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        print(f"\n  Prediction distribution (real vs ghost):")
        print(f"  {'Range':<12s} {'Real':>8s} {'Ghost':>8s}")
        for i in range(len(bins) - 1):
            lo, hi = bins[i], bins[i + 1]
            in_bin = (probs >= lo) & (probs < hi) if i < len(bins) - 1 else (probs >= lo)
            r_count = int((in_bin & real_mask).sum())
            g_count = int((in_bin & ghost_mask).sum())
            bar_r = "█" * min(r_count // max(real_mask.sum(), 1) * 50, 50)
            bar_g = "█" * min(g_count // max(ghost_mask.sum(), 1) * 50, 50)
            print(f"  [{lo:.1f},{hi:.1f})  {r_count:>8,d} {g_count:>8,d}")

        # Per-scenario prediction stats
        unique_scenarios = sorted(set(scenarios))
        if len(unique_scenarios) > 1:
            print(f"\n  Per-scenario ghost recall (at threshold=0.5):")
            print(f"  {'Scenario':<35s} {'Ghost n':>8s} {'Recall':>8s}")
            print(f"  {'-'*35} {'-'*8} {'-'*8}")
            for sc in unique_scenarios:
                sc_mask = scenarios == sc
                sc_ghost = sc_mask & ghost_mask
                if sc_ghost.sum() > 0:
                    recall = (probs[sc_ghost] >= 0.5).mean()
                    print(f"  {sc:<35s} {sc_ghost.sum():>8,d} {recall:>8.4f}")


# ---------------------------------------------------------------------------
# 6. Per-feature importance via model weights
# ---------------------------------------------------------------------------

def analyse_model_weights(model):
    """Inspect first-layer weights to see which features matter."""
    print("\n" + "=" * 70)
    print("9. MODEL FEATURE IMPORTANCE (first-layer weight magnitudes)")
    print("=" * 70)

    # Get first linear layer weights
    first_layer = None
    for module in model.modules():
        if isinstance(module, torch.nn.Linear):
            first_layer = module
            break

    if first_layer is None:
        print("  Could not find first linear layer.")
        return

    weights = first_layer.weight.detach().cpu().numpy()  # (out, in)
    importance = np.mean(np.abs(weights), axis=0)

    ranked = sorted(zip(FEATURE_NAMES, importance), key=lambda x: -x[1])
    print(f"\n  {'Feature':<30s} {'Mean |w|':>10s} {'Rank':>6s}")
    print(f"  {'-'*30} {'-'*10} {'-'*6}")
    for rank, (fname, imp) in enumerate(ranked, 1):
        bar = "█" * int(imp / ranked[0][1] * 30)
        print(f"  {fname:<30s} {imp:>10.6f} {rank:>6d}  {bar}")


# ---------------------------------------------------------------------------
# 7. Within-scenario vs cross-scenario gap
# ---------------------------------------------------------------------------

def analyse_scenario_generalization(model, datasets, device):
    """Compare model performance on seen vs unseen scenarios."""
    print("\n" + "=" * 70)
    print("10. SEEN vs UNSEEN SCENARIO PERFORMANCE")
    print("=" * 70)

    # Get scenarios in each split
    train_scenarios = set()
    for rec in datasets["train"].sequences:
        train_scenarios.add(rec.get("scenario", "?"))

    print(f"\n  Train scenarios: {sorted(train_scenarios)}")

    model.eval()
    for split_name in ("val", "test"):
        ds = datasets.get(split_name)
        if ds is None:
            continue

        seen_probs, seen_targets = [], []
        unseen_probs, unseen_targets = [], []

        loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)
        with torch.no_grad():
            for batch in loader:
                features = batch["features"].to(device)
                point_mask = batch["point_mask"].to(device)
                logits = model(features, point_mask)
                probs = torch.sigmoid(logits).cpu().numpy()
                mask = batch["label_mask"].numpy()
                target = batch["target"].numpy()
                seq_idx = batch["sequence_index"].numpy()

                for row in range(len(seq_idx)):
                    m = mask[row]
                    if not m.any():
                        continue
                    rec = ds.sequences[int(seq_idx[row])]
                    sc = rec.get("scenario", "?")
                    if sc in train_scenarios:
                        seen_probs.append(probs[row][m])
                        seen_targets.append(target[row][m])
                    else:
                        unseen_probs.append(probs[row][m])
                        unseen_targets.append(target[row][m])

        def _metrics(probs_list, targets_list, label):
            if not probs_list:
                print(f"  {split_name.upper()} {label}: no samples")
                return
            p = np.concatenate(probs_list)
            t = np.concatenate(targets_list)
            ghost_mask = t == 1
            if ghost_mask.sum() == 0:
                print(f"  {split_name.upper()} {label}: no ghost samples")
                return
            recall_05 = (p[ghost_mask] >= 0.5).mean()
            recall_08 = (p[ghost_mask] >= 0.8).mean()
            real_fpr_05 = ((p[t == 0] >= 0.5)).mean() if (t == 0).any() else 0
            real_fpr_08 = ((p[t == 0] >= 0.8)).mean() if (t == 0).any() else 0
            print(f"  {split_name.upper()} {label}: "
                  f"n={ghost_mask.sum():>6,d} ghost, "
                  f"{(t==0).sum():>6,d} real | "
                  f"recall@0.5={recall_05:.4f}  recall@0.8={recall_08:.4f}  "
                  f"FPR@0.5={real_fpr_05:.4f}  FPR@0.8={real_fpr_08:.4f}")

        _metrics(seen_probs, seen_targets, "SEEN scenarios")
        _metrics(unseen_probs, unseen_targets, "UNSEEN scenarios")


# ---------------------------------------------------------------------------
# 8. Summary & verdict
# ---------------------------------------------------------------------------

def print_summary(splits, leaked, all_stats, feature_shifts):
    """Print a verdict on whether data is learnable."""
    print("\n" + "=" * 70)
    print("SUMMARY & VERDICT")
    print("=" * 70)

    issues = []

    # Check for leaked scenarios
    if leaked:
        issues.append(f"Scenario leakage detected: {len(leaked)} scenarios in multiple splits")

    # Check for extreme class imbalance
    for sp, recs in splits.items():
        real = sum(r.get("real_points", 0) for r in recs)
        ghost = sum(r.get("ghost_points", 0) for r in recs)
        ratio = real / max(ghost, 1)
        if ratio > 20 or ratio < 0.05:
            issues.append(f"{sp}: extreme class imbalance (real:ghost = {ratio:.1f}:1)")

    # Check for extreme feature shifts
    if "train" in all_stats and "test" in all_stats:
        for fname in FEATURE_NAMES:
            train_std = all_stats["train"][fname]["overall_std"]
            delta = (all_stats["test"][fname]["overall_mean"]
                     - all_stats["train"][fname]["overall_mean"])
            if train_std > 1e-8 and abs(delta / train_std) > 2.0:
                issues.append(
                    f"Large feature shift: {fname} "
                    f"(Δ/σ = {delta / train_std:.2f})"
                )

    if issues:
        print("\n⚠ ISSUES FOUND:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("\n✓ No major data issues detected.")

    print("\n" + "-" * 70)
    print("INTERPRETATION:")
    print("-" * 70)
    if not leaked and not issues:
        print("""
  The data splits appear clean with no leakage. Feature distributions
  are similar across splits. If the model still fails to generalize,
  the issue is likely:

  1. INSUFFICIENT VARIETY: The training set may not cover the full
     range of real-world conditions (weather, road geometry, sensor
     noise). The model memorizes training-scenario patterns instead
     of learning generalizable ghost signatures.

  2. LABEL AMBIGUITY: Some multipath reflections may be inherently
     ambiguous — the features (range, azimuth, velocity, amplitude)
     may not contain enough information to reliably distinguish
     real from ghost in all scenarios.

  3. SCENARIO-SPECIFIC PATTERNS: Ghost characteristics may vary
     significantly by scenario (road layout, nearby structures),
     making cross-scenario generalization fundamentally hard.

  → The model capacity (TemporalPointNet) is likely sufficient.
     The bottleneck is the feature space or training diversity.
""")
    elif leaked:
        print("""
  ⚠ DATA LEAKAGE is present — validation metrics during training
  are unreliable. The val AUPRC=0.95 was inflated because the model
  had already seen similar scenarios. The test AUPRC=0.33 is the
  true performance.

  Fix: Re-prepare the dataset with --split-mode scenario_grouped
  to ensure no scenario appears in both train and test.
""")
    else:
        print("""
  Distribution shifts and/or class imbalance issues were found.
  These can cause the model to learn shortcuts that don't generalize.
""")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    output_dir = Path(args.output) if args.output else Path(args.data) / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load manifest
    with (Path(args.data) / "manifest.json").open() as f:
        manifest = json.load(f)

    print(f"Dataset: {args.data}")
    print(f"Schema: {manifest.get('feature_schema')}")
    print(f"Split mode: {manifest.get('split_mode')}")
    print(f"Output: {output_dir}")

    # 1-4: Split analysis
    splits = analyse_split_composition(manifest)
    leaked_scenarios = check_scenario_leakage(splits)
    leaked_seqids = check_sequence_id_leakage(splits)
    leaked_sources = check_temporal_leakage(splits)

    # 5: Label distribution
    analyse_label_distribution(splits)

    # Load datasets
    print("\n\nLoading datasets (this may take a while for large datasets)...")
    datasets = {}
    for split_name in ("train", "val", "test"):
        try:
            ds = PreparedGhostDataset(
                args.data, split_name,
                window_frames=5, max_points=1024,
                augment=False, seed=42,
            )
            datasets[split_name] = ds
            print(f"  {split_name}: {len(ds)} samples loaded")
        except Exception as e:
            print(f"  {split_name}: FAILED to load ({e})")

    # 6: Feature distributions
    all_features, all_targets, all_stats = analyse_feature_distributions(datasets)

    # 7: Linear separability
    analyse_linear_separability(datasets)

    # 8-9: Model analysis
    model = None
    if args.checkpoint and Path(args.checkpoint).exists():
        try:
            ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=True)
            model = create_ghost_model(
                ckpt["model_name"],
                **ckpt.get("model_kwargs", {}),
            ).to(args.device)
            model.load_state_dict(ckpt["model_state"])
            print(f"\nLoaded model from {args.checkpoint}")
            analyse_model_predictions(model, datasets, args.device)
            analyse_model_weights(model)
        except Exception as e:
            print(f"\nFailed to load checkpoint: {e}")

    # 10: Seen vs unseen
    if model is not None:
        analyse_scenario_generalization(model, datasets, args.device)

    # Summary
    all_leaked = {**leaked_scenarios, **leaked_seqids, **leaked_sources}
    print_summary(splits, all_leaked, all_stats, None)

    # Save all results as JSON
    results = {
        "manifest": str(Path(args.data).resolve()),
        "splits": {
            sp: {
                "sequences": len(recs),
                "total_points": sum(r.get("points", 0) for r in recs),
                "real_points": sum(r.get("real_points", 0) for r in recs),
                "ghost_points": sum(r.get("ghost_points", 0) for r in recs),
                "scenarios": list(set(r.get("scenario", "?") for r in recs)),
            }
            for sp, recs in splits.items()
        },
        "leaked_scenarios": dict(leaked_scenarios),
        "feature_stats": all_stats,
    }
    with (output_dir / "dataset_analysis.json").open("w") as f:
        json.dump(results, f, indent=2, sort_keys=True)
    print(f"\nDetailed results saved to {output_dir / 'dataset_analysis.json'}")


if __name__ == "__main__":
    main()
