#!/usr/bin/env python3
"""
Model Validation System for Crash Predictor
=============================================

Generates comprehensive validation report with:
  - ROC Curve + AUC
  - Precision-Recall Curve + AUC-PR
  - Confusion Matrix Heatmap
  - Prediction Distribution (safe vs crash)
  - Threshold Sweep Analysis
  - Feature Importance (permutation)
  - Scenario-based split evaluation

Usage:
    python validate_model.py
    python validate_model.py --data dataset_crash/data.csv --model model/crash_mlp.pt
"""

import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import pickle
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_recall_curve, roc_curve,
    confusion_matrix, classification_report,
    f1_score, precision_score, recall_score, accuracy_score
)
from sklearn.preprocessing import StandardScaler

# ============================================================================
# Config
# ============================================================================
DATA_PATH = 'dataset_crash/data.csv'
MODEL_PATH = 'model/crash_mlp.pt'
SCALER_PATH = 'model/scaler.pkl'
OUTPUT_DIR = 'validation_results'

FEATURE_COLS = [
    'ego_speed', 'ego_acceleration', 'nearest_distance',
    'relative_velocity', 'ttc', 'obstacle_speed',
    'obstacle_type', 'lateral_offset', 'ego_steering',
    'rear_distance', 'rear_relative_velocity', 'rear_ttc',
    'rear_obstacle_speed', 'rear_obstacle_type'
]
LABEL_COL = 'collision_within_2s'


# ============================================================================
# MLP Model (must match train_mlp.py)
# ============================================================================
class CrashMLP(nn.Module):
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
# Scenario-Based Split
# ============================================================================
def scenario_based_split(df, test_ratio=0.2, seed=42):
    """Split by scenario_id so test set contains entirely unseen scenarios."""
    rng = np.random.RandomState(seed)
    scenarios = df['scenario_id'].unique()
    rng.shuffle(scenarios)
    split_idx = int(len(scenarios) * (1 - test_ratio))
    train_scenarios = scenarios[:split_idx]
    test_scenarios = scenarios[split_idx:]
    train_df = df[df['scenario_id'].isin(train_scenarios)].reset_index(drop=True)
    test_df = df[df['scenario_id'].isin(test_scenarios)].reset_index(drop=True)
    return train_df, test_df


# ============================================================================
# Plots
# ============================================================================
def plot_roc_curve(y_true, y_prob, save_path):
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    auc_val = roc_auc_score(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color='#2196F3', lw=2.5, label=f'ROC Curve (AUC = {auc_val:.4f})')
    ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5, label='Random (AUC = 0.5)')
    ax.fill_between(fpr, tpr, alpha=0.1, color='#2196F3')
    ax.set_xlabel('False Positive Rate', fontsize=13)
    ax.set_ylabel('True Positive Rate', fontsize=13)
    ax.set_title('ROC Curve — Crash Predictor', fontsize=15, fontweight='bold')
    ax.legend(fontsize=12, loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.01])
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  ✅ ROC curve saved → {save_path}")
    return auc_val


def plot_pr_curve(y_true, y_prob, save_path):
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    auc_pr = average_precision_score(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recall, precision, color='#FF5722', lw=2.5, label=f'PR Curve (AUC-PR = {auc_pr:.4f})')
    baseline = y_true.sum() / len(y_true)
    ax.axhline(y=baseline, color='gray', linestyle='--', lw=1, label=f'Baseline ({baseline:.3f})')
    ax.fill_between(recall, precision, alpha=0.1, color='#FF5722')
    ax.set_xlabel('Recall', fontsize=13)
    ax.set_ylabel('Precision', fontsize=13)
    ax.set_title('Precision-Recall Curve — Crash Predictor', fontsize=15, fontweight='bold')
    ax.legend(fontsize=12, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.01])
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  ✅ PR curve saved → {save_path}")
    return auc_pr


def plot_confusion_matrix(y_true, y_pred, save_path):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    fig.colorbar(im, ax=ax, shrink=0.8)

    labels = ['Safe (0)', 'Crash (1)']
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_yticklabels(labels, fontsize=12)
    ax.set_xlabel('Predicted', fontsize=13)
    ax.set_ylabel('Actual', fontsize=13)
    ax.set_title('Confusion Matrix', fontsize=15, fontweight='bold')

    for i in range(2):
        for j in range(2):
            color = 'white' if cm[i, j] > cm.max() / 2 else 'black'
            ax.text(j, i, f'{cm[i, j]:,}', ha='center', va='center',
                    fontsize=18, fontweight='bold', color=color)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  ✅ Confusion matrix saved → {save_path}")


def plot_prediction_distribution(y_true, y_prob, save_path):
    fig, ax = plt.subplots(figsize=(9, 6))
    safe_probs = y_prob[y_true == 0]
    crash_probs = y_prob[y_true == 1]

    ax.hist(safe_probs, bins=50, alpha=0.6, color='#4CAF50', label=f'Safe frames (n={len(safe_probs):,})', density=True)
    if len(crash_probs) > 0:
        ax.hist(crash_probs, bins=50, alpha=0.6, color='#F44336', label=f'Crash frames (n={len(crash_probs):,})', density=True)
    ax.axvline(x=0.5, color='black', linestyle='--', lw=1.5, label='Threshold = 0.5')
    ax.set_xlabel('Predicted Crash Probability', fontsize=13)
    ax.set_ylabel('Density', fontsize=13)
    ax.set_title('Prediction Distribution — Safe vs Crash Frames', fontsize=15, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  ✅ Prediction distribution saved → {save_path}")


def plot_threshold_sweep(y_true, y_prob, save_path):
    thresholds = np.arange(0.05, 0.96, 0.05)
    precisions, recalls, f1s = [], [], []

    for t in thresholds:
        pred = (y_prob >= t).astype(int)
        tp = ((pred == 1) & (y_true == 1)).sum()
        fp = ((pred == 1) & (y_true == 0)).sum()
        fn = ((pred == 0) & (y_true == 1)).sum()
        p = tp / max(1, tp + fp)
        r = tp / max(1, tp + fn)
        f = 2 * p * r / max(1e-8, p + r)
        precisions.append(p)
        recalls.append(r)
        f1s.append(f)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(thresholds, precisions, 'o-', color='#2196F3', lw=2, label='Precision')
    ax.plot(thresholds, recalls, 's-', color='#FF9800', lw=2, label='Recall')
    ax.plot(thresholds, f1s, '^-', color='#9C27B0', lw=2, label='F1-Score')
    ax.axvline(x=0.5, color='gray', linestyle='--', lw=1, alpha=0.5)
    ax.set_xlabel('Decision Threshold', fontsize=13)
    ax.set_ylabel('Score', fontsize=13)
    ax.set_title('Threshold Sweep — Precision / Recall / F1', fontsize=15, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([-0.02, 1.02])
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  ✅ Threshold sweep saved → {save_path}")


def plot_feature_importance(model, X_test, y_test, feature_names, scaler, device, save_path):
    """Permutation importance: shuffle each feature, measure AUC drop."""
    model.eval()

    # Baseline AUC
    X_scaled = scaler.transform(X_test)
    with torch.no_grad():
        base_prob = model(torch.tensor(X_scaled, dtype=torch.float32).to(device)).cpu().numpy()
    base_auc = roc_auc_score(y_test, base_prob)

    importances = []
    for i in range(X_test.shape[1]):
        X_perm = X_test.copy()
        np.random.shuffle(X_perm[:, i])
        X_perm_scaled = scaler.transform(X_perm)
        with torch.no_grad():
            perm_prob = model(torch.tensor(X_perm_scaled, dtype=torch.float32).to(device)).cpu().numpy()
        perm_auc = roc_auc_score(y_test, perm_prob)
        importances.append(base_auc - perm_auc)

    # Sort
    indices = np.argsort(importances)[::-1]
    sorted_names = [feature_names[i] for i in indices]
    sorted_imp = [importances[i] for i in indices]

    fig, ax = plt.subplots(figsize=(9, 6))
    colors = ['#F44336' if v > 0 else '#9E9E9E' for v in sorted_imp]
    bars = ax.barh(range(len(sorted_names)), sorted_imp, color=colors, edgecolor='white')
    ax.set_yticks(range(len(sorted_names)))
    ax.set_yticklabels(sorted_names, fontsize=11)
    ax.set_xlabel('AUC Drop When Feature Shuffled', fontsize=13)
    ax.set_title('Feature Importance (Permutation)', fontsize=15, fontweight='bold')
    ax.invert_yaxis()
    ax.grid(True, axis='x', alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  ✅ Feature importance saved → {save_path}")
    return dict(zip(feature_names, importances))


# ============================================================================
# Text Report
# ============================================================================
def write_report(report_path, metrics, auc_roc, auc_pr, feat_imp, n_train, n_test,
                 train_scenarios, test_scenarios, pos_test, neg_test):
    with open(report_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("CRASH PREDICTOR — VALIDATION REPORT\n")
        f.write("=" * 70 + "\n\n")

        f.write("DATA SPLIT (Scenario-Based)\n")
        f.write(f"  Train scenarios: {len(train_scenarios)} → {n_train:,} frames\n")
        f.write(f"  Test scenarios:  {len(test_scenarios)} → {n_test:,} frames\n")
        f.write(f"  Test positive:   {pos_test:,} ({pos_test/max(1,n_test)*100:.2f}%)\n")
        f.write(f"  Test negative:   {neg_test:,} ({neg_test/max(1,n_test)*100:.2f}%)\n\n")

        f.write("KEY METRICS\n")
        f.write(f"  AUC-ROC:         {auc_roc:.4f}\n")
        f.write(f"  AUC-PR:          {auc_pr:.4f}\n")
        for k, v in metrics.items():
            f.write(f"  {k:<18s} {v:.4f}\n")

        f.write("\nFEATURE IMPORTANCE (AUC drop)\n")
        if feat_imp:
            for name, imp in sorted(feat_imp.items(), key=lambda x: -x[1]):
                f.write(f"  {name:<22s} {imp:+.4f}\n")

        f.write("\n" + "=" * 70 + "\n")
    print(f"  ✅ Report saved → {report_path}")


# ============================================================================
# Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description='Validate crash prediction model')
    parser.add_argument('--data', default=DATA_PATH)
    parser.add_argument('--model', default=MODEL_PATH)
    parser.add_argument('--scaler', default=SCALER_PATH)
    parser.add_argument('--output', default=OUTPUT_DIR)
    parser.add_argument('--threshold', type=float, default=0.5)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    print("=" * 70)
    print("CRASH PREDICTOR — MODEL VALIDATION")
    print("=" * 70)

    # ---- Check files ----
    for path, name in [(args.data, 'Data'), (args.model, 'Model'), (args.scaler, 'Scaler')]:
        if not os.path.exists(path):
            print(f"  ❌ {name} not found: {path}")
            return
        print(f"  ✅ {name}: {path}")

    # ---- Load data ----
    df = pd.read_csv(args.data)
    print(f"\n  Loaded {len(df):,} rows, {df['scenario_id'].nunique()} scenarios")

    # Check features exist
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        print(f"  ❌ Missing columns: {missing}")
        return

    # ---- Scenario-based split ----
    train_df, test_df = scenario_based_split(df, test_ratio=0.2)

    train_scenarios = train_df['scenario_id'].unique()
    test_scenarios = test_df['scenario_id'].unique()
    print(f"\n  Train: {len(train_scenarios)} scenarios → {len(train_df):,} frames")
    print(f"  Test:  {len(test_scenarios)} scenarios → {len(test_df):,} frames")

    X_test = test_df[FEATURE_COLS].values.astype(np.float32)
    y_test = test_df[LABEL_COL].values.astype(np.float32)

    pos_test = int(y_test.sum())
    neg_test = len(y_test) - pos_test
    print(f"  Test positive: {pos_test:,} ({pos_test/max(1,len(y_test))*100:.2f}%)")
    print(f"  Test negative: {neg_test:,}")

    if pos_test < 2:
        print(f"\n  ❌ Not enough positive test samples ({pos_test}). Need more crash data.")
        return

    # ---- Load model + scaler ----
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Fit scaler on train data (same as training)
    X_train = train_df[FEATURE_COLS].values.astype(np.float32)
    scaler = StandardScaler()
    scaler.fit(X_train)

    # Also load saved scaler for comparison
    with open(args.scaler, 'rb') as f:
        saved_scaler = pickle.load(f)

    model = CrashMLP(input_dim=len(FEATURE_COLS)).to(device)
    state = torch.load(args.model, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    print(f"\n  Model loaded on {device}")

    # ---- Predict on test set ----
    X_test_scaled = saved_scaler.transform(X_test)
    with torch.no_grad():
        tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
        y_prob = model(tensor).cpu().numpy()

    y_pred = (y_prob >= args.threshold).astype(int)

    # ---- Compute metrics ----
    print(f"\n{'='*70}")
    print(f"RESULTS (threshold={args.threshold})")
    print(f"{'='*70}")

    auc_roc = roc_auc_score(y_test, y_prob)
    auc_pr = average_precision_score(y_test, y_prob)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    metrics = {
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1-Score': f1,
    }

    print(f"  AUC-ROC:    {auc_roc:.4f}")
    print(f"  AUC-PR:     {auc_pr:.4f}")
    print(f"  Accuracy:   {acc:.4f}")
    print(f"  Precision:  {prec:.4f}")
    print(f"  Recall:     {rec:.4f}")
    print(f"  F1-Score:   {f1:.4f}")

    print(f"\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Safe', 'Crash'], zero_division=0))

    # ---- Threshold sweep (print) ----
    print(f"\n  Threshold Analysis:")
    print(f"  {'Thresh':>8s} {'Prec':>8s} {'Recall':>8s} {'F1':>8s} {'FP':>6s} {'FN':>6s}")
    print(f"  {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*6} {'-'*6}")
    for thresh in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        pred_t = (y_prob >= thresh).astype(int)
        tp = ((pred_t == 1) & (y_test == 1)).sum()
        fp = ((pred_t == 1) & (y_test == 0)).sum()
        fn = ((pred_t == 0) & (y_test == 1)).sum()
        p = tp / max(1, tp + fp)
        r = tp / max(1, tp + fn)
        f = 2 * p * r / max(1e-8, p + r)
        print(f"  {thresh:8.1f} {p:8.3f} {r:8.3f} {f:8.3f} {int(fp):6d} {int(fn):6d}")

    # ---- Generate all plots ----
    print(f"\n{'='*70}")
    print("GENERATING PLOTS")
    print(f"{'='*70}")

    plot_roc_curve(y_test, y_prob, os.path.join(args.output, 'roc_curve.png'))
    plot_pr_curve(y_test, y_prob, os.path.join(args.output, 'pr_curve.png'))
    plot_confusion_matrix(y_test, y_pred, os.path.join(args.output, 'confusion_matrix.png'))
    plot_prediction_distribution(y_test, y_prob, os.path.join(args.output, 'prediction_dist.png'))
    plot_threshold_sweep(y_test, y_prob, os.path.join(args.output, 'threshold_analysis.png'))

    feat_imp = plot_feature_importance(
        model, X_test, y_test, FEATURE_COLS, saved_scaler, device,
        os.path.join(args.output, 'feature_importance.png'))

    # ---- Write text report ----
    write_report(
        os.path.join(args.output, 'validation_report.txt'),
        metrics, auc_roc, auc_pr, feat_imp,
        len(train_df), len(test_df),
        train_scenarios, test_scenarios, pos_test, neg_test)

    print(f"\n{'='*70}")
    print(f"VALIDATION COMPLETE")
    print(f"  All outputs in: {args.output}/")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
