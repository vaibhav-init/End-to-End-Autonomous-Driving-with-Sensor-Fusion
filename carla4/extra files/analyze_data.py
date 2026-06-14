#!/usr/bin/env python3
"""
Analyze collected crash data before training.
Run: python analyze_data.py
"""

import pandas as pd
import numpy as np
import os

DATA_PATH = 'dataset_crash/data.csv'

def main():
    if not os.path.exists(DATA_PATH):
        print(f"❌ No data file at {DATA_PATH}")
        return

    df = pd.read_csv(DATA_PATH)

    print("=" * 70)
    print("CRASH DATA ANALYSIS")
    print("=" * 70)

    # ---- Basic stats ----
    print(f"\n📊 DATASET SIZE")
    print(f"  Total rows:     {len(df):,}")
    print(f"  Scenarios:      {df['scenario_id'].nunique()}")
    print(f"  Duration:       {len(df) / 20 / 60:.1f} minutes ({len(df)/20:.0f}s at 20 FPS)")

    # ---- Label distribution ----
    pos = (df['collision_within_2s'] == 1).sum()
    neg = (df['collision_within_2s'] == 0).sum()
    ratio = pos / max(1, len(df)) * 100
    print(f"\n🏷️  LABEL DISTRIBUTION")
    print(f"  Positive (crash within 2s):  {pos:,}  ({ratio:.2f}%)")
    print(f"  Negative (safe):             {neg:,}  ({100-ratio:.2f}%)")
    print(f"  Imbalance ratio:             1:{neg//max(1,pos)}")

    if pos == 0:
        print(f"\n  ⚠️  NO POSITIVE LABELS! No crashes recorded.")
        print(f"      Need more data or adjust crash probabilities.")

    # ---- Feature statistics ----
    features = ['ego_speed', 'ego_acceleration', 'nearest_distance',
                'relative_velocity', 'ttc', 'obstacle_speed',
                'obstacle_type', 'lateral_offset', 'ego_steering',
                'rear_distance', 'rear_relative_velocity', 'rear_ttc',
                'rear_obstacle_speed', 'rear_obstacle_type']

    print(f"\n📈 FEATURE STATISTICS")
    print(f"  {'Feature':<20s} {'Mean':>8s} {'Std':>8s} {'Min':>8s} {'Max':>8s}")
    print(f"  {'-'*20} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    for f in features:
        if f in df.columns:
            print(f"  {f:<20s} {df[f].mean():8.2f} {df[f].std():8.2f} "
                  f"{df[f].min():8.2f} {df[f].max():8.2f}")

    # ---- Positive vs Negative comparison ----
    if pos > 0:
        print(f"\n🔍 POSITIVE vs NEGATIVE FRAMES (key features)")
        print(f"  {'Feature':<20s} {'Safe (avg)':>10s} {'Crash (avg)':>12s} {'Difference':>10s}")
        print(f"  {'-'*20} {'-'*10} {'-'*12} {'-'*10}")
        key_features = ['ego_speed', 'nearest_distance', 'relative_velocity', 'ttc']
        for f in key_features:
            if f in df.columns:
                safe_mean = df[df['collision_within_2s'] == 0][f].mean()
                crash_mean = df[df['collision_within_2s'] == 1][f].mean()
                diff = crash_mean - safe_mean
                print(f"  {f:<20s} {safe_mean:10.2f} {crash_mean:12.2f} {diff:+10.2f}")

    # ---- Per-scenario breakdown ----
    print(f"\n📋 PER-SCENARIO BREAKDOWN")
    print(f"  {'Scenario':<10s} {'Frames':>8s} {'Crashes':>8s} {'AvgSpeed':>10s} {'MinTTC':>8s}")
    print(f"  {'-'*10} {'-'*8} {'-'*8} {'-'*10} {'-'*8}")
    for sid in sorted(df['scenario_id'].unique()):
        sdf = df[df['scenario_id'] == sid]
        crashes = (sdf['collision_within_2s'] == 1).sum()
        avg_spd = sdf['ego_speed'].mean()
        min_ttc = sdf['ttc'].min()
        print(f"  {sid:<10d} {len(sdf):8d} {crashes:8d} {avg_spd:10.1f} {min_ttc:8.2f}")

    # ---- Data quality checks ----
    print(f"\n⚠️  DATA QUALITY")
    zero_speed = (df['ego_speed'] < 0.1).sum()
    zero_pct = zero_speed / len(df) * 100
    print(f"  Frames with speed ≈ 0:   {zero_speed:,} ({zero_pct:.1f}%)")
    if zero_pct > 50:
        print(f"     → Too much idle data! Ego is stopped most of the time.")

    max_dist = (df['nearest_distance'] >= 49).sum()
    max_pct = max_dist / len(df) * 100
    print(f"  Frames with no obstacle: {max_dist:,} ({max_pct:.1f}%)")

    bad_accel = (df['ego_acceleration'].abs() > 50).sum()
    print(f"  Frames with |accel|>50:  {bad_accel:,} (noise)")

    # ---- Recommendation ----
    print(f"\n💡 RECOMMENDATION")
    if len(df) < 10000:
        print(f"  ❌ Only {len(df):,} rows. Need at least 20,000+ for decent MLP.")
        print(f"     Keep collecting! (~17 minutes more at 20 FPS)")
    elif len(df) < 50000:
        print(f"  ⚠️  {len(df):,} rows. Usable but more data = better model.")
        print(f"     Target: 50,000-100,000 rows (40-80 minutes of data)")
    else:
        print(f"  ✅ {len(df):,} rows. Good amount for MLP training!")

    if pos < 100:
        print(f"  ❌ Only {pos} positive frames. Need at least 200+ for learning.")
        print(f"     Increase crash probabilities or collect longer.")
    elif pos < 500:
        print(f"  ⚠️  {pos} positive frames. Okay with oversampling. More is better.")
    else:
        print(f"  ✅ {pos} positive frames. Good for training!")

    print(f"\n  IDEAL TARGETS:")
    print(f"    Rows:      50,000 - 100,000 (40-80 min of driving)")
    print(f"    Positive:  500 - 2,000 frames (1-4% of total)")
    print(f"    Crashes:   15-50 collision events")
    print(f"    Scenarios: 20-40 scenarios")

    print(f"\n{'=' * 70}")


if __name__ == '__main__':
    main()
