#!/usr/bin/env python3
"""Power analysis for the closed-loop scenario runs.

Reads a ``per_run_metrics.csv`` produced by ``analyze_results.py`` and answers
the question that gates every remaining simulator hour:

    how many seeds per arm are needed to detect an effect of size delta?

The primary endpoint is phantom-brake events per km; jerk and collision rate
are reported alongside. For a two-sample comparison with n runs per arm:

    MDE = (z_{1-alpha/2} + z_{power}) * sqrt(2 * sigma^2 / n)

This is an approximation, stated as such. It is still far better than running
another 40-run arm whose binomial interval is 22 points wide.

Usage:
  python3 power_analysis.py comparison/per_run_metrics.csv --out-dir power
"""

import argparse
import math
import os

import numpy as np
import pandas as pd

Z_ALPHA = 1.959963984540054
Z_POWER = 0.8416212335729143


def _clean(series):
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    return values[np.isfinite(values)]


def mde_continuous(std, n, alpha_z=Z_ALPHA, power_z=Z_POWER):
    if n <= 1 or not np.isfinite(std):
        return float("nan")
    return (alpha_z + power_z) * math.sqrt(2.0 * std * std / n)


def mde_proportion(p, n, alpha_z=Z_ALPHA, power_z=Z_POWER):
    if n <= 1 or not np.isfinite(p):
        return float("nan")
    variance = p * (1.0 - p)
    return (alpha_z + power_z) * math.sqrt(2.0 * variance / n)


def bootstrap_mean_ci(values, reps=5000, seed=42):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(reps, values.size), replace=True)
    means = samples.mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def analyse(df, alpha_z=Z_ALPHA, power_z=Z_POWER):
    rows = []
    group_cols = [c for c in ("driver", "scenario") if c in df.columns]
    if not group_cols:
        group_cols = ["driver"]
    for key, group in df.groupby(group_cols, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        row = dict(zip(group_cols, key))
        phantom = _clean(group.get("phantom_brake_per_km", pd.Series(dtype=float)))
        jerk = _clean(group.get("jerk_rms_mps3", pd.Series(dtype=float)))
        collided = _clean(group.get("collided", pd.Series(dtype=float)))
        row["n_runs"] = int(len(group))
        row["phantom_n"] = int(phantom.size)
        row["phantom_mean_per_km"] = (
            float(np.mean(phantom)) if phantom.size else float("nan")
        )
        row["phantom_std_per_km"] = (
            float(np.std(phantom, ddof=1)) if phantom.size > 1 else float("nan")
        )
        row["phantom_ci_low"], row["phantom_ci_high"] = bootstrap_mean_ci(phantom)
        row["phantom_mde_n"] = mde_continuous(
            row["phantom_std_per_km"], phantom.size, alpha_z, power_z
        )
        row["jerk_mean"] = float(np.mean(jerk)) if jerk.size else float("nan")
        row["jerk_std"] = (
            float(np.std(jerk, ddof=1)) if jerk.size > 1 else float("nan")
        )
        row["jerk_mde_n"] = mde_continuous(
            row["jerk_std"], jerk.size, alpha_z, power_z
        )
        row["collision_rate"] = (
            float(np.mean(collided)) if collided.size else float("nan")
        )
        row["collision_mde_n"] = mde_proportion(
            row["collision_rate"], collided.size, alpha_z, power_z
        )
        rows.append(row)
    return pd.DataFrame(rows)


def required_n(std, delta, alpha_z=Z_ALPHA, power_z=Z_POWER):
    if not np.isfinite(std) or delta <= 0.0:
        return float("nan")
    return math.ceil(2.0 * ((alpha_z + power_z) * std / delta) ** 2)


def required_n_proportion(p, delta, alpha_z=Z_ALPHA, power_z=Z_POWER):
    if not np.isfinite(p) or delta <= 0.0:
        return float("nan")
    variance = p * (1.0 - p)
    return math.ceil(2.0 * ((alpha_z + power_z) ** 2) * variance / (delta ** 2))


def write_markdown(summary, path):
    lines = [
        "# Power analysis",
        "",
        "Primary endpoint: phantom-brake events per km. "
        "MDE is the minimum detectable difference between two independent "
        "arms of the observed size.",
        "",
        "| driver | scenario | n | phantom/km mean | sd | 95% CI | MDE | jerk MDE | collision MDE |",
        "|---|---|---:|---:|---:|---|---:|---:|---:|",
    ]
    for _, row in summary.iterrows():
        ci = (
            f"{row['phantom_ci_low']:.2f}–{row['phantom_ci_high']:.2f}"
            if np.isfinite(row["phantom_ci_low"])
            else "—"
        )
        lines.append(
            f"| {row.get('driver', '')} | {row.get('scenario', '')} | "
            f"{row['n_runs']} | {row['phantom_mean_per_km']:.2f} | "
            f"{row['phantom_std_per_km']:.2f} | {ci} | "
            f"{row['phantom_mde_n']:.2f} | {row['jerk_mde_n']:.2f} | "
            f"{row['collision_mde_n']:.3f} |"
        )
    with open(path, "w") as handle:
        handle.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("per_run_metrics")
    parser.add_argument("--out-dir", default="power")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--power", type=float, default=0.80)
    args = parser.parse_args()
    df = pd.read_csv(args.per_run_metrics)
    # Normalize the binomial endpoint if analyze_results wrote booleans.
    if "collided" in df.columns:
        df["collided"] = (
            df["collided"].astype(str).str.lower().isin(("true", "1"))
            .astype(int)
        )
    alpha_z = abs(float(np.round(
        __import__("scipy.stats", fromlist=["norm"]).norm.ppf(args.alpha / 2.0),
        12,
    )))
    power_z = float(
        __import__("scipy.stats", fromlist=["norm"]).norm.ppf(args.power)
    )
    summary = analyse(df, alpha_z, power_z)
    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = os.path.join(args.out_dir, "power_summary.csv")
    md_path = os.path.join(args.out_dir, "power_summary.md")
    summary.to_csv(csv_path, index=False)
    write_markdown(summary, md_path)
    print(summary.to_string(index=False))
    print(f"\nwrote {csv_path}\nwrote {md_path}")

    print("\nRuns per arm needed to detect a phantom-brake effect:")
    for _, row in summary.iterrows():
        std = row["phantom_std_per_km"]
        if not np.isfinite(std):
            continue
        print(f"  {row.get('driver')} S{row.get('scenario')}:")
        for delta in (1.0, 2.0, 5.0, 10.0):
            print(
                f"    delta={delta:4.1f}/km -> "
                f"{required_n(std, delta, alpha_z, power_z)} runs/arm"
            )


if __name__ == "__main__":
    main()
