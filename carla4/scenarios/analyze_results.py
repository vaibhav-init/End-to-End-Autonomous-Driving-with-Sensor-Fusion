#!/usr/bin/env python3
"""
Compare drivers on the NHTSA scenarios from their ground-truth logs.

Reads the per-tick CSVs written by GroundTruthLogger under each driver's result
root, computes LONGITUDINAL metrics per run, and emits:
  - <out>/summary.csv          : collision rate + metric means per (driver, scenario, fog)
  - <out>/cdf_s<id>.png        : CDF of closest-approach distance and min TTC per scenario
  - <out>/collision_rate.png   : collision rate by fog, per driver/scenario
  - a summary table printed to stdout

Usage:
  python analyze_results.py --runs pcla=results_pcla mlp=results_mlp --out comparison

Each --runs entry is `label=dir`; `dir` is the --output-root used by run_all.py
(it contains results_s1/.. with files like s1_fog0_seed42.csv).
"""

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

# Sim rate used by the scenarios (carla4/scenarios/config.py FPS); used to turn
# reaction step counts into seconds.
FPS = 20
OBSTACLE_PRESENT_MAX_M = 100.0   # GT distance below this => an obstacle is in front
BRAKE_REACTION_THRESHOLD = 0.3   # brake command counted as "reacting"


def parse_runs(run_args):
    runs = {}
    for item in run_args:
        if "=" not in item:
            raise SystemExit(f"--runs entry must be label=dir, got: {item}")
        label, path = item.split("=", 1)
        runs[label.strip()] = path.strip()
    return runs


def per_run_metrics(df):
    """Longitudinal metrics for a single run (one CSV)."""
    collided = bool((df["collision_occurred"] == 1).any())

    present = df["gt_distance_to_npc_m"] < OBSTACLE_PRESENT_MAX_M
    dist_present = df.loc[present, "gt_distance_to_npc_m"]
    min_dist = float(dist_present.min()) if not dist_present.empty else np.nan

    ttc = df.loc[present, "time_to_collision_s"]
    ttc = ttc[(ttc > 0) & (ttc < 900)]
    min_ttc = float(ttc.min()) if not ttc.empty else np.nan

    # Strongest deceleration (most negative acceleration), reported as positive m/s^2
    peak_decel = float(max(0.0, -df["ego_accel_mps2"].min()))

    # Reaction latency: first obstacle-present step -> first hard brake after it
    reaction_s = np.nan
    if present.any():
        first_present = df.index[present][0]
        after = df.loc[first_present:]
        braked = after[after["brake"] > BRAKE_REACTION_THRESHOLD]
        if not braked.empty:
            reaction_s = float((braked.index[0] - first_present) / FPS)

    return {
        "collided": collided,
        "min_dist_m": min_dist,
        "min_ttc_s": min_ttc,
        "peak_decel_mps2": peak_decel,
        "reaction_s": reaction_s,
    }


def load_runs(runs):
    """Return a long DataFrame: one row per (driver, scenario, fog, seed) run."""
    records = []
    for label, root in runs.items():
        csvs = sorted(glob.glob(os.path.join(root, "**", "*.csv"), recursive=True))
        # Ignore aggregate summaries written by run_all.py
        csvs = [c for c in csvs if os.path.basename(c) != "summary_all.csv"]
        if not csvs:
            print(f"  [warn] no per-run CSVs found under {root} for '{label}'")
        for path in csvs:
            try:
                df = pd.read_csv(path)
            except Exception as exc:  # noqa: BLE001
                print(f"  [warn] skip {path}: {exc}")
                continue
            if df.empty or "scenario_id" not in df.columns:
                print(f"  [warn] skip {path}: empty or missing columns")
                continue
            metrics = per_run_metrics(df)
            records.append({
                "driver": label,
                "scenario": int(df["scenario_id"].iloc[0]),
                "fog": int(df["fog_density"].iloc[0]),
                "seed": int(df["seed"].iloc[0]),
                "file": os.path.basename(path),
                **metrics,
            })
    return pd.DataFrame.from_records(records)


def build_summary(df):
    """Collision rate + metric means per (driver, scenario, fog)."""
    grp = df.groupby(["driver", "scenario", "fog"])
    summary = grp.agg(
        n_runs=("collided", "size"),
        n_collisions=("collided", "sum"),
        mean_min_dist_m=("min_dist_m", "mean"),
        mean_min_ttc_s=("min_ttc_s", "mean"),
        mean_peak_decel=("peak_decel_mps2", "mean"),
        mean_reaction_s=("reaction_s", "mean"),
    ).reset_index()
    summary["collision_rate"] = summary["n_collisions"] / summary["n_runs"]
    cols = ["driver", "scenario", "fog", "n_runs", "n_collisions",
            "collision_rate", "mean_min_dist_m", "mean_min_ttc_s",
            "mean_peak_decel", "mean_reaction_s"]
    return summary[cols].sort_values(["scenario", "fog", "driver"])


def _cdf(ax, values, label, point_labels=None, unit="", driver_idx=0):
    """Plot a smooth CDF curve, labelling each point with what it represents.

    Uses monotonic interpolation (PCHIP) to draw a smooth S-curve through the
    empirical CDF points, with markers at the actual data points.

    Args:
        point_labels: list of strings parallel to *values* (before NaN removal
            and sorting) describing each point, e.g. weather condition names.
        unit: suffix like 'm' or 's' appended after the numeric value.
        driver_idx: index of this driver (0, 1, …) — used to alternate label
            placement above vs below the curve so different drivers don't overlap.
    """
    from scipy.interpolate import PchipInterpolator

    # Pair values with labels, drop NaNs, sort by value
    if point_labels is not None:
        pairs = [(v, lbl) for v, lbl in zip(values, point_labels)
                 if not np.isnan(v)]
    else:
        pairs = [(v, None) for v in values if not np.isnan(v)]
    if not pairs:
        return
    pairs.sort(key=lambda p: p[0])
    vals = np.array([p[0] for p in pairs])
    plabels = [p[1] for p in pairs]

    # Empirical CDF: (x_i, i/n)
    y_cdf = np.arange(1, vals.size + 1) / vals.size

    # Build knot points for smooth interpolation:
    #   start at (x_min - margin, 0), pass through data, end at (x_max + margin, 1)
    x_range = vals[-1] - vals[0] if vals.size > 1 else 1.0
    margin = max(x_range * 0.15, 0.1)
    x_knots = np.concatenate([[vals[0] - margin], vals, [vals[-1] + margin]])
    y_knots = np.concatenate([[0.0], y_cdf, [1.0]])

    # Smooth interpolation (monotonic so CDF never decreases)
    if x_knots.size >= 2:
        # Need unique x values for interpolation
        # If duplicates exist, nudge them slightly
        for j in range(1, len(x_knots)):
            if x_knots[j] <= x_knots[j - 1]:
                x_knots[j] = x_knots[j - 1] + 1e-6

        interp = PchipInterpolator(x_knots, y_knots)
        x_smooth = np.linspace(x_knots[0], x_knots[-1], 200)
        y_smooth = np.clip(interp(x_smooth), 0.0, 1.0)

        # Plot smooth curve
        line = ax.plot(x_smooth, y_smooth, linewidth=2,
                       label=f"{label} (n={vals.size})")
    else:
        line = ax.plot(vals, y_cdf, linewidth=2,
                       label=f"{label} (n={vals.size})")

    color = line[0].get_color()

    # Plot actual data points as markers
    ax.scatter(vals, y_cdf, color=color, s=40, zorder=5, edgecolors="white",
               linewidths=0.8)

    # Alternate label placement: even drivers above, odd drivers below.
    base_y_sign = 1 if driver_idx % 2 == 0 else -1
    va = "bottom" if base_y_sign > 0 else "top"

    for i, (xv, yv) in enumerate(zip(vals, y_cdf)):
        plbl = plabels[i]
        if plbl:
            txt = f"{plbl}: {xv:.1f}{unit}"
        else:
            txt = f"{xv:.1f}{unit}"
        # Stagger: alternate between two vertical offsets per point index
        y_offset = base_y_sign * (10 + 12 * (i % 2))
        ax.annotate(txt,
                    xy=(xv, yv),
                    textcoords="offset points", xytext=(6, y_offset),
                    fontsize=6.5, color=color, fontweight="bold",
                    va=va, ha="left")


def plot_cdfs(df, out_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001
        print(f"  [warn] matplotlib unavailable, skipping plots: {exc}")
        return

    drivers = sorted(df["driver"].unique())
    scenarios = sorted(df["scenario"].unique())

    # Weather fog-code → human-readable name (matches compare_drivers.py)
    weather_names = {
        1: "Dark Night", 2: "Dense Fog", 3: "Clear Day", 4: "Night+Fog+Rain",
        80: "Heavy Rain", 50: "Moderate Rain", 20: "Light Rain", 0: "Clear",
    }

    for sid in scenarios:
        sdf = df[df["scenario"] == sid]
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        for didx, label in enumerate(drivers):
            ddf = sdf[sdf["driver"] == label]
            # Build per-point weather labels aligned with the value lists
            wlabels = [weather_names.get(int(f), f"fog={f}") for f in ddf["fog"]]
            _cdf(axes[0], ddf["min_dist_m"].tolist(), label,
                 point_labels=wlabels, unit="m", driver_idx=didx)
            _cdf(axes[1], ddf["min_ttc_s"].tolist(), label,
                 point_labels=wlabels, unit="s", driver_idx=didx)
        axes[0].set(title=f"S{sid} — closest approach CDF",
                    xlabel="min distance to NPC (m)", ylabel="P(X ≤ x)")
        axes[1].set(title=f"S{sid} — min TTC CDF",
                    xlabel="min time-to-collision (s)", ylabel="P(X ≤ x)")
        for ax in axes:
            ax.grid(True, alpha=0.3)
            ax.legend()
        fig.tight_layout()
        path = os.path.join(out_dir, f"cdf_s{sid}.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  wrote {path}")

    # Collision-rate by fog, one line per (driver, scenario)
    summary = build_summary(df)
    fig, ax = plt.subplots(figsize=(12, 7))

    # Track annotations at each (fog, rate) coordinate to stagger overlaps
    coord_count = {}  # (fog, rate) -> number of labels already placed there
    series_idx = 0
    for label in drivers:
        for sid in scenarios:
            sub = summary[(summary["driver"] == label) & (summary["scenario"] == sid)]
            sub = sub.sort_values("fog")
            if sub.empty:
                continue
            line = ax.plot(sub["fog"], sub["collision_rate"], marker="o",
                           label=f"{label} S{sid}")
            color = line[0].get_color()

            # Label each point with driver, scenario, and rate
            for _, row in sub.iterrows():
                fog_val = int(row["fog"])
                rate_pct = row["collision_rate"] * 100
                # Compact label: "MLP S4: 25%"
                point_label = f"{label.upper()} S{sid}: {rate_pct:.0f}%"

                # Stagger vertically when multiple labels land on same coords
                key = (fog_val, round(row["collision_rate"], 3))
                n_prev = coord_count.get(key, 0)
                coord_count[key] = n_prev + 1

                # Spread labels: alternate above/below, increasing offset
                direction = 1 if n_prev % 2 == 0 else -1
                y_offset = direction * (8 + 12 * (n_prev // 2))
                va = "bottom" if direction > 0 else "top"

                ax.annotate(point_label,
                            xy=(row["fog"], row["collision_rate"]),
                            textcoords="offset points",
                            xytext=(6, y_offset),
                            fontsize=6.5, color=color, fontweight="bold",
                            ha="left", va=va)
            series_idx += 1

    ax.set(title="Collision rate by weather condition",
           xlabel="Weather preset", ylabel="collision rate",
           ylim=(-0.05, 1.05))
    # Use weather names on x-axis instead of raw fog codes
    fog_vals = sorted(summary["fog"].unique())
    ax.set_xticks(fog_vals)
    ax.set_xticklabels([weather_names.get(int(f), f"fog={f}") for f in fog_vals],
                       rotation=30, ha="right", fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = os.path.join(out_dir, "collision_rate.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  wrote {path}")


def main():
    parser = argparse.ArgumentParser(description="Compare drivers from GT logs")
    parser.add_argument("--runs", nargs="+", required=True,
                        help="label=dir pairs, e.g. pcla=results_pcla mlp=results_mlp")
    parser.add_argument("--out", default="comparison", help="output directory")
    args = parser.parse_args()

    runs = parse_runs(args.runs)
    os.makedirs(args.out, exist_ok=True)

    print("=" * 72)
    print("DRIVER COMPARISON")
    print("=" * 72)
    for label, root in runs.items():
        print(f"  {label:8s} <- {root}")

    df = load_runs(runs)
    if df.empty:
        print("  No runs loaded — nothing to compare.")
        sys.exit(1)

    raw_path = os.path.join(args.out, "per_run_metrics.csv")
    df.to_csv(raw_path, index=False)
    print(f"  wrote {raw_path} ({len(df)} runs)")

    summary = build_summary(df)
    summary_path = os.path.join(args.out, "summary.csv")
    summary.to_csv(summary_path, index=False)
    print(f"  wrote {summary_path}")

    print("\n" + "=" * 72)
    print("SUMMARY (collision rate + longitudinal metric means)")
    print("=" * 72)
    with pd.option_context("display.width", 200,
                           "display.max_columns", None,
                           "display.float_format", lambda v: f"{v:.2f}"):
        print(summary.to_string(index=False))

    print()
    plot_cdfs(df, args.out)
    print("=" * 72)


if __name__ == "__main__":
    main()
