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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metrics import peak_deceleration_mps2, longitudinal_cost_metrics  # noqa: E402

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


def critical_event_index(df, present):
    """Use explicit event markers, falling back to legacy GT heuristics."""
    if "critical_event" in df.columns:
        marked = pd.to_numeric(
            df["critical_event"], errors="coerce"
        ).fillna(0) > 0
        if marked.any():
            return marked.index[marked][0]
    critical = df[
        present
        & (df["gt_relative_velocity"] > 2.0)
        & (df["gt_distance_to_npc_m"] < 80.0)
    ]
    return critical.index[0] if not critical.empty else None


def per_run_metrics(df):
    """Longitudinal metrics for a single run (one CSV)."""
    collided = bool((df["collision_occurred"] == 1).any())

    present = df["gt_distance_to_npc_m"] < OBSTACLE_PRESENT_MAX_M
    event_idx = critical_event_index(df, present)
    evaluation_present = present.copy()
    if event_idx is not None:
        evaluation_present &= df.index >= event_idx
    dist_present = df.loc[evaluation_present, "gt_distance_to_npc_m"]
    min_dist = float(dist_present.min()) if not dist_present.empty else np.nan

    ttc = df.loc[evaluation_present, "time_to_collision_s"]
    ttc = ttc[(ttc > 0) & (ttc < 900)]
    min_ttc = float(ttc.min()) if not ttc.empty else np.nan

    # Strongest deceleration (most negative acceleration), reported as positive m/s^2
    peak_decel = peak_deceleration_mps2(df)

    # Reaction latency: explicit critical event -> first hard brake after it.
    reaction_s = np.nan
    if event_idx is not None:
        after = df.loc[event_idx:]
        braked = after[after["brake"] > BRAKE_REACTION_THRESHOLD]
        if not braked.empty:
            reaction_s = float((braked.index[0] - event_idx) / FPS)

    pre_event_brake_fraction = np.nan
    if event_idx is not None:
        pre_event = df.loc[df.index < event_idx]
        if not pre_event.empty:
            pre_event_brake_fraction = float(
                (pre_event["brake"] > BRAKE_REACTION_THRESHOLD).mean()
            )

    ghost_selected_fraction = np.nan
    if "radar_selected_source" in df.columns:
        source = df["radar_selected_source"].astype(str)
        ghost_selected_fraction = float((source == "ghost").mean())

    return {
        "collided": collided,
        "min_dist_m": min_dist,
        "min_ttc_s": min_ttc,
        "peak_decel_mps2": peak_decel,
        "reaction_s": reaction_s,
        "pre_event_brake_fraction": pre_event_brake_fraction,
        "ghost_selected_fraction": ghost_selected_fraction,
        # The false-positive side of the trade-off; see metrics.py.
        **longitudinal_cost_metrics(df, fps=FPS),
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
            target_speed = (
                pd.to_numeric(
                    df["test_target_speed_kmh"], errors="coerce"
                ).dropna()
                if "test_target_speed_kmh" in df.columns
                else pd.Series(dtype=float)
            )
            event_distance = (
                pd.to_numeric(
                    df["test_event_distance_m"], errors="coerce"
                ).dropna()
                if "test_event_distance_m" in df.columns
                else pd.Series(dtype=float)
            )
            records.append({
                "driver": label,
                "scenario": int(df["scenario_id"].iloc[0]),
                "fog": int(df["fog_density"].iloc[0]),
                "seed": int(df["seed"].iloc[0]),
                "file": os.path.basename(path),
                "test_target_speed_kmh": (
                    float(target_speed.iloc[0])
                    if not target_speed.empty
                    else np.nan
                ),
                "test_event_distance_m": (
                    float(event_distance.iloc[0])
                    if not event_distance.empty
                    else np.nan
                ),
                **metrics,
            })
    return pd.DataFrame.from_records(records)


def validate_comparison_matrix(df):
    """Reject duplicate, unpaired, or mixed-profile comparison inputs."""
    key_cols = ["scenario", "fog", "seed"]
    duplicates = df.duplicated(["driver", *key_cols], keep=False)
    if duplicates.any():
        rows = df.loc[duplicates, ["driver", *key_cols, "file"]]
        raise RuntimeError(
            "Duplicate result keys found; use clean output roots:\n"
            + rows.to_string(index=False)
        )

    drivers = sorted(df["driver"].unique())
    if len(drivers) > 1:
        reference = set(
            map(tuple, df[df["driver"] == drivers[0]][key_cols].to_numpy())
        )
        for driver in drivers[1:]:
            candidate = set(
                map(tuple, df[df["driver"] == driver][key_cols].to_numpy())
            )
            if candidate != reference:
                raise RuntimeError(
                    f"Run matrix for '{driver}' does not match "
                    f"'{drivers[0]}'. Missing/extra keys: "
                    f"{sorted(reference ^ candidate)}"
                )

    for scenario, scenario_df in df.groupby("scenario"):
        profiles = scenario_df[
            ["test_target_speed_kmh", "test_event_distance_m"]
        ].dropna(how="all").drop_duplicates()
        if len(profiles) > 1:
            raise RuntimeError(
                f"S{scenario} contains mixed stress profiles:\n"
                + profiles.to_string(index=False)
            )


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
        mean_pre_event_brake_fraction=("pre_event_brake_fraction", "mean"),
        phantom_brake_events=("phantom_brake_events", "sum"),
        distance_km=("distance_km", "sum"),
        mean_jerk_rms_mps3=("jerk_rms_mps3", "mean"),
        std_jerk_rms_mps3=("jerk_rms_mps3", "std"),
        mean_ghost_selected_fraction=("ghost_selected_fraction", "mean"),
    ).reset_index()
    summary["collision_rate"] = summary["n_collisions"] / summary["n_runs"]
    summary["phantom_brake_per_km"] = summary["phantom_brake_events"] / summary[
        "distance_km"
    ].replace(0.0, np.nan)
    cols = ["driver", "scenario", "fog", "n_runs", "n_collisions",
            "collision_rate", "mean_min_dist_m", "mean_min_ttc_s",
            "mean_peak_decel", "mean_reaction_s",
            "mean_pre_event_brake_fraction",
            "phantom_brake_events", "distance_km", "phantom_brake_per_km",
            "mean_jerk_rms_mps3", "std_jerk_rms_mps3",
            "mean_ghost_selected_fraction"]
    return summary[cols].sort_values(["scenario", "fog", "driver"])


def _cdf(ax, values, label, point_labels=None, unit="", driver_idx=0):
    """Plot an empirical CDF and label each observation with its run ID.

    Args:
        point_labels: list of strings parallel to *values* (before NaN removal
            and sorting), such as ``M01`` or ``P04``.
        unit: suffix like 'm' or 's' appended after the numeric value.
        driver_idx: used to place different drivers' labels on opposite sides
            of their markers.
    """
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

    # Draw a true step ECDF. Smooth interpolation creates values that were
    # never observed and makes a small experiment look more precise than it is.
    x_range = vals[-1] - vals[0] if vals.size > 1 else 1.0
    margin = max(x_range * 0.15, 0.1)
    x_step = np.concatenate([[vals[0] - margin], vals,
                             [vals[-1] + margin]])
    y_step = np.concatenate([[0.0], y_cdf, [1.0]])
    line = ax.step(
        x_step,
        y_step,
        where="post",
        linewidth=1.4,
        linestyle="--",
        alpha=0.65,
        label=f"{label.upper()} (n={vals.size})",
    )

    color = line[0].get_color()

    # Add a monotonic smooth guide over the empirical steps. The dashed ECDF,
    # markers, and run IDs remain the statistical observations; this solid
    # curve is only a visual aid.
    try:
        from scipy.interpolate import PchipInterpolator

        unique_x, counts = np.unique(vals, return_counts=True)
        if unique_x.size >= 2:
            unique_y = np.cumsum(counts) / vals.size
            smooth_x_knots = np.concatenate([
                [unique_x[0] - margin],
                unique_x,
                [unique_x[-1] + margin],
            ])
            smooth_y_knots = np.concatenate([[0.0], unique_y, [1.0]])
            interpolator = PchipInterpolator(
                smooth_x_knots, smooth_y_knots
            )
            smooth_x = np.linspace(
                smooth_x_knots[0], smooth_x_knots[-1], 300
            )
            smooth_y = np.clip(interpolator(smooth_x), 0.0, 1.0)
            ax.plot(
                smooth_x,
                smooth_y,
                color=color,
                linewidth=2.4,
                alpha=0.9,
                label="_nolegend_",
                zorder=3,
            )
    except ImportError:
        # The empirical graph remains complete when SciPy is unavailable.
        pass

    # Plot actual data points as markers
    ax.scatter(vals, y_cdf, color=color, s=40, zorder=5, edgecolors="white",
               linewidths=0.8)

    # Labels are deliberately short. Their complete descriptions appear in
    # the run-key tables below the plots, where text cannot overlap.
    base_y_sign = 1 if driver_idx % 2 == 0 else -1
    va = "bottom" if base_y_sign > 0 else "top"
    for i, (xv, yv) in enumerate(zip(vals, y_cdf)):
        plbl = plabels[i]
        txt = plbl if plbl else f"{xv:.1f}{unit}"
        y_offset = base_y_sign * 7
        ax.annotate(txt,
                    xy=(xv, yv),
                    textcoords="offset points", xytext=(4, y_offset),
                    fontsize=6.2, color=color, fontweight="bold",
                    va=va, ha="left", clip_on=False,
                    bbox={"boxstyle": "round,pad=0.12", "facecolor": "white",
                          "edgecolor": "none", "alpha": 0.72})


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

    # Human-readable names used in titles and run-key tables.
    weather_names = {
        1: "Dark Night", 2: "Dense Fog", 3: "Clear Day",
        4: "Night + Fog + Rain",
        80: "Heavy Rain", 50: "Moderate Rain", 20: "Light Rain", 0: "Clear",
    }
    scenario_names = {
        1: "Lead Vehicle Stopped (NHTSA Scenario #25)",
        2: "Lead Vehicle Decelerating (NHTSA Scenario #4)",
        3: "Lead Vehicle Moving at Lower Constant Speed "
           "(NHTSA Scenario #12)",
        4: "Vehicle Cut-In from Adjacent Lane",
    }

    for sid in scenarios:
        sdf = df[df["scenario"] == sid].copy()
        scenario_name = scenario_names.get(sid, f"Scenario S{sid}")

        # The top row contains the two CDFs. The lower row contains one
        # complete run-key table per driver so long labels never cover data.
        fig = plt.figure(figsize=(max(18, 9 * len(drivers)), 14))
        grid = fig.add_gridspec(
            2, 2, height_ratios=[2.4, 2.0], hspace=0.28, wspace=0.12
        )
        axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])]
        table_grid = grid[1, :].subgridspec(
            1, max(1, len(drivers)), wspace=0.12
        )
        table_axes = [
            fig.add_subplot(table_grid[0, i]) for i in range(len(drivers))
        ]

        for didx, label in enumerate(drivers):
            ddf = sdf[sdf["driver"] == label].sort_values(
                ["fog", "seed"]
            ).copy()
            prefix = {"mlp": "M", "pcla": "P"}.get(
                str(label).lower(), str(label)[:2].upper()
            )
            ddf["run_id"] = [
                f"{prefix}{i:02d}" for i in range(1, len(ddf) + 1)
            ]

            _cdf(axes[0], ddf["min_dist_m"].tolist(), label,
                 point_labels=ddf["run_id"].tolist(), unit="m",
                 driver_idx=didx)
            _cdf(axes[1], ddf["min_ttc_s"].tolist(), label,
                 point_labels=ddf["run_id"].tolist(), unit="s",
                 driver_idx=didx)

            table_rows = []
            collision_rows = []
            for row_idx, (_, row) in enumerate(ddf.iterrows(), start=1):
                collided = bool(row["collided"])
                if collided:
                    collision_rows.append(row_idx)
                table_rows.append([
                    row["run_id"],
                    weather_names.get(int(row["fog"]), f"Fog code {row['fog']}"),
                    str(int(row["seed"])),
                    "COLLISION" if collided else "Safe",
                    f"{row['min_dist_m']:.2f}",
                    f"{row['min_ttc_s']:.2f}",
                ])

            table_ax = table_axes[didx]
            table_ax.axis("off")
            table_ax.set_title(
                f"{label.upper()} run key — Scenario S{sid}: {scenario_name}",
                fontsize=10, fontweight="bold", pad=8,
            )
            run_table = table_ax.table(
                cellText=table_rows,
                colLabels=[
                    "Run ID", "Complete weather condition", "Seed", "Outcome",
                    "Min distance (m)", "Min TTC (s)",
                ],
                cellLoc="center",
                colLoc="center",
                bbox=[0.0, 0.0, 1.0, 0.95],
                colWidths=[0.09, 0.30, 0.09, 0.14, 0.19, 0.16],
            )
            run_table.auto_set_font_size(False)
            run_table.set_fontsize(7.2)
            for col in range(6):
                run_table[(0, col)].set_facecolor("#d9e5f2")
                run_table[(0, col)].set_text_props(weight="bold")
            for row_idx in collision_rows:
                for col in range(6):
                    run_table[(row_idx, col)].set_facecolor("#f8d7da")
                run_table[(row_idx, 3)].set_text_props(
                    color="#9c1c24", weight="bold"
                )

        axes[0].set(title="Closest-approach ECDF — larger distance is safer",
                    xlabel="min distance to NPC (m)", ylabel="P(X ≤ x)")
        axes[1].set(title="Minimum-TTC ECDF — larger TTC is safer",
                    xlabel="min time-to-collision (s)", ylabel="P(X ≤ x)")
        for ax in axes:
            ax.grid(True, alpha=0.3)
            ax.legend()
            ax.set_ylim(-0.04, 1.09)
        fig.suptitle(
            f"Scenario S{sid}: {scenario_name}\n"
            "Markers and dashed steps are empirical observations; "
            "solid curves are smooth visual guides",
            fontsize=15, fontweight="bold", y=0.995,
        )
        path = os.path.join(out_dir, f"cdf_s{sid}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
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
                point_label = (
                    f"{label.upper()} — Scenario S{sid}: "
                    f"{rate_pct:.0f}% "
                    f"({int(row['n_collisions'])}/{int(row['n_runs'])})"
                )

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

    if len(scenarios) == 1:
        sid = scenarios[0]
        collision_title = (
            f"Collision rate — Scenario S{sid}: "
            f"{scenario_names.get(sid, f'Scenario S{sid}')}"
        )
    else:
        collision_title = "Collision rate by complete scenario and weather condition"
    ax.set(title=collision_title,
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
    validate_comparison_matrix(df)

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
