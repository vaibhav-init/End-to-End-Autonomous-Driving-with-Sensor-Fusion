#!/usr/bin/env python3
"""
Human-readable comparison of MLP vs PCLA driver performance.

Reads per-tick GT CSVs and prints a clear table with actual numbers:
  - Collision? (yes/no)
  - Stopping distance (how far from NPC the ego stopped)
  - Reaction time (seconds from the scenario event to first hard brake)
  - Peak deceleration (strongest braking in m/s²)
  - Ego speed at closest approach
  - Time to stop (seconds from the scenario event to ego speed < 1 km/h)

Usage:
  # Compare both drivers
  python compare_drivers.py --runs mlp=results_mlp pcla=results_pcla

  # Just one driver
  python compare_drivers.py --runs mlp=results_mlp
"""

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metrics import longitudinal_cost_metrics  # noqa: E402

FPS = 20
OBSTACLE_PRESENT_MAX_M = 100.0
BRAKE_THRESHOLD = 0.3  # brake > this = "reacting"


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


def analyze_single_run(csv_path):
    """Extract clear metrics from one scenario run CSV."""
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        return {"error": str(e)}

    if df.empty or "scenario_id" not in df.columns:
        return {"error": "empty or missing columns"}

    scenario = int(df["scenario_id"].iloc[0])
    fog = int(df["fog_density"].iloc[0])
    seed = int(df["seed"].iloc[0])

    # Did it collide?
    collided = bool((df["collision_occurred"] == 1).any())

    # Only look at steps where obstacle is actually present
    present = df["gt_distance_to_npc_m"] < OBSTACLE_PRESENT_MAX_M
    event_idx = critical_event_index(df, present)
    evaluation_present = present.copy()
    if event_idx is not None:
        evaluation_present &= df.index >= event_idx
    dist_present = df.loc[evaluation_present, "gt_distance_to_npc_m"]

    # Stopping distance = minimum distance to NPC
    if not dist_present.empty:
        stopping_dist = float(dist_present.min())
    else:
        stopping_dist = float("nan")

    # Ego speed at closest approach
    if not dist_present.empty:
        closest_idx = dist_present.idxmin()
        speed_at_closest = float(df.loc[closest_idx, "gt_ego_speed_kmh"])
    else:
        speed_at_closest = float("nan")

    # Collision speed (if collided)
    if collided:
        collision_rows = df[df["collision_occurred"] == 1]
        collision_speed = float(collision_rows.iloc[0]["gt_ego_speed_kmh"])
    else:
        collision_speed = float("nan")

    # Reaction time: explicit scenario event → first hard brake.
    reaction_s = float("nan")
    if event_idx is not None:
        after_event = df.loc[event_idx:]
        braked = after_event[after_event["brake"] > BRAKE_THRESHOLD]
        if not braked.empty:
            reaction_steps = braked.index[0] - event_idx
            reaction_s = float(reaction_steps / FPS)

    pre_event_brake_fraction = float("nan")
    if event_idx is not None:
        pre_event = df.loc[df.index < event_idx]
        if not pre_event.empty:
            pre_event_brake_fraction = float(
                (pre_event["brake"] > BRAKE_THRESHOLD).mean()
            )

    # Peak deceleration (most negative acceleration → reported positive)
    peak_decel = float(max(0.0, -df["ego_accel_mps2"].min()))

    # Time to stop: critical event → ego speed < 1 km/h
    time_to_stop_s = float("nan")
    if event_idx is not None:
        after_event = df.loc[event_idx:]
        stopped = after_event[after_event["gt_ego_speed_kmh"] < 1.0]
        if not stopped.empty:
            stop_steps = stopped.index[0] - event_idx
            time_to_stop_s = float(stop_steps / FPS)

    # Min TTC
    ttc = (
        df.loc[evaluation_present, "time_to_collision_s"]
        if evaluation_present.any()
        else pd.Series(dtype=float)
    )
    ttc = ttc[(ttc > 0) & (ttc < 900)]
    min_ttc = float(ttc.min()) if not ttc.empty else float("nan")

    cost = longitudinal_cost_metrics(df, fps=FPS)
    ghost_selected = "—"
    if "radar_selected_source" in df.columns:
        ghost_selected = round(
            100.0 * float((df["radar_selected_source"].astype(str) == "ghost").mean()),
            1,
        )

    return {
        "scenario": scenario,
        "fog": fog,
        "seed": seed,
        "collided": collided,
        "phantom_events": cost["phantom_brake_events"],
        "phantom_per_km": (
            round(cost["phantom_brake_per_km"], 2)
            if not np.isnan(cost["phantom_brake_per_km"])
            else "—"
        ),
        "distance_km": round(cost["distance_km"], 3),
        "jerk_rms": (
            round(cost["jerk_rms_mps3"], 2)
            if not np.isnan(cost["jerk_rms_mps3"])
            else "—"
        ),
        "ghost_selected_pct": ghost_selected,
        "stopping_dist_m": round(stopping_dist, 1),
        "speed_at_closest_kmh": round(speed_at_closest, 1),
        "collision_speed_kmh": round(collision_speed, 1) if not np.isnan(collision_speed) else "—",
        "reaction_time_s": round(reaction_s, 2) if not np.isnan(reaction_s) else "—",
        "pre_event_brake_pct": (
            round(100.0 * pre_event_brake_fraction, 1)
            if not np.isnan(pre_event_brake_fraction)
            else "—"
        ),
        "peak_decel_mps2": round(peak_decel, 1),
        "time_to_stop_s": round(time_to_stop_s, 2) if not np.isnan(time_to_stop_s) else "—",
        "min_ttc_s": round(min_ttc, 2) if not np.isnan(min_ttc) else "—",
    }


def load_all_runs(runs):
    """Load all CSVs for all drivers."""
    all_results = []
    for label, root in runs.items():
        csvs = sorted(glob.glob(os.path.join(root, "**", "*.csv"), recursive=True))
        csvs = [c for c in csvs if os.path.basename(c) != "summary_all.csv"]
        if not csvs:
            print(f"  ⚠️  No CSVs found under {root} for '{label}'")
            continue
        for path in csvs:
            result = analyze_single_run(path)
            if "error" in result:
                print(f"  ⚠️  Skip {path}: {result['error']}")
                continue
            result["driver"] = label
            result["file"] = os.path.basename(path)
            all_results.append(result)
    return all_results


SCENARIO_NAMES = {
    1: "S1: Lead Vehicle Stopped",
    2: "S2: Lead Vehicle Decelerating",
    3: "S3: Lead Vehicle Constant Speed",
    4: "S4: Cut-In from Adjacent Lane",
    5: "S5: Ghost-Exposure Drive",
}

WEATHER_NAMES = {
    1: "Dark Night",
    2: "Dense Fog",
    3: "Clear Day",
    4: "Night+Fog+Rain",
    # Legacy names (in case old results are loaded)
    80: "Heavy Rain",
    50: "Moderate Rain",
    20: "Light Rain",
    0: "Clear",
}


def print_results(results):
    """Print a human-readable comparison table."""
    if not results:
        print("  No results to display.")
        return

    drivers = sorted(set(r["driver"] for r in results))
    scenarios = sorted(set(r["scenario"] for r in results))

    for sid in scenarios:
        scenario_name = SCENARIO_NAMES.get(sid, f"S{sid}")
        print(f"\n{'=' * 94}")
        print(f"  {scenario_name}")
        print(f"{'=' * 94}")

        for driver in drivers:
            runs = [r for r in results if r["scenario"] == sid and r["driver"] == driver]
            if not runs:
                continue

            print(f"\n  Driver: {driver.upper()}")
            print(f"  {'─' * 118}")
            print(f"  {'Weather':<14} {'Seed':<5} {'Collision':<9} {'Stop Dist':<10} "
                  f"{'React':<8} {'Pre-Brake':<10} {'PeakDecel':<10} {'T→Stop':<8} "
                  f"{'MinTTC':<8} {'Phantom/km':<11} {'Jerk':<7} {'Ghost%'}")
            print(f"  {'─' * 118}")

            for r in sorted(runs, key=lambda x: (-x["fog"], x["seed"])):
                weather = WEATHER_NAMES.get(r["fog"], f"fog={r['fog']}")
                collision_str = "💥 YES" if r["collided"] else "✅ No"
                pre_brake = (
                    f"{r['pre_event_brake_pct']:.1f}%"
                    if r["pre_event_brake_pct"] != "—"
                    else "—"
                )
                print(f"  {weather:<14} {r['seed']:<5d} {collision_str:<9} "
                      f"{r['stopping_dist_m']:>6.1f}m   "
                      f"{str(r['reaction_time_s']):>6}s "
                      f"{pre_brake:>9} "
                      f"{r['peak_decel_mps2']:>6.1f}m/s² "
                      f"{str(r['time_to_stop_s']):>6}s "
                      f"{str(r['min_ttc_s']):>6}s "
                      f"{str(r['phantom_per_km']):>10} "
                      f"{str(r['jerk_rms']):>6} "
                      f"{str(r['ghost_selected_pct']):>6}")

            # Averages
            avg_stop = np.nanmean([r["stopping_dist_m"] for r in runs])
            n_collisions = sum(1 for r in runs if r["collided"])
            react_vals = [r["reaction_time_s"] for r in runs
                         if r["reaction_time_s"] != "—"]
            avg_react = np.mean(react_vals) if react_vals else float("nan")
            avg_decel = np.mean([r["peak_decel_mps2"] for r in runs])
            pre_brake_vals = [
                r["pre_event_brake_pct"]
                for r in runs
                if r["pre_event_brake_pct"] != "—"
            ]
            avg_pre_brake = (
                np.mean(pre_brake_vals)
                if pre_brake_vals
                else float("nan")
            )

            avg_react_str = (
                f"{avg_react:.2f}s" if not np.isnan(avg_react) else "—"
            )
            avg_pre_str = (
                f"{avg_pre_brake:.1f}%"
                if not np.isnan(avg_pre_brake)
                else "—"
            )
            total_phantom = sum(r["phantom_events"] for r in runs)
            total_km = sum(r["distance_km"] for r in runs)
            phantom_per_km = (
                f"{total_phantom / total_km:.2f}" if total_km > 1e-6 else "—"
            )
            jerk_values = [r["jerk_rms"] for r in runs if r["jerk_rms"] != "—"]
            jerk_str = (
                f"{np.mean(jerk_values):.2f}±{np.std(jerk_values):.2f}"
                if jerk_values
                else "—"
            )
            print(f"  {'─' * 118}")
            print(
                f"  {'AVERAGE':<20} {n_collisions}/{len(runs)} hits  "
                f"{avg_stop:>6.1f}m   {avg_react_str:>7} "
                f"{avg_pre_str:>9} {avg_decel:>6.1f}m/s²   "
                f"phantom {total_phantom} events over {total_km:.2f} km "
                f"= {phantom_per_km}/km   jerk {jerk_str} m/s³"
            )

    # Cross-driver comparison
    if len(drivers) > 1:
        print(f"\n\n{'=' * 80}")
        print(f"  HEAD-TO-HEAD COMPARISON")
        print(f"{'=' * 80}")

        for sid in scenarios:
            scenario_name = SCENARIO_NAMES.get(sid, f"S{sid}")
            print(f"\n  {scenario_name}:")

            for driver in drivers:
                runs = [r for r in results
                        if r["scenario"] == sid and r["driver"] == driver]
                if not runs:
                    continue
                n_col = sum(1 for r in runs if r["collided"])
                avg_stop = np.nanmean([r["stopping_dist_m"] for r in runs])
                react_vals = [r["reaction_time_s"] for r in runs
                             if r["reaction_time_s"] != "—"]
                avg_react = np.mean(react_vals) if react_vals else float("nan")

                react_str = f"{avg_react:.2f}s" if not np.isnan(avg_react) else "—"
                print(f"    {driver.upper():<8}: "
                      f"collisions={n_col}/{len(runs)}  "
                      f"avg_stop_dist={avg_stop:.1f}m  "
                      f"avg_reaction={react_str}")


def main():
    parser = argparse.ArgumentParser(
        description="Human-readable driver comparison from GT logs")
    parser.add_argument("--runs", nargs="+", required=True,
                        help="label=dir pairs, e.g. mlp=results_mlp pcla=results_pcla")
    args = parser.parse_args()

    runs = parse_runs(args.runs)
    print("=" * 80)
    print("DRIVER PERFORMANCE ANALYSIS")
    print("=" * 80)
    for label, root in runs.items():
        print(f"  {label:<8} ← {root}")

    results = load_all_runs(runs)
    if not results:
        print("\n  No valid runs found. Check your paths.")
        sys.exit(1)

    print(f"\n  Loaded {len(results)} runs total")
    print_results(results)
    print(f"\n{'=' * 80}")


if __name__ == "__main__":
    main()
