#!/usr/bin/env python3
"""
Master Runner — All Scenarios
==============================

Runs the NHTSA-aligned scenarios (S1 stopped lead, S2 decelerating lead, S4
cut-in) and the S5 ghost-exposure drive across seeds, weather presets and
towns, each `(town, scenario, fog, seed)` in a fresh subprocess.

Usage:
  # Start CARLA first, then from carla4/scenarios:
  python run_all.py --driver mlp --model-dir ../model_x --radar-backend realistic

  # One scenario, one seed
  python run_all.py --scenarios 1 --seeds 42

  # The paired ghost comparison: same model, same seeds, ghosts off / on
  python run_all.py ... --radar-multipath-mode off      --output-root results_A_clean
  python run_all.py ... --radar-multipath-mode geometry --output-root results_A_ghosts
  python run_all.py ... --radar-multipath-mode geometry --radar-ghost-oracle \
                        --output-root results_B_oracle

Output:
  <output-root>/results_sN/*.csv           GT logs (+ .detections.npz sidecars)
  <output-root>/<town>/results_sN/*.csv    when more than one town is given
  <output-root>/summary_all.csv            one row per run
"""

import argparse
import csv
import os
import subprocess
import sys
import time

_CARLA4_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _CARLA4_DIR not in sys.path:
    sys.path.insert(0, _CARLA4_DIR)

from config import (
    DEFAULT_TOWN,
    FOG_LADDER,
    RANDOM_SEEDS,
    S1_OBSTACLE_DISTANCE,
    S1_SPAWN_SPEED_KMH,
    S2_NPC_INITIAL_GAP,
    S2_NPC_SPEED_KMH,
    SCENARIO_DURATION_S,
)
from drivers import DRIVER_NAMES
from radar import add_radar_arguments


SCENARIO_MODULES = {
    1: ("s1_lead_vehicle_stopped", "results_s1"),
    2: ("s2_lead_vehicle_decelerating", "results_s2"),
    # 3: ("s3_lead_vehicle_constant_speed", "results_s3"),  # S3 disabled for now
    4: ("s4_cut_in", "results_s4"),
    5: ("s5_ghost_exposure", "results_s5"),
}
DEFAULT_SCENARIOS = [1, 2, 4, 5]


def _radar_passthrough(args):
    """Command-line radar flags forwarded verbatim to each scenario process."""

    cmd = ["--radar-backend", args.radar_backend]
    if args.radar_profile:
        cmd += ["--radar-profile", args.radar_profile]
    if args.radar_config:
        cmd += ["--radar-config", args.radar_config]
    if args.radar_seed is not None:
        cmd += ["--radar-seed", str(args.radar_seed)]
    if args.radar_ghost_detector:
        cmd += ["--radar-ghost-detector", args.radar_ghost_detector]
    if args.radar_ghost_threshold is not None:
        cmd += ["--radar-ghost-threshold", str(args.radar_ghost_threshold)]
    if args.radar_ghost_device:
        cmd += ["--radar-ghost-device", args.radar_ghost_device]
    if args.radar_ghost_oracle:
        cmd += ["--radar-ghost-oracle"]
    if args.radar_multipath_mode:
        cmd += ["--radar-multipath-mode", args.radar_multipath_mode]
    if args.radar_ghost_rate_scale is not None:
        cmd += ["--radar-ghost-rate-scale", str(args.radar_ghost_rate_scale)]
    if args.radar_ghost_snr_offset_db is not None:
        cmd += ["--radar-ghost-snr-offset-db", str(args.radar_ghost_snr_offset_db)]
    return cmd


def _scenario_passthrough(args, sid):
    if sid == 1:
        return [
            "--target-speed-kmh", str(args.s1_target_speed_kmh),
            "--obstacle-distance-m", str(args.s1_obstacle_distance_m),
            "--stage-stable-s", str(args.s1_stage_stable_s),
            "--stage-speed-tolerance-kmh", str(args.s1_stage_speed_tolerance_kmh),
        ]
    if sid == 2:
        cmd = [
            "--target-speed-kmh", str(args.s2_target_speed_kmh),
            "--initial-gap", str(args.s2_gap_m),
            "--stage-gap", str(args.s2_gap_m),
            "--handover-settle-s", str(args.s2_handover_settle_s),
        ]
        if args.s2_no_stage:
            cmd.append("--no-stage-approach")
        return cmd
    if sid == 5:
        return ["--duration-s", str(args.s5_duration_s)]
    return []


def _parse_result(stdout):
    collision = "?"
    min_dist = "?"
    for line in stdout.splitlines():
        lowered = line.lower()
        if "collision=" in lowered and "min_dist" in lowered:
            parts = line.split("collision=")
            if len(parts) > 1:
                collision = parts[1].split()[0].strip()
            parts2 = line.split("min_dist=")
            if len(parts2) > 1:
                min_dist = parts2[1].split()[0].strip()
        if "❌" in line and "failed" in lowered:
            collision = "error"
    return collision, min_dist


def main():
    parser = argparse.ArgumentParser(description="Run all NHTSA scenarios")
    parser.add_argument("--scenarios", type=int, nargs="+",
                        default=DEFAULT_SCENARIOS,
                        help="Scenarios to run (1, 2, 4, 5; 3 is disabled)")
    parser.add_argument("--fog", type=int, nargs="+", default=FOG_LADDER,
                        help="Weather presets to test (default: clear day only)")
    parser.add_argument("--seeds", type=int, nargs="+", default=RANDOM_SEEDS,
                        help="Random seeds; each also seeds the radar")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--towns", nargs="+", default=[DEFAULT_TOWN],
                        help=(
                            "towns to run, each in its own results subfolder when "
                            "more than one is given. Any town must have multi-lane "
                            "straight road for the highway spawner (Town04 does)"
                        ))
    parser.add_argument(
        "--safety-rules",
        action="store_true",
        help=(
            "re-enable the hardcoded emergency-brake overrides in the mlp "
            "driver (ablation arm). Off by default so the model decides."
        ),
    )
    parser.add_argument("--driver", choices=list(DRIVER_NAMES), default="mlp",
                        help="Longitudinal control source for every run")
    parser.add_argument("--model-dir", default="../model_throttle_brake",
                        help="model directory (for --driver mlp or transformer)")
    parser.add_argument("--pcla-agent", default="tfv6_visiononly",
                        help="PCLA agent name (for --driver pcla)")
    add_radar_arguments(parser)
    parser.add_argument("--output-root", default=None,
                        help="Root dir for results (default: results_<driver>[_<backend>])")
    parser.add_argument("--timeout", type=int, default=300,
                        help="Per-run subprocess timeout in seconds")
    parser.add_argument("--s1-target-speed-kmh", type=float, default=S1_SPAWN_SPEED_KMH)
    parser.add_argument("--s1-obstacle-distance-m", type=float, default=S1_OBSTACLE_DISTANCE)
    parser.add_argument("--s1-stage-stable-s", type=float, default=1.0)
    parser.add_argument("--s1-stage-speed-tolerance-kmh", type=float, default=2.0)
    parser.add_argument("--s2-target-speed-kmh", type=float, default=S2_NPC_SPEED_KMH)
    parser.add_argument("--s2-gap-m", type=float, default=S2_NPC_INITIAL_GAP)
    parser.add_argument("--s2-handover-settle-s", type=float, default=1.0)
    parser.add_argument("--s2-no-stage", action="store_true",
                        help="Disable controlled S2 staging (not recommended for comparisons)")
    parser.add_argument("--s5-duration-s", type=float, default=SCENARIO_DURATION_S[5],
                        help="Length of the S5 ghost-exposure drive")
    args = parser.parse_args()
    if min(
        args.s1_target_speed_kmh,
        args.s1_obstacle_distance_m,
        args.s1_stage_stable_s,
        args.s1_stage_speed_tolerance_kmh,
        args.s5_duration_s,
    ) <= 0.0:
        parser.error("scenario parameters must be positive")
    if args.output_root is None:
        if args.driver in ("mlp", "transformer") and args.radar_backend != "native":
            args.output_root = f"results_{args.driver}_{args.radar_backend}"
        else:
            args.output_root = f"results_{args.driver}"
    os.makedirs(args.output_root, exist_ok=True)

    total_runs = len(args.towns) * len(args.scenarios) * len(args.fog) * len(args.seeds)
    print("=" * 64)
    print("MASTER RUNNER — ALL NHTSA SCENARIOS")
    print("=" * 64)
    print(f"  Towns:           {args.towns}")
    print(f"  Driver:          {args.driver}")
    print(f"  Radar backend:   {args.radar_backend}")
    if args.radar_backend == "realistic":
        print(f"  Radar profile:   {args.radar_profile or 'model/generic_lrr_v1'}")
        if args.radar_config:
            print(f"  Radar config:    {args.radar_config}")
        if args.radar_multipath_mode:
            print(f"  Multipath:       {args.radar_multipath_mode} (runtime override)")
        if args.radar_ghost_rate_scale is not None:
            print(f"  Ghost rate x:    {args.radar_ghost_rate_scale}")
        if args.radar_ghost_snr_offset_db is not None:
            print(f"  Ghost SNR:       {args.radar_ghost_snr_offset_db:+g} dB")
        if args.radar_ghost_oracle:
            print("  Ghost filter:    ORACLE (ground-truth ceiling)")
        elif args.radar_ghost_detector:
            print(f"  Ghost detector:  {args.radar_ghost_detector}")
    print(f"  Output root:     {args.output_root}")
    print(f"  Scenarios:       {args.scenarios}")
    print(f"  Fog levels:      {args.fog}")
    print(f"  Seeds:           {args.seeds}")
    print(f"  Total runs:      {total_runs}")
    print(f"  CARLA:           {args.host}:{args.port}")
    print("  Subprocess mode: each run in a fresh Python process")
    print("=" * 64)
    print()

    all_results = []
    run_number = 0
    start = time.time()
    radar_flags = _radar_passthrough(args)

    for town in args.towns:
        town_root = (
            args.output_root
            if len(args.towns) == 1
            else os.path.join(args.output_root, town)
        )
        for sid in args.scenarios:
            if sid not in SCENARIO_MODULES:
                print(f"  Unknown scenario {sid}, skipping")
                continue
            module_name, output_dir = SCENARIO_MODULES[sid]
            scenario_script = f"{module_name}.py"

            for fog in args.fog:
                for seed in args.seeds:
                    run_number += 1
                    print(f"\n  [{run_number}/{total_runs}] {town} S{sid} fog={fog} seed={seed}")
                    print(f"  {'─' * 50}")
                    record = {
                        "town": town,
                        "scenario": sid,
                        "fog": fog,
                        "seed": seed,
                        "collision": "error",
                        "min_distance_m": -1.0,
                        "returncode": None,
                    }
                    cmd = [
                        sys.executable, scenario_script,
                        "--host", args.host,
                        "--port", str(args.port),
                        "--town", town,
                        "--fog", str(fog),
                        "--seeds", str(seed),
                        "--output", os.path.join(town_root, output_dir),
                        "--driver", args.driver,
                        "--model-dir", args.model_dir,
                        "--pcla-agent", args.pcla_agent,
                        *radar_flags,
                        *_scenario_passthrough(args, sid),
                    ]
                    if args.safety_rules:
                        cmd.append("--safety-rules")
                    try:
                        result = subprocess.run(
                            cmd,
                            capture_output=True,
                            text=True,
                            timeout=args.timeout,
                        )
                        for line in result.stdout.splitlines():
                            if line.strip():
                                print(f"    {line}")
                        record["returncode"] = result.returncode
                        if result.returncode != 0:
                            print(f"    ❌ Process exited with code {result.returncode}")
                            for line in result.stderr.splitlines()[-8:]:
                                print(f"    ⚠️  {line}")
                        else:
                            collision, min_dist = _parse_result(result.stdout)
                            record["collision"] = collision
                            record["min_distance_m"] = min_dist
                    except subprocess.TimeoutExpired:
                        print(f"    ❌ Timed out after {args.timeout}s")
                        record["collision"] = "timeout"
                    except Exception as exc:  # noqa: BLE001
                        print(f"    ❌ Failed: {exc}")
                    all_results.append(record)

    elapsed = time.time() - start
    print(f"\n{'=' * 64}")
    print("ALL DONE")
    print(f"  Total runs:   {run_number}")
    print(f"  Time elapsed: {elapsed:.0f}s ({elapsed / 60:.1f}min)")
    print("=" * 64)

    summary_path = os.path.join(args.output_root, "summary_all.csv")
    with open(summary_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "town", "scenario", "fog", "seed", "collision", "min_distance_m", "returncode",
        ])
        writer.writeheader()
        writer.writerows(all_results)
    print(f"\n  Summary saved: {summary_path}")

    print(f"\n{'=' * 64}")
    print("SUMMARY TABLE")
    print("=" * 64)
    print(f"{'Town':>10} {'Scen':>5} {'Fog':>5} {'Seed':>5} {'Collision':>10} {'MinDist':>8}")
    for record in all_results:
        print(f"{record['town']:>10} {record['scenario']:5d} {record['fog']:5d} "
              f"{record['seed']:5d} {str(record['collision']):>10s} "
              f"{str(record['min_distance_m']):>8s}")
    print("=" * 64)
    print("\n  Next: python3 analyze_results.py --runs <label>=<output-root> ... --out comparison")


if __name__ == "__main__":
    main()
