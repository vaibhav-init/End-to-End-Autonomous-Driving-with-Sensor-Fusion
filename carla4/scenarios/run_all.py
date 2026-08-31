#!/usr/bin/env python3
"""
Master Runner — All Scenarios
==============================

Runs all three NHTSA-aligned scenarios across the full fog ladder
with multiple seeds using CARLA Traffic Manager autopilot as the baseline.

Usage (on remote machine via AnyDesk):
  # Start CARLA first
  cd /opt/carla-simulator && ./CarlaUE4.sh

  # Run all scenarios
  cd /home/vaibhav/carla-claude/carla4/scenarios
  python run_all.py

  # Run one scenario
  python run_all.py --scenarios 1

  # Quick test (1 fog level, 1 seed)
  python run_all.py --fog 0 --seeds 42

Output:
  results_s1/  — GT CSVs for Scenario 1
  results_s2/  — GT CSVs for Scenario 2
  results_s3/  — GT CSVs for Scenario 3
  summary_all.csv  — Aggregated results table
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
    FOG_LADDER,
    RANDOM_SEEDS,
    S1_OBSTACLE_DISTANCE,
    S1_SPAWN_SPEED_KMH,
    S2_NPC_INITIAL_GAP,
    S2_NPC_SPEED_KMH,
)
from radar import add_radar_arguments


SCENARIO_MODULES = {
    1: ("s1_lead_vehicle_stopped", "results_s1"),
    2: ("s2_lead_vehicle_decelerating", "results_s2"),
    # 3: ("s3_lead_vehicle_constant_speed", "results_s3"),  # S3 disabled for now
    4: ("s4_cut_in", "results_s4"),
}


def main():
    parser = argparse.ArgumentParser(description="Run all NHTSA scenarios")
    parser.add_argument("--scenarios", type=int, nargs="+",
                        default=[1, 2, 4],  # S3 disabled for now
                        help="Scenarios to run (1, 2, 4; 3 is disabled)")
    parser.add_argument("--fog", type=int, nargs="+", default=FOG_LADDER,
                        help="Fog densities to test")
    parser.add_argument("--seeds", type=int, nargs="+", default=RANDOM_SEEDS,
                        help="Random seeds")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--town", default="Town04")
    parser.add_argument(
        "--safety-rules",
        action="store_true",
        help=(
            "re-enable the hardcoded emergency-brake overrides in the mlp "
            "driver (ablation arm). Off by default so the model decides."
        ),
    )
    parser.add_argument("--driver", choices=["pcla", "mlp", "idm"], default="mlp",
                        help="Longitudinal control source for every run")
    parser.add_argument("--model-dir", default="../model_throttle_brake",
                        help="MLP model directory (for --driver mlp)")
    parser.add_argument("--pcla-agent", default="tfv6_visiononly",
                        help="PCLA agent name (for --driver pcla)")
    add_radar_arguments(parser)
    parser.add_argument("--output-root", default=None,
                        help="Root dir for results (default: results_<driver>)")
    parser.add_argument("--timeout", type=int, default=300,
                        help="Per-run subprocess timeout in seconds")
    parser.add_argument(
        "--s1-target-speed-kmh",
        type=float,
        default=S1_SPAWN_SPEED_KMH,
        help="S1 stress-test speed before obstacle appearance",
    )
    parser.add_argument(
        "--s1-obstacle-distance-m",
        type=float,
        default=S1_OBSTACLE_DISTANCE,
        help="S1 stopped-obstacle distance",
    )
    parser.add_argument(
        "--s1-stage-stable-s",
        type=float,
        default=1.0,
        help="Required stable time at the S1 target speed",
    )
    parser.add_argument(
        "--s1-stage-speed-tolerance-kmh",
        type=float,
        default=2.0,
        help="Allowed S1 staging speed error",
    )
    parser.add_argument(
        "--s2-target-speed-kmh",
        type=float,
        default=S2_NPC_SPEED_KMH,
    )
    parser.add_argument(
        "--s2-gap-m",
        type=float,
        default=S2_NPC_INITIAL_GAP,
    )
    parser.add_argument("--s2-handover-settle-s", type=float, default=1.0)
    parser.add_argument(
        "--s2-no-stage",
        action="store_true",
        help="Disable controlled S2 staging (not recommended for comparisons)",
    )
    args = parser.parse_args()
    if min(
        args.s1_target_speed_kmh,
        args.s1_obstacle_distance_m,
        args.s1_stage_stable_s,
        args.s1_stage_speed_tolerance_kmh,
    ) <= 0.0:
        parser.error("S1 stress-test parameters must be positive")
    if args.output_root is None:
        if args.driver == "mlp" and args.radar_backend != "native":
            args.output_root = f"results_mlp_{args.radar_backend}"
        else:
            args.output_root = f"results_{args.driver}"
    os.makedirs(args.output_root, exist_ok=True)

    total_runs = len(args.scenarios) * len(args.fog) * len(args.seeds)
    print("=" * 64)
    print("MASTER RUNNER — ALL NHTSA SCENARIOS")
    print("=" * 64)
    print(f"  Town:            {args.town}")
    print(f"  Driver:          {args.driver}")
    print(f"  Radar backend:   {args.radar_backend}")
    if args.radar_backend == "realistic":
        print(f"  Radar profile:   {args.radar_profile or 'generic_lrr_v1'}")
        if args.radar_config:
            print(f"  Radar config:    {args.radar_config}")
        if args.radar_ghost_detector:
            print(f"  Ghost detector:  {args.radar_ghost_detector}")
    print(f"  Output root:     {args.output_root}")
    print(f"  Scenarios:       {args.scenarios}")
    print(f"  Fog levels:      {args.fog}")
    print(f"  Seeds:           {args.seeds}")
    print(f"  Total runs:      {total_runs}")
    print(f"  CARLA:           {args.host}:{args.port}")
    print(f"  Subprocess mode: each run in a fresh Python process")
    print("=" * 64)
    print()

    all_results = []
    run_number = 0
    start = time.time()

    for sid in args.scenarios:
        if sid not in SCENARIO_MODULES:
            print(f"  Unknown scenario {sid}, skipping")
            continue

        module_name, output_dir = SCENARIO_MODULES[sid]
        scenario_script = f"{module_name}.py"

        for fog in args.fog:
            for seed in args.seeds:
                run_number += 1
                print(f"\n  [{run_number}/{total_runs}] Running S{sid} fog={fog} seed={seed}")
                print(f"  {'─' * 50}")

                try:
                    cmd = [
                        sys.executable, scenario_script,
                        "--host", args.host,
                        "--port", str(args.port),
                        "--town", args.town,
                        "--fog", str(fog),
                        "--seeds", str(seed),
                        "--output", os.path.join(args.output_root, output_dir),
                        "--driver", args.driver,
                        "--model-dir", args.model_dir,
                        "--pcla-agent", args.pcla_agent,
                        "--radar-backend", args.radar_backend,
                    ]
                    if args.safety_rules:
                        cmd.append("--safety-rules")
                    if args.radar_profile:
                        cmd.extend(["--radar-profile", args.radar_profile])
                    if args.radar_config:
                        cmd.extend(["--radar-config", args.radar_config])
                    if args.radar_seed is not None:
                        cmd.extend(["--radar-seed", str(args.radar_seed)])
                    if args.radar_ghost_detector:
                        cmd.extend(
                            [
                                "--radar-ghost-detector",
                                args.radar_ghost_detector,
                            ]
                        )
                    if args.radar_ghost_threshold is not None:
                        cmd.extend(
                            [
                                "--radar-ghost-threshold",
                                str(args.radar_ghost_threshold),
                            ]
                        )
                    if args.radar_ghost_device:
                        cmd.extend(
                            ["--radar-ghost-device", args.radar_ghost_device]
                        )
                    if sid == 1:
                        cmd.extend([
                            "--target-speed-kmh",
                            str(args.s1_target_speed_kmh),
                            "--obstacle-distance-m",
                            str(args.s1_obstacle_distance_m),
                            "--stage-stable-s",
                            str(args.s1_stage_stable_s),
                            "--stage-speed-tolerance-kmh",
                            str(args.s1_stage_speed_tolerance_kmh),
                        ])
                    elif sid == 2:
                        cmd.extend([
                            "--target-speed-kmh",
                            str(args.s2_target_speed_kmh),
                            "--initial-gap",
                            str(args.s2_gap_m),
                            "--stage-gap",
                            str(args.s2_gap_m),
                            "--handover-settle-s",
                            str(args.s2_handover_settle_s),
                        ])
                        if args.s2_no_stage:
                            cmd.append("--no-stage-approach")
                    result = subprocess.run(
                        cmd,
                        capture_output=True,
                        text=True,
                        timeout=args.timeout,
                    )

                    # Print live output
                    for line in result.stdout.splitlines():
                        if line.strip():
                            print(f"    {line}")

                    if result.returncode != 0:
                        print(f"    ❌ Process exited with code {result.returncode}")
                        if result.stderr:
                            for line in result.stderr.splitlines()[-5:]:
                                print(f"    ⚠️  {line}")
                        all_results.append({
                            "scenario": sid,
                            "fog": fog,
                            "seed": seed,
                            "collision": "error",
                            "min_distance_m": -1.0,
                            "rows": 0,
                        })
                    else:
                        # Try to parse result from stdout
                        collision = "?"
                        min_dist = "?"
                        rows = "?"
                        for line in result.stdout.splitlines():
                            if "collision=" in line.lower() and "min_dist" in line.lower():
                                parts = line.split("collision=")
                                if len(parts) > 1:
                                    collision = parts[1].split()[0].strip()
                                parts2 = line.split("min_dist=")
                                if len(parts2) > 1:
                                    min_dist = parts2[1].split()[0].strip()
                            # Check for CRASHED status in output
                            if "❌" in line and "failed" in line.lower():
                                collision = "error"

                        # Also check if summary was printed
                        for line in result.stdout.splitlines():
                            if "collisions" in line.lower() and "runs" in line.lower():
                                parts = line.split()
                                for i, p in enumerate(parts):
                                    if p == "collisions" and i > 0:
                                        try:
                                            rows = parts[i - 1]
                                        except (ValueError, IndexError):
                                            pass

                        all_results.append({
                            "scenario": sid,
                            "fog": fog,
                            "seed": seed,
                            "collision": collision,
                            "min_distance_m": min_dist,
                            "rows": rows,
                        })

                except subprocess.TimeoutExpired:
                    print(f"    ❌ Timed out after 120s")
                    all_results.append({
                        "scenario": sid,
                        "fog": fog,
                        "seed": seed,
                        "collision": "timeout",
                        "min_distance_m": -1.0,
                        "rows": 0,
                    })
                except Exception as e:
                    print(f"    ❌ Failed: {e}")
                    all_results.append({
                        "scenario": sid,
                        "fog": fog,
                        "seed": seed,
                        "collision": "error",
                        "min_distance_m": -1.0,
                        "rows": 0,
                    })

    elapsed = time.time() - start
    print(f"\n{'=' * 64}")
    print("ALL DONE")
    print(f"  Total runs:   {run_number}")
    print(f"  Time elapsed: {elapsed:.0f}s ({elapsed / 60:.1f}min)")
    print("=" * 64)

    # Save summary
    summary_path = os.path.join(args.output_root, "summary_all.csv")
    with open(summary_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "scenario", "fog", "seed", "collision", "min_distance_m", "rows"
        ])
        w.writeheader()
        w.writerows(all_results)
    print(f"\n  Summary saved: {summary_path}")

    # Print summary table
    print(f"\n{'=' * 64}")
    print("SUMMARY TABLE")
    print("=" * 64)
    print(f"{'Scen':>5} {'Fog':>5} {'Seed':>5} {'Collision':>10} {'MinDist':>8} {'Rows':>6}")
    print(f"{'─' * 5} {'─' * 5} {'─' * 5} {'─' * 10} {'─' * 8} {'─' * 6}")
    for r in all_results:
        print(f"{r['scenario']:5d} {r['fog']:5d} {r['seed']:5d} "
              f"{str(r['collision']):>10s} {str(r['min_distance_m']):>8s} "
              f"{str(r['rows']):>6s}")
    print("=" * 64)


if __name__ == "__main__":
    main()
