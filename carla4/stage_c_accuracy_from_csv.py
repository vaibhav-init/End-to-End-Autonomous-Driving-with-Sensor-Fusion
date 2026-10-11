"""Stage C accuracy comparison from the validator's own per-sample errors.

Uses the latency-aligned selected-target errors that
``validate_radar_accuracy.py`` already logged (``realistic_*_error_latency_aligned_*``),
restricted to frames where the correct lead was reported, versus the real-data
matched-pair core statistics (robust sigma produced by
``calibrate_rgd_real_data.py``).  This avoids polluting "sensor accuracy" with
the unmatched-ghost selection bias present in a many-detections-one-target
scenario, and merges the sim velocity behaviour with the dataset's estimator
inflation caveat spelled out in ``carla4/2026-10-09.md``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from typing import Any, Dict, List, Tuple

import numpy as np

sys_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "..")
import sys
sys.path.insert(0, sys_path)

from carla4.calibrate_rgd_real_data import robust_sigma  # noqa: E402

CORRECT_TARGET_SOURCE = "direct"


def selected_target_accuracy(csv_path: str) -> Dict[str, Any]:
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            try:
                correct = int(float(row["realistic_correct_target"]))
                if not correct:
                    continue
                rows.append({
                    "range_err_m": float(row["realistic_distance_error_latency_aligned_m"]),
                    "vr_err_mps": float(row["realistic_velocity_error_latency_aligned_mps"]),
                    "obstacle_speed_err": float(row.get("realistic_obstacle_speed_error_current_mps", "nan") or "nan"),
                    "azimuth_deg": float(row["realistic_selected_azimuth_deg"]),
                    "correct_source": row.get("realistic_target_identity_method", ""),
                })
            except (KeyError, TypeError, ValueError):
                continue
    if not rows:
        return {"n": 0}
    rng = np.asarray([row["range_err_m"] for row in rows])
    vr = np.asarray([row["vr_err_mps"] for row in rows])
    rng_sigma, rng_n = robust_sigma(rng)
    vr_sigma, vr_n = robust_sigma(vr)
    return {
        "n_correct_frames": len(rows),
        "range_error_mean_m": float(rng.mean()),
        "range_error_mae_m": float(np.abs(rng).mean()),
        "range_core_sigma_m": rng_sigma,
        "range_core_n": rng_n,
        "velocity_error_mean_mps": float(vr.mean()),
        "velocity_core_sigma_mps": vr_sigma,
        "velocity_core_n": vr_n,
    }


def load_real_reference(report_path: str) -> Dict[str, Any]:
    with open(report_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    picked = {}
    for report in payload.get("reports", []):
        core = report.get("core_stats", {})
        picked[report["channel"]] = {
            "range_sigma_m": core.get("range_sigma_m"),
            "velocity_sigma_mps": core.get("velocity_sigma_mps"),
            "n_core_pairs": core.get("n_core_pairs"),
        }
    return picked


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage_c_dir")
    parser.add_argument("--real-calibration-report", required=True,
                        help="JSON from calibrate_rgd_real_data.py")
    parser.add_argument("--out", default="stage_c_accuracy_report.json")
    args = parser.parse_args()

    sim = selected_target_accuracy(os.path.join(args.stage_c_dir, "radar_samples.csv"))
    real = load_real_reference(args.real_calibration_report)
    payload = {"sim": sim, "real_references": real}

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    print("SIM (selected target, latency aligned):", json.dumps(sim, indent=2))
    for name, values in real.items():
        print(f"REAL {name}: sigma_r={values['range_sigma_m']:.3f} m  "
              f"sigma_v={values['velocity_sigma_mps']:.3f} m/s  n={values['n_core_pairs']}")
    if sim.get("n_correct_frames"):
        print(f"\nverdict: sim core range sigma {sim['range_core_sigma_m']:.3f} m vs real ~{real.get('nuScenes_merged_front_summary', {}).get('range_sigma_m', float('nan')):.3f} m;"
              f" velocity sigma {sim['velocity_core_sigma_mps']:.3f} m/s vs real ~{real.get('nuScenes_merged_front_summary', {}).get('velocity_sigma_mps', float('nan')):.3f} m/s")
    print(f"report -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
