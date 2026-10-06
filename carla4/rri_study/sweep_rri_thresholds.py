"""One-at-a-time sensitivity of the RRI result to each modelling threshold.

Answers the question "which thresholds are the conclusion resting on?" by
varying one parameter at a time around a fixed baseline and reporting the two
outcomes that matter: phantom detections per scan, and the fraction of scans in
which a phantom won track association.

Two classes of parameter are swept, and the distinction matters.

*Modelling assumptions* (the ``rri_*`` fields) describe the interference: how
much leaks, how coherent it is, how strong it must be before CFAR reports it.
These are not measured, so sweeping them and reporting the spread is the
legitimate result. A conclusion that survives the whole sweep is robust; one
that flips inside it is conditional and must be stated as such.

*Detector criteria* (the association gates, the detection midpoint) decide what
counts as a phantom worth reporting. Relaxing these raises the phantom rate
without making the interference any stronger, so it is reported for
completeness and labelled as a scoring choice rather than a physical knob.

Run:
    python3 sweep_rri_thresholds.py
    python3 sweep_rri_thresholds.py --isolation-db 15 --coherence 0.3
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

# Allow running from the repository root.
_CARLA_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _CARLA_DIR not in sys.path:
    sys.path.insert(0, _CARLA_DIR)

from radar.realistic_core import load_realistic_radar_config

from run_carla_rri_closed_loop import _read, _run_group, _write_csv

# name -> (values, class). "assumption" parameters are physics we chose;
# "criterion" parameters decide what counts as a reportable phantom.
SWEEPS = {
    # --- modelling assumptions ---
    "rri_antenna_isolation_db": (
        [30.0, 25.0, 20.0, 15.0, 10.0, 5.0],
        "assumption",
    ),
    "rri_interference_coherence": (
        [0.0, 0.1, 0.3, 0.6, 1.0],
        "assumption",
    ),
    "rri_min_inr_db_for_phantom": (
        [12.0, 8.0, 5.0, 3.0, 0.0, -3.0],
        "assumption",
    ),
    "rri_phantom_detection_slope": (
        [0.05, 0.10, 0.20, 0.40, 0.80],
        "assumption",
    ),
    "rri_desensitisation_db": (
        [0.0, 1.0, 3.0, 6.0, 12.0],
        "assumption",
    ),
    "rri_max_ghost_orders": (
        [0, 1, 2, 3],
        "assumption",
    ),
    "rri_chirp_period_s": (
        [2.0e-6, 1.0e-6, 5.0e-7],
        "assumption",
    ),
    "rri_coherent_processing_chirps": (
        [16, 32, 64, 128],
        "assumption",
    ),
    # --- detector criteria ---
    "association_range_gate_m": (
        [1.0, 2.0, 4.0, 8.0],
        "criterion",
    ),
    "association_azimuth_gate_deg": (
        [1.0, 2.0, 5.0, 10.0],
        "criterion",
    ),
    "detection_snr_midpoint_db": (
        [4.0, 6.0, 8.0, 12.0],
        "criterion",
    ),
    "confirmation_hits": (
        [1, 2, 3, 5],
        "criterion",
    ),
}


def _row(scenario, isolation_db, coherence, parameter, value, klass, summary):
    return {
        "scenario": scenario,
        "parameter": parameter,
        "value": value,
        "class": klass,
        "isolation_db": float(isolation_db),
        "coherence": float(coherence),
        "peak_inr_db": summary["peak_inr_db"],
        "phantom_points_per_scan": summary["phantom_points_per_scan"],
        "phantom_scan_fraction": summary["phantom_scan_fraction"],
        "phantom_selected_count": summary["phantom_selected_count"],
        "phantom_selected_fraction": summary["phantom_selected_fraction"],
        "clutter_points_per_scan": summary["clutter_points_per_scan"],
        "direct_points_per_scan": summary["direct_points_per_scan"],
        "scans": summary["scans"],
    }


def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--geometry", default=os.path.join(here, "carla_rri_geometry.csv")
    )
    parser.add_argument(
        "--scenario",
        default="rear_radar_tailgating/closing",
        help="The only scenario where the conclusion is non-trivial.",
    )
    parser.add_argument("--isolation-db", type=float, default=15.0)
    parser.add_argument("--coherence", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default=here)
    args = parser.parse_args()

    rows = _read(args.geometry)
    subset = [row for row in rows if row["scenario"] == args.scenario]
    if not subset:
        raise SystemExit(
            f"scenario {args.scenario!r} not in {args.geometry}; "
            f"have: {sorted({r['scenario'] for r in rows})}"
        )

    out_rows = []
    for parameter, (values, klass) in SWEEPS.items():
        for value in values:
            summary, _ = _run_group(
                subset,
                args.isolation_db,
                args.coherence,
                args.seed,
                extra_overrides={parameter: value},
            )
            out_rows.append(
                _row(
                    args.scenario,
                    args.isolation_db,
                    args.coherence,
                    parameter,
                    value,
                    klass,
                    summary,
                )
            )

    os.makedirs(args.output_dir, exist_ok=True)
    path = os.path.join(args.output_dir, "threshold_sensitivity.csv")
    _write_csv(path, out_rows)
    print(f"baseline: isolation {args.isolation_db:g} dB, coherence {args.coherence:g}")
    print(f"wrote {path}\n")

    for parameter, (values, klass) in SWEEPS.items():
        print(f"--- {parameter}  ({klass}) ---")
        subset_rows = [r for r in out_rows if r["parameter"] == parameter]
        for row in subset_rows:
            flag = (
                f"{row['phantom_selected_fraction']:6.1%}"
                if row["phantom_selected_count"]
                else "     -"
            )
            print(
                f"   {row['value']:>10} : peakINR {row['peak_inr_db']:5.1f} dB"
                f"  phantom/scan {row['phantom_points_per_scan']:6.3f}"
                f"  selected {flag}"
            )
        print()


if __name__ == "__main__":
    main()
