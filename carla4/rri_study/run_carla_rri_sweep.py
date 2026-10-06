"""Run the RRI model over CARLA-collected inter-radar geometry.

Reads ``carla_rri_geometry.csv`` (from ``collect_carla_rri_geometry.py``) and
drives ``radar.rri`` over the measured geometry, so the interference result is
attributable to CARLA's kinematics rather than to an analytic guess about where
two vehicles would be.

The split of responsibility is deliberate and worth stating plainly:

* CARLA supplies range, azimuth, closing rate and boresight alignment -- real
  road geometry, real lane curvature, real vehicle poses.
* ``radar.rri`` supplies the interference physics: link budget, coherent
  rejection, phantom generation, desensitisation and false-alarm inflation.
* Nothing here simulates a waveform, and CARLA's own radar is not consulted.

Per scan this reports the interference-to-noise ratio, the phantom detections
the model would emit, and how many of those the tracker would select -- the
last one being the phantom-braking hazard.

Run:
    python3 run_carla_rri_sweep.py
    python3 run_carla_rri_sweep.py --isolation-db 15
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys

# Allow running from the repository root.
_CARLA_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _CARLA_DIR not in sys.path:
    sys.path.insert(0, _CARLA_DIR)

from radar.rri import InterferingRadar, RRIParameters

TX_POWER_DBM = 10.0
TX_GAIN_DBI = 8.0


def _read(path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def _params(isolation_db, coherence):
    return RRIParameters(
        antenna_isolation_db=float(isolation_db),
        interference_coherence=float(coherence),
    )


def _sweep(rows, params):
    """Evaluate the interference at each collected scan."""

    out = []
    for row in rows:
        interferer = InterferingRadar(
            object_id=1,
            distance_m=float(row["range_m"]),
            azimuth_rad=math.radians(float(row["azimuth_deg"])),
            relative_velocity_mps=float(row["closing_mps"]),
            transmit_power_dbm=TX_POWER_DBM,
            antenna_gain_dbi=TX_GAIN_DBI,
            boresight_alignment=float(row["boresight_alignment"]),
            label=row["scenario"],
        )
        from radar.rri import evaluate_rri

        # A deterministic stream per scan keeps this reproducible without
        # pulling in the sensor model: only the phantom draw varies.
        import numpy as np

        report = evaluate_rri(
            [interferer],
            params,
            rng=np.random.default_rng([int(row["frame"]), int(isolation_seed(params))]),
        )
        out.append(
            {
                "scenario": row["scenario"],
                "frame": int(row["frame"]),
                "time_s": float(row["time_s"]),
                "range_m": float(row["range_m"]),
                "azimuth_deg": float(row["azimuth_deg"]),
                "closing_mps": float(row["closing_mps"]),
                "boresight_alignment": float(row["boresight_alignment"]),
                "max_inr_db": round(report.max_inr_db, 3),
                "desensitisation_db": round(report.desensitisation_db, 3),
                "false_alarm_multiplier": round(report.false_alarm_multiplier, 3),
                "phantom_count": report.phantom_count,
                "phantom_ranges_m": ";".join(
                    f"{item['distance_m']:.2f}" for item in report.phantoms
                ),
            }
        )
    return out


def isolation_seed(params):
    """Stable seed component so repeated runs agree."""

    return int(abs(params.antenna_isolation_db) * 100) % 100000


def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--geometry",
        default=os.path.join(here, "carla_rri_geometry.csv"),
        help="Output of collect_carla_rri_geometry.py.",
    )
    parser.add_argument(
        "--isolation-db",
        default="40,25,20,15,10",
        help="Comma-separated mount isolations to evaluate the geometry against.",
    )
    parser.add_argument("--coherence", type=float, default=0.3)
    parser.add_argument("--output-dir", default=here)
    args = parser.parse_args()

    if not os.path.exists(args.geometry):
        raise SystemExit(
            f"missing {args.geometry}\n"
            "collect it first:\n"
            "  python3 carla4/rri_study/collect_carla_rri_geometry.py"
        )

    rows = _read(args.geometry)
    isolations = sorted(
        {float(v) for v in str(args.isolation_db).split(",") if v.strip()},
        reverse=True,
    )

    all_rows = []
    for isolation_db in isolations:
        all_rows.extend(_sweep(rows, _params(isolation_db, args.coherence)))

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "carla_rri_sweep.csv")
    with open(out_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)

    # Per-scenario summary at the worst isolation tested.
    worst_iso = min(isolations)
    print(f"geometry: {args.geometry}  ({len(rows)} scans)")
    print(f"wrote {out_path}\n")
    print(f"{'scenario':34s} {'iso':>5s} {'maxINR':>7s} {'scans w/ phantom':>17s} {'first phantom at':>18s}")
    for scenario in dict.fromkeys(row["scenario"] for row in rows):
        for isolation_db in isolations:
            subset = [
                row
                for row in all_rows
                if row["scenario"] == scenario
                and abs(row["max_inr_db"] - 0.0) > 1e-12
            ]
            subset_all = [
                row
                for row in all_rows
                if row["scenario"] == scenario
            ]
            hits = [row for row in subset_all if row["phantom_count"] > 0]
            peak = max((row["max_inr_db"] for row in subset_all), default=0.0)
            first = f"{hits[0]['range_m']:.1f} m" if hits else "-"
            print(
                f"{scenario:34s} {isolation_db:5.0f} {peak:7.1f} "
                f"{len(hits):10d}/{len(subset_all):<6d} {first:>18s}"
            )


if __name__ == "__main__":
    main()
