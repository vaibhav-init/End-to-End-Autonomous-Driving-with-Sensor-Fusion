"""Closed-loop RRI: CARLA geometry driven through the full sensor model.

``run_carla_rri_sweep.py`` evaluates the interference physics and reports how
many phantoms the model *would* emit. That is not the hazard. The hazard is a
phantom that wins track association, because the controller then brakes for an
object that is not there. Answering that needs the whole chain -- detection
probability, quantisation, the association gate, track confirmation and the
longitudinal selector -- so this script drives
``RealisticRadarModel.step()`` with the collected geometry and reports what the
tracker actually did.

What the sensor sees each scan
------------------------------
The neighbour vehicle is both a real target and an interferer, which is the
honest situation: its front radar leaks into ours *and* its body reflects. It
is therefore injected twice -- once as an ``IdealRadarTarget`` at the measured
range/azimuth/Doppler, once as an ``InterferingRadar``. Additional real targets
(another vehicle further back, a pedestrian crossing) are synthesised in the
same sensor frame so the tracker has something to choose between; without a
competing target the selection metric cannot distinguish "picked the phantom"
from "picked the only thing there".

Sensor frame
------------
Geometry is consumed in the frame of the radar under test, whose boresight is
whatever the collector recorded. For the tailgating case that is a rear-facing
mount, so a bearing near 0 degrees means directly behind ego.

Run:
    python3 run_carla_rri_closed_loop.py
    python3 run_carla_rri_closed_loop.py --isolation-db 10 --coherence 0.3
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from collections import defaultdict

# Allow running from the repository root.
_CARLA_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _CARLA_DIR not in sys.path:
    sys.path.insert(0, _CARLA_DIR)

from radar.realistic_core import (
    IdealRadarTarget,
    RealisticRadarModel,
    load_realistic_radar_config,
    realistic_radar_config_signature,
)
from radar.rri import InterferingRadar

TX_POWER_DBM = 10.0
TX_GAIN_DBI = 8.0
CYCLE_TIME_S = 0.1

# Background traffic in the radar's frame. Fixed geometry on purpose: the study
# varies the interference, not the scene, so these are held constant and only
# exist to give the selector a choice.
BACK_VEHICLE_RANGE_M = 45.0
BACK_VEHICLE_AZIMUTH_DEG = 4.0
BACK_VEHICLE_SNR_DB = 30.0
CROSSING_PEDESTRIAN_RANGE_M = 28.0
CROSSING_PEDESTRIAN_AZIMUTH_DEG = 22.0
CROSSING_PEDESTRIAN_SNR_DB = 24.0

# Range resolution is 0.15 m and azimuth 1.8 deg, so a neighbour at 3 m and one
# at 45 m are far apart in any metric the tracker uses.
PHANTOM_SELECTED_SOURCE = "phantom"


def _snr_for_range(distance_m, reference_snr_db, reference_range_m=35.0):
    return reference_snr_db + 40.0 * math.log10(reference_range_m / max(distance_m, 1.0))


def _read(path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def _targets_for(row):
    """Real targets in the radar's frame for one collected scan."""

    neighbour_range = float(row["range_m"])
    neighbour_azimuth = float(row["azimuth_deg"])
    neighbour_closing = float(row["closing_mps"])

    targets = [
        # The neighbour itself: its front radar leaks into ours and its body
        # reflects. The neighbour's RCS is that of a small car.
        IdealRadarTarget(
            object_id=1,
            semantic_tag=10,
            distance_m=neighbour_range,
            azimuth_rad=math.radians(neighbour_azimuth),
            relative_velocity_mps=neighbour_closing,
            snr_db=_snr_for_range(neighbour_range, 32.0),
            velocity_xy_mps=(neighbour_closing, 0.0),
            lateral_extent_m=0.9,
        ),
        IdealRadarTarget(
            object_id=2,
            semantic_tag=10,
            distance_m=BACK_VEHICLE_RANGE_M,
            azimuth_rad=math.radians(BACK_VEHICLE_AZIMUTH_DEG),
            relative_velocity_mps=0.0,
            snr_db=_snr_for_range(BACK_VEHICLE_RANGE_M, BACK_VEHICLE_SNR_DB),
            velocity_xy_mps=(0.0, 0.0),
            lateral_extent_m=0.9,
        ),
        IdealRadarTarget(
            object_id=3,
            semantic_tag=12,
            distance_m=CROSSING_PEDESTRIAN_RANGE_M,
            azimuth_rad=math.radians(CROSSING_PEDESTRIAN_AZIMUTH_DEG),
            relative_velocity_mps=-1.2,
            snr_db=_snr_for_range(
                CROSSING_PEDESTRIAN_RANGE_M, CROSSING_PEDESTRIAN_SNR_DB
            ),
            velocity_xy_mps=(-1.2, 0.6),
            lateral_extent_m=0.4,
        ),
    ]
    return targets


def _interferer_for(row):
    return InterferingRadar(
        object_id=1,
        distance_m=float(row["range_m"]),
        azimuth_rad=math.radians(float(row["azimuth_deg"])),
        relative_velocity_mps=float(row["closing_mps"]),
        transmit_power_dbm=TX_POWER_DBM,
        antenna_gain_dbi=TX_GAIN_DBI,
        boresight_alignment=float(row["boresight_alignment"]),
        label="neighbour_front_radar",
    )


def _run_group(rows, isolation_db, coherence, seed):
    overrides = {
        "rri_mode": "parametric",
        "rri_antenna_isolation_db": float(isolation_db),
        "rri_interference_coherence": float(coherence),
        # Multipath off so a wall ghost is never counted as an RRI phantom.
        "multipath_mode": "off",
        "cycle_time_s": CYCLE_TIME_S,
    }
    config = load_realistic_radar_config("rgd_regime_v1", overrides=overrides)
    model = RealisticRadarModel(config, seed=seed)

    totals = defaultdict(int)
    per_scan = []
    for index, row in enumerate(rows):
        timestamp_s = float(row["time_s"])
        selected = model.step(
            _targets_for(row),
            timestamp_s=timestamp_s,
            interferers=[_interferer_for(row)],
        )
        _, points = model.latest_points()
        diagnostics = model.diagnostics()

        phantom_points = sum(point.source == "phantom" for point in points)
        record = {
            "scenario": row["scenario"],
            "isolation_db": float(isolation_db),
            "coherence": float(coherence),
            "frame": int(row["frame"]),
            "time_s": timestamp_s,
            "range_m": float(row["range_m"]),
            "azimuth_deg": float(row["azimuth_deg"]),
            "boresight_alignment": float(row["boresight_alignment"]),
            "max_inr_db": diagnostics["rri_max_inr_db"],
            "phantom_points": phantom_points,
            "clutter_points": sum(p.source == "clutter" for p in points),
            "direct_points": sum(p.source == "direct" for p in points),
            "phantom_detections": diagnostics["rri_phantom_detection_count"],
            "false_alarm_multiplier": diagnostics["rri_false_alarm_multiplier"],
            "selected_source": selected.source or "",
            "selected_distance_m": round(float(selected.distance_m), 3),
            "selected_relative_velocity_mps": round(
                float(selected.relative_velocity_mps), 4
            ),
            "confirmed_tracks": diagnostics["confirmed_track_count"],
        }
        per_scan.append(record)

        if index < 30:
            # Discard the tracker's confirmation window.
            continue
        totals["scans"] += 1
        totals["phantom_scans"] += int(phantom_points > 0)
        totals["phantom_points"] += phantom_points
        totals["clutter_points"] += sum(p.source == "clutter" for p in points)
        totals["direct_points"] += sum(p.source == "direct" for p in points)
        totals["phantom_selected"] += int(
            selected.source == PHANTOM_SELECTED_SOURCE
        )
        totals["peak_inr"] = max(totals["peak_inr"], diagnostics["rri_max_inr_db"])

    scans = max(totals["scans"], 1)
    summary = {
        "scenario": rows[0]["scenario"] if rows else "",
        "isolation_db": float(isolation_db),
        "coherence": float(coherence),
        "scans": int(totals["scans"]),
        "peak_inr_db": round(totals["peak_inr"], 2),
        "phantom_points_per_scan": round(totals["phantom_points"] / scans, 4),
        "clutter_points_per_scan": round(totals["clutter_points"] / scans, 4),
        "direct_points_per_scan": round(totals["direct_points"] / scans, 4),
        "phantom_scan_fraction": round(totals["phantom_scans"] / scans, 4),
        "phantom_selected_count": int(totals["phantom_selected"]),
        "phantom_selected_fraction": round(totals["phantom_selected"] / scans, 4),
    }
    return summary, per_scan


def _write_csv(path, rows):
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--geometry", default=os.path.join(here, "carla_rri_geometry.csv")
    )
    parser.add_argument("--isolation-db", default="40,25,20,15,10")
    parser.add_argument(
        "--coherence",
        default="0.1,0.3,0.6",
        help="Interference coherence fractions. This is the least-constrained "
        "parameter in the model, so it is swept rather than fixed.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default=here)
    args = parser.parse_args()

    if not os.path.exists(args.geometry):
        raise SystemExit(
            f"missing {args.geometry}\nrun collect_carla_rri_geometry.py first"
        )

    rows = _read(args.geometry)
    isolations = sorted(
        {float(v) for v in str(args.isolation_db).split(",") if v.strip()}, reverse=True
    )
    coherences = sorted(
        {float(v) for v in str(args.coherence).split(",") if v.strip()}
    )

    summaries = []
    per_scan = []
    for scenario in dict.fromkeys(row["scenario"] for row in rows):
        subset = [row for row in rows if row["scenario"] == scenario]
        for coherence in coherences:
            for isolation_db in isolations:
                summary, scans = _run_group(
                    subset, isolation_db, coherence, args.seed
                )
                summaries.append(summary)
                per_scan.extend(scans)

    os.makedirs(args.output_dir, exist_ok=True)
    summary_path = os.path.join(args.output_dir, "closed_loop_summary.csv")
    scan_path = os.path.join(args.output_dir, "closed_loop_per_scan.csv")
    _write_csv(summary_path, summaries)
    _write_csv(scan_path, per_scan)

    config = load_realistic_radar_config(
        "rgd_regime_v1", overrides={"rri_mode": "parametric", "multipath_mode": "off"}
    )
    print(f"signature {realistic_radar_config_signature(config)}")
    print(f"wrote {summary_path}\nwrote {scan_path}\n")

    header = (
        f"{'scenario':30s} {'coh':>4s} {'iso':>4s} {'peakINR':>8s} "
        f"{'phantom/scan':>13s} {'direct/scan':>12s} {'clutter/scan':>13s} "
        f"{'PHANTOM SELECTED':>17s}"
    )
    print(header)
    for item in summaries:
        flag = (
            f"{item['phantom_selected_count']}/{item['scans']}"
            if item["phantom_selected_count"]
            else "-"
        )
        print(
            f"{item['scenario']:30s} {item['coherence']:4.1f} {item['isolation_db']:4.0f} "
            f"{item['peak_inr_db']:8.1f} {item['phantom_points_per_scan']:13.3f} "
            f"{item['direct_points_per_scan']:12.3f} {item['clutter_points_per_scan']:13.3f} "
            f"{flag:>17s}"
        )


if __name__ == "__main__":
    main()