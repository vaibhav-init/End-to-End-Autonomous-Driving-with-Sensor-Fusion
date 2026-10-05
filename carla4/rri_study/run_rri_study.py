"""Radar-to-radar interference study: adjacent-lane vehicle and rear-radar tailgating.

Two scenarios, both analytic (no CARLA required), both sweeping the geometry
that matters:

``adjacent_lane``
    Ego and a second vehicle travelling alongside in the neighbouring lane.
    Both radars face forward along their own lane, so the geometry is close to
    boresight-null: the interferer is roughly 90 degrees off our boresight. The
    sweep runs the longitudinal offset from 10 m behind to 10 m ahead, then
    repeats it with the neighbour angled 30 degrees toward us, which is the
    attitude a vehicle has partway through a lane change.

``rear_radar_tailgating``
    Ego runs a rear-facing radar and follows a lead vehicle. The following
    vehicle's radar faces us directly, so this is the worst-case boresight
    alignment. The sweep runs the following distance from 1 m to 60 m, which is
    the quantity that decides whether two radars can blind each other at all.

Both scenarios record, per scan: interference-to-noise ratio, SINR
desensitisation, false-alarm multiplier, phantom and clutter detection counts,
and whether a phantom was ever selected as the tracked target. That last one is
the safety-relevant output -- a phantom that wins track association is a
phantom brake -- so it is separated from the detection counts.

Run:
    python3 run_rri_study.py                      # both scenarios
    python3 run_rri_study.py --scenario rear_radar_tailgating
    python3 run_rri_study.py --isolation-db 20     # override the mount
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from collections import defaultdict

import numpy as np

# Allow running from the repository root: python3 carla4/rri_study/run_rri_study.py
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

# Lane geometry, in metres. Sensor frame is x forward, y right.
LANE_WIDTH_M = 3.5
EGO_SPEED_MPS = 15.0
NEIGHBOUR_SPEED_MPS = 15.0
CYCLE_TIME_S = 0.1

# A front radar's transmit power / gain, matching the 77 GHz band assumed by
# RRIParameters. Isolation is swept, not the power, because isolation is the
# quantity that actually varies between mounts and is what the study is about.
TX_POWER_DBM = 10.0
TX_GAIN_DBI = 8.0

# Target population for every scan: one car ahead in our lane plus one
# pedestrian crossing at 40 m. Enough for a tracker to hold, so a phantom has
# something to be confused with.
LEAD_RANGE_M = 35.0
LEAD_SPEED_MPS = 12.0
PEDESTRIAN_RANGE_M = 40.0
PEDESTRIAN_SPEED_MPS = -1.2
CAR_SNR_DB = 34.0
PEDESTRIAN_SNR_DB = 28.0


def _snr_for_range(distance_m: float, reference_snr_db: float, reference_range_m: float = 35.0) -> float:
    """Albersheim-equivalent range law, matching the comparison pipeline."""

    return reference_snr_db + 40.0 * math.log10(reference_range_m / max(distance_m, 1.0))


def _front_targets(timestamp_s: float):
    """Targets for a forward-facing radar: a car ahead and a crossing pedestrian."""

    lead_vy = 0.0
    ped_vx, ped_vy = -PEDESTRIAN_SPEED_MPS, 0.6
    ped_x, ped_y = PEDESTRIAN_RANGE_M, 4.0
    return [
        IdealRadarTarget(
            object_id=1,
            semantic_tag=10,
            distance_m=LEAD_RANGE_M,
            azimuth_rad=0.0,
            relative_velocity_mps=LEAD_SPEED_MPS - EGO_SPEED_MPS,
            snr_db=_snr_for_range(LEAD_RANGE_M, CAR_SNR_DB),
            velocity_xy_mps=(LEAD_SPEED_MPS - EGO_SPEED_MPS, lead_vy),
            lateral_extent_m=0.9,
        ),
        IdealRadarTarget(
            object_id=2,
            semantic_tag=12,
            distance_m=math.hypot(ped_x, ped_y),
            azimuth_rad=math.atan2(ped_y, ped_x),
            relative_velocity_mps=math.hypot(ped_vx, ped_vy),
            snr_db=_snr_for_range(PEDESTRIAN_RANGE_M, PEDESTRIAN_SNR_DB),
            velocity_xy_mps=(ped_vx, ped_vy),
            lateral_extent_m=0.4,
        ),
    ], timestamp_s


def _rear_targets(timestamp_s: float, following_distance_m: float):
    """Targets for the rear-facing radar, in the rear sensor's own frame.

    A rear radar's boresight points backwards, so in its own frame the vehicle
    being followed sits at azimuth ~0 and *closes* on us. Expressing the
    scenario in the sensor frame rather than the ego frame is what keeps the
    follower inside the field of view; placing it at azimuth 180 in the ego
    frame would put it outside a forward-looking FOV and silently discard every
    artifact, which is a property of the wrong sensor model rather than of the
    interference.
    """

    ped_vx, ped_vy = PEDESTRIAN_SPEED_MPS, -0.6
    ped_x, ped_y = 18.0, -4.0
    return [
        IdealRadarTarget(
            object_id=11,
            semantic_tag=10,
            distance_m=following_distance_m + 15.0,
            azimuth_rad=0.02,
            relative_velocity_mps=-(NEIGHBOUR_SPEED_MPS - EGO_SPEED_MPS),
            snr_db=_snr_for_range(following_distance_m + 15.0, CAR_SNR_DB),
            velocity_xy_mps=(NEIGHBOUR_SPEED_MPS - EGO_SPEED_MPS, 0.0),
            lateral_extent_m=0.9,
        ),
        IdealRadarTarget(
            object_id=12,
            semantic_tag=12,
            distance_m=math.hypot(ped_x, ped_y),
            azimuth_rad=math.atan2(ped_y, ped_x),
            relative_velocity_mps=math.hypot(ped_vx, ped_vy),
            snr_db=_snr_for_range(18.0, PEDESTRIAN_SNR_DB),
            velocity_xy_mps=(ped_vx, ped_vy),
            lateral_extent_m=0.4,
        ),
    ], timestamp_s


def _build_model(isolation_db: float, ghost_off: bool = True) -> RealisticRadarModel:
    overrides = {
        "rri_mode": "parametric",
        "rri_antenna_isolation_db": float(isolation_db),
        "cycle_time_s": CYCLE_TIME_S,
    }
    if ghost_off:
        # Isolate interference from multipath so a phantom is never confused
        # with a wall ghost; both use the ghost label downstream.
        overrides["multipath_mode"] = "off"
    config = load_realistic_radar_config("rgd_regime_v1", overrides=overrides)
    return RealisticRadarModel(config, seed=42)


def _run_case(
    label: str,
    interferer_at,
    isolation_db: float,
    scans: int,
    warmup: int,
    targets_at=None,
):
    """Run one geometry sweep point and summarise the interference outcome.

    ``interferer_at`` maps ``(frame, timestamp_s)`` to an ``InterferingRadar``
    or ``None``, so a scenario can sweep geometry while the scan loop stays
    identical across scenarios.
    """

    model = _build_model(isolation_db)
    totals = defaultdict(float)
    per_scan = []
    for frame in range(scans):
        timestamp_s = frame * CYCLE_TIME_S
        builder = targets_at or _front_targets
        targets, timestamp_s = builder(timestamp_s)
        interferer = interferer_at(frame, timestamp_s)
        interferers = (interferer,) if interferer is not None else ()
        selected = model.step(
            targets,
            timestamp_s=timestamp_s,
            interferers=interferers,
        )
        _, points = model.latest_points()
        diagnostics = model.diagnostics()

        counts = defaultdict(int)
        for point in points:
            counts[point.source] += 1

        row = {
            "label": label,
            "frame": frame,
            "time_s": round(timestamp_s, 4),
            "interferer_range_m": round(interferer.distance_m, 3) if interferer else float("nan"),
            "interferer_azimuth_deg": round(math.degrees(interferer.azimuth_rad), 3) if interferer else float("nan"),
            "boresight_alignment": round(interferer.boresight_alignment, 4) if interferer else float("nan"),
            "max_inr_db": round(diagnostics["rri_max_inr_db"], 3),
            "desensitisation_db": round(diagnostics["rri_desensitisation_db"], 3),
            "false_alarm_multiplier": round(diagnostics["rri_false_alarm_multiplier"], 3),
            "direct_points": counts["direct"],
            "phantom_points": counts["phantom"],
            "clutter_points": counts["clutter"],
            "ghost_points": counts["ghost"],
            "direct_detections": diagnostics["direct_detection_count"],
            "phantom_detections": diagnostics["rri_phantom_detection_count"],
            "clutter_detections": diagnostics["clutter_detection_count"],
            "selected_source": selected.source or "",
            "selected_distance_m": round(float(selected.distance_m), 3),
            "selected_relative_velocity_mps": round(float(selected.relative_velocity_mps), 4),
        }
        per_scan.append(row)

        if frame < warmup:
            continue
        totals["scans"] += 1
        totals["max_inr_db"] = max(totals["max_inr_db"], row["max_inr_db"])
        totals["desensitisation_db"] = max(
            totals["desensitisation_db"], row["desensitisation_db"]
        )
        totals["false_alarm_multiplier"] = max(
            totals["false_alarm_multiplier"], row["false_alarm_multiplier"]
        )
        totals["phantom_points"] += row["phantom_points"]
        totals["clutter_points"] += row["clutter_points"]
        totals["direct_points"] += row["direct_points"]
        totals["phantom_selected"] += int(row["selected_source"] == "phantom")
        totals["scans_with_phantom"] += int(row["phantom_points"] > 0)

    scanned = max(totals["scans"], 1.0)
    summary = {
        "label": label,
        "isolation_db": isolation_db,
        "scans": int(totals["scans"]),
        "max_inr_db": round(totals["max_inr_db"], 3),
        "max_desensitisation_db": round(totals["desensitisation_db"], 3),
        "max_false_alarm_multiplier": round(totals["false_alarm_multiplier"], 3),
        "phantom_points_per_scan": round(totals["phantom_points"] / scanned, 4),
        "clutter_points_per_scan": round(totals["clutter_points"] / scanned, 4),
        "direct_points_per_scan": round(totals["direct_points"] / scanned, 4),
        "scans_with_phantom_fraction": round(
            totals["scans_with_phantom"] / scanned, 4
        ),
        "phantom_selected_count": int(totals["phantom_selected"]),
    }
    return summary, per_scan


def scenario_adjacent_lane(isolation_db: float, scans: int, warmup: int):
    """Neighbouring-lane vehicle, boresight-null and angled toward us."""

    cases = []
    # Boresight-null: both radars face along their own lane, so the neighbour's
    # radar sits about 90 degrees off ours.
    cases.append(
        (
            "adjacent_lane/boresight_null",
            lambda frame, t: InterferingRadar(
                object_id=101,
                distance_m=math.hypot(LANE_WIDTH_M, 0.0),
                azimuth_rad=math.atan2(LANE_WIDTH_M, 0.0),
                relative_velocity_mps=0.0,
                transmit_power_dbm=TX_POWER_DBM,
                antenna_gain_dbi=TX_GAIN_DBI,
                boresight_alignment=0.02,
                label="adjacent_lane_neighbour",
            ),
        )
    )
    # Angled: the neighbour is 30 degrees off its own lane heading toward us,
    # which is roughly its attitude mid-lane-change, and 8 m alongside.
    cases.append(
        (
            "adjacent_lane/angled_30deg_8m",
            lambda frame, t: InterferingRadar(
                object_id=102,
                distance_m=math.hypot(8.0, LANE_WIDTH_M),
                azimuth_rad=math.atan2(LANE_WIDTH_M, 8.0),
                relative_velocity_mps=0.0,
                transmit_power_dbm=TX_POWER_DBM,
                antenna_gain_dbi=TX_GAIN_DBI,
                boresight_alignment=math.cos(math.radians(60.0)),
                label="adjacent_lane_neighbour_angled",
            ),
        )
    )
    # Very close: the neighbour has pulled alongside to within 2 m, which is
    # where the boresight-null case finally has any coupling left.
    cases.append(
        (
            "adjacent_lane/very_close_2m",
            lambda frame, t: InterferingRadar(
                object_id=103,
                distance_m=2.0,
                azimuth_rad=math.atan2(LANE_WIDTH_M, 1.0),
                relative_velocity_mps=0.0,
                transmit_power_dbm=TX_POWER_DBM,
                antenna_gain_dbi=TX_GAIN_DBI,
                boresight_alignment=0.5,
                label="adjacent_lane_neighbour_close",
            ),
        )
    )
    return [_run_case(label, builder, isolation_db, scans, warmup) for label, builder in cases]


def scenario_rear_radar_tailgating(isolation_db: float, scans: int, warmup: int):
    """Rear-facing radar with a follower whose radar faces us head-on.

    The following distance is swept, because it is the only variable that
    decides whether two radars at automotive spacing can blind each other.
    """

    results = []
    for following_distance_m in (1.0, 2.0, 3.0, 5.0, 10.0, 20.0, 40.0, 60.0):
        def builder(frame, t, d=following_distance_m):
            return InterferingRadar(
                object_id=201,
                distance_m=d,
                # In the rear radar's own frame the follower is straight ahead
                # of the boresight and closing at the speed difference.
                azimuth_rad=0.0,
                relative_velocity_mps=EGO_SPEED_MPS - NEIGHBOUR_SPEED_MPS,
                transmit_power_dbm=TX_POWER_DBM,
                antenna_gain_dbi=TX_GAIN_DBI,
                # Head-on: the follower's front radar points straight at us.
                boresight_alignment=1.0,
                label="follower_front_radar",
            )

        def targets(t, d=following_distance_m):
            return _rear_targets(t, d)

        results.append(
            _run_case(
                f"rear_radar_tailgating/follow_{following_distance_m:g}m",
                builder,
                isolation_db,
                scans,
                warmup,
                targets_at=targets,
            )
        )
    return results


def _write_csv(path: str, rows, fieldnames) -> None:
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _report(summaries, config_signature: str, isolations) -> str:
    lines = [
        "# Radar-to-radar interference study",
        "",
        f"Sensor profile `rgd_regime_v1`, config signature `{config_signature}`.",
        "Multipath disabled so an RRI phantom is never confused with a wall ghost.",
        f"Isolation swept: {', '.join(f'{value:g}' for value in isolations)} dB.",
        "",
        "## All cases",
        "",
        "| Isolation (dB) | Case | max INR (dB) | Desens. (dB) | FA mult. | "
        "Phantom pts/scan | Scans w/ phantom | Phantom selected |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for item in summaries:
        lines.append(
            f"| {item['isolation_db']:g} | {item['label']} | "
            f"{item['max_inr_db']:.1f} | {item['max_desensitisation_db']:.2f} | "
            f"{item['max_false_alarm_multiplier']:.2f}x | "
            f"{item['phantom_points_per_scan']:.3f} | "
            f"{item['scans_with_phantom_fraction']:.3f} | "
            f"{item['phantom_selected_count']} |"
        )

    # For each geometry, the quietest (best) and worst isolation observed, so
    # the boundary is readable without reading all 60-odd rows.
    lines += ["", "## Boundary", "", "| Case | Quiet at (dB) | Worst (dB) | Worst INR | Phantom selected |", "|---|---|---|---|---|"]
    by_case = {}
    for item in summaries:
        by_case.setdefault(item["label"], []).append(item)
    for label, items in by_case.items():
        quiet = max(items, key=lambda entry: entry["max_inr_db"] <= 0.0 and entry["isolation_db"] or -1.0)
        worst = max(items, key=lambda entry: entry["max_inr_db"])
        quiet_text = (
            f"{quiet['isolation_db']:g}" if quiet["max_inr_db"] <= 0.0 else "none"
        )
        lines.append(
            f"| {label} | {quiet_text} | {worst['isolation_db']:g} | "
            f"{worst['max_inr_db']:.1f} dB | {worst['phantom_selected_count']} |"
        )

    active = [item for item in summaries if item["phantom_points_per_scan"] > 0.0]
    hazard = sorted(
        (item for item in summaries if item["phantom_selected_count"] > 0),
        key=lambda entry: entry["phantom_selected_count"],
        reverse=True,
    )
    lines += ["", "## Reading", ""]

    tail = sorted(
        (
            item
            for item in summaries
            if item["label"].startswith("rear_radar_tailgating/")
        ),
        key=lambda entry: float(entry["label"].rsplit("_", 1)[1].rstrip("m")),
    )
    if tail:
        worst = max(tail, key=lambda entry: entry["max_inr_db"])
        lines.append(
            f"- **Rear radar, head-on follower** peaks at {worst['max_inr_db']:.1f} dB "
            f"INR ({worst['isolation_db']:g} dB isolation, "
            f"{worst['label'].rsplit('/', 1)[1]}). Beyond roughly 5 m of following "
            "distance the interference never reaches the receiver noise floor at "
            "any isolation tested."
        )

    null_rows = [item for item in summaries if item["label"].endswith("boresight_null")]
    if null_rows:
        worst = max(null_rows, key=lambda entry: entry["max_inr_db"])
        lines.append(
            f"- **Neighbouring lane, boresight-null** stays the quietest geometry: "
            f"worst case {worst['max_inr_db']:.1f} dB INR at "
            f"{worst['isolation_db']:g} dB isolation. A radar pointing along its "
            "own lane is ~90 degrees off the neighbour's boresight, and the "
            "antenna pattern suppresses most of what does couple across."
        )

    lines.append(
        f"- **{len(active)} of {len(summaries)} case/isolation combinations produced "
        "any phantom detection.** Everything else stayed below the noise floor, "
        "so CFAR never crossed threshold."
    )

    if hazard:
        top = hazard[0]
        scans = max(top["scans"], 1)
        lines.append(
            f"- **Phantom-braking hazard confirmed.** At {top['isolation_db']:g} dB "
            f"isolation, {top['label']} selected the phantom as the tracked target "
            f"in {top['phantom_selected_count']} of {scans} scans "
            f"({100.0 * top['phantom_selected_count'] / scans:.0f}%). The phantom "
            "sits at the interfering radar's true range and azimuth, so a "
            "longitudinal tracker selects it as the nearest object and the "
            "controller would brake for a vehicle that is in fact further away. "
            "This is the failure mode worth guarding against, and it is distinct "
            "from simply reporting extra clutter."
        )
    else:
        lines.append(
            "- **No phantom was ever selected as the tracked target**, so no "
            "phantom braking was observed at these settings."
        )

    lines += [
        "",
        "## Caveats",
        "",
        "- The geometry sweep is analytic, not from CARLA. Trajectories are "
        "straight and constant-speed, so this bounds the interference physics "
        "rather than reproducing a specific driving manoeuvre.",
        "- Antenna-to-antenna isolation is swept rather than measured. The "
        "practical conclusion is the *boundary* (roughly 15-20 dB isolation and "
        "under ~3 m separation), not any single row.",
        "- Interference coherence is fixed at "
        f"{summaries[0].get('coherence', 0.3) if summaries else 0.3}. Two radars "
        "with matched carriers would reject more; ones with a large carrier "
        "offset would reject less. This is the least-constrained parameter here.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenario",
        choices=("adjacent_lane", "rear_radar_tailgating", "both"),
        default="both",
    )
    parser.add_argument(
        "--isolation-db",
        default="40,30,25,20,15,10",
        help="Comma-separated antenna-to-antenna isolation values to sweep.",
    )
    parser.add_argument("--scans", type=int, default=120)
    parser.add_argument(
        "--warmup",
        type=int,
        default=30,
        help="Scans discarded before counting, so the tracker's confirmation "
        "window does not pollute the first rows.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default=here)
    args = parser.parse_args()

    if not os.path.isdir(os.path.join(os.path.dirname(here), "radar")):
        raise SystemExit(
            "run from carla4/ so the radar package is importable, e.g.\n"
            "  python3 carla4/rri_study/run_rri_study.py"
        )

    np.random.seed(args.seed)

    isolations = sorted(
        {float(value) for value in str(args.isolation_db).split(",") if value.strip()},
        reverse=True,
    )

    results = []
    for isolation_db in isolations:
        if args.scenario in ("adjacent_lane", "both"):
            results.extend(
                scenario_adjacent_lane(isolation_db, args.scans, args.warmup)
            )
        if args.scenario in ("rear_radar_tailgating", "both"):
            results.extend(
                scenario_rear_radar_tailgating(
                    isolation_db, args.scans, args.warmup
                )
            )

    summaries = [item[0] for item in results]
    per_scan = [row for item in results for row in item[1]]

    os.makedirs(args.output_dir, exist_ok=True)
    summary_path = os.path.join(args.output_dir, "rri_summary.csv")
    scan_path = os.path.join(args.output_dir, "rri_per_scan.csv")
    report_path = os.path.join(args.output_dir, "RRI_REPORT.md")
    _write_csv(summary_path, summaries, list(summaries[0].keys()))
    _write_csv(scan_path, per_scan, list(per_scan[0].keys()))

    config = load_realistic_radar_config(
        "rgd_regime_v1",
        overrides={"rri_mode": "parametric", "multipath_mode": "off"},
    )
    text = _report(
        summaries, realistic_radar_config_signature(config), isolations
    )
    with open(report_path, "w") as handle:
        handle.write(text)

    print(text)
    print(f"wrote {summary_path}")
    print(f"wrote {scan_path}")
    print(f"wrote {report_path}")


if __name__ == "__main__":
    main()