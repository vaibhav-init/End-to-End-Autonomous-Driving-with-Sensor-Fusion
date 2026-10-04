"""Run the repository's realistic radar model on the shared truth scenario.

Consumes ``truth_scenario.csv`` (from ``generate_truth_scenario.py``) and
steps :class:`radar.realistic_core.RealisticRadarModel` once per frame:

* the pedestrian enters as an ``IdealRadarTarget`` (direct),
* the analytic mirror ghost enters via ``multipath_targets`` -- exactly the
  geometry-mode contract used by the CARLA front end,
* the profile is ``rgd_regime_v1`` with ``max_range_m`` overridden to 153 m so
  the envelope matches MATLAB's configuration in ``run_matlab_radar.m``.

Output: ``python_detections.csv`` with one row per emitted point:
sensor, frame, source_scan_index, time_s, point_type, object_id,
x_m, y_m, range_m, azimuth_deg, radial_velocity_mps, snr_db

``source_scan_index`` is the scan whose ideal targets produced the point; the
model delays delivery by ``latency_scans``, so error statistics are computed
against the truth of the SOURCE frame, not the delivery frame.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys

# Allow running from the repository root: python3 carla4/compare_matlab/...
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_CARLA_DIR = os.path.join(_REPO_ROOT, "carla4")
if _CARLA_DIR not in sys.path:
    sys.path.insert(0, _CARLA_DIR)

from radar.realistic_core import (  # noqa: E402
    IdealRadarTarget,
    RealisticRadarModel,
    load_realistic_radar_config,
)

PROFILE = "rgd_regime_v1"
MAX_RANGE_M = 153.0  # RGD range gate; rgd_regime_v1 keeps the 100 m default

GHOST_BOUNCE_TYPE = "type1"
GHOST_BOUNCE_ORDER = 2


def _read_truth(path: str) -> list[dict]:
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def run(truth_path: str, output_path: str, seed: int) -> None:
    truth = _read_truth(truth_path)
    frames: dict[int, list[dict]] = {}
    for row in truth:
        frames.setdefault(int(row["frame"]), []).append(row)

    config = load_realistic_radar_config(PROFILE, max_range_m=MAX_RANGE_M)
    model = RealisticRadarModel(config, seed=seed)

    fieldnames = [
        "sensor",
        "frame",
        "source_scan_index",
        "time_s",
        "point_type",
        "object_id",
        "x_m",
        "y_m",
        "range_m",
        "azimuth_deg",
        "radial_velocity_mps",
        "snr_db",
    ]

    n_rows = 0
    with open(output_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        for frame in sorted(frames):
            rows = frames[frame]
            timestamp_s = float(rows[0]["time_s"])

            ideal_targets = []
            multipath_targets = []
            for row in rows:
                x, y = float(row["x_m"]), float(row["y_m"])
                target = IdealRadarTarget(
                    object_id=int(row["object_id"]),
                    # Both direct and ghost returns belong to the pedestrian
                    # class (RGD ghosts are multipath returns OF a road user).
                    semantic_tag=12,
                    distance_m=math.hypot(x, y),
                    azimuth_rad=math.atan2(y, x),
                    relative_velocity_mps=float(row["radial_velocity_mps"]),
                    snr_db=float(row["snr_db"]),
                    point_count=1,
                    lateral_extent_m=0.4,
                    source="direct" if row["entity_type"] == "direct" else "ghost",
                    parent_object_id=1 if row["entity_type"] == "ghost" else 0,
                    bounce_type=(
                        GHOST_BOUNCE_TYPE if row["entity_type"] == "ghost" else "direct"
                    ),
                    bounce_order=GHOST_BOUNCE_ORDER if row["entity_type"] == "ghost" else 1,
                    velocity_xy_mps=(
                        float(row["vx_mps"]),
                        float(row["vy_mps"]),
                    ),
                )
                if row["entity_type"] == "direct":
                    ideal_targets.append(target)
                else:
                    multipath_targets.append(target)

            model.step(
                ideal_targets,
                timestamp_s=timestamp_s,
                multipath_targets=multipath_targets,
            )
            _, points = model.latest_points()
            diagnostics = model.diagnostics()
            source_scan = int(diagnostics.get("delivered_source_scan_index") or frame)

            for point in points:
                az_deg = math.degrees(point.azimuth_rad)
                writer.writerow(
                    {
                        "sensor": f"python_{PROFILE}",
                        "frame": frame,
                        "source_scan_index": source_scan,
                        "time_s": round(timestamp_s, 6),
                        "point_type": point.source,
                        "object_id": point.truth_object_id,
                        "x_m": round(point.distance_m * math.cos(point.azimuth_rad), 6),
                        "y_m": round(point.distance_m * math.sin(point.azimuth_rad), 6),
                        "range_m": round(point.distance_m, 6),
                        "azimuth_deg": round(az_deg, 6),
                        "radial_velocity_mps": round(point.relative_velocity_mps, 6),
                        "snr_db": round(point.snr_db, 6),
                    }
                )
                n_rows += 1

    print(f"wrote {output_path}  ({n_rows} points over {len(frames)} frames)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    here = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument("--truth", default=os.path.join(here, "truth_scenario.csv"))
    parser.add_argument("--output", default=os.path.join(here, "python_detections.csv"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not os.path.exists(args.truth):
        raise SystemExit(
            f"truth file not found: {args.truth}\n"
            "run generate_truth_scenario.py first"
        )
    run(args.truth, args.output, args.seed)


if __name__ == "__main__":
    main()
