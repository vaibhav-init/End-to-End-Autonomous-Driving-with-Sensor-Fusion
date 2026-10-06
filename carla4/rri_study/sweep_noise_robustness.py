"""Is the driving policy's input stable when sensor noise changes?

The precision attribution (``attribute_precision_gap.py``) showed that our
per-point error is dominated by extended-target point spread rather than
measurement noise. That matters here: the controller never sees a point. It
sees ``RadarModelOutput``, the filtered track the selector returns.

So the question worth answering is not "how noisy is the sensor" but "how much
does the sensor's *noise* move the number the policy acts on". This script
holds the scenario fixed, scales every noise floor by a factor, and reports two
things per scale:

* per-point error, which should grow roughly in proportion to the scale, and
* selected-track error, which should grow much more slowly if the tracker's
  own filtering is rejecting the noise.

The ratio between those two growth rates is the robustness claim. It is a claim
about this model rather than a cross-check against someone else's, and it does
not need CARLA, a GPU, or a trained policy -- the controller's input is exactly
what is being measured.

Run:
    python3 sweep_noise_robustness.py
    python3 sweep_noise_robustness.py --scales 0.25 0.5 1 2 4 8
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
)

FPS = 10.0
# The target has to sit inside the ego path or the selector correctly discards
# it as a crossing road user and there is no selected track to measure. A
# lead-vehicle geometry keeps the selector engaged, which is the point: the
# number being measured is the one a controller would act on.
PED_Y_M = 0.6
PED_X_NEAR_M = 12.0
PED_X_FAR_M = 25.0
PED_SPEED_MPS = 1.4
LATERAL_EXTENT_M = 0.4
SNR_DB = 49.5

# The floors that scale. SNR-scaled terms are left alone: they already decay to
# nothing at this SNR, so scaling them would change nothing measurable and
# would obscure the effect.
BASE_FLOORS = {
    "range_noise_floor_m": 0.06,
    "azimuth_noise_floor_deg": 0.20,
    "doppler_noise_floor_mps": 0.04,
}


def _truth_positions(frames):
    span = PED_X_FAR_M - PED_X_NEAR_M
    half = span / PED_SPEED_MPS
    period = 2.0 * half
    out = []
    for frame in range(frames):
        t = (frame - 1) / FPS
        phase = math.fmod(t, period)
        if phase <= half:
            x = PED_X_NEAR_M + PED_SPEED_MPS * phase
            vx = PED_SPEED_MPS
        else:
            x = PED_X_FAR_M - PED_SPEED_MPS * (phase - half)
            vx = -PED_SPEED_MPS
        radial = -(x * vx) / math.hypot(x, PED_Y_M)
        out.append((frame, t, x, vx, radial))
    return out


def _run(scale, truth, seed, extended_points):
    overrides = {
        "multipath_mode": "off",
        "emit_extended_points": bool(extended_points),
    }
    for name, value in BASE_FLOORS.items():
        overrides[name] = value * float(scale)
    config = load_realistic_radar_config("rgd_regime_v1", overrides=overrides)
    model = RealisticRadarModel(config, seed=seed)

    by_frame = {frame: (t, x, vx, radial) for frame, t, x, vx, radial in truth}
    point_err = defaultdict(list)
    track_err = {"range": [], "velocity": []}
    track_ids = []
    selected_count = 0
    scored = 0
    # Skip the tracker's confirmation window, otherwise early scans are scored
    # before a track exists and the selected fraction reads as zero.
    warmup = 40

    for frame, t, x, vx, radial in truth:
        if frame <= warmup:
            model.step(
                [
                    IdealRadarTarget(
                        object_id=1,
                        semantic_tag=12,
                        distance_m=math.hypot(x, PED_Y_M),
                        azimuth_rad=math.atan2(PED_Y_M, x),
                        relative_velocity_mps=radial,
                        snr_db=SNR_DB,
                        point_count=1,
                        lateral_extent_m=LATERAL_EXTENT_M,
                        velocity_xy_mps=(vx, 0.0),
                    )
                ],
                timestamp_s=t,
            )
            continue
        target = IdealRadarTarget(
            object_id=1,
            semantic_tag=12,
            distance_m=math.hypot(x, PED_Y_M),
            azimuth_rad=math.atan2(PED_Y_M, x),
            relative_velocity_mps=radial,
            snr_db=SNR_DB,
            point_count=1,
            lateral_extent_m=LATERAL_EXTENT_M,
            velocity_xy_mps=(vx, 0.0),
        )
        selected = model.step([target], timestamp_s=t)
        diagnostics = model.diagnostics()
        source_scan = diagnostics.get("delivered_source_scan_index")
        if source_scan is None:
            source_scan = frame
        if source_scan not in by_frame:
            continue
        _, truth_x, _, truth_radial = by_frame[source_scan]

        _, points = model.latest_points()
        for point in points:
            if point.source != "direct":
                continue
            point_err["range"].append(
                point.distance_m - math.hypot(truth_x, PED_Y_M)
            )
            point_err["azimuth"].append(
                point.azimuth_rad - math.atan2(PED_Y_M, truth_x)
            )
            point_err["velocity"].append(point.relative_velocity_mps - truth_radial)

        scored += 1
        if selected.track_id:
            selected_count += 1
            track_ids.append(selected.track_id)
            track_err["range"].append(
                selected.distance_m - math.hypot(truth_x, PED_Y_M)
            )
            track_err["velocity"].append(
                selected.relative_velocity_mps - truth_radial
            )

    def rms(values):
        if not values:
            return float("nan")
        return math.sqrt(sum(v * v for v in values) / len(values))

    switches = sum(
        1 for a, b in zip(track_ids, track_ids[1:]) if a != b
    )
    return {
        "scale": float(scale),
        "extended_points": bool(extended_points),
        "scans": scored,
        "selected_fraction": round(selected_count / max(scored, 1), 4),
        "track_id_switches": switches,
        "point_range_rms": round(rms(point_err["range"]), 5),
        "point_azimuth_rms": round(rms(point_err["azimuth"]), 5),
        "point_velocity_rms": round(rms(point_err["velocity"]), 5),
        "track_range_rms": round(rms(track_err["range"]), 5),
        "track_velocity_rms": round(rms(track_err["velocity"]), 5),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scales", type=float, nargs="+",
                        default=[0.25, 0.5, 1.0, 2.0, 4.0, 8.0])
    parser.add_argument("--output-dir",
                        default=os.path.dirname(os.path.abspath(__file__)))
    args = parser.parse_args()

    truth = _truth_positions(args.frames)
    rows = []
    for extended in (True, False):
        for scale in args.scales:
            rows.append(_run(scale, truth, args.seed, extended))

    os.makedirs(args.output_dir, exist_ok=True)
    path = os.path.join(args.output_dir, "noise_robustness.csv")
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {path}\n")

    for extended in (True, False):
        label = "multi-point echo (default)" if extended else "single point per target"
        print(f"=== {label} ===")
        print(
            f"{'floor scale':>12s} {'point rng':>10s} {'point v':>9s} "
            f"{'TRACK rng':>10s} {'TRACK v':>9s} {'selected':>9s} {'id switch':>10s}"
        )
        base = next(
            (r for r in rows if r["extended_points"] == extended and r["scale"] == 1.0),
            None,
        )
        for row in rows:
            if row["extended_points"] != extended:
                continue
            growth = ""
            if base and base["point_range_rms"] > 0:
                growth = (
                    f"  pts {row['point_range_rms'] / base['point_range_rms']:5.2f}x"
                    f"  track {row['track_range_rms'] / base['track_range_rms']:5.2f}x"
                )
            print(
                f"{row['scale']:12.2f} {row['point_range_rms']:10.4f} "
                f"{row['point_velocity_rms']:9.4f} {row['track_range_rms']:10.4f} "
                f"{row['track_velocity_rms']:9.4f} "
                f"{row['selected_fraction']:9.3f} {row['track_id_switches']:10d}{growth}"
            )
        print()


if __name__ == "__main__":
    main()