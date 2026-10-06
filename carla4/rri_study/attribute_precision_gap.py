"""Attribute the precision gap between our sensor model and MATLAB's.

The comparison in ``compare_matlab`` measures our radar's range error at 0.157 m
and MATLAB's at 0.010 m. That gap has two independent candidate causes, and they
need different fixes, so it is worth knowing which one dominates before
changing anything:

1. **Measurement noise floor.** ``range_noise_floor_m`` and friends are the error
   that survives at high SNR, where the ``*_snr_scale_*`` terms have decayed to
   nothing. MATLAB has no such floor -- its only error is quantisation onto the
   declared resolution grid.
2. **Extended-target point spread.** ``emit_extended_points`` expands each
   detection into ~8 points spread across the object's footprint, which adds
   geometric spread on top of measurement noise. MATLAB returns a single point
   per target and has no such spread.

This script measures each contribution by disabling one at a time, on the same
analytic scenario ``compare_matlab`` uses, and reports the error against truth.
It does not change any default: it only reads configurations.
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

from radar.realistic_core import (
    IdealRadarTarget,
    RealisticRadarModel,
    load_realistic_radar_config,
)

FPS = 10.0
PED_Y_M = 3.0
PED_X_NEAR_M = 12.0
PED_X_FAR_M = 25.0
PED_SPEED_MPS = 1.4
LATERAL_EXTENT_M = 0.4
SNR_DB = 49.5

# Cases: (label, overrides). "floors_zeroed" removes the high-SNR noise floor,
# leaving quantisation as the only error, which is MATLAB's regime.
CASES = {
    "baseline": {},
    "extended_points_off": {"emit_extended_points": False},
    "floors_zeroed": {
        "range_noise_floor_m": 0.0,
        "azimuth_noise_floor_deg": 0.0,
        "doppler_noise_floor_mps": 0.0,
        "snr_fluctuation_std_db": 0.0,
        "error_correlation": 0.0,
    },
    "floors_zeroed_and_points_off": {
        "emit_extended_points": False,
        "range_noise_floor_m": 0.0,
        "azimuth_noise_floor_deg": 0.0,
        "doppler_noise_floor_mps": 0.0,
        "snr_fluctuation_std_db": 0.0,
        "error_correlation": 0.0,
    },
    "floors_halved": {
        "range_noise_floor_m": 0.03,
        "azimuth_noise_floor_deg": 0.10,
        "doppler_noise_floor_mps": 0.02,
    },
    "floors_quartered": {
        "range_noise_floor_m": 0.015,
        "azimuth_noise_floor_deg": 0.05,
        "doppler_noise_floor_mps": 0.01,
    },
}


def _truth_positions(frames):
    span = PED_X_FAR_M - PED_X_NEAR_M
    half = span / PED_SPEED_MPS
    period = 2.0 * half
    out = []
    for frame in range(frames):
        t = (frame - 1) / FPS
        phase = math.fmod(t, period)
        x = (
            PED_X_NEAR_M + PED_SPEED_MPS * phase
            if phase <= half
            else PED_X_FAR_M - PED_SPEED_MPS * (phase - half)
        )
        vx = PED_SPEED_MPS if phase <= half else -PED_SPEED_MPS
        radial = -(x * vx) / math.hypot(x, PED_Y_M)
        out.append((t, x, vx, radial))
    return out


def _measure(overrides, truth, seed=42):
    # latency_scans=0 so a scan's points belong to that scan's truth. The
    # default delay hands back an older scan, and scoring it against the
    # current target produces metre-scale nonsense. Clutter is filtered out
    # rather than gated, since a clutter point at a random range would dominate
    # an ungated RMS.
    merged = {"multipath_mode": "off", "latency_scans": 0, **overrides}
    config = load_realistic_radar_config("rgd_regime_v1", overrides=merged)
    model = RealisticRadarModel(config, seed=seed)

    errs = {"range": [], "azimuth": [], "velocity": []}
    for t, x, vx, radial in truth:
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
        model.step([target], timestamp_s=t)
        _, points = model.latest_points()
        for point in points:
            if point.source != "direct":
                continue
            errs["range"].append(point.distance_m - math.hypot(x, PED_Y_M))
            errs["azimuth"].append(point.azimuth_rad - math.atan2(PED_Y_M, x))
            errs["velocity"].append(point.relative_velocity_mps - radial)

    out = {"latency_scans": 0}
    for key, values in errs.items():
        if values:
            out[f"{key}_rms"] = math.sqrt(sum(v * v for v in values) / len(values))
            out[f"{key}_n"] = len(values)
        else:
            out[f"{key}_rms"] = float("nan")
            out[f"{key}_n"] = 0
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    truth = _truth_positions(args.frames)
    rows = []
    for label, overrides in CASES.items():
        result = _measure(overrides, truth, args.seed)
        result["case"] = label
        result["points_per_scan"] = result["range_n"] / args.frames
        rows.append(result)

    print(
        f"{'case':30s} {'pts/scan':>9s} {'range RMS':>10s} "
        f"{'azim RMS':>10s} {'dopp RMS':>10s}"
    )
    print(f"{'':30s} {'(m)':>9s} {'(m)':>10s} {'(deg)':>10s} {'(m/s)':>10s}")
    for row in rows:
        print(
            f"{row['case']:30s} {row['points_per_scan']:9.2f} "
            f"{row['range_rms']:10.4f} {row['azimuth_rms']:10.4f} "
            f"{row['velocity_rms']:10.4f}"
        )

    reference = next(r for r in rows if r["case"] == "baseline")
    closest = next(r for r in rows if r["case"] == "floors_zeroed_and_points_off")
    print(
        f"\nMATLAB reference for comparison: range 0.0100 m, "
        f"azimuth 0.1779 deg, radial vel 0.0044 m/s, 1 point per target."
    )
    print(
        f"\nBaseline -> closest MATLAB-like configuration: range "
        f"{reference['range_rms']:.4f} -> {closest['range_rms']:.4f} m, "
        f"azimuth {reference['azimuth_rms']:.4f} -> {closest['azimuth_rms']:.4f} deg, "
        f"Doppler {reference['velocity_rms']:.4f} -> {closest['velocity_rms']:.4f} m/s"
    )


if __name__ == "__main__":
    main()
