"""Analytic ground-truth scenario for the Python-vs-MATLAB radar comparison.

This is the single source of truth both radar models consume.  It reproduces
the verified RGD-regime collection geometry (see
``collect_carla_radar_ghosts.py``) without needing CARLA:

* stationary ego, sensor at the origin (x forward, y right, z ignored),
* a pedestrian walking away from and back toward the sensor at 1.4 m/s,
* a planar guardrail parallel to the walk direction,
* the analytic type-1 order-2 multipath ghost: the mirror image of the
  pedestrian across the guardrail plane, with a 2nd-order bounce loss
  (5 dB) plus the guardrail material loss (1 dB).

Coordinate and sign conventions (identical on both model sides):
* x forward, y RIGHT (CARLA convention; the RGD exporter flips y on export),
* azimuth measured from boresight, positive to the right,
* radial velocity positive = CLOSING (Python model contract).  MATLAB reports
  opening-positive range rate; ``matlab/run_matlab_radar.m`` flips the sign.

SNR truth model: ``snr_db(r) = SNR_ANCHOR_DB + 40*log10(REFERENCE_RANGE_M / r)``
is the SNR that MATLAB's ``radarDataGenerator`` assigns a ReferenceRCS target
at range r for DetectionProbability=0.9 / FalseAlarmRate=1e-6 (Albersheim).
Feeding the same SNR law to the Python model removes amplitude-domain bias
from the comparison, so the measured differences are error-model differences.

Output: ``truth_scenario.csv`` with one row per entity per frame:
frame, time_s, entity, entity_type, object_id, x_m, y_m, vx_mps, vy_mps,
range_m, azimuth_deg, radial_velocity_mps, snr_db, rcs_dbsm
"""

from __future__ import annotations

import argparse
import csv
import math
import os

import numpy as np

# Comparison envelope (matches profile rgd_regime_v1 + RGD 153 m range gate).
FPS = 10.0
DURATION_S = 38.5
MAX_RANGE_M = 153.0
MAX_UNAMBIGUOUS_DOPPLER_MPS = 44.3

# SNR anchor: Albersheim SNR (dB) for Pd=0.9 at Pfa=1e-6 at the reference range.
SNR_ANCHOR_DB = 13.2
REFERENCE_RANGE_M = 100.0

# Scenario geometry (sensor frame, metres).
PED_Y_M = 3.0            # lateral offset of the walk line (right of boresight)
WALL_Y_M = 4.0           # guardrail plane, parallel to x
PED_X_NEAR_M = 12.0      # nearest walk point
PED_X_FAR_M = 25.0       # farthest walk point
PED_SPEED_MPS = 1.4      # matches the RGD pedestrian regime

# Ghost physics (identical priors to realistic_core / profiles).
SECOND_ORDER_BOUNCE_LOSS_DB = 5.0
GUARDRAIL_MATERIAL_LOSS_DB = 1.0
GHOST_SNR_LOSS_DB = SECOND_ORDER_BOUNCE_LOSS_DB + GUARDRAIL_MATERIAL_LOSS_DB

# MATLAB-only inputs (radarDataGenerator computes SNR from RCS + range).
PED_RCS_DBSM = 10.0      # = ReferenceRCS used in the MATLAB script
GHOST_RCS_DBSM = PED_RCS_DBSM - GHOST_SNR_LOSS_DB

DIRECT_OBJECT_ID = 1
GHOST_OBJECT_ID = 2


def _walk_position(t_s: float) -> tuple[float, float]:
    """Pedestrian x position and velocity: triangular 12->25->12 m cycles."""

    span = PED_X_FAR_M - PED_X_NEAR_M
    half_period = span / PED_SPEED_MPS
    period = 2.0 * half_period
    phase = math.fmod(t_s, period)
    if phase <= half_period:
        x = PED_X_NEAR_M + PED_SPEED_MPS * phase
        vx = PED_SPEED_MPS
    else:
        x = PED_X_FAR_M - PED_SPEED_MPS * (phase - half_period)
        vx = -PED_SPEED_MPS
    return x, vx


def _radial_velocity(x: float, y: float, vx: float, vy: float) -> float:
    """Closing-positive radial velocity of a point moving with (vx, vy)."""

    r = math.hypot(x, y)
    if r < 1e-9:
        return 0.0
    return -((x * vx + y * vy) / r)


def _truth_snr(x: float, y: float, loss_db: float) -> float:
    r = max(math.hypot(x, y), 1e-6)
    return SNR_ANCHOR_DB + 40.0 * math.log10(REFERENCE_RANGE_M / r) - loss_db


def generate_rows() -> list[dict]:
    rows = []
    n_frames = int(round(DURATION_S * FPS))
    for frame in range(1, n_frames + 1):
        t = (frame - 1) / FPS
        x, vx = _walk_position(t)
        y, vy = PED_Y_M, 0.0
        r = math.hypot(x, y)
        az = math.degrees(math.atan2(y, x))

        rows.append(
            {
                "frame": frame,
                "time_s": round(t, 6),
                "entity": "pedestrian",
                "entity_type": "direct",
                "object_id": DIRECT_OBJECT_ID,
                "x_m": round(x, 6),
                "y_m": round(y, 6),
                "vx_mps": round(vx, 6),
                "vy_mps": round(vy, 6),
                "range_m": round(r, 6),
                "azimuth_deg": round(az, 6),
                "radial_velocity_mps": round(_radial_velocity(x, y, vx, vy), 6),
                "snr_db": round(_truth_snr(x, y, 0.0), 6),
                "rcs_dbsm": PED_RCS_DBSM,
            }
        )

        # Type-1 order-2 ghost: mirror image across the guardrail plane
        # y = WALL_Y_M.  Lateral motion flips sign under reflection; the
        # longitudinal component does not.
        gx, gy = x, 2.0 * WALL_Y_M - y
        rows.append(
            {
                "frame": frame,
                "time_s": round(t, 6),
                "entity": "mirror_type1_order2",
                "entity_type": "ghost",
                "object_id": GHOST_OBJECT_ID,
                "x_m": round(gx, 6),
                "y_m": round(gy, 6),
                "vx_mps": round(vx, 6),
                "vy_mps": round(-vy, 6),
                "range_m": round(math.hypot(gx, gy), 6),
                "azimuth_deg": round(math.degrees(math.atan2(gy, gx)), 6),
                "radial_velocity_mps": round(
                    _radial_velocity(gx, gy, vx, -vy), 6
                ),
                "snr_db": round(_truth_snr(gx, gy, GHOST_SNR_LOSS_DB), 6),
                "rcs_dbsm": GHOST_RCS_DBSM,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default=os.path.dirname(os.path.abspath(__file__)),
        help="Directory for truth_scenario.csv (default: alongside this file).",
    )
    args = parser.parse_args()

    rows = generate_rows()
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "truth_scenario.csv")

    fieldnames = [
        "frame",
        "time_s",
        "entity",
        "entity_type",
        "object_id",
        "x_m",
        "y_m",
        "vx_mps",
        "vy_mps",
        "range_m",
        "azimuth_deg",
        "radial_velocity_mps",
        "snr_db",
        "rcs_dbsm",
    ]
    with open(out_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    n_frames = int(round(DURATION_S * FPS))
    ranges = np.array([row["range_m"] for row in rows if row["entity"] == "pedestrian"])
    print(f"wrote {out_path}")
    print(
        f"frames: {n_frames}  entities/frame: 2  "
        f"pedestrian range: {ranges.min():.1f}-{ranges.max():.1f} m  "
        f"ghost SNR loss: {GHOST_SNR_LOSS_DB:.1f} dB"
    )


if __name__ == "__main__":
    main()
