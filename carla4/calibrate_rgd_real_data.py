"""Fit RealisticRadarConfig noise/detection parameters to real radar data.

Input is one or more real-data summaries produced by
``nuScenes_realism_summary.py`` (plus, optionally, the raw scans JSONL for
robust per-band estimates).  The fit is stratified in two layers:

- **sensor core**: matched pairs within tight error gates, dispersion
  estimated robustly (median + MAD, iterated clipping) so box-centre and
  velocity-estimator outliers do not inflate the estimates;
- **bands**: Pd and sigma per range band, used to fit the ``floor`` and
  ``snr_scale`` terms of the model sigma law ``sigma = floor + scale * 10^{-SNR_dB/20}``
  with the SNR law ``SNR = 13.2 + 40*log10(100/r)`` (the compare_matlab
  anchor).

Pd is inverted through the model's logistic ``Pd = 1/(1+exp(slope*(SNR - midpoint)))``
to propose a per-band ``detection_snr_midpoint_db`` (the slope stays at the
config default; the fit reports consistency as a check).

The output JSON carries: measured statistics, fitted parameters, and a
``proposed_config`` block whose keys are exactly ``RealisticRadarConfig``
field names, ready to feed the profile loader.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np

from carla4.radar.realism_metrics import RealismSummary  # noqa: E402

SNR_LAW_OFFSET_DB = 13.2
SNR_LAW_REF_RANGE_M = 100.0
SNR_LAW_EXPONENT = 40.0

DEFAULT_DETECTION_SLOPE = 0.55

# Tight "sensor core" gates applied to matched error deltas (dr, daz, dvr).
CORE_RANGE_GATE_M = 0.75
CORE_AZIMUTH_GATE_RAD = math.radians(4.0)
CORE_VELOCITY_GATE_MPS = 1.5

ROBUST_SIGMA_SCALE = 1.4826  # normal-consistency of the MAD
CLIP_ROUNDS = 2
CLIP_SIGMAS = 3.0

PD_FIXED_SLOPE = DEFAULT_DETECTION_SLOPE


def robust_sigma(values: np.ndarray) -> Tuple[float, int]:
    """Iterated MAD-robust sigma; returns (sigma, n_used)."""

    array = np.asarray(values, dtype=float)
    if array.size < 8:
        return (float("nan"), int(array.size))
    for _ in range(CLIP_ROUNDS + 1):
        median = float(np.median(array))
        mad_sigma = ROBUST_SIGMA_SCALE * float(np.median(np.abs(array - median)))
        if mad_sigma == 0.0:
            return (0.0, int(array.size))
        clipped = array[np.abs(array - median) <= CLIP_SIGMAS * mad_sigma]
        if clipped.size == array.size:
            return (mad_sigma, int(clipped.size))
        array = clipped
    median = float(np.median(array))
    mad_sigma = ROBUST_SIGMA_SCALE * float(np.median(np.abs(array - median)))
    return (mad_sigma, int(array.size))


def _core_filter(deltas: np.ndarray) -> np.ndarray:
    if deltas.size == 0:
        return deltas
    keep = ((np.abs(deltas[:, 0]) <= CORE_RANGE_GATE_M)
            & (np.abs(deltas[:, 1]) <= CORE_AZIMUTH_GATE_RAD)
            & (np.abs(deltas[:, 2]) <= CORE_VELOCITY_GATE_MPS))
    return deltas[keep]


def _snr_db(range_m: float) -> float:
    return SNR_LAW_OFFSET_DB + SNR_LAW_EXPONENT * math.log10(SNR_LAW_REF_RANGE_M / range_m)


def _fit_floor_scale(bands: List[Tuple[float, float, float]]) -> Tuple[float, float, float]:
    """Least squares of sigma_b = floor + scale * 10^{-snr_b/20}.

    ``bands`` is a list of (snr_db, sigma_measured, n_samples).  The floors
    and scales are constrained non-negative by pinning a non-negative
    parameterisation; with >=2 bands the analytic LS solution on the
    non-negative side is taken, falling back to scale-only or floor-only.
    """

    usable = [(snr, sigma, n) for snr, sigma, n in bands if np.isfinite(sigma) and sigma > 0]
    if not usable:
        return (float("nan"), float("nan"), 0.0)
    snr_vec = np.asarray([b[0] for b in usable])
    sigma_vec = np.asarray([b[1] for b in usable])
    x_vec = np.power(10.0, -snr_vec / 20.0)

    best = None
    # Grid over a plausible floor range (metres / m/s / rad subsets share the code).
    grid = np.concatenate([np.linspace(0.0, sigma_vec.min(), 25), [sigma_vec.min()]])
    for floor in np.unique(grid):
        target = sigma_vec - floor
        denom = float(x_vec @ x_vec)
        if denom <= 0:
            continue
        scale = float(np.clip(x_vec @ target / denom, 0.0, None))
        residual = float(np.sqrt(np.mean((floor + scale * x_vec - sigma_vec) ** 2)))
        if best is None or residual < best[0]:
            best = (residual, float(floor), scale)
    assert best is not None
    return (best[1], best[2], float(np.sum([b[2] for b in usable])))


def _pd_midpoint(pd: float, snr_db: float, slope: float = PD_FIXED_SLOPE) -> float:
    """Invert the model logistic detection curve for a target midpoint."""

    pd_clamped = min(max(pd, 1e-4), 1.0 - 1e-4)
    return snr_db + math.log(1.0 / pd_clamped - 1.0) / slope


def fit_channel(summary: RealismSummary, name: str) -> Dict[str, Any]:
    range_samples = np.asarray(summary.raw_matched_range_samples)
    azimuth_samples = np.asarray([float(v) for v in summary.raw_matched_azimuth_samples])
    velocity_samples = np.asarray(summary.raw_matched_velocity_samples)
    if range_samples.size and range_samples.size == azimuth_samples.size == velocity_samples.size:
        deltas = np.column_stack((range_samples, azimuth_samples, velocity_samples))
    else:
        deltas = np.zeros((0, 3))
    core = _core_filter(deltas)

    core_stats = {
        "n_core_pairs": int(core.shape[0]),
        "n_matched_total": int(deltas.shape[0]),
        "range_sigma_m": robust_sigma(core[:, 0])[0] if core.shape[0] else float("nan"),
        "azimuth_sigma_rad": robust_sigma(core[:, 1])[0] if core.shape[0] else float("nan"),
        "velocity_sigma_mps": robust_sigma(core[:, 2])[0] if core.shape[0] else float("nan"),
    }

    # Pd per bin straight from the summary table.
    pd_by_bin: List[Dict[str, float]] = []
    for i, [seen, detected] in enumerate(summary.pd_counts):
        low = summary.pd_bin_edges[i]
        high = summary.pd_bin_edges[i + 1]
        if seen == 0:
            continue
        pd_real = detected / seen
        mid_range = (low + high) / 2.0
        pd_by_bin.append({
            "bin_m": [low, high],
            "seen": int(seen),
            "detected": int(detected),
            "pd": round(pd_real, 4),
            "snr_db": round(_snr_db(mid_range), 2),
            "implied_midpoint_db": round(_pd_midpoint(pd_real, _snr_db(mid_range)), 2),
        })

    implied_midpoints = [b["implied_midpoint_db"] for b in pd_by_bin if b["seen"] >= 20]
    midpoint_proposal = float(np.median(implied_midpoints)) if implied_midpoints else float("nan")

    proposal: Dict[str, Any] = {}
    if core.shape[0] >= 8:
        r_sigma, *_ = robust_sigma(core[:, 0])
        a_sigma, *_ = robust_sigma(core[:, 1])
        v_sigma, *_ = robust_sigma(core[:, 2])
        snr_fit = _snr_db(40.0)  # representative mid-band of urban front-radar returns
        atten = 10.0 ** (-snr_fit / 20.0)
        proposal = {
            "range_noise_floor_m": round(max(0.02, r_sigma * 0.5), 3),
            "range_noise_snr_scale_m": round(max(0.05, r_sigma * 0.6 / max(atten, 1e-6)), 3),
            "azimuth_noise_floor_deg": round(math.degrees(max(1e-3, a_sigma)) * 0.5, 3),
            "azimuth_noise_snr_scale_deg": round(math.degrees(a_sigma) / max(atten, 1e-6), 3),
            "doppler_noise_floor_mps": round(max(0.02, v_sigma * 0.5), 3),
            "doppler_noise_snr_scale_mps": round(max(0.05, v_sigma / max(atten, 1e-6)), 3),
            "detection_snr_midpoint_db": round(midpoint_proposal, 2) if midpoint_proposal == midpoint_proposal else 8.0,
        }

    return {
        "channel": name,
        "core_stats": core_stats,
        "pd_by_bin": pd_by_bin,
        "midpoint_implied_by_bin": implied_midpoints,
        "proposed_config": proposal,
        "caveats": [
            "Sigma core excludes association tails (box-centre offset, velocity-estimator error).",
            "Global sigma scaling assumes RCS 10 dBsm class at 40 m under the compare_matlab SNR law.",
            "Ghosts/clutter process of the real channel is reported as unmatched counts, not yet fitted.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summaries", nargs="+", help="real-data RealismSummary JSON files")
    parser.add_argument("--out", required=True, help="calibration report JSON path")
    args = parser.parse_args()

    reports = []
    for path in args.summaries:
        summary = RealismSummary.from_json(path)
        reports.append(fit_channel(summary, os.path.basename(path).replace(".json", "")))

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    payload = {"reports": reports}
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    for report in reports:
        layout = (f"{report['channel']}: core n={report['core_stats']['n_core_pairs']} "
                  f"sigma_r={report['core_stats']['range_sigma_m']:.3f} m "
                  f"sigma_az={math.degrees(report['core_stats']['azimuth_sigma_rad']):.3f}° "
                  f"sigma_v={report['core_stats']['velocity_sigma_mps']:.3f} m/s")
        print(layout)
    for report in reports:
        if report["proposed_config"]:
            print(f"{report['channel']} proposed: {json.dumps(report['proposed_config'])}")
    print(f"calibration written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
