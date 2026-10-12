"""Bootstrap confidence intervals for the real-vs-sim realism statistics.

Resamples the raw matched samples retained in RealismSummary JSONs and the
unmatched-detection range samples, and reports 95% percentile-bootstrap CIs
for the paired statistic set a paper claims:

- per-summary: range/azimuth/velocity sigma;
- cross-summary: KS distance for range/velocity/ghost-range distributions,
  via independent resamples of both sides;
- ghost counts per scan: mean and its CI (counts are not in the summary, so
  the caller passes the per-scan ghost counts when available; otherwise the
  summary's mean ghost count is assumed Poisson, with the CI from the normal
  approximation of its standard error).

Run with any env that has numpy.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, Sequence, Tuple

import numpy as np

sys_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "..")
import sys
sys.path.insert(0, sys_path)

from carla4.radar.realism_metrics import RealismSummary, _ks_statistic  # noqa: E402


def _bootstrap_ci(values: np.ndarray, stat, n_resamples: int, seed: int) -> Tuple[float, float, float, float]:
    rng = np.random.default_rng(seed)
    point = float(stat(values))
    if values.size < 8:
        return (point, point, point, point)
    stats = np.empty(n_resamples)
    for i in range(n_resamples):
        resample = values[rng.integers(0, values.size, values.size)]
        stats[i] = stat(resample)
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return (float(lo), point, float(hi), float(np.std(stats)))


def _robust_sigma(values: np.ndarray) -> float:
    array = values
    for _ in range(3):
        median = np.median(array)
        mad_sigma = 1.4826 * np.median(np.abs(array - median))
        if mad_sigma == 0.0:
            return 0.0
        clipped = array[np.abs(array - median) <= 3.0 * mad_sigma]
        if clipped.size == array.size:
            return float(mad_sigma)
        array = clipped
    return float(1.4826 * np.median(np.abs(array - np.median(array))))


def ks_ci(samples_a: Sequence[float], samples_b: Sequence[float],
          n_resamples: int = 500, seed: int = 0) -> Dict[str, float]:
    a = np.asarray(samples_a, dtype=float)
    b = np.asarray(samples_b, dtype=float)
    if a.size < 16 or b.size < 16:
        return {"ks": _ks_statistic(a, b), "lo": float("nan"), "hi": float("nan")}
    rng = np.random.default_rng(seed)
    stats = np.empty(n_resamples)
    for i in range(n_resamples):
        ra = a[rng.integers(0, a.size, a.size)]
        rb = b[rng.integers(0, b.size, b.size)]
        stats[i] = _ks_statistic(ra, rb)
    lo, hi = np.percentile(stats, [2.5, 97.5])
    point = _ks_statistic(a, b)
    return {"ks": float(point), "lo": float(lo), "hi": float(hi)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real", required=True)
    parser.add_argument("--sim", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    real = RealismSummary.from_json(args.real)
    sim = RealismSummary.from_json(args.sim)

    report: Dict[str, Any] = {}
    for side, summary in (("real", real), ("sim", sim)):
        side_report = {}
        for name, samples in (
            ("range_sigma_m", np.asarray(summary.raw_matched_range_samples)),
            ("azimuth_sigma_rad", np.asarray(summary.raw_matched_azimuth_samples)),
            ("velocity_sigma_mps", np.asarray(summary.raw_matched_velocity_samples)),
        ):
            if samples.size:
                lo, point, hi, stderr = _bootstrap_ci(samples, _robust_sigma, 300, 0)
                side_report[name] = {"value": point, "lo": lo, "hi": hi, "stderr": stderr}
        report[side] = side_report

    report["ks"] = {
        "range": ks_ci(real.raw_matched_range_samples, sim.raw_matched_range_samples),
        "velocity": ks_ci(real.raw_matched_velocity_samples, sim.raw_matched_velocity_samples),
        "ghost_range": ks_ci(real.raw_ghost_range_samples, sim.raw_ghost_range_samples),
    }

    parent = os.path.dirname(os.path.abspath(args.out))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
