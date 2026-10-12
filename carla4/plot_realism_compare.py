"""Render real-vs-sim comparisons from RealismSummary JSONs.

Produces one PNG next to the output compare JSON (or to the given path):
- matched-pair error histograms (range, azimuth, velocity) for real vs sim;
- Pd-vs-range table bars;
- detections-per-scan and unmatched-detection rates as reference bars.

Run with the PCLA env (needs matplotlib).
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List, Sequence, Tuple

import numpy as np

sys_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "..")
import sys
sys.path.insert(0, sys_path)

from carla4.radar.realism_metrics import RealismSummary  # noqa: E402


def _paired_hist(ax, real: Sequence[float], sim: Sequence[float], label: str,
                 bins: int = 41) -> None:
    ax.hist(np.asarray(real, dtype=float), bins=bins, alpha=0.55, density=True,
            label=f"real n={len(real)}")
    ax.hist(np.asarray(sim, dtype=float), bins=bins, alpha=0.55, density=True,
            label=f"sim n={len(sim)}")
    ax.set_xlabel(label)
    ax.set_ylabel("probability density")
    ax.legend()


def plot_reports(real_path: str, sim_path: str, out_path: str) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    real = RealismSummary.from_json(real_path)
    sim = RealismSummary.from_json(sim_path)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle("Real (nuScenes ARS408) vs simulation (rgd_regime_v1, calibrated)")

    _paired_hist(axes[0, 0], real.raw_matched_range_samples, sim.raw_matched_range_samples,
                 "range error det - GT centre [m]")
    _paired_hist(axes[0, 1], real.raw_matched_azimuth_samples, sim.raw_matched_azimuth_samples,
                 "azimuth error det - GT centre [rad]")
    _paired_hist(axes[0, 2], real.raw_matched_velocity_samples, sim.raw_matched_velocity_samples,
                 "radial velocity error [m/s]")

    ax = axes[1, 0]
    width = 0.38
    edges = real.pd_bin_edges
    centers = [(low + high) / 2.0 for low, high in zip(edges[:-1], edges[1:])]
    real_pd = [det / seen if seen else float("nan") for seen, det in real.pd_counts]
    sim_pd = [det / seen if seen else float("nan") for seen, det in sim.pd_counts]
    ax.bar(np.array(centers) - width / 2, real_pd, width, label="real")
    ax.bar(np.array(centers) + width / 2, sim_pd, width, label="sim")
    ax.set_xlabel("GT range bin centre [m]")
    ax.set_ylabel("P(resolve target in scan)")
    ax.set_title("detection probability vs range")
    ax.legend()

    ax = axes[1, 1]
    labels = ["dets/scan", "targets/scan", "ghosts/scan"]
    real_vals = [real.detections_per_scan_mean, real.targets_per_scan_mean,
                 real.ghosts_per_scan_mean]
    sim_vals = [sim.detections_per_scan_mean, sim.targets_per_scan_mean,
                sim.ghosts_per_scan_mean]
    x = np.arange(len(labels))
    ax.bar(x - width / 2, real_vals, width, label="real")
    ax.bar(x + width / 2, sim_vals, width, label="sim")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title("scan composition (scenario density differs)")
    ax.legend()

    ax = axes[1, 2]
    ax.axis("off")
    verdict_lines = [
        ("range sigma r/s", f"{real.range_error_std_m:.3f} / {sim.range_error_std_m:.3f} m"),
        ("velocity sigma r/s", f"{real.velocity_error_std_mps:.3f} / {sim.velocity_error_std_mps:.3f} m/s"),
        ("ghost fraction r/s", f"{real.ghost_fraction:.2f} / {sim.ghost_fraction:.2f}"),
        ("scans r/s", f"{real.n_scans} / {sim.n_scans}"),
    ]
    table = "\n".join(f"{k:22s}{v}" for k, v in verdict_lines)
    ax.text(0.02, 0.85, table, family="monospace", fontsize=12)

    fig.tight_layout()
    parent = os.path.dirname(os.path.abspath(out_path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real", required=True)
    parser.add_argument("--sim", required=True)
    parser.add_argument("--out", default="real_vs_sim_report.png")
    args = parser.parse_args()
    path = plot_reports(args.real, args.sim, args.out)
    print(f"figure -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
