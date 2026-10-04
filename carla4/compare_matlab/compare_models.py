"""Statistically compare the Python and MATLAB radar models on shared truth.

Loads ``truth_scenario.csv`` plus the two detection CSVs and writes
``comparison_report.md`` (and optional PNG plots) measuring, per model:

1. point budget: mean points/frame by type (direct / ghost / clutter)
2. detection probability vs range for the direct pedestrian
3. geometry error vs analytic truth: range / azimuth / radial velocity
   (bias, RMS, std) with a two-sample KS test Python-vs-MATLAB
4. ghost fidelity: ghost points vs the analytic mirror image (range and
   azimuth delta) and ghost-minus-direct SNR (expected about -6 dB)
5. sanity: points outside the Doppler ambiguity window, detections outside
   the FOV/range envelope

Alignment: Python points are matched to the truth of ``source_scan_index``
(the model delays delivery by ``latency_scans``); MATLAB points carry their
own frame.  A point matches a truth entity within a range/azimuth gate.

Usage:
    python3 compare_models.py                       # both models
    python3 compare_models.py --open                # also open the report
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from collections import defaultdict

import numpy as np

try:
    from scipy import stats as scipy_stats
except ImportError:  # KS tests become unavailable; the rest still runs
    scipy_stats = None

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

HERE = os.path.dirname(os.path.abspath(__file__))

RANGE_GATE_M = 2.0
AZ_GATE_DEG = 5.0
PD_RANGE_BIN_M = 5.0
MAX_RANGE_M = 153.0
MAX_UNAMBIGUOUS_DOPPLER_MPS = 44.3
FOV_HALF_DEG = 70.0

ENTITY_BY_OBJECT_ID = {1: "pedestrian", 2: "mirror_type1_order2"}


def _read_csv(path):
    if not os.path.exists(path):
        return None
    with open(path, newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key in (
            "frame",
            "source_scan_index",
            "object_id",
            "x_m",
            "y_m",
            "range_m",
            "azimuth_deg",
            "radial_velocity_mps",
            "snr_db",
        ):
            try:
                row[key] = float(row[key])
            except (KeyError, ValueError):
                row[key] = float("nan")
    return rows


def _truth_by_frame(rows):
    frames = defaultdict(dict)
    for row in rows:
        frames[int(row["frame"])][row["entity"]] = row
    return frames


def _truth_frame_of(point):
    """Truth frame a detection belongs to: source scan when present, else own."""

    src = point.get("source_scan_index")
    if src is not None and src == src:  # `src != src` is the NaN test
        return int(src)
    return int(point["frame"])


def _match(points, truth_entity):
    """Match each point to the nearest truth row of one entity (in-place gate)."""

    matched = []
    for point in points:
        frame = _truth_frame_of(point)
        entity = truth_by_frame_all.get(frame, {}).get(truth_entity)
        if entity is None:
            continue
        d_range = point["range_m"] - entity["range_m"]
        d_az = point["azimuth_deg"] - entity["azimuth_deg"]
        if abs(d_range) <= RANGE_GATE_M and abs(d_az) <= AZ_GATE_DEG:
            matched.append((point, d_range, d_az, point["radial_velocity_mps"] - entity["radial_velocity_mps"]))
    return matched


def _frame_coverage(matched, key):
    """Fraction of truth frames carrying this entity that got >=1 matched point.

    Returns ``(coverage, n_truth_frames)``.  Uses the source scan frame so a
    latency-delayed python point counts against the frame it was actually
    generated from, matching how :func:`_match` aligns points to truth.
    """

    entity = ENTITY_BY_OBJECT_ID[1 if key == "direct" else 2]
    covered = {_truth_frame_of(point) for point, *_ in matched}
    n_truth = sum(1 for frame in truth_by_frame_all if entity in truth_by_frame_all[frame])
    if n_truth == 0:
        return float("nan"), 0
    return len(covered & set(truth_by_frame_all)) / n_truth, n_truth


def _summarize_errors(values, label):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return f"| {label} | n/a | n/a | n/a | n/a |"
    return (
        f"| {label} | {values.size} | {np.mean(values):+.4f} | "
        f"{np.sqrt(np.mean(values ** 2)):.4f} | {np.std(values):.4f} |"
    )


def _pd_vs_range(model_rows, label, lines):
    direct = [r for r in model_rows if r["point_type"] == "direct"]
    if not direct:
        lines.append(f"\n### Pd vs range — {label}\n\nNo direct points.")
        return
    bins = np.arange(0.0, MAX_RANGE_M + PD_RANGE_BIN_M, PD_RANGE_BIN_M)
    edges, counts, hits = [], [], []
    for lo in bins[:-1]:
        hi = lo + PD_RANGE_BIN_M
        frames_in_bin = {
            int(r["frame"])
            for r in truth_rows
            if lo <= r["range_m"] < hi and r["entity"] == "pedestrian"
        }
        if not frames_in_bin:
            continue
        detected = {
            int(r["frame"])
            for r in direct
            if lo <= r["range_m"] < hi
        }
        edges.append(0.5 * (lo + hi))
        counts.append(len(frames_in_bin))
        hits.append(len(detected & frames_in_bin) / len(frames_in_bin))
    lines.append(f"\n### Detection probability vs range — {label}\n")
    lines.append("| Range bin centre (m) | Truth frames | Pd |")
    lines.append("|---|---|---|")
    for e, c, h in zip(edges, counts, hits):
        lines.append(f"| {e:.1f} | {c} | {h:.3f} |")


def _model_report(name, rows, lines):
    lines.append(f"\n## Model: {name}\n")
    if rows is None:
        lines.append("No detection CSV found — run this model's script first.\n")
        return None

    by_type = defaultdict(list)
    for row in rows:
        by_type[row["point_type"]].append(row)

    n_frames = max(int(r["frame"]) for r in rows)
    lines.append("### Point budget\n")
    lines.append("| Type | Points | Points/frame |")
    lines.append("|---|---|---|")
    for ptype in ("direct", "ghost", "clutter"):
        pts = by_type.get(ptype, [])
        lines.append(f"| {ptype} | {len(pts)} | {len(pts) / n_frames:.2f} |")
    lines.append(f"| total | {len(rows)} | {len(rows) / n_frames:.2f} |")

    # Sanity: Doppler ambiguity and envelope violations.
    rr = np.array([r["radial_velocity_mps"] for r in rows], dtype=float)
    rr_ok = rr[np.isfinite(rr)]
    n_wrap = int(np.sum(np.abs(rr_ok) > MAX_UNAMBIGUOUS_DOPPLER_MPS + 1e-6))
    az = np.array([r["azimuth_deg"] for r in rows], dtype=float)
    n_fov = int(np.sum(np.abs(az) > FOV_HALF_DEG + 1e-6))
    rng = np.array([r["range_m"] for r in rows], dtype=float)
    n_env = int(np.sum((rng > MAX_RANGE_M + 1e-6) | (rng < 0.0)))
    lines.append(
        f"\nSanity: Doppler-wrap violations {n_wrap}, FOV violations {n_fov}, "
        f"range-envelope violations {n_env}."
    )

    direct_matched = _match(by_type.get("direct", []), "pedestrian")
    lines.append("\n### Geometry error vs analytic truth — direct pedestrian\n")
    lines.append("| Error | n | Bias | RMS | Std |")
    lines.append("|---|---|---|---|---|")
    lines.append(_summarize_errors([m[1] for m in direct_matched], "range (m)"))
    lines.append(_summarize_errors([m[2] for m in direct_matched], "azimuth (deg)"))
    lines.append(_summarize_errors([m[3] for m in direct_matched], "radial vel (m/s)"))

    ghost_matched = _match(by_type.get("ghost", []), "mirror_type1_order2")
    lines.append("\n### Ghost fidelity vs analytic mirror image\n")
    lines.append("| Error | n | Bias | RMS | Std |")
    lines.append("|---|---|---|---|---|")
    lines.append(_summarize_errors([m[1] for m in ghost_matched], "range (m)"))
    lines.append(_summarize_errors([m[2] for m in ghost_matched], "azimuth (deg)"))
    lines.append(_summarize_errors([m[3] for m in ghost_matched], "radial vel (m/s)"))

    direct_snr, ghost_snr = [], []
    for point, *_ in direct_matched:
        direct_snr.append(point["snr_db"])
    for point, *_ in ghost_matched:
        ghost_snr.append(point["snr_db"])
    if direct_snr and ghost_snr:
        diff = np.mean(ghost_snr) - np.mean(direct_snr)
        lines.append(
            f"\nMean SNR direct {np.mean(direct_snr):.2f} dB, ghost "
            f"{np.mean(ghost_snr):.2f} dB, difference {diff:+.2f} dB "
            "(truth loss: -6.0 dB)."
        )

    _pd_vs_range(rows, name, lines)
    return {
        "direct": direct_matched,
        "ghost": ghost_matched,
        "rows": rows,
        "n_frames": n_frames,
    }


def _ks_block(python_summary, matlab_summary, lines):
    if scipy_stats is None:
        lines.append(
            "\n> KS tests skipped: scipy not installed "
            "(pip install scipy). Bias/RMS/Std comparisons above still stand.\n"
        )
        return
    if python_summary is None or matlab_summary is None:
        return
    lines.append("\n## Python vs MATLAB distribution tests (two-sample KS)\n")
    lines.append("| Error | n_py | n_ml | KS statistic | p-value |")
    lines.append("|---|---|---|---|---|")
    for key, label in (("direct", "direct"), ("ghost", "ghost")):
        for idx, err_label in ((1, "range (m)"), (2, "azimuth (deg)"), (3, "radial vel (m/s)")):
            a = np.array([m[idx] for m in python_summary[key]], dtype=float)
            b = np.array([m[idx] for m in matlab_summary[key]], dtype=float)
            a, b = a[np.isfinite(a)], b[np.isfinite(b)]
            if a.size < 2 or b.size < 2:
                lines.append(f"| {label} {err_label} | {a.size} | {b.size} | n/a | n/a |")
                continue
            stat, p = scipy_stats.ks_2samp(a, b)
            lines.append(f"| {label} {err_label} | {a.size} | {b.size} | {stat:.4f} | {p:.4f} |")


def _plots(python_summary, matlab_summary, out_dir):
    if plt is None:
        return []
    written = []
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, idx, label in zip(axes, (1, 2, 3), ("range error (m)", "azimuth error (deg)", "radial-vel error (m/s)")):
        for summary, name, color in ((python_summary, "python", "tab:blue"), (matlab_summary, "matlab", "tab:orange")):
            if summary is None:
                continue
            vals = np.array([m[idx] for m in summary["direct"]], dtype=float)
            vals = vals[np.isfinite(vals)]
            if vals.size:
                lo, hi = np.percentile(vals, [0.5, 99.5])
                grid = np.linspace(lo, hi, 40)
                ax.hist(vals, bins=grid, alpha=0.5, label=name, color=color, density=True)
        ax.set_xlabel(label)
        ax.set_ylabel("density")
        ax.legend()
    fig.suptitle("Direct-pedestrian error distributions (matched to analytic truth)")
    path = os.path.join(out_dir, "error_histograms.png")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    written.append(path)

    fig, ax = plt.subplots(figsize=(7, 5))
    for summary, name, color in ((python_summary, "python", "tab:blue"), (matlab_summary, "matlab", "tab:orange")):
        if summary is None:
            continue
        pts = [r for r in summary["rows"] if r["point_type"] != "clutter"]
        if pts:
            ax.scatter([p["x_m"] for p in pts], [p["y_m"] for p in pts], s=4, alpha=0.3, label=name, color=color)
    if truth_rows:
        ped = [r for r in truth_rows if r["entity"] == "pedestrian"]
        ghost = [r for r in truth_rows if r["entity"] == "mirror_type1_order2"]
        ax.plot([r["x_m"] for r in ped], [r["y_m"] for r in ped], "k-", lw=2, label="truth ped")
        ax.plot([r["x_m"] for r in ghost], [r["y_m"] for r in ghost], "k--", lw=1.5, label="truth mirror ghost")
        ax.plot([], [], " ", label=f"guardrail y=4.0 m")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m, right)")
    ax.set_aspect("equal")
    ax.legend(fontsize=8)
    fig.suptitle("Detections vs analytic truth geometry")
    path = os.path.join(out_dir, "geometry_scatter.png")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    written.append(path)
    return written


def main():
    global truth_rows, truth_by_frame_all

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--truth", default=os.path.join(HERE, "truth_scenario.csv"))
    parser.add_argument("--python", default=os.path.join(HERE, "python_detections.csv"))
    parser.add_argument("--matlab", default=os.path.join(HERE, "matlab_detections.csv"))
    parser.add_argument("--output-dir", default=HERE)
    parser.add_argument("--open", action="store_true", help="print report to stdout")
    args = parser.parse_args()

    truth_rows = _read_csv(args.truth)
    if not truth_rows:
        raise SystemExit(f"truth file missing or empty: {args.truth}")
    truth_by_frame_all = _truth_by_frame(truth_rows)

    python_rows = _read_csv(args.python)
    matlab_rows = _read_csv(args.matlab)

    n_frames = len({int(r["frame"]) for r in truth_rows})
    lines = [
        "# Python vs MATLAB radar comparison report",
        "",
        f"Truth scenario: {args.truth}",
        f"Frames: {n_frames} at 10 Hz (38.5 s), pedestrian 12-25 m, guardrail y=+4 m.",
        "Envelope: rgd_regime_v1 + 153 m range gate (10 Hz, 140 deg FOV, "
        "0.15 m / 1.8 deg / 0.087 m/s resolutions, +/-44.3 m/s Doppler).",
        "",
    ]

    python_summary = _model_report("python (RealisticRadarModel, rgd_regime_v1)", python_rows, lines)
    matlab_summary = _model_report("matlab (radarDataGenerator, matched envelope)", matlab_rows, lines)
    _ks_block(python_summary, matlab_summary, lines)

    if python_summary and matlab_summary:
        lines.append("\n## Headline\n")
        lines.append(
            "Two different quantities are reported below because they are NOT the "
            "same thing and neither is a detection probability:\n"
        )
        lines.append("| Quantity | python | matlab |")
        lines.append("|---|---|---|")
        for key, label in (("direct", "Direct"), ("ghost", "Ghost")):
            py_pts = len(python_summary[key])
            ml_pts = len(matlab_summary[key])
            py_cov, n_truth = _frame_coverage(python_summary[key], key)
            ml_cov, _ = _frame_coverage(matlab_summary[key], key)
            lines.append(
                f"| {label} matched points/frame | {py_pts / max(n_frames, 1):.2f} "
                f"| {ml_pts / max(n_frames, 1):.2f} |"
            )
            lines.append(
                f"| {label} frame coverage (>=1 matched point) | "
                f"{py_cov:.3f} | {ml_cov:.3f} |"
            )
        lines.append(
            f"\nFrame coverage is out of {n_truth} truth frames carrying that "
            "entity. Use frame coverage, not points/frame, to compare the two "
            "models: python returns a multi-point extended echo per target "
            "while matlab returns a single point per target, so points/frame "
            "reflects the echo model rather than detection performance."
        )

    plots = _plots(python_summary, matlab_summary, args.output_dir)
    if plots:
        lines.append("\nPlots: " + ", ".join(os.path.basename(p) for p in plots))

    report_path = os.path.join(args.output_dir, "comparison_report.md")
    with open(report_path, "w") as handle:
        handle.write("\n".join(lines) + "\n")
    print(f"wrote {report_path}")
    if args.open:
        print("\n".join(lines))


if __name__ == "__main__":
    main()
