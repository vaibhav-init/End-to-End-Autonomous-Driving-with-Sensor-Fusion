"""Realism metrics for automotive radar target-list models.

The module scores a sequence of radar scans (real or simulated) against scene
ground truth and reports the distributions a "comparable to real radar"
claim must reproduce:

- detections per scan,
- probability of detection versus range,
- range / azimuth / radial-velocity error statistics of matched detections,
- unmatched-detection (ghost / clutter) rate and its range distribution,
- Doppler-ambiguity misassignment rate of matched detections.

The result is a :class:`RealismSummary` that serialises to JSON.  Two
summaries -- one from real data, one from the simulator -- can then be
compared with :func:`compare_summaries`, which reports histogram distances
and KS statistics where raw samples are available.  This is the statistical
anchor for fitting ``RealisticRadarConfig`` (``rgd_regime_v1``) parameters
against measured radar data.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import math
import os
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

# Association gates: a detection matches a target only inside all three.
RANGE_GATE_M = 1.5
AZIMUTH_GATE_RAD = math.radians(8.0)
RADIAL_VELOCITY_GATE_MPS = 6.0

# A matched pair whose velocity error exceeds this margin is counted as a
# Doppler-ambiguity misassignment instead of measurement noise.
DOPPLER_MISASSIGNMENT_MPS = 2.0

# Range bins in metres used for the Pd table and ghost-range histogram.
PD_RANGE_BINS = (0.0, 25.0, 50.0, 75.0, 100.0, 150.0, 250.0)

DETECTIONS_PER_SCAN_HIST_BINS = 60


@dataclass
class Scan:
    """One radar scan with its detections and scene ground truth.

    ``detections`` and ``targets`` are rows of arrays: distance [m] from the
    radar, azimuth [rad] from boresight (positive right), radial velocity
    [m/s] positive closing. ``detection_truth_ids`` and ``target_truth_ids``
    optionally carry per-scan ground-truth association (the simulator knows
    it exactly; the extraction adapter for real data fills it by the same
    nearest-neighbour gate used here and stores it for reproducibility).
    """

    detections: np.ndarray                      # shape (Nd, >=3)
    targets: np.ndarray                         # shape (Nt, >=3)
    detection_truth_ids: Optional[np.ndarray] = None
    target_truth_ids: Optional[np.ndarray] = None
    target_is_direct: Optional[np.ndarray] = None  # bool per target


@dataclass
class MatchStats:
    """Association outcomes of one scan."""

    matched_delta: np.ndarray                   # (n_matched, 3) det - target
    matched_indices: List[Tuple[int, int]] = field(default_factory=list)
    unmatched_detections: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    matched_velocity_err: np.ndarray = field(default_factory=lambda: np.zeros((0,)))
    matched_detection_snrless: List[Tuple[int, int]] = field(default_factory=list)


def _validate_scan_rows(rows: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(rows, dtype=float)
    if array.ndim != 2 or array.shape[1] < 3:
        raise ValueError(f"{name} must have columns [range, azimuth, radial_velocity]")
    return array


def _greedy_associate(detections: np.ndarray, targets: np.ndarray) -> MatchStats:
    """Greedy nearest association inside the gates.

    Candidates are paired by ascending normalised cost; a detection and a
    target are each used at most once. The ordering keeps association
    deterministic for identical scans.
    """

    stats = MatchStats(matched_delta=np.zeros((0, 3)))
    if detections.shape[0] == 0 or targets.shape[0] == 0:
        stats.unmatched_detections = detections[:, :3] if detections.shape[0] else detections[:0, :3]
        return stats

    cost = np.full((detections.shape[0], targets.shape[0]), np.inf)
    for det_i, det in enumerate(detections):
        for tgt_i, tgt in enumerate(targets):
            d_range = abs(det[0] - tgt[0])
            d_az = abs(math.remainder(det[1] - tgt[1], math.tau))
            d_vr = abs(det[2] - tgt[2])
            if d_range <= RANGE_GATE_M and d_az <= AZIMUTH_GATE_RAD and d_vr < RADIAL_VELOCITY_GATE_MPS:
                cost[det_i, tgt_i] = (d_range / RANGE_GATE_M
                                      + d_az / AZIMUTH_GATE_RAD
                                      + d_vr / RADIAL_VELOCITY_GATE_MPS)
    flat = np.argsort(cost, axis=None, kind="stable")
    det_used: set = set()
    tgt_used: set = set()
    deltas = []
    pairs = []
    vel_errs = []
    for index in flat:
        det_i, tgt_i = np.unravel_index(index, cost.shape)
        if not np.isfinite(cost[det_i, tgt_i]) or det_i in det_used or tgt_i in tgt_used:
            continue
        det_used.add(int(det_i))
        tgt_used.add(int(tgt_i))
        delta = detections[det_i, :3] - targets[tgt_i, :3]
        deltas.append(delta)
        pairs.append((int(det_i), int(tgt_i)))
        vel_errs.append(float(delta[2]))
    if deltas:
        stats.matched_delta = np.asarray(deltas, dtype=float)
        stats.matched_indices = pairs
        stats.matched_velocity_err = np.asarray(vel_errs, dtype=float)
    keep = [i for i in range(detections.shape[0]) if i not in det_used]
    stats.unmatched_detections = detections[keep, :3] if keep else detections[:0, :3]
    return stats


def _range_bin_index(range_m: float) -> int:
    """Bin index of ``range_m`` in PD_RANGE_BINS, clamped to interior bins."""

    if range_m <= PD_RANGE_BINS[0]:
        return 0
    if range_m >= PD_RANGE_BINS[-1]:
        return len(PD_RANGE_BINS) - 2
    return int(np.searchsorted(PD_RANGE_BINS, range_m, side="right") - 1)


def _bin_centers(edges: Sequence[float]) -> List[float]:
    return [(float(edges[i]) + float(edges[i + 1])) / 2.0 for i in range(len(edges) - 1)]


@dataclass
class RealismSummary:
    """Distribution-level portrait of a scan sequence."""

    n_scans: int = 0
    detections_per_scan_mean: float = 0.0
    detections_per_scan_std: float = 0.0
    detections_per_scan_hist: List[int] = field(default_factory=list)  # ints, bins=60 over [0, DETECTIONS_PER_SCAN_HIST_BINS]
    targets_per_scan_mean: float = 0.0

    # Pd table: for each PD_RANGE_BINS logical cell, [seen, detected] counts,
    # and the list of range-bin edges used.
    pd_bin_edges: List[float] = field(default_factory=list)
    pd_counts: List[List[int]] = field(default_factory=list)            # per bin: [n_target_scans, n_detected]

    # Measurement-error statistics over matched pairs: delta = det - target.
    range_error_mean_m: float = 0.0
    range_error_std_m: float = 0.0
    range_error_quantiles: List[float] = field(default_factory=list)     # 50/90/95 %
    azimuth_error_mean_rad: float = 0.0
    azimuth_error_std_rad: float = 0.0
    azimuth_error_quantiles_rad: List[float] = field(default_factory=list)
    velocity_error_mean_mps: float = 0.0
    velocity_error_std_mps: float = 0.0
    velocity_error_quantiles_mps: List[float] = field(default_factory=list)
    n_matched: int = 0

    # Doppler-ambiguity misassignment among matched pairs.
    velocity_misassignment_rate: float = 0.0

    # Ghost / clutter portrait of unmatched detections.
    ghosts_per_scan_mean: float = 0.0
    ghost_fraction: float = 0.0
    ghost_range_edges: List[float] = field(default_factory=list)
    ghost_range_counts: List[int] = field(default_factory=list)

    n_detection_samples: int = 0                                        # matched pairs retained for KS (capped)
    raw_matched_velocity_samples: List[float] = field(default_factory=list)
    raw_matched_range_samples: List[float] = field(default_factory=list)
    raw_matched_azimuth_samples: List[float] = field(default_factory=list)
    raw_ghost_range_samples: List[float] = field(default_factory=list)

    RAW_SAMPLE_CAP = 20000

    def to_json(self, path: str) -> str:
        parent = os.path.dirname(os.path.abspath(path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(asdict(self), handle, indent=2)
        return path

    @classmethod
    def from_json(cls, path: str) -> "RealismSummary":
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return cls(**{key: value for key, value in payload.items()
                      if key not in ("RAW_SAMPLE_CAP",)})


def summarize_scans(scans: Iterable[Scan],
                    keep_raw_samples: bool = True) -> RealismSummary:
    """Reduce a scan sequence to a :class:`RealismSummary`."""

    summary = RealismSummary()
    summary.pd_bin_edges = list(PD_RANGE_BINS)
    summary.pd_counts = [[0, 0] for _ in range(len(PD_RANGE_BINS) - 1)]
    summary.ghost_range_edges = list(PD_RANGE_BINS)
    summary.ghost_range_counts = [0] * (len(PD_RANGE_BINS) - 1)
    det_counts = []
    tgt_counts = []
    matched = np.empty((0, 3))
    ghost_rows: List[np.ndarray] = []
    misassigned = 0

    for scan in scans:
        dets = _validate_scan_rows(scan.detections, "detections")
        tgts = _validate_scan_rows(scan.targets, "targets")
        det_count = int(dets.shape[0])
        det_counts.append(det_count)
        tgt_count = int(tgts.shape[0])
        tgt_counts.append(tgt_count)

        stats = _greedy_associate(dets, tgts)
        if stats.matched_delta.size:
            matched = stats.matched_delta if matched.size == 0 else np.vstack((matched, stats.matched_delta))
            misassigned += int(np.sum(np.abs(stats.matched_velocity_err) > DOPPLER_MISASSIGNMENT_MPS))

        # Every ground-truth target in the gate contributes one "seen" vote.
        for tgt in tgts:
            bin_index = _range_bin_index(float(tgt[0]))
            summary.pd_counts[bin_index][0] += 1
        # Count detected targets per bin using matched pairs.
        if stats.matched_delta.size:
            for tgt_i in {pair[1] for pair in stats.matched_indices}:
                tgt_range = tgts[tgt_i, 0]
                summary.pd_counts[_range_bin_index(float(tgt_range))][1] += 1

        if stats.unmatched_detections.size:
            rows = np.atleast_2d(stats.unmatched_detections)
            for row in rows:
                ghost_bin = _range_bin_index(float(row[0]))
                summary.ghost_range_counts[ghost_bin] += 1
            ghost_rows.append(rows[:, 0].ravel())

    summary.n_scans = len(det_counts)
    if det_counts:
        counts = np.asarray(det_counts, dtype=float)
        summary.detections_per_scan_mean = float(counts.mean())
        summary.detections_per_scan_std = float(counts.std(ddof=1)) if len(counts) > 1 else 0.0
        hist, _ = np.histogram(counts, bins=DETECTIONS_PER_SCAN_HIST_BINS,
                               range=(0, DETECTIONS_PER_SCAN_HIST_BINS))
        summary.detections_per_scan_hist = [int(v) for v in hist]
    summary.targets_per_scan_mean = float(np.mean(tgt_counts)) if tgt_counts else 0.0

    if matched.size:
        rng = matched[:, 0]
        az = matched[:, 1]
        vel = matched[:, 2]
        summary.n_matched = int(matched.shape[0])
        summary.range_error_mean_m = float(rng.mean())
        summary.range_error_std_m = float(rng.std(ddof=1)) if rng.size > 1 else 0.0
        summary.range_error_quantiles = [float(q) for q in np.percentile(rng, [50, 90, 95])]
        summary.azimuth_error_mean_rad = float(az.mean())
        summary.azimuth_error_std_rad = float(az.std(ddof=1)) if az.size > 1 else 0.0
        summary.azimuth_error_quantiles_rad = [float(q) for q in np.percentile(az, [50, 90, 95])]
        summary.velocity_error_mean_mps = float(vel.mean())
        summary.velocity_error_std_mps = float(vel.std(ddof=1)) if vel.size > 1 else 0.0
        summary.velocity_error_quantiles_mps = [float(q) for q in np.percentile(vel, [50, 90, 95])]
        summary.velocity_misassignment_rate = misassigned / matched.shape[0]
        if keep_raw_samples:
            cap = RealismSummary.RAW_SAMPLE_CAP
            summary.raw_matched_range_samples = rng[:cap].tolist()
            summary.raw_matched_azimuth_samples = az[:cap].tolist()
            summary.raw_matched_velocity_samples = vel[:cap].tolist()
            summary.n_detection_samples = int(min(rng.size, cap))

    dets_per_scan = sum(det_counts)
    ghosts = float(sum(int(r.size) for r in ghost_rows))
    summary.ghosts_per_scan_mean = ghosts / summary.n_scans if summary.n_scans else 0.0
    summary.ghost_fraction = ghosts / dets_per_scan if dets_per_scan else 0.0
    if ghost_rows and keep_raw_samples:
        flat_ghost = np.concatenate(ghost_rows)
        cap = RealismSummary.RAW_SAMPLE_CAP
        summary.raw_ghost_range_samples = flat_ghost.ravel()[:cap].tolist()

    return summary


def _ks_statistic(samples_a: Sequence[float], samples_b: Sequence[float]) -> float:
    a = np.asarray(samples_a, dtype=float)
    b = np.asarray(samples_b, dtype=float)
    if a.size < 2 or b.size < 2:
        return float("nan")
    grid = np.sort(np.concatenate((a, b)))
    cdf_a = np.searchsorted(np.sort(a), grid, side="right") / a.size
    cdf_b = np.searchsorted(np.sort(b), grid, side="right") / b.size
    return float(np.max(np.abs(cdf_a - cdf_b)))


def _l1_distance(hist_a: Sequence[float], hist_b: Sequence[float]) -> float:
    """L1/2 distance between two count histograms sharing one support."""

    a = np.asarray(hist_a, dtype=float)
    b = np.asarray(hist_b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("histograms must share support")
    total_a = float(a.sum())
    total_b = float(b.sum())
    if total_a == 0.0 or total_b == 0.0:
        return 1.0
    probability_a = a / total_a
    probability_b = b / total_b
    return float(np.abs(probability_a - probability_b).sum() / 2.0)


def compare_summaries(reference: RealismSummary, candidate: RealismSummary) -> Dict[str, Any]:
    """Compare a simulator summary with a real-data reference.

    Returns a flat report dictionary; histogram distances are L1/2 for the
    discrete distributions, KS statistics are reported only when both
    summaries retained raw samples for that metric.
    """

    report: Dict[str, Any] = {}
    report["detections_per_scan"] = {
        "reference_mean": reference.detections_per_scan_mean,
        "candidate_mean": candidate.detections_per_scan_mean,
        "hist_l1": _l1_distance(reference.detections_per_scan_hist,
                                candidate.detections_per_scan_hist),
    }
    report["pd_table"] = {
        "reference_bins": reference.pd_counts,
        "candidate_bins": candidate.pd_counts,
        "bin_edges": reference.pd_bin_edges,
    }
    err = {}
    for key, ref_val, cand_val in (
        ("range_error_mean_m", reference.range_error_mean_m, candidate.range_error_mean_m),
        ("range_error_std_m", reference.range_error_std_m, candidate.range_error_std_m),
        ("azimuth_error_mean_rad", reference.azimuth_error_mean_rad, candidate.azimuth_error_mean_rad),
        ("azimuth_error_std_rad", reference.azimuth_error_std_rad, candidate.azimuth_error_std_rad),
        ("velocity_error_mean_mps", reference.velocity_error_mean_mps, candidate.velocity_error_mean_mps),
        ("velocity_error_std_mps", reference.velocity_error_std_mps, candidate.velocity_error_std_mps),
        ("velocity_misassignment_rate", reference.velocity_misassignment_rate,
         candidate.velocity_misassignment_rate),
        ("ghosts_per_scan_mean", reference.ghosts_per_scan_mean, candidate.ghosts_per_scan_mean),
        ("ghost_fraction", reference.ghost_fraction, candidate.ghost_fraction),
    ):
        err[key] = {"reference": ref_val, "candidate": cand_val,
                    "delta": (cand_val - ref_val) if ref_val is not None and cand_val is not None else None}
    if reference.raw_matched_range_samples and candidate.raw_matched_range_samples:
        err["range_error_ks"] = _ks_statistic(reference.raw_matched_range_samples,
                                              candidate.raw_matched_range_samples)
    if reference.raw_matched_velocity_samples and candidate.raw_matched_velocity_samples:
        err["velocity_error_ks"] = _ks_statistic(reference.raw_matched_velocity_samples,
                                                 candidate.raw_matched_velocity_samples)
    if reference.raw_ghost_range_samples and candidate.raw_ghost_range_samples:
        err["ghost_range_ks"] = _ks_statistic(reference.raw_ghost_range_samples,
                                              candidate.raw_ghost_range_samples)
    report["statistics"] = err
    return report
