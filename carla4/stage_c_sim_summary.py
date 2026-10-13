"""Build the simulator's RealismSummary from a validate_radar_accuracy run.

Reads ``radar_details.jsonl`` + ``radar_samples.csv`` produced by
``carla4/validate_radar_accuracy.py``:

- detections per scan = ``backends.realistic.debug.delivered_detections``
  (the post-tracker delivered target list, ghosts included), reduced to
  ``[range, azimuth, radial_velocity_closing]``;
- ground-truth targets per scan = the lead vehicle box from the CSV
  (``gt_center_range_m``, ``gt_azimuth_deg``, ``gt_closing_velocity_mps``).

The scans are reduced with the same ``summarize_scans`` as the real data and
compared against a real-data summary (e.g. the nuScenes merged-front one) via
``compare_summaries``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Any, Dict, List, Tuple

import numpy as np

sys_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "..")
import sys
sys.path.insert(0, sys_path)

from carla4.radar.realism_metrics import (  # noqa: E402
    RealismSummary,
    Scan,
    compare_summaries,
    summarize_scans,
)


def _det_rows(detections: List[Dict[str, Any]]) -> np.ndarray:
    rows = []
    for det in detections:
        try:
            rows.append([
                float(det["distance_m"]),
                float(det["azimuth_rad"]),
                float(det["relative_velocity_mps"]),
            ])
        except (KeyError, TypeError, ValueError):
            continue
    if not rows:
        return np.zeros((0, 3))
    return np.asarray(rows, dtype=float)


def load_gt_by_sample(csv_path: str) -> Dict[int, Tuple[np.ndarray, int]]:
    import csv

    out: Dict[int, Tuple[np.ndarray, int]] = {}
    with open(csv_path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            try:
                sample = int(float(row["sample_index"]))
                target = np.array([
                    float(row["gt_center_range_m"]),
                    math.radians(float(row["gt_azimuth_deg"])),
                    float(row["gt_closing_velocity_mps"]),
                ])
                lead_id = int(float(row.get("lead_actor_id") or 0))
                out.setdefault(sample, (target, lead_id))
            except (KeyError, TypeError, ValueError):
                continue
    return out


def load_gt_by_world_frame(csv_path: str) -> Dict[int, np.ndarray]:
    """GT keyed by the CARLA world frame, for latency-aligned association."""

    import csv

    out: Dict[int, np.ndarray] = {}
    with open(csv_path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            try:
                frame = int(float(row["world_frame"]))
                target = np.array([
                    float(row["gt_center_range_m"]),
                    math.radians(float(row["gt_azimuth_deg"])),
                    float(row["gt_closing_velocity_mps"]),
                ])
                out.setdefault(frame, target)
            except (KeyError, TypeError, ValueError):
                continue
    return out


def build_sim_scans(details_path: str, gt_by_sample: Dict[int, np.ndarray],
                    gt_by_frame: Dict[int, np.ndarray]) -> List[Scan]:
    scans = []
    with open(details_path, "r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            sample = int(record.get("sample_index", 0))
            detections = _det_rows(
                record.get("backends", {}).get("realistic", {}).get("debug", {}).get("delivered_detections", [])
            )
            # The CSV ground truth at sample_index is what the validator's own
            # error columns are evaluated against, so keep the association
            # consistent with it. (Matching by aligned_frame shifted the bias
            # the wrong way; the sensor's aligned frame numbering does not
            # map onto the CSV world_frame column one-to-one.)
            aligned_frame = (
                record.get("backends", {}).get("realistic", {}).get("aligned_ground_truth_frame")
            )
            entry = gt_by_sample.get(sample)
            if entry is None:
                scans.append(Scan(detections=detections, targets=np.zeros((0, 3))))
                continue
            gt, lead_id = entry
            targets = np.asarray([gt], dtype=float)
            truth_ids = np.asarray([lead_id], dtype=np.int64) if lead_id else None
            scans.append(Scan(detections=detections, targets=targets,
                              target_truth_ids=truth_ids))
    return scans


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage_c_dir", help="output of validate_radar_accuracy.py")
    parser.add_argument("--real-summary", default=None,
                        help="real-data RealismSummary JSON to compare against")
    parser.add_argument("--sim-summary-out", default=None)
    parser.add_argument("--compare-out", default=None)
    args = parser.parse_args()

    details = os.path.join(args.stage_c_dir, "radar_details.jsonl")
    samples = os.path.join(args.stage_c_dir, "radar_samples.csv")
    gt_by_sample = load_gt_by_sample(samples)
    gt_by_frame = load_gt_by_world_frame(samples)
    scans = build_sim_scans(details, gt_by_sample, gt_by_frame)
    sim_summary = summarize_scans(scans)
    print(f"sim scans={sim_summary.n_scans} "
          f"dets/scan={sim_summary.detections_per_scan_mean:.1f} "
          f"ghosts/scan={sim_summary.ghosts_per_scan_mean:.1f} "
          f"range_sigma={sim_summary.range_error_std_m:.3f} m "
          f"range_mean={sim_summary.range_error_mean_m:.3f} m "
          f"vr_sigma={sim_summary.velocity_error_std_mps:.3f} m/s "
          f"misassign={sim_summary.velocity_misassignment_rate:.3f} "
          f"miss_run_mean={sim_summary.miss_run_mean:.2f} "
          f"count_autocorr={sim_summary.detection_count_lag1_autocorr:.3f}")

    if args.sim_summary_out:
        sim_summary.to_json(args.sim_summary_out)

    if args.real_summary:
        real_summary = RealismSummary.from_json(args.real_summary)
        report = compare_summaries(real_summary, sim_summary)
        if args.compare_out:
            parent = os.path.dirname(os.path.abspath(args.compare_out))
            if parent:
                os.makedirs(parent, exist_ok=True)
            with open(args.compare_out, "w", encoding="utf-8") as handle:
                json.dump(report, handle, indent=2)
        table = report["statistics"]
        print("\nreal-vs-sim key deltas (reference=real, candidate=sim):")
        for key in ("range_error_mean_m", "range_error_std_m",
                    "velocity_error_mean_mps", "velocity_error_std_mps",
                    "velocity_misassignment_rate",
                    "ghosts_per_scan_mean", "ghost_fraction"):
            block = table.get(key)
            if block:
                print(f"  {key:32s} real={block['reference']:8.3f} sim={block['candidate']:8.3f} delta={block['delta']:+.3f}")
        for key in ("range_error_ks", "azimuth_error_ks", "velocity_error_ks", "ghost_range_ks",
                    "miss_run_hist_l1"):
            if key in table:
                print(f"  {key:32s} {table[key]:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
