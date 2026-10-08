"""Build the real-radar RealismSummary from extracted nuScenes scans.

Reads the JSONL produced by ``nuScenes_radar_extract.py`` and writes one
reference summary JSON per requested variant:

- ``<prefix>_summary.json`` for every channel separately, and
- ``<prefix>_merged_front_summary.json`` merging the three front radars per
  keyframe into one virtual wide-FOV scan (targets de-duplicated across
  channels with the same nearest-neighbour gates as the metrics module).

The summaries are the real-data anchors that the simulator configuration
(``rgd_regime_v1``) must match.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np

sys.path.insert(0, os.path.join(os.path.abspath(os.path.dirname(__file__)), ".."))

from carla4.radar.realism_metrics import RealismSummary, Scan, summarize_scans  # noqa: E402

FRONT_CHANNELS = ("RADAR_FRONT", "RADAR_FRONT_LEFT", "RADAR_FRONT_RIGHT")


def _channel_of(row: Dict[str, Any]) -> str:
    return row["channel"]


def _scan_rows(dets: Iterable[List[float]], tgts: Iterable[List[float]]) -> Scan:
    detections = np.asarray(list(dets), dtype=float)
    targets = np.asarray(list(tgts), dtype=float)
    if detections.size == 0:
        detections = np.zeros((0, 3))
    elif detections.ndim == 1:
        detections = detections.reshape(1, -1)
    if targets.size == 0:
        targets = np.zeros((0, 6))
    elif targets.ndim == 1:
        targets = targets.reshape(1, -1)
    return Scan(detections=detections, targets=targets)


def _dedupe_targets(frames: Iterable[np.ndarray]) -> np.ndarray:
    """Greedy merge of the same physical target reported by multiple radars."""

    merged: List[np.ndarray] = []
    for frame in frames:
        for target in np.atleast_2d(frame):
            duplicate = False
            for existing in merged:
                if (abs(target[0] - existing[0]) < 1.5
                        and abs(np.remainder(target[1] - existing[1], np.pi)) < 0.15):
                    duplicate = True
                    break
            if not duplicate:
                merged.append(np.asarray(target, dtype=float))
    if not merged:
        return np.zeros((0, 6))
    return np.vstack(merged)


def build_summaries(rows: Iterable[Dict[str, Any]], out_dir: str,
                    prefix: str) -> List[Tuple[str, RealismSummary]]:
    by_channel: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    by_keyframe: Dict[Tuple[float, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        channel = _channel_of(row)
        by_channel[channel].append(row)
        if channel in FRONT_CHANNELS:
            key = (float(row["timestamp"]), row["sample_token"])
            by_keyframe[key].append(row)

    outputs: List[Tuple[str, RealismSummary]] = []
    for channel in sorted(by_channel):
        scans = []
        for row in by_channel[channel]:
            scans.append(_scan_rows(row["detections"], row["targets"]))
        summary = summarize_scans(scans)
        name = f"{prefix}_{channel.lower()}_summary.json"
        summary.to_json(os.path.join(out_dir, name))
        outputs.append((name, summary))

    merged_scans = []
    for _key, rows_at_frame in sorted(by_keyframe.items()):
        det_arrays = [np.asarray(r["detections"], dtype=float) for r in rows_at_frame]
        det_arrays = [d.reshape(1, -1) if d.size and d.ndim == 1 else d for d in det_arrays]
        det_arrays = [d for d in det_arrays if d.size and d.ndim == 2]
        detections = np.vstack(det_arrays) if det_arrays else np.zeros((0, 3))
        tgt_arrays = [np.atleast_2d(np.asarray(r["targets"], dtype=float)) for r in rows_at_frame]
        tgt_arrays = [t for t in tgt_arrays if t.size and t.ndim == 2 and t.shape[0] > 0]
        targets = _dedupe_targets(tgt_arrays)
        merged_scans.append(Scan(detections=detections, targets=targets))
    merged_summary = summarize_scans(merged_scans)
    merged_name = f"{prefix}_merged_front_summary.json"
    merged_summary.to_json(os.path.join(out_dir, merged_name))
    outputs.append((merged_name, merged_summary))
    return outputs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scans", help="JSONL from nuScenes_radar_extract.py(.gz)")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--prefix", default="nuScenes")
    args = parser.parse_args()

    opener = gzip.open if args.scans.endswith(".gz") else open
    rows = []
    with opener(args.scans, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    os.makedirs(args.out_dir, exist_ok=True)
    for name, summary in build_summaries(rows, args.out_dir, args.prefix):
        print(f"{name}: scans={summary.n_scans} "
              f"dets/scan={summary.detections_per_scan_mean:.1f} "
              f"targets/scan={summary.targets_per_scan_mean:.1f} "
              f"ghosts/scan={summary.ghosts_per_scan_mean:.2f} "
              f"range_err_sigma={summary.range_error_std_m:.2f}m "
              f"vr_sigma={summary.velocity_error_std_mps:.2f}m/s "
              f"vd_mis={summary.velocity_misassignment_rate:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
