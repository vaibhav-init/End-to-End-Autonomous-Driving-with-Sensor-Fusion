#!/usr/bin/env python3
"""Thin an expanded CARLA ghost export down to the real prepared point density.

The CFAR-style export expansion in ``radar/ghost_detection/export_expansion.py``
overshoots badly: measured on ``ghost_carla_zeroshot_v2`` it emits ~2150 points
per scan against the ~252 that ``prepare_radar_ghost_dataset.py`` actually
produces from real Radar Ghost Dataset sequences. That 8.5x mismatch is not
cosmetic. Two of the three schema-v2 features are frame-relative:

- ``relative_log_amplitude`` centres each point on its frame's *median* log
  amplitude, so the reference is computed over populations of wildly different
  size;
- ``local_density_ratio`` counts neighbours inside a *fixed physical* gate
  (1.5 m / 2 deg), so packing 8.5x the points into the same volume inflates the
  raw counts, and normalising by the frame mean only partly cancels it.

Uniform random thinning is the right correction and is statistically equivalent
to having collected with a smaller ``--points-per-detection``: the expansion
draws its per-detection count from a Poisson distribution, and thinning a
Poisson process by probability p yields Poisson(p*lambda) while leaving the
position, amplitude and Doppler marginals untouched. Labels are inherited
point-wise, so class proportions are preserved in expectation.

This avoids a fresh CARLA collection (~2 min/sequence) for the same result.
"""

import argparse
import hashlib
from pathlib import Path
import shutil

import h5py
import numpy as np


# Median points per (sensor, frame) measured on the prepared real RGD splits.
REAL_POINTS_PER_SCAN = 250


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="collected CARLA H5 tree")
    parser.add_argument("--output", required=True, help="thinned H5 tree")
    parser.add_argument(
        "--target-points",
        type=int,
        default=REAL_POINTS_PER_SCAN,
        help=(
            "points kept per (sensor, frame); default matches the measured "
            "real prepared density"
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace an existing output tree",
    )
    return parser.parse_args()


def _scan_keys(radar):
    """Group row indices by (sensor, frame), the unit a real scan corresponds to."""

    names = radar.dtype.names or ()
    frame = np.asarray(radar["frame"]).astype(np.int64, copy=False)
    if "sensor" in names:
        sensor_raw = np.asarray(radar["sensor"])
        _, sensor = np.unique(sensor_raw, return_inverse=True)
    else:
        sensor = np.zeros(len(frame), dtype=np.int64)
    keys = sensor.astype(np.int64) * (int(frame.max()) + 1 if len(frame) else 1)
    keys = keys + frame
    order = np.argsort(keys, kind="stable")
    sorted_keys = keys[order]
    boundaries = np.flatnonzero(sorted_keys[1:] != sorted_keys[:-1]) + 1
    starts = np.concatenate((np.array((0,), dtype=np.int64), boundaries))
    ends = np.concatenate(
        (boundaries, np.array((len(sorted_keys),), dtype=np.int64))
    )
    return order, starts, ends


def decimate_file(source, destination, target_points, seed):
    with h5py.File(source, "r") as handle:
        if "radar" not in handle:
            raise ValueError(f"H5 file has no 'radar' dataset: {source}")
        radar = np.copy(handle["radar"])
        attributes = {key: handle.attrs[key] for key in handle.attrs}
    if radar.dtype.names is None:
        raise ValueError(f"Radar entry must be a structured array: {source}")
    if not len(radar):
        raise ValueError(f"Radar entry is empty: {source}")

    order, starts, ends = _scan_keys(radar)
    # Seed per file so a rerun reproduces the same thinning. hashlib rather
    # than hash(): the builtin is salted per process, so it would silently
    # give a different subsample on every run.
    digest = hashlib.sha256(
        f"{Path(source).name}:{int(seed)}".encode("utf-8")
    ).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "big"))
    keep = []
    for start, end in zip(starts, ends):
        group = order[start:end]
        if len(group) <= target_points:
            keep.append(group)
            continue
        keep.append(rng.choice(group, target_points, replace=False))
    kept = np.sort(np.concatenate(keep))
    thinned = radar[kept]

    destination.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(destination, "w") as handle:
        handle.create_dataset("radar", data=thinned, compression="gzip")
        for key, value in attributes.items():
            handle.attrs[key] = value
    return len(radar), len(thinned), len(starts)


def main():
    args = parse_args()
    if args.target_points < 1:
        raise ValueError("--target-points must be positive")
    source_root = Path(args.input)
    output_root = Path(args.output)
    if output_root.exists():
        if not args.overwrite:
            raise SystemExit(
                f"{output_root} already exists; pass --overwrite to replace it"
            )
        shutil.rmtree(output_root)

    files = sorted(source_root.rglob("*.h5"))
    if not files:
        raise SystemExit(f"No .h5 files under {source_root}")
    total_before = total_after = total_scans = 0
    for path in files:
        relative = path.relative_to(source_root)
        before, after, scans = decimate_file(
            path,
            output_root / relative,
            args.target_points,
            args.seed,
        )
        total_before += before
        total_after += after
        total_scans += scans
        print(
            f"  {relative}: {before:>9,d} -> {after:>8,d} points "
            f"({scans} scans, {after / max(scans, 1):.0f}/scan)"
        )
    print()
    print(
        f"{len(files)} files | {total_before:,d} -> {total_after:,d} points "
        f"({total_after / max(total_before, 1) * 100:.1f}% kept) | "
        f"{total_after / max(total_scans, 1):.0f} points/scan "
        f"(target {args.target_points})"
    )
    print(f"Wrote {output_root}")


if __name__ == "__main__":
    main()
