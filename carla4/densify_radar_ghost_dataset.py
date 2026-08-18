#!/usr/bin/env python3
"""Statistical densification for the CARLA -> RGD sim-to-real ghost pipeline.

The Radar Ghost Dataset v1.1 records ~800 raw CFAR detections per frame while
the CARLA target-list collector emits only tens of points per frame. A PointNet
trained on sparse CARLA point sets generalizes poorly to dense real point sets,
so this script synthesizes point clouds whose *spread statistics match the real
RGD train split*.

Two subcommands:

1. ``stencil`` -- measure, **from the RGD Train split only**, the per-class
   point stencil: the within-target spatial covariance (dx, dy), the Doppler
   variance (dv), and the amplitude distribution, for pedestrians (class 1)
   and cyclists (class 2). The RGD validation and test splits are never read.

2. ``densify`` -- for every labeled CARLA point whose class has a stencil,
   sample N synthetic points around it with a multivariate Gaussian built from
   that stencil, inherit the parent's real/ghost label, and write a compatible
   H5 tree that ``prepare_radar_ghost_dataset.py`` can consume unchanged.

Physics constraints honored during densification:

- The sampled point keeps the parent's radial velocity as its mean; only the
  measured per-detection Doppler spread is added, and the result is clamped to
  the RGD unambiguous Doppler envelope (+/-44.3 m/s). The parent's v_r = v*cos
  (theta) relation is therefore preserved.
- Sampled positions are clamped to the RGD sensor envelope: range 0.15-153 m,
  azimuth +/-70 deg (the same envelope as the ``rgd_regime_v1`` profile).
- range/azimuth are recomputed from the sampled (x, y) so x_cc = r*cos(phi),
  y_cc = r*sin(phi) stays exact.
- Background (label 0), noise (-2), ignore (-1), and any class without a
  stencil (e.g. vehicle labels 3/4/5) pass through unmodified: applying a
  pedestrian stencil to a wall return would corrupt the point-cloud context.
"""

import argparse
import json
from pathlib import Path
import re
import uuid

import h5py
import numpy as np

from radar.ghost_detection.labels import decode_cmto_label


RGD_RANGE_MIN_M = 0.15
RGD_RANGE_MAX_M = 153.0
RGD_FOV_RAD = np.deg2rad(70.0)
RGD_MAX_DOPPLER_MPS = 44.3
RGD_POINTS_PER_FRAME = 800

DENSIFIED_DTYPE = np.dtype(
    [
        ("frame", np.int64),
        ("frame_timestamp", np.float64),
        ("timestamp", np.float64),
        ("sensor", "S8"),
        ("x_cc", np.float32),
        ("y_cc", np.float32),
        ("r_sc", np.float32),
        ("phi_sc", np.float32),
        ("vr_sc", np.float32),
        ("amp", np.float32),
        ("uuid", "S36"),
        ("label_id", np.int32),
        ("instance_id", np.int64),
        ("source", "S16"),
        ("parent_object_id", np.int64),
        ("reflector_id", np.int64),
        ("bounce_type", "S16"),
        ("bounce_order", np.int8),
        ("path_length_m", np.float32),
    ]
)

CLASS_NAMES = {1: "pedestrian", 2: "cyclist"}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    stencil_parser = subparsers.add_parser(
        "stencil",
        help="measure the RGD train-split point stencil",
    )
    stencil_parser.add_argument(
        "--input",
        required=True,
        help=(
            "RGD source: either a prepared dataset directory (manifest.json "
            "with NPZ sequences) or the unpacked original/ H5 tree"
        ),
    )
    stencil_parser.add_argument("--output", required=True, help="stencil JSON path")
    stencil_parser.add_argument(
        "--split",
        default="train",
        help="RGD split to read (default train; never use val/test for the stencil)",
    )
    stencil_parser.add_argument(
        "--class-ids",
        type=int,
        nargs="+",
        default=(1, 2),
        help="RGD classes to measure (1 pedestrian, 2 cyclist)",
    )
    stencil_parser.add_argument(
        "--radius-m",
        type=float,
        default=2.0,
        help="cluster radius used to group points of the same target",
    )

    densify_parser = subparsers.add_parser(
        "densify",
        help="synthesize dense point clouds around CARLA labeled points",
    )
    densify_parser.add_argument(
        "--carla-input",
        required=True,
        help="CARLA collector output directory (H5 files, recursive)",
    )
    densify_parser.add_argument("--stencil", required=True, help="stencil JSON path")
    densify_parser.add_argument("--output", required=True, help="densified H5 directory")
    densify_parser.add_argument(
        "--points-per-frame",
        type=float,
        default=RGD_POINTS_PER_FRAME,
        help="target total point count per frame (RGD ~800)",
    )
    densify_parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Stencil measurement (RGD train split only)
# ---------------------------------------------------------------------------


def _cluster_centered_vectors(x, y, vr, radius_m):
    """Return pooled within-cluster centered (dx, dy, dv) vectors.

    Points of one class within one (sensor, frame) group are clustered by
    proximity; each cluster approximates the detections of one target. The
    centered vectors (relative to the cluster centroid) pool into the
    per-target spread statistics.
    """

    n = len(x)
    if n < 2:
        return []
    radius2 = float(radius_m) ** 2
    assigned = np.zeros(n, dtype=np.bool_)
    pooled = []
    for seed_index in range(n):
        if assigned[seed_index]:
            continue
        member = np.zeros(n, dtype=np.bool_)
        member[seed_index] = True
        assigned[seed_index] = True
        stack = [seed_index]
        while stack:
            index = stack.pop()
            dist2 = (x - x[index]) ** 2 + (y - y[index]) ** 2
            for candidate in np.flatnonzero((~assigned) & (dist2 <= radius2)):
                member[candidate] = True
                assigned[candidate] = True
                stack.append(int(candidate))
        count = int(member.sum())
        if count < 2:
            continue
        centroid = np.array(
            (
                float(x[member].mean()),
                float(y[member].mean()),
                float(vr[member].mean()),
            )
        )
        centered = np.stack(
            (x[member] - centroid[0], y[member] - centroid[1], vr[member] - centroid[2]),
            axis=-1,
        )
        pooled.append(centered)
    return pooled


def _sequence_class_ids(sequence):
    """Return per-point RGD class ids from prepared or raw sequences."""

    names = None
    if hasattr(sequence, "dtype") and sequence.dtype.names is not None:
        names = sequence.dtype.names
    if names is not None and "class_id" in names:
        return np.asarray(sequence["class_id"])
    if isinstance(sequence, dict) and "class_id" in sequence:
        return np.asarray(sequence["class_id"])
    # Raw H5 path: decode CMTO labels on the fly.
    label_ids = sequence["label_id"]
    return np.asarray(
        [decode_cmto_label(int(value)).class_id for value in label_ids],
        dtype=np.int8,
    )


def _sample_position_xy(sequence, class_id):
    """Return masked per-class (x, y, vr, amp, frame, sensor) or None."""

    class_mask = _sequence_class_ids(sequence)
    mask = class_mask == class_id
    if not np.any(mask):
        return None
    # Structured numpy arrays use dtype.names; dicts use ``in`` directly.
    names = getattr(getattr(sequence, "dtype", None), "names", None)
    has_x = (
        (isinstance(sequence, dict) and "x_cc" in sequence)
        or (names is not None and "x_cc" in names)
    )
    if has_x:
        x = np.asarray(sequence["x_cc"], dtype=np.float64)[mask]
        y = np.asarray(sequence["y_cc"], dtype=np.float64)[mask]
    else:
        r = np.asarray(sequence["r_sc"], dtype=np.float64)[mask]
        phi = np.asarray(sequence["phi_sc"], dtype=np.float64)[mask]
        x = r * np.cos(phi)
        y = r * np.sin(phi)
    return (
        x,
        y,
        np.asarray(sequence["vr_sc"], dtype=np.float64)[mask],
        np.asarray(sequence["amp"], dtype=np.float64)[mask],
        np.asarray(sequence["frame"])[mask],
        np.asarray(sequence["sensor"])[mask],
    )


def _consecutive_frame_groups(order, frame, sensor):
    """Split a sorted index array into runs sharing (sensor, frame)."""

    if len(order) == 0:
        return []
    change = np.flatnonzero(
        (frame[order][1:] != frame[order][:-1])
        | (sensor[order][1:] != sensor[order][:-1])
    )
    return np.split(order, change + 1)


def _measure_stencil(sequences, class_ids, radius_m):
    """Measure per-class stencils from prepared or raw RGD sequences."""

    class_stats = {}
    for class_id in class_ids:
        name = CLASS_NAMES.get(int(class_id), f"class_{int(class_id)}")
        centered_blocks = []
        amplitude_values = []
        point_count = 0
        cluster_count = 0
        for sequence in sequences:
            sample = _sample_position_xy(sequence, class_id)
            if sample is None:
                continue
            x, y, vr, amp, frame, sensor = sample
            point_count += len(x)
            amplitude_values.append(amp)
            order = np.lexsort((frame, sensor))
            for group in _consecutive_frame_groups(order, frame, sensor):
                if len(group) < 2:
                    continue
                blocks = _cluster_centered_vectors(
                    x[group],
                    y[group],
                    vr[group],
                    radius_m,
                )
                centered_blocks.extend(blocks)
                cluster_count += len(blocks)
        if not centered_blocks and not amplitude_values:
            class_stats[str(int(class_id))] = {
                "name": name,
                "point_count": 0,
                "cluster_count": 0,
                "available": False,
            }
            continue
        centered = np.concatenate(centered_blocks, axis=0) if centered_blocks else np.empty((0, 3))
        if len(centered):
            cov_3d = (centered.T @ centered) / len(centered)
            mean_offset = centered.mean(axis=0)
        else:
            cov_3d = np.zeros((3, 3))
            mean_offset = np.zeros(3)
        amplitudes = np.concatenate(amplitude_values)
        log1p = np.log1p(np.maximum(amplitudes, 0.0))
        class_stats[str(int(class_id))] = {
            "name": name,
            "point_count": int(point_count),
            "cluster_count": int(cluster_count),
            "offset_sample_count": int(len(centered)),
            "available": len(centered) > 0,
            "mean_offset_xy": mean_offset[:2].tolist(),
            "cov_3d": cov_3d.tolist(),
            "cov_xy": cov_3d[:2, :2].tolist(),
            "std_dv": float(np.sqrt(max(cov_3d[2, 2], 0.0))),
            "log1p_amp_mean": float(log1p.mean()),
            "log1p_amp_std": float(log1p.std()),
            "amp_median": float(np.median(amplitudes)),
            "amp_p10": float(np.percentile(amplitudes, 10)),
            "amp_p90": float(np.percentile(amplitudes, 90)),
        }
    return class_stats


def _iter_prepared_sequences(input_root, split):
    """Read prepared NPZ sequences matching *split* from a manifest.

    Returns a **list** (or ``None`` when there is no manifest), not a
    generator.  Calling a generator function always returns a truthy
    generator object — even when ``return None`` appears inside — which
    would silently steal the fallback to the raw-H5 path.
    """

    manifest_path = input_root / "manifest.json"
    if not manifest_path.is_file():
        return None
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    sequences = []
    for record in manifest.get("sequences", ()):
        if record.get("split") != split:
            continue
        path = input_root / record["path"]
        with np.load(path, allow_pickle=False) as archive:
            sequences.append({name: np.copy(archive[name]) for name in archive.files})
    return sequences or None


def _iter_raw_h5_sequences(input_root, split):
    # Check the split name as an actual directory component in the path,
    # which is far more robust than regex-matching the string form.
    split_lower = split.lower()
    paths = sorted(
        set(
            list(input_root.rglob("*.h5")) + list(input_root.rglob("*.hdf5"))
        )
    )
    for path in paths:
        if not any(part.lower() == split_lower for part in path.parts):
            continue
        with h5py.File(path, "r") as handle:
            radar = np.copy(handle["radar"])
        yield radar


def run_stencil(args):
    input_root = Path(args.input).resolve()
    output_path = Path(args.output)
    if output_path.suffix.lower() != ".json":
        output_path = output_path / "rgd_stencil.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    prepared = _iter_prepared_sequences(input_root, args.split)
    if prepared is not None:
        sequences = list(prepared)
        source = "prepared"
    else:
        sequences = list(_iter_raw_h5_sequences(input_root, args.split))
        source = "raw h5"
    if not sequences:
        details = []
        manifest_path = input_root / "manifest.json"
        if manifest_path.is_file():
            try:
                with manifest_path.open("r", encoding="utf-8") as handle:
                    manifest = json.load(handle)
                split_counts = {}
                for record in manifest.get("sequences", ()):
                    key = record.get("split")
                    split_counts[key] = split_counts.get(key, 0) + 1
                details.append(f"manifest.json present, sequences per split={split_counts}")
            except Exception as exc:  # pragma: no cover
                details.append(f"manifest.json unreadable: {exc}")
        else:
            details.append("no manifest.json found")
        h5_count = len(list(input_root.rglob("*.h5"))) + len(
            list(input_root.rglob("*.hdf5"))
        )
        details.append(f"h5 files found: {h5_count}")
        raise FileNotFoundError(
            f"No {args.split} RGD sequences found under {input_root}; "
            + "; ".join(details)
            + ". Point --input at the prepared dataset dir (manifest.json) or "
            "the raw original/ H5 tree (train/val/test subdirectories)."
        )
    class_stats = _measure_stencil(sequences, args.class_ids, args.radius_m)
    stencil = {
        "schema_version": 1,
        "source": "Radar Ghost Dataset v1.1",
        "input_root": str(input_root),
        "input_format": source,
        "split_used": args.split,
        "radius_m": args.radius_m,
        "class_ids": [int(value) for value in args.class_ids],
        "envelope": {
            "range_min_m": RGD_RANGE_MIN_M,
            "range_max_m": RGD_RANGE_MAX_M,
            "fov_deg": float(np.rad2deg(RGD_FOV_RAD)),
            "max_doppler_mps": RGD_MAX_DOPPLER_MPS,
        },
        "classes": class_stats,
    }
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(stencil, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"Wrote stencil to {output_path}")
    for class_id, stats in class_stats.items():
        if not stats.get("available"):
            print(
                f"  class {class_id} ({stats['name']}): NO samples in {args.split} "
                "split; densification will skip this class"
            )
            continue
        print(
            f"  class {class_id} ({stats['name']}): points={stats['point_count']}, "
            f"clusters={stats['cluster_count']}, "
            f"std_dx={float(np.sqrt(stats['cov_xy'][0][0])):.3f} m, "
            f"std_dy={float(np.sqrt(stats['cov_xy'][1][1])):.3f} m, "
            f"std_dv={stats['std_dv']:.3f} m/s, "
            f"amp_median={stats['amp_median']:.3g}, "
            f"log1p_amp(mu={stats['log1p_amp_mean']:.3f}, "
            f"sigma={stats['log1p_amp_std']:.3f})"
        )
    return stencil


# ---------------------------------------------------------------------------
# Densification
# ---------------------------------------------------------------------------


def _regularize_covariance(covariance):
    """Clamp eigenvalues so degenerate stencils still sample stably."""

    covariance = np.asarray(covariance, dtype=np.float64)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    eigenvalues = np.clip(eigenvalues, 1.0e-6, None)
    return (eigenvectors * eigenvalues) @ eigenvectors.T


def _class_id_of(label_id):
    return int(decode_cmto_label(int(label_id)).class_id)


def _passthrough_row(radar, index):
    """One output row equal to the parent point (no synthesis)."""

    parent = radar[index]
    return (
        index,
        0.0,
        0.0,
        float(parent["x_cc"]),
        float(parent["y_cc"]),
        float(parent["vr_sc"]),
        float(parent["amp"]),
    )


def _densify_frame(
    group,
    radar,
    stencil_classes,
    points_per_frame,
    seed,
):
    """Return one output row per emitted point for a (sensor, frame) group."""

    densifiable_indices = [
        index
        for index in group
        if _class_id_of(radar["label_id"][index]) in stencil_classes
    ]
    if not densifiable_indices:
        return [_passthrough_row(radar, index) for index in group]

    # Scale N so the whole frame reaches the RGD raw-detection density
    # (~800 points/frame) while non-stencil points pass through untouched.
    pass_count = len(group) - len(densifiable_indices)
    per_point = max(
        1,
        int(round((points_per_frame - pass_count) / len(densifiable_indices))),
    )
    rng = np.random.default_rng(seed)
    rows = []
    for index in group:
        class_id = _class_id_of(radar["label_id"][index])
        stats = stencil_classes.get(class_id) if class_id in stencil_classes else None
        if stats is None or not stats.get("available"):
            rows.append(_passthrough_row(radar, index))
            continue
        covariance = _regularize_covariance(stats["cov_3d"])
        x_parent = float(radar["x_cc"][index])
        y_parent = float(radar["y_cc"][index])
        vr_parent = float(radar["vr_sc"][index])
        amp_mu = float(stats["log1p_amp_mean"])
        amp_sigma = float(stats["log1p_amp_std"])
        for _ in range(per_point):
            delta = np.zeros(3)
            for _attempt in range(16):
                candidate = rng.multivariate_normal(np.zeros(3), covariance)
                x = x_parent + candidate[0]
                y = y_parent + candidate[1]
                r = float(np.hypot(x, y))
                phi = float(np.arctan2(y, x))
                vr = float(
                    np.clip(
                        vr_parent + candidate[2],
                        -RGD_MAX_DOPPLER_MPS,
                        RGD_MAX_DOPPLER_MPS,
                    )
                )
                if (
                    RGD_RANGE_MIN_M <= r <= RGD_RANGE_MAX_M
                    and abs(phi) <= RGD_FOV_RAD
                ):
                    delta = candidate
                    break
            if amp_sigma > 0.0:
                amp = float(
                    np.expm1(amp_mu + amp_sigma * rng.standard_normal())
                )
                amp = max(amp, 1.0e-4)
            else:
                amp = float(radar["amp"][index])
            rows.append(
                (
                    index,
                    float(delta[0]),
                    float(delta[1]),
                    x_parent + float(delta[0]),
                    y_parent + float(delta[1]),
                    vr_parent + float(delta[2]),
                    amp,
                )
            )
    return rows


def _copy_rows(radar, rows, sequence_stem):
    """Materialize the structured output array from parent indices + deltas."""

    out = np.empty(len(rows), dtype=DENSIFIED_DTYPE)
    for row_index, (parent_index, _dx, _dy, x, y, vr, amp) in enumerate(rows):
        parent = radar[parent_index]
        row = out[row_index]
        row["frame"] = parent["frame"]
        row["frame_timestamp"] = parent["frame_timestamp"]
        row["timestamp"] = parent["timestamp"]
        row["sensor"] = parent["sensor"]
        row["x_cc"] = x
        row["y_cc"] = y
        row["r_sc"] = float(np.hypot(x, y))
        row["phi_sc"] = float(np.arctan2(y, x))
        row["vr_sc"] = vr
        row["amp"] = amp
        row["uuid"] = str(
            uuid.uuid5(
                uuid.NAMESPACE_OID,
                f"carla-radar-densified-{sequence_stem}-{row_index}",
            )
        ).encode("ascii")
        row["label_id"] = parent["label_id"]
        row["instance_id"] = parent["instance_id"]
        row["source"] = parent["source"]
        row["parent_object_id"] = parent["parent_object_id"]
        row["reflector_id"] = parent["reflector_id"]
        row["bounce_type"] = parent["bounce_type"]
        row["bounce_order"] = parent["bounce_order"]
        row["path_length_m"] = parent["path_length_m"]
    return out


def _sensor_sort_key(values):
    """Map sensor identifiers (bytes/strings/ints) to small integer codes."""

    seen = {}
    result = np.zeros(len(values), dtype=np.int8)
    for index, value in enumerate(values):
        if isinstance(value, bytes):
            text = value.decode("utf-8", errors="replace").strip().lower()
        else:
            text = str(value).strip().lower()
        if text not in seen:
            seen[text] = len(seen)
        result[index] = seen[text]
    return result


def run_densify(args):
    carla_root = Path(args.carla_input).resolve()
    stencil_path = Path(args.stencil)
    output_root = Path(args.output).resolve()
    with stencil_path.open("r", encoding="utf-8") as handle:
        stencil = json.load(handle)
    # JSON keys are strings; convert once so per-point class lookups (ints)
    # match.
    stencil_classes = {
        int(key): value for key, value in stencil.get("classes", {}).items()
    }
    input_paths = sorted(carla_root.rglob("*.h5")) + sorted(carla_root.rglob("*.hdf5"))
    if not input_paths:
        raise FileNotFoundError(f"No H5 files found below {carla_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    summary = []
    for file_index, path in enumerate(input_paths, start=1):
        relative = path.relative_to(carla_root)
        out_path = output_root / relative
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(path, "r") as handle:
            radar = np.copy(handle["radar"])
            attrs = {key: value for key, value in handle.attrs.items()}
        frame_values = np.asarray(radar["frame"], dtype=np.int64)
        sensor_values = _sensor_sort_key(radar["sensor"])
        order = np.lexsort((np.arange(len(radar)), frame_values, sensor_values))
        radar = radar[order]
        frame_values = frame_values[order]
        sensor_values = sensor_values[order]
        group_starts = np.concatenate(
            (
                np.array((0,), dtype=np.int64),
                np.flatnonzero(
                    (frame_values[1:] != frame_values[:-1])
                    | (sensor_values[1:] != sensor_values[:-1])
                )
                + 1,
            )
        )
        group_ends = np.concatenate(
            (group_starts[1:], np.array((len(radar),), dtype=np.int64))
        )
        rows = []
        for group_index, (start, end) in enumerate(zip(group_starts, group_ends)):
            group = list(range(int(start), int(end)))
            rows.extend(
                _densify_frame(
                    group,
                    radar,
                    stencil_classes,
                    args.points_per_frame,
                    args.seed * 1_000_003 + file_index * 9_176 + group_index * 37,
                )
            )
        densified = _copy_rows(radar, rows, path.stem)
        with h5py.File(out_path, "w") as handle:
            handle.create_dataset(
                "radar",
                data=densified,
                compression="gzip",
                shuffle=True,
            )
            for key, value in attrs.items():
                handle.attrs[key] = value
            handle.attrs["densified"] = True
            handle.attrs["densification_stencil"] = str(stencil_path)
            handle.attrs["densification_points_per_frame"] = float(
                args.points_per_frame
            )
        labeled = densified["label_id"] > 0
        real = int(np.count_nonzero(labeled & (densified["label_id"] % 10 == 1)))
        ghost = int(np.count_nonzero(labeled & (densified["label_id"] % 10 != 1)))
        summary.append(
            {
                "source": str(relative),
                "output": str(out_path.relative_to(output_root)),
                "input_points": int(len(radar)),
                "output_points": int(len(densified)),
                "real_points": real,
                "ghost_points": ghost,
                "frames": int(len(group_starts)),
            }
        )
        print(
            f"[{file_index}/{len(input_paths)}] {relative}: "
            f"{len(radar)} -> {len(densified)} points "
            f"(real={real}, ghost={ghost})"
        )
    report = {
        "schema_version": 1,
        "stencil": str(stencil_path),
        "points_per_frame": float(args.points_per_frame),
        "seed": args.seed,
        "files": summary,
    }
    with (output_root / "densification_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    total_in = sum(record["input_points"] for record in summary)
    total_out = sum(record["output_points"] for record in summary)
    print(
        f"Densified {len(summary)} sequences: {total_in} -> {total_out} points "
        f"in {output_root}"
    )


def main():
    args = parse_args()
    if args.command == "stencil":
        run_stencil(args)
    elif args.command == "densify":
        run_densify(args)
    else:  # pragma: no cover
        raise SystemExit(f"unknown command {args.command!r}")


if __name__ == "__main__":
    main()
