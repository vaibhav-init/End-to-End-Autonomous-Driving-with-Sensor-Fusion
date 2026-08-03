#!/usr/bin/env python3
"""Prepare Radar Ghost Dataset v1.1 H5 files for point-set training."""

import argparse
import hashlib
import json
from pathlib import Path
import re

import h5py
import numpy as np

from radar.ghost_detection.features import (
    FEATURE_NAMES,
    FEATURE_SCHEMA_VERSION,
)
from radar.ghost_detection.labels import decode_label_arrays


SCENARIO_PATTERN = re.compile(r"scenario[-_](\d+)", re.IGNORECASE)
STANDARD_SENSOR_CODES = {"left": 0, "right": 1, "front": 2}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="unpacked original/ or virtual/")
    parser.add_argument("--output", required=True, help="prepared artifact directory")
    parser.add_argument(
        "--split-mode",
        choices=("official", "scenario_grouped", "all_train"),
        default="official",
        help="official benchmark split or a strict scenario-disjoint split",
    )
    parser.add_argument(
        "--include-sketchy",
        action="store_true",
        help="use negative four-digit unsure labels as supervised labels",
    )
    parser.add_argument(
        "--exclude-undecided",
        action="store_true",
        help="ignore annotated multipath whose bounce type/order is undecided",
    )
    return parser.parse_args()


def _field(data, name, aliases=(), required=True, default=None):
    names = data.dtype.names or ()
    for candidate in (name,) + tuple(aliases):
        if candidate in names:
            return np.asarray(data[candidate])
    if required:
        raise ValueError(
            f"Radar dataset is missing required column {name!r}; has {names}"
        )
    if callable(default):
        return default(len(data))
    return np.full(len(data), default)


def _sensor_strings(values):
    result = []
    for value in values:
        if isinstance(value, bytes):
            text = value.decode("utf-8", errors="replace")
        else:
            text = str(value)
        result.append(text.strip().lower())
    return result


def _sensor_codes(values):
    texts = _sensor_strings(values)
    unknown = sorted(set(texts) - set(STANDARD_SENSOR_CODES))
    mapping = dict(STANDARD_SENSOR_CODES)
    for index, name in enumerate(unknown, start=3):
        mapping[name] = index
    return np.asarray([mapping[name] for name in texts], dtype=np.int8), mapping


def _attribute_text(attributes, name, default=None):
    value = attributes.get(name, default)
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if value is None:
        return None
    return str(value)


def _scenario_name(path):
    match = SCENARIO_PATTERN.search(path.name)
    return f"scenario-{int(match.group(1)):02d}" if match else path.stem


def _official_split(path):
    for part in reversed(path.parts):
        lowered = part.lower()
        if lowered in ("train", "val", "test"):
            return lowered
    match = re.search(r"_(train|val|test)(?:\.h5)?$", path.name.lower())
    if match:
        return match.group(1)
    raise ValueError(
        f"Cannot infer official split for {path}; use --split-mode all_train "
        "or place files under train/val/test directories"
    )


def _scenario_split_map(paths):
    scenarios = sorted(
        {_scenario_name(path) for path in paths},
        key=lambda value: hashlib.sha256(value.encode("utf-8")).hexdigest(),
    )
    if len(scenarios) < 3:
        raise ValueError(
            "scenario_grouped needs at least three distinct scenario IDs"
        )
    train_count = max(1, int(round(0.70 * len(scenarios))))
    val_count = max(1, int(round(0.15 * len(scenarios))))
    if train_count + val_count >= len(scenarios):
        train_count = len(scenarios) - 2
        val_count = 1
    mapping = {}
    for index, scenario in enumerate(scenarios):
        if index < train_count:
            split = "train"
        elif index < train_count + val_count:
            split = "val"
        else:
            split = "test"
        mapping[scenario] = split
    return mapping


def _split_for(path, mode, scenario_splits=None):
    if mode == "all_train":
        return "train"
    if mode == "official":
        return _official_split(path)
    return scenario_splits[_scenario_name(path)]


def _safe_output_name(path, input_root):
    relative = str(path.relative_to(input_root))
    suffix = hashlib.sha256(relative.encode("utf-8")).hexdigest()[:10]
    return f"{path.stem}-{suffix}.npz"


def prepare_file(path, input_root, output_root, args, scenario_splits=None):
    with h5py.File(path, "r") as handle:
        if "radar" not in handle:
            raise ValueError(f"H5 file has no 'radar' dataset: {path}")
        radar = np.copy(handle["radar"])
        source_metadata = {
            "scene": _attribute_text(handle.attrs, "scene"),
            "town": _attribute_text(handle.attrs, "town"),
            "weather": _attribute_text(handle.attrs, "weather"),
            "seed": int(handle.attrs["seed"]) if "seed" in handle.attrs else None,
        }
    if radar.dtype.names is None:
        raise ValueError(f"Radar entry must be a NumPy structured array: {path}")

    frame = _field(radar, "frame").astype(np.int64, copy=False)
    frame_timestamp = _field(
        radar,
        "frame_timestamp",
        aliases=("timestamp",),
    ).astype(np.float64, copy=False)
    sensor_raw = _field(
        radar,
        "sensor",
        required=False,
        default=lambda size: np.full(size, "front", dtype="U5"),
    )
    sensor, sensor_mapping = _sensor_codes(sensor_raw)
    range_m = _field(radar, "r_sc", aliases=("range", "range_m")).astype(
        np.float32,
        copy=False,
    )
    azimuth = _field(
        radar,
        "phi_sc",
        aliases=("azimuth", "azimuth_rad"),
    ).astype(np.float32, copy=False)
    velocity = _field(
        radar,
        "vr_sc",
        aliases=("doppler", "radial_velocity_mps"),
    ).astype(np.float32, copy=False)
    amplitude = _field(radar, "amp", aliases=("amplitude",)).astype(
        np.float32,
        copy=False,
    )
    label_id = _field(radar, "label_id").astype(np.int32, copy=False)
    decoded = decode_label_arrays(
        label_id,
        include_sketchy=args.include_sketchy,
        include_undecided=not args.exclude_undecided,
    )
    finite = (
        np.isfinite(frame_timestamp)
        & np.isfinite(range_m)
        & np.isfinite(azimuth)
        & np.isfinite(velocity)
        & np.isfinite(amplitude)
        & (range_m > 0.0)
    )
    order = np.lexsort((frame_timestamp[finite], frame[finite], sensor[finite]))

    arrays = {
        "frame": frame[finite][order],
        "frame_timestamp": frame_timestamp[finite][order].astype(np.float32),
        "sensor": sensor[finite][order],
        "r_sc": range_m[finite][order],
        "phi_sc": azimuth[finite][order],
        "vr_sc": velocity[finite][order],
        "amp": amplitude[finite][order],
        "label_id": label_id[finite][order],
    }
    arrays.update(
        {
            name: values[finite][order]
            for name, values in decoded.items()
        }
    )
    for optional in ("x_cc", "y_cc", "instance_id", "group"):
        if optional in radar.dtype.names:
            arrays[optional] = np.asarray(radar[optional])[finite][order]

    output_name = _safe_output_name(path, input_root)
    output_relative = Path("sequences") / output_name
    np.savez_compressed(output_root / output_relative, **arrays)
    sensor_frames = {}
    for sensor_code in np.unique(arrays["sensor"]):
        frames = np.unique(
            arrays["frame"][arrays["sensor"] == sensor_code]
        )
        sensor_frames[str(int(sensor_code))] = frames.astype(int).tolist()
    target = arrays["target"]
    return {
        "name": path.stem,
        "path": str(output_relative),
        "source_path": str(path.relative_to(input_root)),
        "scenario": source_metadata["scene"] or _scenario_name(path),
        "sequence_id": _scenario_name(path),
        "town": source_metadata["town"],
        "weather": source_metadata["weather"],
        "seed": source_metadata["seed"],
        "split": _split_for(path, args.split_mode, scenario_splits),
        "points": int(len(target)),
        "labeled_points": int(np.count_nonzero(target >= 0)),
        "real_points": int(np.count_nonzero(target == 0)),
        "ghost_points": int(np.count_nonzero(target == 1)),
        "sensor_frames": sensor_frames,
        "sensor_mapping": sensor_mapping,
    }


def main():
    args = parse_args()
    input_root = Path(args.input).resolve()
    output_root = Path(args.output).resolve()
    paths = sorted(input_root.rglob("*.h5")) + sorted(input_root.rglob("*.hdf5"))
    paths = sorted(set(paths))
    if not paths:
        raise FileNotFoundError(f"No H5 files found below {input_root}")
    scenario_splits = (
        _scenario_split_map(paths)
        if args.split_mode == "scenario_grouped"
        else None
    )
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "sequences").mkdir(parents=True, exist_ok=True)

    sequences = []
    failures = []
    for index, path in enumerate(paths, start=1):
        try:
            record = prepare_file(
                path,
                input_root,
                output_root,
                args,
                scenario_splits=scenario_splits,
            )
            sequences.append(record)
            print(
                f"[{index}/{len(paths)}] {record['split']:5s} {path.name}: "
                f"real={record['real_points']} ghost={record['ghost_points']}"
            )
        except Exception as exc:
            failures.append(
                {
                    "path": str(path),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            print(f"[{index}/{len(paths)}] FAILED {path}: {exc}")

    manifest = {
        "schema_version": 1,
        "source": "Radar Ghost Dataset v1.1 or compatible CARLA export",
        "input_root": str(input_root),
        "split_mode": args.split_mode,
        "include_sketchy": bool(args.include_sketchy),
        "include_undecided": not bool(args.exclude_undecided),
        "scenario_splits": scenario_splits,
        "feature_schema": FEATURE_SCHEMA_VERSION,
        "feature_names": list(FEATURE_NAMES),
        "sequences": sequences,
        "failures": failures,
    }
    with (output_root / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"Prepared {len(sequences)} sequences in {output_root}")
    if failures:
        raise RuntimeError(
            f"{len(failures)} H5 files failed; inspect manifest.json before training"
        )


if __name__ == "__main__":
    main()
