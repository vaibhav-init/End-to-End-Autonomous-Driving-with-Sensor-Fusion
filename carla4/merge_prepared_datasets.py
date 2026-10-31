#!/usr/bin/env python3
"""Merge prepared ghost datasets into one manifest without copying points.

Sequence paths are rewritten to absolute paths, so the output directory only
contains a manifest. Names are prefixed with the source directory name to keep
them unique. All inputs must share the same feature schema.

Example:
  python3 merge_prepared_datasets.py \
    --input artifacts/ghost_carla_calibrated_v7s_prepared \
            artifacts/carla_ghost_measphenom_v3_prepared \
    --output artifacts/ghost_synth_combined
"""

import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--prefix",
        action="store_true",
        help="prefix sequence names with the source directory (default: on)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    sources = [Path(path) for path in args.input]
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    merged = None
    sequences = []
    for source in sources:
        with (source / "manifest.json").open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        if merged is None:
            merged = {
                key: value
                for key, value in manifest.items()
                if key not in ("sequences", "split_protocol", "subset_protocol")
            }
        elif manifest.get("feature_schema") != merged.get("feature_schema"):
            raise ValueError(
                f"{source}: feature schema {manifest.get('feature_schema')!r} "
                f"does not match {merged.get('feature_schema')!r}"
            )
        prefix = source.name
        for record in manifest["sequences"]:
            record = dict(record)
            record["name"] = f"{prefix}__{record['name']}"
            record["source_dataset"] = str(source.resolve())
            record["path"] = str((source / record["path"]).resolve())
            sequences.append(record)
        print(f"{source}: {len(manifest['sequences'])} sequences")

    merged["sequences"] = sequences
    merged["merge_protocol"] = {
        "sources": [str(source.resolve()) for source in sources],
        "sequence_count": len(sequences),
    }
    with (output / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(merged, handle, indent=2, sort_keys=True)
        handle.write("\n")

    for split in ("train", "val", "test"):
        selected = [record for record in sequences if record.get("split") == split]
        frames = sum(
            len(values)
            for record in selected
            for values in record.get("sensor_frames", {}).values()
        )
        print(f"{split:5s} sequences={len(selected):3d} frames={frames:6d}")
    print(f"wrote {output / 'manifest.json'}")


if __name__ == "__main__":
    main()
