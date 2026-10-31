#!/usr/bin/env python3
"""Build a scenario-disjoint train/val split from a prepared ghost dataset.

The official Radar Ghost Dataset split is sequence-level: its val sequences
often come from the same scenarios as train, so validation scores are
optimistic and do not predict the unseen-scenario test split. This utility
keeps the official test split untouched, pools the official train+val
sequences, and holds out whole scenarios for validation using the same
deterministic hash ordering as ``prepare_radar_ghost_dataset.py``.

The output directory contains a new manifest and a symlink to the source
``sequences`` directory, so no point data is copied.

Example:
  python3 make_scenario_split.py \
    --input artifacts/ghost_real_official_v2 \
    --output artifacts/ghost_real_scenario_val \
    --val-fraction 0.2
"""

import argparse
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path
import re


SCENARIO_PATTERN = re.compile(r"scenario[-_](\d+)", re.IGNORECASE)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="prepared dataset directory")
    parser.add_argument("--output", required=True, help="new prepared directory")
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.2,
        help="fraction of pooled scenarios held out for validation",
    )
    parser.add_argument(
        "--val-scenarios",
        nargs="*",
        default=None,
        help="explicit scenario ids to hold out, overriding --val-fraction",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="replace an existing output directory",
    )
    return parser.parse_args()


def scenario_key(record):
    match = SCENARIO_PATTERN.search(record.get("name", ""))
    if match:
        return f"scenario-{int(match.group(1)):02d}"
    value = record.get("scenario")
    if value:
        match = SCENARIO_PATTERN.search(str(value))
        if match:
            return f"scenario-{int(match.group(1)):02d}"
    raise ValueError(f"cannot infer scenario for {record.get('name')!r}")


def main():
    args = parse_args()
    if not 0.0 < args.val_fraction < 1.0:
        raise ValueError("--val-fraction must be in (0, 1)")
    source = Path(args.input)
    output = Path(args.output)
    with (source / "manifest.json").open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)

    pooled = [
        record
        for record in manifest["sequences"]
        if record.get("split") in ("train", "val")
    ]
    test = [
        record for record in manifest["sequences"] if record.get("split") == "test"
    ]
    if not pooled:
        raise ValueError("no train/val sequences to re-split")

    by_scenario = defaultdict(list)
    for record in pooled:
        by_scenario[scenario_key(record)].append(record)

    scenarios = sorted(
        by_scenario,
        key=lambda value: hashlib.sha256(value.encode("utf-8")).hexdigest(),
    )
    if args.val_scenarios:
        val_scenarios = {
            f"scenario-{int(value):02d}" if value.isdigit() else value
            for value in args.val_scenarios
        }
        missing = val_scenarios - set(scenarios)
        if missing:
            raise ValueError(f"unknown scenarios: {sorted(missing)}")
    else:
        count = max(1, int(round(args.val_fraction * len(scenarios))))
        val_scenarios = set(scenarios[:count])

    for record in pooled:
        record["split"] = (
            "val" if scenario_key(record) in val_scenarios else "train"
        )

    output.mkdir(parents=True, exist_ok=True)
    sequences_link = output / "sequences"
    if sequences_link.exists() or sequences_link.is_symlink():
        if not args.force:
            raise FileExistsError(
                f"{sequences_link} exists; pass --force to replace it"
            )
        if sequences_link.is_symlink() or sequences_link.is_file():
            sequences_link.unlink()
        else:
            raise ValueError(f"{sequences_link} is a real directory; refusing to remove")
    os.symlink((source / "sequences").resolve(), sequences_link)

    manifest["sequences"] = pooled + test
    manifest["split_protocol"] = {
        "mode": "scenario_disjoint",
        "source": str(source.resolve()),
        "val_scenarios": sorted(val_scenarios),
        "val_fraction": args.val_fraction,
        "official_test_preserved": True,
    }
    with (output / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")

    for split in ("train", "val", "test"):
        records = [record for record in manifest["sequences"] if record["split"] == split]
        frames = sum(
            len(values)
            for record in records
            for values in record.get("sensor_frames", {}).values()
        )
        print(
            f"{split:5s} sequences={len(records):3d} frames={frames:6d} "
            f"scenarios={len({scenario_key(record) for record in records})}"
        )
    print(f"val scenarios: {', '.join(sorted(val_scenarios))}")
    print(f"wrote {output / 'manifest.json'}")


if __name__ == "__main__":
    main()
