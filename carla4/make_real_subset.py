#!/usr/bin/env python3
"""Create a label-scarce training subset from a prepared real dataset.

Keeps the official validation and test splits untouched and samples a small
number of training sequences, stratified across scenarios and target classes.
This is the regime where synthetic pretraining should matter: the same test
split, but only a handful of real labelled recordings.

Example:
  python3 make_real_subset.py \
    --input artifacts/ghost_real_official_v2 \
    --output artifacts/ghost_real_subset_5 \
    --train-sequences 5 --seed 42 --force
"""

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
import random
import re


CLASS_PATTERN = re.compile(r"_(ped|cycl|car|truck|bus)_", re.IGNORECASE)
SCENARIO_PATTERN = re.compile(r"scenario[-_](\d+)", re.IGNORECASE)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--train-sequences", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def scenario_of(record):
    match = SCENARIO_PATTERN.search(record.get("name", ""))
    return f"scenario-{int(match.group(1)):02d}" if match else "scenario-unknown"


def class_of(record):
    match = CLASS_PATTERN.search(record.get("name", ""))
    return match.group(1).lower() if match else "unknown"


def main():
    args = parse_args()
    source = Path(args.input)
    output = Path(args.output)
    with (source / "manifest.json").open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)

    train = [
        record for record in manifest["sequences"] if record.get("split") == "train"
    ]
    if args.train_sequences < 1 or args.train_sequences > len(train):
        raise ValueError(
            f"--train-sequences must be in [1, {len(train)}]"
        )

    rng = random.Random(args.seed)
    by_scenario = defaultdict(list)
    for record in train:
        by_scenario[scenario_of(record)].append(record)
    scenarios = sorted(by_scenario)
    rng.shuffle(scenarios)
    for records in by_scenario.values():
        rng.shuffle(records)

    selected = []
    class_counts = defaultdict(int)
    while len(selected) < args.train_sequences:
        made_progress = False
        for scenario in scenarios:
            if len(selected) >= args.train_sequences:
                break
            available = [
                record
                for record in by_scenario[scenario]
                if record not in selected
            ]
            if not available:
                continue
            available.sort(key=lambda record: class_counts[class_of(record)])
            chosen = available[0]
            selected.append(chosen)
            class_counts[class_of(chosen)] += 1
            made_progress = True
        if not made_progress:
            break

    selected_names = {record["name"] for record in selected}
    for record in manifest["sequences"]:
        if record.get("split") == "train":
            record["split"] = (
                "train" if record["name"] in selected_names else "unused"
            )
    manifest["subset_protocol"] = {
        "source": str(source.resolve()),
        "train_sequences": args.train_sequences,
        "seed": args.seed,
        "selected": sorted(selected_names),
        "class_counts": dict(class_counts),
    }

    output.mkdir(parents=True, exist_ok=True)
    link = output / "sequences"
    if link.exists() or link.is_symlink():
        if not args.force:
            raise FileExistsError(f"{link} exists; pass --force to replace it")
        link.unlink()
    os.symlink((source / "sequences").resolve(), link)
    with (output / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print(
        f"train={len(selected)} sequences "
        f"({dict(class_counts)}) across "
        f"{len({scenario_of(record) for record in selected})} scenarios"
    )
    for record in selected:
        print(f"  {record['name']}")
    print(f"wrote {output / 'manifest.json'}")


if __name__ == "__main__":
    main()
