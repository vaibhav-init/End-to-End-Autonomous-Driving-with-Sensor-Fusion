#!/usr/bin/env python3
"""Summarise a collected throttle/brake dataset before training on it.

Exists so the numbers that decide whether a collection is usable are one
command rather than a multi-line snippet pasted into a shell -- where `import`
is ImageMagick's screen-capture tool, which grabs the pointer and hangs the
terminal.

The four that matter:

* **braking while moving** -- the model is learning when to slow down. A
  dataset dominated by free driving is why the shipped model never braked.
  Frames held at a standstill are excluded: a stationary car reports full
  brake, which would count queueing as if it were deceleration.
* **target detected** -- how often the radar had anything to report. Near zero
  means the sensor is misconfigured, not that the road was empty.
* **stopped frames** -- training downsamples these to ~15%, so a run that
  spends its time queueing yields far less than its row count suggests.
* **ghost selected** -- how often a multipath ghost was the controlling
  target. This is the phenomenon the filter is meant to remove, and the
  within-30 m share is the part of it that can actually provoke a brake.
"""

import argparse
import json
import os

import pandas as pd


def summarise(directory):
    csv_path = os.path.join(directory, "data.csv")
    frame = pd.read_csv(csv_path)
    total = len(frame)

    config_path = os.path.join(directory, "dataset_config.json")
    config = {}
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as handle:
            config = json.load(handle)

    def fraction(mask):
        return float(mask.mean()) if total else 0.0

    radar_range = float(config.get("radar_range_m", 100.0))
    detected = fraction(frame["distance_t-0"] < radar_range * 0.99)
    stopped_mask = frame["ego_speed_now"] < 0.3
    stopped = fraction(stopped_mask)
    # Braking *while moving* is the only part the model can learn a
    # deceleration from. A car held at a standstill also reports full brake,
    # so counting every braking frame conflates "slowing down" with "parked
    # behind a queue" and flatters the dataset by roughly the stopped
    # fraction.
    braking = fraction((frame["autopilot_brake"] > 0.1) & ~stopped_mask)
    ghost = ghost_near = None
    if "radar_selected_source" in frame:
        ghost_mask = frame["radar_selected_source"] == "ghost"
        ghost = fraction(ghost_mask)
        # A ghost only matters if it is close enough to make the controller
        # lift off or brake. One at 80 m while cruising changes nothing, so
        # the total ghost rate overstates how much there is to fix.
        ghost_near = fraction(ghost_mask & (frame["distance_t-0"] < 30.0))

    print(f"  dataset          {csv_path}")
    for key in ("town", "radar_backend", "radar_profile", "weather_mode",
                "leading_distance_m", "ignore_lights_pct", "scenarios"):
        if config.get(key) is not None:
            print(f"  {key:16s} {config[key]}")
    print(f"  rows             {total:,}")
    print(f"  target detected  {detected:.3f}")
    print(f"  braking (moving) {braking:.3f}")
    print(f"  stopped frames   {stopped:.3f}")
    if ghost is not None:
        print(f"  ghost selected   {ghost:.3f}")
        print(f"  ghost within 30m {ghost_near:.3f}  ({int(ghost_near * total):,} frames)")

    # Rules of thumb, not thresholds anyone measured -- they exist to catch a
    # collection that is obviously not worth training on.
    warnings = []
    if braking < 0.04:
        warnings.append(
            "almost no braking: the model cannot learn to slow down. Check "
            "--leading-distance-m and whether traffic is dense enough"
        )
    if detected < 0.20:
        warnings.append(
            "the radar rarely saw anything. Check --radar-points-per-second "
            "and the profile before trusting this data"
        )
    if stopped > 0.35:
        warnings.append(
            f"{stopped:.0%} of frames are stationary. Training keeps ~15% of "
            "them, so the usable dataset is much smaller than the row count"
        )
    for warning in warnings:
        print(f"  WARNING: {warning}")
    return warnings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "directories",
        nargs="+",
        help="collection output directories (each holding data.csv)",
    )
    args = parser.parse_args()
    for index, directory in enumerate(args.directories):
        if index:
            print()
        summarise(directory)


if __name__ == "__main__":
    main()
