#!/usr/bin/env python3
"""Distribution-level fidelity of a synthetic ghost export against real RGD.

For every statistic in `radar/ghost_detection/statistics.py` this computes the
1-D Wasserstein distance between the real test split and the synthetic set,
and next to it the same distance between the real test split and the real
train split. That second number is the floor: two halves of the same real
sensor already differ by that much, so a synthetic set is "indistinguishable
at this statistic" when its distance sits near the floor and "far" when the
ratio is large. The table is the simulator-fidelity result; a detector-based
check of the same question is `evaluate_cross_domain.py`.

Nothing here is fitted to the test split, and nothing here reads labels
other than the real/ghost target and the bounce family that both domains
already carry.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np

from calibrate_ghost_profile import load_split
from radar.ghost_detection.statistics import (
    FAMILY_NAMES,
    summarize,
    wasserstein_1d,
)


COMPARED = [
    ("points_per_frame", "points per scan", 1.0),
    ("real_per_frame", "labelled real points per scan", 1.0),
    ("ghost_per_frame", "labelled ghost points per scan", 1.0),
    ("ghost_fraction_per_frame", "ghost share of labelled points", 1.0),
    ("background_per_frame", "unlabelled points per scan", 1.0),
    ("frame_median_amp_db", "frame median amplitude [dB, absolute]", 1.0),
    ("real_rel_amp_db", "real amplitude rel. frame median [dB]", 1.0),
    ("ghost_rel_amp_db", "ghost amplitude rel. frame median [dB]", 1.0),
    ("real_range_m", "real point range [m]", 1.0),
    ("ghost_range_m", "ghost point range [m]", 1.0),
    ("real_abs_doppler_mps", "real |Doppler| [m/s]", 1.0),
    ("ghost_abs_doppler_mps", "ghost |Doppler| [m/s]", 1.0),
    ("object_points", "points per real object", 1.0),
    ("object_range_spread_m", "within-object range spread [m]", 1.0),
    ("object_doppler_std_mps", "within-object Doppler std [m/s]", 1.0),
    ("ghost_lifetime_frames", "ghost run length [scans]", 1.0),
    ("ghost_fading_std_db", "ghost per-run amplitude std [dB]", 1.0),
]
for _family in FAMILY_NAMES[:3]:
    COMPARED += [
        (f"{_family}_delta_range_m", f"{_family}: ghost-parent range [m]", 1.0),
        (f"{_family}_delta_azimuth_rad", f"{_family}: ghost-parent azimuth [deg]", 180.0 / math.pi),
        (f"{_family}_delta_amp_db", f"{_family}: ghost-parent amplitude [dB]", 1.0),
        (f"{_family}_doppler_ratio", f"{_family}: ghost/parent Doppler", 1.0),
    ]


def compare(real_test, synthetic, real_reference):
    rows = []
    for key, label, scale in COMPARED:
        a = np.asarray(real_test.get(key, ()), dtype=np.float64) * scale
        b = np.asarray(synthetic.get(key, ()), dtype=np.float64) * scale
        c = np.asarray(real_reference.get(key, ()), dtype=np.float64) * scale
        distance = wasserstein_1d(a, b)
        floor = wasserstein_1d(a, c)
        rows.append(
            {
                "statistic": key,
                "label": label,
                "real_test": summarize(a),
                "synthetic": summarize(b),
                "w1_real_vs_synthetic": distance,
                "w1_real_test_vs_real_train": floor,
                "ratio": (distance / floor) if floor and floor > 1.0e-9 and not math.isnan(distance) else float("nan"),
            }
        )
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--real", required=True, help="prepared real RGD directory")
    parser.add_argument("--synthetic", required=True, help="prepared synthetic directory")
    parser.add_argument("--split", default="test", help="real split to score against")
    parser.add_argument("--synthetic-split", default=None,
                        help="synthetic split (default: same as --split; a set with no "
                             "test split can be scored with its val split)")
    parser.add_argument("--reference-split", default="train",
                        help="real split that sets the floor (default train)")
    parser.add_argument("--amplitude", choices=("auto", "linear", "db"), default="auto")
    parser.add_argument("--output", default=None, help="JSON report path")
    args = parser.parse_args()

    real_test, n_real, _ = load_split(args.real, args.split, args.amplitude)
    # The real data may carry no instance link between ghosts and parents
    # (RGD does not); then the synthetic side must use the same same-class
    # parent so ghost-parent offsets and lifetimes are comparable.
    parent_mode = "auto" if real_test["_meta"]["instance_pairs"] > 0 else "class"
    real_ref, n_ref, _ = load_split(args.real, args.reference_split, args.amplitude, parent_mode)
    synthetic_split = args.synthetic_split or args.split
    synthetic, n_syn, _ = load_split(args.synthetic, synthetic_split, args.amplitude, parent_mode)
    rows = compare(real_test, synthetic, real_ref)

    print("=" * 110)
    print("GHOST FIDELITY: real test vs synthetic, with the real-vs-real floor")
    print("=" * 110)
    print(f"  real {args.real} [{args.split}] {n_real} seq, amp unit {real_test['_meta']['amplitude_unit']}, "
          f"parent link instance={real_test['_meta']['instance_pairs']:,} class={real_test['_meta']['class_pairs']:,}")
    print(f"  synthetic {args.synthetic} [{synthetic_split}] {n_syn} seq, amp unit {synthetic['_meta']['amplitude_unit']}, "
          f"parent mode {parent_mode}")
    print(f"  floor: real [{args.reference_split}] {n_ref} seq")
    print()
    print(f"  {'statistic':<44} {'real med':>10} {'synth med':>10} {'W1':>9} {'floor':>9} {'ratio':>7}")
    print(f"  {'-' * 44} {'-' * 10} {'-' * 10} {'-' * 9} {'-' * 9} {'-' * 7}")
    for row in rows:
        real_med = row["real_test"].get("median", float("nan"))
        syn_med = row["synthetic"].get("median", float("nan"))
        if row["real_test"].get("count", 0) == 0 and row["synthetic"].get("count", 0) == 0:
            continue
        print(f"  {row['label']:<44} {real_med:>10.3f} {syn_med:>10.3f} "
              f"{row['w1_real_vs_synthetic']:>9.3f} {row['w1_real_test_vs_real_train']:>9.3f} "
              f"{row['ratio']:>7.2f}")
    print()
    print("  ratio ~1: synthetic is as close to real test as real train is (matched).")
    print("  ratio >>1: the statistic still separates the domains; that is the gap to close.")
    print("  detector-based check: evaluate_cross_domain.py with the real-trained checkpoint.")
    print("=" * 110)

    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "real": str(Path(args.real).resolve()),
                    "synthetic": str(Path(args.synthetic).resolve()),
                    "split": args.split,
                    "reference_split": args.reference_split,
                    "real_meta": real_test["_meta"],
                    "synthetic_meta": synthetic["_meta"],
                    "rows": rows,
                },
                handle, indent=2, sort_keys=True,
            )
        print(f"  report: {path}")


if __name__ == "__main__":
    main()
