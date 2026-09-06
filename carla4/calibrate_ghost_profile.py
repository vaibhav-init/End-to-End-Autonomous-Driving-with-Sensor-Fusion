#!/usr/bin/env python3
"""Fit the radar profile's unfitted priors to the Radar Ghost Dataset.

Reads the prepared RGD **train** split only (val and test are never opened,
so the fidelity numbers measured on them afterwards are honest), measures
the statistics that the simulator's priors stand in for, and writes:

  <output>/rgd_statistics.json        every measured distribution, summarised
  <output>/calibrated_overrides.json  a --radar-config override file with the
                                      parameters that map directly onto the
                                      measurements, each with its derivation

What can be fitted from a labelled point cloud, and how:

  points_per_object_mean         median labelled points per real object per scan
  point_footprint_scale          measured within-object range spread against
                                 the class footprint prior (uniform depth D
                                 has a 10-90 percentile spread of 0.8 D)
  micro_doppler_scale            measured within-object Doppler std against
                                 the prior's implied std for the same class
  multipath_second_order_loss_db ghost-minus-parent amplitude, with the
                                 model's own 40 log10 spreading term and an
                                 assumed reflection loss removed
  multipath_third_order_loss_db  same for the third-order family
  multipath_fading_std_db        std of a ghost run's per-scan amplitude
  multipath_fading_correlation   lag-1 autocorrelation of the same series

What cannot be fitted this way and is only reported: the ghost-to-real
ratio (the image-method geometry sets it; use --radar-ghost-rate-scale to
match), ghost lifetimes, per-family range and azimuth offsets (a check on
the geometry, not a knob), and anything about clutter or dropout, which RGD
does not label.

Amplitude units: RGD stores an amplitude; whether it is linear or already dB
is detected and printed. The relative statistics used for fitting cancel a
constant unit mismatch; absolute amplitude never enters a fitted parameter.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np

from radar.extended_target import CLASS_FOOTPRINT_M, MICRO_DOPPLER_AMPLITUDE, MICRO_DOPPLER_NOISE_MPS
from radar.ghost_detection.statistics import (
    CLASS_NAMES,
    merge_statistics,
    sequence_statistics,
    summarize,
    summarize_statistics,
)
from radar.realistic_core import load_realistic_radar_config


# Reflection loss the image method charges at a typical incidence; only the
# residual after this and spreading is attributed to the per-bounce prior.
ASSUMED_REFLECTION_LOSS_DB = 2.0


def load_split(prepared_dir, split, amplitude_mode, parent_mode="auto"):
    root = Path(prepared_dir)
    with (root / "manifest.json").open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    records = [r for r in manifest.get("sequences", ()) if r.get("split") == split]
    if not records:
        raise ValueError(f"No sequences with split {split!r} in {root}")
    parts = []
    for record in records:
        with np.load(root / record["path"], allow_pickle=False) as archive:
            sequence = {name: np.copy(archive[name]) for name in archive.files}
        parts.append(sequence_statistics(sequence, amplitude_mode, parent_mode))
    return merge_statistics(parts), len(records), manifest


def _implied_micro_doppler_std(class_id):
    low, high = MICRO_DOPPLER_AMPLITUDE.get(int(class_id), (0.05, 0.10))
    amplitude = 0.5 * (low + high)
    return math.sqrt(0.5 * amplitude * amplitude + MICRO_DOPPLER_NOISE_MPS ** 2)


def derive_overrides(stats, base_profile, synthetic_stats=None, base_overrides=None):
    """Map measured statistics onto profile parameters, with derivations.

    Direct fits (points per object, footprint, micro-Doppler, fading,
    bounce loss, ghost points per cluster) come from the real data alone.

    Relative fits need a synthetic reference collected with
    ``base_overrides`` (the overrides that produced it): ghost rate,
    road-user amplitude and ghost amplitude are then corrected by the
    real-minus-synthetic residual on top of the base values, so each
    collect-and-refit round moves the synthetic distribution toward the
    real one instead of restarting from the priors.
    """

    base = load_realistic_radar_config(base_profile)
    base_overrides = dict(base_overrides or {})
    overrides = {}
    notes = {}
    # Closed-loop corrections that must win over the direct fits below.
    relative = {}

    real_rel_amp = stats["real_rel_amp_db"]
    ghost_rel_amp = stats["ghost_rel_amp_db"]
    if synthetic_stats is not None:
        syn_real_rel = synthetic_stats["real_rel_amp_db"]
        syn_ghost_rel = synthetic_stats["ghost_rel_amp_db"]
        if real_rel_amp.size >= 100 and syn_real_rel.size >= 100:
            # Road users sit above the frame median in the synthetic scans;
            # close the gap by raising the static background, which leaves
            # the road-user link budget (car detection range) untouched.
            previous = float(base_overrides.get("static_snr_offset_db", 0.0))
            residual = float(np.median(real_rel_amp)) - float(np.median(syn_real_rel))
            overrides["static_snr_offset_db"] = float(np.clip(round(previous - residual, 2), -40.0, 80.0))
            notes["static_snr_offset_db"] = (
                f"previous {previous:+.2f} dB - (real labelled-real amplitude rel. frame median "
                f"{np.median(real_rel_amp):+.2f} dB - synthetic {np.median(syn_real_rel):+.2f} dB)"
            )
        if ghost_rel_amp.size >= 100 and syn_ghost_rel.size >= 100 and real_rel_amp.size and syn_real_rel.size:
            previous = float(base_overrides.get("ghost_snr_offset_db", 0.0))
            real_gap = float(np.median(ghost_rel_amp)) - float(np.median(real_rel_amp))
            syn_gap = float(np.median(syn_ghost_rel)) - float(np.median(syn_real_rel))
            overrides["ghost_snr_offset_db"] = float(np.clip(round(previous + (real_gap - syn_gap), 2), -40.0, 40.0))
            notes["ghost_snr_offset_db"] = (
                f"previous {previous:+.2f} dB + (real ghost-minus-real median gap {real_gap:+.2f} dB "
                f"- synthetic gap {syn_gap:+.2f} dB)"
            )
        real_fm = stats["frame_median_amp_db"]
        syn_fm = synthetic_stats["frame_median_amp_db"]
        if real_fm.size >= 10 and syn_fm.size >= 10:
            previous = float(base_overrides.get("amplitude_gain_db", 0.0))
            residual = float(np.median(real_fm)) - float(np.median(syn_fm))
            overrides["amplitude_gain_db"] = float(round(previous + residual, 2))
            notes["amplitude_gain_db"] = (
                f"previous {previous:+.2f} dB + (real frame-median amplitude {np.median(real_fm):.2f} dB "
                f"- synthetic {np.median(syn_fm):.2f} dB), 20log10 of the stored linear amplitude"
            )
        real_bg = stats["background_per_frame"]
        syn_bg = synthetic_stats["background_per_frame"]
        if real_bg.size >= 10 and syn_bg.size >= 10 and float(np.mean(syn_bg)) > 0.0:
            previous = float(base_overrides.get("static_points_per_cluster_mean", 1.0))
            value = previous * float(np.mean(real_bg)) / float(np.mean(syn_bg))
            overrides["static_points_per_cluster_mean"] = float(np.clip(round(value, 3), 1.0, 20.0))
            overrides["expand_static_points"] = bool(overrides["static_points_per_cluster_mean"] > 1.0)
            notes["static_points_per_cluster_mean"] = (
                f"previous {previous:.3f} x (mean unlabelled points per scan: real {np.mean(real_bg):.1f} "
                f"/ synthetic {np.mean(syn_bg):.1f})"
            )
        for stat_key, knob, default, lower, upper in (
            ("object_range_spread_m", "point_footprint_scale", 1.0, 0.2, 5.0),
            ("object_doppler_std_mps", "micro_doppler_scale", 1.0, 0.1, 5.0),
            ("ghost_fading_std_db", "multipath_fading_std_db", float(base.multipath_fading_std_db), 0.0, 12.0),
        ):
            real_values = stats[stat_key]
            syn_values = synthetic_stats[stat_key]
            if real_values.size >= 10 and syn_values.size >= 10 and float(np.median(syn_values)) > 1.0e-9:
                previous = float(base_overrides.get(knob, default))
                ratio = float(np.median(real_values)) / float(np.median(syn_values))
                relative[knob] = (
                    float(np.clip(round(previous * ratio, 3), lower, upper)),
                    f"previous {previous:.3f} x (real median {stat_key} {np.median(real_values):.3f} "
                    f"/ synthetic {np.median(syn_values):.3f})",
                )

    cluster_points = stats["ghost_cluster_points"]
    object_points = stats["object_points"]
    if cluster_points.size >= 20 and object_points.size >= 20:
        ratio = float(np.mean(cluster_points)) / max(float(np.mean(object_points)), 1e-6)
        overrides["ghost_points_scale"] = float(np.clip(round(ratio, 3), 0.05, 2.0))
        notes["ghost_points_scale"] = (
            f"mean points per ghost cluster ({np.mean(cluster_points):.2f}) / mean points "
            f"per real object ({np.mean(object_points):.2f}) over {cluster_points.size} clusters"
        )

    real_rate = stats["ghost_clusters_per_object"]
    if synthetic_stats is not None and real_rate.size >= 20:
        synthetic_rate = synthetic_stats["ghost_clusters_per_object"]
        if synthetic_rate.size >= 20 and float(np.mean(synthetic_rate)) > 0.0:
            previous = float(base_overrides.get("ghost_rate_scale", 1.0))
            scale = previous * float(np.mean(real_rate)) / float(np.mean(synthetic_rate))
            overrides["ghost_rate_scale"] = float(np.clip(round(scale, 3), 0.01, 1.0))
            notes["ghost_rate_scale"] = (
                f"previous {previous:.3f} x (mean ghost clusters per real object per scan: "
                f"real {np.mean(real_rate):.3f} / synthetic {np.mean(synthetic_rate):.3f})"
            )

    object_points = stats["object_points"]
    if object_points.size:
        value = float(np.median(object_points))
        overrides["points_per_object_mean"] = max(1.0, round(value, 2))
        notes["points_per_object_mean"] = (
            f"median labelled points per real object per scan over {object_points.size} objects"
        )

    classes = stats["object_class"]
    spreads = stats["object_range_spread_m"]
    if spreads.size and classes.size == spreads.size:
        scales, weights = [], []
        for class_id in np.unique(classes):
            depth_prior = CLASS_FOOTPRINT_M.get(int(class_id), (2.0, 1.0))[0]
            selected = spreads[classes == class_id]
            if selected.size < 10:
                continue
            observed_depth = float(np.median(selected)) / 0.8
            scales.append(observed_depth / depth_prior)
            weights.append(selected.size)
        if scales:
            scale = float(np.average(scales, weights=weights))
            overrides["point_footprint_scale"] = float(np.clip(round(scale, 3), 0.2, 5.0))
            notes["point_footprint_scale"] = (
                "median 10-90 range spread per object / 0.8 against the class depth prior, "
                f"count-weighted over classes {sorted(int(c) for c in np.unique(classes))}"
            )

    doppler_std = stats["object_doppler_std_mps"]
    if doppler_std.size and classes.size == doppler_std.size:
        best_class = None
        best_count = 0
        for class_id in np.unique(classes):
            count = int((classes == class_id).sum())
            if int(class_id) in (1, 2) and count > best_count:
                best_class, best_count = int(class_id), count
        if best_class is not None and best_count >= 10:
            observed = float(np.median(doppler_std[classes == best_class]))
            implied = _implied_micro_doppler_std(best_class)
            overrides["micro_doppler_scale"] = float(np.clip(round(observed / implied, 3), 0.1, 5.0))
            notes["micro_doppler_scale"] = (
                f"median within-object Doppler std for {CLASS_NAMES[best_class]} "
                f"({observed:.3f} m/s) / prior-implied std ({implied:.3f} m/s)"
            )

    for name, key, reflections in (
        ("type1_second", "multipath_second_order_loss_db", 1),
        ("type2_third", "multipath_third_order_loss_db", 2),
    ):
        delta = stats[f"{name}_delta_amp_db"]
        spreading = stats[f"{name}_spreading_db"]
        if delta.size >= 20 and spreading.size >= 20:
            bounce = -float(np.median(delta)) - float(np.median(spreading)) - reflections * ASSUMED_REFLECTION_LOSS_DB
            overrides[key] = float(np.clip(round(bounce, 2), 0.0, 30.0))
            notes[key] = (
                f"-(median ghost-parent amp dB {np.median(delta):+.2f}) - median spreading "
                f"{np.median(spreading):.2f} dB - {reflections}x{ASSUMED_REFLECTION_LOSS_DB} dB "
                f"assumed reflection, from {delta.size} {name} pairs (prior {getattr(base, key):.1f})"
            )
    if "multipath_second_order_loss_db" not in overrides:
        delta = stats["type2_second_delta_amp_db"]
        spreading = stats["type2_second_spreading_db"]
        if delta.size >= 20:
            bounce = -float(np.median(delta)) - float(np.median(spreading)) - ASSUMED_REFLECTION_LOSS_DB
            overrides["multipath_second_order_loss_db"] = float(np.clip(round(bounce, 2), 0.0, 30.0))
            notes["multipath_second_order_loss_db"] = (
                f"from {delta.size} type2_second pairs (no type1_second pairs available)"
            )

    fading = stats["ghost_fading_std_db"]
    if fading.size >= 10:
        overrides["multipath_fading_std_db"] = float(np.clip(round(float(np.median(fading)), 2), 0.0, 12.0))
        notes["multipath_fading_std_db"] = (
            f"median per-run std of the per-scan ghost amplitude over {fading.size} runs "
            f"(prior {base.multipath_fading_std_db:.1f})"
        )
    lag1 = stats["ghost_fading_lag1"]
    if lag1.size >= 10:
        overrides["multipath_fading_correlation"] = float(np.clip(round(float(np.median(lag1)), 3), 0.0, 0.98))
        notes["multipath_fading_correlation"] = (
            f"median lag-1 autocorrelation of per-scan ghost amplitude over {lag1.size} runs "
            f"(prior {base.multipath_fading_correlation:.2f})"
        )
    for knob, (value, note) in relative.items():
        overrides[knob] = value
        notes[knob] = note
    return overrides, notes


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", required=True, help="prepared RGD directory (manifest.json)")
    parser.add_argument("--split", default="train", help="split to fit on; never val or test")
    parser.add_argument("--output", required=True, help="directory for the statistics and overrides")
    parser.add_argument("--base-profile", default="rgd_regime_v1",
                        help="profile whose priors the overrides replace")
    parser.add_argument("--amplitude", choices=("auto", "linear", "db"), default="auto",
                        help="unit of the stored amp field (default: detect)")
    parser.add_argument("--synthetic", default=None,
                        help="prepared synthetic set collected with ghost_rate_scale 1 "
                             "(the smoke sequence); enables the ghost_rate_scale fit")
    parser.add_argument("--synthetic-split", default="train")
    parser.add_argument("--base-overrides", default=None,
                        help="the overrides JSON the synthetic reference was collected with; "
                             "relative fits are corrections on top of it")
    args = parser.parse_args()
    if args.split in ("val", "test"):
        parser.error("fit on train only; val/test are for the fidelity check")
    if args.synthetic and not args.base_overrides:
        parser.error("--synthetic needs --base-overrides (the JSON that produced it)")

    stats, sequences, manifest = load_split(args.data, args.split, args.amplitude)
    meta = stats["_meta"]
    summaries = summarize_statistics(stats)
    synthetic_stats = None
    base_overrides = {}
    if args.synthetic:
        synthetic_stats, _n, _m = load_split(
            args.synthetic, args.synthetic_split, args.amplitude, "class"
        )
        with open(args.base_overrides, "r", encoding="utf-8") as handle:
            base_overrides = json.load(handle)
    overrides, notes = derive_overrides(stats, args.base_profile, synthetic_stats, base_overrides)
    # Carry forward relative knobs that this round could not refit.
    for key in (
        "ghost_rate_scale", "road_user_snr_offset_db", "static_snr_offset_db", "ghost_snr_offset_db",
        "amplitude_gain_db", "static_points_per_cluster_mean", "expand_static_points",
    ):
        if key in base_overrides and key not in overrides:
            overrides[key] = base_overrides[key]
            notes[key] = "carried forward from --base-overrides (not refit this round)"
    overrides["profile_name"] = f"{args.base_profile}_rgd_calibrated"

    labeled = meta["real_points"] + meta["ghost_points"]
    ghost_to_real = meta["ghost_points"] / max(meta["real_points"], 1)
    parent_mode = "instance_id" if meta["instance_pairs"] >= meta["class_pairs"] else "same-class centroid"

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "source": str(Path(args.data).resolve()),
        "split": args.split,
        "sequences": sequences,
        "feature_schema": manifest.get("feature_schema"),
        "amplitude_unit_detected": meta["amplitude_unit"],
        "parent_association": parent_mode,
        "instance_pairs": meta["instance_pairs"],
        "class_pairs": meta["class_pairs"],
        "frames": meta["frames"],
        "points": meta["points"],
        "real_points": meta["real_points"],
        "ghost_points": meta["ghost_points"],
        "ghost_to_real_ratio": ghost_to_real,
        "statistics": summaries,
        "per_class_object_points": {
            CLASS_NAMES.get(int(c), str(int(c))): summarize(stats["object_points"][stats["object_class"] == c])
            for c in np.unique(stats["object_class"])
        } if stats["object_class"].size else {},
    }
    with (output / "rgd_statistics.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
    with (output / "calibrated_overrides.json").open("w", encoding="utf-8") as handle:
        json.dump(overrides, handle, indent=2, sort_keys=True)
    with (output / "calibration_notes.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "base_profile": args.base_profile,
                "assumed_reflection_loss_db": ASSUMED_REFLECTION_LOSS_DB,
                "derivations": notes,
                "not_fitted": {
                    "ghost_to_real_ratio_target": ghost_to_real,
                    "ghost_lifetime_frames": summaries.get("ghost_lifetime_frames"),
                    "note": "match the ratio with --radar-ghost-rate-scale; lifetimes and "
                            "per-family offsets are geometry checks, see evaluate_ghost_fidelity.py",
                },
            },
            handle, indent=2, sort_keys=True,
        )

    print("=" * 72)
    print("RGD PROFILE CALIBRATION")
    print("=" * 72)
    print(f"  source           {args.data} [{args.split}] {sequences} sequences")
    print(f"  amplitude unit   {meta['amplitude_unit']} (detected)")
    print(f"  parent link      {parent_mode}  (instance pairs {meta['instance_pairs']:,}, class pairs {meta['class_pairs']:,})")
    print(f"  points           {meta['points']:,}  labelled real {meta['real_points']:,}  ghost {meta['ghost_points']:,}  "
          f"ghost:real {ghost_to_real:.3f}")
    print(f"  points/frame     median {summaries['points_per_frame'].get('median', float('nan')):.0f}")
    print(f"  points/object    median {summaries['object_points'].get('median', float('nan')):.1f}")
    for name in ("type1_second", "type2_second", "type2_third"):
        block = summaries[f"{name}_delta_amp_db"]
        if block.get("count"):
            print(f"  {name:<14} n={block['count']:6d}  amp {block['median']:+.1f} dB  "
                  f"range {summaries[f'{name}_delta_range_m']['median']:+.1f} m  "
                  f"az {math.degrees(summaries[f'{name}_delta_azimuth_rad']['median']):+.1f} deg")
    life = summaries["ghost_lifetime_frames"]
    if life.get("count"):
        print(f"  ghost lifetime   median {life['median']:.0f} scans over {life['count']} runs")
    rate = summaries["ghost_clusters_per_object"]
    if rate.get("count"):
        print(f"  ghost clusters   mean {rate['mean']:.3f} per real object per scan"
              + (f"  (synthetic at scale 1: {np.mean(synthetic_stats['ghost_clusters_per_object']):.3f})"
                 if synthetic_stats is not None and synthetic_stats["ghost_clusters_per_object"].size else
                 "  (pass --synthetic <prepared smoke> to fit ghost_rate_scale)"))
    print()
    print("  derived overrides:")
    for key, value in sorted(overrides.items()):
        if key == "profile_name":
            continue
        print(f"    {key:<32} {value}")
        print(f"      {notes.get(key, '')}")
    print()
    print(f"  wrote {output / 'calibrated_overrides.json'}")
    print("  use:  --radar-profile rgd_regime_v1 --radar-config", output / "calibrated_overrides.json")
    print("=" * 72)


if __name__ == "__main__":
    main()
