#!/usr/bin/env python3
"""Explain wrong target selections from a radar validation directory.

This is intentionally dependency-free and does not import CARLA. It consumes
the metadata and detailed JSONL already written by validate_radar_accuracy.py.
"""

import argparse
from collections import Counter
import json
import math
import os


SEMANTIC_NAMES = {
    0: "Unlabeled",
    1: "Roads",
    2: "SideWalks",
    3: "Building",
    4: "Wall",
    5: "Fence",
    6: "Pole",
    7: "TrafficLight",
    8: "TrafficSign",
    9: "Vegetation",
    10: "Terrain",
    11: "Sky",
    12: "Pedestrian",
    13: "Rider",
    14: "Car",
    15: "Truck",
    16: "Bus",
    17: "Train",
    18: "Motorcycle",
    19: "Bicycle",
    20: "Static",
    21: "Dynamic",
    22: "Other",
    23: "Water",
    24: "RoadLine",
    25: "Ground",
    26: "Bridge",
    27: "RailTrack",
    28: "GuardRail",
}


def _load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _target_id(item, field):
    value = item.get(field)
    return None if value is None else int(value)


def _find_by_id(items, field, object_id):
    return [
        item
        for item in items
        if _target_id(item, field) == int(object_id)
    ]


def _track_eligibility(track, config):
    if not bool(track.get("confirmed", False)):
        return False, "lead_track_unconfirmed"
    if float(track.get("confidence", 0.0)) < float(
        config["minimum_track_confidence"]
    ):
        return False, "lead_track_low_confidence"

    distance = float(track["distance_m"])
    minimum_distance = float(config["minimum_forward_distance_m"])
    if distance < minimum_distance:
        return False, "lead_track_below_minimum_range"

    azimuth = float(track["azimuth_rad"])
    lateral = distance * math.sin(azimuth)
    half_width = (
        float(config["path_half_width_m"])
        + float(config["path_width_growth_per_m"]) * distance
    )
    if abs(lateral) > half_width:
        return False, "lead_track_outside_path_corridor"
    if distance * math.cos(azimuth) < minimum_distance:
        return False, "lead_track_not_forward"
    return True, "eligible"


def _wrong_selection_reason(debug, lead_id, config):
    ideal = _find_by_id(
        debug.get("ideal_targets", ()),
        "object_id",
        lead_id,
    )
    if not ideal:
        return "lead_not_extracted", None, None

    generated = _find_by_id(
        debug.get("generated_detections", ()),
        "truth_object_id",
        lead_id,
    )
    if not generated:
        return "lead_direct_detection_missing", ideal[0], None

    delivered = _find_by_id(
        debug.get("delivered_detections", ()),
        "truth_object_id",
        lead_id,
    )
    if not delivered:
        return "lead_not_delivered_after_latency", ideal[0], None

    lead_tracks = _find_by_id(
        debug.get("tracks", ()),
        "truth_object_id",
        lead_id,
    )
    if not lead_tracks:
        return "lead_detection_lost_during_association", ideal[0], None

    eligible = []
    rejection_reasons = []
    for track in lead_tracks:
        accepted, reason = _track_eligibility(track, config)
        if accepted:
            eligible.append(track)
        else:
            rejection_reasons.append(reason)
    if not eligible:
        reason = Counter(rejection_reasons).most_common(1)[0][0]
        return reason, ideal[0], lead_tracks[0]

    lead_track = min(
        eligible,
        key=lambda item: float(item["distance_m"])
        * math.cos(float(item["azimuth_rad"])),
    )
    selected = debug.get("selected", {})
    selected_source = str(selected.get("source", "none"))
    selected_tag = int(selected.get("semantic_tag", 0) or 0)
    selected_distance = float(selected.get("distance_m", math.inf))
    selected_forward = selected_distance * math.cos(
        float(selected.get("azimuth_rad", 0.0))
    )
    lead_forward = float(lead_track["distance_m"]) * math.cos(
        float(lead_track["azimuth_rad"])
    )
    if selected_forward <= lead_forward + 1.0e-6:
        return (
            f"closer_{selected_source}_{SEMANTIC_NAMES.get(selected_tag, selected_tag)}",
            ideal[0],
            lead_track,
        )
    return "selection_order_invariant_violation", ideal[0], lead_track


def _format_target(item, id_field):
    if not item:
        return "none"
    object_id = item.get(id_field)
    tag = int(item.get("semantic_tag", 0) or 0)
    distance = item.get("distance_m")
    azimuth = math.degrees(float(item.get("azimuth_rad", 0.0)))
    return (
        f"id={object_id} tag={tag}:{SEMANTIC_NAMES.get(tag, 'Unknown')} "
        f"range={float(distance):.2f}m az={azimuth:+.2f}deg"
    )


def analyze(validation_dir, backend, max_examples):
    metadata_path = os.path.join(validation_dir, "metadata.json")
    details_path = os.path.join(validation_dir, "radar_details.jsonl")
    metadata = _load_json(metadata_path)
    lead_id = int(metadata["actors"]["lead"]["id"])
    radar_metadata = metadata["radars"].get(backend, {})
    config = radar_metadata.get("radar_config")
    if backend != "realistic" or not config:
        raise RuntimeError(
            "Forensic reason classification currently requires the realistic "
            "backend and its resolved radar_config in metadata.json"
        )

    relevant = 0
    correct = 0
    wrong = 0
    no_output = 0
    reasons = Counter()
    selected_sources = Counter()
    selected_tags = Counter()
    selected_ids = Counter()
    examples = []

    with open(details_path, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            values = record.get("backends", {}).get(backend)
            if not values or not bool(values.get("observable", False)):
                continue
            relevant += 1
            identity = values.get("target_identity", {})
            selected_id = identity.get("target_id")
            if selected_id is not None and int(selected_id) == lead_id:
                correct += 1
                continue

            wrong += 1
            debug = values.get("debug", {})
            selected = debug.get("selected", {})
            if not selected.get("track_id"):
                no_output += 1
            selected_source = str(selected.get("source", "none"))
            selected_tag = int(selected.get("semantic_tag", 0) or 0)
            selected_truth_id = selected.get("truth_object_id")
            selected_sources[selected_source] += 1
            selected_tags[selected_tag] += 1
            selected_ids[str(selected_truth_id)] += 1

            reason, lead_ideal, lead_track = _wrong_selection_reason(
                debug,
                lead_id,
                config,
            )
            reasons[reason] += 1
            if len(examples) < max_examples:
                examples.append(
                    {
                        "frame": record.get("world_frame"),
                        "reason": reason,
                        "selected": _format_target(
                            selected,
                            "truth_object_id",
                        ),
                        "lead_ideal": _format_target(
                            lead_ideal,
                            "object_id",
                        ),
                        "lead_track": _format_target(
                            lead_track,
                            "truth_object_id",
                        ),
                    }
                )

    print("=" * 88)
    print("RADAR TARGET-SELECTION FORENSICS — COPY THIS BLOCK BACK TO CODEX")
    print("=" * 88)
    print(f"Directory: {os.path.abspath(validation_dir)}")
    print(f"Backend/profile: {backend} / {radar_metadata.get('radar_profile')}")
    print(f"Lead actor ID: {lead_id}")
    print(
        f"Path-relevant frames: {relevant} | correct: {correct} | "
        f"wrong: {wrong} | no output: {no_output}"
    )
    print("Wrong-selection reasons:")
    for reason, count in reasons.most_common():
        print(f"  {reason}: {count}")
    print(f"Wrong selected sources: {dict(selected_sources)}")
    print(
        "Wrong selected tags: "
        + str(
            {
                f"{tag}:{SEMANTIC_NAMES.get(tag, 'Unknown')}": count
                for tag, count in selected_tags.most_common()
            }
        )
    )
    print(f"Most common wrong truth IDs: {selected_ids.most_common(10)}")
    if examples:
        print("First wrong-selection examples:")
        for example in examples:
            print(f"  frame={example['frame']} reason={example['reason']}")
            print(f"    selected:   {example['selected']}")
            print(f"    lead ideal: {example['lead_ideal']}")
            print(f"    lead track: {example['lead_track']}")
    print("=" * 88)


def main():
    parser = argparse.ArgumentParser(
        description="Explain target-selection errors from radar_details.jsonl"
    )
    parser.add_argument("validation_dir")
    parser.add_argument("--backend", default="realistic")
    parser.add_argument("--max-examples", type=int, default=8)
    args = parser.parse_args()
    if args.max_examples < 0:
        parser.error("--max-examples cannot be negative")
    analyze(args.validation_dir, args.backend, args.max_examples)


if __name__ == "__main__":
    main()
