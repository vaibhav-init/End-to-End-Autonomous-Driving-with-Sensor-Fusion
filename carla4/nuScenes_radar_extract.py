"""Extract nuScenes radar sweeps and annotated ground truth into canonical scans.

Produces one JSONL row per (keyframe sample, radar channel):

    {
      "scene", "sample_token", "channel", "timestamp",
      "ego_speed_mps": float,
      "detections": [[range_m, azimuth_rad, vr_closing_mps, rcs_db, dynprop, id], ...],
      "targets":    [[range_m, azimuth_rad, vr_closing_mps, category], ...],
    }

Conventions (must match ``carla4/radar/realism_metrics.py``):
  azimuth: boresight zero, positive RIGHT;
  radial velocity: positive CLOSING.

Radar point velocities (vx) are stored in the sensor frame as velocity of the
object relative to the sensor.  ``--doppler-sign`` selects the flip that
yields positive closing values; ``auto`` validates the choice per channel
against annotated parked vehicles ahead of the ego (their measured closing
speed must track ``ego_speed``).
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import os
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
from pyquaternion import Quaternion

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import RadarPointCloud

DEFAULT_RADAR_CHANNELS = (
    "RADAR_FRONT",
    "RADAR_FRONT_LEFT",
    "RADAR_FRONT_RIGHT",
    "RADAR_BACK_LEFT",
    "RADAR_BACK_RIGHT",
)

# Radar pcd row layout after decode (ARS408): x, y, z, dynprop, id, rcs,
# vx, vy, comp_rate, sensitivity, invalid_state, ambiguous_state.
DET_X, DET_Y, DET_RCS, DET_VX, DET_DYNPROP, DET_ID = 0, 1, 5, 6, 3, 4

TARGET_CATEGORY_PREFIXES = ("vehicle.", "human.pedestrian.")


def _row(rows: Sequence[Sequence[float]]) -> List[List[float]]:
    return [[float(v) for v in row] for row in rows]


def _quaternion(record: Dict[str, Any]) -> Quaternion:
    return Quaternion(record["rotation"])


def _sensor_frame_of(sample_data: Dict[str, Any], nusc: NuScenes):
    csd = nusc.get("calibrated_sensor", sample_data["calibrated_sensor_token"])
    return np.asarray(csd["translation"], dtype=float), _quaternion(csd)


def _global_point_to_sensor(point_global: np.ndarray,
                            ego_pose: Dict[str, Any],
                            sensor_translation: np.ndarray,
                            sensor_rotation_global: Quaternion,
                            ego_rotation_global: Quaternion) -> np.ndarray:
    """Sensor-frame position of a global point.

    Chain: global --(ego pose)--> ego --(calibrated sensor)--> sensor.
    """

    point_ego = ego_rotation_global.rotation_matrix.T @ (point_global - np.asarray(ego_pose["translation"], dtype=float))
    return sensor_rotation_global.rotation_matrix.T @ (point_ego - sensor_translation)


def _global_vector_to_sensor(vector_global: np.ndarray,
                             ego_rotation_global: Quaternion,
                             sensor_rotation_global: Quaternion) -> np.ndarray:
    """Sensor-frame vector of a global express, honouring ego/sensor chain."""

    vector_ego = ego_rotation_global.rotation_matrix.T @ vector_global
    return sensor_rotation_global.rotation_matrix.T @ vector_ego


def _ego_velocity_vectors(nusc: NuScenes, sample_token: str) -> Tuple[np.ndarray, float]:
    """Global ego velocity vector from consecutive ego poses at keyframes."""

    def pose_of(sample: Dict[str, Any]):
        sd = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
        pose = nusc.get("ego_pose", sd["ego_pose_token"])
        return float(sample["timestamp"]), np.asarray(pose["translation"][:2], dtype=float)

    sample_prev = None
    current = nusc.get("sample", sample_token)
    poses: List[Tuple[float, np.ndarray]] = [pose_of(current)]
    walk = current
    for _ in range(3):
        if not walk["prev"]:
            break
        walk = nusc.get("sample", walk["prev"])
        poses.append(pose_of(walk))
    walk = current
    for _ in range(3):
        if not walk["next"]:
            break
        walk = nusc.get("sample", walk["next"])
        poses.append(pose_of(walk))

    poses.sort(key=lambda pair: pair[0])
    velocities: List[np.ndarray] = []
    for earlier, later in zip(poses, poses[1:]):
        dt = (later[0] - earlier[0]) / 1e6
        if abs(dt) < 1e-6:
            continue
        velocities.append((later[1] - earlier[1]) / dt)
    if not velocities:
        return np.zeros(2), 0.0
    median_velocity = np.median(np.asarray(velocities), axis=0)
    return median_velocity, float(np.linalg.norm(median_velocity))


def _annotation_rows(nusc: NuScenes, sample_token: str) -> List[Dict[str, Any]]:
    """Category-filtered annotations with global velocity and parked flag."""

    rows: List[Dict[str, Any]] = []
    sample = nusc.get("sample", sample_token)
    for token in sample["annotations"]:
        ann = nusc.get("sample_annotation", token)
        if not ann["category_name"].startswith(TARGET_CATEGORY_PREFIXES):
            continue
        velocity = nusc.box_velocity(token)
        parked = any(nusc.get("attribute", t)["name"] == "vehicle.parked"
                     for t in ann["attribute_tokens"])
        rows.append({
            "token": token,
            "position": np.asarray(ann["translation"], dtype=float),
            "velocity": np.asarray(velocity, dtype=float),
            "category": ann["category_name"],
            "parked": parked,
        })
    return rows


def _waypoint_chain(nusc: NuScenes, scene_token: str) -> Iterator[Dict[str, Any]]:
    """Yield keyframe samples of one scene earliest to latest."""

    scene = nusc.get("scene", scene_token)
    token = scene["first_sample_token"]
    while token:
        sample = nusc.get("sample", token)
        yield sample
        token = sample["next"]


def _channel_row(nusc: NuScenes, scene_name: str, sample: Dict[str, Any],
                 channel: str, doppler_sign: int) -> Optional[Dict[str, Any]]:
    if channel not in sample["data"]:
        return None
    sample_data = nusc.get("sample_data", sample["data"][channel])
    pcd_path = os.path.join(nusc.dataroot, sample_data["filename"])
    if not os.path.exists(pcd_path):
        return None

    lidar_data = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
    ego_pose = nusc.get("ego_pose", lidar_data["ego_pose_token"])
    ego_rotation_global = _quaternion(ego_pose)
    sensor_translation, sensor_rotation_global = _sensor_frame_of(sample_data, nusc)

    ego_velocity_global, ego_speed = _ego_velocity_vectors(nusc, sample["token"])

    pointcloud = RadarPointCloud.from_file(pcd_path)
    rows = pointcloud.points
    detections: List[List[float]] = []
    targets: List[List[float]] = []
    target_labels: List[str] = []
    target_parked: List[float] = []

    for row in rows:
        x_sensor, y_sensor = float(row[DET_X]), float(row[DET_Y])
        range_m = math.hypot(x_sensor, y_sensor)
        if range_m <= 0.5:
            continue
        azimuth = math.atan2(y_sensor, x_sensor)
        vr_sensor = float(row[DET_VX]) * doppler_sign
        detections.append([range_m, azimuth, vr_sensor, float(row[DET_RCS]),
                           float(row[DET_DYNPROP]), float(row[DET_ID])])

    for ann in _annotation_rows(nusc, sample["token"]):
        if not np.all(np.isfinite(ann["velocity"])):
            continue
        point_sensor = _global_point_to_sensor(ann["position"], ego_pose,
                                               sensor_translation, sensor_rotation_global,
                                               ego_rotation_global)
        range_m = float(np.hypot(point_sensor[0], point_sensor[1]))
        if range_m <= 0.5:
            continue
        azimuth = math.atan2(point_sensor[1], point_sensor[0])
        rel_velocity_global = ann["velocity"] - ego_velocity_global
        rel_velocity_sensor = _global_vector_to_sensor(rel_velocity_global,
                                                       ego_rotation_global,
                                                       sensor_rotation_global)
        distance = max(range_m, 1e-6)
        radial_unit = point_sensor[:2] / distance
        vr_closing = -float(np.dot(radial_unit, rel_velocity_sensor[:2]))
        width_m, length_m, height_m = (float(v) for v in ann["size"])
        targets.append([range_m, azimuth, vr_closing, length_m, width_m, height_m])
        target_labels.append(ann["category"])
        target_parked.append(1.0 if ann["parked"] else 0.0)

    return {
        "scene": scene_name,
        "sample_token": sample["token"],
        "channel": channel,
        "timestamp": float(sample["timestamp"]),
        "ego_speed_mps": ego_speed,
        "detections": _row(detections),
        "targets": _row(targets),
        "target_labels": target_labels,
        "target_parked": target_parked,
    }


def resolve_doppler_sign(nusc: NuScenes, channel: str,
                         max_samples: int = 30) -> int:
    """+1 if the raw vx flip already yields closing-positive data, else -1.

    Validation signal: for annotated parked vehicles ahead of the ego
    (|azimuth| < 10 deg), the measured closing speed of detections inside the
    gate should track the ego speed.  Both flips are scored over recent
    keyframes; the one matching parked-vehicle closing motion wins.
    """

    scores: Dict[int, List[float]] = {1: [], -1: []}
    used = 0
    for scene in nusc.scene:
        if used >= max_samples:
            break
        for sample in _waypoint_chain(nusc, scene["token"]):
            if used >= max_samples:
                break
            row = _channel_row(nusc, scene["name"], sample, channel, 1)
            used += 1
            if row is None or row["ego_speed_mps"] < 1.0:
                continue
            static_targets = [tgt for tgt, label, parked
                              in zip(row["targets"], row["target_labels"], row["target_parked"])
                              if parked > 0.0 and abs(tgt[1]) < math.radians(10)]
            if not static_targets:
                continue
            static_gt = float(np.median([tgt[2] for tgt in static_targets]))
            det_ranges = np.asarray([det[0] for det in row["detections"]])
            det_vr = np.asarray([det[2] for det in row["detections"]])
            if det_ranges.size == 0:
                continue
            for sign in (1, -1):
                closing = det_vr if sign == 1 else -det_vr
                scores[sign].append(float(np.median(np.abs(closing - static_gt))))
    if not scores[1] and not scores[-1]:
        return 1
    mean_1 = float(np.mean(scores[1])) if scores[1] else float("inf")
    mean_minus1 = float(np.mean(scores[-1])) if scores[-1] else float("inf")
    return 1 if mean_1 <= mean_minus1 else -1


def extract_dataset(nusc: NuScenes, channels: Sequence[str],
                    doppler_signs: str) -> Iterator[Dict[str, Any]]:
    signs: Dict[str, int] = {}
    for channel in channels:
        if doppler_signs == "auto":
            signs[channel] = resolve_doppler_sign(nusc, channel)
        elif doppler_signs == "plus_closing":
            signs[channel] = 1
        else:
            signs[channel] = -1
    print("resolved doppler signs:", signs)
    for scene in nusc.scene:
        for sample in _waypoint_chain(nusc, scene["token"]):
            for channel in channels:
                row = _channel_row(nusc, scene["name"], sample, channel, signs[channel])
                if row is not None:
                    yield row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataroot", help="path to the extracted nuScenes root (contains v1.0-mini/)")
    parser.add_argument("--output", default="nuScenes_radar_scans.jsonl")
    parser.add_argument("--version", default="v1.0-mini")
    parser.add_argument("--channels", nargs="*", default=list(DEFAULT_RADAR_CHANNELS))
    parser.add_argument("--doppler-sign", choices=("auto", "plus_closing", "plus_away"),
                        default="auto")
    parser.add_argument("--max-scenes", type=int, default=0, help="limit scenes for smoke tests")
    args = parser.parse_args()

    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=True)
    scene_filter = nusc.scene if not args.max_scenes else nusc.scene[:args.max_scenes]
    if args.max_scenes:
        nusc.scene = scene_filter  # type: ignore[misc]

    gz = args.output.endswith(".gz")
    opener = gzip.open if gz else open
    count = 0
    with opener(args.output, "wt", encoding="utf-8") as handle:
        for row in extract_dataset(nusc, args.channels, args.doppler_sign):
            handle.write(json.dumps(row) + "\n")
            count += 1
    print(f"wrote {count} scans to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
