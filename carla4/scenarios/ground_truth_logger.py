#!/usr/bin/env python3
"""
Ground Truth Logger
===================

Records every simulation tick's metrics to a CSV file for later analysis.

Fields (from the thesis plan):
  step, fog_density, scenario_id,
  gt_ego_speed_kmh, gt_npc_speed_kmh,
  gt_distance_to_npc_m, gt_relative_velocity,
  gt_tl_state, throttle, brake, steer,
  collision_occurred, time_to_collision_s
"""

import csv
import os
import math
import sys

_CARLA4_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _CARLA4_DIR not in sys.path:
    sys.path.insert(0, _CARLA4_DIR)

from radar.detection_log import DetectionLog, sidecar_path  # noqa: E402


# Corridor used to decide whether a ground-truth actor is "in path". Matches
# the radar selector's corridor (half width plus growth with range) so the
# phantom-brake metric and the sensor agree on what counts as ahead.
PATH_HALF_WIDTH_M = 1.8
PATH_WIDTH_GROWTH_PER_M = 0.004
PATH_MAX_FORWARD_M = 100.0


def actor_in_ego_path(ego, actor):
    """Whether ``actor`` sits ahead of the ego inside the driving corridor.

    Uses transforms only, so it works on any road segment without map
    queries. Lateral extent of the actor is included, as the selector does.
    """

    if ego is None or actor is None or not actor.is_alive:
        return False
    ego_transform = ego.get_transform()
    forward = ego_transform.get_forward_vector()
    right = ego_transform.get_right_vector()
    delta = actor.get_location() - ego_transform.location
    longitudinal = delta.x * forward.x + delta.y * forward.y
    lateral = delta.x * right.x + delta.y * right.y
    if not 0.0 < longitudinal < PATH_MAX_FORWARD_M:
        return False
    half_width = PATH_HALF_WIDTH_M + PATH_WIDTH_GROWTH_PER_M * longitudinal
    try:
        extent = float(actor.bounding_box.extent.y)
    except (AttributeError, RuntimeError):
        extent = 0.9
    return abs(lateral) - extent <= half_width


def nearest_in_path_actor(world, ego, type_filters=("vehicle.*", "walker.pedestrian.*")):
    """Privileged nearest in-path actor ahead of the ego, or None.

    Returns ``(actor, distance_m)``. Used by the exposure scenario, where any
    vehicle in the traffic may become the legitimate reason to brake, and by
    the privileged collection teacher.
    """

    best = None
    best_distance = float("inf")
    ego_transform = ego.get_transform()
    forward = ego_transform.get_forward_vector()
    right = ego_transform.get_right_vector()
    for pattern in type_filters:
        for actor in world.get_actors().filter(pattern):
            if actor.id == ego.id:
                continue
            try:
                delta = actor.get_location() - ego_transform.location
            except RuntimeError:
                continue
            longitudinal = delta.x * forward.x + delta.y * forward.y
            if not 0.0 < longitudinal < PATH_MAX_FORWARD_M:
                continue
            lateral = delta.x * right.x + delta.y * right.y
            half_width = PATH_HALF_WIDTH_M + PATH_WIDTH_GROWTH_PER_M * longitudinal
            try:
                extent = float(actor.bounding_box.extent.y)
            except (AttributeError, RuntimeError):
                extent = 0.5
            if abs(lateral) - extent > half_width:
                continue
            distance = math.sqrt(delta.x ** 2 + delta.y ** 2)
            if distance < best_distance:
                best = actor
                best_distance = distance
    return best, (best_distance if best is not None else None)


class GroundTruthLogger:
    """Records per-tick telemetry for CDF analysis and collision statistics."""

    FIELDS = [
        "step",
        "fog_density",
        "scenario_id",
        "seed",
        "test_target_speed_kmh",
        "test_event_distance_m",
        "gt_ego_speed_kmh",
        "gt_npc_speed_kmh",
        "gt_distance_to_npc_m",
        "gt_relative_velocity",
        "gt_npc_in_path",
        "gt_tl_state",
        "throttle",
        "brake",
        "steer",
        "critical_event",
        "collision_occurred",
        "time_to_collision_s",
        "ego_accel_mps2",
        "min_distance_so_far_m",
        "radar_backend",
        "radar_profile",
        "radar_config_signature",
        "radar_sensor_frame",
        "radar_sensor_timestamp_s",
        "radar_scan_index",
        "radar_ideal_target_count",
        "radar_multipath_mode",
        "radar_reflector_count",
        "radar_multipath_ideal_target_count",
        "radar_generated_detection_count",
        "radar_rejected_detection_count",
        "radar_delivered_detection_count",
        "radar_direct_detection_count",
        "radar_dropped_direct_count",
        "radar_ghost_detection_count",
        "radar_clutter_detection_count",
        "radar_interference_active",
        "radar_active_ghost_count",
        "radar_active_track_count",
        "radar_confirmed_track_count",
        "radar_selected_track_id",
        "radar_selected_truth_object_id",
        "radar_selected_truth_parent_object_id",
        "radar_selected_reflector_id",
        "radar_selected_bounce_type",
        "radar_selected_bounce_order",
        "radar_selected_path_length_m",
        "radar_selected_ghost_probability",
        "radar_selected_source",
        "radar_selected_confidence",
        "radar_selected_azimuth_deg",
        "radar_point_detection_count",
        "radar_ghost_point_count",
        "radar_distance_m",
        "radar_relative_velocity_mps",
        "radar_obstacle_speed_mps",
        "radar_last_error",
        "controller_target_speed_mps",
    ]

    def __init__(
        self,
        output_dir,
        scenario_id,
        fog_density,
        seed,
        target_speed_kmh=None,
        event_distance_m=None,
    ):
        os.makedirs(output_dir, exist_ok=True)
        self.filepath = os.path.join(
            output_dir,
            f"s{scenario_id}_fog{fog_density}_seed{seed}.csv",
        )
        self.scenario_id = scenario_id
        self.fog_density = fog_density
        self.seed = seed
        self.target_speed_kmh = target_speed_kmh
        self.event_distance_m = event_distance_m
        self._file = open(self.filepath, "w", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=self.FIELDS)
        self._writer.writeheader()
        self._rows = 0
        self._collision_count = 0
        self._min_distance = float("inf")
        # Per-scan detection lists go to a sidecar next to the CSV; see
        # radar/detection_log.py. Written on close() only if anything arrived.
        self._detections = DetectionLog()
        self.detections_path = sidecar_path(self.filepath)

    def log(
        self,
        step,
        ego_speed_kmh,
        npc_speed_kmh=None,
        distance_to_npc=None,
        relative_velocity=None,
        throttle=0.0,
        brake=0.0,
        steer=0.0,
        critical_event=False,
        collision=False,
        tl_state=0,
        ego_accel=0.0,
        radar_diagnostics=None,
        npc_in_path=None,
        detections=None,
    ):
        """Record one tick of ground truth data.

        ``npc_in_path`` is the ground-truth answer to "is the logged actor a
        legitimate reason to brake"; the phantom-brake metric depends on it.
        ``detections`` is the driver's latest point-level scan (a dict from
        ``get_detections``) and lands in the sidecar.
        """
        if distance_to_npc is not None:
            self._min_distance = min(self._min_distance, distance_to_npc)
        if detections is not None:
            self._detections.append(
                step,
                detections.get("timestamp"),
                detections.get("scan_index", 0),
                detections.get("detections", ()),
            )

        # Compute TTC safely
        rel_vel = relative_velocity if relative_velocity is not None else 0.0
        dist = distance_to_npc if distance_to_npc is not None else 999.0
        if rel_vel > 0.1 and dist > 0.1:
            ttc = dist / rel_vel
        else:
            ttc = 999.0

        if collision:
            self._collision_count += 1

        radar = radar_diagnostics or {}
        row = {
            "step": step,
            "fog_density": self.fog_density,
            "scenario_id": self.scenario_id,
            "seed": self.seed,
            "test_target_speed_kmh": (
                round(self.target_speed_kmh, 4)
                if self.target_speed_kmh is not None
                else ""
            ),
            "test_event_distance_m": (
                round(self.event_distance_m, 4)
                if self.event_distance_m is not None
                else ""
            ),
            "gt_ego_speed_kmh": round(ego_speed_kmh, 4),
            "gt_npc_speed_kmh": round(npc_speed_kmh, 4) if npc_speed_kmh is not None else "",
            "gt_distance_to_npc_m": round(dist, 4),
            "gt_relative_velocity": round(rel_vel, 4),
            "gt_npc_in_path": (
                "" if npc_in_path is None else int(bool(npc_in_path))
            ),
            "gt_tl_state": tl_state,
            "throttle": round(throttle, 4),
            "brake": round(brake, 4),
            "steer": round(steer, 4),
            "critical_event": 1 if critical_event else 0,
            "collision_occurred": 1 if collision else 0,
            "time_to_collision_s": round(ttc, 4),
            "ego_accel_mps2": round(ego_accel, 4),
            "min_distance_so_far_m": round(self._min_distance, 4),
            "radar_backend": radar.get("backend", ""),
            "radar_profile": radar.get("profile", ""),
            "radar_config_signature": radar.get("config_signature", ""),
            "radar_sensor_frame": radar.get("frame", ""),
            "radar_sensor_timestamp_s": radar.get("timestamp", ""),
            "radar_scan_index": radar.get("scan_index", ""),
            "radar_ideal_target_count": radar.get("ideal_target_count", ""),
            "radar_multipath_mode": radar.get("multipath_mode", ""),
            "radar_reflector_count": radar.get("reflector_count", ""),
            "radar_multipath_ideal_target_count": radar.get(
                "multipath_ideal_target_count", ""
            ),
            "radar_generated_detection_count": radar.get(
                "generated_detection_count", ""
            ),
            "radar_rejected_detection_count": radar.get(
                "rejected_detection_count", ""
            ),
            "radar_delivered_detection_count": radar.get(
                "delivered_detection_count", ""
            ),
            "radar_direct_detection_count": radar.get(
                "direct_detection_count", ""
            ),
            "radar_dropped_direct_count": radar.get(
                "dropped_direct_count", ""
            ),
            "radar_ghost_detection_count": radar.get(
                "ghost_detection_count", ""
            ),
            "radar_clutter_detection_count": radar.get(
                "clutter_detection_count", ""
            ),
            "radar_interference_active": radar.get(
                "interference_active", ""
            ),
            "radar_active_ghost_count": radar.get("active_ghost_count", ""),
            "radar_active_track_count": radar.get("active_track_count", ""),
            "radar_confirmed_track_count": radar.get(
                "confirmed_track_count", ""
            ),
            "radar_selected_track_id": radar.get("selected_track_id", ""),
            "radar_selected_truth_object_id": radar.get(
                "selected_truth_object_id", ""
            ),
            "radar_selected_truth_parent_object_id": radar.get(
                "selected_truth_parent_object_id", ""
            ),
            "radar_selected_reflector_id": radar.get(
                "selected_reflector_id", ""
            ),
            "radar_selected_bounce_type": radar.get(
                "selected_bounce_type", ""
            ),
            "radar_selected_bounce_order": radar.get(
                "selected_bounce_order", ""
            ),
            "radar_selected_path_length_m": radar.get(
                "selected_path_length_m", ""
            ),
            "radar_selected_ghost_probability": radar.get(
                "selected_ghost_probability", ""
            ),
            "radar_selected_source": radar.get("selected_source", ""),
            "radar_selected_confidence": radar.get(
                "selected_confidence", ""
            ),
            "radar_selected_azimuth_deg": radar.get(
                "selected_azimuth_deg", ""
            ),
            "radar_point_detection_count": radar.get(
                "point_detection_count", ""
            ),
            "radar_ghost_point_count": radar.get("ghost_point_count", ""),
            "radar_distance_m": radar.get("controller_distance_m", ""),
            "radar_relative_velocity_mps": radar.get(
                "controller_relative_velocity_mps", ""
            ),
            "radar_obstacle_speed_mps": radar.get(
                "controller_obstacle_speed_mps", ""
            ),
            "radar_last_error": radar.get("last_error", ""),
            "controller_target_speed_mps": radar.get(
                "controller_target_speed_mps", ""
            ),
        }
        self._writer.writerow(row)
        self._rows += 1

    @property
    def row_count(self):
        return self._rows

    @property
    def has_collision(self):
        return self._collision_count > 0

    @property
    def min_distance(self):
        return self._min_distance

    def close(self):
        if self._file and not self._file.closed:
            self._file.close()
            if self._detections.frame_count:
                try:
                    self._detections.save(self.detections_path)
                except OSError as exc:
                    print(f"  [logger] could not write {self.detections_path}: {exc}")

    def __del__(self):
        self.close()


def compute_vehicle_speed(vehicle):
    """Return speed in km/h from a CARLA vehicle."""
    v = vehicle.get_velocity()
    return math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2) * 3.6


def distance_between(actor_a, actor_b):
    """Euclidean distance between two actors in metres."""
    return actor_a.get_location().distance(actor_b.get_location())
