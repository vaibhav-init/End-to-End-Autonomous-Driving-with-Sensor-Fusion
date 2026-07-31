#!/usr/bin/env python3
"""Compare all forward-radar backends with CARLA 0.9.16 ground truth.

This is a controlled, simulator-backed validation harness. It spawns an ego
vehicle and one lead vehicle, runs CARLA synchronously, waits for radar sensor
frames, and records:

* current and latency-aligned ground truth;
* scalar output and errors for native, C-Shenron-derived, and realistic radar;
* target identity where the backend exposes it;
* every native radar return and every extracted/delivered/tracked target;
* semantic-LiDAR tag histograms using the CARLA 0.9.16 tag table;
* callback errors, frame lag, misses, false selections, ghosts, and clutter.

The script imports only CARLA, NumPy (through the radar package), and the
Python standard library. It does not require pandas, PyTorch, or YOLO.
"""

import argparse
from collections import Counter
import csv
from datetime import datetime
import json
import math
import os
import platform
import random
import re
import sys
import time

from radar import (
    CARLA_0916_SEMANTIC_TAGS,
    DEFAULT_REALISTIC_RADAR_PROFILE,
    RADAR_BACKENDS,
    REALISTIC_RADAR_PROFILES,
    SEMANTIC_LIDAR_DTYPE,
    create_front_radar,
    describe_radar_configuration,
    semantic_material_name,
)
from radar.validation import BackendAccuracy


EXPECTED_CARLA_VERSION = "0.9.16"
IDENTITY_BACKENDS = frozenset(("cshenron", "realistic"))

BASE_FIELDS = [
    "sample_index",
    "world_frame",
    "simulation_time_s",
    "wall_elapsed_s",
    "collision_count",
    "ego_actor_id",
    "lead_actor_id",
    "ego_speed_mps",
    "ego_speed_kmh",
    "lead_speed_mps",
    "lead_speed_kmh",
    "ego_accel_mps2",
    "lead_accel_mps2",
    "ego_x",
    "ego_y",
    "ego_z",
    "ego_yaw_deg",
    "lead_x",
    "lead_y",
    "lead_z",
    "lead_yaw_deg",
    "gt_center_range_m",
    "gt_surface_range_m",
    "gt_longitudinal_m",
    "gt_lateral_m",
    "gt_azimuth_deg",
    "gt_elevation_deg",
    "gt_bbox_azimuth_min_deg",
    "gt_bbox_azimuth_max_deg",
    "gt_bbox_elevation_min_deg",
    "gt_bbox_elevation_max_deg",
    "gt_closing_velocity_mps",
    "gt_radial_obstacle_speed_mps",
]

BACKEND_FIELDS = [
    "sensor_frame",
    "sensor_timestamp_s",
    "sensor_frame_lag",
    "sensor_synchronized",
    "lead_observable",
    "reported_detection",
    "target_truth_id",
    "correct_target",
    "target_identity_method",
    "target_association_distance_m",
    "selected_source",
    "distance_m",
    "relative_velocity_mps",
    "obstacle_speed_mps",
    "distance_error_current_m",
    "velocity_error_current_mps",
    "obstacle_speed_error_current_mps",
    "distance_error_latency_aligned_m",
    "velocity_error_latency_aligned_mps",
    "latency_alignment_frames",
    "raw_detection_count",
    "candidate_count",
    "raw_return_count",
    "target_count",
    "target_semantic_tag",
    "target_snr_db",
    "ideal_target_count",
    "generated_detection_count",
    "delivered_detection_count",
    "delivered_source_scan_index",
    "direct_detection_count",
    "dropped_direct_count",
    "ghost_detection_count",
    "active_ghost_count",
    "clutter_detection_count",
    "interference_active",
    "active_track_count",
    "confirmed_track_count",
    "selected_track_id",
    "selected_confidence",
    "selected_azimuth_deg",
    "last_error",
]


def _vector_components(vector):
    return (float(vector.x), float(vector.y), float(vector.z))


def _vector_magnitude(vector):
    x, y, z = _vector_components(vector)
    return math.sqrt(x * x + y * y + z * z)


def _copy_location(carla, location):
    return carla.Location(
        x=float(location.x),
        y=float(location.y),
        z=float(location.z),
    )


def _actor_center(carla, actor):
    transform = actor.get_transform()
    box_location = _copy_location(carla, actor.bounding_box.location)
    return transform.transform(box_location)


def _distance_to_actor_box(carla, world_point, actor):
    """Euclidean distance from a world point to an actor's oriented box."""

    actor_transform = actor.get_transform()
    point_actor = actor_transform.inverse_transform(
        _copy_location(carla, world_point)
    )
    bounding_box = actor.bounding_box
    box_transform = carla.Transform(
        _copy_location(carla, bounding_box.location),
        carla.Rotation(
            pitch=float(bounding_box.rotation.pitch),
            yaw=float(bounding_box.rotation.yaw),
            roll=float(bounding_box.rotation.roll),
        ),
    )
    point_box = box_transform.inverse_transform(point_actor)
    extent = bounding_box.extent
    dx = max(abs(float(point_box.x)) - float(extent.x), 0.0)
    dy = max(abs(float(point_box.y)) - float(extent.y), 0.0)
    dz = max(abs(float(point_box.z)) - float(extent.z), 0.0)
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def _ground_truth(carla, ego, lead, sensor_transform):
    sensor_location = sensor_transform.location
    lead_center = _actor_center(carla, lead)
    center_local = sensor_transform.inverse_transform(
        _copy_location(carla, lead_center)
    )
    dx = float(lead_center.x - sensor_location.x)
    dy = float(lead_center.y - sensor_location.y)
    dz = float(lead_center.z - sensor_location.z)
    center_range = max(math.sqrt(dx * dx + dy * dy + dz * dz), 1.0e-9)
    horizontal = math.hypot(float(center_local.x), float(center_local.y))

    ego_velocity = ego.get_velocity()
    lead_velocity = lead.get_velocity()
    relative_x = float(ego_velocity.x - lead_velocity.x)
    relative_y = float(ego_velocity.y - lead_velocity.y)
    relative_z = float(ego_velocity.z - lead_velocity.z)
    closing_speed = (
        relative_x * dx + relative_y * dy + relative_z * dz
    ) / center_range
    ego_speed = _vector_magnitude(ego_velocity)
    box_vertices = lead.bounding_box.get_world_vertices(
        lead.get_transform()
    )
    vertex_azimuths = []
    vertex_elevations = []
    for vertex in box_vertices:
        local = sensor_transform.inverse_transform(
            _copy_location(carla, vertex)
        )
        if float(local.x) <= 0.0:
            continue
        vertex_horizontal = math.hypot(float(local.x), float(local.y))
        vertex_azimuths.append(
            math.degrees(math.atan2(float(local.y), float(local.x)))
        )
        vertex_elevations.append(
            math.degrees(
                math.atan2(float(local.z), max(vertex_horizontal, 1.0e-9))
            )
        )
    if not vertex_azimuths:
        vertex_azimuths = [
            math.degrees(
                math.atan2(float(center_local.y), float(center_local.x))
            )
        ]
        vertex_elevations = [
            math.degrees(
                math.atan2(float(center_local.z), max(horizontal, 1.0e-9))
            )
        ]

    return {
        "center_range_m": center_range,
        "surface_range_m": _distance_to_actor_box(
            carla,
            sensor_location,
            lead,
        ),
        "longitudinal_m": float(center_local.x),
        "lateral_m": float(center_local.y),
        "azimuth_deg": math.degrees(
            math.atan2(float(center_local.y), float(center_local.x))
        ),
        "elevation_deg": math.degrees(
            math.atan2(float(center_local.z), max(horizontal, 1.0e-9))
        ),
        "bbox_azimuth_min_deg": min(vertex_azimuths),
        "bbox_azimuth_max_deg": max(vertex_azimuths),
        "bbox_elevation_min_deg": min(vertex_elevations),
        "bbox_elevation_max_deg": max(vertex_elevations),
        "closing_velocity_mps": closing_speed,
        "radial_obstacle_speed_mps": max(0.0, ego_speed - closing_speed),
        "ego_speed_mps": ego_speed,
        "lead_speed_mps": _vector_magnitude(lead_velocity),
    }


def _backend_envelope(radar):
    if radar.backend == "native":
        return {
            "horizontal_fov_deg": 10.0,
            "min_elevation_deg": -1.0,
            "max_elevation_deg": 1.0,
            "max_range_m": radar._range,
        }
    config = (
        radar.realistic_config
        if radar.backend == "realistic"
        else radar.config
    )
    return {
        "horizontal_fov_deg": float(config.horizontal_fov_deg),
        "min_elevation_deg": float(config.min_elevation_deg),
        "max_elevation_deg": float(config.max_elevation_deg),
        "max_range_m": float(config.max_range_m),
    }


def _is_observable(ground_truth, envelope):
    horizontal_min = -envelope["horizontal_fov_deg"] / 2.0
    horizontal_max = envelope["horizontal_fov_deg"] / 2.0
    return bool(
        ground_truth["longitudinal_m"] > 0.0
        and ground_truth["surface_range_m"] < envelope["max_range_m"]
        and ground_truth["bbox_azimuth_max_deg"] >= horizontal_min
        and ground_truth["bbox_azimuth_min_deg"] <= horizontal_max
        and ground_truth["bbox_elevation_max_deg"]
        >= envelope["min_elevation_deg"]
        and ground_truth["bbox_elevation_min_deg"]
        <= envelope["max_elevation_deg"]
    )


def _native_hit_association(
    carla,
    radar,
    debug_snapshot,
    lead,
    tolerance_m,
):
    selected = debug_snapshot.get("selected_detection")
    if selected is None:
        return None, None
    depth = float(selected["distance_m"])
    azimuth = float(selected["azimuth_rad"])
    altitude = float(selected["altitude_rad"])
    cos_altitude = math.cos(altitude)
    local_hit = carla.Location(
        x=depth * cos_altitude * math.cos(azimuth),
        y=depth * cos_altitude * math.sin(azimuth),
        z=depth * math.sin(altitude),
    )
    world_hit = radar.sensor.get_transform().transform(local_hit)
    distance = _distance_to_actor_box(carla, world_hit, lead)
    target_id = int(lead.id) if distance <= tolerance_m else -1
    return target_id, distance


def _sensor_state(radar, state, diagnostics, debug_snapshot):
    backend = radar.backend
    if backend == "native":
        reported = debug_snapshot.get("selected_detection") is not None
        target_id = None
        source = "native" if reported else "none"
    elif backend == "cshenron":
        target_id = diagnostics.get("target_object_id")
        reported = target_id is not None
        source = "direct" if reported else "none"
    else:
        target_id = diagnostics.get("selected_truth_object_id")
        reported = diagnostics.get("selected_track_id") is not None
        source = diagnostics.get("selected_source", "none")
    return {
        "reported": bool(reported),
        "target_id": target_id,
        "source": source,
        "distance_m": float(state["distance"]),
        "relative_velocity_mps": float(state["relative_velocity"]),
        "obstacle_speed_mps": float(state["obstacle_speed"]),
    }


def _wait_for_sensor_frames(radars, world_frame, timeout_s):
    deadline = time.monotonic() + timeout_s
    diagnostics = {}
    while True:
        diagnostics = {
            backend: radar.diagnostics()
            for backend, radar in radars.items()
        }
        missing = [
            backend
            for backend, values in diagnostics.items()
            if int(values.get("frame", -1)) < int(world_frame)
        ]
        if not missing or time.monotonic() >= deadline:
            return diagnostics, missing
        time.sleep(0.001)


def _configure_blueprint(blueprint, role_name, rng):
    if blueprint.has_attribute("role_name"):
        blueprint.set_attribute("role_name", role_name)
    if blueprint.has_attribute("color"):
        colors = blueprint.get_attribute("color").recommended_values
        if colors:
            blueprint.set_attribute("color", rng.choice(colors))
    if blueprint.has_attribute("is_invincible"):
        blueprint.set_attribute("is_invincible", "true")
    return blueprint


def _spawn_vehicle_pair(carla, world, gap_m, vehicle_filter, seed):
    carla_map = world.get_map()
    spawn_points = list(carla_map.get_spawn_points())
    if not spawn_points:
        raise RuntimeError("The current map has no vehicle spawn points")
    blueprints = list(
        world.get_blueprint_library().filter(vehicle_filter)
    )
    blueprints = [
        blueprint
        for blueprint in blueprints
        if not blueprint.has_attribute("number_of_wheels")
        or int(blueprint.get_attribute("number_of_wheels")) == 4
    ]
    if not blueprints:
        raise RuntimeError(
            f"No four-wheel vehicle blueprint matches '{vehicle_filter}'"
        )

    rng = random.Random(seed)
    rng.shuffle(spawn_points)
    for ego_transform in spawn_points:
        waypoint = carla_map.get_waypoint(
            ego_transform.location,
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )
        if waypoint is None or waypoint.is_junction:
            continue
        ahead = waypoint.next(float(gap_m))
        if not ahead:
            continue
        lead_waypoint = min(
            ahead,
            key=lambda item: abs(item.lane_id - waypoint.lane_id),
        )
        lead_transform = lead_waypoint.transform
        ego_transform.location.z += 0.25
        lead_transform.location.z += 0.25

        ego_template = rng.choice(blueprints)
        lead_template = rng.choice(blueprints)
        ego_blueprint = world.get_blueprint_library().find(
            ego_template.id
        )
        lead_blueprint = world.get_blueprint_library().find(
            lead_template.id
        )
        _configure_blueprint(
            ego_blueprint,
            "radar_validation_ego",
            rng,
        )
        _configure_blueprint(
            lead_blueprint,
            "radar_validation_lead",
            rng,
        )

        ego = world.try_spawn_actor(ego_blueprint, ego_transform)
        if ego is None:
            continue
        lead = world.try_spawn_actor(lead_blueprint, lead_transform)
        if lead is not None:
            return ego, lead
        ego.destroy()
    raise RuntimeError(
        "Could not spawn a same-lane ego/lead pair. Stop other traffic or "
        "try another town."
    )


def _set_tm_speed(traffic_manager, actor, speed_kmh):
    if hasattr(traffic_manager, "set_desired_speed"):
        traffic_manager.set_desired_speed(actor, float(speed_kmh))
        return
    speed_limit = max(float(actor.get_speed_limit()), 1.0)
    difference = 100.0 * (1.0 - float(speed_kmh) / speed_limit)
    traffic_manager.vehicle_percentage_speed_difference(actor, difference)


def _configure_traffic_manager(
    traffic_manager,
    ego,
    lead,
    port,
    ego_speed_kmh,
    lead_speed_kmh,
):
    ego.set_autopilot(True, port)
    lead.set_autopilot(True, port)
    for actor in (ego, lead):
        traffic_manager.auto_lane_change(actor, False)
    traffic_manager.distance_to_leading_vehicle(ego, 5.0)
    _set_tm_speed(traffic_manager, ego, ego_speed_kmh)
    _set_tm_speed(traffic_manager, lead, lead_speed_kmh)


def _spawn_collision_sensor(carla, world, ego, collision_state):
    blueprint = world.get_blueprint_library().find(
        "sensor.other.collision"
    )
    sensor = world.spawn_actor(
        blueprint,
        carla.Transform(),
        attach_to=ego,
    )

    def callback(event):
        collision_state["count"] += 1
        collision_state["last_frame"] = int(event.frame)
        collision_state["other_actor_id"] = int(event.other_actor.id)

    sensor.listen(callback)
    return sensor


def _actor_metadata(actor):
    box = actor.bounding_box
    return {
        "id": int(actor.id),
        "type_id": actor.type_id,
        "bounding_box": {
            "location": {
                "x": float(box.location.x),
                "y": float(box.location.y),
                "z": float(box.location.z),
            },
            "extent": {
                "x": float(box.extent.x),
                "y": float(box.extent.y),
                "z": float(box.extent.z),
            },
            "rotation": {
                "pitch": float(box.rotation.pitch),
                "yaw": float(box.rotation.yaw),
                "roll": float(box.rotation.roll),
            },
        },
    }


def _weather_metadata(weather):
    names = (
        "cloudiness",
        "precipitation",
        "precipitation_deposits",
        "wind_intensity",
        "sun_azimuth_angle",
        "sun_altitude_angle",
        "fog_density",
        "fog_distance",
        "fog_falloff",
        "wetness",
        "scattering_intensity",
        "mie_scattering_scale",
        "rayleigh_scattering_scale",
        "dust_storm",
    )
    return {
        name: float(getattr(weather, name))
        for name in names
        if hasattr(weather, name)
    }


def _write_json(path, value):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)


def _prepare_output(path, overwrite):
    if path is None:
        path = datetime.now().strftime("radar_validation_%Y%m%d_%H%M%S")
    path = os.path.abspath(path)
    if os.path.isdir(path) and os.listdir(path) and not overwrite:
        raise RuntimeError(
            f"Output directory is not empty: {path}. Use a new path or "
            "--overwrite."
        )
    os.makedirs(path, exist_ok=True)
    return path


def _version_is_0916(version):
    match = re.search(r"\d+\.\d+\.\d+", str(version))
    return bool(match and match.group(0) == EXPECTED_CARLA_VERSION)


def _assess_carla_versions(client_version, server_version):
    """Handle both packaged semver strings and source-build Git hashes."""

    client_text = str(client_version)
    server_text = str(server_version)
    if _version_is_0916(client_text) and _version_is_0916(server_text):
        return {
            "accepted": True,
            "mode": "verified_semantic_version",
            "message": "client and server report CARLA 0.9.16",
        }
    client_semver = re.search(r"\d+\.\d+\.\d+", client_text)
    server_semver = re.search(r"\d+\.\d+\.\d+", server_text)
    if (
        client_text == server_text
        and client_text
        and client_semver is None
        and server_semver is None
    ):
        return {
            "accepted": True,
            "mode": "matching_source_build_id",
            "message": (
                "client and server expose the same source-build identifier "
                f"'{client_text}'; semantic version will be capability-audited"
            ),
        }
    return {
        "accepted": False,
        "mode": "unverified_or_mismatched",
        "message": (
            f"client/server version identifiers are "
            f"{client_text!r}/{server_text!r}"
        ),
    }


def _probe_required_sensor_capabilities(world):
    requirements = {
        "sensor.other.radar": (
            "horizontal_fov",
            "vertical_fov",
            "range",
            "points_per_second",
            "sensor_tick",
        ),
        "sensor.lidar.ray_cast_semantic": (
            "channels",
            "range",
            "points_per_second",
            "rotation_frequency",
            "upper_fov",
            "lower_fov",
            "horizontal_fov",
            "sensor_tick",
        ),
    }
    result = {"blueprints": {}, "missing": []}
    library = world.get_blueprint_library()
    for blueprint_id, attributes in requirements.items():
        matches = list(library.filter(blueprint_id))
        if not matches:
            result["missing"].append(f"blueprint:{blueprint_id}")
            continue
        blueprint = matches[0]
        available = {
            name: bool(blueprint.has_attribute(name))
            for name in attributes
        }
        result["blueprints"][blueprint_id] = available
        result["missing"].extend(
            f"{blueprint_id}.{name}"
            for name, present in available.items()
            if not present
        )
    return result


def _format_ratio(value):
    return "n/a" if value is None else f"{100.0 * value:.2f}%"


def _format_metric(value, suffix=""):
    return "n/a" if value is None else f"{value:.4f}{suffix}"


def _print_summary(summary):
    print("\n" + "=" * 88)
    print("RADAR VALIDATION SUMMARY — COPY THIS BLOCK BACK TO CODEX")
    print("=" * 88)
    print(
        f"CARLA client/server: {summary['carla_client_version']} / "
        f"{summary['carla_server_version']}"
    )
    print(
        f"Map: {summary['map']} | samples: {summary['samples']} | "
        f"collisions: {summary['collision_count']}"
    )
    for backend, values in summary["backends"].items():
        range_stats = values[
            "selected_output_range_error_latency_aligned_m"
        ]
        velocity_stats = values[
            "selected_output_velocity_error_latency_aligned_mps"
        ]
        line = (
            f"{backend:10s} detection="
            f"{_format_ratio(values['detection_rate_when_observable'])} "
            f"miss={_format_ratio(values['miss_rate_when_observable'])} "
        )
        if values["identity_available"]:
            line += (
                f"correct_target="
                f"{_format_ratio(values['correct_target_rate'])} "
            )
        else:
            line += "correct_target=n/a(native has no IDs) "
        line += (
            f"range_MAE={_format_metric(range_stats['mae'], 'm')} "
            f"range_RMSE={_format_metric(range_stats['rmse'], 'm')} "
            f"velocity_MAE={_format_metric(velocity_stats['mae'], 'm/s')} "
            f"unsynced={values['unsynchronized_frames']} "
            f"callback_errors={values['callback_error_frames']}"
        )
        print(line)
    for backend, tag_values in summary[
        "semantic_tag_return_totals"
    ].items():
        observed = ", ".join(
            f"{tag}:{entry['name']}={entry['count']}"
            for tag, entry in tag_values.items()
        )
        print(
            f"{backend:10s} semantic_tags=[{observed}] "
            f"lead_tag14_frames="
            f"{summary['lead_tag14_extracted_frames'].get(backend, 0)}"
        )
    for warning in summary["warnings"]:
        print(f"WARNING: {warning}")
    print(f"CSV:      {summary['files']['samples_csv']}")
    print(f"Details:  {summary['files']['details_jsonl']}")
    print(f"Metadata: {summary['files']['metadata_json']}")
    print(f"Summary:  {summary['files']['summary_json']}")
    print("=" * 88)


def _cleanup_sensor(sensor):
    if sensor is None:
        return
    try:
        if sensor.is_alive:
            sensor.stop()
            sensor.destroy()
    except RuntimeError:
        pass


def _cleanup_actor(actor):
    if actor is None:
        return
    try:
        if actor.is_alive:
            actor.destroy()
    except RuntimeError:
        pass


def _backend_csv_values(
    backend,
    sensor,
    diagnostics,
    ground_truth,
    aligned_ground_truth,
    observable,
    synchronized,
    frame_lag,
    correct_target,
    latency_frames,
):
    reported = sensor["reported"]
    distance_error = (
        sensor["distance_m"] - ground_truth["surface_range_m"]
        if observable and reported
        else None
    )
    velocity_error = (
        sensor["relative_velocity_mps"]
        - ground_truth["closing_velocity_mps"]
        if observable and reported
        else None
    )
    obstacle_error = (
        sensor["obstacle_speed_mps"]
        - ground_truth["radial_obstacle_speed_mps"]
        if observable and reported
        else None
    )
    aligned_range_error = (
        sensor["distance_m"] - aligned_ground_truth["surface_range_m"]
        if observable and reported
        else None
    )
    aligned_velocity_error = (
        sensor["relative_velocity_mps"]
        - aligned_ground_truth["closing_velocity_mps"]
        if observable and reported
        else None
    )
    values = {
        "sensor_frame": diagnostics.get("frame"),
        "sensor_timestamp_s": diagnostics.get("timestamp"),
        "sensor_frame_lag": frame_lag,
        "sensor_synchronized": int(synchronized),
        "lead_observable": int(observable),
        "reported_detection": int(reported),
        "target_truth_id": sensor["target_id"],
        "correct_target": (
            "" if correct_target is None else int(correct_target)
        ),
        "target_identity_method": sensor.get("identity_method", ""),
        "target_association_distance_m": sensor.get(
            "association_distance_m"
        ),
        "selected_source": sensor["source"],
        "distance_m": sensor["distance_m"],
        "relative_velocity_mps": sensor["relative_velocity_mps"],
        "obstacle_speed_mps": sensor["obstacle_speed_mps"],
        "distance_error_current_m": distance_error,
        "velocity_error_current_mps": velocity_error,
        "obstacle_speed_error_current_mps": obstacle_error,
        "distance_error_latency_aligned_m": aligned_range_error,
        "velocity_error_latency_aligned_mps": aligned_velocity_error,
        "latency_alignment_frames": latency_frames,
    }
    for name in BACKEND_FIELDS:
        values.setdefault(name, diagnostics.get(name))
    if backend == "realistic":
        values["target_semantic_tag"] = diagnostics.get(
            "selected_semantic_tag"
        )
    return {
        f"{backend}_{name}": (
            "" if values.get(name) is None else values.get(name)
        )
        for name in BACKEND_FIELDS
    }


def _base_csv_values(
    sample_index,
    world_frame,
    simulation_time,
    wall_elapsed,
    collision_count,
    ego,
    lead,
    ground_truth,
):
    ego_transform = ego.get_transform()
    lead_transform = lead.get_transform()
    ego_acceleration = ego.get_acceleration()
    lead_acceleration = lead.get_acceleration()
    return {
        "sample_index": sample_index,
        "world_frame": world_frame,
        "simulation_time_s": simulation_time,
        "wall_elapsed_s": wall_elapsed,
        "collision_count": collision_count,
        "ego_actor_id": ego.id,
        "lead_actor_id": lead.id,
        "ego_speed_mps": ground_truth["ego_speed_mps"],
        "ego_speed_kmh": ground_truth["ego_speed_mps"] * 3.6,
        "lead_speed_mps": ground_truth["lead_speed_mps"],
        "lead_speed_kmh": ground_truth["lead_speed_mps"] * 3.6,
        "ego_accel_mps2": _vector_magnitude(ego_acceleration),
        "lead_accel_mps2": _vector_magnitude(lead_acceleration),
        "ego_x": ego_transform.location.x,
        "ego_y": ego_transform.location.y,
        "ego_z": ego_transform.location.z,
        "ego_yaw_deg": ego_transform.rotation.yaw,
        "lead_x": lead_transform.location.x,
        "lead_y": lead_transform.location.y,
        "lead_z": lead_transform.location.z,
        "lead_yaw_deg": lead_transform.rotation.yaw,
        "gt_center_range_m": ground_truth["center_range_m"],
        "gt_surface_range_m": ground_truth["surface_range_m"],
        "gt_longitudinal_m": ground_truth["longitudinal_m"],
        "gt_lateral_m": ground_truth["lateral_m"],
        "gt_azimuth_deg": ground_truth["azimuth_deg"],
        "gt_elevation_deg": ground_truth["elevation_deg"],
        "gt_bbox_azimuth_min_deg": ground_truth[
            "bbox_azimuth_min_deg"
        ],
        "gt_bbox_azimuth_max_deg": ground_truth[
            "bbox_azimuth_max_deg"
        ],
        "gt_bbox_elevation_min_deg": ground_truth[
            "bbox_elevation_min_deg"
        ],
        "gt_bbox_elevation_max_deg": ground_truth[
            "bbox_elevation_max_deg"
        ],
        "gt_closing_velocity_mps": ground_truth[
            "closing_velocity_mps"
        ],
        "gt_radial_obstacle_speed_mps": ground_truth[
            "radial_obstacle_speed_mps"
        ],
    }


def build_argument_parser():
    parser = argparse.ArgumentParser(
        description=(
            "CARLA 0.9.16 synchronized native/C-Shenron/realistic radar "
            "accuracy experiment"
        )
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--tm-port", type=int, default=8000)
    parser.add_argument(
        "--town",
        default=None,
        help="optional town to load; default keeps the current CARLA world",
    )
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--duration-s", type=float, default=30.0)
    parser.add_argument("--warmup-s", type=float, default=3.0)
    parser.add_argument("--spawn-gap-m", type=float, default=45.0)
    parser.add_argument("--ego-speed-kmh", type=float, default=45.0)
    parser.add_argument("--lead-speed-kmh", type=float, default=25.0)
    parser.add_argument("--radar-range-m", type=float, default=100.0)
    parser.add_argument(
        "--backends",
        nargs="+",
        choices=RADAR_BACKENDS,
        default=list(RADAR_BACKENDS),
    )
    parser.add_argument(
        "--radar-profile",
        choices=REALISTIC_RADAR_PROFILES,
        default=DEFAULT_REALISTIC_RADAR_PROFILE,
    )
    parser.add_argument("--radar-config", default=None)
    parser.add_argument("--radar-seed", type=int, default=None)
    parser.add_argument(
        "--native-points-per-second",
        type=int,
        default=3000,
    )
    parser.add_argument(
        "--native-association-tolerance-m",
        type=float,
        default=0.75,
        help=(
            "maximum native hit-point distance from the lead bounding box "
            "for validator-only target association"
        ),
    )
    parser.add_argument(
        "--semantic-points-per-second",
        type=int,
        default=240000,
    )
    parser.add_argument("--vehicle-filter", default="vehicle.*")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--sensor-wait-timeout-s",
        type=float,
        default=2.0,
    )
    parser.add_argument(
        "--detailed-log-every",
        type=int,
        default=1,
        help="write detailed JSON every N samples; 0 disables it",
    )
    parser.add_argument("--print-every-s", type=float, default=1.0)
    parser.add_argument("--fog-density", type=float, default=None)
    parser.add_argument("--precipitation", type=float, default=None)
    parser.add_argument("--wetness", type=float, default=None)
    parser.add_argument("--dust-storm", type=float, default=None)
    parser.add_argument("--no-rendering", action="store_true")
    parser.add_argument("--output", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--allow-version-mismatch",
        action="store_true",
        help="run even when client/server are not CARLA 0.9.16",
    )
    return parser


def _validate_arguments(parser, args):
    positive = (
        "fps",
        "duration_s",
        "spawn_gap_m",
        "radar_range_m",
        "sensor_wait_timeout_s",
        "print_every_s",
    )
    for name in positive:
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.warmup_s < 0:
        parser.error("--warmup-s cannot be negative")
    if args.native_points_per_second <= 0:
        parser.error("--native-points-per-second must be positive")
    if args.native_association_tolerance_m < 0:
        parser.error("--native-association-tolerance-m cannot be negative")
    if args.semantic_points_per_second <= 0:
        parser.error("--semantic-points-per-second must be positive")
    if args.detailed_log_every < 0:
        parser.error("--detailed-log-every cannot be negative")
    for name in (
        "fog_density",
        "precipitation",
        "wetness",
        "dust_storm",
    ):
        value = getattr(args, name)
        if value is not None and not 0.0 <= value <= 100.0:
            parser.error(f"--{name.replace('_', '-')} must be in [0, 100]")


def main():
    parser = build_argument_parser()
    args = parser.parse_args()
    _validate_arguments(parser, args)
    output_dir = _prepare_output(args.output, args.overwrite)
    csv_path = os.path.join(output_dir, "radar_samples.csv")
    details_path = os.path.join(output_dir, "radar_details.jsonl")
    metadata_path = os.path.join(output_dir, "metadata.json")
    summary_path = os.path.join(output_dir, "summary.json")

    try:
        import carla
    except ImportError as exc:
        raise RuntimeError(
            "CARLA Python API is not importable. Activate the CARLA 0.9.16 "
            "environment or add its wheel/egg to PYTHONPATH."
        ) from exc

    client = carla.Client(args.host, args.port)
    client.set_timeout(30.0)
    client_version = client.get_client_version()
    server_version = client.get_server_version()
    version_assessment = _assess_carla_versions(
        client_version,
        server_version,
    )
    if not args.allow_version_mismatch and not version_assessment["accepted"]:
        raise RuntimeError(
            "This validator requires matching CARLA 0.9.16-compatible client "
            f"and server APIs: {version_assessment['message']}. Use "
            "--allow-version-mismatch only for diagnostics."
        )

    world = client.get_world()
    if args.town:
        current_town = world.get_map().name.split("/")[-1]
        if current_town != args.town:
            print(f"Loading {args.town} ...")
            world = client.load_world(args.town)
    capability_audit = _probe_required_sensor_capabilities(world)
    if capability_audit["missing"]:
        raise RuntimeError(
            "CARLA build is missing required radar/semantic-LiDAR "
            "capabilities: " + ", ".join(capability_audit["missing"])
        )

    original_settings = world.get_settings()
    original_weather = world.get_weather()
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 1.0 / args.fps
    if args.no_rendering:
        settings.no_rendering_mode = True

    traffic_manager = client.get_trafficmanager(args.tm_port)
    ego = None
    lead = None
    collision_sensor = None
    radars = {}
    collision_state = {
        "count": 0,
        "last_frame": None,
        "other_actor_id": None,
    }
    semantic_totals = {
        backend: Counter()
        for backend in args.backends
        if backend in IDENTITY_BACKENDS
    }
    semantic_lead_frames = Counter()
    metrics = {
        backend: BackendAccuracy(
            backend,
            # Native has no target IDs, but this controlled validator labels
            # its selected hit geometrically against the lead bounding box.
            identity_available=True,
        )
        for backend in args.backends
    }
    ground_truth_history = {
        backend: {}
        for backend in args.backends
    }
    samples_written = 0
    run_start = time.monotonic()

    try:
        world.apply_settings(settings)
        traffic_manager.set_synchronous_mode(True)
        traffic_manager.set_random_device_seed(args.seed)

        requested_weather = world.get_weather()
        changed_weather = False
        for argument_name, carla_name in (
            ("fog_density", "fog_density"),
            ("precipitation", "precipitation"),
            ("wetness", "wetness"),
            ("dust_storm", "dust_storm"),
        ):
            value = getattr(args, argument_name)
            if value is not None and hasattr(requested_weather, carla_name):
                setattr(requested_weather, carla_name, float(value))
                changed_weather = True
        if changed_weather:
            world.set_weather(requested_weather)

        ego, lead = _spawn_vehicle_pair(
            carla,
            world,
            args.spawn_gap_m,
            args.vehicle_filter,
            args.seed,
        )
        _configure_traffic_manager(
            traffic_manager,
            ego,
            lead,
            args.tm_port,
            args.ego_speed_kmh,
            args.lead_speed_kmh,
        )
        collision_sensor = _spawn_collision_sensor(
            carla,
            world,
            ego,
            collision_state,
        )

        radar_seed = (
            args.radar_seed if args.radar_seed is not None else args.seed
        )
        for backend in args.backends:
            points_per_second = (
                args.native_points_per_second
                if backend == "native"
                else args.semantic_points_per_second
            )
            radars[backend] = create_front_radar(
                ego,
                world,
                range_m=args.radar_range_m,
                backend=backend,
                fps=args.fps,
                points_per_second=points_per_second,
                profile_name=(
                    args.radar_profile
                    if backend == "realistic"
                    else None
                ),
                config_path=(
                    args.radar_config
                    if backend == "realistic"
                    else None
                ),
                seed=radar_seed,
                capture_debug=True,
            )

        metadata = {
            "schema_version": 1,
            "created_at": datetime.now().astimezone().isoformat(),
            "expected_carla_version": EXPECTED_CARLA_VERSION,
            "carla_client_version": client_version,
            "carla_server_version": server_version,
            "carla_version_assessment": version_assessment,
            "sensor_capability_audit": capability_audit,
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "map": world.get_map().name,
            "arguments": vars(args),
            "actors": {
                "ego": _actor_metadata(ego),
                "lead": _actor_metadata(lead),
            },
            "weather": _weather_metadata(world.get_weather()),
            "semantic_lidar_0916_contract": {
                "record_dtype": str(SEMANTIC_LIDAR_DTYPE),
                "record_size_bytes": SEMANTIC_LIDAR_DTYPE.itemsize,
                "fields": [
                    "x:f4",
                    "y:f4",
                    "z:f4",
                    "cos_incidence:f4",
                    "object_id:u4",
                    "semantic_tag:u4",
                ],
                "tags": {
                    str(tag): name
                    for tag, name in CARLA_0916_SEMANTIC_TAGS.items()
                },
                "material_mapping": {
                    str(tag): semantic_material_name(tag)
                    for tag in CARLA_0916_SEMANTIC_TAGS
                },
                "compatibility_note": (
                    "The adapter uses CARLA 0.9.16's six-field semantic-LiDAR "
                    "record and 0..28 semantic table. It does not reuse the "
                    "offset material-label table from older C-Shenron data "
                    "conversion code; only the material scattering equations "
                    "are ported."
                ),
            },
            "radars": {},
            "files": {
                "samples_csv": csv_path,
                "details_jsonl": details_path,
                "metadata_json": metadata_path,
                "summary_json": summary_path,
            },
        }
        for backend in args.backends:
            points_per_second = (
                args.native_points_per_second
                if backend == "native"
                else args.semantic_points_per_second
            )
            metadata["radars"][backend] = describe_radar_configuration(
                backend=backend,
                range_m=args.radar_range_m,
                fps=args.fps,
                points_per_second=points_per_second,
                profile_name=(
                    args.radar_profile
                    if backend == "realistic"
                    else None
                ),
                config_path=(
                    args.radar_config
                    if backend == "realistic"
                    else None
                ),
            )
            metadata["radars"][backend]["envelope"] = _backend_envelope(
                radars[backend]
            )
        _write_json(metadata_path, metadata)

        print("=" * 88)
        print("CARLA 0.9.16 RADAR VS GROUND-TRUTH VALIDATION")
        print("=" * 88)
        print(f"Client/server:  {client_version} / {server_version}")
        print(f"Version mode:   {version_assessment['mode']}")
        print(f"Version audit:  {version_assessment['message']}")
        print(f"Map:            {world.get_map().name}")
        print(f"Ego/lead IDs:   {ego.id} / {lead.id}")
        print(f"Backends:       {', '.join(args.backends)}")
        print(f"Radar range:    {args.radar_range_m:.1f} m")
        print(
            f"Semantic LiDAR: {args.semantic_points_per_second:,} points/s "
            "(CARLA 0.9.16 ffffII layout)"
        )
        print(f"Output:         {output_dir}")
        print(
            "C-Shenron note: material/scattering equations are retained, but "
            "CARLA integration and tags are explicitly 0.9.16."
        )
        print("=" * 88)

        fieldnames = list(BASE_FIELDS)
        for backend in args.backends:
            fieldnames.extend(
                f"{backend}_{name}" for name in BACKEND_FIELDS
            )

        warmup_frames = int(round(args.warmup_s * args.fps))
        sample_frames = int(round(args.duration_s * args.fps))
        print(f"Warming sensors and trackers for {warmup_frames} frames ...")
        for _ in range(warmup_frames):
            current_speed = _vector_magnitude(ego.get_velocity())
            for radar in radars.values():
                radar.update_ego_speed(current_speed)
            world_frame = world.tick()
            _wait_for_sensor_frames(
                radars,
                world_frame,
                args.sensor_wait_timeout_s,
            )
            for backend, radar in radars.items():
                sensor_transform = radar.sensor.get_transform()
                ground_truth_history[backend][world_frame] = _ground_truth(
                    carla,
                    ego,
                    lead,
                    sensor_transform,
                )

        print(f"Recording {sample_frames} synchronized frames ...")
        with open(
            csv_path,
            "w",
            newline="",
            encoding="utf-8",
        ) as csv_handle, open(
            details_path,
            "w",
            encoding="utf-8",
        ) as details_handle:
            writer = csv.DictWriter(csv_handle, fieldnames=fieldnames)
            writer.writeheader()

            for sample_index in range(sample_frames):
                if not ego.is_alive or not lead.is_alive:
                    print("Actor destroyed; stopping collection early")
                    break

                pre_tick_speed = _vector_magnitude(ego.get_velocity())
                for radar in radars.values():
                    radar.update_ego_speed(pre_tick_speed)

                world_frame = world.tick()
                snapshot = world.get_snapshot()
                diagnostics_by_backend, missing = _wait_for_sensor_frames(
                    radars,
                    world_frame,
                    args.sensor_wait_timeout_s,
                )
                ego_speed = _vector_magnitude(ego.get_velocity())
                for radar in radars.values():
                    radar.update_ego_speed(ego_speed)

                ground_truth_by_backend = {}
                for backend, radar in radars.items():
                    ground_truth = _ground_truth(
                        carla,
                        ego,
                        lead,
                        radar.sensor.get_transform(),
                    )
                    ground_truth_by_backend[backend] = ground_truth
                    ground_truth_history[backend][
                        world_frame
                    ] = ground_truth

                common_ground_truth = ground_truth_by_backend[
                    args.backends[0]
                ]
                csv_row = _base_csv_values(
                    sample_index,
                    world_frame,
                    snapshot.timestamp.elapsed_seconds,
                    time.monotonic() - run_start,
                    collision_state["count"],
                    ego,
                    lead,
                    common_ground_truth,
                )
                detail_record = {
                    "sample_index": sample_index,
                    "world_frame": int(world_frame),
                    "simulation_time_s": float(
                        snapshot.timestamp.elapsed_seconds
                    ),
                    "collision_state": collision_state.copy(),
                    "ground_truth": ground_truth_by_backend,
                    "backends": {},
                }

                display_values = []
                for backend, radar in radars.items():
                    diagnostics = diagnostics_by_backend[backend]
                    state = radar.get()
                    debug_snapshot = radar.debug_snapshot()
                    sensor = _sensor_state(
                        radar,
                        state,
                        diagnostics,
                        debug_snapshot,
                    )
                    if backend == "native" and sensor["reported"]:
                        (
                            sensor["target_id"],
                            sensor["association_distance_m"],
                        ) = _native_hit_association(
                            carla,
                            radar,
                            debug_snapshot,
                            lead,
                            args.native_association_tolerance_m,
                        )
                        sensor["identity_method"] = (
                            "validator_hit_point_to_lead_obb"
                        )
                    elif backend in IDENTITY_BACKENDS:
                        sensor["identity_method"] = (
                            "semantic_truth_id_diagnostic_only"
                        )
                        sensor["association_distance_m"] = None
                    else:
                        sensor["identity_method"] = ""
                        sensor["association_distance_m"] = None
                    sensor_frame = int(diagnostics.get("frame", -1))
                    frame_lag = int(world_frame) - sensor_frame
                    synchronized = backend not in missing and frame_lag == 0
                    latency_frames = int(
                        diagnostics.get("configured_latency_scans", 0) or 0
                    )
                    aligned_frame = sensor_frame - latency_frames
                    aligned_ground_truth = ground_truth_history[
                        backend
                    ].get(
                        aligned_frame,
                        ground_truth_by_backend[backend],
                    )
                    envelope = _backend_envelope(radar)
                    observable = _is_observable(
                        aligned_ground_truth,
                        envelope,
                    )

                    range_error_current = (
                        sensor["distance_m"]
                        - ground_truth_by_backend[backend]["surface_range_m"]
                        if observable and sensor["reported"]
                        else None
                    )
                    velocity_error_current = (
                        sensor["relative_velocity_mps"]
                        - ground_truth_by_backend[backend][
                            "closing_velocity_mps"
                        ]
                        if observable and sensor["reported"]
                        else None
                    )
                    range_error_aligned = (
                        sensor["distance_m"]
                        - aligned_ground_truth["surface_range_m"]
                        if observable and sensor["reported"]
                        else None
                    )
                    velocity_error_aligned = (
                        sensor["relative_velocity_mps"]
                        - aligned_ground_truth["closing_velocity_mps"]
                        if observable and sensor["reported"]
                        else None
                    )
                    correct_target = metrics[backend].update(
                        observable=observable,
                        reported=sensor["reported"],
                        target_id=sensor["target_id"],
                        lead_id=lead.id,
                        synchronized=synchronized,
                        callback_error=bool(
                            diagnostics.get("last_error")
                        ),
                        frame_lag=frame_lag,
                        range_error_current=range_error_current,
                        velocity_error_current=velocity_error_current,
                        range_error_aligned=range_error_aligned,
                        velocity_error_aligned=velocity_error_aligned,
                    )

                    csv_row.update(
                        _backend_csv_values(
                            backend,
                            sensor,
                            diagnostics,
                            ground_truth_by_backend[backend],
                            aligned_ground_truth,
                            observable,
                            synchronized,
                            frame_lag,
                            correct_target,
                            latency_frames,
                        )
                    )
                    detail_record["backends"][backend] = {
                        "state": state,
                        "diagnostics": diagnostics,
                        "debug": debug_snapshot,
                        "observable": observable,
                        "synchronized": synchronized,
                        "aligned_ground_truth_frame": aligned_frame,
                        "errors": {
                            "range_current_m": range_error_current,
                            "velocity_current_mps": velocity_error_current,
                            "range_latency_aligned_m": range_error_aligned,
                            "velocity_latency_aligned_mps": (
                                velocity_error_aligned
                            ),
                        },
                        "target_identity": {
                            "method": sensor["identity_method"],
                            "target_id": sensor["target_id"],
                            "association_distance_m": sensor[
                                "association_distance_m"
                            ],
                        },
                    }

                    tag_counts = debug_snapshot.get(
                        "semantic_tag_counts",
                        {},
                    )
                    for tag, entry in tag_counts.items():
                        semantic_totals.get(backend, Counter())[
                            int(tag)
                        ] += int(entry["count"])
                    ideal_targets = debug_snapshot.get(
                        "ideal_targets",
                        (),
                    )
                    if any(
                        int(target.get("object_id", -1)) == int(lead.id)
                        and int(target.get("semantic_tag", -1)) == 14
                        for target in ideal_targets
                    ):
                        semantic_lead_frames[backend] += 1

                    if sensor["reported"]:
                        display_values.append(
                            f"{backend}={sensor['distance_m']:.1f}m/"
                            f"{sensor['relative_velocity_mps']:+.1f}m/s"
                        )
                    else:
                        display_values.append(f"{backend}=MISS")

                writer.writerow(csv_row)
                samples_written += 1
                if (
                    args.detailed_log_every > 0
                    and sample_index % args.detailed_log_every == 0
                ):
                    details_handle.write(
                        json.dumps(detail_record, sort_keys=True) + "\n"
                    )

                print_interval = max(
                    1,
                    int(round(args.print_every_s * args.fps)),
                )
                if (
                    sample_index % print_interval == 0
                    or sample_index + 1 == sample_frames
                ):
                    print(
                        f"[{sample_index + 1:04d}/{sample_frames:04d}] "
                        f"GT={common_ground_truth['surface_range_m']:.1f}m/"
                        f"{common_ground_truth['closing_velocity_mps']:+.1f}m/s "
                        + " | ".join(display_values)
                    )

                oldest_frame = int(world_frame) - 200
                for history in ground_truth_history.values():
                    for old_frame in tuple(history):
                        if old_frame < oldest_frame:
                            del history[old_frame]

        warnings = []
        if version_assessment["mode"] == "matching_source_build_id":
            warnings.append(
                "CARLA reports a matching source-build identifier rather than "
                "a semantic version; required 0.9.16 sensor capabilities "
                "passed, and the runtime tag histogram must confirm semantics"
            )
        elif not version_assessment["accepted"]:
            warnings.append(
                "CARLA version mismatch was explicitly allowed: "
                f"{version_assessment['message']}"
            )
        for backend, counts in semantic_totals.items():
            unknown = sorted(
                tag for tag in counts if tag not in CARLA_0916_SEMANTIC_TAGS
            )
            if unknown:
                warnings.append(
                    f"{backend} observed unknown semantic tag IDs {unknown}"
                )
            if sum(counts.values()) > 0 and set(counts) == {0}:
                warnings.append(
                    f"{backend} semantic raw_data contained only tag 0; "
                    "check the CARLA 0.9.16 client/server wheel and raw layout"
                )
            if semantic_lead_frames[backend] == 0:
                warnings.append(
                    f"{backend} never extracted lead actor {lead.id} as "
                    "CARLA 0.9.16 semantic tag 14 (Car)"
                )

        summary = {
            "schema_version": 1,
            "carla_client_version": client_version,
            "carla_server_version": server_version,
            "carla_version_assessment": version_assessment,
            "sensor_capability_audit": capability_audit,
            "map": world.get_map().name,
            "samples": samples_written,
            "collision_count": collision_state["count"],
            "collision_state": collision_state,
            "semantic_lidar_record_size_bytes": (
                SEMANTIC_LIDAR_DTYPE.itemsize
            ),
            "backends": {
                backend: accumulator.summary()
                for backend, accumulator in metrics.items()
            },
            "semantic_tag_return_totals": {
                backend: {
                    str(tag): {
                        "name": CARLA_0916_SEMANTIC_TAGS.get(
                            tag,
                            f"Unknown({tag})",
                        ),
                        "count": count,
                    }
                    for tag, count in sorted(counts.items())
                }
                for backend, counts in semantic_totals.items()
            },
            "lead_tag14_extracted_frames": dict(semantic_lead_frames),
            "warnings": warnings,
            "files": metadata["files"],
        }
        _write_json(summary_path, summary)
        metadata["result"] = {
            "samples": samples_written,
            "collision_state": collision_state,
            "semantic_tag_return_totals": summary[
                "semantic_tag_return_totals"
            ],
            "lead_tag14_extracted_frames": dict(semantic_lead_frames),
            "warnings": warnings,
        }
        _write_json(metadata_path, metadata)
        _print_summary(summary)
    except KeyboardInterrupt:
        print("\nInterrupted; CSV/JSONL data collected so far was retained.")
    finally:
        for radar in radars.values():
            radar.cleanup()
        _cleanup_sensor(collision_sensor)
        if ego is not None:
            try:
                ego.set_autopilot(False, args.tm_port)
            except RuntimeError:
                pass
        if lead is not None:
            try:
                lead.set_autopilot(False, args.tm_port)
            except RuntimeError:
                pass
        _cleanup_actor(lead)
        _cleanup_actor(ego)
        try:
            traffic_manager.set_synchronous_mode(False)
        except RuntimeError:
            pass
        try:
            world.set_weather(original_weather)
            world.apply_settings(original_settings)
        except RuntimeError:
            pass


if __name__ == "__main__":
    main()
