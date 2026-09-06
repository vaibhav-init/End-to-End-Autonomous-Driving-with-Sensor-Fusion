#!/usr/bin/env python3
"""Collect path-labeled geometry multipath target lists in CARLA 0.9.16."""

import argparse
import json
import math
from pathlib import Path
import random
import subprocess
import sys
import time
import uuid

import h5py
import numpy as np

from radar import create_front_radar
from radar.ghost_detection.export_expansion import expand_detection_points
from radar.ghost_detection.features import snr_db_to_amplitude
from radar.multipath import ReflectorSegment, generate_multipath_targets
from radar.realistic_core import IdealRadarTarget


RADAR_DTYPE = np.dtype(
    [
        ("frame", np.int64),
        ("frame_timestamp", np.float64),
        ("timestamp", np.float64),
        ("sensor", "S8"),
        ("x_cc", np.float32),
        ("y_cc", np.float32),
        ("r_sc", np.float32),
        ("phi_sc", np.float32),
        ("vr_sc", np.float32),
        ("amp", np.float32),
        ("uuid", "S36"),
        ("label_id", np.int32),
        ("instance_id", np.int64),
        ("source", "S16"),
        ("parent_object_id", np.int64),
        ("reflector_id", np.int64),
        ("bounce_type", "S16"),
        ("bounce_order", np.int8),
        ("path_length_m", np.float32),
    ]
)


# Radar Ghost Dataset v1.1 records a stationary ego with a pedestrian/cyclist
# main object moving near a reflective surface. These constants let the same
# collector reproduce that regime (or keep the original vehicle smoke test).
TARGET_TYPES = ("vehicle", "pedestrian", "cyclist")
TARGET_SEMANTIC_TAGS = {"vehicle": 14, "pedestrian": 12, "cyclist": 18}
TARGET_SPEEDS_MPS = {"vehicle": 3.0, "pedestrian": 1.4, "cyclist": 4.5}
# Placements the solver accepts can still be invisible to the semantic LiDAR
# (a walker parked inside terrain behind a guardrail). Try this many
# reflectors before giving the sequence up.
MAX_PLACEMENT_ATTEMPTS = 4


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--traffic-manager-port", type=int, default=8000)
    parser.add_argument("--town", default="Town04")
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="train")
    parser.add_argument("--sequences", type=int, default=20)
    parser.add_argument("--duration", type=float, default=38.5)
    parser.add_argument("--warmup", type=float, default=3.0)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--vehicles", type=int, default=45)
    parser.add_argument("--walkers", type=int, default=25)
    parser.add_argument(
        "--lead-distance",
        type=float,
        default=25.0,
        help="initial spawn distance for the controlled dynamic radar target",
    )
    parser.add_argument(
        "--target-type",
        choices=TARGET_TYPES,
        default="vehicle",
        help=(
            "main multipath object class: vehicle (CARLA car), pedestrian "
            "(CARLA walker), or cyclist (CARLA two-wheel motorcycle). The "
            "Radar Ghost Dataset recordings use pedestrian/cyclist main "
            "objects walking away from and back toward the stationary ego."
        ),
    )
    parser.add_argument("--range", dest="range_m", type=float, default=100.0)
    parser.add_argument("--points-per-second", type=int, default=240000)
    parser.add_argument(
        "--radar-timeout",
        type=float,
        default=30.0,
        help="seconds to wait for each processed semantic-LiDAR radar frame",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="do not update the CARLA spectator camera",
    )
    parser.add_argument(
        "--camera-view",
        choices=("target", "chase"),
        default="target",
        help="spectator view: controlled-target close-up or scenarios-style chase",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--radar-config", help="optional geometry-profile overrides")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="skip sequences that already have an H5 and summary sidecar",
    )
    parser.add_argument(
        "--expand-points",
        dest="expand_points",
        action="store_true",
        default=True,
        help=(
            "expand each grouped detection into CFAR-like surface points "
            "with per-point amplitude/Doppler statistics (default on; this "
            "is what makes the export distribution match real RGD frames)"
        ),
    )
    parser.add_argument(
        "--no-expand-points",
        dest="expand_points",
        action="store_false",
        help="write one row per grouped detection (legacy behaviour)",
    )
    parser.add_argument(
        "--points-per-detection",
        type=float,
        default=12.0,
        help="mean expanded sub-points per detection when --expand-points",
    )
    parser.add_argument(
        "--label-scope",
        choices=("main", "all"),
        default="main",
        help=(
            "main (default): only the controlled target and its ghosts carry "
            "CMTO labels, every other actor is background, as the Radar Ghost "
            "Dataset annotates one main object; all: label every road user"
        ),
    )
    parser.add_argument(
        "--target-range-min",
        type=float,
        default=4.0,
        help="closest accepted controlled-target range in metres (RGD median is ~7 m)",
    )
    parser.add_argument(
        "--target-range-max",
        type=float,
        default=16.0,
        help="farthest accepted controlled-target range in metres",
    )
    parser.add_argument(
        "--surface-distance-max",
        type=float,
        default=4.5,
        help=(
            "largest accepted target-to-reflector distance in metres. Real "
            "second-order ghosts sit ~1.7 m behind their parent, which needs "
            "the walker within a few metres of the surface"
        ),
    )
    parser.add_argument(
        "--surface-distance-target",
        type=float,
        default=2.0,
        help=(
            "preferred target-to-reflector distance in metres; among equally "
            "robust placements the closest gap wins. The second-order ghost "
            "trails its parent by about this gap (real: ~1.7 m)"
        ),
    )
    parser.add_argument(
        "--target-speed-mps",
        type=float,
        default=None,
        help=(
            "walking/riding speed of the controlled target; default is the "
            f"per-type table {TARGET_SPEEDS_MPS}"
        ),
    )
    parser.add_argument(
        "--motion-direction",
        choices=("tangent", "radial"),
        default="tangent",
        help=(
            "walk the controlled target along the reflector (tangent) or "
            "along the radar line of sight (radial). Real ghost/parent "
            "Doppler ratios near 0.9 need the object moving parallel to the "
            "surface; radial motion into an oblique wall gives ~0.5"
        ),
    )
    parser.add_argument(
        "--type2-azimuth-max-deg",
        type=float,
        default=40.0,
        help=(
            "largest accepted |ghost - parent| azimuth of the mirrored "
            "(type-2) path. Real type-2 ghosts appear ~16 deg off the parent; "
            "a guardrail beside the ego puts them at 60+ deg"
        ),
    )
    parser.add_argument(
        "--points-source",
        choices=("sensor", "expand"),
        default="sensor",
        help=(
            "sensor (default): write the point-level list the radar itself "
            "emits, so a calibrated profile's points-per-object, footprint and "
            "micro-Doppler settings reach the export; expand: the legacy "
            "export-time expansion of the grouped detections"
        ),
    )
    parser.add_argument(
        "--sequence-retries",
        type=int,
        default=1,
        help="retries after a sequence worker exits or crashes",
    )
    parser.add_argument(
        "--worker-index",
        type=int,
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


def _carla_module():
    import carla

    return carla


def _class_id(semantic_tag):
    return {
        12: 1,
        13: 2,
        19: 2,
        14: 3,
        21: 3,
        15: 4,
        16: 4,
        17: 4,
        18: 5,
    }.get(int(semantic_tag), 3)


def _expand_detection_points(detection, rng, mean_points):
    """Delegate to the shared CFAR-emulating expansion (see export_expansion)."""

    return expand_detection_points(
        detection,
        rng,
        mean_points,
        snr_to_amplitude=snr_db_to_amplitude,
    )


def _cmto_label(detection, main_object_ids=None):
    """CMTO code for one detection.

    With ``main_object_ids`` set, only the main object's direct returns and
    the ghosts it casts are labelled; other road users become background
    (label 0), which is how the Radar Ghost Dataset annotates a scene.
    """

    source = detection.get("source", "")
    if source == "clutter":
        return -2
    semantic_tag = int(detection.get("semantic_tag", 0))
    if semantic_tag not in (12, 13, 14, 15, 16, 17, 18, 19, 21):
        return 0
    if main_object_ids:
        owner = (
            int(detection.get("truth_parent_object_id") or 0)
            if source == "ghost"
            else int(detection.get("truth_object_id") or 0)
        )
        if owner not in main_object_ids:
            return 0
    class_id = _class_id(semantic_tag)
    if source == "direct":
        return class_id * 1000 + 11
    bounce_type = {"type1": 1, "type2": 2}.get(
        detection.get("bounce_type"),
        0,
    )
    physical_order = int(detection.get("bounce_order", 0))
    encoded_order = {1: 1, 2: 2, 3: 4}.get(physical_order, 0)
    return class_id * 1000 + bounce_type * 10 + encoded_order


def _detection_row(sequence_id, frame, timestamp, index, detection,
                   main_object_ids=None, amplitude_gain=1.0):
    distance = float(detection["distance_m"])
    # Convert CARLA x-forward/y-right to the real dataset's y-left convention.
    azimuth = -float(detection["azimuth_rad"])
    x_sensor = distance * math.cos(azimuth)
    y_sensor = distance * math.sin(azimuth)
    parent_id = int(
        detection.get("truth_parent_object_id")
        or detection.get("truth_object_id", 0)
    )
    identifier = str(
        uuid.uuid5(
            uuid.NAMESPACE_OID,
            f"carla-radar-{sequence_id}-{frame}-{index}",
        )
    ).encode("ascii")
    return (
        int(frame),
        float(timestamp),
        float(timestamp),
        b"front",
        x_sensor,
        y_sensor,
        distance,
        azimuth,
        -float(detection["relative_velocity_mps"]),
        float(snr_db_to_amplitude(detection["snr_db"])) * float(amplitude_gain),
        identifier,
        _cmto_label(detection, main_object_ids),
        parent_id,
        str(detection.get("source", "")).encode("ascii", errors="replace"),
        parent_id,
        int(detection.get("reflector_id", 0)),
        str(detection.get("bounce_type", "direct")).encode(
            "ascii",
            errors="replace",
        ),
        int(detection.get("bounce_order", 1)),
        float(detection.get("path_length_m", 2.0 * distance)),
    )


def _set_weather(world, sequence_index):
    carla = _carla_module()
    presets = (
        carla.WeatherParameters.ClearNoon,
        carla.WeatherParameters.CloudyNoon,
        carla.WeatherParameters.WetNoon,
        carla.WeatherParameters.MidRainyNoon,
    )
    world.set_weather(presets[sequence_index % len(presets)])
    return sequence_index % len(presets)


def _update_spectator_camera(spectator, ego, target=None, view="target"):
    """Show the controlled target, with the scenarios/ chase view available."""

    if spectator is None or ego is None or not ego.is_alive:
        return
    carla = _carla_module()
    ego_transform = ego.get_transform()
    if view == "target" and target is not None and target.is_alive:
        target_location = target.get_transform().location
        delta_x = float(target_location.x - ego_transform.location.x)
        delta_y = float(target_location.y - ego_transform.location.y)
        horizontal_distance = max(math.hypot(delta_x, delta_y), 1.0e-6)
        direction_x = delta_x / horizontal_distance
        direction_y = delta_y / horizontal_distance
        stand_off = min(12.0, max(5.0, 0.3 * horizontal_distance))
        camera_location = carla.Location(
            x=float(target_location.x) - stand_off * direction_x,
            y=float(target_location.y) - stand_off * direction_y,
            z=float(target_location.z) + 6.0,
        )
        yaw = math.degrees(math.atan2(direction_y, direction_x))
        pitch = math.degrees(math.atan2(-6.0, stand_off))
        spectator.set_transform(
            carla.Transform(
                camera_location,
                carla.Rotation(pitch=pitch, yaw=yaw),
            )
        )
        return

    # This is the exact third-person placement used by scenarios/.
    spectator.set_transform(
        carla.Transform(
            ego_transform.location
            - ego_transform.get_forward_vector() * 15.0
            + carla.Location(z=8.0),
            carla.Rotation(
                pitch=-20.0,
                yaw=float(ego_transform.rotation.yaw),
            ),
        )
    )


def _spawn_vehicles(
    client,
    world,
    traffic_manager,
    count,
    seed,
    lead_distance_m,
    target_type="vehicle",
):
    carla = _carla_module()
    rng = random.Random(seed)
    blueprints = [
        blueprint
        for blueprint in world.get_blueprint_library().filter("vehicle.*")
        if not blueprint.id.endswith(("isetta", "carlacola", "cybertruck", "t2"))
    ]
    road_vehicle_blueprints = []
    for blueprint in blueprints:
        if not blueprint.has_attribute("number_of_wheels"):
            continue
        try:
            if blueprint.get_attribute("number_of_wheels").as_int() == 4:
                road_vehicle_blueprints.append(blueprint)
        except (RuntimeError, ValueError):
            continue
    if not road_vehicle_blueprints:
        road_vehicle_blueprints = blueprints
    if target_type == "cyclist":
        # There is no cyclist actor in CARLA; a two-wheel motorcycle is the
        # closest proxy and is tagged Motorcycle (18) by the semantic LiDAR.
        controlled_blueprints = [
            blueprint
            for blueprint in blueprints
            if blueprint.has_attribute("number_of_wheels")
            and blueprint.get_attribute("number_of_wheels").as_int() == 2
        ]
        if not controlled_blueprints:
            controlled_blueprints = road_vehicle_blueprints
    else:
        controlled_blueprints = road_vehicle_blueprints
    spawn_points = list(world.get_map().get_spawn_points())
    rng.shuffle(spawn_points)
    if not blueprints or not spawn_points:
        raise RuntimeError("Town has no usable vehicle blueprints/spawn points")

    ego_blueprint = rng.choice(road_vehicle_blueprints)
    if ego_blueprint.has_attribute("role_name"):
        ego_blueprint.set_attribute("role_name", "hero")
    ego = None
    ego_spawn_index = None
    for index, transform in enumerate(spawn_points):
        ego = world.try_spawn_actor(ego_blueprint, transform)
        if ego is not None:
            ego_spawn_index = index
            break
    if ego is None:
        raise RuntimeError("Unable to spawn ego vehicle")
    # This is a stationary radar experiment, not a driving episode. Disabling
    # ego physics keeps every semantic sweep in the same sensor coordinates.
    ego.set_simulate_physics(False)

    actor_ids = []
    controlled = None
    if target_type == "pedestrian":
        # The controlled main object is a walker, matching the Radar Ghost
        # Dataset's pedestrian recordings. Physics is disabled so kinematic
        # control keeps it exactly on the validated multipath path while
        # CARLA still reports the velocity set below.
        walker_blueprints = list(
            world.get_blueprint_library().filter("walker.pedestrian.*")
        )
        if not walker_blueprints:
            raise RuntimeError("Town has no walker blueprints")
        rng.shuffle(walker_blueprints)
        # try_spawn_actor at a single navigation point is unreliable, so try
        # many candidate spots. The actor is immediately repositioned onto the
        # validated multipath path, so the spawn location only needs to be
        # collision-free.
        locations = []
        for _ in range(24):
            location = world.get_random_location_from_navigation()
            if location is not None:
                locations.append(location)
        locations.append(
            ego.get_transform().location
            + carla.Location(x=2.0, y=2.0, z=1.0)
        )
        controlled = None
        for blueprint in walker_blueprints:
            if blueprint.has_attribute("is_invincible"):
                blueprint.set_attribute("is_invincible", "false")
            for location in locations:
                controlled = world.try_spawn_actor(
                    blueprint,
                    carla.Transform(location),
                )
                if controlled is not None:
                    break
            if controlled is not None:
                break
        if controlled is None:
            raise RuntimeError("Unable to spawn the pedestrian radar target")
        # Physics is disabled so kinematic teleports keep the walker exactly
        # on the validated multipath path. CARLA reports zero velocity for
        # such actors; the radar adapter falls back to the transform
        # derivative, so pedestrian Doppler is still correct.
        controlled.set_simulate_physics(False)
        actor_ids.append(controlled.id)
    else:
        ego_waypoint = world.get_map().get_waypoint(
            ego.get_transform().location,
            project_to_road=True,
        )
        if ego_waypoint is not None:
            candidate_distances = (
                float(lead_distance_m),
                float(lead_distance_m) + 5.0,
                max(12.0, float(lead_distance_m) - 5.0),
            )
            for distance in candidate_distances:
                lead_waypoints = list(ego_waypoint.next(distance))
                rng.shuffle(lead_waypoints)
                for waypoint in lead_waypoints:
                    transform = waypoint.transform
                    transform.location.z += 0.35
                    blueprint = rng.choice(controlled_blueprints)
                    if blueprint.has_attribute("role_name"):
                        blueprint.set_attribute("role_name", "radar_target")
                    controlled = world.try_spawn_actor(blueprint, transform)
                    if controlled is not None:
                        break
                if controlled is not None:
                    break
        if controlled is None:
            raise RuntimeError(
                "Unable to spawn the guaranteed lead radar target ahead of ego"
            )
        controlled.apply_control(carla.VehicleControl(hand_brake=True))
        actor_ids.append(controlled.id)

    batch = []
    available_spawn_points = [
        transform
        for index, transform in enumerate(spawn_points)
        if index != ego_spawn_index
        and transform.location.distance(controlled.get_transform().location) > 8.0
    ]
    for transform in available_spawn_points[: max(0, count - 1)]:
        blueprint = rng.choice(blueprints)
        if blueprint.has_attribute("role_name"):
            blueprint.set_attribute("role_name", "autopilot")
        batch.append(
            carla.command.SpawnActor(blueprint, transform).then(
                carla.command.SetAutopilot(
                    carla.command.FutureActor,
                    True,
                    traffic_manager.get_port(),
                )
            )
        )
    responses = client.apply_batch_sync(batch, True)
    actor_ids.extend(
        response.actor_id for response in responses if not response.error
    )
    return ego, controlled, actor_ids


def _spawn_walkers(client, world, count, seed):
    carla = _carla_module()
    rng = random.Random(seed)
    walker_blueprints = list(
        world.get_blueprint_library().filter("walker.pedestrian.*")
    )
    spawn_batch = []
    for _ in range(count):
        location = world.get_random_location_from_navigation()
        if location is None or not walker_blueprints:
            continue
        transform = carla.Transform(location)
        blueprint = rng.choice(walker_blueprints)
        if blueprint.has_attribute("is_invincible"):
            blueprint.set_attribute("is_invincible", "false")
        spawn_batch.append(carla.command.SpawnActor(blueprint, transform))
    responses = client.apply_batch_sync(spawn_batch, True)
    walker_ids = [response.actor_id for response in responses if not response.error]
    controller_blueprint = world.get_blueprint_library().find(
        "controller.ai.walker"
    )
    controller_batch = [
        carla.command.SpawnActor(
            controller_blueprint,
            carla.Transform(),
            walker_id,
        )
        for walker_id in walker_ids
    ]
    controller_responses = client.apply_batch_sync(controller_batch, True)
    controller_ids = [
        response.actor_id
        for response in controller_responses
        if not response.error
    ]
    world.tick()
    for controller_id in controller_ids:
        controller = world.get_actor(controller_id)
        destination = world.get_random_location_from_navigation()
        if controller is None or destination is None:
            continue
        controller.start()
        controller.go_to_location(destination)
        controller.set_max_speed(rng.uniform(0.8, 1.8))
    return walker_ids, controller_ids


def _wait_for_radar_frame(radar, frame, timeout_s):
    deadline = time.monotonic() + timeout_s
    diagnostics = {}
    while time.monotonic() < deadline:
        diagnostics = radar.diagnostics()
        callback_error = diagnostics.get("last_error")
        if callback_error:
            raise RuntimeError(
                f"Radar callback failed while waiting for frame {frame}: "
                f"{callback_error}"
            )
        if int(diagnostics.get("frame", -1)) >= int(frame):
            return diagnostics
        time.sleep(0.002)
    raise TimeoutError(
        f"Radar callback did not reach CARLA frame {frame} within "
        f"{timeout_s:.1f}s; latest radar frame="
        f"{diagnostics.get('frame', -1)}, raw returns="
        f"{diagnostics.get('raw_return_count', 0)}, last error="
        f"{diagnostics.get('last_error')!r}"
    )


def _probe_target(actor_id, target_xy_m, semantic_tag=14):
    distance = float(np.linalg.norm(target_xy_m))
    return IdealRadarTarget(
        object_id=int(actor_id),
        semantic_tag=int(semantic_tag),
        distance_m=distance,
        azimuth_rad=math.atan2(float(target_xy_m[1]), float(target_xy_m[0])),
        relative_velocity_mps=0.0,
        snr_db=45.0,
        point_count=20,
        lateral_extent_m=2.0,
        parent_object_id=int(actor_id),
        path_length_m=2.0 * distance,
    )


def _wrap_angle(angle_rad):
    return (float(angle_rad) + math.pi) % (2.0 * math.pi) - math.pi


def _controlled_target_candidates(
    reflector,
    actor_id,
    config,
    semantic_tag=14,
    range_min_m=0.0,
    range_max_m=None,
    surface_distance_max_m=None,
    type2_azimuth_max_rad=None,
    surface_distance_target_m=None,
    motion_direction="tangent",
):
    """Yield target positions whose path is accepted by the production solver.

    ``range_min_m``/``range_max_m`` restrict the target's range from the
    sensor. The Radar Ghost Dataset's main object walks a few metres from a
    parked car (median 7 m), so the default band in the CLI keeps the
    synthetic scene in the same regime instead of wherever a guardrail
    happens to sit.
    """

    point = np.asarray(reflector.point_xy_m, dtype=np.float64)
    tangent = np.asarray(reflector.tangent_xy, dtype=np.float64)
    normal = np.asarray(reflector.normal_xy, dtype=np.float64)
    line_normal_coordinate = float(np.dot(point, normal))
    line_tangent_coordinate = float(np.dot(point, tangent))
    if abs(line_normal_coordinate) < 1.0e-6:
        return

    # Close surfaces first: real ghost-to-parent range offsets are ~1.3 m for
    # second-order paths, which needs the target within a few metres of the
    # reflector, not the 4-14 m the original smoke test used.
    surface_distances = (1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 11.0, 14.0)
    if range_max_m is None:
        range_max_m = 0.78 * float(config.max_range_m)
    reflection_offsets = (
        0.0,
        -0.12 * float(reflector.length_m),
        0.12 * float(reflector.length_m),
    )
    half_fov = math.radians(float(config.horizontal_fov_deg) / 2.0)
    for surface_distance in surface_distances:
        if surface_distance > float(
            config.multipath_max_target_surface_distance_m
        ):
            continue
        if surface_distance_max_m is not None and surface_distance > float(
            surface_distance_max_m
        ):
            continue
        for reflection_offset in reflection_offsets:
            desired_tangent = line_tangent_coordinate + reflection_offset
            target_xy = (
                (line_normal_coordinate + surface_distance) * normal
                + desired_tangent
                * (line_normal_coordinate - surface_distance)
                / line_normal_coordinate
                * tangent
            )
            target_range = float(np.linalg.norm(target_xy))
            target_azimuth = math.atan2(
                float(target_xy[1]),
                float(target_xy[0]),
            )
            if (
                target_xy[0] <= 3.0
                or target_range < float(range_min_m)
                or target_range > min(float(range_max_m), 0.78 * float(config.max_range_m))
                or abs(target_azimuth) > 0.82 * half_fov
            ):
                continue

            paths = generate_multipath_targets(
                [_probe_target(actor_id, target_xy, semantic_tag)],
                [reflector],
                config,
            )
            if not paths:
                continue
            # Ghost-to-parent azimuth offset, the statistic the calibration
            # compares (real second-order type-2 ghosts sit ~16 deg off
            # their parent).
            if type2_azimuth_max_rad is not None and any(
                path.bounce_type == "type2"
                and abs(
                    _wrap_angle(float(path.azimuth_rad) - target_azimuth)
                ) > float(type2_azimuth_max_rad)
                for path in paths
            ):
                continue

            # The target extractor observes a vehicle surface rather than its
            # actor origin. Prefer geometry that remains valid for nearby
            # points, so the plan is robust to that centroid displacement.
            robust_offsets = (
                -1.0 * tangent,
                1.0 * tangent,
                -0.75 * normal,
                0.75 * normal,
            )
            robust_count = 0
            for offset in robust_offsets:
                nearby_paths = generate_multipath_targets(
                    [_probe_target(actor_id, target_xy + offset, semantic_tag)],
                    [reflector],
                    config,
                )
                robust_count += int(bool(nearby_paths))
            # Probe the geometry along the line the object will actually
            # walk. Real ghost/parent Doppler ratios (~0.9 second order,
            # ~0.8 third order) only come out of motion parallel to the
            # reflector; radial motion into an oblique wall mirrors the
            # velocity and halves the ghost Doppler.
            radial_unit = target_xy / max(
                float(np.linalg.norm(target_xy)),
                1.0e-9,
            )
            motion_unit = tangent if motion_direction == "tangent" else radial_unit
            motion_amplitude = 0.75
            for candidate_amplitude in (1.5, 2.5, 4.0):
                endpoint_paths = [
                    generate_multipath_targets(
                        [
                            _probe_target(
                                actor_id,
                                target_xy
                                + sign * candidate_amplitude * motion_unit,
                                semantic_tag,
                            )
                        ],
                        [reflector],
                        config,
                    )
                    for sign in (-1.0, 1.0)
                ]
                if not all(endpoint_paths):
                    break
                motion_amplitude = candidate_amplitude
            path_families = sorted(
                {
                    f"{path.bounce_type}-order{path.bounce_order}"
                    for path in paths
                }
            )
            # Among equally robust placements prefer the gap closest to the
            # requested one: the type-1 ghost trails its parent by about the
            # gap, and the real offset is ~1.7 m.
            gap_penalty = (
                abs(surface_distance - float(surface_distance_target_m))
                if surface_distance_target_m is not None
                else surface_distance
            )
            score = (
                robust_count,
                len(path_families),
                -gap_penalty,
                motion_amplitude,
                min(float(path.snr_db) for path in paths),
                min(float(reflector.length_m), 20.0),
                -target_range,
            )
            yield {
                "score": score,
                "target_xy_m": target_xy,
                "surface_distance_m": surface_distance,
                "reflector": reflector,
                "path_families": path_families,
                "robust_count": robust_count,
                "motion_amplitude_m": motion_amplitude,
                "motion_unit_sensor": np.asarray(motion_unit, dtype=np.float64),
            }


def _configure_controlled_target(
    radar,
    world,
    target_actor,
    semantic_tag=14,
    target_speed_mps=3.0,
    target_type="vehicle",
    range_min_m=0.0,
    range_max_m=None,
    surface_distance_max_m=None,
    type2_azimuth_max_rad=None,
    surface_distance_target_m=None,
    motion_direction="tangent",
    exclude_reflectors=(),
):
    """Put a CARLA actor into geometry known to produce physical paths."""

    snapshot = radar.debug_snapshot()
    reflector_dicts = snapshot.get("reflectors", ())
    reflectors = [ReflectorSegment(**item) for item in reflector_dicts]
    candidates = []
    for reflector in reflectors:
        candidates.extend(
            _controlled_target_candidates(
                reflector,
                target_actor.id,
                radar.realistic_config,
                semantic_tag,
                range_min_m=range_min_m,
                range_max_m=range_max_m,
                surface_distance_max_m=surface_distance_max_m,
                type2_azimuth_max_rad=type2_azimuth_max_rad,
                surface_distance_target_m=surface_distance_target_m,
                motion_direction=motion_direction,
            )
        )
    constrained = (
        range_min_m > 0.0
        or range_max_m is not None
        or surface_distance_max_m is not None
        or type2_azimuth_max_rad is not None
    )
    if not candidates and constrained:
        print(
            f"  no placement inside range {range_min_m:.0f}-"
            f"{'any' if range_max_m is None else f'{range_max_m:.0f}'} m, "
            f"surface <= {surface_distance_max_m}, type-2 azimuth <= "
            f"{'any' if type2_azimuth_max_rad is None else f'{math.degrees(type2_azimuth_max_rad):.0f} deg'}; "
            "falling back to any geometry (the scene will not match RGD)"
        )
        for reflector in reflectors:
            candidates.extend(
                _controlled_target_candidates(
                    reflector,
                    target_actor.id,
                    radar.realistic_config,
                    semantic_tag,
                )
            )
    if exclude_reflectors:
        excluded = {int(item) for item in exclude_reflectors}
        candidates = [
            item
            for item in candidates
            if int(item["reflector"].reflector_id) not in excluded
        ]
    if not candidates:
        observed_tags = sorted(
            {int(reflector.semantic_tag) for reflector in reflectors}
        )
        raise RuntimeError(
            "The radar observed reflector surfaces but could not find a "
            "valid controlled multipath placement. "
            f"reflectors={len(reflectors)}, tags={observed_tags}. "
            "Try another seed or continue to a reflector-rich road before "
            "changing the geometry gates."
        )

    candidate = max(candidates, key=lambda item: item["score"])
    reflector = candidate["reflector"]
    target_xy = candidate["target_xy_m"]
    sensor_transform = radar.sensor.get_transform()
    sensor_yaw_rad = math.radians(float(sensor_transform.rotation.yaw))
    cosine = math.cos(sensor_yaw_rad)
    sine = math.sin(sensor_yaw_rad)
    world_x = (
        float(sensor_transform.location.x)
        + cosine * float(target_xy[0])
        - sine * float(target_xy[1])
    )
    world_y = (
        float(sensor_transform.location.y)
        + sine * float(target_xy[0])
        + cosine * float(target_xy[1])
    )
    carla = _carla_module()
    waypoint = world.get_map().get_waypoint(
        carla.Location(
            x=world_x,
            y=world_y,
            z=float(sensor_transform.location.z),
        ),
        project_to_road=True,
    )
    world_z = (
        float(waypoint.transform.location.z) + 0.35
        if waypoint is not None
        else float(sensor_transform.location.z) - 0.65
    )

    # Motion direction: the candidate's probed unit vector (reflector
    # tangent by default, or the sensor-to-target line) in sensor
    # coordinates, rotated into world coordinates. Only the projection onto
    # the line of sight reaches the Doppler axis, so the expected mean |vr|
    # is the walking speed times |cos| of the angle to the line of sight.
    target_range = max(float(np.linalg.norm(target_xy)), 1.0e-9)
    radial_sensor = np.asarray(target_xy, dtype=np.float64) / target_range
    motion_sensor = np.asarray(candidate["motion_unit_sensor"], dtype=np.float64)
    motion_dir_world = np.array(
        (
            cosine * motion_sensor[0] - sine * motion_sensor[1],
            sine * motion_sensor[0] + cosine * motion_sensor[1],
        ),
        dtype=np.float64,
    )
    expected_radial_speed = float(target_speed_mps) * abs(
        float(np.dot(motion_sensor, radial_sensor))
    )
    tangent = np.asarray(reflector.tangent_xy, dtype=np.float64)
    tangent_world = np.array(
        (
            cosine * tangent[0] - sine * tangent[1],
            sine * tangent[0] + cosine * tangent[1],
        ),
        dtype=np.float64,
    )
    target_yaw = math.degrees(
        math.atan2(float(tangent_world[1]), float(tangent_world[0]))
    )
    target_actor.set_simulate_physics(False)
    target_actor.set_transform(
        carla.Transform(
            carla.Location(x=world_x, y=world_y, z=world_z),
            carla.Rotation(yaw=target_yaw),
        )
    )
    target_actor.set_target_velocity(carla.Vector3D())
    return {
        "actor": target_actor,
        "base_world_xy": np.array((world_x, world_y), dtype=np.float64),
        "world_z": world_z,
        "motion_dir_world": motion_dir_world,
        "tangent_world": tangent_world,
        "yaw": target_yaw,
        "semantic_tag": int(semantic_tag),
        "target_type": str(target_type),
        "target_speed_mps": float(target_speed_mps),
        "expected_radial_speed_mps": expected_radial_speed,
        "motion_direction": str(motion_direction),
        "amplitude_m": float(candidate["motion_amplitude_m"]),
        # Triangular (constant-speed) profile: with period = 4*amplitude/speed
        # the target walks at exactly the configured speed between the two
        # validated radial endpoints, so mean |radial velocity| ≈ speed.
        "period_s": max(
            4.0,
            4.0
            * float(candidate["motion_amplitude_m"])
            / max(float(target_speed_mps), 0.5),
        ),
        "reflector_id": int(reflector.reflector_id),
        "reflector_tag": int(reflector.semantic_tag),
        "reflector_length_m": float(reflector.length_m),
        "base_target_range_m": float(np.linalg.norm(target_xy)),
        "target_surface_distance_m": float(candidate["surface_distance_m"]),
        "expected_path_families": candidate["path_families"],
        "robust_probe_count": int(candidate["robust_count"]),
        "motion_amplitude_m": float(candidate["motion_amplitude_m"]),
    }


def _triangular_offset_speed(elapsed_s, amplitude_m, period_s):
    """Constant-speed back-and-forth motion between -/+ amplitude.

    A triangular position wave keeps |speed| = 4*amplitude/period at all
    times except the instantaneous reversal, matching how the Radar Ghost
    Dataset records its main object: a pedestrian/cyclist walking away from
    and back toward the stationary ego at roughly constant speed. With the
    period set to 4*amplitude/speed, the mean radial Doppler equals the
    configured walking speed.
    """

    half_period = float(period_s) / 2.0
    phase = float(elapsed_s) % float(period_s)
    speed = 4.0 * float(amplitude_m) / max(float(period_s), 1.0e-9)
    if phase < half_period:
        offset = -float(amplitude_m) + speed * phase
    else:
        offset = float(amplitude_m) - speed * (phase - half_period)
        speed = -speed
    return offset, speed


def _update_controlled_target(plan, step, fps):
    carla = _carla_module()
    elapsed = float(step) / float(fps)
    offset, speed = _triangular_offset_speed(
        elapsed,
        float(plan["amplitude_m"]),
        float(plan["period_s"]),
    )
    # Back-and-forth motion along the planned direction; the Doppler the
    # radar sees is the projection onto the line of sight.
    location_xy = plan["base_world_xy"] + offset * plan["motion_dir_world"]
    velocity_xy = speed * plan["motion_dir_world"]
    # All target types are driven kinematically: the exact transform keeps
    # the multipath geometry valid, and the target velocity is reported to
    # the radar adapter (directly by CARLA for vehicles, or through the
    # transform-derivative fallback for walkers).
    plan["actor"].set_transform(
        carla.Transform(
            carla.Location(
                x=float(location_xy[0]),
                y=float(location_xy[1]),
                z=float(plan["world_z"]),
            ),
            carla.Rotation(yaw=float(plan["yaw"])),
        )
    )
    plan["actor"].set_target_velocity(
        carla.Vector3D(
            x=float(velocity_xy[0]),
            y=float(velocity_xy[1]),
            z=0.0,
        )
    )


def _validate_controlled_target(radar, world, plan, fps, timeout_s):
    """Require the placed CARLA actor itself to produce a multipath path."""

    target_id = int(plan["actor"].id)
    last_dynamic_ids = []
    for step in range(12):
        _update_controlled_target(plan, step, fps)
        frame = world.tick()
        _wait_for_radar_frame(radar, frame, timeout_s=timeout_s)
        snapshot = radar.debug_snapshot()
        ideal_targets = snapshot.get("ideal_targets", ())
        last_dynamic_ids = [
            int(target.get("object_id", 0))
            for target in ideal_targets
            if int(target.get("semantic_tag", -1))
            in (12, 13, 14, 15, 16, 17, 18, 19, 21)
        ]
        matching_paths = [
            target
            for target in snapshot.get("multipath_ideal_targets", ())
            if int(target.get("parent_object_id", 0)) == target_id
        ]
        if matching_paths:
            plan["validated_path_families"] = sorted(
                {
                    f"{path.get('bounce_type')}-order"
                    f"{int(path.get('bounce_order', 0))}"
                    for path in matching_paths
                }
            )
            plan["validation_multipath_count"] = len(matching_paths)
            return step + 1
    raise RuntimeError(
        "The controlled CARLA vehicle did not produce an observed multipath "
        "target, so collection was stopped instead of writing false labels. "
        f"target_id={target_id}, reflector_id={plan['reflector_id']}, "
        f"reflector_tag={plan['reflector_tag']}, "
        f"last_dynamic_ids={last_dynamic_ids}. Try a different --seed."
    )


def collect_sequence(client, args, sequence_index):
    carla = _carla_module()
    world = client.load_world(args.town)
    original_settings = world.get_settings()
    traffic_manager = client.get_trafficmanager(args.traffic_manager_port)
    radar = None
    ego = None
    controlled_target = None
    controlled_plan = None
    spectator = None
    npc_ids = []
    walker_ids = []
    walker_controller_ids = []
    rows = []
    diagnostics_summary = {}
    aggregate = {
        "capture_frames": 0,
        "max_reflector_count": 0,
        "max_multipath_count": 0,
        "sum_reflector_count": 0,
        "sum_multipath_count": 0,
        "sum_dynamic_ideal_targets": 0,
    }
    sequence_seed = args.seed + sequence_index * 1009
    semantic_tag = TARGET_SEMANTIC_TAGS[args.target_type]
    target_speed_mps = (
        float(args.target_speed_mps)
        if args.target_speed_mps is not None
        else TARGET_SPEEDS_MPS[args.target_type]
    )
    try:
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 1.0 / args.fps
        settings.no_rendering_mode = False
        world.apply_settings(settings)
        traffic_manager.set_synchronous_mode(True)
        traffic_manager.set_random_device_seed(sequence_seed)
        weather_index = _set_weather(world, sequence_index)
        ego, controlled_target, npc_ids = _spawn_vehicles(
            client,
            world,
            traffic_manager,
            args.vehicles,
            sequence_seed,
            args.lead_distance,
            args.target_type,
        )
        if not args.headless:
            spectator = world.get_spectator()
            _update_spectator_camera(
                spectator,
                ego,
                controlled_target,
                args.camera_view,
            )
        walker_ids, walker_controller_ids = _spawn_walkers(
            client,
            world,
            args.walkers,
            sequence_seed + 17,
        )
        radar = create_front_radar(
            ego,
            world,
            range_m=args.range_m,
            backend="realistic",
            fps=args.fps,
            points_per_second=args.points_per_second,
            profile_name="rgd_regime_v1",
            config_path=args.radar_config,
            seed=sequence_seed,
            capture_debug=True,
        )
        # A newly spawned CARLA sensor can skip its second spawn-time frame.
        # Advance twice but wait for the first frame: this proves that one
        # complete scan exists without deadlocking on a frame CARLA omitted.
        startup_frame = world.tick()
        _update_spectator_camera(
            spectator,
            ego,
            controlled_target,
            args.camera_view,
        )
        world.tick()
        _wait_for_radar_frame(
            radar,
            startup_frame,
            timeout_s=args.radar_timeout,
        )
        placement_kwargs = dict(
            range_min_m=args.target_range_min,
            range_max_m=args.target_range_max,
            surface_distance_max_m=args.surface_distance_max,
            type2_azimuth_max_rad=math.radians(args.type2_azimuth_max_deg),
            surface_distance_target_m=args.surface_distance_target,
            motion_direction=args.motion_direction,
        )
        controlled_plan = _configure_controlled_target(
            radar,
            world,
            controlled_target,
            semantic_tag,
            target_speed_mps,
            args.target_type,
            **placement_kwargs,
        )
        amplitude_gain = 10.0 ** (
            float(getattr(radar.realistic_config, "amplitude_gain_db", 0.0)) / 20.0
        )
        main_object_ids = (
            {int(controlled_target.id)} if args.label_scope == "main" else None
        )
        _update_spectator_camera(
            spectator,
            ego,
            controlled_target,
            args.camera_view,
        )
        def _describe_plan(plan):
            print(
                "  Controlled target "
                f"{controlled_target.id} (type={args.target_type}, "
                f"tag={semantic_tag}) at "
                f"{plan['base_target_range_m']:.1f} m; "
                f"reflector tag={plan['reflector_tag']}, "
                f"surface gap="
                f"{plan['target_surface_distance_m']:.1f} m; "
                f"motion={plan['motion_direction']} "
                f"(expected |vr| {plan['expected_radial_speed_mps']:.2f} m/s); "
                f"camera={args.camera_view}"
            )

        _describe_plan(controlled_plan)
        tried_reflectors = {int(controlled_plan["reflector_id"])}
        for placement_attempt in range(MAX_PLACEMENT_ATTEMPTS):
            try:
                controlled_step = _validate_controlled_target(
                    radar,
                    world,
                    controlled_plan,
                    args.fps,
                    args.radar_timeout,
                )
                break
            except RuntimeError as validation_error:
                if placement_attempt + 1 >= MAX_PLACEMENT_ATTEMPTS:
                    raise
                print(
                    f"  placement rejected: {validation_error}\n"
                    "  trying another reflector"
                )
                try:
                    controlled_plan = _configure_controlled_target(
                        radar,
                        world,
                        controlled_target,
                        semantic_tag,
                        target_speed_mps,
                        args.target_type,
                        exclude_reflectors=tried_reflectors,
                        **placement_kwargs,
                    )
                except RuntimeError:
                    raise validation_error
                tried_reflectors.add(int(controlled_plan["reflector_id"]))
                _describe_plan(controlled_plan)
        controlled_plan["placement_attempts"] = placement_attempt + 1
        warmup_frames = int(round(args.warmup * args.fps))
        capture_frames = int(round(args.duration * args.fps))
        for _ in range(warmup_frames):
            _update_controlled_target(
                controlled_plan,
                controlled_step,
                args.fps,
            )
            controlled_step += 1
            if args.camera_view == "chase":
                _update_spectator_camera(
                    spectator,
                    ego,
                    controlled_target,
                    args.camera_view,
                )
            frame = world.tick()
            _wait_for_radar_frame(
                radar,
                frame,
                timeout_s=args.radar_timeout,
            )
        for _ in range(capture_frames):
            _update_controlled_target(
                controlled_plan,
                controlled_step,
                args.fps,
            )
            controlled_step += 1
            if args.camera_view == "chase":
                _update_spectator_camera(
                    spectator,
                    ego,
                    controlled_target,
                    args.camera_view,
                )
            frame = world.tick()
            diagnostics = _wait_for_radar_frame(
                radar,
                frame,
                timeout_s=args.radar_timeout,
            )
            snapshot = radar.debug_snapshot()
            radar_frame = int(diagnostics.get("frame", frame))
            timestamp = float(
                diagnostics.get("timestamp") or radar_frame / args.fps
            )
            detections = snapshot.get("generated_detections", ())
            ideal_targets = snapshot.get("ideal_targets", ())
            reflector_count = int(diagnostics.get("reflector_count", 0))
            multipath_count = int(
                diagnostics.get("multipath_ideal_target_count", 0)
            )
            dynamic_ideal_count = sum(
                int(target.get("semantic_tag", -1))
                in (12, 13, 14, 15, 16, 17, 18, 19, 21)
                for target in ideal_targets
            )
            aggregate["capture_frames"] += 1
            aggregate["max_reflector_count"] = max(
                aggregate["max_reflector_count"],
                reflector_count,
            )
            aggregate["max_multipath_count"] = max(
                aggregate["max_multipath_count"],
                multipath_count,
            )
            aggregate["sum_reflector_count"] += reflector_count
            aggregate["sum_multipath_count"] += multipath_count
            aggregate["sum_dynamic_ideal_targets"] += dynamic_ideal_count
            frame_rng = np.random.default_rng(
                args.seed * 1_000_003
                + sequence_index * 10_007
                + radar_frame
            )
            if args.points_source == "sensor":
                # The radar's own point emission (delivered scan, one cycle of
                # latency): calibrated points-per-object, footprint and
                # micro-Doppler flow straight into the export.
                for point_index, point in enumerate(
                    snapshot.get("point_detections", ())
                ):
                    rows.append(
                        _detection_row(
                            sequence_index,
                            radar_frame,
                            timestamp,
                            point_index,
                            point,
                            main_object_ids=main_object_ids,
                            amplitude_gain=amplitude_gain,
                        )
                    )
            else:
                for detection_index, detection in enumerate(detections):
                    if args.expand_points:
                        expanded = _expand_detection_points(
                            detection,
                            frame_rng,
                            args.points_per_detection,
                        )
                    else:
                        expanded = [detection]
                    for point_index, point in enumerate(expanded):
                        rows.append(
                            _detection_row(
                                sequence_index,
                                radar_frame,
                                timestamp,
                                (detection_index << 8) + point_index,
                                point,
                                main_object_ids=main_object_ids,
                                amplitude_gain=amplitude_gain,
                            )
                        )
            diagnostics_summary = {
                "last_frame": radar_frame,
                "last_reflector_count": int(
                    reflector_count
                ),
                "last_multipath_count": int(
                    multipath_count
                ),
                "radar_profile": diagnostics.get("profile"),
                "radar_config_signature": diagnostics.get(
                    "config_signature"
                ),
                "weather_index": weather_index,
                "target_type": args.target_type,
                "target_semantic_tag": int(semantic_tag),
                "target_speed_mps_expected": float(
                    (controlled_plan or {}).get(
                        "expected_radial_speed_mps", target_speed_mps
                    )
                ),
                "motion_direction": (controlled_plan or {}).get("motion_direction"),
                "radar_fps": int(args.fps),
                "radar_fov_deg": float(
                    radar.realistic_config.horizontal_fov_deg
                ),
                "radar_cycle_time_s": float(
                    radar.realistic_config.cycle_time_s
                ),
                "ego_speed_mps": float(
                    np.linalg.norm(
                        (
                            ego.get_velocity().x,
                            ego.get_velocity().y,
                            ego.get_velocity().z,
                        )
                    )
                ),
                "controlled_target_id": int(controlled_target.id),
                "controlled_reflector_id": controlled_plan[
                    "reflector_id"
                ],
                "controlled_reflector_tag": controlled_plan[
                    "reflector_tag"
                ],
                "controlled_reflector_length_m": controlled_plan[
                    "reflector_length_m"
                ],
                "controlled_target_range_m": controlled_plan[
                    "base_target_range_m"
                ],
                "controlled_target_surface_distance_m": controlled_plan[
                    "target_surface_distance_m"
                ],
                "controlled_expected_path_families": controlled_plan[
                    "expected_path_families"
                ],
                "controlled_robust_probe_count": controlled_plan[
                    "robust_probe_count"
                ],
                "controlled_motion_amplitude_m": controlled_plan[
                    "motion_amplitude_m"
                ],
                "controlled_placement_attempts": int(
                    controlled_plan.get("placement_attempts", 1)
                ),
                "controlled_validated_path_families": controlled_plan[
                    "validated_path_families"
                ],
                "controlled_validation_multipath_count": controlled_plan[
                    "validation_multipath_count"
                ],
                **aggregate,
            }
    finally:
        if radar is not None:
            radar.cleanup()
        for controller_id in walker_controller_ids:
            controller = world.get_actor(controller_id)
            if controller is not None:
                try:
                    controller.stop()
                except RuntimeError:
                    pass
        destroy_ids = list(npc_ids) + list(walker_controller_ids) + list(walker_ids)
        if ego is not None:
            destroy_ids.append(ego.id)
        if destroy_ids:
            client.apply_batch(
                [carla.command.DestroyActor(actor_id) for actor_id in destroy_ids]
            )
        traffic_manager.set_synchronous_mode(False)
        world.apply_settings(original_settings)
    return np.asarray(rows, dtype=RADAR_DTYPE), diagnostics_summary


def _add_point_statistics(counts, radar_data, diagnostics):
    """Augment a sequence summary with label-class and Doppler checks."""

    labeled = radar_data["label_id"] > 0
    classes, class_counts = np.unique(
        (radar_data["label_id"][labeled] // 1000) % 10,
        return_counts=True,
    )
    counts["label_class_histogram"] = {
        int(cls): int(count) for cls, count in zip(classes, class_counts)
    }

    ghost_mask = (radar_data["label_id"] > 0) & (
        radar_data["label_id"] % 10 != 1
    )
    families = {}
    if np.any(ghost_mask):
        ghost_rows = radar_data[ghost_mask]
        for bounce_type, bounce_order in zip(
            ghost_rows["bounce_type"],
            ghost_rows["bounce_order"],
        ):
            key = f"{bounce_type.decode()}-order{int(bounce_order)}"
            families[key] = families.get(key, 0) + 1
    counts["ghost_family_histogram"] = families

    target_id = int(diagnostics.get("controlled_target_id", -1))
    direct_mask = (
        (radar_data["instance_id"] == target_id)
        & (radar_data["source"] == b"direct")
    )
    speeds = np.abs(radar_data["vr_sc"][direct_mask]).astype(np.float64)
    counts["direct_target_detection_count"] = int(np.count_nonzero(direct_mask))
    counts["direct_target_speed_mean_mps"] = float(
        np.mean(speeds) if speeds.size else 0.0
    )
    counts["direct_target_speed_max_mps"] = float(
        np.max(speeds) if speeds.size else 0.0
    )


def _verification_block(counts):
    """Render a copyable PASS/FAIL block for RGD-regime verification."""

    target_type = counts.get("target_type", "?")
    expected_class = _class_id(TARGET_SEMANTIC_TAGS.get(target_type, 14))
    issues = []
    lines = []

    def check(label, ok, detail=""):
        status = "PASS" if ok else "FAIL"
        if not ok:
            issues.append(label)
        lines.append(f"  [{status}] {label}: {detail}")

    lines.append("=" * 74)
    lines.append("RGD REGIME COLLECTION VERIFICATION — COPY THIS BLOCK BACK")
    lines.append("=" * 74)
    lines.append(f"file: {counts.get('path', '')}")
    lines.append(
        f"target_type: {target_type} "
        f"(tag {counts.get('target_semantic_tag', '?')}, "
        f"expected RGD class {expected_class})"
    )
    lines.append(
        f"fps: {counts.get('radar_fps', '?')}  "
        f"cycle: {counts.get('radar_cycle_time_s', '?')}s  "
        f"fov: {counts.get('radar_fov_deg', '?')} deg  "
        f"profile: {counts.get('radar_profile', '?')}"
    )
    lines.append(
        f"reflector: id={counts.get('controlled_reflector_id', '?')} "
        f"tag={counts.get('controlled_reflector_tag', '?')} "
        f"length={counts.get('controlled_reflector_length_m', '?')}m"
    )
    lines.append(
        "validated_path_families: "
        f"{counts.get('controlled_validated_path_families', '?')}"
    )
    lines.append(
        f"points: {counts.get('points', 0)}  "
        f"real: {counts.get('real', 0)}  "
        f"ghost: {counts.get('ghost', 0)}  "
        f"classes: {counts.get('label_class_histogram', {})}"
    )
    frames = max(int(counts.get("capture_frames", 0) or 0), 1)
    lines.append(
        f"per scan: {counts.get('points', 0) / frames:.0f} points, "
        f"{counts.get('real', 0) / frames:.1f} labelled real, "
        f"{counts.get('ghost', 0) / frames:.1f} labelled ghost   "
        "(RGD train per sensor-scan: 336 points, 6.9 real, 1.8 ghost)"
    )
    lines.append(
        f"target range: {float(counts.get('controlled_target_range_m', float('nan'))):.1f} m "
        "(RGD main object median ~7 m)"
    )
    lines.append(
        f"direct target: "
        f"{counts.get('direct_target_detection_count', 0)} detections, "
        "mean |vr|="
        f"{counts.get('direct_target_speed_mean_mps', 0.0):.3f} m/s "
        "(expected ~"
        f"{counts.get('target_speed_mps_expected', 0.0)} m/s), "
        "max |vr|="
        f"{counts.get('direct_target_speed_max_mps', 0.0):.3f} m/s"
    )
    lines.append(f"ghost_families: {counts.get('ghost_family_histogram', {})}")
    lines.append("-" * 74)
    check(
        "ego stationary",
        float(counts.get("ego_speed_mps", float("inf"))) < 0.5,
        f"{counts.get('ego_speed_mps', float('nan')):.3f} m/s",
    )
    check("radar fps = 10", int(counts.get("radar_fps", 0)) == 10)
    check(
        "radar fov = 140 deg",
        abs(float(counts.get("radar_fov_deg", 0.0)) - 140.0) < 1.0e-6,
        f"{counts.get('radar_fov_deg', 0.0)} deg",
    )
    check("ghost points > 0", int(counts.get("ghost", 0)) > 0)
    check("real points > 0", int(counts.get("real", 0)) > 0)
    check(
        "expected class present",
        expected_class in counts.get("label_class_histogram", {}),
        f"class {expected_class} in "
        f"{counts.get('label_class_histogram', {})}",
    )
    check(
        "direct target Doppler alive",
        float(counts.get("direct_target_speed_mean_mps", 0.0)) > 0.05,
    )
    expected_speed = float(counts.get("target_speed_mps_expected", 0.0))
    measured = float(counts.get("direct_target_speed_mean_mps", 0.0))
    measured_max = float(counts.get("direct_target_speed_max_mps", 0.0))
    if expected_speed > 0.0:
        # The expected value is the walking speed projected onto the line of
        # sight. The triangular profile keeps |speed| constant except at the
        # instantaneous reversal, so the mean |vr| should sit close to it.
        check(
            "direct target speed plausible",
            0.1 * expected_speed <= measured <= 1.6 * expected_speed,
            f"mean |vr|={measured:.3f} vs {expected_speed:.3f} m/s "
            f"(max {measured_max:.3f})",
        )
    check(
        "controlled reflector used",
        counts.get("controlled_reflector_id") not in (None, ""),
        str(counts.get("controlled_reflector_id")),
    )
    lines.append("-" * 74)
    if issues:
        lines.append(
            f"RESULT: {len(issues)} check(s) failed: {', '.join(issues)}"
        )
    else:
        lines.append("RESULT: ALL CHECKS PASSED")
    lines.append("=" * 74)
    return "\n".join(lines)


def _sequence_path(args, sequence_index):
    return (
        Path(args.output)
        / args.split
        / (
            f"scenario-{args.town.lower()}-synthetic-{sequence_index:03d}_sequence-01_"
            f"car_{args.split}.h5"
        )
    )


def _write_sequence(path, radar_data, diagnostics, args, sequence_index):
    with h5py.File(path, "w") as handle:
        handle.create_dataset(
            "radar",
            data=radar_data,
            compression="gzip",
            shuffle=True,
        )
        handle.attrs["generator"] = "CARLA 0.9.16 rgd_regime_v1"
        handle.attrs["town"] = args.town
        handle.attrs["seed"] = args.seed + sequence_index * 1009
        handle.attrs["arguments"] = json.dumps(vars(args), sort_keys=True)
        handle.attrs["radar_config_signature"] = diagnostics.get(
            "radar_config_signature",
            "",
        )
        controlled_metadata = {
            key: value
            for key, value in diagnostics.items()
            if key.startswith("controlled_")
        }
        handle.attrs["controlled_target"] = json.dumps(
            controlled_metadata,
            sort_keys=True,
        )


def _sequence_counts(path, radar_data, diagnostics, args):
    counts = {
        "path": str(path),
        "points": int(len(radar_data)),
        "real": int(np.count_nonzero(radar_data["label_id"] % 10 == 1)),
        "ghost": int(
            np.count_nonzero(
                (radar_data["label_id"] > 0)
                & (radar_data["label_id"] % 10 != 1)
            )
        ),
        **diagnostics,
    }
    _add_point_statistics(counts, radar_data, diagnostics)
    return counts


def _run_sequence_worker(args, sequence_index):
    """Collect one sequence in a fresh process.

    A fresh process per sequence prevents native CARLA state from
    accumulating across world reloads (which segfaults after ~7 sequences
    when running everything in one process).
    """

    carla = _carla_module()
    client = carla.Client(args.host, args.port)
    client.set_timeout(30.0)
    path = _sequence_path(args, sequence_index)
    summary_path = path.with_suffix(".summary.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Collecting sequence {sequence_index + 1}/{args.sequences}")
    radar_data, diagnostics = collect_sequence(client, args, sequence_index)
    _write_sequence(path, radar_data, diagnostics, args, sequence_index)
    counts = _sequence_counts(path, radar_data, diagnostics, args)
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(counts, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(counts, sort_keys=True))
    print(_verification_block(counts))
    return counts


def _run_supervisor(args):
    """Run each sequence in its own worker subprocess, with resume/retry."""

    output = Path(args.output) / args.split
    output.mkdir(parents=True, exist_ok=True)
    worker_arguments = list(sys.argv[1:])
    summaries = []
    failures = []
    completed_indices = set()
    for sequence_index in range(args.sequences):
        path = _sequence_path(args, sequence_index)
        summary_path = path.with_suffix(".summary.json")
        if (
            args.resume
            and path.is_file()
            and summary_path.is_file()
        ):
            print(
                f"Skipping completed sequence {sequence_index + 1}/"
                f"{args.sequences}: {summary_path}"
            )
            completed_indices.add(sequence_index)
            continue

        return_code = None
        for attempt in range(args.sequence_retries + 1):
            if attempt:
                print(
                    f"Retrying sequence {sequence_index + 1}/"
                    f"{args.sequences} after worker exit {return_code}"
                )
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                *worker_arguments,
                "--worker-index",
                str(sequence_index),
            ]
            result = subprocess.run(command)
            return_code = int(result.returncode)
            if return_code == 0 and summary_path.is_file():
                completed_indices.add(sequence_index)
                break
        if return_code != 0 or not summary_path.is_file():
            failures.append(
                {
                    "sequence_index": sequence_index,
                    "town": args.town,
                    "worker_return_code": return_code,
                }
            )

    for sequence_index in range(args.sequences):
        if sequence_index not in completed_indices:
            continue
        summary_path = _sequence_path(args, sequence_index).with_suffix(
            ".summary.json"
        )
        with summary_path.open("r", encoding="utf-8") as handle:
            summaries.append(json.load(handle))
    summary_path = Path(args.output) / f"collection_{args.town.lower()}_{args.split}.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {"sequences": summaries, "failed_sequences": failures},
            handle,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")
    print(
        f"Ready: {len(summaries)} sequences, {len(failures)} failures, "
        f"summary={summary_path}"
    )
    if failures:
        raise RuntimeError(
            "Some CARLA sequence workers failed; rerun the same command with "
            "--resume after checking that the CARLA server is alive"
        )


def main():
    args = parse_args()
    if (
        args.sequences < 1
        or args.duration <= 0.0
        or args.fps < 1
        or args.radar_timeout <= 0.0
        or args.lead_distance <= 0.0
        or args.sequence_retries < 0
    ):
        raise ValueError(
            "sequences, duration, fps, radar-timeout, lead-distance, and "
            "sequence-retries must be valid"
        )
    if args.worker_index is None:
        _run_supervisor(args)
        return
    if not 0 <= args.worker_index < args.sequences:
        raise ValueError("worker-index is outside the requested sequence range")
    _run_sequence_worker(args, args.worker_index)


if __name__ == "__main__":
    main()
