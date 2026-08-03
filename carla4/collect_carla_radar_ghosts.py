#!/usr/bin/env python3
"""Collect path-labeled geometry multipath target lists in CARLA 0.9.16."""

import argparse
import json
import math
from pathlib import Path
import random
import time
import uuid

import h5py
import numpy as np

from radar import create_front_radar
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


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--traffic-manager-port", type=int, default=8000)
    parser.add_argument("--town", default="Town04")
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="train")
    parser.add_argument("--sequences", type=int, default=20)
    parser.add_argument("--duration", type=float, default=45.0)
    parser.add_argument("--warmup", type=float, default=3.0)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--vehicles", type=int, default=45)
    parser.add_argument("--walkers", type=int, default=25)
    parser.add_argument(
        "--lead-distance",
        type=float,
        default=25.0,
        help="initial spawn distance for the controlled dynamic radar target",
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
        help="do not update the CARLA spectator chase camera",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--radar-config", help="optional geometry-profile overrides")
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


def _cmto_label(detection):
    source = detection.get("source", "")
    if source == "clutter":
        return -2
    semantic_tag = int(detection.get("semantic_tag", 0))
    if semantic_tag not in (12, 13, 14, 15, 16, 17, 18, 19, 21):
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


def _detection_row(sequence_id, frame, timestamp, index, detection):
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
        float(snr_db_to_amplitude(detection["snr_db"])),
        identifier,
        _cmto_label(detection),
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


def _update_spectator_camera(spectator, ego):
    """Match the scenarios/ third-person chase-camera placement."""

    if spectator is None or ego is None or not ego.is_alive:
        return
    carla = _carla_module()
    ego_transform = ego.get_transform()
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
    ego_waypoint = world.get_map().get_waypoint(
        ego.get_transform().location,
        project_to_road=True,
    )
    lead = None
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
                blueprint = rng.choice(road_vehicle_blueprints)
                if blueprint.has_attribute("role_name"):
                    blueprint.set_attribute("role_name", "radar_target")
                lead = world.try_spawn_actor(blueprint, transform)
                if lead is not None:
                    break
            if lead is not None:
                break
    if lead is None:
        raise RuntimeError(
            "Unable to spawn the guaranteed lead radar target ahead of ego"
        )
    lead.apply_control(carla.VehicleControl(hand_brake=True))
    actor_ids.append(lead.id)

    batch = []
    available_spawn_points = [
        transform
        for index, transform in enumerate(spawn_points)
        if index != ego_spawn_index
        and transform.location.distance(lead.get_transform().location) > 8.0
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
    return ego, lead, actor_ids


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


def _probe_target(actor_id, target_xy_m):
    distance = float(np.linalg.norm(target_xy_m))
    return IdealRadarTarget(
        object_id=int(actor_id),
        semantic_tag=14,
        distance_m=distance,
        azimuth_rad=math.atan2(float(target_xy_m[1]), float(target_xy_m[0])),
        relative_velocity_mps=0.0,
        snr_db=45.0,
        point_count=20,
        lateral_extent_m=2.0,
        parent_object_id=int(actor_id),
        path_length_m=2.0 * distance,
    )


def _controlled_target_candidates(reflector, actor_id, config):
    """Yield target positions whose path is accepted by the production solver."""

    point = np.asarray(reflector.point_xy_m, dtype=np.float64)
    tangent = np.asarray(reflector.tangent_xy, dtype=np.float64)
    normal = np.asarray(reflector.normal_xy, dtype=np.float64)
    line_normal_coordinate = float(np.dot(point, normal))
    line_tangent_coordinate = float(np.dot(point, tangent))
    if abs(line_normal_coordinate) < 1.0e-6:
        return

    surface_distances = (4.0, 6.0, 8.0, 11.0, 14.0)
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
                or target_range > 0.78 * float(config.max_range_m)
                or abs(target_azimuth) > 0.82 * half_fov
            ):
                continue

            paths = generate_multipath_targets(
                [_probe_target(actor_id, target_xy)],
                [reflector],
                config,
            )
            if not paths:
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
                    [_probe_target(actor_id, target_xy + offset)],
                    [reflector],
                    config,
                )
                robust_count += int(bool(nearby_paths))
            path_families = sorted(
                {
                    f"{path.bounce_type}-order{path.bounce_order}"
                    for path in paths
                }
            )
            score = (
                robust_count,
                len(path_families),
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
            }


def _configure_controlled_target(radar, world, target_actor):
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
            )
        )
    if not candidates:
        observed_tags = sorted(
            {int(reflector.semantic_tag) for reflector in reflectors}
        )
        raise RuntimeError(
            "The radar observed reflector surfaces but could not find a "
            "valid controlled multipath placement. "
            f"reflectors={len(reflectors)}, tags={observed_tags}. "
            "Try another Town04 seed before changing the geometry gates."
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
        "tangent_world": tangent_world,
        "yaw": target_yaw,
        "amplitude_m": min(0.75, max(0.25, 0.08 * reflector.length_m)),
        "period_s": 8.0,
        "reflector_id": int(reflector.reflector_id),
        "reflector_tag": int(reflector.semantic_tag),
        "reflector_length_m": float(reflector.length_m),
        "base_target_range_m": float(np.linalg.norm(target_xy)),
        "target_surface_distance_m": float(candidate["surface_distance_m"]),
        "expected_path_families": candidate["path_families"],
        "robust_probe_count": int(candidate["robust_count"]),
    }


def _update_controlled_target(plan, step, fps):
    carla = _carla_module()
    omega = 2.0 * math.pi / float(plan["period_s"])
    elapsed = float(step) / float(fps)
    offset = float(plan["amplitude_m"]) * math.sin(omega * elapsed)
    speed = float(plan["amplitude_m"]) * omega * math.cos(omega * elapsed)
    location_xy = plan["base_world_xy"] + offset * plan["tangent_world"]
    velocity_xy = speed * plan["tangent_world"]
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
        )
        if not args.headless:
            spectator = world.get_spectator()
            _update_spectator_camera(spectator, ego)
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
            profile_name="geometry_multipath_v1",
            config_path=args.radar_config,
            seed=sequence_seed,
            capture_debug=True,
        )
        # A newly spawned CARLA sensor can skip its second spawn-time frame.
        # Advance twice but wait for the first frame: this proves that one
        # complete scan exists without deadlocking on a frame CARLA omitted.
        startup_frame = world.tick()
        _update_spectator_camera(spectator, ego)
        world.tick()
        _wait_for_radar_frame(
            radar,
            startup_frame,
            timeout_s=args.radar_timeout,
        )
        controlled_plan = _configure_controlled_target(
            radar,
            world,
            controlled_target,
        )
        controlled_step = _validate_controlled_target(
            radar,
            world,
            controlled_plan,
            args.fps,
            args.radar_timeout,
        )
        warmup_frames = int(round(args.warmup * args.fps))
        capture_frames = int(round(args.duration * args.fps))
        for _ in range(warmup_frames):
            _update_controlled_target(
                controlled_plan,
                controlled_step,
                args.fps,
            )
            controlled_step += 1
            _update_spectator_camera(spectator, ego)
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
            _update_spectator_camera(spectator, ego)
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
            for detection_index, detection in enumerate(detections):
                rows.append(
                    _detection_row(
                        sequence_index,
                        radar_frame,
                        timestamp,
                        detection_index,
                        detection,
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


def main():
    args = parse_args()
    if (
        args.sequences < 1
        or args.duration <= 0.0
        or args.fps < 1
        or args.radar_timeout <= 0.0
        or args.lead_distance <= 0.0
    ):
        raise ValueError(
            "sequences, duration, fps, radar-timeout, and lead-distance "
            "must be positive"
        )
    carla = _carla_module()
    client = carla.Client(args.host, args.port)
    client.set_timeout(30.0)
    output = Path(args.output) / args.split
    output.mkdir(parents=True, exist_ok=True)
    collection_summary = []
    for sequence_index in range(args.sequences):
        print(f"Collecting sequence {sequence_index + 1}/{args.sequences}")
        radar_data, diagnostics = collect_sequence(client, args, sequence_index)
        path = output / (
            f"scenario-{args.town.lower()}-synthetic-{sequence_index:03d}_sequence-01_"
            f"car_{args.split}.h5"
        )
        with h5py.File(path, "w") as handle:
            handle.create_dataset(
                "radar",
                data=radar_data,
                compression="gzip",
                shuffle=True,
            )
            handle.attrs["generator"] = "CARLA 0.9.16 geometry_multipath_v1"
            handle.attrs["town"] = args.town
            handle.attrs["seed"] = args.seed + sequence_index * 1009
            handle.attrs["arguments"] = json.dumps(
                vars(args),
                sort_keys=True,
            )
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
        collection_summary.append(counts)
        print(json.dumps(counts, sort_keys=True))
    with (
        Path(args.output)
        / f"collection_{args.town.lower()}_{args.split}.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(collection_summary, handle, indent=2, sort_keys=True)
        handle.write("\n")


if __name__ == "__main__":
    main()
