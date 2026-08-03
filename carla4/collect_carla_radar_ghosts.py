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
    parser.add_argument("--range", dest="range_m", type=float, default=100.0)
    parser.add_argument("--points-per-second", type=int, default=240000)
    parser.add_argument(
        "--radar-timeout",
        type=float,
        default=30.0,
        help="seconds to wait for each processed semantic-LiDAR radar frame",
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


def _spawn_vehicles(client, world, traffic_manager, count, seed):
    carla = _carla_module()
    rng = random.Random(seed)
    blueprints = [
        blueprint
        for blueprint in world.get_blueprint_library().filter("vehicle.*")
        if not blueprint.id.endswith(("isetta", "carlacola", "cybertruck", "t2"))
    ]
    spawn_points = list(world.get_map().get_spawn_points())
    rng.shuffle(spawn_points)
    if not blueprints or not spawn_points:
        raise RuntimeError("Town has no usable vehicle blueprints/spawn points")

    ego_blueprint = rng.choice(blueprints)
    if ego_blueprint.has_attribute("role_name"):
        ego_blueprint.set_attribute("role_name", "hero")
    ego = None
    for transform in spawn_points:
        ego = world.try_spawn_actor(ego_blueprint, transform)
        if ego is not None:
            break
    if ego is None:
        raise RuntimeError("Unable to spawn ego vehicle")
    ego.set_autopilot(True, traffic_manager.get_port())

    batch = []
    for transform in spawn_points[1 : count + 1]:
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
    actor_ids = [response.actor_id for response in responses if not response.error]
    return ego, actor_ids


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


def collect_sequence(client, args, sequence_index):
    carla = _carla_module()
    world = client.load_world(args.town)
    original_settings = world.get_settings()
    traffic_manager = client.get_trafficmanager(args.traffic_manager_port)
    radar = None
    ego = None
    npc_ids = []
    walker_ids = []
    walker_controller_ids = []
    rows = []
    diagnostics_summary = {}
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
        ego, npc_ids = _spawn_vehicles(
            client,
            world,
            traffic_manager,
            args.vehicles,
            sequence_seed,
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
            profile_name="geometry_multipath_v1",
            config_path=args.radar_config,
            seed=sequence_seed,
            capture_debug=True,
        )
        # A newly spawned CARLA sensor is not guaranteed to emit on its first
        # simulation tick. Advance two startup ticks, then require at least the
        # first one to have been processed before entering the frame-by-frame
        # collection loop. This avoids deadlocking on a skipped spawn tick.
        startup_frame = world.tick()
        world.tick()
        _wait_for_radar_frame(
            radar,
            startup_frame,
            timeout_s=args.radar_timeout,
        )
        warmup_frames = int(round(args.warmup * args.fps))
        capture_frames = int(round(args.duration * args.fps))
        for _ in range(warmup_frames):
            frame = world.tick()
            _wait_for_radar_frame(
                radar,
                frame,
                timeout_s=args.radar_timeout,
            )
        for _ in range(capture_frames):
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
                    diagnostics.get("reflector_count", 0)
                ),
                "last_multipath_count": int(
                    diagnostics.get("multipath_ideal_target_count", 0)
                ),
                "radar_profile": diagnostics.get("profile"),
                "radar_config_signature": diagnostics.get(
                    "config_signature"
                ),
                "weather_index": weather_index,
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
    ):
        raise ValueError(
            "sequences, duration, fps, and radar-timeout must be positive"
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
