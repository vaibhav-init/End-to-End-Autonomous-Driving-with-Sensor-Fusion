#!/usr/bin/env python3
"""Collect diverse real/ghost radar target lists during normal CARLA driving."""

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import random

import h5py
import numpy as np

from collect_carla_radar_ghosts import (
    RADAR_DTYPE,
    _carla_module,
    _configure_controlled_target,
    _detection_row,
    _spawn_walkers,
    _update_controlled_target,
    _update_spectator_camera,
    _wait_for_radar_frame,
)
from radar import create_front_radar


@dataclass(frozen=True)
class ScenePreset:
    name: str
    vehicle_scale: float
    walker_scale: float
    ego_speed_kmh: float
    npc_speed_difference_min: float
    npc_speed_difference_max: float
    following_distance_min_m: float
    following_distance_max_m: float
    lane_changes: bool


SCENE_PRESETS = {
    "urban_dense": ScenePreset(
        name="urban_dense",
        vehicle_scale=1.30,
        walker_scale=1.20,
        ego_speed_kmh=30.0,
        npc_speed_difference_min=5.0,
        npc_speed_difference_max=35.0,
        following_distance_min_m=2.5,
        following_distance_max_m=6.0,
        lane_changes=True,
    ),
    "highway_flow": ScenePreset(
        name="highway_flow",
        vehicle_scale=1.50,
        walker_scale=0.20,
        ego_speed_kmh=65.0,
        npc_speed_difference_min=-15.0,
        npc_speed_difference_max=15.0,
        following_distance_min_m=4.0,
        following_distance_max_m=10.0,
        lane_changes=True,
    ),
    "pedestrian_dense": ScenePreset(
        name="pedestrian_dense",
        vehicle_scale=0.70,
        walker_scale=2.00,
        ego_speed_kmh=25.0,
        npc_speed_difference_min=10.0,
        npc_speed_difference_max=40.0,
        following_distance_min_m=3.0,
        following_distance_max_m=7.0,
        lane_changes=False,
    ),
    "mixed_weather": ScenePreset(
        name="mixed_weather",
        vehicle_scale=1.00,
        walker_scale=1.00,
        ego_speed_kmh=40.0,
        npc_speed_difference_min=-5.0,
        npc_speed_difference_max=30.0,
        following_distance_min_m=3.0,
        following_distance_max_m=8.0,
        lane_changes=True,
    ),
}
DIVERSE_SCENE_NAMES = tuple(SCENE_PRESETS)
DYNAMIC_TAGS = frozenset((12, 13, 14, 15, 16, 17, 18, 19, 21))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--traffic-manager-port", type=int, default=8000)
    parser.add_argument(
        "--towns",
        nargs="+",
        default=("Town03", "Town04"),
        help="towns rotated across sequences",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="train")
    parser.add_argument("--sequences", type=int, default=8)
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--warmup", type=float, default=3.0)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument(
        "--vehicles",
        type=int,
        default=45,
        help="base NPC count before applying each scene's density scale",
    )
    parser.add_argument(
        "--walkers",
        type=int,
        default=25,
        help="base pedestrian count before applying each scene's density scale",
    )
    parser.add_argument(
        "--scenes",
        nargs="+",
        choices=tuple(SCENE_PRESETS) + ("diverse",),
        default=("diverse",),
        help="scene presets to rotate; diverse enables all presets",
    )
    parser.add_argument("--range", dest="range_m", type=float, default=100.0)
    parser.add_argument("--points-per-second", type=int, default=240000)
    parser.add_argument("--radar-timeout", type=float, default=30.0)
    parser.add_argument("--event-duration", type=float, default=8.0)
    parser.add_argument("--event-cooldown", type=float, default=4.0)
    parser.add_argument("--event-retry", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--radar-config", help="optional geometry-profile overrides")
    parser.add_argument(
        "--camera-view",
        choices=("chase", "target"),
        default="chase",
    )
    parser.add_argument("--headless", action="store_true")
    return parser.parse_args()


def _scene_names(requested):
    if "diverse" in requested:
        if len(requested) != 1:
            raise ValueError("Use --scenes diverse alone or list explicit scenes")
        return DIVERSE_SCENE_NAMES
    return tuple(dict.fromkeys(requested))


def _weather_for_sequence(world, sequence_index, scene_name):
    carla = _carla_module()
    weather_presets = (
        ("clear_noon", carla.WeatherParameters.ClearNoon),
        ("cloudy_noon", carla.WeatherParameters.CloudyNoon),
        ("wet_noon", carla.WeatherParameters.WetNoon),
        ("mid_rain_noon", carla.WeatherParameters.MidRainyNoon),
        ("soft_rain_sunset", carla.WeatherParameters.SoftRainSunset),
    )
    scene_offset = DIVERSE_SCENE_NAMES.index(scene_name)
    name, weather = weather_presets[
        (sequence_index + scene_offset) % len(weather_presets)
    ]
    world.set_weather(weather)
    return name


def _four_wheel_blueprints(world):
    result = []
    for blueprint in world.get_blueprint_library().filter("vehicle.*"):
        if blueprint.id.endswith(("isetta", "carlacola", "cybertruck", "t2")):
            continue
        if not blueprint.has_attribute("number_of_wheels"):
            continue
        try:
            if blueprint.get_attribute("number_of_wheels").as_int() == 4:
                result.append(blueprint)
        except (RuntimeError, ValueError):
            continue
    return result


def _set_desired_speed(traffic_manager, actor, speed_kmh):
    try:
        traffic_manager.set_desired_speed(actor, float(speed_kmh))
        return
    except (AttributeError, RuntimeError):
        pass
    speed_limit = max(float(actor.get_speed_limit()), 20.0)
    percentage = 100.0 * (1.0 - float(speed_kmh) / speed_limit)
    traffic_manager.vehicle_percentage_speed_difference(
        actor,
        max(-50.0, min(90.0, percentage)),
    )


def _spawn_driving_scene(
    client,
    world,
    traffic_manager,
    preset,
    vehicle_count,
    seed,
):
    carla = _carla_module()
    rng = random.Random(seed)
    blueprints = _four_wheel_blueprints(world)
    spawn_points = list(world.get_map().get_spawn_points())
    rng.shuffle(spawn_points)
    if not blueprints or len(spawn_points) < 2:
        raise RuntimeError("Town has insufficient four-wheel vehicles/spawn points")

    used_indices = set()

    def spawn_named_actor(role_name):
        for index, transform in enumerate(spawn_points):
            if index in used_indices:
                continue
            blueprint = rng.choice(blueprints)
            if blueprint.has_attribute("role_name"):
                blueprint.set_attribute("role_name", role_name)
            actor = world.try_spawn_actor(blueprint, transform)
            if actor is not None:
                used_indices.add(index)
                return actor
        return None

    ego = spawn_named_actor("hero")
    event_target = spawn_named_actor("multipath_event_target")
    if ego is None or event_target is None:
        for actor in (ego, event_target):
            if actor is not None and actor.is_alive:
                actor.destroy()
        raise RuntimeError("Unable to spawn ego and multipath event vehicle")

    port = traffic_manager.get_port()
    ego.set_autopilot(True, port)
    _set_desired_speed(traffic_manager, ego, preset.ego_speed_kmh)
    traffic_manager.distance_to_leading_vehicle(ego, 6.0)
    traffic_manager.ignore_lights_percentage(ego, 0.0)
    traffic_manager.ignore_signs_percentage(ego, 0.0)
    traffic_manager.ignore_walkers_percentage(ego, 0.0)
    traffic_manager.auto_lane_change(ego, preset.lane_changes)

    # The event target is kinematic only while it traverses a reflector. All
    # other actors, including ego, remain normal Traffic Manager participants.
    event_target.set_simulate_physics(False)
    event_target.set_target_velocity(carla.Vector3D())

    batch = []
    for index, transform in enumerate(spawn_points):
        if index in used_indices or len(batch) >= vehicle_count:
            continue
        blueprint = rng.choice(blueprints)
        if blueprint.has_attribute("role_name"):
            blueprint.set_attribute("role_name", "autopilot")
        batch.append(
            carla.command.SpawnActor(blueprint, transform).then(
                carla.command.SetAutopilot(
                    carla.command.FutureActor,
                    True,
                    port,
                )
            )
        )
    responses = client.apply_batch_sync(batch, True)
    npc_ids = [response.actor_id for response in responses if not response.error]
    for actor_id in npc_ids:
        actor = world.get_actor(actor_id)
        if actor is None:
            continue
        speed_difference = rng.uniform(
            preset.npc_speed_difference_min,
            preset.npc_speed_difference_max,
        )
        traffic_manager.vehicle_percentage_speed_difference(
            actor,
            speed_difference,
        )
        traffic_manager.distance_to_leading_vehicle(
            actor,
            rng.uniform(
                preset.following_distance_min_m,
                preset.following_distance_max_m,
            ),
        )
        traffic_manager.ignore_lights_percentage(actor, 0.0)
        traffic_manager.ignore_signs_percentage(actor, 0.0)
        traffic_manager.ignore_walkers_percentage(actor, 0.0)
        traffic_manager.auto_lane_change(actor, preset.lane_changes)
    return ego, event_target, [event_target.id] + npc_ids


def _park_event_target(target, ego):
    carla = _carla_module()
    ego_transform = ego.get_transform()
    location = (
        ego_transform.location
        - ego_transform.get_forward_vector() * 35.0
        + carla.Location(z=50.0)
    )
    target.set_transform(
        carla.Transform(
            location,
            carla.Rotation(yaw=float(ego_transform.rotation.yaw)),
        )
    )
    target.set_target_velocity(carla.Vector3D())


def _dynamic_target_count(ideal_targets):
    return sum(
        int(target.get("semantic_tag", -1)) in DYNAMIC_TAGS
        for target in ideal_targets
    )


def _event_record(plan, frame):
    return {
        "start_frame": int(frame),
        "target_range_m": float(plan["base_target_range_m"]),
        "surface_distance_m": float(plan["target_surface_distance_m"]),
        "motion_amplitude_m": float(plan["motion_amplitude_m"]),
        "reflector_id": int(plan["reflector_id"]),
        "reflector_tag": int(plan["reflector_tag"]),
        "reflector_length_m": float(plan["reflector_length_m"]),
        "expected_path_families": list(plan["expected_path_families"]),
    }


def collect_sequence(client, args, sequence_index, town, preset):
    carla = _carla_module()
    world = client.load_world(town)
    original_settings = world.get_settings()
    traffic_manager = client.get_trafficmanager(args.traffic_manager_port)
    sequence_seed = args.seed + sequence_index * 1009
    vehicle_count = max(0, int(round(args.vehicles * preset.vehicle_scale)))
    walker_count = max(0, int(round(args.walkers * preset.walker_scale)))
    radar = None
    ego = None
    event_target = None
    actor_ids = []
    walker_ids = []
    controller_ids = []
    spectator = None
    rows = []
    events = []
    event_failures = []
    aggregate = {
        "capture_frames": 0,
        "frames_with_dynamic_targets": 0,
        "frames_with_multipath": 0,
        "max_dynamic_ideal_targets": 0,
        "max_multipath_count": 0,
        "sum_dynamic_ideal_targets": 0,
        "sum_multipath_count": 0,
        "sum_reflector_count": 0,
    }
    diagnostics_summary = {}
    try:
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 1.0 / args.fps
        settings.no_rendering_mode = False
        world.apply_settings(settings)
        traffic_manager.set_synchronous_mode(True)
        traffic_manager.set_random_device_seed(sequence_seed)
        try:
            traffic_manager.set_hybrid_physics_mode(True)
            traffic_manager.set_hybrid_physics_radius(max(args.range_m, 70.0))
        except (AttributeError, RuntimeError):
            pass
        weather_name = _weather_for_sequence(
            world,
            sequence_index,
            preset.name,
        )
        ego, event_target, actor_ids = _spawn_driving_scene(
            client,
            world,
            traffic_manager,
            preset,
            vehicle_count,
            sequence_seed,
        )
        _park_event_target(event_target, ego)
        walker_ids, controller_ids = _spawn_walkers(
            client,
            world,
            walker_count,
            sequence_seed + 17,
        )
        if not args.headless:
            spectator = world.get_spectator()
            _update_spectator_camera(
                spectator,
                ego,
                event_target,
                args.camera_view,
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
        startup_frame = world.tick()
        world.tick()
        _wait_for_radar_frame(
            radar,
            startup_frame,
            timeout_s=args.radar_timeout,
        )
        for _ in range(int(round(args.warmup * args.fps))):
            if spectator is not None:
                _update_spectator_camera(
                    spectator,
                    ego,
                    event_target,
                    args.camera_view,
                )
            frame = world.tick()
            _wait_for_radar_frame(
                radar,
                frame,
                timeout_s=args.radar_timeout,
            )

        event_plan = None
        event_local_step = 0
        event_end_step = -1
        next_event_step = 0
        event_duration_frames = max(1, int(round(args.event_duration * args.fps)))
        event_cooldown_frames = max(0, int(round(args.event_cooldown * args.fps)))
        event_retry_frames = max(1, int(round(args.event_retry * args.fps)))
        capture_frames = int(round(args.duration * args.fps))
        for capture_step in range(capture_frames):
            if event_plan is not None and capture_step >= event_end_step:
                _park_event_target(event_target, ego)
                event_plan = None
                next_event_step = capture_step + event_cooldown_frames

            if event_plan is None and capture_step >= next_event_step:
                try:
                    event_plan = _configure_controlled_target(
                        radar,
                        world,
                        event_target,
                    )
                    event_local_step = 0
                    event_end_step = capture_step + event_duration_frames
                    events.append(
                        _event_record(
                            event_plan,
                            radar.diagnostics().get("frame", -1),
                        )
                    )
                    print(
                        f"    event {len(events)}: "
                        f"range={event_plan['base_target_range_m']:.1f}m "
                        f"motion=+/-{event_plan['motion_amplitude_m']:.1f}m "
                        f"reflector_tag={event_plan['reflector_tag']}"
                    )
                except RuntimeError as exc:
                    event_failures.append(str(exc))
                    next_event_step = capture_step + event_retry_frames

            if event_plan is not None:
                _update_controlled_target(
                    event_plan,
                    event_local_step,
                    args.fps,
                )
                event_local_step += 1

            if spectator is not None:
                _update_spectator_camera(
                    spectator,
                    ego,
                    event_target,
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
            dynamic_count = _dynamic_target_count(ideal_targets)
            multipath_count = int(
                diagnostics.get("multipath_ideal_target_count", 0)
            )
            reflector_count = int(diagnostics.get("reflector_count", 0))
            aggregate["capture_frames"] += 1
            aggregate["frames_with_dynamic_targets"] += int(dynamic_count > 0)
            aggregate["frames_with_multipath"] += int(multipath_count > 0)
            aggregate["max_dynamic_ideal_targets"] = max(
                aggregate["max_dynamic_ideal_targets"],
                dynamic_count,
            )
            aggregate["max_multipath_count"] = max(
                aggregate["max_multipath_count"],
                multipath_count,
            )
            aggregate["sum_dynamic_ideal_targets"] += dynamic_count
            aggregate["sum_multipath_count"] += multipath_count
            aggregate["sum_reflector_count"] += reflector_count
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
                "town": town,
                "scene": preset.name,
                "weather": weather_name,
                "seed": sequence_seed,
                "vehicles_requested": vehicle_count,
                "vehicles_spawned": len(actor_ids) - 1,
                "walkers_requested": walker_count,
                "walkers_spawned": len(walker_ids),
                "ego_speed_kmh": math.sqrt(
                    ego.get_velocity().x ** 2
                    + ego.get_velocity().y ** 2
                    + ego.get_velocity().z ** 2
                )
                * 3.6,
                "multipath_events": len(events),
                "multipath_event_failures": len(event_failures),
                "radar_profile": diagnostics.get("profile"),
                "radar_config_signature": diagnostics.get(
                    "config_signature"
                ),
                **aggregate,
            }
    finally:
        if radar is not None:
            radar.cleanup()
        for controller_id in controller_ids:
            controller = world.get_actor(controller_id)
            if controller is not None:
                try:
                    controller.stop()
                except RuntimeError:
                    pass
        destroy_ids = list(actor_ids) + list(controller_ids) + list(walker_ids)
        if ego is not None:
            destroy_ids.append(ego.id)
        if destroy_ids:
            client.apply_batch(
                [carla.command.DestroyActor(actor_id) for actor_id in destroy_ids]
            )
        traffic_manager.set_synchronous_mode(False)
        world.apply_settings(original_settings)
    return (
        np.asarray(rows, dtype=RADAR_DTYPE),
        diagnostics_summary,
        events,
        event_failures,
    )


def _write_sequence(path, radar_data, diagnostics, events, failures, args):
    with h5py.File(path, "w") as handle:
        handle.create_dataset(
            "radar",
            data=radar_data,
            compression="gzip",
            shuffle=True,
        )
        handle.attrs["generator"] = "CARLA 0.9.16 diverse geometry_multipath_v1"
        handle.attrs["town"] = diagnostics["town"]
        handle.attrs["scene"] = diagnostics["scene"]
        handle.attrs["weather"] = diagnostics["weather"]
        handle.attrs["seed"] = diagnostics["seed"]
        handle.attrs["arguments"] = json.dumps(vars(args), sort_keys=True)
        handle.attrs["radar_config_signature"] = diagnostics.get(
            "radar_config_signature",
            "",
        )
        handle.attrs["multipath_events"] = json.dumps(events, sort_keys=True)
        handle.attrs["multipath_event_failures"] = json.dumps(failures)


def main():
    args = parse_args()
    if (
        args.sequences < 1
        or args.duration <= 0.0
        or args.fps < 1
        or args.vehicles < 0
        or args.walkers < 0
        or args.radar_timeout <= 0.0
        or args.event_duration <= 0.0
        or args.event_cooldown < 0.0
        or args.event_retry <= 0.0
    ):
        raise ValueError("Collection counts, durations, fps, and timeout are invalid")
    scene_names = _scene_names(args.scenes)
    carla = _carla_module()
    client = carla.Client(args.host, args.port)
    client.set_timeout(30.0)
    output = Path(args.output) / args.split
    output.mkdir(parents=True, exist_ok=True)
    summaries = []
    failed_sequences = []
    for sequence_index in range(args.sequences):
        scene_block = sequence_index // len(args.towns)
        scene_name = scene_names[scene_block % len(scene_names)]
        preset = SCENE_PRESETS[scene_name]
        town = args.towns[sequence_index % len(args.towns)]
        print(
            f"Collecting sequence {sequence_index + 1}/{args.sequences}: "
            f"scene={scene_name} town={town}"
        )
        try:
            radar_data, diagnostics, events, event_failures = collect_sequence(
                client,
                args,
                sequence_index,
                town,
                preset,
            )
            filename = (
                f"scenario-{sequence_index:04d}-{scene_name}-{town.lower()}-"
                f"{args.split}.h5"
            )
            path = output / filename
            _write_sequence(
                path,
                radar_data,
                diagnostics,
                events,
                event_failures,
                args,
            )
            counts = {
                "path": str(path),
                "points": int(len(radar_data)),
                "real": int(
                    np.count_nonzero(radar_data["label_id"] % 10 == 1)
                ),
                "ghost": int(
                    np.count_nonzero(
                        (radar_data["label_id"] > 0)
                        & (radar_data["label_id"] % 10 != 1)
                    )
                ),
                **diagnostics,
            }
            summaries.append(counts)
            print(json.dumps(counts, sort_keys=True))
        except Exception as exc:
            failure = {
                "sequence_index": sequence_index,
                "scene": scene_name,
                "town": town,
                "error": f"{type(exc).__name__}: {exc}",
            }
            failed_sequences.append(failure)
            print(json.dumps(failure, sort_keys=True))

    summary_path = Path(args.output) / f"collection_{args.split}.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "sequences": summaries,
                "failed_sequences": failed_sequences,
            },
            handle,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")
    if not summaries:
        raise RuntimeError("Every requested CARLA sequence failed")
    print(
        f"Ready: {len(summaries)} sequences, "
        f"{len(failed_sequences)} failures, summary={summary_path}"
    )


if __name__ == "__main__":
    main()
