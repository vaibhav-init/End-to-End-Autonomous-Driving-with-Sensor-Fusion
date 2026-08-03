"""CARLA-native, C-Shenron, and temporal realistic radar adapters."""

from dataclasses import asdict, replace
import math
import os
import threading

import numpy as np

from .cshenron_core import (
    CShenronConfig,
    decode_semantic_lidar,
    extract_targets,
    semantic_tag_name,
)
from .realistic_core import (
    DEFAULT_REALISTIC_RADAR_PROFILE,
    REALISTIC_RADAR_PROFILES,
    IdealRadarTarget,
    RadarEnvironment,
    RealisticRadarModel,
    load_realistic_radar_config,
    realistic_radar_config_dict,
    realistic_radar_config_signature,
)
from .multipath import extract_reflector_segments, generate_multipath_targets


RADAR_BACKENDS = ("native", "cshenron", "realistic")
_BACKEND_ALIASES = {
    "native": "native",
    "carla": "native",
    "cshenron": "cshenron",
    "c-shenron": "cshenron",
    "shenron": "cshenron",
    "realistic": "realistic",
    "temporal": "realistic",
    "phenomenological": "realistic",
}
_DYNAMIC_ACTOR_TAGS = frozenset((12, 13, 14, 15, 16, 17, 18, 19, 21))


def normalize_radar_backend(backend=None):
    """Resolve a backend name, using CARLA_RADAR_BACKEND when omitted."""

    value = backend or os.environ.get("CARLA_RADAR_BACKEND", "native")
    normalized = _BACKEND_ALIASES.get(str(value).strip().lower())
    if normalized is None:
        raise ValueError(
            f"Unknown radar backend '{value}'. Choose from {RADAR_BACKENDS}."
        )
    return normalized


def add_radar_arguments(parser):
    """Add shared radar backend and realism-profile options."""

    parser.add_argument(
        "--radar-backend",
        choices=RADAR_BACKENDS,
        default=normalize_radar_backend(),
        help=(
            "forward radar implementation (default: CARLA_RADAR_BACKEND or native)"
        ),
    )
    parser.add_argument(
        "--radar-profile",
        choices=REALISTIC_RADAR_PROFILES,
        default=os.environ.get("CARLA_RADAR_PROFILE"),
        help=(
            "built-in profile for --radar-backend realistic "
            f"(default: {DEFAULT_REALISTIC_RADAR_PROFILE})"
        ),
    )
    parser.add_argument(
        "--radar-config",
        default=os.environ.get("CARLA_RADAR_CONFIG"),
        help="optional JSON overrides for --radar-backend realistic",
    )
    parser.add_argument(
        "--radar-seed",
        type=int,
        default=None,
        help="sensor-error RNG seed (default: collection/scenario seed or 42)",
    )
    parser.add_argument(
        "--radar-ghost-detector",
        default=os.environ.get("CARLA_RADAR_GHOST_DETECTOR"),
        help="optional trained real-vs-multipath detector checkpoint",
    )
    parser.add_argument(
        "--radar-ghost-threshold",
        type=float,
        default=(
            float(os.environ["CARLA_RADAR_GHOST_THRESHOLD"])
            if os.environ.get("CARLA_RADAR_GHOST_THRESHOLD")
            else None
        ),
        help="override the detector checkpoint's rejection threshold",
    )
    parser.add_argument(
        "--radar-ghost-device",
        default=os.environ.get("CARLA_RADAR_GHOST_DEVICE", "cpu"),
        help="PyTorch device for online ghost filtering (default: cpu)",
    )
    return parser


def _carla_module():
    # Keep module import dependency-light so the NumPy core can be unit tested
    # on machines that do not have the CARLA Python egg installed.
    import carla

    return carla


def _empty_state(range_m):
    return {
        "distance": float(range_m),
        "relative_velocity": 0.0,
        "obstacle_speed": 0.0,
    }


def resolve_realistic_radar_config(
    range_m,
    fps=20,
    profile_name=None,
    config_path=None,
    config=None,
):
    """Resolve the exact realistic configuration used at runtime."""

    return load_realistic_radar_config(
        profile_name=profile_name,
        config_path=config_path,
        config=config,
        max_range_m=float(range_m),
        cycle_time_s=1.0 / max(float(fps), 1.0),
    )


def describe_radar_configuration(
    backend,
    range_m,
    fps=20,
    points_per_second=None,
    profile_name=None,
    config_path=None,
    config=None,
    ghost_detector_path=None,
    ghost_threshold=None,
):
    """Return serializable metadata for dataset/model compatibility checks."""

    backend = normalize_radar_backend(backend)
    metadata = {
        "radar_backend": backend,
        "radar_range_m": float(range_m),
    }
    if points_per_second is not None:
        metadata["radar_points_per_second"] = int(points_per_second)
    if backend == "realistic":
        resolved = resolve_realistic_radar_config(
            range_m=range_m,
            fps=fps,
            profile_name=profile_name,
            config_path=config_path,
            config=config,
        )
        metadata.update(
            {
                "radar_profile": resolved.profile_name,
                "radar_config_signature": realistic_radar_config_signature(
                    resolved
                ),
                "radar_config": realistic_radar_config_dict(resolved),
            }
        )
        if ghost_detector_path:
            from .ghost_detection.runtime import checkpoint_metadata

            detector = checkpoint_metadata(
                ghost_detector_path,
                threshold=ghost_threshold,
            )
            metadata.update(
                {
                    "radar_ghost_detector": os.path.basename(
                        ghost_detector_path
                    ),
                    "radar_ghost_detector_signature": detector["signature"],
                    "radar_ghost_threshold": detector["threshold"],
                    "radar_ghost_model": detector["model_name"],
                    "radar_ghost_feature_schema": detector["feature_schema"],
                    "radar_ghost_window_frames": detector["window_frames"],
                    "radar_ghost_max_points": detector["max_points"],
                }
            )
    return metadata


_LOGGED_DIAGNOSTIC_KEYS = (
    "backend",
    "profile",
    "config_signature",
    "frame",
    "timestamp",
    "scan_index",
    "ideal_target_count",
    "multipath_mode",
    "reflector_count",
    "multipath_ideal_target_count",
    "generated_detection_count",
    "accepted_detection_count",
    "rejected_detection_count",
    "delivered_detection_count",
    "delivered_source_scan_index",
    "configured_latency_scans",
    "direct_detection_count",
    "dropped_direct_count",
    "ghost_detection_count",
    "clutter_detection_count",
    "interference_active",
    "active_ghost_count",
    "active_track_count",
    "confirmed_track_count",
    "selected_track_id",
    "selected_truth_object_id",
    "selected_semantic_tag",
    "selected_source",
    "selected_truth_parent_object_id",
    "selected_reflector_id",
    "selected_bounce_type",
    "selected_bounce_order",
    "selected_path_length_m",
    "selected_ghost_probability",
    "selected_confidence",
    "selected_azimuth_deg",
    "selected_lateral_extent_m",
    "ghost_detector_signature",
    "ghost_detector_threshold",
    "path_curvature_per_m",
    "last_error",
)


def radar_diagnostics_row(radar):
    """Flatten stable scalar diagnostics for CSV logging."""

    diagnostics = radar.diagnostics()
    return {
        f"radar_{key}": diagnostics.get(key, "")
        for key in _LOGGED_DIAGNOSTIC_KEYS
    }


class NativeFrontRadar:
    """The original narrow CARLA radar with the existing scalar contract."""

    backend = "native"

    def __init__(
        self,
        vehicle,
        world,
        range_m=50.0,
        points_per_second=3000,
        capture_debug=False,
        **_ignored,
    ):
        self.latest = _empty_state(range_m)
        self._ego_speed = 0.0
        self._range = float(range_m)
        self._lock = threading.Lock()
        self._frame = -1
        self._timestamp = None
        self._raw_detection_count = 0
        self._candidate_count = 0
        self._selected_detection = None
        self._detection_snapshot = []
        self._capture_debug = bool(capture_debug)

        carla = _carla_module()
        blueprint = world.get_blueprint_library().find("sensor.other.radar")
        blueprint.set_attribute("horizontal_fov", "10")
        blueprint.set_attribute("vertical_fov", "2")
        blueprint.set_attribute("range", str(range_m))
        blueprint.set_attribute("points_per_second", str(points_per_second))
        transform = carla.Transform(
            carla.Location(x=2.5, z=1.0),
            carla.Rotation(pitch=2.0),
        )
        self.sensor = world.spawn_actor(blueprint, transform, attach_to=vehicle)
        self.sensor.listen(self._on_radar)

    def _on_radar(self, data):
        nearest_dist = self._range
        nearest_vel = 0.0
        candidate_count = 0
        selected_detection = None
        detections = []
        for detection in data:
            accepted = not (
                abs(detection.azimuth) > 0.3
                or detection.depth < 1.0
                or detection.altitude < -0.02
            )
            detection_record = {
                "distance_m": float(detection.depth),
                "azimuth_rad": float(detection.azimuth),
                "altitude_rad": float(detection.altitude),
                "carla_velocity_mps": float(detection.velocity),
                "closing_velocity_mps": -float(detection.velocity),
                "passes_adapter_filter": accepted,
            }
            if self._capture_debug:
                detections.append(detection_record)
            if not accepted:
                continue
            candidate_count += 1
            if detection.depth < nearest_dist:
                nearest_dist = detection.depth
                nearest_vel = detection.velocity
                selected_detection = detection_record.copy()

        relative_velocity = -nearest_vel
        with self._lock:
            ego_speed = self._ego_speed
            self._frame = int(data.frame)
            timestamp = getattr(data, "timestamp", None)
            self._timestamp = (
                float(timestamp) if timestamp is not None else None
            )
            self._raw_detection_count = len(data)
            self._candidate_count = candidate_count
            self._selected_detection = selected_detection
            self._detection_snapshot = detections
            self.latest = {
                "distance": nearest_dist,
                "relative_velocity": relative_velocity,
                "obstacle_speed": max(0.0, ego_speed - relative_velocity),
            }

    def update_ego_speed(self, speed):
        with self._lock:
            self._ego_speed = float(speed)

    def get(self):
        with self._lock:
            return self.latest.copy()

    def diagnostics(self):
        with self._lock:
            return {
                "backend": self.backend,
                "frame": self._frame,
                "timestamp": self._timestamp,
                "raw_detection_count": self._raw_detection_count,
                "candidate_count": self._candidate_count,
                "selected_azimuth_deg": (
                    math.degrees(
                        self._selected_detection["azimuth_rad"]
                    )
                    if self._selected_detection
                    else None
                ),
            }

    def debug_snapshot(self):
        with self._lock:
            return {
                "detections": [
                    detection.copy()
                    for detection in self._detection_snapshot
                ],
                "selected_detection": (
                    self._selected_detection.copy()
                    if self._selected_detection
                    else None
                ),
            }

    def cleanup(self):
        sensor = self.sensor
        self.sensor = None
        if sensor and sensor.is_alive:
            try:
                sensor.stop()
                sensor.destroy()
            except RuntimeError:
                pass


class CShenronFrontRadar:
    """C-Shenron material-aware target-list adapter for CARLA 0.9.16.

    A stock semantic LiDAR supplies geometry, incidence cosine, semantic tag,
    and actor ID. The NumPy core applies C-Shenron-derived material scattering
    and signal gating; this adapter adds actor-relative radial velocity and
    exposes the repository's existing three-scalar radar interface.
    """

    backend = "cshenron"

    def __init__(
        self,
        vehicle,
        world,
        range_m=50.0,
        fps=20,
        points_per_second=240000,
        seed=42,
        config=None,
        capture_debug=False,
        **_ignored,
    ):
        self.vehicle = vehicle
        self.world = world
        self._range = float(range_m)
        self._ego_speed = 0.0
        self._seed = int(seed)
        self._lock = threading.Lock()
        self._target = None
        self._frame = -1
        self._target_count = 0
        self._targets = ()
        self._timestamp = None
        self._raw_return_count = 0
        self._semantic_tag_counts = {}
        self._capture_debug = bool(capture_debug)
        self._last_error = None
        self._reported_error = False
        self.config = replace(
            config or CShenronConfig(),
            max_range_m=float(range_m),
        )

        carla = _carla_module()
        blueprint = world.get_blueprint_library().find(
            "sensor.lidar.ray_cast_semantic"
        )
        attributes = {
            "channels": "32",
            "range": str(range_m),
            "points_per_second": str(points_per_second),
            "rotation_frequency": str(float(fps)),
            "upper_fov": "10",
            "lower_fov": "-10",
            "horizontal_fov": "360",
            "sensor_tick": str(1.0 / max(float(fps), 1.0)),
        }
        for name, value in attributes.items():
            if blueprint.has_attribute(name):
                blueprint.set_attribute(name, value)

        transform = carla.Transform(carla.Location(x=2.5, z=1.0))
        self.sensor = world.spawn_actor(blueprint, transform, attach_to=vehicle)
        self.sensor.listen(self._on_semantic_lidar)

    def _on_semantic_lidar(self, measurement):
        try:
            returns = decode_semantic_lidar(measurement.raw_data)
            targets = extract_targets(returns, self.config)
            target = targets[0] if targets else None
            tag_counts = {}
            if self._capture_debug:
                tags, counts = np.unique(
                    returns["semantic_tag"],
                    return_counts=True,
                )
                tag_counts = {
                    int(tag): int(count)
                    for tag, count in zip(tags, counts)
                }
            with self._lock:
                self._target = target
                self._target_count = len(targets)
                self._targets = (
                    tuple(targets) if self._capture_debug else ()
                )
                self._frame = int(measurement.frame)
                timestamp = getattr(measurement, "timestamp", None)
                self._timestamp = (
                    float(timestamp) if timestamp is not None else None
                )
                self._raw_return_count = len(returns)
                self._semantic_tag_counts = tag_counts
                self._last_error = None
        except Exception as exc:
            with self._lock:
                self._last_error = f"{type(exc).__name__}: {exc}"
                should_report = not self._reported_error
                self._reported_error = True
            if should_report:
                print(f"  [radar:cshenron] callback failed: {exc}")

    @staticmethod
    def _vector_components(vector):
        return np.array((vector.x, vector.y, vector.z), dtype=np.float64)

    def _target_closing_speed(self, target, ego_velocity=None, ego_yaw_rad=None):
        if ego_velocity is None:
            ego_velocity = self._vector_components(self.vehicle.get_velocity())
        obstacle_velocity = np.zeros(3, dtype=np.float64)
        if ego_yaw_rad is None:
            ego_yaw_rad = math.radians(
                self.vehicle.get_transform().rotation.yaw
            )
        local_x, local_y, local_z = target.direction
        direction = np.array(
            (
                math.cos(ego_yaw_rad) * local_x
                - math.sin(ego_yaw_rad) * local_y,
                math.sin(ego_yaw_rad) * local_x
                + math.cos(ego_yaw_rad) * local_y,
                local_z,
            ),
            dtype=np.float64,
        )
        direction /= max(np.linalg.norm(direction), 1.0e-9)

        actor = None
        if (
            target.object_id > 0
            and target.semantic_tag in _DYNAMIC_ACTOR_TAGS
        ):
            try:
                actor = self.world.get_actor(target.object_id)
            except RuntimeError:
                actor = None

        if actor is not None and actor.is_alive:
            try:
                obstacle_velocity = self._vector_components(actor.get_velocity())
            except RuntimeError:
                actor = None

        radial_relative_velocity = float(
            np.dot(obstacle_velocity - ego_velocity, direction)
        )
        return -radial_relative_velocity

    def update_ego_speed(self, speed):
        with self._lock:
            self._ego_speed = float(speed)

    def get(self):
        with self._lock:
            target = self._target
            frame = self._frame
            ego_speed = self._ego_speed

        if target is None:
            return _empty_state(self._range)

        closing_speed = self._target_closing_speed(target)
        noise_seed = (
            self._seed * 1000003
            + frame * 9176
            + target.object_id * 37
            + target.semantic_tag
        ) & 0xFFFFFFFF
        rng = np.random.default_rng(noise_seed)
        noisy_range = target.distance_m + float(rng.normal(0.0, 0.15))
        range_resolution = self.config.speed_of_light_mps / (2.0 * 0.256e9)
        distance = round(noisy_range / range_resolution) * range_resolution
        distance = min(self._range, max(1.0, distance))
        relative_velocity = closing_speed + float(rng.normal(0.0, 0.15))
        return {
            "distance": float(distance),
            "relative_velocity": float(relative_velocity),
            "obstacle_speed": max(0.0, ego_speed - relative_velocity),
        }

    def diagnostics(self):
        with self._lock:
            target = self._target
            return {
                "backend": self.backend,
                "frame": self._frame,
                "timestamp": self._timestamp,
                "raw_return_count": self._raw_return_count,
                "target_count": self._target_count,
                "target_object_id": target.object_id if target else None,
                "target_semantic_tag": target.semantic_tag if target else None,
                "target_snr_db": target.snr_db if target else None,
                "last_error": self._last_error,
            }

    def debug_snapshot(self):
        with self._lock:
            return {
                "ideal_targets": [
                    asdict(target) for target in self._targets
                ],
                "semantic_tag_counts": {
                    str(tag): {
                        "name": semantic_tag_name(tag),
                        "count": count,
                    }
                    for tag, count in self._semantic_tag_counts.items()
                },
            }

    def cleanup(self):
        sensor = self.sensor
        self.sensor = None
        if sensor and sensor.is_alive:
            try:
                sensor.stop()
                sensor.destroy()
            except RuntimeError:
                pass


class RealisticFrontRadar(CShenronFrontRadar):
    """Material-aware target-list radar with temporal sensor imperfections.

    Semantic LiDAR supplies scene geometry and occlusion.  C-Shenron-derived
    material scattering supplies ideal target SNR.  ``RealisticRadarModel``
    then applies a configurable sensor/error model and target tracker before
    reducing the result to the repository's scalar longitudinal contract.
    """

    backend = "realistic"

    def __init__(
        self,
        vehicle,
        world,
        range_m=100.0,
        fps=20,
        points_per_second=240000,
        seed=42,
        profile_name=None,
        config_path=None,
        config=None,
        capture_debug=False,
        ghost_detector_path=None,
        ghost_threshold=None,
        ghost_device="cpu",
        **_ignored,
    ):
        self.vehicle = vehicle
        self.world = world
        self._range = float(range_m)
        self._ego_speed = 0.0
        self._lock = threading.Lock()
        self._frame = -1
        self._timestamp = None
        self._last_error = None
        self._reported_error = False
        self._ideal_target_count = 0
        self._reflector_count = 0
        self._state = _empty_state(range_m)
        self._model_snapshot = {}
        self._raw_return_count = 0
        self._semantic_tag_counts = {}
        self._reflector_snapshot = []
        self._capture_debug = bool(capture_debug)
        self._path_curvature_per_m = 0.0
        self.realistic_config = resolve_realistic_radar_config(
            range_m=range_m,
            fps=fps,
            profile_name=profile_name,
            config_path=config_path,
            config=config,
        )
        detection_filter = None
        if ghost_detector_path:
            from .ghost_detection.runtime import RuntimeGhostFilter

            detection_filter = RuntimeGhostFilter(
                ghost_detector_path,
                threshold=ghost_threshold,
                device=ghost_device,
            )
        self._detection_filter = detection_filter
        self.model = RealisticRadarModel(
            config=self.realistic_config,
            seed=seed,
            capture_debug=self._capture_debug,
            detection_filter=detection_filter,
        )

        # Let the temporal probability-of-detection model, rather than the
        # C-Shenron core's fixed threshold, decide whether a weak target exists.
        self.config = replace(
            CShenronConfig(),
            max_range_m=float(range_m),
            horizontal_fov_deg=self.realistic_config.horizontal_fov_deg,
            min_elevation_deg=self.realistic_config.min_elevation_deg,
            max_elevation_deg=self.realistic_config.max_elevation_deg,
            min_points_per_target=1,
            min_snr_db=-40.0,
        )

        carla = _carla_module()
        blueprint = world.get_blueprint_library().find(
            "sensor.lidar.ray_cast_semantic"
        )
        attributes = {
            "channels": "32",
            "range": str(range_m),
            "points_per_second": str(points_per_second),
            "rotation_frequency": str(float(fps)),
            "upper_fov": str(self.realistic_config.max_elevation_deg),
            "lower_fov": str(self.realistic_config.min_elevation_deg),
            # A complete sweep per callback prevents a rotating partial sector
            # from masquerading as radar dropout; the core applies radar FOV.
            "horizontal_fov": "360",
            "sensor_tick": str(1.0 / max(float(fps), 1.0)),
        }
        for name, value in attributes.items():
            if blueprint.has_attribute(name):
                blueprint.set_attribute(name, value)

        transform = carla.Transform(carla.Location(x=2.5, z=1.0))
        self.sensor = world.spawn_actor(
            blueprint,
            transform,
            attach_to=vehicle,
        )
        self.sensor.listen(self._on_semantic_lidar)

    def _read_environment(self):
        try:
            weather = self.world.get_weather()
        except (AttributeError, RuntimeError):
            return RadarEnvironment()

        def normalized(name):
            return float(np.clip(getattr(weather, name, 0.0) / 100.0, 0.0, 1.0))

        return RadarEnvironment(
            precipitation=normalized("precipitation"),
            wetness=normalized("wetness"),
            fog=normalized("fog_density"),
            dust=normalized("dust_storm"),
        )

    def _update_path_curvature(self, ego_velocity):
        """Estimate driven-path curvature from production-like ego motion.

        Automotive target selectors commonly consume yaw rate from the ego
        motion/IMU interface. CARLA exposes angular velocity in deg/s, so the
        planar curvature is yaw_rate / speed in 1/m.
        """

        ego_speed = float(np.linalg.norm(ego_velocity[:2]))
        if ego_speed < 1.0:
            raw_curvature = 0.0
        else:
            try:
                angular_velocity = self.vehicle.get_angular_velocity()
                raw_curvature = math.radians(
                    float(angular_velocity.z)
                ) / ego_speed
            except (AttributeError, RuntimeError):
                raw_curvature = 0.0
        raw_curvature = float(
            np.clip(
                raw_curvature,
                -self.realistic_config.max_abs_path_curvature_per_m,
                self.realistic_config.max_abs_path_curvature_per_m,
            )
        )
        gain = self.realistic_config.path_curvature_filter_gain
        self._path_curvature_per_m += gain * (
            raw_curvature - self._path_curvature_per_m
        )
        return self._path_curvature_per_m

    def _on_semantic_lidar(self, measurement):
        try:
            returns = decode_semantic_lidar(measurement.raw_data)
            targets = extract_targets(returns, self.config)
            tag_counts = {}
            if self._capture_debug:
                tags, counts = np.unique(
                    returns["semantic_tag"],
                    return_counts=True,
                )
                tag_counts = {
                    int(tag): int(count)
                    for tag, count in zip(tags, counts)
                }
            ego_velocity = self._vector_components(
                self.vehicle.get_velocity()
            )
            ego_yaw_rad = math.radians(
                self.vehicle.get_transform().rotation.yaw
            )
            path_curvature_per_m = self._update_path_curvature(ego_velocity)
            ideal_targets = []
            for target in targets:
                ideal_targets.append(
                    IdealRadarTarget(
                        object_id=target.object_id,
                        semantic_tag=target.semantic_tag,
                        distance_m=target.distance_m,
                        azimuth_rad=math.atan2(
                            target.direction[1],
                            target.direction[0],
                        ),
                        relative_velocity_mps=self._target_closing_speed(
                            target,
                            ego_velocity=ego_velocity,
                            ego_yaw_rad=ego_yaw_rad,
                        ),
                        snr_db=target.snr_db,
                        point_count=target.point_count,
                        lateral_extent_m=target.lateral_extent_m,
                        parent_object_id=target.object_id,
                        path_length_m=2.0 * target.distance_m,
                    )
                )
            reflectors = extract_reflector_segments(
                returns,
                self.realistic_config,
            )
            multipath_paths = generate_multipath_targets(
                ideal_targets,
                reflectors,
                self.realistic_config,
            )
            multipath_targets = [
                IdealRadarTarget(
                    object_id=path.object_id,
                    semantic_tag=path.semantic_tag,
                    distance_m=path.distance_m,
                    azimuth_rad=path.azimuth_rad,
                    relative_velocity_mps=path.relative_velocity_mps,
                    snr_db=path.snr_db,
                    lateral_extent_m=path.lateral_extent_m,
                    source="ghost",
                    parent_object_id=path.parent_object_id,
                    reflector_id=path.reflector_id,
                    bounce_type=path.bounce_type,
                    bounce_order=path.bounce_order,
                    path_length_m=path.path_length_m,
                )
                for path in multipath_paths
            ]
            timestamp = getattr(measurement, "timestamp", None)
            selected = self.model.step(
                ideal_targets,
                timestamp_s=timestamp,
                environment=self._read_environment(),
                path_curvature_per_m=path_curvature_per_m,
                multipath_targets=multipath_targets,
            )
            model_snapshot = self.model.debug_snapshot()
            state = {
                "distance": selected.distance_m,
                "relative_velocity": selected.relative_velocity_mps,
            }
            with self._lock:
                self._frame = int(measurement.frame)
                self._timestamp = (
                    float(timestamp) if timestamp is not None else None
                )
                self._ideal_target_count = len(ideal_targets)
                self._reflector_count = len(reflectors)
                self._raw_return_count = len(returns)
                self._semantic_tag_counts = tag_counts
                self._reflector_snapshot = (
                    [asdict(reflector) for reflector in reflectors]
                    if self._capture_debug
                    else []
                )
                self._model_snapshot = model_snapshot
                self._state = state
                self._last_error = None
        except Exception as exc:
            with self._lock:
                self._last_error = f"{type(exc).__name__}: {exc}"
                should_report = not self._reported_error
                self._reported_error = True
            if should_report:
                print(f"  [radar:realistic] callback failed: {exc}")

    def get(self):
        with self._lock:
            state = self._state.copy()
            ego_speed = self._ego_speed
        relative_velocity = state["relative_velocity"]
        return {
            "distance": float(state["distance"]),
            "relative_velocity": float(relative_velocity),
            "obstacle_speed": max(0.0, ego_speed - relative_velocity),
        }

    def diagnostics(self):
        model_diagnostics = self.model.diagnostics()
        with self._lock:
            model_diagnostics.update(
                {
                    "backend": self.backend,
                    "frame": self._frame,
                    "timestamp": self._timestamp,
                    "raw_return_count": self._raw_return_count,
                    "ideal_target_count": self._ideal_target_count,
                    "reflector_count": self._reflector_count,
                    "last_error": self._last_error,
                }
            )
            if self._detection_filter is not None:
                model_diagnostics["ghost_detector_signature"] = (
                    self._detection_filter.signature
                )
                model_diagnostics["ghost_detector_threshold"] = (
                    self._detection_filter.threshold
                )
        return model_diagnostics

    def debug_snapshot(self):
        with self._lock:
            result = self._model_snapshot.copy()
            for key in (
                "ideal_targets",
                "multipath_ideal_targets",
                "generated_detections",
                "accepted_detections",
                "rejected_detections",
                "delivered_detections",
                "tracks",
            ):
                result[key] = [
                    item.copy()
                    for item in self._model_snapshot.get(key, ())
                ]
            if "selected" in self._model_snapshot:
                result["selected"] = self._model_snapshot[
                    "selected"
                ].copy()
            result["semantic_tag_counts"] = {
                str(tag): {
                    "name": semantic_tag_name(tag),
                    "count": count,
                }
                for tag, count in self._semantic_tag_counts.items()
            }
            result["reflectors"] = [
                reflector.copy() for reflector in self._reflector_snapshot
            ]
            return result


def create_front_radar(
    vehicle,
    world,
    range_m=50.0,
    backend=None,
    fps=20,
    **kwargs,
):
    """Create a radar backend with a common get/update/cleanup contract."""

    backend = normalize_radar_backend(backend)
    if backend == "native":
        return NativeFrontRadar(vehicle, world, range_m=range_m, **kwargs)
    radar_class = (
        CShenronFrontRadar if backend == "cshenron" else RealisticFrontRadar
    )
    return radar_class(
        vehicle,
        world,
        range_m=range_m,
        fps=fps,
        **kwargs,
    )


# Backward-compatible symbol for code that imports FrontRadar directly.
FrontRadar = NativeFrontRadar
