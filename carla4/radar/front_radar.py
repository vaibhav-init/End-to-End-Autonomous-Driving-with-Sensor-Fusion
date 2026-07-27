"""CARLA-native and C-Shenron-compatible forward-radar adapters."""

from dataclasses import replace
import math
import os
import threading

import numpy as np

from .cshenron_core import CShenronConfig, decode_semantic_lidar, extract_targets


RADAR_BACKENDS = ("native", "cshenron")
_BACKEND_ALIASES = {
    "native": "native",
    "carla": "native",
    "cshenron": "cshenron",
    "c-shenron": "cshenron",
    "shenron": "cshenron",
}


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
    """Add the shared radar backend option to an argparse parser."""

    parser.add_argument(
        "--radar-backend",
        choices=RADAR_BACKENDS,
        default=normalize_radar_backend(),
        help=(
            "forward radar implementation (default: CARLA_RADAR_BACKEND or native)"
        ),
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


class NativeFrontRadar:
    """The original narrow CARLA radar with the existing scalar contract."""

    backend = "native"

    def __init__(self, vehicle, world, range_m=50.0, points_per_second=3000, **_ignored):
        self.latest = _empty_state(range_m)
        self._ego_speed = 0.0
        self._range = float(range_m)
        self._lock = threading.Lock()

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
        for detection in data:
            if (
                abs(detection.azimuth) > 0.3
                or detection.depth < 1.0
                or detection.altitude < -0.02
            ):
                continue
            if detection.depth < nearest_dist:
                nearest_dist = detection.depth
                nearest_vel = detection.velocity

        relative_velocity = -nearest_vel
        with self._lock:
            ego_speed = self._ego_speed
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
        return {"backend": self.backend}

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
            with self._lock:
                self._target = target
                self._target_count = len(targets)
                self._frame = int(measurement.frame)
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

    def _target_closing_speed(self, target):
        ego_velocity = self._vector_components(self.vehicle.get_velocity())
        obstacle_velocity = np.zeros(3, dtype=np.float64)
        direction = None

        actor = None
        if target.object_id > 0:
            try:
                actor = self.world.get_actor(target.object_id)
            except RuntimeError:
                actor = None

        if actor is not None and actor.is_alive:
            try:
                obstacle_velocity = self._vector_components(actor.get_velocity())
                actor_location = actor.get_location()
                ego_location = self.vehicle.get_location()
                delta = np.array(
                    (
                        actor_location.x - ego_location.x,
                        actor_location.y - ego_location.y,
                        actor_location.z - ego_location.z,
                    ),
                    dtype=np.float64,
                )
                norm = np.linalg.norm(delta)
                if norm > 1.0e-6:
                    direction = delta / norm
            except RuntimeError:
                actor = None

        if direction is None:
            yaw = math.radians(self.vehicle.get_transform().rotation.yaw)
            local_x, local_y, local_z = target.direction
            direction = np.array(
                (
                    math.cos(yaw) * local_x - math.sin(yaw) * local_y,
                    math.sin(yaw) * local_x + math.cos(yaw) * local_y,
                    local_z,
                ),
                dtype=np.float64,
            )
            direction /= max(np.linalg.norm(direction), 1.0e-9)

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
                "target_count": self._target_count,
                "target_object_id": target.object_id if target else None,
                "target_semantic_tag": target.semantic_tag if target else None,
                "target_snr_db": target.snr_db if target else None,
                "last_error": self._last_error,
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
    return CShenronFrontRadar(
        vehicle,
        world,
        range_m=range_m,
        fps=fps,
        **kwargs,
    )


# Backward-compatible symbol for code that imports FrontRadar directly.
FrontRadar = NativeFrontRadar
