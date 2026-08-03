"""Deterministic planar multipath geometry for automotive radar targets.

The implementation uses the image (mirror) method on locally planar semantic
LiDAR surfaces.  It deliberately returns path-level ground truth separately
from the observable target so CARLA IDs and reflector geometry never need to
become detector inputs.
"""

from dataclasses import dataclass
import hashlib
import math

import numpy as np


REFLECTOR_TAGS = frozenset((3, 4, 5, 20, 26, 28))
DYNAMIC_TARGET_TAGS = frozenset((12, 13, 14, 15, 16, 17, 18, 19, 21))


@dataclass(frozen=True)
class ReflectorSegment:
    """A locally planar vertical reflector in radar-sensor coordinates."""

    reflector_id: int
    semantic_tag: int
    point_xy_m: tuple
    tangent_xy: tuple
    normal_xy: tuple
    length_m: float
    rms_residual_m: float
    point_count: int
    reflection_loss_db: float


@dataclass(frozen=True)
class MultipathTarget:
    """One physically related virtual target before sensor imperfections."""

    object_id: int
    parent_object_id: int
    reflector_id: int
    semantic_tag: int
    distance_m: float
    azimuth_rad: float
    relative_velocity_mps: float
    snr_db: float
    path_length_m: float
    bounce_type: str
    bounce_order: int
    reflection_point_xy_m: tuple
    lateral_extent_m: float = 0.0


def _zigzag(value):
    value = int(value)
    return 2 * value if value >= 0 else -2 * value - 1


def _stable_negative_id(*parts):
    payload = ":".join(str(part) for part in parts).encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    value = int.from_bytes(digest, "big") & ((1 << 62) - 1)
    return -(10_000_000_000 + value)


def _reflection_loss_db(semantic_tag):
    # Conservative priors.  They are intentionally visible and are augmented
    # by configurable per-bounce loss in RealisticRadarConfig.
    return {
        5: 1.5,   # metal fence
        28: 1.0,  # guard rail
        4: 3.0,   # wall
        3: 4.0,   # building facade
        20: 4.5,  # generic static object
        26: 3.5,  # bridge
    }.get(int(semantic_tag), 5.0)


def extract_reflector_segments(returns, config):
    """Fit short 2-D reflector segments to semantic-LiDAR surface returns."""

    if returns.size == 0 or getattr(config, "multipath_mode", "off") != "geometry":
        return []

    x = returns["x"].astype(np.float64, copy=False)
    y = returns["y"].astype(np.float64, copy=False)
    z = returns["z"].astype(np.float64, copy=False)
    tags = returns["semantic_tag"].astype(np.int64, copy=False)
    ranges = np.hypot(x, y)
    cell_size = float(config.multipath_reflector_cell_size_m)
    mask = (
        np.isfinite(x)
        & np.isfinite(y)
        & np.isfinite(z)
        & np.isin(tags, tuple(REFLECTOR_TAGS))
        & (x > 0.5)
        & (ranges <= float(config.max_range_m))
        & (z >= float(config.multipath_reflector_min_height_m))
        & (z <= float(config.multipath_reflector_max_height_m))
    )
    indices = np.flatnonzero(mask)
    if indices.size == 0:
        return []

    selected_x = x[indices]
    selected_y = y[indices]
    selected_tags = tags[indices]
    grid_x = np.floor(selected_x / cell_size).astype(np.int64)
    grid_y = np.floor(selected_y / cell_size).astype(np.int64)
    order = np.lexsort((grid_y, grid_x, selected_tags))
    ordered_tags = selected_tags[order]
    ordered_grid_x = grid_x[order]
    ordered_grid_y = grid_y[order]
    changes = (
        (ordered_tags[1:] != ordered_tags[:-1])
        | (ordered_grid_x[1:] != ordered_grid_x[:-1])
        | (ordered_grid_y[1:] != ordered_grid_y[:-1])
    )
    starts = np.concatenate(
        (np.array((0,), dtype=np.int64), np.flatnonzero(changes) + 1)
    )
    ends = np.concatenate(
        (starts[1:], np.array((len(order),), dtype=np.int64))
    )

    reflectors = []
    minimum_points = int(config.multipath_reflector_min_points)
    for start, end in zip(starts, ends):
        if end - start < minimum_points:
            continue
        local = order[start:end]
        points = np.column_stack((selected_x[local], selected_y[local]))
        centroid = np.median(points, axis=0)
        centered = points - centroid
        covariance = centered.T @ centered / max(len(points) - 1, 1)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        tangent = eigenvectors[:, int(np.argmax(eigenvalues))]
        tangent /= max(float(np.linalg.norm(tangent)), 1.0e-12)
        normal = np.array((-tangent[1], tangent[0]), dtype=np.float64)
        # Give the normal a stable sign: it points from the surface toward the
        # radar whenever possible.
        if float(np.dot(normal, -centroid)) < 0.0:
            normal = -normal
            tangent = -tangent

        along = centered @ tangent
        residual = centered @ normal
        low, high = np.quantile(along, (0.02, 0.98))
        length = float(high - low)
        rms = float(np.sqrt(np.mean(np.square(residual))))
        if (
            length < float(config.multipath_reflector_min_length_m)
            or rms > float(config.multipath_reflector_max_residual_m)
        ):
            continue

        tag = int(ordered_tags[start])
        grid_cell_x = int(ordered_grid_x[start])
        grid_cell_y = int(ordered_grid_y[start])
        reflector_id = _stable_negative_id(
            "reflector",
            tag,
            _zigzag(grid_cell_x),
            _zigzag(grid_cell_y),
        )
        reflectors.append(
            ReflectorSegment(
                reflector_id=reflector_id,
                semantic_tag=tag,
                point_xy_m=(float(centroid[0]), float(centroid[1])),
                tangent_xy=(float(tangent[0]), float(tangent[1])),
                normal_xy=(float(normal[0]), float(normal[1])),
                length_m=length,
                rms_residual_m=rms,
                point_count=len(points),
                reflection_loss_db=_reflection_loss_db(tag),
            )
        )

    reflectors.sort(
        key=lambda item: math.hypot(item.point_xy_m[0], item.point_xy_m[1])
    )
    return reflectors[: int(config.multipath_max_reflectors)]


def _mirror_point(point, line_point, normal):
    signed_distance = float(np.dot(point - line_point, normal))
    return point - 2.0 * signed_distance * normal


def _specular_point(target_xy, reflector, config):
    line_point = np.asarray(reflector.point_xy_m, dtype=np.float64)
    tangent = np.asarray(reflector.tangent_xy, dtype=np.float64)
    normal = np.asarray(reflector.normal_xy, dtype=np.float64)
    target_mirror = _mirror_point(target_xy, line_point, normal)

    radar_side = float(np.dot(-line_point, normal))
    target_side = float(np.dot(target_xy - line_point, normal))
    if radar_side * target_side <= 0.0:
        return None

    denominator = float(np.dot(target_mirror, normal))
    if abs(denominator) <= 1.0e-9:
        return None
    fraction = float(np.dot(line_point, normal) / denominator)
    if not 0.0 < fraction < 1.0:
        return None
    point = fraction * target_mirror
    segment_offset = abs(float(np.dot(point - line_point, tangent)))
    if segment_offset > 0.5 * reflector.length_m + float(
        config.multipath_segment_margin_m
    ):
        return None

    incident = point / max(float(np.linalg.norm(point)), 1.0e-12)
    incidence_cosine = abs(float(np.dot(incident, normal)))
    if incidence_cosine < float(config.multipath_min_incidence_cosine):
        return None
    target_surface_distance = abs(target_side)
    if target_surface_distance > float(config.multipath_max_target_surface_distance_m):
        return None
    return point, target_mirror, incidence_cosine


def _path_snr_db(target, effective_range_m, reflector, bounce_order, config):
    direct_range = max(float(target.distance_m), 1.0)
    spreading_loss = 40.0 * math.log10(
        max(float(effective_range_m), 1.0) / direct_range
    )
    if bounce_order == 2:
        bounce_loss = float(config.multipath_second_order_loss_db)
        reflection_loss = reflector.reflection_loss_db
    else:
        bounce_loss = float(config.multipath_third_order_loss_db)
        reflection_loss = 2.0 * reflector.reflection_loss_db
    return float(target.snr_db) - spreading_loss - bounce_loss - reflection_loss


def _path_closing_speeds(target, target_xy, target_mirror, normal):
    target_unit = target_xy / max(float(np.linalg.norm(target_xy)), 1.0e-12)
    mirror_unit = target_mirror / max(
        float(np.linalg.norm(target_mirror)), 1.0e-12
    )
    # The source target exposes radial motion only.  Treating its unobserved
    # tangential velocity as zero is explicit and keeps the approximation
    # radar-observable rather than querying privileged CARLA actor velocity.
    target_velocity = -float(target.relative_velocity_mps) * target_unit
    mirror_velocity = target_velocity - 2.0 * normal * float(
        np.dot(target_velocity, normal)
    )
    direct_rate = float(np.dot(target_unit, target_velocity))
    mirror_rate = float(np.dot(mirror_unit, mirror_velocity))
    return -0.5 * (direct_rate + mirror_rate), -mirror_rate


def generate_multipath_targets(targets, reflectors, config):
    """Generate deterministic second/third-order ghost hypotheses."""

    if getattr(config, "multipath_mode", "off") != "geometry":
        return []

    half_fov = math.radians(float(config.horizontal_fov_deg) / 2.0)
    ghosts = []
    for target in targets:
        if int(target.semantic_tag) not in DYNAMIC_TARGET_TAGS:
            continue
        target_xy = np.array(
            (
                float(target.distance_m) * math.cos(float(target.azimuth_rad)),
                float(target.distance_m) * math.sin(float(target.azimuth_rad)),
            ),
            dtype=np.float64,
        )
        candidates = []
        for reflector in reflectors:
            result = _specular_point(target_xy, reflector, config)
            if result is None:
                continue
            reflection_point, target_mirror, _ = result
            mirror_distance = float(np.linalg.norm(target_mirror))
            direct_distance = float(np.linalg.norm(target_xy))
            second_range = 0.5 * (direct_distance + mirror_distance)
            third_range = mirror_distance
            if second_range - direct_distance < float(
                config.multipath_min_range_separation_m
            ):
                continue
            reflection_azimuth = math.atan2(
                float(reflection_point[1]),
                float(reflection_point[0]),
            )
            normal = np.asarray(reflector.normal_xy, dtype=np.float64)
            second_closing, third_closing = _path_closing_speeds(
                target,
                target_xy,
                target_mirror,
                normal,
            )

            path_specs = [
                (
                    "type1",
                    2,
                    second_range,
                    float(target.azimuth_rad),
                    second_closing,
                ),
                (
                    "type2",
                    2,
                    second_range,
                    reflection_azimuth,
                    second_closing,
                ),
            ]
            if bool(config.multipath_enable_third_order):
                path_specs.append(
                    (
                        "type2",
                        3,
                        third_range,
                        reflection_azimuth,
                        third_closing,
                    )
                )

            for bounce_type, order, distance, azimuth, closing in path_specs:
                if (
                    distance > float(config.max_range_m)
                    or abs(azimuth) > half_fov
                ):
                    continue
                snr_db = _path_snr_db(
                    target,
                    distance,
                    reflector,
                    order,
                    config,
                )
                ghost_id = _stable_negative_id(
                    "ghost",
                    target.object_id,
                    reflector.reflector_id,
                    bounce_type,
                    order,
                )
                candidates.append(
                    MultipathTarget(
                        object_id=ghost_id,
                        parent_object_id=int(target.object_id),
                        reflector_id=int(reflector.reflector_id),
                        semantic_tag=int(target.semantic_tag),
                        distance_m=float(distance),
                        azimuth_rad=float(azimuth),
                        relative_velocity_mps=float(closing),
                        snr_db=float(snr_db),
                        path_length_m=float(2.0 * distance),
                        bounce_type=bounce_type,
                        bounce_order=order,
                        reflection_point_xy_m=(
                            float(reflection_point[0]),
                            float(reflection_point[1]),
                        ),
                        lateral_extent_m=float(target.lateral_extent_m),
                    )
                )

        candidates.sort(key=lambda item: (-item.snr_db, item.distance_m))
        ghosts.extend(candidates[: int(config.multipath_max_ghosts_per_target)])
    return ghosts
