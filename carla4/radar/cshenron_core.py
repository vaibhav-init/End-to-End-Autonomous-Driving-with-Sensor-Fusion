"""NumPy-only C-Shenron material/scattering compatibility core.

The upstream C-Shenron implementation converts semantic LiDAR returns into
raw ADC samples and then a range-angle image. This project consumes a compact
target list instead, so this module retains the semantic/material mapping,
surface scattering, range/angle gating, and signal thresholding while stopping
before the expensive ADC synthesis.
"""

from dataclasses import dataclass
from enum import IntEnum
import math

import numpy as np


SEMANTIC_LIDAR_DTYPE = np.dtype(
    [
        ("x", np.float32),
        ("y", np.float32),
        ("z", np.float32),
        ("cos_incidence", np.float32),
        ("object_id", np.uint32),
        ("semantic_tag", np.uint32),
    ]
)

# CARLA 0.9.16 semantic IDs. CARLA changed this table in 0.9.14, so these
# values must not be replaced with the offset table used by older C-Shenron
# data-conversion code.
CARLA_0916_SEMANTIC_TAGS = {
    0: "Unlabeled",
    1: "Roads",
    2: "SideWalks",
    3: "Building",
    4: "Wall",
    5: "Fence",
    6: "Pole",
    7: "TrafficLight",
    8: "TrafficSign",
    9: "Vegetation",
    10: "Terrain",
    11: "Sky",
    12: "Pedestrian",
    13: "Rider",
    14: "Car",
    15: "Truck",
    16: "Bus",
    17: "Train",
    18: "Motorcycle",
    19: "Bicycle",
    20: "Static",
    21: "Dynamic",
    22: "Other",
    23: "Water",
    24: "RoadLine",
    25: "Ground",
    26: "Bridge",
    27: "RailTrack",
    28: "GuardRail",
}


class Material(IntEnum):
    UNLABELLED = 0
    WOOD = 1
    CONCRETE = 2
    HUMAN = 3
    METAL = 4


# CARLA 0.9.16 semantic tags:
# https://carla.readthedocs.io/en/0.9.16/ref_sensors/#semantic-lidar-sensor
_TAG_TO_MATERIAL = np.array(
    [
        Material.UNLABELLED,  # 0  Unlabeled
        Material.CONCRETE,    # 1  Roads
        Material.CONCRETE,    # 2  Sidewalks
        Material.CONCRETE,    # 3  Building
        Material.CONCRETE,    # 4  Wall
        Material.METAL,       # 5  Fence
        Material.METAL,       # 6  Pole
        Material.METAL,       # 7  TrafficLight
        Material.METAL,       # 8  TrafficSign
        Material.WOOD,        # 9  Vegetation
        Material.CONCRETE,    # 10 Terrain
        Material.UNLABELLED,  # 11 Sky
        Material.HUMAN,       # 12 Pedestrian
        Material.HUMAN,       # 13 Rider
        Material.METAL,       # 14 Car
        Material.METAL,       # 15 Truck
        Material.METAL,       # 16 Bus
        Material.METAL,       # 17 Train
        Material.METAL,       # 18 Motorcycle
        Material.METAL,       # 19 Bicycle
        Material.CONCRETE,    # 20 Static
        Material.METAL,       # 21 Dynamic
        Material.UNLABELLED,  # 22 Other
        Material.UNLABELLED,  # 23 Water
        Material.UNLABELLED,  # 24 RoadLine
        Material.CONCRETE,    # 25 Ground
        Material.CONCRETE,    # 26 Bridge
        Material.METAL,       # 27 RailTrack
        Material.METAL,       # 28 GuardRail
    ],
    dtype=np.uint8,
)

# Surfaces that may represent a longitudinal obstacle. Roads, sidewalks,
# terrain, vegetation, and sky are deliberately excluded.
_OBSTACLE_TAGS = frozenset(
    (3, 4, 5, 6, 7, 8, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 26, 27, 28)
)
_DYNAMIC_TAGS = frozenset((12, 13, 14, 15, 16, 17, 18, 19, 21))


@dataclass(frozen=True)
class CShenronConfig:
    """Configuration for C-Shenron-compatible target extraction."""

    max_range_m: float = 100.0
    horizontal_fov_deg: float = 10.0
    min_elevation_deg: float = -8.0
    max_elevation_deg: float = 8.0
    carrier_frequency_hz: float = 77.0e9
    speed_of_light_mps: float = 3.0e8
    voxel_azimuth_deg: float = 2.0
    voxel_elevation_deg: float = 2.0
    min_points_per_target: int = 2
    noise_power: float = 1.0e-4
    min_snr_db: float = 6.0
    include_specular: bool = False
    static_range_bin_m: float = 1.5
    static_angle_bin_deg: float = 1.0


@dataclass(frozen=True)
class RadarTarget:
    """One signal-qualified target in semantic-LiDAR sensor coordinates."""

    object_id: int
    semantic_tag: int
    distance_m: float
    direction: tuple
    snr_db: float
    point_count: int
    lateral_extent_m: float = 0.0


def decode_semantic_lidar(raw_data):
    """Return a zero-copy structured view of CARLA semantic-LiDAR bytes."""

    view = memoryview(raw_data)
    if len(view) % SEMANTIC_LIDAR_DTYPE.itemsize:
        raise ValueError(
            "Unexpected semantic-LiDAR buffer size "
            f"{len(view)} (record size {SEMANTIC_LIDAR_DTYPE.itemsize})"
        )
    return np.frombuffer(view, dtype=SEMANTIC_LIDAR_DTYPE)


def semantic_tag_name(tag):
    """Return the CARLA 0.9.16 name, preserving unknown/custom tag IDs."""

    value = int(tag)
    return CARLA_0916_SEMANTIC_TAGS.get(value, f"Unknown({value})")


def semantic_material_name(tag):
    """Return the material assigned to a CARLA 0.9.16 semantic tag."""

    value = int(tag)
    if not 0 <= value < len(_TAG_TO_MATERIAL):
        return Material.UNLABELLED.name
    return Material(int(_TAG_TO_MATERIAL[value])).name


def map_semantic_materials(tags):
    """Map CARLA 0.9.16 semantic tags to C-Shenron material classes."""

    tags = np.asarray(tags)
    result = np.full(tags.shape, Material.UNLABELLED, dtype=np.uint8)
    valid = (tags >= 0) & (tags < len(_TAG_TO_MATERIAL))
    result[valid] = _TAG_TO_MATERIAL[tags[valid].astype(np.int64)]
    return result


def cshenron_return_power(
    ranges_m,
    cos_incidence,
    materials,
    config=None,
):
    """Estimate per-point received power using C-Shenron's surface model.

    CARLA reports the cosine of the incidence angle. The public C-Shenron path
    feeds that column into an angle function; this port converts it back to an
    angle first, which is the physically consistent interpretation for 0.9.16.
    """

    config = config or CShenronConfig()
    ranges = np.maximum(np.asarray(ranges_m, dtype=np.float64), 1.0)
    cosine = np.clip(np.abs(np.asarray(cos_incidence, dtype=np.float64)), 1.0e-4, 1.0)
    material = np.asarray(materials, dtype=np.int64)
    material = np.clip(material, int(Material.UNLABELLED), int(Material.METAL))

    # Values adapted from C-Shenron Sceneset.get_loss_3.
    roughness = np.array((0.0, 0.0017, 0.0017, 0.01, 0.00005))
    permittivity = np.array((1.0, 2.0, 5.24, 15.0, 100000.0))

    sin_sq = np.maximum(0.0, 1.0 - cosine * cosine)
    root = np.sqrt(np.maximum(permittivity[material] - sin_sq, 1.0e-9))
    reflection_sq = np.square((cosine - root) / np.maximum(cosine + root, 1.0e-9))
    wavelength_factor = (
        4.0
        * np.pi
        * roughness[material]
        * cosine
        * config.carrier_frequency_hz
        / config.speed_of_light_mps
    )
    smooth_fraction = np.exp(-0.5 * np.square(wavelength_factor))
    scatter_fraction = 1.0 - smooth_fraction

    lobe = (0.9 * np.power(cosine, 8.0) + 0.1) / 1.09
    scatter = reflection_sq * scatter_fraction * np.square(lobe)

    specular = np.zeros_like(scatter)
    if config.include_specular:
        incidence = np.arccos(cosine)
        specular = (
            reflection_sq
            * smooth_fraction
            * (incidence < math.radians(2.0))
            * 4.0
        )

    voxel_solid_angle = (
        math.radians(config.voxel_azimuth_deg)
        * math.radians(config.voxel_elevation_deg)
    )
    incident_scale = 3282.0 * voxel_solid_angle
    return (
        incident_scale
        * (scatter + specular)
        * (100.0 * 25.0 / 9.0)
        / np.square(ranges)
    )


def extract_targets(returns, config=None):
    """Convert semantic returns to signal-qualified forward radar targets."""

    config = config or CShenronConfig()
    if returns.size == 0:
        return []

    x = returns["x"].astype(np.float64, copy=False)
    y = returns["y"].astype(np.float64, copy=False)
    z = returns["z"].astype(np.float64, copy=False)
    ranges = np.sqrt(x * x + y * y + z * z)
    azimuth = np.arctan2(y, x)
    elevation = np.arctan2(z, np.hypot(x, y))
    tags = returns["semantic_tag"].astype(np.int64, copy=False)

    finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    obstacle = np.isin(tags, tuple(_OBSTACLE_TAGS))
    mask = (
        finite
        & obstacle
        & (x > 0.8)
        & (ranges >= 1.0)
        & (ranges < config.max_range_m)
        & (np.abs(azimuth) <= math.radians(config.horizontal_fov_deg / 2.0))
        & (elevation >= math.radians(config.min_elevation_deg))
        & (elevation <= math.radians(config.max_elevation_deg))
    )
    if not np.any(mask):
        return []

    x = x[mask]
    y = y[mask]
    z = z[mask]
    ranges = ranges[mask]
    azimuth = azimuth[mask]
    tags = tags[mask]
    object_ids = returns["object_id"][mask].astype(np.int64, copy=False)
    cosine = returns["cos_incidence"][mask].astype(np.float64, copy=False)
    materials = map_semantic_materials(tags)
    powers = cshenron_return_power(ranges, cosine, materials, config)

    # Actor IDs are a useful grouping key for dynamic extended objects. Static
    # geometry is different: one CARLA object/mesh can span a long wall,
    # building, or guard rail, and collapsing that whole surface to one return
    # creates a nonphysical target. Split every static surface into stable
    # range/angle/tag cells, irrespective of the CARLA object ID.
    keys = object_ids.copy()
    static = ~np.isin(tags, tuple(_DYNAMIC_TAGS)) | (object_ids <= 0)
    if np.any(static):
        range_bin = np.floor(
            ranges[static] / config.static_range_bin_m
        ).astype(np.int64)
        angle_bin = np.floor(
            np.degrees(azimuth[static]) / config.static_angle_bin_deg
        ).astype(np.int64)
        # Zig-zag encoding handles signed angle bins. Two Cantor pairings give
        # a deterministic collision-free integer ID for non-negative inputs.
        angle_key = np.where(
            angle_bin >= 0,
            2 * angle_bin,
            -2 * angle_bin - 1,
        )
        first_sum = range_bin + angle_key
        first_pair = first_sum * (first_sum + 1) // 2 + angle_key
        static_tags = tags[static]
        second_sum = first_pair + static_tags
        static_key = (
            second_sum * (second_sum + 1) // 2 + static_tags
        )
        keys[static] = -(1 + static_key)

    targets = []
    # Sorting once avoids rescanning every return for every group. That matters
    # for the realistic backend's wide field of view and dense semantic LiDAR.
    order = np.argsort(keys, kind="stable")
    sorted_keys = keys[order]
    group_starts = np.concatenate(
        (
            np.array((0,), dtype=np.int64),
            np.flatnonzero(sorted_keys[1:] != sorted_keys[:-1]) + 1,
        )
    )
    group_ends = np.concatenate(
        (group_starts[1:], np.array((len(order),), dtype=np.int64))
    )
    for start, end in zip(group_starts, group_ends):
        key = int(sorted_keys[start])
        indices = order[start:end]
        if len(indices) < config.min_points_per_target:
            continue
        signal_power = float(np.sum(powers[indices]))
        snr_db = 10.0 * math.log10(
            max(signal_power, np.finfo(float).tiny) / config.noise_power
        )
        if snr_db < config.min_snr_db:
            continue

        target_ranges = ranges[indices]
        distance = float(np.quantile(target_ranges, 0.10))

        # Range represents the front scattering surface, but using that one
        # quantile's nearest LiDAR point for angle makes an extended vehicle
        # jump between its visible corners.  A target-list radar reports a
        # cluster angle, so use robust angular centroids across the object's
        # returns.  This keeps ego-corridor gating stable on curves without
        # using the CARLA actor transform or other privileged ground truth.
        target_azimuth = float(np.median(azimuth[indices]))
        target_elevation = float(
            np.median(
                np.arctan2(
                    z[indices],
                    np.hypot(x[indices], y[indices]),
                )
            )
        )
        cos_elevation = math.cos(target_elevation)
        direction = (
            cos_elevation * math.cos(target_azimuth),
            cos_elevation * math.sin(target_azimuth),
            math.sin(target_elevation),
        )
        target_lateral = distance * math.sin(target_azimuth)
        lateral_low, lateral_high = np.quantile(
            y[indices],
            (0.05, 0.95),
        )
        lateral_extent = max(
            abs(float(lateral_low) - target_lateral),
            abs(float(lateral_high) - target_lateral),
        )
        tag_counts = np.bincount(tags[indices], minlength=len(_TAG_TO_MATERIAL))
        semantic_tag = int(np.argmax(tag_counts))
        object_id = key
        targets.append(
            RadarTarget(
                object_id=object_id,
                semantic_tag=semantic_tag,
                distance_m=distance,
                direction=direction,
                snr_db=snr_db,
                point_count=len(indices),
                lateral_extent_m=float(lateral_extent),
            )
        )

    targets.sort(key=lambda target: target.distance_m)
    return targets
