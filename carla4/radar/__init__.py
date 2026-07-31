"""Pluggable forward-radar backends for the CARLA driving pipeline."""

from .cshenron_core import (
    CARLA_0916_SEMANTIC_TAGS,
    SEMANTIC_LIDAR_DTYPE,
    semantic_material_name,
    semantic_tag_name,
)
from .front_radar import (
    RADAR_BACKENDS,
    CShenronFrontRadar,
    FrontRadar,
    NativeFrontRadar,
    RealisticFrontRadar,
    add_radar_arguments,
    create_front_radar,
    describe_radar_configuration,
    normalize_radar_backend,
    radar_diagnostics_row,
    resolve_realistic_radar_config,
)
from .realistic_core import (
    DEFAULT_REALISTIC_RADAR_PROFILE,
    REALISTIC_RADAR_PROFILES,
    IdealRadarTarget,
    RadarEnvironment,
    RealisticRadarConfig,
    RealisticRadarModel,
    load_realistic_radar_config,
    realistic_radar_config_signature,
)

__all__ = [
    "DEFAULT_REALISTIC_RADAR_PROFILE",
    "CARLA_0916_SEMANTIC_TAGS",
    "RADAR_BACKENDS",
    "REALISTIC_RADAR_PROFILES",
    "SEMANTIC_LIDAR_DTYPE",
    "CShenronFrontRadar",
    "FrontRadar",
    "IdealRadarTarget",
    "NativeFrontRadar",
    "RadarEnvironment",
    "RealisticFrontRadar",
    "RealisticRadarConfig",
    "RealisticRadarModel",
    "add_radar_arguments",
    "create_front_radar",
    "describe_radar_configuration",
    "load_realistic_radar_config",
    "normalize_radar_backend",
    "radar_diagnostics_row",
    "realistic_radar_config_signature",
    "resolve_realistic_radar_config",
    "semantic_material_name",
    "semantic_tag_name",
]
