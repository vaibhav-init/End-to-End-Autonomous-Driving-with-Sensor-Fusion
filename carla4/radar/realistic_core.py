"""Temporal target-list radar model for simulator-independent testing.

This module deliberately stops at the automotive-radar target-list boundary.
Geometry, occlusion, material class, and incidence-dependent return strength
come from the C-Shenron compatibility core.  This layer models effects that
are absent from CARLA and from the lightweight C-Shenron adapter:

* SNR-dependent detection and measurement error
* temporally correlated noise and missed detections
* sensor cadence, quantization, ambiguity, and latency
* unstructured clutter and interference bursts
* persistent multipath-like ghost targets
* nearest-neighbour association, M-of-N confirmation, and track deletion
* ego-path gating before reducing the target list to the scalar MLP contract

The built-in parameters are research priors, not a claim that a particular
commercial radar has been reproduced.  ``load_realistic_radar_config`` accepts
JSON overrides so parameters fitted from real sequences can replace them.
"""

from collections import deque
from dataclasses import asdict, dataclass, field, fields, replace
import hashlib
import json
import math
import os

import numpy as np


DEFAULT_REALISTIC_RADAR_PROFILE = "generic_lrr_v1"
REALISTIC_RADAR_PROFILES = (
    "ideal_target_list_v1",
    "gaussian_baseline_v1",
    "generic_lrr_v1",
    "geometry_multipath_v1",
)
_PROFILE_DIRECTORY = os.path.join(os.path.dirname(__file__), "profiles")


@dataclass(frozen=True)
class RealisticRadarConfig:
    """Configuration of the phenomenological radar and its target tracker."""

    schema_version: int = 1
    profile_name: str = DEFAULT_REALISTIC_RADAR_PROFILE

    # Sensor envelope.  The default 50 ms cycle matches this repository's
    # synchronous 20 Hz control loop; RadarScenes reports an average 60 ms.
    max_range_m: float = 100.0
    horizontal_fov_deg: float = 120.0
    min_elevation_deg: float = -8.0
    max_elevation_deg: float = 8.0
    cycle_time_s: float = 0.05
    range_resolution_m: float = 0.15
    doppler_resolution_mps: float = 0.1 / 3.6
    azimuth_resolution_boresight_deg: float = 0.5
    azimuth_resolution_edge_deg: float = 2.0
    max_unambiguous_doppler_mps: float = 70.0

    # Measurement errors.  Each sigma is floor + scale / sqrt(linear SNR).
    range_noise_floor_m: float = 0.06
    range_noise_snr_scale_m: float = 0.45
    doppler_noise_floor_mps: float = 0.04
    doppler_noise_snr_scale_mps: float = 0.30
    azimuth_noise_floor_deg: float = 0.20
    azimuth_noise_snr_scale_deg: float = 1.50
    snr_fluctuation_std_db: float = 1.5
    error_correlation: float = 0.82

    # Probability of detection and correlated dropout state.
    detection_snr_midpoint_db: float = 8.0
    detection_snr_slope: float = 0.55
    min_detection_probability: float = 0.02
    max_detection_probability: float = 0.995
    dropout_enter_probability: float = 0.012
    dropout_exit_probability: float = 0.35
    dropout_detection_scale: float = 0.08

    # Target-list clutter and mutual interference.
    false_alarms_per_scan: float = 0.08
    stationary_clutter_fraction: float = 0.75
    interference_enter_probability: float = 0.002
    interference_exit_probability: float = 0.30
    interference_detection_scale: float = 0.25
    interference_clutter_multiplier: float = 8.0

    # Persistent multipath-like target priors.  Geometry-aware calibration can
    # replace these priors through a JSON profile.
    ghost_start_probability: float = 0.004
    ghost_survival_probability: float = 0.94
    ghost_max_age_scans: int = 40
    max_active_ghosts: int = 4
    ghost_snr_loss_db: float = 8.0
    ghost_min_range_bias_m: float = 1.0
    ghost_max_range_bias_m: float = 18.0
    wet_road_ghost_multiplier: float = 2.0

    # Multipath source. ``probabilistic`` preserves the original stochastic
    # target-list model. ``geometry`` accepts deterministic image-method paths
    # fitted to semantic-LiDAR walls/guardrails. ``off`` disables both.
    multipath_mode: str = "probabilistic"
    multipath_reflector_cell_size_m: float = 8.0
    multipath_reflector_min_points: int = 8
    multipath_reflector_min_length_m: float = 1.5
    multipath_reflector_max_residual_m: float = 0.45
    multipath_reflector_min_height_m: float = -1.5
    multipath_reflector_max_height_m: float = 3.0
    multipath_max_reflectors: int = 48
    multipath_segment_margin_m: float = 1.0
    multipath_min_incidence_cosine: float = 0.015
    multipath_max_target_surface_distance_m: float = 18.0
    multipath_min_range_separation_m: float = 0.30
    multipath_second_order_loss_db: float = 5.0
    multipath_third_order_loss_db: float = 9.0
    multipath_enable_third_order: bool = True
    multipath_max_ghosts_per_target: int = 6

    # Weather is deliberately modest at 77 GHz.  Values represent additional
    # dB loss over 100 m at a normalized CARLA weather setting of 1.0.
    rain_attenuation_db_per_100m: float = 1.0
    fog_attenuation_db_per_100m: float = 0.05
    dust_attenuation_db_per_100m: float = 0.50

    # Processing delay and target tracker.
    latency_scans: int = 1
    confirmation_hits: int = 2
    confirmation_window: int = 3
    deletion_misses: int = 4
    association_range_gate_m: float = 3.0
    association_azimuth_gate_deg: float = 4.0
    association_doppler_gate_mps: float = 3.0
    range_filter_gain: float = 0.65
    azimuth_filter_gain: float = 0.55
    doppler_filter_gain: float = 0.60
    minimum_track_confidence: float = 0.55

    # Scalar-controller target selection.  Lateral gating avoids selecting an
    # adjacent-lane return merely because it is the nearest radar target.
    path_half_width_m: float = 1.8
    path_width_growth_per_m: float = 0.004
    minimum_forward_distance_m: float = 1.0
    path_curvature_filter_gain: float = 0.25
    max_abs_path_curvature_per_m: float = 0.08
    max_path_lateral_offset_m: float = 8.0
    non_road_user_priority_penalty_m: float = 15.0


@dataclass(frozen=True)
class RadarEnvironment:
    """Normalized environmental conditions in the range [0, 1]."""

    precipitation: float = 0.0
    wetness: float = 0.0
    fog: float = 0.0
    dust: float = 0.0

    def clamped(self):
        return RadarEnvironment(
            precipitation=float(np.clip(self.precipitation, 0.0, 1.0)),
            wetness=float(np.clip(self.wetness, 0.0, 1.0)),
            fog=float(np.clip(self.fog, 0.0, 1.0)),
            dust=float(np.clip(self.dust, 0.0, 1.0)),
        )


@dataclass(frozen=True)
class IdealRadarTarget:
    """One ideal material-qualified target before sensor imperfections."""

    object_id: int
    semantic_tag: int
    distance_m: float
    azimuth_rad: float
    relative_velocity_mps: float
    snr_db: float
    point_count: int = 1
    lateral_extent_m: float = 0.0
    source: str = "direct"
    parent_object_id: int = 0
    reflector_id: int = 0
    bounce_type: str = "direct"
    bounce_order: int = 1
    path_length_m: float = 0.0


@dataclass(frozen=True)
class RadarDetection:
    """One corrupted target-list detection."""

    distance_m: float
    azimuth_rad: float
    relative_velocity_mps: float
    snr_db: float
    source: str
    truth_object_id: int
    semantic_tag: int
    lateral_extent_m: float = 0.0
    truth_parent_object_id: int = 0
    reflector_id: int = 0
    bounce_type: str = "direct"
    bounce_order: int = 1
    path_length_m: float = 0.0
    ghost_probability: float = 0.0


@dataclass(frozen=True)
class RadarModelOutput:
    """Selected longitudinal target after tracking and path gating."""

    distance_m: float
    relative_velocity_mps: float
    track_id: int = 0
    confidence: float = 0.0
    source: str = "none"
    truth_object_id: int = 0
    semantic_tag: int = 0
    azimuth_rad: float = 0.0
    lateral_extent_m: float = 0.0
    truth_parent_object_id: int = 0
    reflector_id: int = 0
    bounce_type: str = "direct"
    bounce_order: int = 1
    path_length_m: float = 0.0
    ghost_probability: float = 0.0


@dataclass
class _Track:
    track_id: int
    distance_m: float
    azimuth_rad: float
    relative_velocity_mps: float
    snr_db: float
    source: str
    truth_object_id: int
    semantic_tag: int
    lateral_extent_m: float = 0.0
    truth_parent_object_id: int = 0
    reflector_id: int = 0
    bounce_type: str = "direct"
    bounce_order: int = 1
    path_length_m: float = 0.0
    ghost_probability: float = 0.0
    age: int = 1
    hits: int = 1
    misses: int = 0
    confirmed: bool = False
    hit_history: deque = field(default_factory=deque)


@dataclass
class _Ghost:
    ghost_id: int
    parent_object_id: int
    semantic_tag: int
    range_bias_m: float
    azimuth_bias_rad: float
    velocity_scale: float
    snr_loss_db: float
    distance_m: float
    azimuth_rad: float
    relative_velocity_mps: float
    snr_db: float
    lateral_extent_m: float = 0.0
    age: int = 0


def _profile_path(profile_name):
    if profile_name not in REALISTIC_RADAR_PROFILES:
        raise ValueError(
            "Unknown realistic radar profile "
            f"'{profile_name}'. Choose from {REALISTIC_RADAR_PROFILES}."
        )
    return os.path.join(_PROFILE_DIRECTORY, f"{profile_name}.json")


def _read_json_object(path):
    with open(path, "r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Radar configuration must be a JSON object: {path}")
    return value


def _config_from_mapping(mapping):
    valid_names = {item.name for item in fields(RealisticRadarConfig)}
    unknown = sorted(set(mapping) - valid_names)
    if unknown:
        raise ValueError(
            "Unknown realistic radar configuration field(s): "
            + ", ".join(unknown)
        )
    config = RealisticRadarConfig(**mapping)
    _validate_config(config)
    return config


def _validate_probability(name, value):
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {value}")


def _validate_config(config):
    if not isinstance(config.profile_name, str) or not config.profile_name.strip():
        raise ValueError("profile_name must be a non-empty string")
    if config.schema_version != 1:
        raise ValueError(
            f"Unsupported radar config schema {config.schema_version}; expected 1"
        )
    integer_fields = (
        "schema_version",
        "ghost_max_age_scans",
        "max_active_ghosts",
        "multipath_reflector_min_points",
        "multipath_max_reflectors",
        "multipath_max_ghosts_per_target",
        "latency_scans",
        "confirmation_hits",
        "confirmation_window",
        "deletion_misses",
    )
    for name in integer_fields:
        value = getattr(config, name)
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"{name} must be an integer")
    numeric_fields = (
        item.name
        for item in fields(RealisticRadarConfig)
        if item.name
        not in (
            "profile_name",
            "multipath_mode",
            "multipath_enable_third_order",
        )
    )
    for name in numeric_fields:
        value = getattr(config, name)
        if isinstance(value, bool) or not isinstance(
            value,
            (int, float, np.integer, np.floating),
        ):
            raise ValueError(f"{name} must be numeric")
        if not math.isfinite(float(value)):
            raise ValueError(f"{name} must be finite")
    positive = (
        "max_range_m",
        "horizontal_fov_deg",
        "cycle_time_s",
        "detection_snr_slope",
        "association_range_gate_m",
        "association_azimuth_gate_deg",
        "association_doppler_gate_mps",
        "path_half_width_m",
        "max_path_lateral_offset_m",
        "multipath_reflector_cell_size_m",
        "multipath_reflector_min_length_m",
        "multipath_max_target_surface_distance_m",
    )
    for name in positive:
        if getattr(config, name) <= 0.0:
            raise ValueError(f"{name} must be positive")
    non_negative = (
        "range_resolution_m",
        "doppler_resolution_mps",
        "max_unambiguous_doppler_mps",
        "range_noise_floor_m",
        "range_noise_snr_scale_m",
        "doppler_noise_floor_mps",
        "doppler_noise_snr_scale_mps",
        "azimuth_noise_floor_deg",
        "azimuth_noise_snr_scale_deg",
        "azimuth_resolution_boresight_deg",
        "azimuth_resolution_edge_deg",
        "snr_fluctuation_std_db",
        "false_alarms_per_scan",
        "latency_scans",
        "minimum_forward_distance_m",
        "path_width_growth_per_m",
        "max_abs_path_curvature_per_m",
        "non_road_user_priority_penalty_m",
        "ghost_snr_loss_db",
        "ghost_min_range_bias_m",
        "ghost_max_range_bias_m",
        "rain_attenuation_db_per_100m",
        "fog_attenuation_db_per_100m",
        "dust_attenuation_db_per_100m",
        "multipath_reflector_max_residual_m",
        "multipath_max_reflectors",
        "multipath_segment_margin_m",
        "multipath_min_range_separation_m",
        "multipath_second_order_loss_db",
        "multipath_third_order_loss_db",
        "multipath_max_ghosts_per_target",
    )
    for name in non_negative:
        if getattr(config, name) < 0:
            raise ValueError(f"{name} must be non-negative")
    probabilities = (
        "min_detection_probability",
        "max_detection_probability",
        "dropout_enter_probability",
        "dropout_exit_probability",
        "dropout_detection_scale",
        "stationary_clutter_fraction",
        "interference_enter_probability",
        "interference_exit_probability",
        "interference_detection_scale",
        "ghost_start_probability",
        "ghost_survival_probability",
        "error_correlation",
        "range_filter_gain",
        "azimuth_filter_gain",
        "doppler_filter_gain",
        "minimum_track_confidence",
        "path_curvature_filter_gain",
    )
    for name in probabilities:
        _validate_probability(name, getattr(config, name))
    if config.min_detection_probability > config.max_detection_probability:
        raise ValueError(
            "min_detection_probability cannot exceed max_detection_probability"
        )
    if config.max_elevation_deg <= config.min_elevation_deg:
        raise ValueError("max_elevation_deg must exceed min_elevation_deg")
    if config.horizontal_fov_deg > 360.0:
        raise ValueError("horizontal_fov_deg cannot exceed 360")
    if not -90.0 <= config.min_elevation_deg < config.max_elevation_deg <= 90.0:
        raise ValueError("elevation limits must lie within [-90, 90]")
    if config.minimum_forward_distance_m >= config.max_range_m:
        raise ValueError("minimum_forward_distance_m must be below max_range_m")
    if (
        config.ghost_max_range_bias_m < config.ghost_min_range_bias_m
    ):
        raise ValueError(
            "ghost_max_range_bias_m cannot be below ghost_min_range_bias_m"
        )
    if (
        config.interference_clutter_multiplier < 0.0
        or config.wet_road_ghost_multiplier < 0.0
    ):
        raise ValueError("clutter and wet-road multipliers must be non-negative")
    if not 1 <= config.confirmation_hits <= config.confirmation_window:
        raise ValueError(
            "confirmation_hits must be between 1 and confirmation_window"
        )
    if config.deletion_misses < 1:
        raise ValueError("deletion_misses must be at least 1")
    if config.ghost_max_age_scans < 1 or config.max_active_ghosts < 0:
        raise ValueError("ghost ages/counts must be valid non-negative integers")
    if config.multipath_mode not in ("off", "probabilistic", "geometry"):
        raise ValueError(
            "multipath_mode must be 'off', 'probabilistic', or 'geometry'"
        )
    if not isinstance(config.multipath_enable_third_order, bool):
        raise ValueError("multipath_enable_third_order must be a boolean")
    if config.multipath_reflector_min_points < 2:
        raise ValueError("multipath_reflector_min_points must be at least 2")
    if (
        config.multipath_reflector_max_height_m
        <= config.multipath_reflector_min_height_m
    ):
        raise ValueError(
            "multipath reflector maximum height must exceed minimum height"
        )
    if not 0.0 <= config.multipath_min_incidence_cosine <= 1.0:
        raise ValueError("multipath_min_incidence_cosine must be in [0, 1]")
    return config


def load_realistic_radar_config(
    profile_name=None,
    config_path=None,
    config=None,
    max_range_m=None,
    cycle_time_s=None,
):
    """Load a built-in profile, embedded mapping, and optional JSON overrides.

    Precedence is: dataclass defaults, built-in profile or embedded ``config``,
    external JSON override, then explicit runtime range/cycle values.
    """

    if isinstance(config, RealisticRadarConfig):
        values = asdict(config)
    elif config is not None:
        if not isinstance(config, dict):
            raise TypeError("config must be RealisticRadarConfig or a mapping")
        values = asdict(RealisticRadarConfig())
        values.update(config)
    else:
        selected = (
            profile_name
            or os.environ.get("CARLA_RADAR_PROFILE")
            or DEFAULT_REALISTIC_RADAR_PROFILE
        )
        values = asdict(RealisticRadarConfig())
        values.update(_read_json_object(_profile_path(selected)))

    if config_path:
        values.update(_read_json_object(config_path))
    if profile_name and config is not None and not config_path:
        embedded_profile = values.get("profile_name")
        if embedded_profile != profile_name:
            raise ValueError(
                f"Embedded radar profile is '{embedded_profile}', but "
                f"'{profile_name}' was requested."
            )
    if max_range_m is not None:
        values["max_range_m"] = float(max_range_m)
    if cycle_time_s is not None:
        values["cycle_time_s"] = float(cycle_time_s)
    return _config_from_mapping(values)


def realistic_radar_config_dict(config):
    """Return a JSON-serializable, validated configuration mapping."""

    _validate_config(config)
    return asdict(config)


def realistic_radar_config_signature(config):
    """Stable short identity used to prevent mixed sensor distributions."""

    payload = json.dumps(
        realistic_radar_config_dict(config),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


class RealisticRadarModel:
    """Stateful target-list error model and longitudinal target tracker."""

    _DYNAMIC_TAGS = frozenset((12, 13, 14, 15, 16, 17, 18, 19, 21))
    _ROAD_USER_TAGS = frozenset((12, 13, 14, 15, 16, 17, 18, 19))
    _REFLECTOR_TAGS = frozenset((3, 4, 5, 20, 26, 28))

    def __init__(
        self,
        config=None,
        seed=42,
        capture_debug=False,
        detection_filter=None,
    ):
        self.config = config or load_realistic_radar_config()
        _validate_config(self.config)
        self._rng = np.random.default_rng(int(seed))
        self._capture_debug = bool(capture_debug)
        self._detection_filter = detection_filter
        self._scan_index = 0
        self._last_timestamp_s = None
        self._dropout_bad = {}
        self._error_state = {}
        self._interference_active = False
        self._latency_queue = deque()
        self._tracks = {}
        self._ghosts = {}
        self._next_track_id = 1
        self._next_ghost_id = -1
        self._diagnostics = {
            "profile": self.config.profile_name,
            "config_signature": realistic_radar_config_signature(self.config),
            "scan_index": 0,
        }
        self._debug_snapshot = {
            "ideal_targets": [],
            "multipath_ideal_targets": [],
            "generated_detections": [],
            "accepted_detections": [],
            "rejected_detections": [],
            "delivered_detections": [],
            "tracks": [],
            "selected": asdict(
                RadarModelOutput(
                    distance_m=self.config.max_range_m,
                    relative_velocity_mps=0.0,
                )
            ),
        }

    @staticmethod
    def _quantize(value, resolution):
        if resolution <= 0.0:
            return float(value)
        return float(round(float(value) / resolution) * resolution)

    def _azimuth_resolution_rad(self, azimuth_rad):
        half_fov = max(math.radians(self.config.horizontal_fov_deg / 2.0), 1.0e-9)
        fraction = min(1.0, abs(azimuth_rad) / half_fov)
        resolution_deg = (
            self.config.azimuth_resolution_boresight_deg
            + fraction
            * (
                self.config.azimuth_resolution_edge_deg
                - self.config.azimuth_resolution_boresight_deg
            )
        )
        return math.radians(max(0.0, resolution_deg))

    def _attenuation_db(self, distance_m, environment):
        scale = max(0.0, float(distance_m)) / 100.0
        return scale * (
            self.config.rain_attenuation_db_per_100m
            * environment.precipitation
            + self.config.fog_attenuation_db_per_100m * environment.fog
            + self.config.dust_attenuation_db_per_100m * environment.dust
        )

    def _detection_probability(self, snr_db):
        exponent = np.clip(
            -self.config.detection_snr_slope
            * (snr_db - self.config.detection_snr_midpoint_db),
            -60.0,
            60.0,
        )
        logistic = 1.0 / (1.0 + math.exp(float(exponent)))
        return (
            self.config.min_detection_probability
            + (
                self.config.max_detection_probability
                - self.config.min_detection_probability
            )
            * logistic
        )

    def _update_correlated_error(self, object_id):
        previous = self._error_state.get(object_id, np.zeros(4, dtype=np.float64))
        rho = self.config.error_correlation
        innovation_scale = math.sqrt(max(0.0, 1.0 - rho * rho))
        updated = rho * previous + innovation_scale * self._rng.normal(size=4)
        self._error_state[object_id] = updated
        return updated

    def _update_dropout_state(self, object_id):
        bad = self._dropout_bad.get(object_id, False)
        if bad:
            if self._rng.random() < self.config.dropout_exit_probability:
                bad = False
        elif self._rng.random() < self.config.dropout_enter_probability:
            bad = True
        self._dropout_bad[object_id] = bad
        return bad

    def _measure_target(self, target, environment, source="direct"):
        error = self._update_correlated_error(target.object_id)
        snr_db = (
            target.snr_db
            - self._attenuation_db(target.distance_m, environment)
            + error[3] * self.config.snr_fluctuation_std_db
        )
        probability = self._detection_probability(snr_db)
        if source == "direct" and self._update_dropout_state(target.object_id):
            probability *= self.config.dropout_detection_scale
        if self._interference_active:
            probability *= self.config.interference_detection_scale
        if self._rng.random() >= probability:
            return None

        inverse_sqrt_snr = 10.0 ** (-max(snr_db, -20.0) / 20.0)
        range_sigma = (
            self.config.range_noise_floor_m
            + self.config.range_noise_snr_scale_m * inverse_sqrt_snr
        )
        doppler_sigma = (
            self.config.doppler_noise_floor_mps
            + self.config.doppler_noise_snr_scale_mps * inverse_sqrt_snr
        )
        azimuth_sigma = math.radians(
            self.config.azimuth_noise_floor_deg
            + self.config.azimuth_noise_snr_scale_deg * inverse_sqrt_snr
        )

        distance = target.distance_m + error[0] * range_sigma
        azimuth = target.azimuth_rad + error[1] * azimuth_sigma
        velocity = target.relative_velocity_mps + error[2] * doppler_sigma

        ambiguity = self.config.max_unambiguous_doppler_mps
        if ambiguity > 0.0:
            velocity = (velocity + ambiguity) % (2.0 * ambiguity) - ambiguity

        distance = self._quantize(distance, self.config.range_resolution_m)
        velocity = self._quantize(
            velocity,
            self.config.doppler_resolution_mps,
        )
        azimuth = self._quantize(
            azimuth,
            self._azimuth_resolution_rad(azimuth),
        )
        distance = float(
            np.clip(
                distance,
                self.config.minimum_forward_distance_m,
                self.config.max_range_m,
            )
        )
        return RadarDetection(
            distance_m=distance,
            azimuth_rad=azimuth,
            relative_velocity_mps=velocity,
            snr_db=float(snr_db),
            source=target.source if target.source != "direct" else source,
            truth_object_id=int(target.object_id),
            semantic_tag=int(target.semantic_tag),
            lateral_extent_m=max(0.0, float(target.lateral_extent_m)),
            truth_parent_object_id=int(target.parent_object_id),
            reflector_id=int(target.reflector_id),
            bounce_type=str(target.bounce_type),
            bounce_order=int(target.bounce_order),
            path_length_m=max(0.0, float(target.path_length_m)),
        )

    def _update_interference(self):
        if self._interference_active:
            if self._rng.random() < self.config.interference_exit_probability:
                self._interference_active = False
        elif self._rng.random() < self.config.interference_enter_probability:
            self._interference_active = True

    def _create_clutter(self):
        rate = self.config.false_alarms_per_scan
        if self._interference_active:
            rate *= self.config.interference_clutter_multiplier
        count = int(self._rng.poisson(rate))
        detections = []
        half_fov = math.radians(self.config.horizontal_fov_deg / 2.0)
        for index in range(count):
            distance = float(
                np.clip(
                    self._rng.exponential(self.config.max_range_m / 3.0),
                    self.config.minimum_forward_distance_m,
                    self.config.max_range_m,
                )
            )
            azimuth = float(self._rng.uniform(-half_fov, half_fov))
            if self._rng.random() < self.config.stationary_clutter_fraction:
                velocity = float(self._rng.normal(0.0, 0.20))
            else:
                velocity = float(self._rng.normal(0.0, 5.0))
            detections.append(
                RadarDetection(
                    distance_m=float(
                        np.clip(
                            self._quantize(
                                distance,
                                self.config.range_resolution_m,
                            ),
                            self.config.minimum_forward_distance_m,
                            self.config.max_range_m,
                        )
                    ),
                    azimuth_rad=self._quantize(
                        azimuth,
                        self._azimuth_resolution_rad(azimuth),
                    ),
                    relative_velocity_mps=self._quantize(
                        velocity,
                        self.config.doppler_resolution_mps,
                    ),
                    snr_db=float(
                        self.config.detection_snr_midpoint_db
                        + self._rng.exponential(2.0)
                    ),
                    source="clutter",
                    truth_object_id=-1000000 - self._scan_index * 1000 - index,
                    semantic_tag=0,
                )
            )
        return detections

    def _spawn_ghosts(self, targets, environment):
        if (
            self.config.max_active_ghosts <= 0
            or self.config.ghost_start_probability <= 0.0
        ):
            return
        reflectors = [
            target
            for target in targets
            if target.semantic_tag in self._REFLECTOR_TAGS
        ]
        sources = [
            target
            for target in targets
            if target.semantic_tag in self._DYNAMIC_TAGS
        ]
        multiplier = 1.0 + environment.wetness * (
            self.config.wet_road_ghost_multiplier - 1.0
        )
        start_probability = min(
            1.0,
            self.config.ghost_start_probability * multiplier,
        )
        active_parents = {ghost.parent_object_id for ghost in self._ghosts.values()}
        for target in sources:
            if len(self._ghosts) >= self.config.max_active_ghosts:
                break
            if target.object_id in active_parents:
                continue
            if self._rng.random() >= start_probability:
                continue

            range_bias = float(
                self._rng.uniform(
                    self.config.ghost_min_range_bias_m,
                    self.config.ghost_max_range_bias_m,
                )
            )
            if reflectors:
                reflector = min(
                    reflectors,
                    key=lambda item: abs(item.azimuth_rad - target.azimuth_rad),
                )
                mirrored = 2.0 * reflector.azimuth_rad - target.azimuth_rad
                azimuth_bias = float(mirrored - target.azimuth_rad)
                range_bias += min(
                    self.config.ghost_max_range_bias_m - range_bias,
                    0.25 * abs(reflector.distance_m - target.distance_m),
                )
            else:
                # Ground/underbody multipath tends to remain near the direct
                # target in angle, unlike a wall-reflected virtual target.
                azimuth_bias = float(
                    self._rng.normal(0.0, math.radians(0.7))
                )

            ghost_id = self._next_ghost_id
            self._next_ghost_id -= 1
            self._ghosts[ghost_id] = _Ghost(
                ghost_id=ghost_id,
                parent_object_id=target.object_id,
                semantic_tag=target.semantic_tag,
                range_bias_m=range_bias,
                azimuth_bias_rad=azimuth_bias,
                velocity_scale=float(self._rng.normal(1.0, 0.05)),
                snr_loss_db=self.config.ghost_snr_loss_db
                + float(self._rng.exponential(2.0)),
                distance_m=target.distance_m + range_bias,
                azimuth_rad=target.azimuth_rad + azimuth_bias,
                relative_velocity_mps=target.relative_velocity_mps,
                snr_db=target.snr_db - self.config.ghost_snr_loss_db,
                lateral_extent_m=target.lateral_extent_m,
            )

    def _update_ghosts(self, targets, environment, dt):
        target_by_id = {target.object_id: target for target in targets}
        detections = []
        expired = []
        for ghost_id, ghost in self._ghosts.items():
            ghost.age += 1
            if (
                ghost.age > self.config.ghost_max_age_scans
                or self._rng.random() > self.config.ghost_survival_probability
            ):
                expired.append(ghost_id)
                continue

            parent = target_by_id.get(ghost.parent_object_id)
            if parent is not None:
                ghost.distance_m = parent.distance_m + ghost.range_bias_m
                ghost.azimuth_rad = parent.azimuth_rad + ghost.azimuth_bias_rad
                ghost.relative_velocity_mps = (
                    parent.relative_velocity_mps * ghost.velocity_scale
                )
                ghost.snr_db = parent.snr_db - ghost.snr_loss_db
                ghost.lateral_extent_m = parent.lateral_extent_m
            else:
                ghost.distance_m -= ghost.relative_velocity_mps * dt

            if not (
                self.config.minimum_forward_distance_m
                <= ghost.distance_m
                <= self.config.max_range_m
            ):
                expired.append(ghost_id)
                continue
            half_fov = math.radians(self.config.horizontal_fov_deg / 2.0)
            if abs(ghost.azimuth_rad) > half_fov:
                expired.append(ghost_id)
                continue

            ideal_ghost = IdealRadarTarget(
                object_id=ghost.ghost_id,
                semantic_tag=ghost.semantic_tag,
                distance_m=ghost.distance_m,
                azimuth_rad=ghost.azimuth_rad,
                relative_velocity_mps=ghost.relative_velocity_mps,
                snr_db=ghost.snr_db,
                lateral_extent_m=ghost.lateral_extent_m,
                source="ghost",
                parent_object_id=ghost.parent_object_id,
                bounce_type="probabilistic",
                bounce_order=0,
                path_length_m=2.0 * ghost.distance_m,
            )
            detection = self._measure_target(
                ideal_ghost,
                environment,
                source="ghost",
            )
            if detection is not None:
                detections.append(detection)

        for ghost_id in expired:
            self._ghosts.pop(ghost_id, None)
            self._error_state.pop(ghost_id, None)
            self._dropout_bad.pop(ghost_id, None)
        return detections

    def _predict_tracks(self, dt):
        for track in self._tracks.values():
            track.distance_m = max(
                self.config.minimum_forward_distance_m,
                track.distance_m - track.relative_velocity_mps * dt,
            )
            track.age += 1

    def _association_cost(self, track, detection):
        distance_delta = abs(track.distance_m - detection.distance_m)
        azimuth_delta = abs(track.azimuth_rad - detection.azimuth_rad)
        velocity_delta = abs(
            track.relative_velocity_mps - detection.relative_velocity_mps
        )
        azimuth_gate = math.radians(
            self.config.association_azimuth_gate_deg
        )
        if (
            distance_delta > self.config.association_range_gate_m
            or azimuth_delta > azimuth_gate
            or velocity_delta > self.config.association_doppler_gate_mps
        ):
            return None
        return (
            distance_delta / self.config.association_range_gate_m
            + azimuth_delta / azimuth_gate
            + velocity_delta / self.config.association_doppler_gate_mps
        )

    def _new_track(self, detection):
        history = deque(maxlen=self.config.confirmation_window)
        history.append(1)
        confirmed = self.config.confirmation_hits <= 1
        track = _Track(
            track_id=self._next_track_id,
            distance_m=detection.distance_m,
            azimuth_rad=detection.azimuth_rad,
            relative_velocity_mps=detection.relative_velocity_mps,
            snr_db=detection.snr_db,
            source=detection.source,
            truth_object_id=detection.truth_object_id,
            semantic_tag=detection.semantic_tag,
            lateral_extent_m=detection.lateral_extent_m,
            truth_parent_object_id=detection.truth_parent_object_id,
            reflector_id=detection.reflector_id,
            bounce_type=detection.bounce_type,
            bounce_order=detection.bounce_order,
            path_length_m=detection.path_length_m,
            ghost_probability=detection.ghost_probability,
            confirmed=confirmed,
            hit_history=history,
        )
        self._tracks[track.track_id] = track
        self._next_track_id += 1

    def _update_track(self, track, detection):
        track.distance_m += self.config.range_filter_gain * (
            detection.distance_m - track.distance_m
        )
        track.azimuth_rad += self.config.azimuth_filter_gain * (
            detection.azimuth_rad - track.azimuth_rad
        )
        track.relative_velocity_mps += self.config.doppler_filter_gain * (
            detection.relative_velocity_mps - track.relative_velocity_mps
        )
        track.snr_db = 0.7 * track.snr_db + 0.3 * detection.snr_db
        track.source = detection.source
        track.truth_object_id = detection.truth_object_id
        track.semantic_tag = detection.semantic_tag
        track.lateral_extent_m += self.config.azimuth_filter_gain * (
            detection.lateral_extent_m - track.lateral_extent_m
        )
        track.truth_parent_object_id = detection.truth_parent_object_id
        track.reflector_id = detection.reflector_id
        track.bounce_type = detection.bounce_type
        track.bounce_order = detection.bounce_order
        track.path_length_m = detection.path_length_m
        track.ghost_probability = detection.ghost_probability
        track.hits += 1
        track.misses = 0
        track.hit_history.append(1)
        if sum(track.hit_history) >= self.config.confirmation_hits:
            track.confirmed = True

    def _miss_track(self, track):
        track.misses += 1
        track.hit_history.append(0)

    def _update_tracks(self, detections, dt):
        self._predict_tracks(dt)
        candidates = []
        for track_id, track in self._tracks.items():
            for detection_index, detection in enumerate(detections):
                cost = self._association_cost(track, detection)
                if cost is not None:
                    candidates.append((cost, track_id, detection_index))
        candidates.sort()

        assigned_tracks = set()
        assigned_detections = set()
        for _, track_id, detection_index in candidates:
            if (
                track_id in assigned_tracks
                or detection_index in assigned_detections
            ):
                continue
            self._update_track(
                self._tracks[track_id],
                detections[detection_index],
            )
            assigned_tracks.add(track_id)
            assigned_detections.add(detection_index)

        for track_id, track in tuple(self._tracks.items()):
            if track_id not in assigned_tracks:
                self._miss_track(track)
            if track.misses >= self.config.deletion_misses:
                del self._tracks[track_id]

        for detection_index, detection in enumerate(detections):
            if detection_index not in assigned_detections:
                self._new_track(detection)

    def _track_confidence(self, track):
        if not track.hit_history:
            return 0.0
        hit_ratio = sum(track.hit_history) / len(track.hit_history)
        maturity = min(1.0, track.hits / max(self.config.confirmation_hits, 1))
        miss_penalty = math.exp(-0.45 * track.misses)
        return float(np.clip(hit_ratio * maturity * miss_penalty, 0.0, 1.0))

    def _path_lateral_offset(self, forward_m, path_curvature_per_m):
        curvature = float(
            np.clip(
                path_curvature_per_m,
                -self.config.max_abs_path_curvature_per_m,
                self.config.max_abs_path_curvature_per_m,
            )
        )
        offset = 0.5 * curvature * forward_m * forward_m
        return float(
            np.clip(
                offset,
                -self.config.max_path_lateral_offset_m,
                self.config.max_path_lateral_offset_m,
            )
        )

    def _select_track(self, path_curvature_per_m=0.0):
        candidates = []
        for track in self._tracks.values():
            confidence = self._track_confidence(track)
            if (
                not track.confirmed
                or confidence < self.config.minimum_track_confidence
                or track.distance_m < self.config.minimum_forward_distance_m
            ):
                continue
            forward_m = track.distance_m * math.cos(track.azimuth_rad)
            if forward_m < self.config.minimum_forward_distance_m:
                continue
            lateral_m = track.distance_m * math.sin(track.azimuth_rad)
            path_lateral_m = self._path_lateral_offset(
                forward_m,
                path_curvature_per_m,
            )
            path_half_width = (
                self.config.path_half_width_m
                + self.config.path_width_growth_per_m * track.distance_m
            )
            target_min_lateral = lateral_m - track.lateral_extent_m
            target_max_lateral = lateral_m + track.lateral_extent_m
            path_min_lateral = path_lateral_m - path_half_width
            path_max_lateral = path_lateral_m + path_half_width
            if (
                target_max_lateral < path_min_lateral
                or target_min_lateral > path_max_lateral
            ):
                continue
            class_penalty = (
                0.0
                if track.semantic_tag in self._ROAD_USER_TAGS
                else self.config.non_road_user_priority_penalty_m
            )
            selection_score = forward_m + class_penalty
            candidates.append(
                (
                    selection_score,
                    forward_m,
                    -confidence,
                    track.track_id,
                    track,
                )
            )

        if not candidates:
            return RadarModelOutput(
                distance_m=self.config.max_range_m,
                relative_velocity_mps=0.0,
            )
        _, _, negative_confidence, _, selected = min(candidates)
        return RadarModelOutput(
            distance_m=float(
                np.clip(
                    selected.distance_m,
                    self.config.minimum_forward_distance_m,
                    self.config.max_range_m,
                )
            ),
            relative_velocity_mps=float(selected.relative_velocity_mps),
            track_id=selected.track_id,
            confidence=-negative_confidence,
            source=selected.source,
            truth_object_id=selected.truth_object_id,
            semantic_tag=selected.semantic_tag,
            azimuth_rad=selected.azimuth_rad,
            lateral_extent_m=selected.lateral_extent_m,
            truth_parent_object_id=selected.truth_parent_object_id,
            reflector_id=selected.reflector_id,
            bounce_type=selected.bounce_type,
            bounce_order=selected.bounce_order,
            path_length_m=selected.path_length_m,
            ghost_probability=selected.ghost_probability,
        )

    def step(
        self,
        ideal_targets,
        timestamp_s=None,
        environment=None,
        path_curvature_per_m=0.0,
        multipath_targets=None,
    ):
        """Advance one sensor cycle and return the selected tracked target."""

        environment = (environment or RadarEnvironment()).clamped()
        path_curvature_per_m = float(path_curvature_per_m)
        if not math.isfinite(path_curvature_per_m):
            path_curvature_per_m = 0.0
        self._scan_index += 1
        if timestamp_s is None or self._last_timestamp_s is None:
            dt = self.config.cycle_time_s
        else:
            dt = float(timestamp_s) - self._last_timestamp_s
            if not 0.0 < dt <= 1.0:
                dt = self.config.cycle_time_s
        if timestamp_s is not None:
            self._last_timestamp_s = float(timestamp_s)

        half_fov = math.radians(self.config.horizontal_fov_deg / 2.0)
        targets = [
            target
            for target in ideal_targets
            if (
                math.isfinite(target.distance_m)
                and math.isfinite(target.azimuth_rad)
                and math.isfinite(target.relative_velocity_mps)
                and math.isfinite(target.snr_db)
                and math.isfinite(target.lateral_extent_m)
                and target.lateral_extent_m >= 0.0
                and self.config.minimum_forward_distance_m
                <= target.distance_m
                <= self.config.max_range_m
                and abs(target.azimuth_rad) <= half_fov
            )
        ]
        geometry_targets = [
            target
            for target in (multipath_targets or ())
            if (
                math.isfinite(target.distance_m)
                and math.isfinite(target.azimuth_rad)
                and math.isfinite(target.relative_velocity_mps)
                and math.isfinite(target.snr_db)
                and math.isfinite(target.lateral_extent_m)
                and target.lateral_extent_m >= 0.0
                and self.config.minimum_forward_distance_m
                <= target.distance_m
                <= self.config.max_range_m
                and abs(target.azimuth_rad) <= half_fov
            )
        ]

        self._update_interference()
        detections = []
        dropped_direct = 0
        for target in targets:
            detection = self._measure_target(target, environment)
            if detection is None:
                dropped_direct += 1
            else:
                detections.append(detection)

        if self.config.multipath_mode == "probabilistic":
            self._spawn_ghosts(targets, environment)
            detections.extend(self._update_ghosts(targets, environment, dt))
        elif self.config.multipath_mode == "geometry":
            for target in geometry_targets:
                detection = self._measure_target(
                    target,
                    environment,
                    source="ghost",
                )
                if detection is not None:
                    detections.append(detection)
        else:
            self._ghosts.clear()
        detections.extend(self._create_clutter())

        generated_counts = {
            source: sum(item.source == source for item in detections)
            for source in ("direct", "ghost", "clutter")
        }
        generated_detection_count = len(detections)
        generated_detections = list(detections)
        rejected = []
        if self._detection_filter is not None:
            detections, rejected = self._detection_filter.filter_detections(
                detections,
                timestamp_s=timestamp_s,
                scan_index=self._scan_index,
            )
        self._latency_queue.append((self._scan_index, detections))
        if len(self._latency_queue) <= self.config.latency_scans:
            delivered_source_scan_index = None
            delivered = []
        else:
            delivered_source_scan_index, delivered = (
                self._latency_queue.popleft()
            )

        self._update_tracks(delivered, dt)
        selected = self._select_track(path_curvature_per_m)
        confirmed_count = sum(track.confirmed for track in self._tracks.values())
        self._diagnostics = {
            "profile": self.config.profile_name,
            "config_signature": realistic_radar_config_signature(self.config),
            "scan_index": self._scan_index,
            "ideal_target_count": len(targets),
            "multipath_mode": self.config.multipath_mode,
            "multipath_ideal_target_count": len(geometry_targets),
            "generated_detection_count": generated_detection_count,
            "accepted_detection_count": len(detections),
            "rejected_detection_count": len(rejected),
            "delivered_detection_count": len(delivered),
            "delivered_source_scan_index": delivered_source_scan_index,
            "configured_latency_scans": self.config.latency_scans,
            "direct_detection_count": generated_counts["direct"],
            "ghost_detection_count": generated_counts["ghost"],
            "clutter_detection_count": generated_counts["clutter"],
            "dropped_direct_count": dropped_direct,
            "active_ghost_count": len(self._ghosts),
            "active_track_count": len(self._tracks),
            "confirmed_track_count": confirmed_count,
            "interference_active": self._interference_active,
            "selected_track_id": selected.track_id or None,
            "selected_truth_object_id": (
                selected.truth_object_id if selected.track_id else None
            ),
            "selected_semantic_tag": (
                selected.semantic_tag if selected.track_id else None
            ),
            "selected_source": selected.source,
            "selected_truth_parent_object_id": (
                selected.truth_parent_object_id if selected.track_id else None
            ),
            "selected_reflector_id": (
                selected.reflector_id if selected.track_id else None
            ),
            "selected_bounce_type": selected.bounce_type,
            "selected_bounce_order": selected.bounce_order,
            "selected_path_length_m": selected.path_length_m,
            "selected_ghost_probability": selected.ghost_probability,
            "selected_confidence": selected.confidence,
            "selected_azimuth_deg": math.degrees(selected.azimuth_rad),
            "selected_lateral_extent_m": selected.lateral_extent_m,
            "path_curvature_per_m": float(path_curvature_per_m),
            "environment": asdict(environment),
        }
        if self._capture_debug:
            self._debug_snapshot = {
                "scan_index": self._scan_index,
                "delivered_source_scan_index": delivered_source_scan_index,
                "path_curvature_per_m": float(path_curvature_per_m),
                "ideal_targets": [asdict(target) for target in targets],
                "multipath_ideal_targets": [
                    asdict(target) for target in geometry_targets
                ],
                "generated_detections": [
                    asdict(detection) for detection in generated_detections
                ],
                "accepted_detections": [
                    asdict(detection) for detection in detections
                ],
                "rejected_detections": [
                    asdict(detection) for detection in rejected
                ],
                "delivered_detections": [
                    asdict(detection) for detection in delivered
                ],
                "tracks": [
                    {
                        "track_id": track.track_id,
                        "distance_m": track.distance_m,
                        "azimuth_rad": track.azimuth_rad,
                        "relative_velocity_mps": (
                            track.relative_velocity_mps
                        ),
                        "snr_db": track.snr_db,
                        "source": track.source,
                        "truth_object_id": track.truth_object_id,
                        "semantic_tag": track.semantic_tag,
                        "lateral_extent_m": track.lateral_extent_m,
                        "truth_parent_object_id": (
                            track.truth_parent_object_id
                        ),
                        "reflector_id": track.reflector_id,
                        "bounce_type": track.bounce_type,
                        "bounce_order": track.bounce_order,
                        "path_length_m": track.path_length_m,
                        "ghost_probability": track.ghost_probability,
                        "age": track.age,
                        "hits": track.hits,
                        "misses": track.misses,
                        "confirmed": track.confirmed,
                        "confidence": self._track_confidence(track),
                        "hit_history": list(track.hit_history),
                    }
                    for track in self._tracks.values()
                ],
                "selected": asdict(selected),
            }
        return selected

    def diagnostics(self):
        """Return a shallow copy of the most recent sensor diagnostics."""

        result = self._diagnostics.copy()
        if "environment" in result:
            result["environment"] = result["environment"].copy()
        return result

    def debug_snapshot(self):
        """Return detailed target/detection/track state for validation logs."""

        result = self._debug_snapshot.copy()
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
                item.copy() for item in self._debug_snapshot.get(key, ())
            ]
        result["selected"] = self._debug_snapshot["selected"].copy()
        return result
