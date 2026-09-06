"""Extended-target point emission shared by the live sensor and the exporter.

A real automotive radar reports many CFAR detections per object: one for
each range-Doppler cell the object's surface lights up, each with its own
amplitude fluctuation and, for road users with moving parts, its own
micro-Doppler offset. The grouped target list that the C-Shenron-derived
front end produces collapses all of that into one point per object, which is
the single largest structural difference between this sensor and a real one.

This module restores the per-object point statistics. It is NumPy-only and
works on any dataclass that carries ``distance_m``, ``azimuth_rad``,
``relative_velocity_mps``, ``snr_db``, ``semantic_tag``, ``lateral_extent_m``
and ``radial_extent_m``: the live sensor expands ``RadarDetection`` objects,
the H5 exporter expands plain dicts through ``export_expansion``.

The footprint and micro-Doppler tables are priors. ``calibrate_ghost_profile.py``
measures the real per-object spread and points-per-object from the Radar Ghost
Dataset train split and emits ``point_footprint_scale``,
``micro_doppler_scale`` and ``points_per_object_mean`` to replace them.
"""

from dataclasses import replace
import math


MAX_POINTS_PER_OBJECT = 64


def class_id_for_semantic_tag(semantic_tag):
    """Map CARLA semantic tags to Radar Ghost Dataset class ids."""

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


# Physical footprint (radial depth m, lateral width m) per RGD class id.
CLASS_FOOTPRINT_M = {
    1: (0.45, 0.50),   # pedestrian
    2: (1.80, 0.70),   # cyclist / rider
    3: (4.40, 1.80),   # car
    4: (9.00, 2.50),   # truck / bus
    5: (2.20, 0.90),   # motorcycle
}

# Micro-Doppler spread amplitude (m/s) per class. Pedestrian limbs swing the
# per-point radial velocity by roughly +/-0.6-1.2 m/s around the torso mean;
# vehicle superstructure/wheel returns vary only slightly. Ghost points
# inherit their parent's spread, exactly as path physics dictates.
MICRO_DOPPLER_AMPLITUDE = {
    1: (0.55, 0.65),
    2: (0.35, 0.45),
    3: (0.06, 0.10),
    4: (0.05, 0.09),
    5: (0.15, 0.25),
}

# Swerling-like per-point amplitude fluctuation, in dB. Matches the
# exp(N(0, 0.23)) multiplicative term the exporter has always used.
POINT_SNR_FLUCTUATION_DB = 2.0
MICRO_DOPPLER_NOISE_MPS = 0.12


def _quantize(value, step):
    if step <= 0.0:
        return float(value)
    return float(round(float(value) / step) * step)


def object_footprint_m(detection, footprint_scale=1.0):
    """Radial depth and lateral width to spread points over.

    Measured extents from the semantic scan win when the front end supplied
    them; otherwise the class prior applies. Extents are half-widths in the
    front end, so the lateral one is doubled.
    """

    class_id = class_id_for_semantic_tag(getattr(detection, "semantic_tag", 14))
    depth, width = CLASS_FOOTPRINT_M.get(class_id, (2.0, 1.0))
    radial = float(getattr(detection, "radial_extent_m", 0.0) or 0.0)
    lateral = float(getattr(detection, "lateral_extent_m", 0.0) or 0.0)
    if radial > 0.2:
        depth = radial
    if lateral > 0.1:
        width = 2.0 * lateral
    scale = max(0.0, float(footprint_scale))
    return depth * scale, width * scale


def expand_detection(
    detection,
    rng,
    mean_points,
    range_resolution_m,
    doppler_resolution_mps,
    azimuth_resolution_rad,
    minimum_range_m,
    maximum_range_m,
    footprint_scale=1.0,
    micro_doppler_scale=1.0,
):
    """Expand one grouped detection into CFAR-like surface points.

    ``azimuth_resolution_rad`` is a callable of azimuth so the sensor's
    boresight-to-edge resolution law applies. Label and truth fields are
    inherited unchanged: every emitted point is exactly as real or as ghostly
    as its cluster, which is what makes counterfactual ghost removal exact.
    """

    depth_m, width_m = object_footprint_m(detection, footprint_scale)
    class_id = class_id_for_semantic_tag(getattr(detection, "semantic_tag", 14))
    low, high = MICRO_DOPPLER_AMPLITUDE.get(class_id, (0.05, 0.10))
    micro_scale = max(0.0, float(micro_doppler_scale))
    micro_amplitude = (low + rng.random() * (high - low)) * micro_scale
    micro_phase = rng.random() * 2.0 * math.pi

    count = int(rng.poisson(max(float(mean_points), 1.0e-6)))
    count = max(1, min(count, MAX_POINTS_PER_OBJECT))

    distance = float(detection.distance_m)
    azimuth = float(detection.azimuth_rad)
    y_sensor = distance * math.sin(azimuth)
    bulk_velocity = float(detection.relative_velocity_mps)
    points = []
    for index in range(count):
        delta_r = (rng.random() - 0.5) * depth_m
        delta_y = (rng.random() - 0.5) * width_m
        along = max(distance + delta_r, 0.5)
        new_x = along * math.cos(azimuth)
        new_y = y_sensor + delta_y
        new_range = max(math.hypot(new_x, new_y), 1.0e-3)
        new_azimuth = math.atan2(new_y, max(new_x, 1.0e-3))
        # Radar-equation trend across the object depth plus per-point
        # fluctuation; 40 log10 because power falls with the fourth power.
        snr_db = (
            float(detection.snr_db)
            + 40.0 * math.log10(max(distance, 1.0e-3) / new_range)
            + rng.normal(0.0, POINT_SNR_FLUCTUATION_DB)
        )
        micro_doppler = (
            micro_amplitude
            * math.sin(micro_phase + 2.0 * math.pi * index / 8.0)
            + rng.normal(0.0, MICRO_DOPPLER_NOISE_MPS * micro_scale)
        )
        quantized_range = _quantize(new_range, range_resolution_m)
        quantized_range = min(maximum_range_m, max(minimum_range_m, quantized_range))
        points.append(
            replace(
                detection,
                distance_m=quantized_range,
                azimuth_rad=_quantize(
                    new_azimuth,
                    float(azimuth_resolution_rad(new_azimuth)),
                ),
                relative_velocity_mps=_quantize(
                    bulk_velocity + micro_doppler,
                    doppler_resolution_mps,
                ),
                snr_db=float(snr_db),
            )
        )
    return points
