"""CFAR-emulating extended-target point expansion for synthetic exports.

Real automotive radars emit many detections per extended object: each
range-Doppler map cell crossing the object passes CFAR, amplitude fluctuates
cell to cell, and per-point radial velocity spreads around the object's bulk
motion (micro-Doppler). A grouped target list collapses all of that into one
point per object per scan — the dominant reason synthetic point clouds fail
to transfer to models that must work on real CFAR output.

This module restores those statistics deterministically at export time. It is
deliberately NumPy-only (no CARLA, no h5py) so it can be unit-tested and
reused by any collector.
"""

import math

import numpy as np

# The footprint and micro-Doppler tables live with the live sensor's
# extended-target emission so the exporter and the sensor agree on them.
from ..extended_target import (  # noqa: F401  (re-exported names)
    CLASS_FOOTPRINT_M as _CLASS_FOOTPRINT_M,
    MICRO_DOPPLER_AMPLITUDE as _MICRO_DOPPLER_AMPLITUDE,
    class_id_for_semantic_tag,
)


# RGD-style resolution grid applied to every expanded point.
RANGE_RESOLUTION_M = 0.15
AZIMUTH_RESOLUTION_RAD = math.radians(1.8)
DOPPLER_RESOLUTION_MPS = 0.087


def _quantize(value, step):
    return round(float(value) / step) * step


def expand_detection_points(detection, rng, mean_points, snr_to_amplitude):
    """Expand one grouped detection into CFAR-like surface points.

    ``detection`` is a dict with at least ``distance_m``, ``azimuth_rad``,
    ``relative_velocity_mps``, ``snr_db`` and ``semantic_tag`` keys plus any
    label metadata. Returns 1..96 copies whose geometry, amplitude and
    Doppler follow CFAR scattering statistics; label metadata is inherited
    unchanged so direct and ghost families expand identically.

    ``snr_to_amplitude`` converts SNR dB to the export amplitude scale (the
    caller injects its own convention; expanded points get a modified SNR so
    downstream conversion stays unchanged).
    """

    class_id = class_id_for_semantic_tag(detection.get("semantic_tag", 14))
    depth_m, width_m = _CLASS_FOOTPRINT_M.get(class_id, (2.0, 1.0))
    lo, hi = _MICRO_DOPPLER_AMPLITUDE.get(class_id, (0.05, 0.10))
    micro_amplitude = lo + rng.random() * (hi - lo)
    micro_phase = rng.random() * 2.0 * math.pi

    distance = float(detection["distance_m"])
    azimuth = float(detection["azimuth_rad"])
    x_sensor = distance * math.cos(azimuth)
    y_sensor = distance * math.sin(azimuth)
    parent_amp = float(snr_to_amplitude(detection["snr_db"]))
    bulk_vr = float(detection["relative_velocity_mps"])

    count = int(rng.poisson(max(mean_points, 1e-6)))
    count = max(1, min(count, 96))
    expanded = []
    for index in range(count):
        delta_r = (rng.random() - 0.5) * depth_m
        delta_y = (rng.random() - 0.5) * width_m
        new_distance = max(distance + delta_r, 0.5)
        new_x = new_distance * math.cos(azimuth)
        new_y = y_sensor + delta_y
        if abs(new_x) < 1.0e-3 and abs(new_y) < 1.0e-3:
            new_azimuth = azimuth
        else:
            new_azimuth = math.atan2(new_y, max(new_x, 1.0e-3))
        # Radar-equation trend across the object depth plus Swerling-like
        # fluctuation; amplitude stays strictly positive.
        amplitude = (
            parent_amp
            * (distance / new_distance) ** 2
            * float(np.exp(rng.normal(0.0, 0.23)))
        )
        micro_doppler = (
            micro_amplitude * math.sin(micro_phase + 2.0 * math.pi * index / 8.0)
            + rng.normal(0.0, 0.12)
        )
        point = dict(detection)
        point["distance_m"] = _quantize(new_distance, RANGE_RESOLUTION_M)
        point["azimuth_rad"] = _quantize(new_azimuth, AZIMUTH_RESOLUTION_RAD)
        point["relative_velocity_mps"] = _quantize(
            bulk_vr + micro_doppler,
            DOPPLER_RESOLUTION_MPS,
        )
        # Overwrite SNR so exported amplitude carries per-point fluctuation;
        # the caller's snr->amplitude conversion applies downstream as before.
        point["snr_db"] = 20.0 * math.log10(max(amplitude, 1.0e-6))
        expanded.append(point)
    return expanded