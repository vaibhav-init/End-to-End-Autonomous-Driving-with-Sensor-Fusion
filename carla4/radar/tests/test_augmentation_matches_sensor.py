"""Augmentation jitter must match the sensor model's point statistics.

The augmentation in ``ghost_detection/dataset.py`` used to carry its own
hard-coded jitter: 0.08 in log-amplitude space and 0.03 m/s in Doppler. The
sensor emits N(0, 2.0) dB of amplitude fluctuation and a per-point Doppler
spread of 0.44 m/s for a pedestrian, so augmentation was smoothing away roughly
3x on amplitude and 20x on Doppler -- precisely the structure a ghost classifier
has to learn.

These tests pin the two together. If someone changes the sensor's point
statistics, these fail rather than letting the augmentation quietly disagree
again.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from radar.extended_target import (
    MICRO_DOPPLER_AMPLITUDE,
    MICRO_DOPPLER_NOISE_MPS,
    POINT_SNR_FLUCTUATION_DB,
    amplitude_jitter_sigma,
    micro_doppler_rms_mps,
)
from radar.ghost_detection.dataset import (
    _MICRO_DOPPLER_SIGMA_BY_CLASS,
    micro_doppler_sigma_for,
)


def test_micro_doppler_rms_matches_the_emitted_points():
    """The analytic RMS must agree with points the sensor actually emits."""

    from dataclasses import replace

    import numpy as np

    from radar.extended_target import expand_detection
    from radar.realistic_core import RadarDetection

    detection = RadarDetection(
        distance_m=30.0,
        azimuth_rad=0.1,
        relative_velocity_mps=5.0,
        snr_db=30.0,
        source="direct",
        truth_object_id=1,
        semantic_tag=12,  # pedestrian -> class 1
        lateral_extent_m=0.9,
    )

    rng = np.random.default_rng(0)
    for class_id, semantic_tag in ((1, 12), (2, 13), (5, 10)):
        points = []
        for _ in range(400):
            expanded = expand_detection(
                replace(detection, semantic_tag=semantic_tag),
                rng,
                mean_points=8.0,
                range_resolution_m=0.15,
                doppler_resolution_mps=0.087,
                azimuth_resolution_rad=lambda angle: math.radians(1.8),
                minimum_range_m=0.15,
                maximum_range_m=153.0,
            )
            points.extend(p.relative_velocity_mps for p in expanded)
        measured = float(np.std(points))
        analytic = micro_doppler_rms_mps(class_id)
        # Quantisation to the Doppler grid suppresses measured spread slightly,
        # so allow a little slack rather than demanding an exact match.
        assert measured == pytest.approx(analytic, rel=0.35), (
            f"class {class_id}: emitted {measured:.4f} m/s vs analytic "
            f"{analytic:.4f} m/s"
        )


def test_pedestrian_spread_is_far_larger_than_the_old_jitter():
    """Guard the specific regression: 0.03 m/s was ~20x too small."""

    assert micro_doppler_rms_mps(1) > 0.30
    assert micro_doppler_rms_mps(1) > 10 * 0.03


def test_larger_targets_spread_faster_than_smaller_ones():
    assert micro_doppler_rms_mps(1) > micro_doppler_rms_mps(2)
    assert micro_doppler_rms_mps(2) > micro_doppler_rms_mps(3)


def test_amplitude_jitter_sigma_reproduces_the_sensor_fluctuation_in_db():
    sigma = amplitude_jitter_sigma()
    assert sigma == pytest.approx(POINT_SNR_FLUCTUATION_DB * math.log(10.0) / 20.0)
    # A lognormal multiplier of exp(N(0, sigma)) is a dB perturbation of
    # 20*sigma/ln(10); check it lands on POINT_SNR_FLUCTUATION_DB.
    assert 20.0 * sigma / math.log(10.0) == pytest.approx(POINT_SNR_FLUCTUATION_DB)
    assert sigma > 0.08, "the old 0.08 was about 3x too small"


def test_background_points_do_not_get_pedestrian_spread():
    """Class 0 must not inherit the pedestrian fallback."""

    sigma = micro_doppler_sigma_for(np.array([0, 1, 2, 3, -1, 99]))
    assert sigma[0] == pytest.approx(micro_doppler_rms_mps(3))
    assert sigma[1] == pytest.approx(micro_doppler_rms_mps(1))
    assert sigma[2] == pytest.approx(micro_doppler_rms_mps(2))
    # Out-of-range and negative ids fall back to the background value rather
    # than indexing out of bounds or picking a pedestrian.
    assert sigma[4] == pytest.approx(micro_doppler_rms_mps(3))
    assert sigma[5] == pytest.approx(micro_doppler_rms_mps(3))


def test_class_table_is_ordered_the_way_the_labels_are():
    assert len(_MICRO_DOPPLER_SIGMA_BY_CLASS) == 6
    values = _MICRO_DOPPLER_SIGMA_BY_CLASS
    assert values[0] < values[1], "background must be quieter than a pedestrian"
    assert values[1] > values[2] > values[3]


def test_micro_doppler_scale_zero_silences_the_spread():
    assert micro_doppler_rms_mps(1, micro_doppler_scale=0.0) == pytest.approx(0.0)


def test_unknown_class_falls_back_to_pedestrian_in_the_model_helper():
    """Documented behaviour of the sensor helper, kept distinct from the
    dataset's background mapping."""

    assert micro_doppler_rms_mps(99) == pytest.approx(micro_doppler_rms_mps(1))
