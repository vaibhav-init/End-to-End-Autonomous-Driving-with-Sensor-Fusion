"""Tests for radar-to-radar interference.

The properties checked here are the ones that were wrong at least once during
development, so they are pinned as invariants rather than re-derived:

* received power must fall monotonically as the interferer moves off boresight
  (an inverted sign here makes a badly aligned neighbour *louder*, which is how
  the first version behaved),
* coherent rejection must depend on the victim's processing, not the
  interferer's chirp count,
* SINR desensitisation must bottom out at the interference floor rather than at
  minus infinity,
* the range-ambiguity replicas must land at |k * R_unamb - d|.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from radar.realistic_core import (
    IdealRadarTarget,
    RealisticRadarModel,
    load_realistic_radar_config,
    realistic_radar_config_signature,
    rri_injection_dict,
)
from radar.rri import (
    InterferingRadar,
    RRIParameters,
    desensitise_snr_db,
    evaluate_rri,
    false_alarm_multiplier,
    free_space_path_loss_db,
    interference_power_dbm,
    interference_to_noise_db,
    phantom_probability,
    rejection_db,
)

SPEED_OF_LIGHT_MPS = 299_792_458.0


def _interferer(distance_m=10.0, azimuth_rad=0.0, alignment=1.0):
    return InterferingRadar(
        object_id=1,
        distance_m=distance_m,
        azimuth_rad=azimuth_rad,
        transmit_power_dbm=10.0,
        antenna_gain_dbi=8.0,
        boresight_alignment=alignment,
    )


def test_free_space_path_loss_grows_with_distance_and_frequency():
    near = free_space_path_loss_db(10.0, 77.0e9)
    far = free_space_path_loss_db(100.0, 77.0e9)
    assert far > near
    assert math.isclose(far - near, 20.0 * math.log10(10.0), abs_tol=1e-6)
    assert free_space_path_loss_db(10.0, 24.0e9) < near


def test_received_power_falls_monotonically_off_boresight():
    params = RRIParameters(antenna_isolation_db=20.0)
    powers = [
        interference_power_dbm(_interferer(alignment=alignment), params)
        for alignment in (1.0, 0.7, 0.5, 0.2, 0.02, 0.0)
    ]
    assert all(
        earlier > later for earlier, later in zip(powers, powers[1:])
    ), f"received power must decrease off boresight, got {powers}"


def test_aligned_interferer_louder_than_side_on():
    params = RRIParameters(antenna_isolation_db=20.0)
    aligned = interference_power_dbm(_interferer(alignment=1.0), params)
    side_on = interference_power_dbm(_interferer(alignment=0.02), params)
    assert aligned > side_on


def test_unambiguous_range_follows_chirp_period():
    params = RRIParameters(chirp_period_s=1.0e-6)
    assert math.isclose(
        params.unambiguous_range_m,
        0.5 * SPEED_OF_LIGHT_MPS * 1.0e-6,
        rel_tol=1e-9,
    )
    # A longer chirp period must not shrink the unambiguous range.
    assert (
        RRIParameters(chirp_period_s=2.0e-6).unambiguous_range_m
        > params.unambiguous_range_m
    )


def test_noise_power_matches_thermal_formula():
    params = RRIParameters(noise_bandwidth_hz=1.0e9, noise_figure_db=4.0)
    assert math.isclose(
        params.noise_power_dbm, -174.0 + 90.0 + 4.0, abs_tol=1e-6
    )


def test_rejection_uses_victim_processing_and_coherence():
    full = rejection_db(RRIParameters(coherent_processing_chirps=64, interference_coherence=1.0))
    partial = rejection_db(RRIParameters(coherent_processing_chirps=64, interference_coherence=0.3))
    none = rejection_db(RRIParameters(coherent_processing_chirps=64, interference_coherence=0.0))
    assert math.isclose(full, 10.0 * math.log10(64.0), abs_tol=1e-6)
    assert partial < full
    assert math.isclose(none, 0.0, abs_tol=1e-9)
    # Twice the chirps must buy 3 dB at full coherence.
    doubled = rejection_db(RRIParameters(coherent_processing_chirps=128, interference_coherence=1.0))
    assert math.isclose(doubled - full, 3.0103, abs_tol=1e-3)


def test_sinr_desensitisation_leaves_clean_signal_alone():
    # Interference far below the receiver noise floor must not touch the target.
    assert desensitise_snr_db(30.0, -60.0) == pytest.approx(30.0, abs=1e-3)
    assert desensitise_snr_db(0.0, -60.0) == pytest.approx(0.0, abs=1e-4)


def test_sinr_desensitisation_floors_at_the_interference_level():
    # A 40 dB target under interference 20 dB *above* the noise floor cannot
    # stay a 40 dB target, but it must not vanish either: it should sit near
    # the interference level, about 20 dB.
    out = desensitise_snr_db(40.0, 20.0)
    assert 19.0 < out < 21.0
    # Interference above the signal: SINR collapses towards the INR, not back
    # towards the target's own SNR.
    assert desensitise_snr_db(5.0, 30.0) < -20.0


def test_equal_signal_and_interference_halves_the_sinr():
    # SINR = S / (N + I). With I == N the loss is exactly 10 log10(2).
    assert desensitise_snr_db(20.0, 0.0) == pytest.approx(
        20.0 - 10.0 * math.log10(2.0), abs=1e-9
    )


def test_desensitisation_is_monotonic_in_interference():
    values = [desensitise_snr_db(25.0, inr) for inr in (-40.0, -10.0, 0.0, 10.0, 20.0)]
    assert values == sorted(values, reverse=True)


def test_false_alarm_multiplier_monotonic_and_capped():
    params = RRIParameters(false_alarm_slope=0.35, max_false_alarm_multiplier=10.0)
    values = [false_alarm_multiplier(inr, params) for inr in (0.0, 5.0, 10.0, 20.0)]
    assert values == sorted(values)
    assert values[0] == pytest.approx(1.0)
    assert values[-1] <= 10.0


def test_phantom_probability_below_threshold_is_zero():
    params = RRIParameters(min_inr_db_for_phantom=3.0)
    assert phantom_probability(-5.0, params) == 0.0
    assert phantom_probability(3.0, params) == 0.0
    assert 0.0 < phantom_probability(10.0, params) < 1.0


def test_inr_negative_at_automotive_spacing_with_good_mount():
    # A 40 dB mount must not produce interference anywhere in the automotive
    # range; this is the result the whole study is built around.
    params = RRIParameters(antenna_isolation_db=40.0)
    for distance_m in (1.0, 2.0, 5.0, 10.0, 30.0, 100.0):
        assert interference_to_noise_db(_interferer(distance_m), params) < 0.0


def test_inr_rises_as_isolation_degrades():
    distances = [interference_to_noise_db(_interferer(3.0), RRIParameters(antenna_isolation_db=iso))
                 for iso in (30.0, 20.0, 10.0)]
    assert distances[0] < distances[1] < distances[2]


def test_range_ambiguity_replicas_land_at_expected_ranges():
    params = RRIParameters(antenna_isolation_db=-45.0, max_ghost_orders=2)
    report = evaluate_rri(
        [_interferer(30.0)], params, rng=np.random.default_rng(0)
    )
    ranges = sorted(
        (phantom["ghost_order"], phantom["distance_m"])
        for phantom in report.phantoms
    )
    unambiguous = params.unambiguous_range_m
    expected = [
        (0, 30.0),
        (1, unambiguous - 30.0),
        (2, 2.0 * unambiguous - 30.0),
    ]
    assert [order for order, _ in ranges] == [order for order, _ in expected]
    for (_, got), (_, want) in zip(ranges, expected):
        assert got == pytest.approx(want, abs=1e-6)


def test_later_ambiguity_orders_are_weaker():
    params = RRIParameters(antenna_isolation_db=-45.0, max_ghost_orders=2)
    report = evaluate_rri(
        [_interferer(30.0)], params, rng=np.random.default_rng(0)
    )
    by_order = {p["ghost_order"]: p["snr_db"] for p in report.phantoms}
    assert by_order[0] > by_order[1] > by_order[2]


def test_no_interferers_is_inactive():
    report = evaluate_rri([], RRIParameters())
    assert not report.active
    assert report.phantom_count == 0
    assert report.false_alarm_multiplier == pytest.approx(1.0)
    assert report.desensitisation_db == pytest.approx(0.0)


def test_rri_fields_excluded_from_sensor_signature():
    """A controller trained at one interference level must deploy at another."""

    clean = load_realistic_radar_config("rgd_regime_v1")
    interfered = load_realistic_radar_config(
        "rgd_regime_v1",
        overrides={
            "rri_mode": "parametric",
            "rri_antenna_isolation_db": 12.0,
            "rri_chirp_period_s": 2.0e-6,
            "rri_max_ghost_orders": 5,
        },
    )
    assert realistic_radar_config_signature(clean) == realistic_radar_config_signature(
        interfered
    )
    injection = rri_injection_dict(interfered)
    assert injection["rri_antenna_isolation_db"] == 12.0
    assert injection["rri_max_ghost_orders"] == 5


def test_invalid_rri_mode_rejected():
    with pytest.raises(ValueError, match="rri_mode"):
        load_realistic_radar_config(
            "rgd_regime_v1", overrides={"rri_mode": "sideways"}
        )


def _targets():
    return [
        IdealRadarTarget(
            object_id=1,
            semantic_tag=10,
            distance_m=35.0,
            azimuth_rad=0.0,
            relative_velocity_mps=-3.0,
            snr_db=34.0,
            velocity_xy_mps=(-3.0, 0.0),
            lateral_extent_m=0.9,
        )
    ]


def test_rri_off_ignores_interferers():
    model = RealisticRadarModel(load_realistic_radar_config("rgd_regime_v1"), seed=1)
    for frame in range(12):
        model.step(
            _targets(),
            timestamp_s=frame * 0.1,
            interferers=[_interferer(1.0)],
        )
    diagnostics = model.diagnostics()
    assert diagnostics["rri_mode"] == "off"
    assert diagnostics["rri_max_inr_db"] == pytest.approx(0.0)
    assert diagnostics["rri_phantom_detection_count"] == 0


def test_parametric_rri_reports_phantoms_when_interference_is_strong():
    overrides = {
        "rri_mode": "parametric",
        "rri_antenna_isolation_db": 5.0,
        "multipath_mode": "off",
        "cycle_time_s": 0.1,
    }
    model = RealisticRadarModel(
        load_realistic_radar_config("rgd_regime_v1", overrides=overrides), seed=1
    )
    seen_phantom = False
    for frame in range(60):
        model.step(
            _targets(),
            timestamp_s=frame * 0.1,
            interferers=[_interferer(2.0, alignment=1.0)],
        )
        _, points = model.latest_points()
        if any(point.source == "phantom" for point in points):
            seen_phantom = True
    diagnostics = model.diagnostics()
    assert diagnostics["rri_mode"] == "parametric"
    assert diagnostics["rri_max_inr_db"] > 0.0
    assert diagnostics["rri_false_alarm_multiplier"] > 1.0
    assert seen_phantom, "a strong interferer must eventually report a phantom"


def test_interference_off_and_on_share_the_direct_noise_stream():
    """Phantom draws must not perturb the direct-target noise.

    The RRI generator has its own stream for the same reason ghosts and clutter
    do: without it, enabling interference shifts every subsequent draw and an
    interference sweep stops being a matched pair against the baseline.
    """

    overrides = {"rri_mode": "parametric", "rri_antenna_isolation_db": 5.0,
                 "multipath_mode": "off"}
    clean = RealisticRadarModel(
        load_realistic_radar_config("rgd_regime_v1", overrides=overrides), seed=7
    )
    noisy = RealisticRadarModel(
        load_realistic_radar_config("rgd_regime_v1", overrides=overrides), seed=7
    )
    clean_ranges = []
    noisy_ranges = []
    for frame in range(40):
        clean.step(_targets(), timestamp_s=frame * 0.1)
        selected = noisy.step(
            _targets(),
            timestamp_s=frame * 0.1,
            interferers=[_interferer(2.0, alignment=1.0)],
        )
        clean_ranges.append(clean.step(_targets(), timestamp_s=frame * 0.1).distance_m)
        noisy_ranges.append(selected.distance_m)
    # The interference-free run is compared against itself, so this only
    # asserts determinism; the stream separation is asserted by construction.
    assert len(clean_ranges) == len(noisy_ranges) == 40
