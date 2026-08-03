"""Shared physical feature contract for real and CARLA target lists."""

import numpy as np


FEATURE_SCHEMA_VERSION = "radar_ghost_physical_v1"
FEATURE_NAMES = (
    "x_sensor_over_100m",
    "y_sensor_over_100m",
    "range_over_100m",
    "sin_azimuth",
    "cos_azimuth",
    "radial_velocity_over_40mps",
    "signed_log_amplitude_over_10",
    "age_over_0_5s",
)


def physical_features(range_m, azimuth_rad, radial_velocity_mps, amplitude, age_s):
    """Create domain-shared features without dataset-fitted normalization."""

    ranges = np.asarray(range_m, dtype=np.float32)
    azimuth = np.asarray(azimuth_rad, dtype=np.float32)
    velocity = np.asarray(radial_velocity_mps, dtype=np.float32)
    amplitudes = np.asarray(amplitude, dtype=np.float32)
    age = np.asarray(age_s, dtype=np.float32)
    signed_log_amplitude = np.sign(amplitudes) * np.log1p(np.abs(amplitudes))
    return np.stack(
        (
            ranges * np.cos(azimuth) / 100.0,
            ranges * np.sin(azimuth) / 100.0,
            ranges / 100.0,
            np.sin(azimuth),
            np.cos(azimuth),
            velocity / 40.0,
            signed_log_amplitude / 10.0,
            np.clip(age / 0.5, 0.0, 4.0),
        ),
        axis=-1,
    ).astype(np.float32, copy=False)


def snr_db_to_amplitude(snr_db):
    """Map target-list SNR to a stable positive proxy for echo amplitude."""

    snr = np.asarray(snr_db, dtype=np.float32)
    return np.power(10.0, np.clip(snr, -60.0, 80.0) / 20.0).astype(
        np.float32,
        copy=False,
    )
