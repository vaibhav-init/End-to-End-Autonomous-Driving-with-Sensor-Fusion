"""Shared physical feature contract for real and CARLA target lists.

Schema v2 notes (zero-shot sim-to-real alignment):

v1 encoded amplitude on an absolute scale (signed_log_amplitude), which does
not transfer between domains: CARLA amplitudes are an SNR-derived proxy while
RGD amplitudes are measured echo power, so the two live on different absolute
scales even when the underlying scene physics agree. v2 therefore adds three
frame-relative statistics computed identically in every domain:

- ``relative_log_amplitude``: log amplitude minus the frame median log
  amplitude. Preserves the discriminative ordering (multipath returns are
  typically weaker than their parent direct return) while cancelling any
  constant gain mismatch between sensors.
- ``doppler_cluster_residual``: distance between a point's radial velocity
  and the median radial velocity of its spatial cluster. Multipath returns
  inherit path-modified Doppler, while direct returns from one rigid object
  share a tight velocity core in both domains.
- ``local_density_ratio``: neighbour count inside a fixed range/azimuth gate
  normalised by the frame mean. CFAR point clouds spread many points across
  extended-object surfaces while isolated artifact points have few
  neighbours; this holds for both real CFAR output and CFAR-emulating
  synthetic export.
"""

import math

import numpy as np


FEATURE_SCHEMA_VERSION = "radar_ghost_physical_v2"
FEATURE_NAMES = (
    "x_sensor_over_100m",
    "y_sensor_over_100m",
    "range_over_100m",
    "sin_azimuth",
    "cos_azimuth",
    "radial_velocity_over_40mps",
    "age_over_0_5s",
    "relative_log_amplitude",
    "log_range_compensated_amplitude",
    "doppler_cluster_residual_over_5mps",
    "local_density_ratio_over_4",
)

# Gates used by the frame-relative statistics. Fixed physical constants, not
# fitted parameters, so they are identical for real and synthetic data.
_DENSITY_RANGE_GATE_M = 1.5
_DENSITY_AZIMUTH_GATE_RAD = math.radians(2.0)
_CLUSTER_RANGE_BIN_M = 2.0
_CLUSTER_AZIMUTH_BIN_RAD = math.radians(3.0)


def _frame_relative_log_amplitude(amplitude):
    """Per-frame median-centred log amplitude, robust to gain mismatch."""

    amplitudes = np.asarray(amplitude, dtype=np.float64).reshape(-1)
    positive = np.maximum(amplitudes, 1.0e-6)
    log_amplitude = np.log(positive)
    median = float(np.median(log_amplitude)) if log_amplitude.size else 0.0
    return np.clip(log_amplitude - median, -10.0, 10.0) / 4.0


def frame_context_statistics(range_m, azimuth_rad, radial_velocity_mps, amplitude):
    """Compute the v2 frame-relative statistics for one complete scan.

    Computed once per frame over *all* points of the scan and then indexed
    per training sample, so the statistics always describe the full point
    cloud exactly like a real sensor frame would. Both the real preparation
    path and the online runtime must call this with the same gates.
    """

    ranges = np.asarray(range_m, dtype=np.float64).reshape(-1)
    azimuth = np.asarray(azimuth_rad, dtype=np.float64).reshape(-1)
    velocity = np.asarray(radial_velocity_mps, dtype=np.float64).reshape(-1)
    amplitude = np.asarray(amplitude, dtype=np.float64).reshape(-1)
    count = ranges.size

    relative_log_amplitude = _frame_relative_log_amplitude(amplitude)

    if count == 0:
        empty = np.zeros(0, dtype=np.float32)
        return (
            relative_log_amplitude,
            empty.copy(),
            empty.copy(),
        )

    # Cluster residual: compare each point against its coarse range/azimuth
    # cell's mean Doppler. Single-point cells fall back to the global frame
    # median so isolated points get a large residual rather than zero.
    range_bins = np.floor(ranges / _CLUSTER_RANGE_BIN_M).astype(np.int64)
    azimuth_bins = np.floor(
        (azimuth + math.pi) / _CLUSTER_AZIMUTH_BIN_RAD
    ).astype(np.int64)
    velocity_finite = np.where(np.isfinite(velocity), velocity, 0.0)
    global_median = float(np.median(velocity_finite))
    keys = np.stack((range_bins, azimuth_bins), axis=1)
    unique_keys, inverse, counts = np.unique(
        keys,
        axis=0,
        return_inverse=True,
        return_counts=True,
    )
    sums = np.bincount(inverse, weights=velocity_finite)
    cell_values = np.full(unique_keys.shape[0], global_median, dtype=np.float64)
    multi = counts > 2
    cell_values[multi] = sums[multi] / counts[multi]
    doppler_residual = np.clip(
        np.abs(velocity_finite - cell_values[inverse]),
        0.0,
        25.0,
    ) / 5.0

    # Local density: neighbour count within a fixed range/azimuth gate,
    # normalised by (4 x) the frame-mean neighbour count.
    dr = ranges[:, None] - ranges[None, :]
    da = azimuth[:, None] - azimuth[None, :]
    near = (np.abs(dr) <= _DENSITY_RANGE_GATE_M) & (
        np.abs(da) <= _DENSITY_AZIMUTH_GATE_RAD
    )
    np.fill_diagonal(near, False)
    neighbour_counts = near.sum(axis=1).astype(np.float64)
    mean_neighbours = float(neighbour_counts.mean())
    if mean_neighbours <= 0.0:
        density_ratio = np.zeros(count, dtype=np.float64)
    else:
        density_ratio = np.clip(
            neighbour_counts / (4.0 * mean_neighbours),
            0.0,
            4.0,
        )

    return (
        relative_log_amplitude.astype(np.float32),
        doppler_residual.astype(np.float32),
        density_ratio.astype(np.float32),
    )


def physical_features(
    range_m,
    azimuth_rad,
    radial_velocity_mps,
    amplitude,
    age_s,
    relative_log_amplitude=None,
    doppler_cluster_residual=None,
    local_density_ratio=None,
):
    """Create domain-shared features without dataset-fitted normalization.

    The three v2 statistics default to zeros so legacy single-target callers
    keep working; batch callers must pass values from
    :func:`frame_context_statistics` computed over the full scan.
    """

    ranges = np.asarray(range_m, dtype=np.float32)
    azimuth = np.asarray(azimuth_rad, dtype=np.float32)
    velocity = np.asarray(radial_velocity_mps, dtype=np.float32)
    amplitudes = np.asarray(amplitude, dtype=np.float32)
    age = np.asarray(age_s, dtype=np.float32)
    if relative_log_amplitude is None:
        relative_log_amplitude = np.zeros_like(amplitudes)
    if doppler_cluster_residual is None:
        doppler_cluster_residual = np.zeros_like(amplitudes)
    if local_density_ratio is None:
        local_density_ratio = np.zeros_like(amplitudes)
    range_compensated = (
        np.log1p(np.maximum(amplitudes, 0.0).astype(np.float64) * ranges.astype(np.float64) ** 2)
        / 10.0
    )
    return np.stack(
        (
            ranges * np.cos(azimuth) / 100.0,
            ranges * np.sin(azimuth) / 100.0,
            ranges / 100.0,
            np.sin(azimuth),
            np.cos(azimuth),
            velocity / 40.0,
            np.clip(age / 0.5, 0.0, 4.0),
            np.asarray(relative_log_amplitude, dtype=np.float32),
            range_compensated.astype(np.float32),
            np.asarray(doppler_cluster_residual, dtype=np.float32),
            np.asarray(local_density_ratio, dtype=np.float32),
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
