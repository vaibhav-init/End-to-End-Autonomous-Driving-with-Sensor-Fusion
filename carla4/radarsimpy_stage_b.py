"""Stage B of the RadarSimPy calibration: ray-tracing reference.

Requires the RadarSimPy pre-built module importable (free tier is
sufficient).  For each canonical scene of
``carla4.radar.radarsimpy_ghost_calibration`` it

1. builds equivalent mesh targets (boxes for the road user and reflectors),
2. runs the FMCW transceiver matched to ``rgd_regime_v1``
   (77 -> 77.9 GHz, 100 us chirp, 6 MHz sampling, 16 chirps, 8 Rx half-lambda array),
3. performs range-Doppler processing, CA-CFAR peak picking and MUSIC DoA,
4. reports detected (range, azimuth, velocity, SNR) alongside the stage-A
   analytic variants ``mix_half`` and ``full_image``.

Deltas feed the ``multipath_*`` amplitude priors (reflection loss, second/
third-order losses) after a few repeated runs.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

sys_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "..")
import sys
sys.path.insert(0, sys_path)

from carla4.radarsimpy_ghost_calibration import (  # noqa: E402
    SCENES,
    Scene,
    analytic_gr,
)

CARRIER_MIN_HZ = 77.0e9
CARRIER_BW_HZ = 0.9e9            # -> 0.167 m range resolution
CHIRP_T_S = 100.0e-6             # sweep rate 9 MHz/us
FS_HZ = 6.0e6                    # beat sampling -> max range ~100 m
PULSES = 16
PRP_S = 150.0e-6
TX_POWER_DBM = 15.0
NOISE_FIGURE_DB = 8.0
N_RX = 1                         # free tier limit: single receiver channel
RX_SPACING_LAMBDA = 0.5
CFAR_ON = 4                      # CA-CFAR 2D parameters
CFAR_OFF = 6
CFAR_PFA = 3.0e-4                # ~ -35 dB threshold factor in square law

PEAK_RING_M = 3.0                # distance tolerance linking sims to analytic


def _lam_m() -> float:
    return 3.0e8 / CARRIER_MIN_HZ


def build_radar(n_rx: int):
    from radarsimpy import Radar, Receiver, Transmitter

    spacing = RX_SPACING_LAMBDA * _lam_m()
    channels = [{"location": (0.0, idx * spacing, 0.0)} for idx in range(n_rx)]
    transmitter = Transmitter(
        f=(CARRIER_MIN_HZ, CARRIER_MIN_HZ + CARRIER_BW_HZ),
        t=CHIRP_T_S,
        tx_power=TX_POWER_DBM,
        prp=PRP_S,
        pulses=PULSES,
    )
    receiver = Receiver(fs=FS_HZ, noise_figure=NOISE_FIGURE_DB, rf_gain=20,
                        channels=channels)
    return Radar(transmitter=transmitter, receiver=receiver)


def _write_mesh(path: str, extents: Sequence[float],
                location: Tuple[float, float, float]) -> str:
    import trimesh

    mesh = trimesh.creation.box(extents=extents)
    mesh.apply_translation([location[0], location[1] + float(extents[1]) / 2.0, location[2]])
    mesh.export(path)
    return path


def scene_mesh_targets(scene: Scene, dirpath: str) -> List[Dict[str, Any]]:
    """Mesh targets equivalent to the canonical scenes (y centred on edge)."""

    x, y = scene.direct_xy
    car_mesh = _write_mesh(os.path.join(dirpath, "car.stl"), (4.0, 2.0, 1.5),
                           (x, y, 0.75))
    plane_point = scene.plane_point_xy
    size = (scene.plane_length_m, 0.4, 3.0)
    glass_mesh = _write_mesh(os.path.join(dirpath, f"plane_{scene.name}.stl"), size,
                             (plane_point[0], plane_point[1], 1.5))
    targets = [
        {"mesh": car_mesh, "location": (float(x), float(y), 0.75),
         "velocity": (0.0, 0.0, 0.0), "rcs": 10.0},
        {"mesh": glass_mesh, "location": (float(plane_point[0]), float(plane_point[1]), 1.5),
         "velocity": (0.0, 0.0, 0.0), "rcs": 20.0},
    ]
    return targets


def _range_axis(n_bins: int) -> np.ndarray:
    beat = np.arange(n_bins) * (FS_HZ / (2.0 * CARRIER_BW_HZ / CHIRP_T_S)) / (2.0 * CARRIER_BW_HZ / CHIRP_T_S)
    # r = c * T * f_b / (2 B) with beat-to-bin mapping k * fs / n_bins
    beat_bins = np.arange(n_bins) * FS_HZ / n_bins
    return beat_bins * CHIRP_T_S * 3.0e8 / (2.0 * CARRIER_BW_HZ)


def _velocity_axis(n_doppler: int) -> np.ndarray:
    k = np.arange(n_doppler) - n_doppler // 2
    doppler = k / (PULSES * PRP_S)
    return doppler * _lam_m() / 4.0  # fmcw doppler->velocity, closing negative


def _extract_peaks(range_doppler_db: np.ndarray, n_bins: int, pfa: float):
    from radarsimpy.processing import cfar_ca_2d

    # cfar_ca_2d returns a boolean-ish mask over the map.
    mask = cfar_ca_2d(range_doppler_db, guard=CFAR_OFF, trailing=CFAR_ON, pfa=pfa)
    mask = np.asarray(mask)
    if mask.ndim != 2:
        raise ValueError(f"cfar mask wrong shape {mask.shape}")
    peaks = np.argwhere(mask)
    detections = []
    for row, col in peaks:
        amp = float(range_doppler_db[row, col])
        detections.append((int(row), int(col), amp))
    return mask, detections


def run_scene(scene: Scene, radar) -> Dict[str, Any]:
    from radarsimpy.simulator import sim_radar
    from radarsimpy.processing import range_doppler_fft

    with tempfile.TemporaryDirectory() as dirpath:
        targets = scene_mesh_targets(scene, dirpath)
        result = sim_radar(radar, targets)
    baseband = result["baseband"] if isinstance(result, dict) else result.get("baseband")
    baseband = np.asarray(baseband)

    # processing ffts axis=2 over (n_rx, n_chirps, n_samples); a single RxC
    # still presents as a 3-D tensor of shape (1, P, N).
    if baseband.ndim == 2:
        baseband = baseband[np.newaxis, ...]
    rd = range_doppler_fft(baseband)
    if rd.ndim == 2:
        rd = rd[np.newaxis, ...]
    maps = [np.asarray(rd[idx]) for idx in range(rd.shape[0])]
    rd_db = 20.0 * np.log10(np.abs(maps[0]) + 1e-12)
    n_bins = _rd_shape(rd_db)
    pfa = 3e-4
    _, peaks = _extract_peaks(rd_db, _rd_shape(rd_db), pfa)

    rng_axis = _range_axis(rd_db.shape[-1])
    vel_axis = _velocity_axis(rd_db.shape[-2]) if rd_db.ndim == 2 else _velocity_axis(rd_db.shape[0])

    noise_floor_db = float(np.percentile(rd_db, 50))
    detections = []
    for row, col, amp in peaks[:64]:
        det = {
            "range_m": float(rng_axis[col] if rd_db.ndim == 2 else rng_axis[0]),
            "velocity_mps": float(vel_axis[row]),
            "snr_db": float(amp - noise_floor_db),
        }
        detections.append(det)
        if rd_db.ndim == 2:
            det["range_m"] = float(rng_axis[col])
        if rd_db.ndim == 2:
            det["azimuth_rad"] = float("nan")

    # Free tier gives a single Rx channel: azimuth is deferred to the
    # two-run interferometry extension. Keep the field for schema stability.
    for det in detections:
        det["azimuth_rad"] = float("nan")

    return {
        "peaks": detections,
        "noise_floor_db": noise_floor_db,
        "shape": list(rd_db.shape),
        "analytic": analytic_gr(scene),
    }


def _rd_shape(rd_db: np.ndarray) -> int:
    return rd_db.shape[-1]


def _music_azimuth(maps: List[np.ndarray], peaks, shape) -> List[float]:
    from radarsimpy.processing import doa_bartlett

    if len(maps) < 2:
        return [float("nan")] * len(peaks)
    angles = []
    lam = _lam_m()
    spacing = RX_SPACING_LAMBDA * lam
    for row, col, _ in peaks:
        snap = np.asarray([m[row if m.shape == shape[:2] else min(row, m.shape[0] - 1),
                             min(col, m.shape[-1] - 1)] for m in maps])[:4]
        covmat = np.outer(snap, np.conj(snap))
        try:
            spec = doa_bartlett(covmat, send_sig=[float("-60"), float("60")], spacing=lam * RX_SPACING_LAMBDA)
            best = float(np.argmax(np.abs(spec[1] if isinstance(spec, tuple) else spec)))
            axis = spec[0] if isinstance(spec, tuple) else np.linspace(-60, 60, len(spec))
            angles.append(math.radians(float(axis[best])))
        except Exception:
            angles.append(float("nan"))
    return angles


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenes", nargs="*", default=["guardrail", "truck"])
    parser.add_argument("--out", default="radarsimpy_stage_b_out.json")
    args = parser.parse_args()

    try:
        import radarsimpy  # noqa: F401
    except Exception as exc:
        print(f"RadarSimPy unavailable: {exc}")
        return 1

    radar = build_radar(N_RX)
    payload: Dict[str, Any] = {"scenes": {}}
    for name in args.scenes:
        scene = SCENES[name]
        payload["scenes"][name] = run_scene(scene, radar)
        rows = payload["scenes"][name]["peaks"]
        print(f"== {name}: {len(rows)} CFAR peaks")
        for det in sorted(rows, key=lambda d: -d["snr_db"])[:10]:
            print(f"   r={det['range_m']:7.2f} m  az={np.degrees(det['azimuth_rad']):7.2f}° "
                  f"v={det['velocity_mps']:6.2f} m/s  snr={det['snr_db']:6.1f} dB")
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
