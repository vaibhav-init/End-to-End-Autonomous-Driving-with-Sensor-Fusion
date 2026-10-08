"""Calibrate multipath priors: analytic mirror method vs the model vs RadarSimPy.

Stage A (always available) checks the repository's own image-method ghost
geometry (``carla4/radar/multipath.py``, ``geometry`` mode) against exact
mirror analytics over three canonical scene archetypes.  Stage B (once the
RadarSimPy pre-built module is placed next to this script) adds the SBR
ray-tracing reference for the same scenes; the deltas drive fitted values of
the ``multipath_*`` and ``ghost_*`` fields of ``RealisticRadarConfig``.

All numbers use the ``rgd_regime_v1`` envelope.  Conventions: radar at the
ego origin, x forward, y right; radial velocity positive closing.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass, replace
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

sys_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "..")  # noqa: E402
import sys
sys.path.insert(0, sys_path)  # noqa: E402

from carla4.radar.multipath import (  # noqa: E402
    ReflectorSegment,
    generate_multipath_targets,
    _mirror_point,
)
from carla4.radar.realistic_core import (  # noqa: E402
    IdealRadarTarget,
    RealisticRadarConfig,
    RealisticRadarModel,
)


def _snr_law_db(range_m: float) -> float:
    """compare_matlab anchor: SNR at 10 dBsm under the rgd_regime_v1 budget."""

    return 13.2 + 40.0 * math.log10(100.0 / max(range_m, 1.0))


@dataclass(frozen=True)
class Scene:
    name: str
    direct_xy: Tuple[float, float]         # direct target position in sensor xy
    plane_point_xy: Tuple[float, float]    # reflector plane anchor
    plane_normal_xy: Tuple[float, float]   # unit normal toward the radar side
    plane_length_m: float
    semantic_tag: int


GUARDRAIL = Scene(name="guardrail", direct_xy=(30.0, 2.0),
                  plane_point_xy=(25.0, 6.0), plane_normal_xy=(0.0, -1.0),
                  plane_length_m=100.0, semantic_tag=4)
TRUCK = Scene(name="truck", direct_xy=(35.0, 1.5),
              plane_point_xy=(20.0, 5.0),
              plane_normal_xy=(math.sin(math.radians(3.0)), -math.cos(math.radians(3.0))),
              plane_length_m=12.0, semantic_tag=4)
BRIDGE = Scene(name="bridge", direct_xy=(25.0, 0.0),
               plane_point_xy=(20.0, 0.0), plane_normal_xy=(0.0, 0.0),
               plane_length_m=60.0, semantic_tag=4)  # overhead plane has no road xy geometry

SCENES: Dict[str, Scene] = {s.name: s for s in (GUARDRAIL, TRUCK, BRIDGE)}


def _plane_reflector(scene: Scene, config: RealisticRadarConfig) -> ReflectorSegment:
    longest = scene.plane_length_m / 2.0
    tangent = (-scene.plane_normal_xy[1], scene.plane_normal_xy[0])
    return ReflectorSegment(
        reflector_id=1,
        semantic_tag=scene.semantic_tag,
        point_xy_m=scene.plane_point_xy,
        tangent_xy=tangent,
        normal_xy=scene.plane_normal_xy,
        length_m=scene.plane_length_m,
        rms_residual_m=0.05,
        point_count=max(int(scene.plane_length_m / 0.5), 8),
        reflection_loss_db=0.0,
    )


def _direct_targets(scene: Scene, config: RealisticRadarConfig) -> List[IdealRadarTarget]:
    x, y = scene.direct_xy
    distance = math.hypot(x, y)
    azimuth = math.atan2(y, x)
    point_count = 20
    lateral = 2.0
    snr = _snr_law_db(distance)
    return [IdealRadarTarget(
        object_id=1,
        semantic_tag=12,  # vehicle in CARLA semantic tags
        distance_m=float(distance),
        azimuth_rad=azimuth,
        relative_velocity_mps=0.0,
        snr_db=float(snr),
        point_count=point_count,
        lateral_extent_m=lateral,
        velocity_xy_mps=(0.0, 0.0),
    )]


def analytic_gr(scene: Scene) -> List[Dict[str, float]]:
    """Mirror-image ghost geometry: the two observable apparent ranges.

    A one-way idler path radar->reflector->target has geometric length
    ``|O->M|`` (mirror identity over the plane).  A two-way round trip mixing
    direct and reflected legs is received at ``(|direct| + |O->M|)/2``.  Both
    appear as separate detections in clutter literature, and the repository
    model emits ``second_range`` (the mixed one) plus ``third_range``
    (``|O->M|``, when third-order is enabled).
    """

    origin = np.zeros(2)
    target_xy = np.asarray(scene.direct_xy, dtype=float)
    plane_point = np.asarray(scene.plane_point_xy, dtype=float)
    normal = np.asarray(scene.plane_normal_xy, dtype=float)
    if np.linalg.norm(normal) < 1e-9:
        return []
    image_xy = np.asarray(_mirror_point(target_xy, plane_point, normal), dtype=float)
    direct_range = float(np.linalg.norm(target_xy))
    image_range = float(np.linalg.norm(image_xy - origin))
    mix_range = 0.5 * (direct_range + image_range)
    azimuth_target = math.atan2(target_xy[1] - origin[1], target_xy[0] - origin[0])
    azimuth_image = math.atan2(image_xy[1] - origin[1], image_xy[0] - origin[0])
    return [
        {"variant": "mix_half", "ghost_range_m": mix_range,
         "direct_range_m": direct_range, "ghost_azimuth_rad": azimuth_target,
         "path_length_m": mix_range, "range_bias_m": mix_range - direct_range},
        {"variant": "full_image", "ghost_range_m": image_range,
         "direct_range_m": direct_range, "ghost_azimuth_rad": azimuth_image,
         "path_length_m": image_range, "range_bias_m": image_range - direct_range},
    ]


def stage_a_plane_plane() -> Dict[str, Any]:
    """Compare the model's geometry-mode ghosts with the mirror analytics."""

    config = RealisticRadarConfig(profile_name="rgd_regime_v1")
    config = replace(config, multipath_mode="geometry")
    report: Dict[str, Any] = {}
    for name, scene in SCENES.items():
        if name == "bridge":
            continue  # overhead planes need the 3-D solver; guarded for stage B
        model = RealisticRadarModel(config=config, seed=0)
        targets = _direct_targets(scene, config)
        reflectors = [_plane_reflector(scene, config)]
        paths = generate_multipath_targets(targets, reflectors, config)
        analytic = analytic_gr(scene)
        rows = []
        for path in paths:
            path_payload = {
                "distance_m": float(path.distance_m),
                "azimuth_rad": float(path.azimuth_rad),
                "path_length_m": float(path.path_length_m),
                "snr_db": float(path.snr_db),
                "bounce_order": int(path.bounce_order),
            }
            if analytic:
                matched = min(analytic, key=lambda a: abs(a["ghost_range_m"] - path.distance_m))
                path_payload["matched_analytic_variant"] = matched["variant"]
                path_payload["analytic_range_m"] = matched["ghost_range_m"]
                path_payload["range_delta_m"] = path.distance_m - matched["ghost_range_m"]
            rows.append(path_payload)
        report[name] = {"n_model_ghosts": len(paths), "model_paths": rows,
                        "analytic": analytic}
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="radarsimpy_ghost_calibration.json")
    args = parser.parse_args()

    payload: Dict[str, Any] = {"stage_a": stage_a_plane_plane(), "stage_b": None}
    module_ready = os.path.isdir(os.path.join(os.path.dirname(__file__), "radarsimpy"))
    payload["radarsimpy_module_present"] = module_ready
    if module_ready:
        try:
            from carla4.radarsimpy_stage_b import run_stage_b  # optional module
            payload["stage_b"] = run_stage_b(SCENES)
        except Exception as exc:
            payload["stage_b"] = {"error": repr(exc)}

    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
