"""The oracle arm is the ghosts-off arm.

`OracleGhostFilter` runs before the latency queue and the tracker, and ghosts
draw from their own RNG stream, so generating ghosts and deleting them by label
produces exactly the detections of never generating them. That makes the oracle
a valid upper bound on ghost filtering, and it makes an oracle-vs-clean
*closed-loop* difference a measurement of the harness's own non-determinism
rather than a cost of filtering.

Pinning it here because the study's headline once rested on reading such a
difference as an effect. If the oracle ever moves after the tracker, this fails,
and that is exactly when the interpretation has to change.
"""

import math
import unittest

from radar.oracle_filter import OracleGhostFilter
from radar.realistic_core import (
    IdealRadarTarget,
    RadarEnvironment,
    RealisticRadarModel,
    load_realistic_radar_config,
)

SCANS = 200


def _scene(scan):
    """A lead car, a pedestrian and a wall, all in slow relative motion."""

    t = scan * 0.1
    return [
        IdealRadarTarget(
            object_id=101, semantic_tag=14, distance_m=40.0 - 0.5 * t,
            azimuth_rad=0.02 * math.sin(0.3 * t), relative_velocity_mps=0.5,
            snr_db=22.0, point_count=6, lateral_extent_m=1.8,
            velocity_xy_mps=(0.5, 0.0), radial_extent_m=1.0),
        IdealRadarTarget(
            object_id=102, semantic_tag=12, distance_m=18.0 + 0.2 * t,
            azimuth_rad=-0.25, relative_velocity_mps=-1.2,
            snr_db=11.0, point_count=2, lateral_extent_m=0.6,
            velocity_xy_mps=(-1.2, 0.4), radial_extent_m=0.3),
        IdealRadarTarget(
            object_id=103, semantic_tag=3, distance_m=25.0,
            azimuth_rad=0.30, relative_velocity_mps=8.0,
            snr_db=25.0, point_count=10, lateral_extent_m=4.0,
            velocity_xy_mps=(8.0, 0.0), radial_extent_m=3.0),
    ]


def _ghosts(scan):
    """What the geometry solver hands the sensor: the lead mirrored in the wall."""

    t = scan * 0.1
    return [IdealRadarTarget(
        object_id=-(10_000_000_000 + 101), semantic_tag=14,
        distance_m=62.0 - 0.4 * t, azimuth_rad=0.34,
        relative_velocity_mps=0.4, snr_db=14.0, point_count=3,
        lateral_extent_m=1.8, source="ghost", parent_object_id=101,
        reflector_id=103, bounce_type="mirror", bounce_order=2,
        path_length_m=62.0, velocity_xy_mps=(0.4, 0.0), radial_extent_m=1.0)]


def _trace(multipath_mode, oracle, scans=SCANS, seed=42):
    config = load_realistic_radar_config(
        profile_name="rgd_regime_v1",
        max_range_m=100.0,
        overrides={"multipath_mode": multipath_mode},
    )
    model = RealisticRadarModel(
        config=config,
        seed=seed,
        detection_filter=OracleGhostFilter() if oracle else None,
    )
    environment = RadarEnvironment()
    trace = []
    for scan in range(scans):
        selected = model.step(
            _scene(scan),
            timestamp_s=scan * 0.05,
            environment=environment,
            multipath_targets=_ghosts(scan) if multipath_mode == "geometry" else None,
        )
        _, points = model.latest_points()
        trace.append((
            round(float(selected.distance_m), 9),
            round(float(selected.relative_velocity_mps), 9),
            round(float(selected.confidence), 9),
            selected.source,
            selected.track_id,
            selected.truth_object_id,
            round(float(selected.azimuth_rad), 9),
            tuple(sorted(round(float(point.distance_m), 9) for point in points)),
        ))
    return trace


class OracleEquivalenceTest(unittest.TestCase):
    def test_oracle_output_equals_multipath_off(self):
        """Arm B (ghosts + perfect filter) is arm C (no ghosts), scan for scan."""

        self.assertEqual(_trace("geometry", True), _trace("off", False))

    def test_unfiltered_ghosts_do_change_the_output(self):
        """The check above is sensitive: without the filter the arms diverge."""

        ghosts_on = _trace("geometry", False)
        ghosts_off = _trace("off", False)
        differing = sum(a != b for a, b in zip(ghosts_on, ghosts_off))
        self.assertGreater(differing, SCANS // 2)


if __name__ == "__main__":
    unittest.main()
