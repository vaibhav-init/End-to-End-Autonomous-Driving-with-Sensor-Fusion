#!/usr/bin/env python3
"""
Ground Truth Logger
===================

Records every simulation tick's metrics to a CSV file for later analysis.

Fields (from the thesis plan):
  step, fog_density, scenario_id,
  gt_ego_speed_kmh, gt_npc_speed_kmh,
  gt_distance_to_npc_m, gt_relative_velocity,
  gt_tl_state, throttle, brake, steer,
  collision_occurred, time_to_collision_s
"""

import csv
import os
import math


class GroundTruthLogger:
    """Records per-tick telemetry for CDF analysis and collision statistics."""

    FIELDS = [
        "step",
        "fog_density",
        "scenario_id",
        "seed",
        "gt_ego_speed_kmh",
        "gt_npc_speed_kmh",
        "gt_distance_to_npc_m",
        "gt_relative_velocity",
        "gt_tl_state",
        "throttle",
        "brake",
        "steer",
        "collision_occurred",
        "time_to_collision_s",
        "ego_accel_mps2",
        "min_distance_so_far_m",
    ]

    def __init__(self, output_dir, scenario_id, fog_density, seed):
        os.makedirs(output_dir, exist_ok=True)
        self.filepath = os.path.join(
            output_dir,
            f"s{scenario_id}_fog{fog_density}_seed{seed}.csv",
        )
        self.scenario_id = scenario_id
        self.fog_density = fog_density
        self.seed = seed
        self._file = open(self.filepath, "w", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=self.FIELDS)
        self._writer.writeheader()
        self._rows = 0
        self._collision_count = 0
        self._min_distance = float("inf")

    def log(
        self,
        step,
        ego_speed_kmh,
        npc_speed_kmh=None,
        distance_to_npc=None,
        relative_velocity=None,
        throttle=0.0,
        brake=0.0,
        steer=0.0,
        collision=False,
        tl_state=0,
        ego_accel=0.0,
    ):
        """Record one tick of ground truth data."""
        if distance_to_npc is not None:
            self._min_distance = min(self._min_distance, distance_to_npc)

        # Compute TTC safely
        rel_vel = relative_velocity if relative_velocity is not None else 0.0
        dist = distance_to_npc if distance_to_npc is not None else 999.0
        if rel_vel > 0.1 and dist > 0.1:
            ttc = dist / rel_vel
        else:
            ttc = 999.0

        if collision:
            self._collision_count += 1

        row = {
            "step": step,
            "fog_density": self.fog_density,
            "scenario_id": self.scenario_id,
            "seed": self.seed,
            "gt_ego_speed_kmh": round(ego_speed_kmh, 4),
            "gt_npc_speed_kmh": round(npc_speed_kmh, 4) if npc_speed_kmh is not None else "",
            "gt_distance_to_npc_m": round(dist, 4),
            "gt_relative_velocity": round(rel_vel, 4),
            "gt_tl_state": tl_state,
            "throttle": round(throttle, 4),
            "brake": round(brake, 4),
            "steer": round(steer, 4),
            "collision_occurred": 1 if collision else 0,
            "time_to_collision_s": round(ttc, 4),
            "ego_accel_mps2": round(ego_accel, 4),
            "min_distance_so_far_m": round(self._min_distance, 4),
        }
        self._writer.writerow(row)
        self._rows += 1

    @property
    def row_count(self):
        return self._rows

    @property
    def has_collision(self):
        return self._collision_count > 0

    @property
    def min_distance(self):
        return self._min_distance

    def close(self):
        if self._file and not self._file.closed:
            self._file.close()

    def __del__(self):
        self.close()


def compute_vehicle_speed(vehicle):
    """Return speed in km/h from a CARLA vehicle."""
    v = vehicle.get_velocity()
    return math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2) * 3.6


def distance_between(actor_a, actor_b):
    """Euclidean distance between two actors in metres."""
    return actor_a.get_location().distance(actor_b.get_location())
