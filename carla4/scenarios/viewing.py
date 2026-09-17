#!/usr/bin/env python3
"""
Keep a finished scenario on screen long enough to watch.

S1, S2 and S4 end on the tick the outcome is decided (a collision, or the ego at
rest near the hazard) and destroy every actor straight away. That is right for
evaluation and useless to a person watching the CARLA window: the car vanishes
the instant it stops. hold_after_end keeps ticking for a few seconds with the
model still in control and the chase camera still following. Nothing is
logged, so the CSV and every metric computed from it are unchanged.
"""

import time

import carla

from config import FPS
from ground_truth_logger import compute_vehicle_speed, distance_between


def follow_ego(spectator, ego):
    """Chase camera: 15 m behind and 8 m above the ego, looking along its heading."""
    ego_t = ego.get_transform()
    spectator.set_transform(carla.Transform(
        ego_t.location - ego_t.get_forward_vector() * 15 + carla.Location(z=8),
        carla.Rotation(pitch=-20, yaw=ego_t.rotation.yaw),
    ))


def hold_after_end(world, ego, driver, spectator, hold_s, collision_flag, hazard=None):
    """Tick on for `hold_s` seconds of real time, unlogged, printing speed and gap."""
    if hold_s <= 0.0 or not ego.is_alive:
        return
    print(f"    ⏸  holding {hold_s:.0f}s for viewing (not logged)")
    reported_collision = collision_flag[0]
    for tick in range(int(hold_s * FPS)):
        ego.apply_control(driver.get_control(ego, world))
        tick_start = time.perf_counter()
        world.tick()
        elapsed = time.perf_counter() - tick_start
        if elapsed < 1.0 / FPS:
            time.sleep(1.0 / FPS - elapsed)
        if not ego.is_alive:
            return
        follow_ego(spectator, ego)

        if collision_flag[0] and not reported_collision:
            print("    💥 collision during the hold (not counted)")
            reported_collision = True
        if tick % FPS == 0:
            gap = (f"{distance_between(ego, hazard):.1f}m"
                   if hazard is not None and hazard.is_alive else "N/A")
            print(f"    hold t={tick // FPS:2d}s  spd={compute_vehicle_speed(ego):5.1f}km/h  "
                  f"dist={gap}")
