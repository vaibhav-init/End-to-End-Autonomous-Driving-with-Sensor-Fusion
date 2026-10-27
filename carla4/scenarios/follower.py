#!/usr/bin/env python3
"""A parameterised following vehicle for rear-end risk measurement.

S5 asks a different question from S1/S2/S4: not "did the ego brake for a real
hazard?" but "if the ego brakes for a phantom, does the vehicle behind it hit
it?".  This module supplies the follower and a simple delayed car-following
controller.  The delay is explicit because ``p(crash | phantom brake)`` is a
function of reaction time, not a property of the ego alone.
"""

from collections import deque
import math

import carla


def _speed_kmh(actor):
    velocity = actor.get_velocity()
    return math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2) * 3.6


def _distance_m(a, b):
    return a.get_location().distance(b.get_location())


def spawn_follower(world, carla_map, ego, gap_m=20.0):
    """Spawn a four-wheel vehicle in the ego's lane, ``gap_m`` behind it."""

    wp = carla_map.get_waypoint(
        ego.get_location(),
        project_to_road=True,
        lane_type=carla.LaneType.Driving,
    )
    if wp is None:
        return None
    bp_lib = world.get_blueprint_library()
    vehicle_bps = [
        bp for bp in bp_lib.filter("vehicle.*")
        if int(bp.get_attribute("number_of_wheels")) >= 4
    ]
    if not vehicle_bps:
        return None
    blueprint = vehicle_bps[0]
    for distance in (gap_m, gap_m * 1.25, gap_m * 0.8, gap_m * 1.5):
        previous = wp.previous(distance)
        if not previous:
            continue
        same_lane = [
            candidate for candidate in previous
            if candidate.road_id == wp.road_id
            and candidate.lane_id == wp.lane_id
        ]
        transform = (same_lane or previous)[0].transform
        transform.location.z += 0.5
        follower = world.try_spawn_actor(blueprint, transform)
        if follower is not None:
            follower.set_target_velocity(ego.get_velocity())
            return follower
    return None


class FollowerController:
    """Delayed proportional car-following controller.

    The follower keeps ``desired_gap_m`` behind the ego, using the ego's speed
    from ``reaction_s`` ago.  Steering is delegated to BasicAgent so the
    follower stays in lane; only throttle/brake are ours.
    """

    def __init__(
        self,
        follower,
        ego,
        carla_map,
        desired_gap_m=20.0,
        reaction_s=0.8,
        fps=20,
        max_speed_kmh=90.0,
        min_gap_m=5.0,
    ):
        self.follower = follower
        self.ego = ego
        self.desired_gap_m = float(desired_gap_m)
        self.reaction_s = float(reaction_s)
        self.fps = int(fps)
        self.min_gap_m = float(min_gap_m)
        self.max_speed_kmh = float(max_speed_kmh)
        self.lookahead_m = 12.0
        delay_frames = max(1, int(round(self.reaction_s * self.fps)))
        self._ego_speeds = deque(
            [_speed_kmh(ego)] * delay_frames,
            maxlen=delay_frames,
        )

    def update(self):
        ego_speed = _speed_kmh(self.ego)
        self._ego_speeds.append(ego_speed)
        delayed_ego_speed = self._ego_speeds[0]
        gap = _distance_m(self.ego, self.follower)
        follower_speed = _speed_kmh(self.follower)

        # Feed forward the *current* ego speed when the gap is healthy, so the
        # follower does not fall behind during acceleration.  Use the delayed
        # speed only when closing a negative gap: that is the reaction lag
        # that makes a phantom brake dangerous.
        if gap >= self.desired_gap_m:
            target_speed = ego_speed + 0.9 * (gap - self.desired_gap_m)
        else:
            target_speed = delayed_ego_speed - 0.9 * (
                self.desired_gap_m - gap
            )
        if gap < self.min_gap_m:
            target_speed = 0.0
        target_speed = max(0.0, min(self.max_speed_kmh, target_speed))

        error = target_speed - follower_speed
        if error >= 0.0:
            throttle = max(0.0, min(1.0, 0.35 * error))
            brake = 0.0
        else:
            throttle = 0.0
            brake = max(0.0, min(1.0, -0.45 * error))

        control = carla.VehicleControl(
            throttle=throttle,
            brake=brake,
            steer=self._steer_toward_ego(),
            hand_brake=False,
        )
        return control, gap, target_speed, brake, follower_speed

    def _steer_toward_ego(self):
        """Pure pursuit toward a point ahead of the ego, not a separate route.

        BasicAgent follows its own 500 m destination; at a junction it can take
        a different branch and stop off-lane. The follower exists only to stay
        behind this ego, so its lateral target is the ego's lane.
        """

        ego_transform = self.ego.get_transform()
        forward = ego_transform.get_forward_vector()
        target = ego_transform.location + forward * self.lookahead_m
        follower_transform = self.follower.get_transform()
        delta = target - follower_transform.location
        target_yaw = math.atan2(delta.y, delta.x)
        current_yaw = math.radians(follower_transform.rotation.yaw)
        error = (target_yaw - current_yaw + math.pi) % (2.0 * math.pi) - math.pi
        return max(-0.7, min(0.7, error / math.radians(30.0)))
