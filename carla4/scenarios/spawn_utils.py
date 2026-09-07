#!/usr/bin/env python3
"""
Spawn utilities for NHTSA scenarios.

Provides:
  - get_highway_spawns(): Generate spawn points on multi-lane (highway) roads only
  - spawn_obstacle_in_ego_direction(): Spawn a stopped vehicle verified to be
    ahead of the ego's actual forward direction
  - spawn_npc_in_ego_direction(): Same but for moving NPCs (S2/S3)
"""

import random
import carla


def get_highway_spawns(carla_map, min_straight_m=80.0, max_yaw_diff=5.0,
                       sample_distance=10.0):
    """
    Generate spawn transforms on highway (multi-lane) road sections.

    Identifies highway roads by checking for parallel driving lanes in the
    same direction — highways typically have 2+ lanes per direction while
    urban roads are single-lane.

    Args:
        carla_map: CARLA map object
        min_straight_m: Minimum straight distance ahead required (metres)
        max_yaw_diff: Maximum heading change over min_straight_m (degrees)
        sample_distance: Distance between sampled waypoints (metres)

    Returns:
        List of carla.Transform objects suitable for spawning on highways
    """
    all_waypoints = carla_map.generate_waypoints(sample_distance)

    highway_spawns = []
    seen_locations = set()  # avoid duplicates

    for wp in all_waypoints:
        if wp.is_junction:
            continue
        if wp.lane_type != carla.LaneType.Driving:
            continue

        # Check for parallel lanes (highway characteristic)
        # Same-direction lanes have lane_id with the same sign
        right = wp.get_right_lane()
        left = wp.get_left_lane()
        has_parallel = False

        if right and right.lane_type == carla.LaneType.Driving:
            if right.lane_id * wp.lane_id > 0:  # same direction
                has_parallel = True
        if left and left.lane_type == carla.LaneType.Driving:
            if left.lane_id * wp.lane_id > 0:  # same direction
                has_parallel = True

        if not has_parallel:
            continue

        # Must have a long straight road ahead
        ahead = wp.next(min_straight_m)
        if not ahead:
            continue

        start_yaw = wp.transform.rotation.yaw
        end_yaw = ahead[0].transform.rotation.yaw
        diff = abs((end_yaw - start_yaw + 180) % 360 - 180)
        if diff > max_yaw_diff:
            continue

        # De-duplicate nearby points (within 8m grid)
        loc = wp.transform.location
        grid_key = (round(loc.x / 8.0), round(loc.y / 8.0))
        if grid_key in seen_locations:
            continue
        seen_locations.add(grid_key)

        tf = wp.transform
        tf.location.z += 0.5  # lift above road surface
        highway_spawns.append(tf)

    return highway_spawns


# A staged actor further off the ego's forward axis than this is in a
# different lane; the radar corridor is 1.8 m half width plus growth.
MAX_STAGED_LATERAL_OFFSET_M = 2.0


def _advance_waypoint(carla_map, ego, ahead_m):
    """
    Walk ahead_m along the road in the ego's actual driving direction.

    Uses dot product between ego forward vector and waypoint direction
    to determine whether to use wp.next() or wp.previous().

    Returns the target waypoint, or None on failure.
    """
    ego_tf = ego.get_transform()
    ego_loc = ego_tf.location
    ego_fwd = ego_tf.get_forward_vector()

    wp = carla_map.get_waypoint(ego_loc, project_to_road=True,
                                lane_type=carla.LaneType.Driving)
    if wp is None:
        return None

    # Check if waypoint direction matches ego's forward direction
    wp_fwd = wp.transform.get_forward_vector()
    dot = ego_fwd.x * wp_fwd.x + ego_fwd.y * wp_fwd.y

    # Use next() or previous() depending on alignment with ego heading
    if dot >= 0:
        advance_fn = lambda w, d: w.next(d)
    else:
        advance_fn = lambda w, d: w.previous(d)

    # Walk forward to the target distance, staying on the ego's own lane.
    # ``next()`` lists every continuation at a fork or ramp, and taking [0]
    # blindly walked the staged obstacle onto a neighbouring carriageway on
    # Town04: the ground truth still reported a closing in-path actor while
    # the radar saw only the guardrail between them, so the scenario scored a
    # perception failure that never existed.
    lane_id = wp.lane_id
    road_id = wp.road_id
    travelled = 0.0
    step = 3.0
    while travelled < ahead_m:
        next_wps = advance_fn(wp, step)
        if not next_wps:
            return None
        same_lane = [w for w in next_wps if w.road_id == road_id and w.lane_id == lane_id]
        same_side = [w for w in next_wps if (w.lane_id > 0) == (lane_id > 0)]
        wp = (same_lane or same_side or next_wps)[0]
        travelled += step

    # Verify the target position is actually ahead of the ego
    target_loc = wp.transform.location
    to_target_x = target_loc.x - ego_loc.x
    to_target_y = target_loc.y - ego_loc.y
    dot_check = ego_fwd.x * to_target_x + ego_fwd.y * to_target_y
    if dot_check < 0:
        # Target ended up behind ego — direction detection failed
        return None

    # The point must sit in the ego's driving corridor, not one lane over,
    # and at roughly the requested distance. ``next()`` on Town04 can jump to
    # a connecting road, which staged S1's 25 m obstacle 200-500 m away (and
    # on the wrong carriageway) in every run before this check existed.
    lateral = abs(-ego_fwd.y * to_target_x + ego_fwd.x * to_target_y)
    if lateral > MAX_STAGED_LATERAL_OFFSET_M:
        return None
    if not 0.5 * ahead_m <= dot_check <= 1.5 * ahead_m + 5.0:
        return None

    return wp


def spawn_obstacle_in_ego_direction(world, carla_map, ego, ahead_m):
    """
    Spawn a stopped vehicle ahead of the ego, verified against the ego's
    actual forward direction.

    Unlike the naive approach of using waypoint.next(), this function
    checks that the spawned obstacle is actually in front of the ego
    by comparing the ego's forward vector with the direction to the obstacle.

    Args:
        world: CARLA world
        carla_map: CARLA map
        ego: Ego vehicle actor
        ahead_m: Distance ahead to place obstacle (metres)

    Returns:
        Spawned vehicle actor, or None on failure
    """
    wp = _advance_waypoint(carla_map, ego, ahead_m)
    if wp is None:
        return None

    bp_lib = world.get_blueprint_library()
    vehicle_bps = [b for b in bp_lib.filter("vehicle.*")
                   if int(b.get_attribute("number_of_wheels")) >= 4]
    bp = random.choice(vehicle_bps)
    transform = wp.transform
    transform.location.z += 0.5
    vehicle = world.try_spawn_actor(bp, transform)
    if vehicle:
        vehicle.apply_control(carla.VehicleControl(brake=1.0, hand_brake=True))
        vehicle.set_target_velocity(carla.Vector3D(0.0, 0.0, 0.0))
    return vehicle


def spawn_npc_in_ego_direction(world, carla_map, ego, ahead_m):
    """
    Spawn a vehicle ahead of ego on the same lane, verified against
    the ego's actual forward direction.

    Same as spawn_obstacle_in_ego_direction but without applying brakes.
    Used for S2 (decelerating NPC) and S3 (constant speed NPC).

    Args:
        world: CARLA world
        carla_map: CARLA map
        ego: Ego vehicle actor
        ahead_m: Distance ahead to place NPC (metres)

    Returns:
        Spawned vehicle actor, or None on failure
    """
    wp = _advance_waypoint(carla_map, ego, ahead_m)
    if wp is None:
        return None

    bp_lib = world.get_blueprint_library()
    vehicle_bps = [b for b in bp_lib.filter("vehicle.*")
                   if int(b.get_attribute("number_of_wheels")) >= 4]
    bp = random.choice(vehicle_bps)
    transform = wp.transform
    transform.location.z += 0.5
    vehicle = world.try_spawn_actor(bp, transform)
    return vehicle
