"""Collect CARLA inter-radar geometry for the radar-to-radar interference study.

Division of labour
------------------
CARLA supplies **kinematics only**: where the two vehicles actually are, where
their radar boresights point, and how the range between the two radars closes.
CARLA's own radar sensor does not model interference between radars, so no
interference physics comes from CARLA -- that is entirely ``radar/rri.py`` in
``run_rri_sweep.py``.

What is recorded per tick is exactly what the interference model needs:

``range_m``
    Distance between the two radar mounting points: the path loss of the
    coupling.
``azimuth_deg``
    Bearing of the other radar in *our* sensor frame. The tailgating case uses
    a rear-facing mounting, expressed in its own boresight frame, because
    rotating a rear radar's geometry into the ego frame would put it outside a
    forward-looking field of view and silently discard every artifact.
``closing_mps``
    Rate of change of ``range_m``, differenced from CARLA's own integrated
    geometry rather than read off the scripted profile.
``boresight_alignment``
    Cosine between the *interfering* radar's boresight and the line back to
    ours. This is the dominant term in the whole study: a radar pointing along
    its own lane radiates almost nothing towards a neighbour in the next lane.

Both vehicles are driven by scripted ``set_transform`` paths rather than
autopilot. An interference study needs the inter-radar vector known exactly,
and a scripted path is the only way to guarantee it; traffic-manager lane
changes leave the geometry to chance.
"""

from __future__ import annotations

import argparse
import csv
import math
import os

import carla

# Radar mounting points in each vehicle's own frame, metres.
FRONT_RADAR_OFFSET = (2.4, 0.0, 0.75)
REAR_RADAR_OFFSET = (-2.4, 0.0, 0.75)

FRONT_MOUNT_YAW_DEG = 0.0
REAR_MOUNT_YAW_DEG = 180.0

EGO_SPEED_MPS = 15.0
LANE_CHANGE_START_S = 4.0
LANE_CHANGE_DURATION_S = 4.0


def _mount_location(transform, offset):
    """World location of a radar at ``offset`` in the given transform's frame."""

    forward = transform.get_forward_vector()
    right = transform.get_right_vector()
    return carla.Location(
        x=transform.location.x + offset[0] * forward.x + offset[1] * right.x,
        y=transform.location.y + offset[0] * forward.y + offset[1] * right.y,
        z=transform.location.z + offset[0] * forward.z + offset[1] * right.z
        + offset[2],
    )


def _boresight(transform, yaw_offset_deg=0.0):
    """Unit vector along a boresight, optionally yawed, in the world frame."""

    forward = transform.get_forward_vector()
    angle = math.radians(yaw_offset_deg)
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    x = forward.x * cos_a - forward.y * sin_a
    y = forward.x * sin_a + forward.y * cos_a
    norm = math.hypot(x, y) or 1.0
    return x / norm, y / norm


def _geometry(our_loc, our_boresight, their_loc, their_boresight):
    """Range, bearing in our frame, and their boresight alignment."""

    dx = their_loc.x - our_loc.x
    dy = their_loc.y - our_loc.y
    dz = their_loc.z - our_loc.z
    distance = max(math.sqrt(dx * dx + dy * dy + dz * dz), 1.0e-9)

    forward_x, forward_y = our_boresight
    norm = math.hypot(forward_x, forward_y) or 1.0
    forward_x, forward_y = forward_x / norm, forward_y / norm
    # Right of our boresight is forward rotated -90 degrees, matching the
    # x-forward / y-right radar convention in carla4/radar.
    right_x, right_y = -forward_y, forward_x

    longitudinal = dx * forward_x + dy * forward_y
    lateral = dx * right_x + dy * right_y
    azimuth_deg = math.degrees(math.atan2(lateral, longitudinal))

    to_us_x, to_us_y = -dx / distance, -dy / distance
    alignment = their_boresight[0] * to_us_x + their_boresight[1] * to_us_y
    return distance, azimuth_deg, alignment


def _planar_distance(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)


def _smoothstep(t):
    t = min(max(t, 0.0), 1.0)
    return t * t * (3.0 - 2.0 * t)


def _spawn(world, blueprint_id, transform):
    blueprint = world.get_blueprint_library().find(blueprint_id)
    return world.spawn_actor(blueprint, transform)


def _offset_from(transform, forward_m, right_m):
    """A transform ``forward_m`` ahead and ``right_m`` across from ``transform``.

    The anchor matters: the neighbour is placed relative to ego's *current*
    transform, not to the spawn point. Anchoring both to the spawn point lets
    ego drive away from a stationary neighbour, which is not the scenario.
    """

    forward = transform.get_forward_vector()
    right = transform.get_right_vector()
    return carla.Transform(
        carla.Location(
            x=transform.location.x + forward_m * forward.x + right_m * right.x,
            y=transform.location.y + forward_m * forward.y + right_m * right.y,
            z=transform.location.z,
        ),
        transform.rotation,
    )


def _collect(
    world,
    ego,
    other,
    base,
    scenario,
    duration_s,
    fps,
    mount_yaw_deg,
    other_mount_offset,
    other_forward_m,
    other_right_m,
):
    """Drive both vehicles along scripted paths and record inter-radar geometry.

    ``other_forward_m`` and ``other_right_m`` are callables of time giving the
    neighbour's offset from ego in ego's own frame, so a lane change is a
    scripted lateral sweep rather than a traffic-manager decision.
    """

    our_offset = FRONT_RADAR_OFFSET if mount_yaw_deg == FRONT_MOUNT_YAW_DEG else REAR_RADAR_OFFSET
    rows = []
    frames = int(round(duration_s * fps))
    previous_range = None
    previous_ego_loc = None
    previous_other_loc = None
    ego_speed = other_speed = 0.0
    for frame in range(frames):
        t = frame / float(fps)
        ego_tf = _offset_from(base, EGO_SPEED_MPS * t, 0.0)
        ego.set_transform(ego_tf)
        # Relative to ego, not to the spawn point: the neighbour has to travel
        # with ego or the scripted gap means nothing.
        other.set_transform(
            _offset_from(ego_tf, other_forward_m(t), other_right_m(t))
        )
        if world.tick() is None:
            break

        ego_tf = ego.get_transform()
        other_tf = other.get_transform()
        our_loc = _mount_location(ego_tf, our_offset)
        their_loc = _mount_location(other_tf, other_mount_offset)
        our_boresight = _boresight(ego_tf, mount_yaw_deg)
        their_boresight = _boresight(other_tf, 0.0)
        distance, azimuth_deg, alignment = _geometry(
            our_loc, our_boresight, their_loc, their_boresight
        )

        if previous_range is None:
            closing = 0.0
            ego_speed = other_speed = 0.0
        else:
            # Positive means closing. Differenced from CARLA's own integrated
            # geometry rather than read off the script.
            #
            # Speeds are differenced the same way instead of using
            # Actor.get_velocity(): a vehicle repositioned with set_transform
            # teleports, and the velocity CARLA reports for a teleport is
            # meaningless (it read 200 m/s here, from being shoved by ambient
            # traffic between scripted poses). A position delta over a fixed
            # step is the only figure that means anything here.
            closing = (previous_range - distance) * fps
            ego_step = _planar_distance(ego_tf.location, previous_ego_loc) * fps
            other_step = _planar_distance(other_tf.location, previous_other_loc) * fps
            ego_speed = 0.5 * (ego_speed + ego_step) if frame > 1 else ego_step
            other_speed = 0.5 * (other_speed + other_step) if frame > 1 else other_step
        previous_range = distance
        previous_ego_loc = ego_tf.location
        previous_other_loc = other_tf.location

        rows.append(
            {
                "scenario": scenario,
                "frame": frame,
                "time_s": round(t, 4),
                "range_m": round(distance, 4),
                "azimuth_deg": round(azimuth_deg, 4),
                "closing_mps": round(closing, 4),
                "boresight_alignment": round(alignment, 5),
                "ego_speed_mps": round(ego_speed, 4),
                "other_speed_mps": round(other_speed, 4),
                "other_right_m": round(other_right_m(t), 4),
            }
        )
    return rows


def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--town", default="Town04")
    parser.add_argument(
        "--output", default=os.path.join(here, "carla_rri_geometry.csv")
    )
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument(
        "--follower-closing-mps",
        type=float,
        default=2.0,
        help="Rate at which the follower's gap shrinks in the tailgating case, "
        "so one run sweeps a safe headway down to a close one.",
    )
    parser.add_argument(
        "--rpc-timeout",
        type=float,
        default=180.0,
        help="Seconds to wait for the simulator RPC. Long by default because a "
        "listening socket is not a ready simulator.",
    )
    parser.add_argument(
        "--load-world",
        action="store_true",
        help="Switch to --town if CARLA is on a different map. Off by default: "
        "a map reload takes minutes on a machine that is already loaded.",
    )
    parser.add_argument(
        "--lane-offset-m",
        type=float,
        default=3.5,
        help="Neighbour lateral offset in the adjacent-lane cases. A real lane "
        "is ~3.5 m, so this is not the same quantity as the following gap.",
    )
    parser.add_argument(
        "--follow-gap-m",
        type=float,
        default=20.0,
        help="Starting longitudinal gap behind ego in the tailgating case; it "
        "closes from there at --follower-closing-mps.",
    )
    args = parser.parse_args()

    # A generous RPC timeout on purpose. CARLA can be listening on the port
    # while still initialising, in which case a short timeout surfaces as a
    # misleading "make sure the simulator is ready" error rather than a wait.
    client = carla.Client(args.host, args.port)
    client.set_timeout(float(args.rpc_timeout))
    world = client.get_world()

    current = world.get_map().name.split("/")[-1]
    if current != args.town:
        if not args.load_world:
            raise SystemExit(
                f"CARLA is on {current}, not {args.town}. Pass --load-world to "
                "switch (slow on a loaded machine) or --town to match."
            )
        world = client.load_world(args.town)
        world.tick()

    # No set_simulator_fps call: in synchronous mode fixed_delta_seconds below
    # is what sets the step, and neither World nor Client exposes that helper in
    # 0.9.16.
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 1.0 / args.fps
    world.apply_settings(settings)

    rows = []
    spawned = []
    try:
        base = world.get_map().get_spawn_points()[0]
        base = carla.Transform(
            carla.Location(base.location.x, base.location.y, base.location.z + 0.5),
            base.rotation,
        )
        ego = _spawn(world, "vehicle.tesla.model3", base)
        spawned.append(ego)

        def neighbour(spawn_forward, spawn_right):
            # The spawn location only has to be collision-free; _collect
            # repositions both vehicles before the first tick, so a scripted
            # offset that is off-road or occupied can still be used as the
            # starting geometry. Spawning at a short offset keeps this valid
            # whatever --gap-m asks for.
            try:
                actor = _spawn(
                    world,
                    "vehicle.tesla.model3",
                    _offset_from(base, spawn_forward, spawn_right),
                )
            except RuntimeError:
                actor = _spawn(
                    world,
                    "vehicle.tesla.model3",
                    _offset_from(base, -6.0, float(spawn_right) * 0.25),
                )
            spawned.append(actor)
            return actor

        lane_offset = args.lane_offset_m
        follow_gap = args.follow_gap_m

        # Scenario A, steady: neighbour held alongside in the adjacent lane.
        other = neighbour(1.0, lane_offset)
        rows.extend(
            _collect(
                world, ego, other, base,
                scenario="adjacent_lane/steady",
                duration_s=args.duration, fps=args.fps,
                mount_yaw_deg=FRONT_MOUNT_YAW_DEG,
                other_mount_offset=FRONT_RADAR_OFFSET,
                other_forward_m=lambda t: 1.0,
                other_right_m=lambda t: lane_offset,
            )
        )
        other.destroy()
        spawned.remove(other)

        # Scenario A, lane change: neighbour sweeps from its lane into ours,
        # which is the attitude where its radar turns towards us.
        other = neighbour(1.0, lane_offset)
        rows.extend(
            _collect(
                world, ego, other, base,
                scenario="adjacent_lane/lane_change",
                duration_s=args.duration, fps=args.fps,
                mount_yaw_deg=FRONT_MOUNT_YAW_DEG,
                other_mount_offset=FRONT_RADAR_OFFSET,
                other_forward_m=lambda t: 1.0,
                other_right_m=lambda t: lane_offset
                * _smoothstep((t - LANE_CHANGE_START_S) / LANE_CHANGE_DURATION_S),
            )
        )
        other.destroy()
        spawned.remove(other)

        # Scenario B: a follower observed by our rear-facing radar. The gap
        # closes at a scripted rate so a single run sweeps from a safe
        # headway down to a dangerous one and crosses the interference
        # boundary in the collected geometry, rather than sitting at one
        # distance the way the analytic sweep does.
        closing_rate = float(args.follower_closing_mps)
        min_gap = 0.5

        def follower_offset(t):
            return -max(min_gap, follow_gap - closing_rate * t)

        other = neighbour(-6.0, 0.0)
        rows.extend(
            _collect(
                world, ego, other, base,
                scenario="rear_radar_tailgating/closing",
                duration_s=args.duration, fps=args.fps,
                mount_yaw_deg=REAR_MOUNT_YAW_DEG,
                other_mount_offset=FRONT_RADAR_OFFSET,
                other_forward_m=follower_offset,
                other_right_m=lambda t: 0.0,
            )
        )
        other.destroy()
        spawned.remove(other)
    finally:
        for actor in spawned:
            try:
                actor.destroy()
            except Exception:
                pass
        try:
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = None
            world.apply_settings(settings)
        except Exception:
            pass

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote {args.output} ({len(rows)} rows)")
    for scenario in dict.fromkeys(row["scenario"] for row in rows):
        subset = [row for row in rows if row["scenario"] == scenario]
        ranges = [row["range_m"] for row in subset]
        aligns = [row["boresight_alignment"] for row in subset]
        azim = [row["azimuth_deg"] for row in subset]
        closings = [row["closing_mps"] for row in subset]
        print(
            f"  {scenario:34s} n={len(subset):4d}"
            f"  range {min(ranges):6.2f}-{max(ranges):6.2f} m"
            f"  az {min(azim):+7.2f}..{max(azim):+7.2f} deg"
            f"  align {min(aligns):+.3f}..{max(aligns):+.3f}"
            f"  closing {min(closings):+.2f}..{max(closings):+.2f} m/s"
        )


if __name__ == "__main__":
    main()