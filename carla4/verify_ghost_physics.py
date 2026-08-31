#!/usr/bin/env python3
"""Verify the Phase 1 ghost-physics fixes against a live CARLA session.

The unit tests cover the maths in isolation. This checks the three properties
actually hold once the sensor is attached to a moving vehicle in a real map,
where reflectors are fitted from semantic LiDAR rather than handed in:

1. **Fading** -- a multipath path's SNR varies scan to scan and the path is
   intermittently detected, instead of being present every frame at a fixed
   level. Without this every ghost confirms through M-of-N exactly like a real
   return, and no persistence test can separate them.
2. **Incidence-dependent loss** -- ghost SNR spreads across paths rather than
   sitting at one value per material.
3. **Tangential Doppler** -- ghost radial velocity differs from its parent's,
   which only happens when the full velocity vector reaches the solver.

Restores world settings and destroys everything it spawned, so it is safe to
run against a session someone else is using.
"""

import argparse
import collections
import math
import statistics

from radar import create_front_radar


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--town", default=None, help="default: keep current map")
    parser.add_argument("--frames", type=int, default=400)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--range", dest="range_m", type=float, default=100.0)
    parser.add_argument("--profile", default="geometry_multipath_v1")
    parser.add_argument("--vehicles", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    import carla

    client = carla.Client(args.host, args.port)
    client.set_timeout(30.0)
    world = client.load_world(args.town) if args.town else client.get_world()
    original_settings = world.get_settings()
    traffic_manager = client.get_trafficmanager()
    spawned = []
    radar = None

    try:
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 1.0 / float(args.fps)
        world.apply_settings(settings)
        traffic_manager.set_synchronous_mode(True)
        traffic_manager.set_random_device_seed(args.seed)

        library = world.get_blueprint_library()
        points = world.get_map().get_spawn_points()
        ego_bp = library.filter("vehicle.tesla.model3")[0]
        ego = None
        for point in points:
            ego = world.try_spawn_actor(ego_bp, point)
            if ego is not None:
                break
        if ego is None:
            raise RuntimeError("could not spawn the ego vehicle")
        spawned.append(ego)
        ego.set_autopilot(True, traffic_manager.get_port())

        vehicle_bps = library.filter("vehicle.*")
        for point in points[1 : 1 + args.vehicles]:
            other = world.try_spawn_actor(vehicle_bps[0], point)
            if other is not None:
                other.set_autopilot(True, traffic_manager.get_port())
                spawned.append(other)

        radar = create_front_radar(
            ego,
            world,
            range_m=args.range_m,
            backend="realistic",
            fps=args.fps,
            radar_profile=args.profile,
            radar_seed=args.seed,
            capture_debug=True,
        )

        per_path_snr = collections.defaultdict(list)
        per_path_frames = collections.Counter()
        families = collections.Counter()
        doppler_deltas = []
        ghost_frames = 0
        observed = 0

        for _ in range(args.frames):
            world.tick()
            snapshot = radar.debug_snapshot() or {}
            detections = snapshot.get("generated_detections") or []
            if not detections:
                continue
            observed += 1
            parents = {
                d.get("truth_object_id"): d.get("relative_velocity_mps")
                for d in detections
                if d.get("source") == "direct"
            }
            saw_ghost = False
            for detection in detections:
                if detection.get("source") != "ghost":
                    continue
                saw_ghost = True
                path_id = detection.get("truth_object_id")
                per_path_snr[path_id].append(float(detection["snr_db"]))
                per_path_frames[path_id] += 1
                families[
                    (detection.get("bounce_type"), detection.get("bounce_order"))
                ] += 1
                parent_velocity = parents.get(
                    detection.get("truth_parent_object_id")
                )
                if parent_velocity is not None:
                    doppler_deltas.append(
                        abs(
                            float(detection["relative_velocity_mps"])
                            - float(parent_velocity)
                        )
                    )
            ghost_frames += int(saw_ghost)

        print("=" * 72)
        print("PHASE 1 GHOST PHYSICS VERIFICATION")
        print("=" * 72)
        print(f"map:            {world.get_map().name}")
        print(f"profile:        {args.profile}")
        print(f"frames ticked:  {args.frames}   with radar output: {observed}")
        print(f"frames w/ghost: {ghost_frames}")
        print(f"distinct paths: {len(per_path_snr)}")
        print(f"families:       {dict(families)}")
        print()

        if not per_path_snr:
            print("RESULT: NO GHOSTS OBSERVED — cannot verify. Try a map with "
                  "more reflective surfaces, or more frames.")
            return

        spreads = [
            max(values) - min(values)
            for values in per_path_snr.values()
            if len(values) > 3
        ]
        duty = [
            per_path_frames[path] / max(observed, 1) for path in per_path_frames
        ]
        all_snr = [v for values in per_path_snr.values() for v in values]

        checks = []
        if spreads:
            median_spread = statistics.median(spreads)
            print(f"[1] fading      per-path SNR spread: median {median_spread:.2f} dB, "
                  f"max {max(spreads):.2f} dB")
            checks.append(("per-path SNR varies (fading)", median_spread > 1.0))
        intermittent = sum(1 for d in duty if d < 0.95)
        print(f"    intermittency: {intermittent}/{len(duty)} paths present in "
              f"<95% of frames")
        checks.append(("paths are intermittent", intermittent > 0))

        snr_range = max(all_snr) - min(all_snr)
        print(f"[2] incidence   ghost SNR across all paths: "
              f"{min(all_snr):.1f}..{max(all_snr):.1f} dB (range {snr_range:.1f})")
        checks.append(("ghost SNR spreads across paths", snr_range > 3.0))

        if doppler_deltas:
            median_delta = statistics.median(doppler_deltas)
            print(f"[3] doppler     |ghost - parent| radial velocity: "
                  f"median {median_delta:.3f} m/s, max {max(doppler_deltas):.3f}")
            checks.append(("ghost Doppler differs from parent", median_delta > 0.05))
        else:
            print("[3] doppler     no ghost/parent pairs observed in one frame")

        print()
        for name, ok in checks:
            print(f"  {'PASS' if ok else 'FAIL'}  {name}")
        print()
        print("RESULT:", "ALL CHECKS PASSED" if all(ok for _, ok in checks)
              else "SOME CHECKS FAILED")
        print("=" * 72)
    finally:
        if radar is not None:
            try:
                radar.cleanup()
            except Exception:
                pass
        for actor in reversed(spawned):
            try:
                actor.destroy()
            except Exception:
                pass
        try:
            traffic_manager.set_synchronous_mode(False)
        except Exception:
            pass
        world.apply_settings(original_settings)
        print("world settings restored, spawned actors destroyed")


if __name__ == "__main__":
    main()
