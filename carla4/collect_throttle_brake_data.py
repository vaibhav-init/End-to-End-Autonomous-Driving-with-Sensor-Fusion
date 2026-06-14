#!/usr/bin/env python3
"""
Throttle/Brake Data Collector (Town01 — Emergency Obstacle Injection)
======================================================================

Ego drives with CARLA Traffic Manager autopilot in Town01.
Records front radar features + autopilot's throttle/brake pedal values.

Emergency Data Injection:
    Every ~20s, a stationary vehicle is spawned 60-80m ahead of the ego.
    The autopilot slams the brakes, generating high-quality emergency braking
    data (brake=1.0 labels with rapidly decreasing distance and TTC).

Labels:
    autopilot_throttle ∈ [0, 1]
    autopilot_brake    ∈ [0, 1]

Usage:
    python collect_throttle_brake_data.py
    python collect_throttle_brake_data.py --duration 1200 --vehicles 40
"""

import os, sys, math, time, random, argparse, traceback, threading
import pandas as pd
import numpy as np
from yolo_perception import CameraManager, YOLOPerception, YOLO_AVAILABLE, TL_STATE_NAMES
import carla

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("⚠️  OpenCV not available — camera preview disabled")

# ============================================================================
# Config
# ============================================================================
CARLA_HOST = '127.0.0.1'
CARLA_PORT = 2000
TOWN = 'Town01'
FPS = 20
MAX_RADAR_RANGE = 50.0

NPC_VEHICLES = 40
NPC_PEDESTRIANS = 30  # fewer peds to avoid tick crashes

SAVE_DIR = 'dataset_throttle_brake'


# ============================================================================
# Front Radar Sensor
# ============================================================================
class FrontRadar:
    """Front-facing radar. Tracks nearest obstacle in a narrow forward cone."""

    def __init__(self, vehicle, world, range_m=50.0):
        self.latest = {
            'distance': range_m,
            'relative_velocity': 0.0,
            'obstacle_speed': 0.0,
        }
        self._ego_speed = 0.0
        self._range = range_m

        bp = world.get_blueprint_library().find('sensor.other.radar')
        bp.set_attribute('horizontal_fov', '10')   # narrow to avoid adjacent lanes
        bp.set_attribute('vertical_fov', '2')
        bp.set_attribute('range', str(range_m))
        bp.set_attribute('points_per_second', '1500')
        tf = carla.Transform(
            carla.Location(x=2.5, z=1.0),
            carla.Rotation(pitch=2.0)
        )
        self.sensor = world.spawn_actor(bp, tf, attach_to=vehicle)
        self.sensor.listen(self._on_radar)

    def _on_radar(self, data):
        nearest_dist = self._range
        nearest_vel = 0.0
        for det in data:
            if abs(det.azimuth) > 0.3 or det.depth < 1.0 or det.altitude < -0.02:
                continue
            if det.depth < nearest_dist:
                nearest_dist = det.depth
                nearest_vel = det.velocity

        rel_vel = -nearest_vel  # positive = closing
        obs_speed = max(0, self._ego_speed - rel_vel)
        self.latest = {
            'distance': nearest_dist,
            'relative_velocity': rel_vel,
            'obstacle_speed': obs_speed,
        }

    def update_ego_speed(self, speed):
        self._ego_speed = speed

    def get(self):
        return self.latest.copy()

    def cleanup(self):
        if self.sensor and self.sensor.is_alive:
            self.sensor.destroy()


# ============================================================================
# Spawn NPC Vehicles (all ignore lights so they keep flowing)
# ============================================================================
def spawn_vehicles(world, client, tm, count):
    bp_lib = world.get_blueprint_library()
    vehicle_bps = [bp for bp in bp_lib.filter('vehicle.*')
                   if int(bp.get_attribute('number_of_wheels')) >= 4]
    spawn_points = world.get_map().get_spawn_points()
    random.shuffle(spawn_points)

    port = tm.get_port()
    batch = []
    for i in range(min(count, len(spawn_points) - 1)):
        bp = random.choice(vehicle_bps)
        if bp.has_attribute('color'):
            bp.set_attribute('color', random.choice(bp.get_attribute('color').recommended_values))
        batch.append(
            carla.command.SpawnActor(bp, spawn_points[i + 1])
            .then(carla.command.SetAutopilot(carla.command.FutureActor, True, port)))

    ids = [r.actor_id for r in client.apply_batch_sync(batch, True) if not r.error]

    for vid in ids:
        v = world.get_actor(vid)
        if v:
            tm.vehicle_percentage_speed_difference(v, random.randint(-10, 40))
            tm.distance_to_leading_vehicle(v, random.uniform(2.0, 6.0))
            tm.ignore_lights_percentage(v, 100)
            tm.ignore_signs_percentage(v, 100)

    print(f"  🚗 Spawned {len(ids)}/{count} NPC vehicles")
    return ids


# ============================================================================
# Spawn Pedestrians
# ============================================================================
def spawn_pedestrians(world, client, count):
    bp_lib = world.get_blueprint_library()
    walker_bps = bp_lib.filter('walker.pedestrian.*')
    ctrl_bp = bp_lib.find('controller.ai.walker')

    walkers = []
    for _ in range(count):
        bp = random.choice(walker_bps)
        if bp.has_attribute('is_invincible'):
            bp.set_attribute('is_invincible', 'false')
        loc = world.get_random_location_from_navigation()
        if loc:
            w = world.try_spawn_actor(bp, carla.Transform(loc))
            if w:
                walkers.append(w)

    controllers = []
    for w in walkers:
        c = world.spawn_actor(ctrl_bp, carla.Transform(), attach_to=w)
        controllers.append(c)

    world.tick()

    for c in controllers:
        dest = world.get_random_location_from_navigation()
        if dest:
            c.start()
            c.go_to_location(dest)
            c.set_max_speed(1.0 + random.random() * 2.0)

    walker_ids = [w.id for w in walkers]
    ctrl_ids = [c.id for c in controllers]
    print(f"  🚶 Spawned {len(walkers)}/{count} pedestrians")
    return walker_ids, ctrl_ids


# ============================================================================
# Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description='Collect throttle/brake data from autopilot')
    parser.add_argument('--host', default=CARLA_HOST)
    parser.add_argument('--port', type=int, default=CARLA_PORT)
    parser.add_argument('--duration', type=int, default=900,
                        help='Total collection time in seconds (default: 900 = 15 min)')
    parser.add_argument('--vehicles', type=int, default=NPC_VEHICLES)
    parser.add_argument('--pedestrians', type=int, default=NPC_PEDESTRIANS)
    parser.add_argument('--output', default=SAVE_DIR)
    args = parser.parse_args()

    total_frames = args.duration * FPS

    print("=" * 70)
    print("THROTTLE/BRAKE DATA COLLECTOR")
    print("=" * 70)
    print(f"  Town:        {TOWN}")
    print(f"  Duration:    {args.duration}s ({args.duration // 60}m {args.duration % 60}s)")
    print(f"  Total frames: {total_frames:,}")
    print(f"  NPC vehicles: {args.vehicles}")
    print(f"  Pedestrians:  {args.pedestrians}")
    print(f"  Output:       {args.output}/data.csv")
    print()
    print("  Labels: autopilot_throttle, autopilot_brake ∈ [0, 1]")
    print("=" * 70)

    os.makedirs(args.output, exist_ok=True)
    csv_path = os.path.join(args.output, 'data.csv')

    # ---- Connect ----
    client = carla.Client(args.host, args.port)
    client.set_timeout(30.0)

    world = client.get_world()
    cur_map = world.get_map().name.split('/')[-1]
    if cur_map != TOWN:
        print(f"\n  🗺️  Loading {TOWN}...")
        world = client.load_world(TOWN)
        time.sleep(3)
    else:
        print(f"\n  🗺️  Already on {TOWN}")

    # Synchronous mode
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 1.0 / FPS
    world.apply_settings(settings)

    tm = client.get_trafficmanager(8000)
    tm.set_synchronous_mode(True)
    world.tick()

    carla_map = world.get_map()
    spawn_points = carla_map.get_spawn_points()

    # Filter for non-junction spawns
    safe_spawns = []
    for sp in spawn_points:
        wp = carla_map.get_waypoint(sp.location, project_to_road=True)
        if wp and not wp.is_junction:
            safe_spawns.append(sp)
    if not safe_spawns:
        safe_spawns = spawn_points
    random.shuffle(safe_spawns)
    print(f"  🛣️  {len(safe_spawns)} safe spawn points (of {len(spawn_points)} total)")

    # ---- Spawn ego ----
    ego_bp = world.get_blueprint_library().find('vehicle.tesla.model3')
    ego = None
    for sp in safe_spawns:
        ego = world.try_spawn_actor(ego_bp, sp)
        if ego:
            break
    if not ego:
        print("  ❌ Failed to spawn ego!")
        return

    # Autopilot — careful driver with good margins
    port = tm.get_port()
    ego.set_autopilot(True, port)
    tm.vehicle_percentage_speed_difference(ego, -10)  # 10% above speed limit (good speed)
    tm.distance_to_leading_vehicle(ego, 15.0)  # Large following distance
    tm.auto_lane_change(ego, True)
    try:
        # Respect ALL traffic rules — careful driver
        tm.ignore_lights_percentage(ego, 0)
        tm.ignore_signs_percentage(ego, 0)
        tm.ignore_walkers_percentage(ego, 0)  # Respect pedestrians
        tm.set_random_device_seed(0)
        tm.set_global_distance_to_leading_vehicle(12.0)  # Large global margin
    except:
        pass
    print(f"\n  🚗 Ego spawned: {ego.type_id}")
    print(f"     Autopilot ON — careful driver, 10% above limit, large margins, obeying traffic")

    # ---- Spawn traffic ----
    npc_ids = spawn_vehicles(world, client, tm, args.vehicles)
    walker_ids, ctrl_ids = spawn_pedestrians(world, client, args.pedestrians)

    # ---- Attach sensors ----
    radar = FrontRadar(ego, world, MAX_RADAR_RANGE)
    camera = CameraManager(ego, world)
    yolo = None
    if YOLO_AVAILABLE:
        yolo = YOLOPerception()
        print(f"  📷 RGB camera + YOLOv8 traffic light detector attached")
    else:
        print(f"  📷 RGB camera attached (YOLO unavailable — features will be 0)")
    print(f"  📡 Front radar attached (range={MAX_RADAR_RANGE}m)")

    # Let everything settle
    for _ in range(40):
        world.tick()

    # ---- Collection loop ----
    print(f"\n{'=' * 70}")
    print(f"  🏁 RECORDING — {args.duration}s ({total_frames:,} frames)")
    print(f"  Auto emergency obstacle every ~20s + manual via ENTER")
    print(f"  Press Ctrl+C to stop early")
    print(f"{'=' * 70}\n")

    data = []
    prev_speed = 0.0
    start_time = time.time()

    # Stats
    brake_frames = 0
    throttle_frames = 0
    coast_frames = 0
    emergency_count = 0

    # Emergency obstacle state
    obstacle_actor = None
    obstacle_stopped_frames = 0
    last_emergency_time = time.time()

    # Stuck detection
    stuck_frames = 0
    respawn_count = 0

    # ---- Manual trigger: press Enter ----
    spawn_requested = threading.Event()

    def key_listener():
        while True:
            try:
                input()
                spawn_requested.set()
            except EOFError:
                break

    listener_thread = threading.Thread(target=key_listener, daemon=True)
    listener_thread.start()
    print("  ⌨️  Press ENTER to spawn obstacle, or wait for auto-spawn every 20s\n")

    try:
        for frame in range(total_frames):
            for attempt in range(3):
                try:
                    world.tick()
                    break
                except RuntimeError as e:
                    if attempt < 2:
                        time.sleep(0.1)
                        continue
                    print(f"  ⚠️  world.tick() failed 3x: {e}")
                    raise

            # ---- Ego state ----
            try:
                vel = ego.get_velocity()
                speed = math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)
                accel = (speed - prev_speed) * FPS if frame > 0 else 0.0
                prev_speed = speed
                ctrl = ego.get_control()
            except Exception as e:
                print(f"  ⚠️  Ego lost! {e}")
                break

            # ---- Auto-trigger emergency obstacle every 20s ----
            now = time.time()
            if (now - last_emergency_time > 20.0
                    and speed > 5.0
                    and obstacle_actor is None):
                spawn_requested.set()
                last_emergency_time = now

            # ---- Spawn obstacle (manual or auto) ----
            if (spawn_requested.is_set()
                    and obstacle_actor is None
                    and speed > 5.0):
                spawn_requested.clear()
                try:
                    ego_wp = carla_map.get_waypoint(
                        ego.get_location(), project_to_road=True)
                    spawn_dist = random.uniform(60.0, 80.0)
                    fwd_wps = ego_wp.next(spawn_dist)
                    if fwd_wps:
                        obs_bp = random.choice([
                            bp for bp in world.get_blueprint_library().filter('vehicle.*')
                            if int(bp.get_attribute('number_of_wheels')) == 4
                        ])
                        obs_tf = fwd_wps[0].transform
                        obs_tf.location.z += 0.5
                        obstacle_actor = world.try_spawn_actor(obs_bp, obs_tf)
                        if obstacle_actor:
                            obstacle_actor.apply_control(
                                carla.VehicleControl(brake=1.0))
                            obstacle_stopped_frames = 0
                            emergency_count += 1
                            # FORCE ego to brake head-on — disable lane change
                            tm.auto_lane_change(ego, False)
                            print(f"\n  🚧 EMERGENCY #{emergency_count}: "
                                  f"Obstacle spawned {spawn_dist:.0f}m ahead "
                                  f"(ego at {speed*3.6:.0f} km/h) — lane change LOCKED")
                except:
                    pass
            elif spawn_requested.is_set():
                spawn_requested.clear()
                if obstacle_actor is not None:
                    print("  ⚠️  Obstacle already active — wait for removal")
                else:
                    print("  ⚠️  Ego too slow — speed up first!")

            # ---- Remove obstacle once ego stops ----
            if obstacle_actor is not None:
                try:
                    obstacle_actor.apply_control(carla.VehicleControl(brake=1.0))
                except:
                    pass

                if speed < 0.3:
                    obstacle_stopped_frames += 1
                else:
                    obstacle_stopped_frames = 0

                # Remove after 1 second stopped
                if obstacle_stopped_frames >= FPS * 1:
                    try:
                        obstacle_actor.destroy()
                    except:
                        pass
                    obstacle_actor = None
                    obstacle_stopped_frames = 0
                    # Re-enable lane change for normal driving
                    tm.auto_lane_change(ego, True)
                    print(f"  ✅ Obstacle removed — lane change UNLOCKED\n")

            # ---- Stuck detection (no emergency active) ----
            if obstacle_actor is None:
                if speed < 0.3:
                    stuck_frames += 1
                else:
                    stuck_frames = 0

                if stuck_frames >= FPS * 5:  # 5 seconds stuck
                    stuck_frames = 0
                    respawn_count += 1
                    new_sp = random.choice(safe_spawns)
                    ego.set_transform(new_sp)
                    ego.set_autopilot(True, port)
                    for _ in range(5):
                        world.tick()
                    print(f"  🔄 Stuck — teleported (respawn #{respawn_count})")
                    last_emergency_time = time.time()

            # ---- Target speed ----
            try:
                target_speed = (ego.get_speed_limit() * 1.10) / 3.6  # m/s
            except:
                target_speed = 10.0

            # ---- Radar ----
            radar.update_ego_speed(speed)
            r = radar.get()

            if r['relative_velocity'] > 0.1:
                ttc = min(r['distance'] / r['relative_velocity'], 10.0)
            else:
                ttc = 10.0

            # ---- Stats ----
            if ctrl.brake > 0.05:
                brake_frames += 1
            elif ctrl.throttle > 0.05:
                throttle_frames += 1
            else:
                coast_frames += 1

            # ---- YOLO traffic light + intersection detection ----
            tl_state = 0
            tl_conf = 0.0
            tl_bbox = None
            approaching_jct = 0
            cam_frame = None
            if yolo is not None:
                cam_frame = camera.get_frame()
                if cam_frame is not None:
                    tl_state, tl_conf, tl_bbox = yolo.detect_traffic_light(cam_frame)
                    approaching_jct = yolo.detect_intersection()

            # ---- Camera preview with YOLO bounding boxes ----
            if CV2_AVAILABLE and cam_frame is not None:
                display = cam_frame.copy()

                # Draw bounding box if traffic light detected
                if tl_bbox is not None:
                    x1, y1, x2, y2 = tl_bbox
                    # Color based on detected state
                    color_map = {
                        0: (200, 200, 200),  # none — gray
                        1: (0, 255, 0),      # green
                        2: (0, 255, 255),     # yellow
                        3: (0, 0, 255),       # red
                    }
                    box_color = color_map.get(tl_state, (200, 200, 200))
                    cv2.rectangle(display, (x1, y1), (x2, y2), box_color, 2)

                    # Label above the box
                    label = f"{TL_STATE_NAMES.get(tl_state, '?')} {tl_conf:.2f}"
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                    cv2.rectangle(display, (x1, y1 - th - 8), (x1 + tw + 4, y1), box_color, -1)
                    cv2.putText(display, label, (x1 + 2, y1 - 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

                # HUD overlay — top left
                tl_str = TL_STATE_NAMES.get(tl_state, 'none')
                hud_lines = [
                    f"Speed: {speed * 3.6:.1f} km/h",
                    f"TL: {tl_str.upper()}",
                    f"Intersection: {'YES' if approaching_jct else 'NO'}",
                    f"Dist: {r['distance']:.1f}m  TTC: {ttc:.1f}s",
                ]
                for i, line in enumerate(hud_lines):
                    cv2.putText(display, line, (10, 25 + i * 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                cv2.imshow('CARLA Camera + YOLO', display)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\n  ⚠️  Quit via camera window")
                    break

            # ---- Record ----
            data.append({
                'frame': frame,
                'timestamp': round(frame / FPS, 3),
                'ego_speed': round(speed, 4),
                'target_speed': round(target_speed, 4),
                'ego_acceleration': round(max(-20, min(20, accel)), 4),
                'distance': round(r['distance'], 4),
                'relative_velocity': round(r['relative_velocity'], 4),
                'ttc': round(ttc, 4),
                'obstacle_speed': round(r['obstacle_speed'], 4),
                'approaching_intersection': approaching_jct,
                'traffic_light_state': tl_state,
                'autopilot_throttle': round(ctrl.throttle, 4),
                'autopilot_brake': round(ctrl.brake, 4),
            })

            # ---- Spectator ----
            try:
                spec = world.get_spectator()
                tf = ego.get_transform()
                spec.set_transform(carla.Transform(
                    tf.location - tf.get_forward_vector() * 12 + carla.Location(z=6),
                    carla.Rotation(pitch=-20, yaw=tf.rotation.yaw)))
            except:
                pass

            # ---- Print progress ----
            if frame % (FPS * 5) == 0 and frame > 0:
                elapsed = time.time() - start_time
                pct = frame / total_frames * 100
                spd_kmh = speed * 3.6
                total = max(1, brake_frames + throttle_frames + coast_frames)
                tl_str = TL_STATE_NAMES.get(tl_state, 'none')
                jct_str = 'JCT' if approaching_jct else '---'
                print(
                    f"  [{frame:>7,}/{total_frames:,}] {pct:4.1f}%  "
                    f"SPD:{spd_kmh:5.1f}km/h  DIST:{r['distance']:5.1f}m  TTC:{ttc:5.1f}s  "
                    f"THR:{ctrl.throttle:4.2f}  BRK:{ctrl.brake:4.2f}  "
                    f"TL:{tl_str:>6s}  {jct_str}  "
                    f"BRK%:{100*brake_frames/total:4.1f}%  "
                    f"EMRG:{emergency_count}  RSPN:{respawn_count}"
                )

            # ---- Save periodically (every 60s) ----
            if frame > 0 and frame % (FPS * 60) == 0:
                pd.DataFrame(data).to_csv(csv_path, index=False)
                print(f"  💾 Saved {len(data):,} frames to {csv_path}")

    except KeyboardInterrupt:
        print(f"\n  ⚠️  Interrupted at frame {frame}")

    # ---- Final save ----
    if data:
        df = pd.DataFrame(data)
        df.to_csv(csv_path, index=False)

    # ---- Cleanup leftover obstacle ----
    if obstacle_actor and obstacle_actor.is_alive:
        try:
            obstacle_actor.destroy()
        except:
            pass

    # ---- Stats ----
    total = max(1, len(data))
    print(f"\n{'=' * 70}")
    print(f"COLLECTION COMPLETE")
    print(f"{'=' * 70}")
    print(f"  Total frames:  {len(data):,}")
    print(f"  Duration:      {len(data)/FPS:.0f}s")
    print(f"  Braking:       {brake_frames:,} frames ({100*brake_frames/total:.1f}%)")
    print(f"  Throttle:      {throttle_frames:,} frames ({100*throttle_frames/total:.1f}%)")
    print(f"  Coasting:      {coast_frames:,} frames ({100*coast_frames/total:.1f}%)")
    print(f"  Emergency injections: {emergency_count}")
    print(f"  Respawns:      {respawn_count}")
    if data:
        throttles = [d['autopilot_throttle'] for d in data]
        brakes = [d['autopilot_brake'] for d in data]
        print(f"  Throttle mean: {sum(throttles)/len(throttles):.3f}")
        print(f"  Brake mean:    {sum(brakes)/len(brakes):.3f}")
    print(f"  Saved to:      {csv_path}")
    print(f"{'=' * 70}")

    # ---- Cleanup ----
    print("\n  Cleaning up...")
    radar.cleanup()
    camera.cleanup()
    if CV2_AVAILABLE:
        cv2.destroyAllWindows()

    try:
        ego.set_autopilot(False, port)
    except:
        pass

    for cid in ctrl_ids:
        try:
            a = world.get_actor(cid)
            if a:
                a.stop()
        except:
            pass

    destroy_ids = npc_ids + ctrl_ids + walker_ids
    if destroy_ids:
        client.apply_batch([carla.command.DestroyActor(x) for x in destroy_ids])

    try:
        ego.destroy()
    except:
        pass

    settings = world.get_settings()
    settings.synchronous_mode = False
    settings.fixed_delta_seconds = None
    world.apply_settings(settings)
    tm.set_synchronous_mode(False)

    print("  ✅ Done!")


if __name__ == '__main__':
    main()
