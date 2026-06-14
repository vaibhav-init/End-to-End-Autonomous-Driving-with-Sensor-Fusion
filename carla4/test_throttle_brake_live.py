#!/usr/bin/env python3
"""
Live Test: Throttle/Brake MLP with BasicAgent Steering
========================================================

Tests the trained throttle/brake model in CARLA:
  - BasicAgent handles steering (pathfinding + lane following)
  - MLP model controls throttle & brake based on front radar
  - Spawns obstacle scenarios to test if the car stops properly

The script spawns challenging situations (stopped vehicles, sudden brakers,
pedestrians) and logs whether the model brakes in time to avoid collision.

Usage:
    python test_throttle_brake_live.py
    python test_throttle_brake_live.py --duration 180 --vehicles 60

Press Ctrl+C to stop.
"""

import os, sys, math, time, random, argparse, traceback, threading
import numpy as np
import carla
import torch
import torch.nn as nn
import pickle
from yolo_perception import (
    CameraManager, YOLOPerception,
    TL_STATE_NAMES, YOLO_AVAILABLE,
)

# Add CARLA agents to path
CARLA_ROOT = os.environ.get('CARLA_ROOT', '/opt/carla-simulator')
AGENTS_PATH = os.path.join(CARLA_ROOT, 'PythonAPI', 'carla')
if AGENTS_PATH not in sys.path:
    sys.path.insert(0, AGENTS_PATH)

try:
    from agents.navigation.basic_agent import BasicAgent
    AGENT_AVAILABLE = True
except ImportError:
    AGENT_AVAILABLE = False
    print("⚠️  BasicAgent not found. Will use manual steering.")
    print(f"   Tried: {AGENTS_PATH}")


# ============================================================================
# Config
# ============================================================================
CARLA_HOST = '127.0.0.1'
CARLA_PORT = 2000
TOWN = 'Town01'
FPS = 20
MAX_RADAR_RANGE = 50.0

MODEL_PATH = 'model_throttle_brake/throttle_brake_mlp.pt'
SCALER_PATH = 'model_throttle_brake/scaler.pkl'

FEATURE_COLS = [
    'ego_speed', 'target_speed', 'ego_acceleration', 'distance',
    'relative_velocity', 'ttc', 'obstacle_speed',
    'approaching_intersection', 'traffic_light_state',
]


# ============================================================================
# MLP Model (must match training — dual output)
# ============================================================================
class ThrottleBrakeMLP(nn.Module):
    def __init__(self, input_dim=9):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 2),
            nn.Sigmoid(),  # [throttle, brake] each in [0, 1]
        )

    def forward(self, x):
        return self.net(x)  # shape: (batch, 2)


# ============================================================================
# Front Radar Sensor
# ============================================================================
class FrontRadar:
    def __init__(self, vehicle, world, range_m=50.0):
        self.latest = {
            'distance': range_m,
            'relative_velocity': 0.0,
            'obstacle_speed': 0.0,
        }
        self._ego_speed = 0.0
        self._range = range_m

        bp = world.get_blueprint_library().find('sensor.other.radar')
        bp.set_attribute('horizontal_fov', '10')  # Much narrower to avoid adjacent lanes
        bp.set_attribute('vertical_fov', '2')  # Narrow beam to avoid ground returns
        bp.set_attribute('range', str(range_m))
        bp.set_attribute('points_per_second', '1500')
        tf = carla.Transform(
            carla.Location(x=2.5, z=1.0),
            carla.Rotation(pitch=2.0)  # Tilt up slightly
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
        rel_vel = -nearest_vel
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
# Collision Recorder
# ============================================================================
class CollisionRecorder:
    COOLDOWN = 3.0
    MIN_IMPULSE = 200.0

    def __init__(self, vehicle, world):
        self.collisions = []
        self._last = {}
        bp = world.get_blueprint_library().find('sensor.other.collision')
        self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=vehicle)
        self.sensor.listen(self._on_collision)

    def _on_collision(self, event):
        now = time.time()
        actor_type = event.other_actor.type_id
        impulse = event.normal_impulse.length()
        if not (actor_type.startswith('vehicle.') or actor_type.startswith('walker.')):
            return
        if impulse < self.MIN_IMPULSE:
            return
        aid = event.other_actor.id
        if aid in self._last and now - self._last[aid] < self.COOLDOWN:
            return
        self._last[aid] = now
        self.collisions.append({
            'time': now, 'actor': actor_type, 'impulse': impulse
        })
        print(f"\n  💥 COLLISION! Hit {actor_type} ({impulse:.0f} N·s)")

    def cleanup(self):
        if self.sensor and self.sensor.is_alive:
            self.sensor.destroy()


# ============================================================================
# Waypoint-based steering fallback (if BasicAgent not available)
# ============================================================================
def compute_waypoint_steer(ego, carla_map):
    """Manual fallback if BasicAgent fails."""
    vel = ego.get_velocity()
    speed = math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)
    lookahead = max(3.0, min(12.0, speed * 1.5))
    wp = carla_map.get_waypoint(ego.get_location(), project_to_road=True)
    if not wp:
        return 0.0

    target_wp = wp
    remaining = lookahead
    while remaining > 0:
        step = min(2.0, remaining)
        nexts = target_wp.next(step)
        if not nexts:
            break
        target_wp = nexts[0]
        remaining -= step

    ego_tf = ego.get_transform()
    target_yaw = math.radians(target_wp.transform.rotation.yaw)
    ego_yaw = math.radians(ego_tf.rotation.yaw)
    err = target_yaw - ego_yaw
    err = (err + math.pi) % (2 * math.pi) - math.pi
    steer = err / math.radians(60)
    return max(-0.7, min(0.7, steer))


# ============================================================================
# Spawn test obstacles
# ============================================================================
def spawn_stopped_vehicle(world, ego, carla_map, ahead_m=35):
    """Spawn a stopped vehicle ahead of ego on the same lane."""
    wp = carla_map.get_waypoint(ego.get_location(), project_to_road=True)
    if not wp:
        return None
    for _ in range(int(ahead_m / 3)):
        nwps = wp.next(3.0)
        if not nwps:
            return None
        wp = nwps[0]

    bp_lib = world.get_blueprint_library()
    vbp = random.choice([bp for bp in bp_lib.filter('vehicle.*')
                         if int(bp.get_attribute('number_of_wheels')) >= 4])
    tf = wp.transform
    tf.location.z += 0.5
    v = world.try_spawn_actor(vbp, tf)
    if v:
        v.apply_control(carla.VehicleControl(brake=1.0))
        v.set_target_velocity(carla.Vector3D(0, 0, 0))
        print(f"  🚧 Stopped vehicle spawned {ahead_m}m ahead")
    return v


def spawn_sudden_braker(world, ego, carla_map, ahead_m=25):
    """Spawn a vehicle ahead that drives then brakes suddenly."""
    wp = carla_map.get_waypoint(ego.get_location(), project_to_road=True)
    if not wp:
        return None
    for _ in range(int(ahead_m / 3)):
        nwps = wp.next(3.0)
        if not nwps:
            return None
        wp = nwps[0]

    bp_lib = world.get_blueprint_library()
    vbp = random.choice([bp for bp in bp_lib.filter('vehicle.*')
                         if int(bp.get_attribute('number_of_wheels')) >= 4])
    tf = wp.transform
    tf.location.z += 0.5
    v = world.try_spawn_actor(vbp, tf)
    if v:
        fwd = tf.get_forward_vector()
        v.enable_constant_velocity(carla.Vector3D(fwd.x * 8.0, fwd.y * 8.0, 0))
        print(f"  🚗💨 Moving vehicle spawned {ahead_m}m ahead (will brake in ~4s)")
    return v


def spawn_pedestrian_crossing(world, ego, carla_map, ahead_m=20):
    """Spawn a pedestrian crossing the road ahead."""
    wp = carla_map.get_waypoint(ego.get_location(), project_to_road=True)
    if not wp:
        return None, None
    for _ in range(int(ahead_m / 3)):
        nwps = wp.next(3.0)
        if not nwps:
            return None, None
        wp = nwps[0]

    bp_lib = world.get_blueprint_library()
    wbp = random.choice(bp_lib.filter('walker.pedestrian.*'))
    if wbp.has_attribute('is_invincible'):
        wbp.set_attribute('is_invincible', 'false')

    tf = wp.transform
    right = tf.get_right_vector()
    tf.location.x -= right.x * 4
    tf.location.y -= right.y * 4
    tf.location.z += 0.5

    walker = world.try_spawn_actor(wbp, tf)
    ctrl = None
    if walker:
        ctrl_bp = bp_lib.find('controller.ai.walker')
        ctrl = world.spawn_actor(ctrl_bp, carla.Transform(), attach_to=walker)
        world.tick()
        # Walk across the road
        dest = wp.transform.location + carla.Location(
            x=right.x * 8, y=right.y * 8)
        ctrl.start()
        ctrl.go_to_location(dest)
        ctrl.set_max_speed(1.8)
        print(f"  🚶 Pedestrian crossing road {ahead_m}m ahead")
    return walker, ctrl


# ============================================================================
# Spawn background traffic
# ============================================================================
def spawn_background_traffic(world, client, tm, count):
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
    print(f"  🚗 {len(ids)} background vehicles")
    return ids


def spawn_background_pedestrians(world, count):
    bp_lib = world.get_blueprint_library()
    walker_bps = bp_lib.filter('walker.pedestrian.*')
    ctrl_bp = bp_lib.find('controller.ai.walker')

    walkers, controllers = [], []
    for _ in range(count):
        bp = random.choice(walker_bps)
        if bp.has_attribute('is_invincible'):
            bp.set_attribute('is_invincible', 'false')
        loc = world.get_random_location_from_navigation()
        if loc:
            w = world.try_spawn_actor(bp, carla.Transform(loc))
            if w:
                walkers.append(w)
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

    print(f"  🚶 {len(walkers)} background pedestrians")
    return [w.id for w in walkers], [c.id for c in controllers]


# ============================================================================
# Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description='Test throttle/brake MLP with BasicAgent steering')
    parser.add_argument('--host', default=CARLA_HOST)
    parser.add_argument('--port', type=int, default=CARLA_PORT)
    parser.add_argument('--model', default=MODEL_PATH)
    parser.add_argument('--scaler', default=SCALER_PATH)
    parser.add_argument('--duration', type=int, default=180, help='Test duration in seconds')
    parser.add_argument('--vehicles', type=int, default=20, help='Background NPC vehicles')
    parser.add_argument('--pedestrians', type=int, default=10, help='Background pedestrians')
    args = parser.parse_args()

    total_frames = args.duration * FPS

    # ---- Load model ----
    print("=" * 70)
    print("THROTTLE/BRAKE LIVE TEST")
    print("=" * 70)
    print(f"  Model:     {args.model}")
    print(f"  Scaler:    {args.scaler}")
    print(f"  Duration:  {args.duration}s")
    print(f"  Vehicles:  {args.vehicles}")
    print(f"  Peds:      {args.pedestrians}")
    print()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = ThrottleBrakeMLP(input_dim=len(FEATURE_COLS)).to(device)
    state = torch.load(args.model, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    print(f"  ✅ Model loaded on {device}")

    with open(args.scaler, 'rb') as f:
        scaler = pickle.load(f)
    print(f"  ✅ Scaler loaded")

    # ---- Connect to CARLA ----
    client = carla.Client(args.host, args.port)
    client.set_timeout(30.0)

    world = client.get_world()
    cur_map = world.get_map().name.split('/')[-1]
    if cur_map != TOWN:
        print(f"\n  🗺️  Loading {TOWN}...")
        world = client.load_world(TOWN)
        time.sleep(3)

    original_settings = world.get_settings()
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 1.0 / FPS
    world.apply_settings(settings)

    tm = client.get_trafficmanager(8000)
    tm.set_synchronous_mode(True)
    world.tick()

    carla_map = world.get_map()
    spawn_points = carla_map.get_spawn_points()

    # ---- Filter spawn points ----
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

    # Tick the world once so the ego vehicle's location is updated from (0,0,0) to its actual spawn point
    # This is critical for BasicAgent to generate a correct route!
    world.tick()

    print(f"\n  🚗 Ego spawned: {ego.type_id}")
    print(f"     Location: ({ego.get_location().x:.0f}, {ego.get_location().y:.0f})")


    # ---- Setup steering controller (BasicAgent for routing) ----
    agent = None
    if AGENT_AVAILABLE:
        agent = BasicAgent(ego, target_speed=30)
        agent.ignore_traffic_lights(active=False)
        agent.ignore_stop_signs(active=False)
        # Give it a random destination far away
        destination = random.choice(spawn_points).location
        agent.set_destination(destination)
        print(f"  🧭 BasicAgent routing initialized")
    else:
        print(f"  🧭 Using manual waypoint steering fallback")

    # ---- Sensors ----
    radar = FrontRadar(ego, world, MAX_RADAR_RANGE)
    collision = CollisionRecorder(ego, world)

    # ---- Camera + YOLO ----
    camera = CameraManager(ego, world)
    yolo = None
    if YOLO_AVAILABLE:
        yolo = YOLOPerception()
        print(f"  📷 RGB camera + YOLOv8 traffic light detector attached")
    else:
        print(f"  📷 RGB camera attached (YOLO unavailable — traffic light state will be 0)")
    print(f"  📡 Front radar + collision sensor attached")

    # ---- Background traffic ----
    npc_ids = spawn_background_traffic(world, client, tm, args.vehicles)
    walker_ids, ctrl_ids = spawn_background_pedestrians(world, args.pedestrians)

    # Let settle
    for _ in range(40):
        world.tick()

    # ---- Test loop ----
    print(f"\n{'=' * 70}")
    print(f"  🏁 LIVE TEST STARTED — {args.duration}s")
    print(f"  BasicAgent steering + MLP throttle/brake")
    print(f"  Press Ctrl+C to stop")
    print(f"{'=' * 70}")
    print()
    print(f"  {'Frame':>7} │ {'Speed':>8} │ {'Dist':>6} │ {'TTC':>6} │ "
          f"{'Action':>7} │ {'Throttle':>8} │ {'Brake':>6} │ Status")
    print(f"  {'─'*7}─┼─{'─'*8}─┼─{'─'*6}─┼─{'─'*6}─┼─"
          f"{'─'*7}─┼─{'─'*8}─┼─{'─'*6}─┼─{'─'*20}")

    prev_speed = 0.0

    # Stats
    total_brake_frames = 0
    total_throttle_frames = 0
    near_miss_count = 0  # got close (<5m) but didn't collide
    min_distance_seen = MAX_RADAR_RANGE

    # ---- Manual obstacle spawning: press Enter ----
    demo_obstacle = None
    demo_count = 0
    demo_stopped_frames = 0

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
    print(f"  ⌨️  Press ENTER to spawn an obstacle ahead!")

    try:
        for frame in range(total_frames):
            world.tick()

            # ---- Ego state ----
            try:
                v = ego.get_velocity()
                speed = math.sqrt(v.x**2 + v.y**2 + v.z**2)
                # CRITICAL FIX: Match training script's signed acceleration exactly
                accel = (speed - prev_speed) * FPS if frame > 0 else 0.0
                prev_speed = speed
            except:
                print("  ⚠️  Ego lost!")
                break

            # ---- Radar ----
            radar.update_ego_speed(speed)
            r = radar.get()

            if r['relative_velocity'] > 0.1:
                ttc = min(r['distance'] / r['relative_velocity'], 10.0)
            else:
                ttc = 10.0

            min_distance_seen = min(min_distance_seen, r['distance'])

            # ---- YOLO traffic light + intersection detection ----
            tl_state = 0
            approaching_jct = 0
            if yolo is not None:
                cam_frame = camera.get_frame()
                if cam_frame is not None:
                    tl_state, tl_conf, tl_bbox = yolo.detect_traffic_light(cam_frame)
                    approaching_jct = yolo.detect_intersection()  # uses cached YOLO results

            # ---- MLP prediction ----
            # Get target speed: speed_limit * 1.10 (matches collector)
            try:
                target_speed = (ego.get_speed_limit() * 1.10) / 3.6  # km/h → m/s
            except:
                target_speed = 10.0

            features = np.array([[
                speed,
                target_speed,
                max(-20, min(20, accel)),
                r['distance'],
                r['relative_velocity'],
                ttc,
                r['obstacle_speed'],
                approaching_jct,
                tl_state,
            ]], dtype=np.float32)

            scaled = scaler.transform(features)
            with torch.no_grad():
                tensor = torch.tensor(scaled, device=device)
                output = model(tensor)  # shape: (1, 2) → [throttle, brake]
                throttle = output[0, 0].item()
                brake = output[0, 1].item()

            # Clamp to [0, 1]
            throttle = max(0.0, min(1.0, throttle))
            brake = max(0.0, min(1.0, brake))

            # Apply both pedals directly — CARLA physics resolves the conflict.
            # The old "winner-takes-all" filter was killing subtle brake signals
            # (e.g. brake=0.35 was zeroed because throttle=0.38).

            # ---- Bad Data Override ----
            # Because the model was trained on TM data, it learned to hold the brake
            # indefinitely when stopped and there is ANY obstacle within 40m.
            # We override this to "creep" forward if the obstacle is > 12m away.
            override = ""
            if speed < 0.5 and r['distance'] > 12.0 and brake > 0.1:
                throttle = 0.35
                brake = 0.0
                override = "  [CREEP OVERRIDE]"

            # ---- Get steering (BasicAgent) ----
            if agent:
                if agent.done():
                    agent.set_destination(random.choice(safe_spawns).location)
                agent_control = agent.run_step()
                steer = agent_control.steer
            else:
                steer = compute_waypoint_steer(ego, carla_map)

            action = throttle - brake  # for display only

            # Neural network is 100% in control of the pedals.
            override = ""

            # ---- Apply hybrid control ----
            ego.apply_control(carla.VehicleControl(
                throttle=throttle,
                steer=steer,
                brake=brake,
            ))

            # ---- Stats ----
            if brake > 0.05:
                total_brake_frames += 1
            if throttle > 0.05:
                total_throttle_frames += 1
            if r['distance'] < 5.0 and speed > 0.5:
                near_miss_count += 1

            # ---- Manual obstacle: spawn on Enter / auto-remove ----
            if (spawn_requested.is_set()
                    and demo_obstacle is None
                    and speed > 5.0):
                spawn_requested.clear()
                try:
                    wp = carla_map.get_waypoint(ego.get_location(), project_to_road=True)
                    spawn_dist = random.uniform(60.0, 80.0)
                    fwd_wps = wp.next(spawn_dist)
                    if fwd_wps:
                        obs_bp = random.choice([
                            bp for bp in world.get_blueprint_library().filter('vehicle.*')
                            if int(bp.get_attribute('number_of_wheels')) == 4
                        ])
                        obs_tf = fwd_wps[0].transform
                        obs_tf.location.z += 0.5
                        demo_obstacle = world.try_spawn_actor(obs_bp, obs_tf)
                        if demo_obstacle:
                            demo_obstacle.apply_control(carla.VehicleControl(brake=1.0))
                            demo_count += 1
                            demo_stopped_frames = 0
                            print(f"\n  🚧 OBSTACLE #{demo_count}: Spawned {spawn_dist:.0f}m ahead! "
                                  f"(ego at {speed*3.6:.0f} km/h) — Model should BRAKE.\n")
                except:
                    pass
            elif spawn_requested.is_set():
                spawn_requested.clear()
                if demo_obstacle is not None:
                    print("  ⚠️  Obstacle already active — wait for auto-removal")
                else:
                    print("  ⚠️  Ego too slow — speed up first!")

            # Auto-remove after ego stops for 2s
            if demo_obstacle is not None:
                try:
                    demo_obstacle.apply_control(carla.VehicleControl(brake=1.0))
                except:
                    pass
                if speed < 0.3:
                    demo_stopped_frames += 1
                else:
                    demo_stopped_frames = 0
                if demo_stopped_frames >= FPS * 2:
                    try:
                        demo_obstacle.destroy()
                    except:
                        pass
                    demo_obstacle = None
                    demo_stopped_frames = 0
                    print(f"\n  ✅ OBSTACLE #{demo_count}: Removed! "
                          f"Model should RESUME. Press ENTER for another.\n")


            # ---- Spectator ----
            try:
                spec = world.get_spectator()
                tf = ego.get_transform()
                spec.set_transform(carla.Transform(
                    tf.location - tf.get_forward_vector() * 12 + carla.Location(z=6),
                    carla.Rotation(pitch=-20, yaw=tf.rotation.yaw)))
            except:
                pass

            # ---- Print ----
            if frame % (FPS * 1) == 0:  # every second
                spd_kmh = speed * 3.6

                if brake > 0.3:
                    status = "🔴 BRAKING"
                elif brake > 0.05:
                    status = "🟡 SLOWING"
                elif throttle > 0.5:
                    status = "🟢 CRUISING"
                elif speed < 0.3:
                    status = "⬜ STOPPED"
                else:
                    status = "🔵 COASTING"

                tl_str = TL_STATE_NAMES.get(tl_state, '?')
                jct_str = '⚠️ JCT' if approaching_jct else ''

                print(
                    f"  {frame:>7,} │ {spd_kmh:5.1f}km/h │ {r['distance']:5.1f}m │ "
                    f"{ttc:5.1f}s │ {action:+6.3f} │ {throttle:8.3f} │ {brake:5.3f} │ STR:{steer:+5.2f} │ "
                    f"TL:{tl_str:6s} │ {jct_str:>6s} │ {status}{override}"
                )

    except KeyboardInterrupt:
        print(f"\n  ⚠️  Stopped at frame {frame}")

    # ---- Summary ----
    total = max(1, frame + 1)
    num_collisions = len(collision.collisions)

    print(f"\n{'=' * 70}")
    print(f"TEST RESULTS")
    print(f"{'=' * 70}")
    print(f"  Duration:          {total / FPS:.0f}s ({total:,} frames)")
    print(f"  Collisions:        {num_collisions}")
    print(f"  Near misses (<5m): {near_miss_count} frames")
    print(f"  Min distance seen: {min_distance_seen:.1f}m")
    print(f"  Braking frames:    {total_brake_frames} ({100*total_brake_frames/total:.1f}%)")
    print(f"  Throttle frames:   {total_throttle_frames} ({100*total_throttle_frames/total:.1f}%)")

    if num_collisions == 0:
        print(f"\n  ✅ ZERO COLLISIONS — model successfully braked for all obstacles!")
    else:
        print(f"\n  ❌ {num_collisions} collisions occurred:")
        for c in collision.collisions:
            print(f"    → Hit {c['actor']} ({c['impulse']:.0f} N·s)")

    print(f"{'=' * 70}")

    # ---- Cleanup ----
    print("\n  Cleaning up...")
    radar.cleanup()
    collision.cleanup()
    camera.cleanup()


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

    world.apply_settings(original_settings)
    tm.set_synchronous_mode(False)
    print("  ✅ Done!")


if __name__ == '__main__':
    main()
