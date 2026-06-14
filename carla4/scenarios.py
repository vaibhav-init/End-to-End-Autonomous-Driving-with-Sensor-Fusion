#!/usr/bin/env python3
"""
Scenario Tests for Throttle/Brake MLP
=======================================

Runs specific, repeatable driving scenarios to test if the trained MLP
can handle critical situations.

Usage:
    python scenarios.py                    # Run all scenarios
    python scenarios.py --scenario 1       # Run scenario #1 only

Scenarios:
    1. Stutter Stop — Ego follows lead vehicle on autopilot.
       Lead suddenly slams brakes. Ego must stop in time.
"""

import os, sys, math, time, random, argparse
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
    from agents.navigation.controller import PIDLateralController
    PID_AVAILABLE = True
except ImportError:
    PID_AVAILABLE = False

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


# ============================================================================
# MLP Model (must match training)
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
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


# ============================================================================
# Sensors
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
        bp.set_attribute('horizontal_fov', '30')
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

    def cleanup(self):
        if self.sensor and self.sensor.is_alive:
            self.sensor.destroy()


class CollisionRecorder:
    COOLDOWN = 2.0

    def __init__(self, vehicle, world):
        self.collisions = []
        self._last = {}
        bp = world.get_blueprint_library().find('sensor.other.collision')
        self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=vehicle)
        self.sensor.listen(self._on_collision)

    def _on_collision(self, event):
        actor_type = event.other_actor.type_id
        impulse = event.normal_impulse
        impulse = math.sqrt(impulse.x**2 + impulse.y**2 + impulse.z**2)
        now = time.time()
        aid = event.other_actor.id
        if aid in self._last and now - self._last[aid] < self.COOLDOWN:
            return
        self._last[aid] = now
        self.collisions.append({
            'time': now, 'actor': actor_type, 'impulse': impulse
        })

    def cleanup(self):
        if self.sensor and self.sensor.is_alive:
            self.sensor.destroy()


# ============================================================================
# Steering helper
# ============================================================================
def compute_steer(ego, carla_map, lat_controller=None):
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

    if lat_controller:
        steer = lat_controller.run_step(target_wp)
        return max(-0.7, min(0.7, steer))

    ego_tf = ego.get_transform()
    target_yaw = math.radians(target_wp.transform.rotation.yaw)
    ego_yaw = math.radians(ego_tf.rotation.yaw)
    err = target_yaw - ego_yaw
    err = (err + math.pi) % (2 * math.pi) - math.pi
    steer = err / math.radians(60)
    return max(-0.7, min(0.7, steer))


# ============================================================================
# Helper: run model for one frame
# ============================================================================
def model_step(ego, radar, model, scaler, device, carla_map, lat_controller, prev_speed,
               yolo=None, camera=None):
    """Run one frame of model inference. Returns (throttle, brake, steer, speed, r, ttc, tl_state, approaching_jct)."""
    vel = ego.get_velocity()
    speed = math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)
    accel = (speed - prev_speed) * FPS

    radar.update_ego_speed(speed)
    r = radar.latest.copy()
    ttc = r['distance'] / max(0.1, r['relative_velocity']) if r['relative_velocity'] > 0.1 else 10.0

    try:
        target_speed = (ego.get_speed_limit() * 1.10) / 3.6
    except:
        target_speed = 9.17

    # YOLO traffic light + intersection detection
    tl_state = 0
    approaching_jct = 0
    if yolo is not None and camera is not None:
        cam_frame = camera.get_frame()
        if cam_frame is not None:
            tl_state, _, _ = yolo.detect_traffic_light(cam_frame)
            approaching_jct = yolo.detect_intersection()  # uses cached YOLO results

    features = np.array([[
        speed, target_speed, max(-20, min(20, accel)),
        r['distance'], r['relative_velocity'], ttc, r['obstacle_speed'],
        approaching_jct, tl_state,
    ]], dtype=np.float32)

    scaled = scaler.transform(features)
    with torch.no_grad():
        tensor = torch.tensor(scaled, device=device)
        output = model(tensor)
        throttle = max(0.0, min(1.0, output[0, 0].item()))
        brake = max(0.0, min(1.0, output[0, 1].item()))

    # Winner-takes-all
    if brake > throttle:
        throttle = 0.0
    else:
        brake = 0.0

    steer = compute_steer(ego, carla_map, lat_controller)

    ego.apply_control(carla.VehicleControl(
        throttle=throttle, steer=steer, brake=brake
    ))

    return throttle, brake, steer, speed, r, ttc, tl_state, approaching_jct


# ============================================================================
# Helper: update spectator
# ============================================================================
def update_spectator(world, ego):
    try:
        spec = world.get_spectator()
        tf = ego.get_transform()
        fv = tf.get_forward_vector()
        spec.set_transform(carla.Transform(
            tf.location + carla.Location(x=-8*fv.x, y=-8*fv.y, z=5),
            carla.Rotation(pitch=-20, yaw=tf.rotation.yaw)
        ))
    except:
        pass


# ============================================================================
# Scenario 1: Stutter Stop
# ============================================================================
def scenario_stutter_stop(client, world, model, scaler, device):
    """
    Ego follows a lead vehicle driven by Traffic Manager autopilot.
    After both are cruising at ~20 km/h, the lead's autopilot is killed
    and full brakes are applied. The ego must stop without collision.
    """
    print("\n" + "=" * 70)
    print("  SCENARIO 1: STUTTER STOP")
    print("  Ego follows a lead vehicle. Lead suddenly slams brakes.")
    print("  Test: Can the model react to sudden deceleration?")
    print("=" * 70)

    carla_map = world.get_map()
    spawn_points = carla_map.get_spawn_points()
    tm = client.get_trafficmanager(8000)

    # ---- Spawn ego (random spawn point for variety) ----
    ego_bp = world.get_blueprint_library().find('vehicle.tesla.model3')
    ego = None
    ego_idx = 0
    random.shuffle(spawn_points)
    for idx in range(len(spawn_points)):
        ego = world.try_spawn_actor(ego_bp, spawn_points[idx])
        if ego:
            ego_idx = idx
            break
    if not ego:
        print("  ❌ Failed to spawn ego!")
        return False

    print(f"  🚗 Ego spawned (spawn point #{ego_idx})")

    # ---- Spawn lead directly in front of ego using waypoints ----
    # Get the road waypoint where the ego is currently sitting
    ego_wp = carla_map.get_waypoint(spawn_points[ego_idx].location)

    # Project 15 meters forward down the exact same lane
    next_wps = ego_wp.next(15.0)

    if not next_wps:
        print("  ❌ No road ahead to spawn lead vehicle! Try a different ego_idx.")
        ego.destroy()
        return False

    lead_tf = next_wps[0].transform
    lead_tf.location.z += 0.5  # Bump Z up slightly to prevent ground clipping

    lead_bp = world.get_blueprint_library().find('vehicle.audi.a2')
    lead = world.try_spawn_actor(lead_bp, lead_tf)

    if not lead:
        print("  ❌ Failed to spawn lead vehicle at the forward waypoint!")
        ego.destroy()
        return False

    # ---- Put lead on TM autopilot (drives naturally on the road) ----
    lead.set_autopilot(True, tm.get_port())
    tm.set_desired_speed(lead, 20)          # 20 km/h
    tm.distance_to_leading_vehicle(lead, 10.0)
    tm.vehicle_percentage_speed_difference(lead, -10)  # slightly faster

    # Let positions settle
    for _ in range(10):
        world.tick()

    lead_dist = ego.get_location().distance(lead.get_location())
    print(f"  🚙 Lead on autopilot, {lead_dist:.0f}m ahead")

    # ---- Setup sensors ----
    radar = FrontRadar(ego, world, MAX_RADAR_RANGE)
    collision = CollisionRecorder(ego, world)
    lat_controller = None
    if PID_AVAILABLE:
        lat_controller = PIDLateralController(ego, K_P=1.95, K_D=0.2, K_I=0.05, dt=1.0/FPS)

    # Camera + YOLO
    camera = CameraManager(ego, world)
    yolo = YOLOPerception() if YOLO_AVAILABLE else None

    # ============================================================
    # PHASE 1: Warm-up — both drive for 10s to build speed
    # ============================================================
    WARMUP_S = 10
    print(f"\n  ⏳ Phase 1: Following lead for {WARMUP_S}s (building speed)...\n")

    prev_speed = 0.0
    for frame in range(FPS * WARMUP_S):
        world.tick()
        throttle, brake, steer, speed, r, ttc, tl_state, approaching_jct = model_step(
            ego, radar, model, scaler, device, carla_map, lat_controller, prev_speed,
            yolo=yolo, camera=camera)
        prev_speed = speed
        update_spectator(world, ego)

        if frame % (FPS * 2) == 0:  # Print every 2s
            t = frame / FPS
            print(f"  [Warmup {t:4.0f}s] {speed*3.6:5.1f} km/h │ "
                  f"dist: {r['distance']:5.1f}m │ thr: {throttle:.3f} │ brk: {brake:.3f}")

    ego_speed_kmh = speed * 3.6
    print(f"\n  ✅ Warm-up done — ego at {ego_speed_kmh:.0f} km/h\n")

    # ============================================================
    # PHASE 2: Lead slams brakes
    # ============================================================
    lead.set_autopilot(False)
    lead.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))
    print("  🛑 LEAD SLAMMED BRAKES!\n")

    BRAKE_S = 8
    min_dist = 999.0

    print(f"  {'Frame':>6} │ {'Time':>5} │ {'Speed':>8} │ {'Dist':>6} │ {'TTC':>6} │ "
          f"{'THR':>5} │ {'BRK':>5} │ Phase")
    print(f"  {'─'*6}─┼─{'─'*5}─┼─{'─'*8}─┼─{'─'*6}─┼─{'─'*6}─┼─"
          f"{'─'*5}─┼─{'─'*5}─┼─{'─'*20}")

    for frame in range(FPS * BRAKE_S):
        world.tick()

        # Keep lead braking
        if lead and lead.is_alive:
            lead.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))

        throttle, brake, steer, speed, r, ttc, tl_state, approaching_jct = model_step(
            ego, radar, model, scaler, device, carla_map, lat_controller, prev_speed,
            yolo=yolo, camera=camera)
        prev_speed = speed
        min_dist = min(min_dist, r['distance'])
        update_spectator(world, ego)

        if frame % 10 == 0:
            t = frame / FPS
            print(f"  {frame:6d} │ {t:5.1f}s │ {speed*3.6:6.1f}km/h │ {r['distance']:5.1f}m │ "
                  f"{ttc:5.1f}s │ {throttle:5.3f} │ {brake:5.3f} │ EGO_MUST_STOP")

    # ============================================================
    # PHASE 3: Remove lead, ego resumes
    # ============================================================
    if lead and lead.is_alive:
        lead.destroy()
        lead = None
    print(f"\n  ✅ Lead removed — ego should resume\n")

    RESUME_S = 7
    for frame in range(FPS * RESUME_S):
        world.tick()
        throttle, brake, steer, speed, r, ttc, tl_state, approaching_jct = model_step(
            ego, radar, model, scaler, device, carla_map, lat_controller, prev_speed,
            yolo=yolo, camera=camera)
        prev_speed = speed
        update_spectator(world, ego)

        if frame % (FPS) == 0:
            t = frame / FPS
            print(f"  [Resume {t:4.0f}s] {speed*3.6:5.1f} km/h │ "
                  f"dist: {r['distance']:5.1f}m │ thr: {throttle:.3f} │ brk: {brake:.3f}")

    # ---- Results ----
    collisions = len(collision.collisions)
    passed = collisions == 0

    print(f"\n{'=' * 70}")
    print(f"  SCENARIO 1 RESULTS: STUTTER STOP")
    print(f"{'=' * 70}")
    print(f"  Collisions:     {collisions}")
    print(f"  Min distance:   {min_dist:.1f}m")
    if passed:
        print(f"\n  ✅ PASSED — Model braked in time for sudden stop!")
    else:
        for c in collision.collisions:
            print(f"  ❌ FAILED — Hit {c['actor']} ({c['impulse']:.0f} N·s)")
    print(f"{'=' * 70}\n")

    # Cleanup
    radar.cleanup()
    collision.cleanup()
    camera.cleanup()
    if ego and ego.is_alive:
        ego.destroy()
    if lead and lead.is_alive:
        lead.destroy()

    return passed


# ============================================================================
# Scenario 2: Red Light Stop
# ============================================================================
def scenario_red_light_stop(client, world, model, scaler, device):
    """
    Ego approaches an intersection with a red traffic light.
    The model must detect the red light and stop before the intersection.
    """
    print("\n" + "=" * 70)
    print("  SCENARIO 2: RED LIGHT STOP")
    print("  Ego approaches an intersection. Traffic light turns red.")
    print("  Test: Can the model stop at a red light?")
    print("=" * 70)

    carla_map = world.get_map()
    spawn_points = carla_map.get_spawn_points()

    # ---- Find a spawn point near a traffic light ----
    traffic_lights = world.get_actors().filter('traffic.traffic_light')
    tl_list = list(traffic_lights)

    if not tl_list:
        print("  ⚠️  No traffic lights found in this map!")
        return None

    # Find spawn points close to traffic lights (within 60m)
    tl_spawns = []
    for sp in spawn_points:
        wp = carla_map.get_waypoint(sp.location, project_to_road=True)
        if wp and not wp.is_junction:
            for tl in tl_list:
                dist = sp.location.distance(tl.get_location())
                if 30 < dist < 60:
                    tl_spawns.append((sp, tl, dist))
                    break

    if not tl_spawns:
        print("  ⚠️  No suitable spawn point near a traffic light!")
        return None

    # Pick the best candidate
    random.shuffle(tl_spawns)
    ego_sp, target_tl, tl_dist = tl_spawns[0]

    # ---- Spawn ego ----
    ego_bp = world.get_blueprint_library().find('vehicle.tesla.model3')
    ego = world.try_spawn_actor(ego_bp, ego_sp)
    if not ego:
        print("  ❌ Failed to spawn ego!")
        return False

    print(f"  🚗 Ego spawned {tl_dist:.0f}m from traffic light")

    # Force the traffic light to red
    target_tl.set_state(carla.TrafficLightState.Red)
    target_tl.set_red_time(30.0)  # Hold red for 30s
    print(f"  🔴 Traffic light forced to RED")

    world.tick()

    # ---- Sensors ----
    radar = FrontRadar(ego, world, MAX_RADAR_RANGE)
    collision = CollisionRecorder(ego, world)
    lat_controller = None
    if PID_AVAILABLE:
        lat_controller = PIDLateralController(ego, K_P=1.95, K_D=0.2, K_I=0.05, dt=1.0/FPS)

    camera = CameraManager(ego, world)
    yolo = YOLOPerception() if YOLO_AVAILABLE else None

    # ---- Phase 1: Accelerate toward intersection (5s) ----
    ACCEL_S = 5
    print(f"\n  ⏳ Phase 1: Accelerating toward intersection ({ACCEL_S}s)...\n")

    prev_speed = 0.0
    for frame in range(FPS * ACCEL_S):
        world.tick()
        # Manual throttle to get up to speed
        steer = compute_steer(ego, carla_map, lat_controller)
        ego.apply_control(carla.VehicleControl(throttle=0.6, steer=steer, brake=0.0))

        vel = ego.get_velocity()
        speed = math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)
        prev_speed = speed
        update_spectator(world, ego)

    print(f"  ✅ Ego at {speed*3.6:.0f} km/h — handing control to model\n")

    # ---- Phase 2: Model drives, should detect red and stop (10s) ----
    APPROACH_S = 10
    detected_red = False
    stopped_at_red = False
    min_dist_to_tl = 999.0

    print(f"  {'Frame':>6} │ {'Time':>5} │ {'Speed':>8} │ {'TL':>6} │ {'JCT':>4} │ "
          f"{'THR':>5} │ {'BRK':>5} │ Phase")
    print(f"  {'─'*6}─┼─{'─'*5}─┼─{'─'*8}─┼─{'─'*6}─┼─{'─'*4}─┼─"
          f"{'─'*5}─┼─{'─'*5}─┼─{'─'*20}")

    for frame in range(FPS * APPROACH_S):
        world.tick()

        throttle, brake, steer, speed, r, ttc, tl_state, approaching_jct = model_step(
            ego, radar, model, scaler, device, carla_map, lat_controller, prev_speed,
            yolo=yolo, camera=camera)
        prev_speed = speed
        update_spectator(world, ego)

        # Track detection
        if tl_state == 3:  # Red
            detected_red = True
        if detected_red and speed < 0.5:
            stopped_at_red = True

        # Distance to traffic light
        dist_to_tl = ego.get_location().distance(target_tl.get_location())
        min_dist_to_tl = min(min_dist_to_tl, dist_to_tl)

        if frame % 10 == 0:
            t = frame / FPS
            tl_str = TL_STATE_NAMES.get(tl_state, '?')
            print(f"  {frame:6d} │ {t:5.1f}s │ {speed*3.6:6.1f}km/h │ {tl_str:>6s} │ "
                  f"{'Y' if approaching_jct else 'N':>4s} │ {throttle:5.3f} │ {brake:5.3f} │ APPROACH_RED")

    # ---- Results ----
    collisions = len(collision.collisions)
    passed = stopped_at_red and collisions == 0

    print(f"\n{'=' * 70}")
    print(f"  SCENARIO 2 RESULTS: RED LIGHT STOP")
    print(f"{'=' * 70}")
    print(f"  Detected red light: {'✅ Yes' if detected_red else '❌ No'}")
    print(f"  Stopped at red:     {'✅ Yes' if stopped_at_red else '❌ No'}")
    print(f"  Collisions:         {collisions}")
    print(f"  Min dist to light:  {min_dist_to_tl:.1f}m")
    if passed:
        print(f"\n  ✅ PASSED — Model detected red light and stopped!")
    elif not detected_red:
        print(f"\n  ❌ FAILED — YOLO did not detect the red light")
    elif not stopped_at_red:
        print(f"\n  ❌ FAILED — Model did not stop at the red light")
    else:
        for c in collision.collisions:
            print(f"  ❌ FAILED — Hit {c['actor']} ({c['impulse']:.0f} N·s)")
    print(f"{'=' * 70}\n")

    # Cleanup
    radar.cleanup()
    collision.cleanup()
    camera.cleanup()
    if ego and ego.is_alive:
        ego.destroy()

    return passed


# ============================================================================
# Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description='Run throttle/brake scenarios')
    parser.add_argument('--scenario', type=int, default=0,
                        help='Run specific scenario (0 = all)')
    args = parser.parse_args()

    # ---- Load model ----
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ThrottleBrakeMLP(input_dim=9).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.eval()

    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)

    print(f"  ✅ Model loaded on {device}")

    # ---- Connect to CARLA ----
    client = carla.Client(CARLA_HOST, CARLA_PORT)
    client.set_timeout(30.0)
    world = client.get_world()

    if world.get_map().name.split('/')[-1] != TOWN:
        print(f"\n  🗺️  Loading {TOWN}...")
        world = client.load_world(TOWN)

    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 1.0 / FPS
    world.apply_settings(settings)

    tm = client.get_trafficmanager(8000)
    tm.set_synchronous_mode(True)
    world.tick()

    # ---- Run scenarios ----
    scenarios = {
        1: ("Stutter Stop", scenario_stutter_stop),
        2: ("Red Light Stop", scenario_red_light_stop),
    }

    results = {}
    to_run = [args.scenario] if args.scenario > 0 else sorted(scenarios.keys())

    print(f"\n{'=' * 70}")
    print(f"  SCENARIO TEST SUITE")
    print(f"  Running: {', '.join(scenarios[s][0] for s in to_run)}")
    print(f"{'=' * 70}")

    for sid in to_run:
        name, func = scenarios[sid]
        try:
            passed = func(client, world, model, scaler, device)
            results[sid] = ("✅ PASSED" if passed else "❌ FAILED", name)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"  ❌ Scenario {sid} crashed: {e}")
            results[sid] = ("💥 CRASHED", name)

    # ---- Summary ----
    print(f"\n{'=' * 70}")
    print(f"  FINAL RESULTS")
    print(f"{'=' * 70}")
    for sid in sorted(results):
        status, name = results[sid]
        print(f"  Scenario {sid}: {name:30s} {status}")
    print(f"{'=' * 70}\n")


if __name__ == '__main__':
    main()
