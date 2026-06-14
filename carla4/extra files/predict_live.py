#!/usr/bin/env python3
"""
Live Crash Probability Visualizer
==================================

Loads trained MLP model, ego drives normally with TM autopilot.
Shows camera feed with crash probability overlay:
  GREEN = safe (< 0.3)
  YELLOW = caution (0.3 - 0.7)
  RED = danger (> 0.7)

Usage:
    python predict_live.py
"""

import carla
import numpy as np
import math
import time
import random
import torch
import torch.nn as nn
import pickle
import pygame
import sys

# ============================================================================
# Config
# ============================================================================
CARLA_HOST = '127.0.0.1'
CARLA_PORT = 2000
TOWN = 'Town01'
FPS = 20
NPC_COUNT = 40
PED_COUNT = 60

# Random throttle (same as data collector)
P_FULL_THROTTLE = 0.01
P_CRASH_WHEN_STOPPED = 0.0001
THROTTLE_BURST_FRAMES = 2 * FPS  # 2 seconds

MODEL_PATH = 'model/crash_mlp.pt'
SCALER_PATH = 'model/scaler.pkl'

WINDOW_W = 1280
WINDOW_H = 720

FEATURE_COLS = [
    'ego_speed', 'ego_acceleration', 'nearest_distance',
    'relative_velocity', 'ttc', 'obstacle_speed', 'obstacle_type',
    'lateral_offset', 'ego_steering',
    'rear_distance', 'rear_relative_velocity', 'rear_ttc',
    'rear_obstacle_speed', 'rear_obstacle_type'
]


# ============================================================================
# MLP Model (must match training)
# ============================================================================
class CrashMLP(nn.Module):
    def __init__(self, input_dim=14):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


# ============================================================================
# Radar
# ============================================================================
class RadarReader:
    def __init__(self, vehicle, world, range_m=50.0):
        self.data = {'distance': range_m, 'relative_velocity': 0.0,
                     'obstacle_speed': 0.0, 'obstacle_type': 2, 'lateral_offset': 0.0}
        self._ego_speed = 0.0
        self._range = range_m

        bp = world.get_blueprint_library().find('sensor.other.radar')
        bp.set_attribute('horizontal_fov', '30')
        bp.set_attribute('vertical_fov', '10')
        bp.set_attribute('range', str(range_m))
        bp.set_attribute('points_per_second', '1500')
        tf = carla.Transform(carla.Location(x=2.5, z=0.7))
        self.sensor = world.spawn_actor(bp, tf, attach_to=vehicle)
        self.sensor.listen(self._cb)

    def _cb(self, data):
        nd, nv, na = self._range, 0.0, 0.0
        for d in data:
            if abs(d.azimuth) > 0.3 or d.depth < 1.0:
                continue
            if d.depth < nd:
                nd, nv, na = d.depth, d.velocity, d.azimuth
        rv = -nv
        self.data = {
            'distance': nd,
            'relative_velocity': rv,
            'obstacle_speed': max(0, self._ego_speed - rv),
            'obstacle_type': 0 if nd < self._range else 2,
            'lateral_offset': nd * math.sin(na) if nd < self._range else 0.0,
        }

    def update_speed(self, s):
        self._ego_speed = s

    def cleanup(self):
        if self.sensor.is_alive:
            self.sensor.destroy()


# ============================================================================
# Rear Radar
# ============================================================================
class RearRadarReader:
    def __init__(self, vehicle, world, range_m=50.0):
        self.data = {'rear_distance': range_m, 'rear_relative_velocity': 0.0,
                     'rear_obstacle_speed': 0.0, 'rear_obstacle_type': 2}
        self._ego_speed = 0.0
        self._range = range_m

        bp = world.get_blueprint_library().find('sensor.other.radar')
        bp.set_attribute('horizontal_fov', '30')
        bp.set_attribute('vertical_fov', '10')
        bp.set_attribute('range', str(range_m))
        bp.set_attribute('points_per_second', '1500')
        tf = carla.Transform(carla.Location(x=-2.5, z=0.7), carla.Rotation(yaw=180))
        self.sensor = world.spawn_actor(bp, tf, attach_to=vehicle)
        self.sensor.listen(self._cb)

    def _cb(self, data):
        nd, nv = self._range, 0.0
        for d in data:
            if abs(d.azimuth) > 0.3 or d.depth < 1.0:
                continue
            if d.depth < nd:
                nd, nv = d.depth, d.velocity
        rv = -nv
        self.data = {
            'rear_distance': nd,
            'rear_relative_velocity': rv,
            'rear_obstacle_speed': max(0, rv + self._ego_speed),
            'rear_obstacle_type': 0 if nd < self._range else 2,
        }

    def update_speed(self, s):
        self._ego_speed = s

    def cleanup(self):
        if self.sensor.is_alive:
            self.sensor.destroy()


# ============================================================================
# Camera
# ============================================================================
class CameraReader:
    def __init__(self, vehicle, world, w, h):
        self.surface = None
        bp = world.get_blueprint_library().find('sensor.camera.rgb')
        bp.set_attribute('image_size_x', str(w))
        bp.set_attribute('image_size_y', str(h))
        bp.set_attribute('fov', '100')
        tf = carla.Transform(carla.Location(x=-6, z=3), carla.Rotation(pitch=-10))
        self.sensor = world.spawn_actor(bp, tf, attach_to=vehicle)
        self.sensor.listen(self._cb)

    def _cb(self, image):
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))[:, :, :3]
        array = array[:, :, ::-1]  # BGR → RGB
        self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))

    def cleanup(self):
        if self.sensor.is_alive:
            self.sensor.destroy()


# ============================================================================
# Main
# ============================================================================
def main():
    # ---- Load model ----
    print("Loading model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = CrashMLP(input_dim=len(FEATURE_COLS)).to(device)
    state = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
    print(f"  Model loaded on {device}")

    # ---- Pygame ----
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
    pygame.display.set_caption('Crash Probability - Live')
    clock = pygame.time.Clock()
    font_big = pygame.font.SysFont('Arial', 48, bold=True)
    font_med = pygame.font.SysFont('Arial', 28)
    font_small = pygame.font.SysFont('Arial', 22)

    # ---- CARLA ----
    print("Connecting to CARLA...")
    client = carla.Client(CARLA_HOST, CARLA_PORT)
    client.set_timeout(30.0)
    world = client.get_world()

    original_settings = world.get_settings()
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 1.0 / FPS
    world.apply_settings(settings)

    tm = client.get_trafficmanager(8000)
    tm.set_synchronous_mode(True)

    carla_map = world.get_map()

    # Clean up existing actors
    for a in world.get_actors().filter('vehicle.*'):
        a.destroy()
    for a in world.get_actors().filter('walker.*'):
        a.destroy()
    for a in world.get_actors().filter('controller.*'):
        a.destroy()
    world.tick()

    # Spawn ego
    ego_bp = world.get_blueprint_library().find('vehicle.tesla.model3')
    spawns = carla_map.get_spawn_points()
    random.shuffle(spawns)
    ego = None
    for sp in spawns:
        ego = world.try_spawn_actor(ego_bp, sp)
        if ego:
            break
    if not ego:
        print("Failed to spawn ego!")
        return

    ego.set_autopilot(True, tm.get_port())
    # Make ego faster + skip 50% red lights
    tm.vehicle_percentage_speed_difference(ego, -20)
    try:
        tm.ignore_lights_percentage(ego, 50)
    except:
        pass
    print(f"  Ego spawned (fast, skips 50% reds)")

    # Spawn NPCs
    port = tm.get_port()
    veh_bps = [bp for bp in world.get_blueprint_library().filter('vehicle.*')
               if int(bp.get_attribute('number_of_wheels')) >= 4]
    batch = []
    random.shuffle(spawns)
    for i in range(min(NPC_COUNT, len(spawns) - 1)):
        bp = random.choice(veh_bps)
        if bp.has_attribute('color'):
            bp.set_attribute('color', random.choice(bp.get_attribute('color').recommended_values))
        batch.append(
            carla.command.SpawnActor(bp, spawns[i + 1])
            .then(carla.command.SetAutopilot(carla.command.FutureActor, True, port)))
    npc_ids = [r.actor_id for r in client.apply_batch_sync(batch, True) if not r.error]
    print(f"  Spawned {len(npc_ids)} NPCs")

    # Spawn pedestrians
    walker_bps = world.get_blueprint_library().filter('walker.pedestrian.*')
    ctrl_bp = world.get_blueprint_library().find('controller.ai.walker')
    walkers, ctrls = [], []
    for _ in range(PED_COUNT):
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
        ctrls.append(c)
    world.tick()
    for c in ctrls:
        dest = world.get_random_location_from_navigation()
        if dest:
            c.start()
            c.go_to_location(dest)
            c.set_max_speed(1.5)
    print(f"  Spawned {len(walkers)} pedestrians")

    # Sensors
    camera = CameraReader(ego, world, WINDOW_W, WINDOW_H)
    radar = RadarReader(ego, world)
    rear_radar = RearRadarReader(ego, world)

    for _ in range(30):
        world.tick()

    # ---- Main loop ----
    print("\n  Running! Press ESC or close window to quit.\n")
    prev_speed = 0.0
    prob_history = []
    running = True
    frame_count = 0
    overriding = False
    override_end = 0

    try:
        while running:
            # Pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    running = False

            world.tick()
            frame_count += 1

            # ---- Random throttle burst (every second) ----
            if overriding and frame_count >= override_end:
                ego.set_autopilot(True, port)
                overriding = False

            if frame_count % FPS == 0 and not overriding:
                try:
                    _v = ego.get_velocity()
                    cur_speed = math.sqrt(_v.x**2 + _v.y**2 + _v.z**2)
                except:
                    cur_speed = 0
                near_now = radar.data
                if cur_speed < 0.5 and near_now['distance'] < 15:
                    if random.random() < P_CRASH_WHEN_STOPPED:
                        ego.set_autopilot(False, port)
                        overriding = True
                        override_end = frame_count + THROTTLE_BURST_FRAMES
                elif cur_speed > 3.0:
                    if random.random() < P_FULL_THROTTLE:
                        ego.set_autopilot(False, port)
                        overriding = True
                        override_end = frame_count + THROTTLE_BURST_FRAMES

            if overriding:
                # Waypoint-following steering + full throttle
                ego_tf = ego.get_transform()
                ego_wp = carla_map.get_waypoint(ego.get_location(), project_to_road=True)
                if ego_wp:
                    targets = ego_wp.next(8.0)
                    if targets:
                        dx = targets[0].transform.location.x - ego_tf.location.x
                        dy = targets[0].transform.location.y - ego_tf.location.y
                        fwd = ego_tf.get_forward_vector()
                        cross = fwd.x * dy - fwd.y * dx
                        steer = max(-1.0, min(1.0, cross * 0.5))
                    else:
                        steer = 0.0
                else:
                    steer = 0.0
                ego.apply_control(carla.VehicleControl(throttle=1.0, steer=steer, brake=0.0))

            # Read ego state
            vel = ego.get_velocity()
            speed = math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)
            accel = (speed - prev_speed) * FPS
            prev_speed = speed
            ctrl = ego.get_control()

            radar.update_speed(speed)
            near = radar.data

            ttc = near['distance'] / near['relative_velocity'] \
                if near['relative_velocity'] > 0.1 else 10.0
            ttc = min(ttc, 10.0)

            # Rear radar
            rear_radar.update_speed(speed)
            rear = rear_radar.data
            rear_ttc = rear['rear_distance'] / rear['rear_relative_velocity'] \
                if rear['rear_relative_velocity'] > 0.1 else 10.0
            rear_ttc = min(rear_ttc, 10.0)

            # Build feature vector (14 features)
            features = np.array([[
                speed, accel, near['distance'], near['relative_velocity'],
                ttc, near['obstacle_speed'], near['obstacle_type'],
                near['lateral_offset'], ctrl.steer,
                rear['rear_distance'], rear['rear_relative_velocity'],
                rear_ttc, rear['rear_obstacle_speed'], rear['rear_obstacle_type']
            ]], dtype=np.float32)

            # Scale & predict
            features_scaled = scaler.transform(features)
            with torch.no_grad():
                tensor = torch.tensor(features_scaled, device=device)
                prob = model(tensor).item()

            prob_history.append(prob)
            if len(prob_history) > 40:
                prob_history.pop(0)
            avg_prob = np.mean(prob_history[-10:])  # Smooth over 0.5s

            # ---- DRAW ----
            screen.fill((0, 0, 0))

            # Camera feed
            if camera.surface:
                screen.blit(camera.surface, (0, 0))

            # Probability bar background
            bar_x, bar_y = 20, 20
            bar_w, bar_h = 300, 50
            pygame.draw.rect(screen, (30, 30, 30), (bar_x - 5, bar_y - 5, bar_w + 10, bar_h + 80), border_radius=10)
            pygame.draw.rect(screen, (60, 60, 60), (bar_x, bar_y + 45, bar_w, bar_h), border_radius=6)

            # Color based on probability
            if avg_prob < 0.3:
                color = (0, 220, 80)       # GREEN
                label = "SAFE"
            elif avg_prob < 0.7:
                color = (255, 200, 0)      # YELLOW
                label = "CAUTION"
            else:
                color = (255, 40, 40)       # RED
                label = "DANGER"

            # Filled bar
            fill_w = int(bar_w * min(avg_prob, 1.0))
            pygame.draw.rect(screen, color, (bar_x, bar_y + 45, fill_w, bar_h), border_radius=6)

            # Text
            prob_text = font_big.render(f"{avg_prob:.0%}", True, color)
            screen.blit(prob_text, (bar_x, bar_y - 10))

            label_text = font_med.render(label, True, color)
            screen.blit(label_text, (bar_x + 160, bar_y + 2))

            # Feature info
            info_y = bar_y + 110
            info_items = [
                f"Speed: {speed*3.6:.0f} km/h",
                f"Front: {near['distance']:.1f}m  TTC: {ttc:.1f}s",
                f"Rear:  {rear['rear_distance']:.1f}m  TTC: {rear_ttc:.1f}s",
                f"Rel.Vel: {near['relative_velocity']:.1f} m/s",
            ]
            for i, txt in enumerate(info_items):
                surf = font_small.render(txt, True, (220, 220, 220))
                pygame.draw.rect(screen, (20, 20, 20, 180),
                                 (bar_x - 2, info_y + i * 28 - 2, 250, 26), border_radius=4)
                screen.blit(surf, (bar_x, info_y + i * 28))

            pygame.display.flip()
            clock.tick(FPS)

    except KeyboardInterrupt:
        pass

    finally:
        print("Cleaning up...")
        camera.cleanup()
        radar.cleanup()
        rear_radar.cleanup()
        for c in ctrls:
            try:
                c.stop()
                c.destroy()
            except:
                pass
        for w in walkers:
            try:
                w.destroy()
            except:
                pass
        client.apply_batch([carla.command.DestroyActor(x) for x in npc_ids])
        ego.destroy()
        world.apply_settings(original_settings)
        pygame.quit()
        print("Done.")


if __name__ == '__main__':
    main()
