#!/usr/bin/env python3
"""
ACCIDENT-style crash data collector. Spawns NPC vehicles with raw vehicle_control
(no ScenarioRunner) using crash geometries from the ACCIDENT benchmark paper.
Outputs CSV compatible with train_mlp.py.
"""
import os, sys, math, time, random, argparse, traceback
import pandas as pd
import carla

CARLA_HOST = '127.0.0.1'
CARLA_PORT = 2000
FPS = 20
MAX_RADAR_RANGE = 50.0
LOOKAHEAD_SECONDS = 2.0
LOOKAHEAD_FRAMES = int(LOOKAHEAD_SECONDS * FPS)
SAVE_DIR = 'dataset_crash_accident'

VEHICLE_POOL = [
    'vehicle.audi.tt', 'vehicle.tesla.model3', 'vehicle.dodge.charger_2020',
    'vehicle.mercedes.coupe_2020', 'vehicle.nissan.patrol_2021',
    'vehicle.mini.cooper_s_2021', 'vehicle.bmw.grandtourer',
    'vehicle.ford.crown', 'vehicle.lincoln.mkz_2017',
    'vehicle.jeep.wrangler_rubicon', 'vehicle.audi.etron',
]

SCENARIOS = [
    {
        'name': 'Town03_HeadOn', 'town': 'Town03', 'type': 'head_on',
        'bg_vehicles': 50, 'bg_peds': 10,
        'ego': {'spawn_point': 48, 'autopilot': {'ignore_vehicles': 60, 'ignore_lights': 100, 'desired_speed': 25, 'path': [50, 58]}},
        'npc1': {'spawn_point': 50, 'control': {'throttle': 0.9, 'steer': -0.01}},
        'npc2': {'location': (-85.0, -15.37, 2.14), 'rotation': (0.04, 90.68, 0.0),
                 'control': {'throttle': 0.0, 'brake': 1.0}},
    },
    {
        'name': 'Town03_TBone', 'town': 'Town03', 'type': 't_bone',
        'bg_vehicles': 50, 'bg_peds': 10,
        'ego': {'spawn_point': 104, 'autopilot': {'ignore_vehicles': 60, 'ignore_lights': 100, 'desired_speed': 25, 'path': [76, 71]}},
        'npc1': {'spawn_point': 106, 'autopilot': {'ignore_vehicles': 100, 'ignore_signs': 100, 'ignore_lights': 100, 'desired_speed': 20, 'path': [81, 76, 71]}},
        'npc2': {'spawn_point': 49, 'autopilot': {'ignore_vehicles': 100, 'ignore_signs': 100, 'ignore_lights': 100, 'desired_speed': 20, 'path': [58, 71, 76]}},
    },
    {
        'name': 'Town05_HeadOn', 'town': 'Town05', 'type': 'head_on',
        'bg_vehicles': 40, 'bg_peds': 10,
        'ego': {'spawn_point': 149, 'autopilot': {'ignore_vehicles': 60, 'ignore_lights': 100, 'desired_speed': 25, 'path': [127, 122, 110]}},
        'npc1': {'location': (-191.09, -60.63, 0.5), 'rotation': (0, 87.8, 0.0),
                 'control': {'throttle': 0.6}},
        'npc2': {'spawn_point': 26, 'autopilot': {'ignore_vehicles': 0, 'ignore_signs': 100, 'ignore_lights': 100, 'desired_speed': 26, 'path': [149, 127, 122, 110]}},
    },
    {
        'name': 'Town05_SideSwipe', 'town': 'Town05', 'type': 'side_swipe',
        'bg_vehicles': 40, 'bg_peds': 10,
        'ego': {'spawn_point': 42, 'autopilot': {'ignore_vehicles': 60, 'ignore_lights': 100, 'desired_speed': 22, 'path': [100, 109, 121]}},
        'npc1': {'spawn_point': 44, 'autopilot': {'ignore_vehicles': 100, 'ignore_signs': 100, 'ignore_lights': 100, 'desired_speed': 22.0, 'path': [100, 109, 121]}},
        'npc2': {'spawn_point': 40, 'autopilot': {'ignore_vehicles': 100, 'ignore_signs': 100, 'ignore_lights': 100, 'desired_speed': 22.5, 'path': [102, 107, 119]}},
    },
    {
        'name': 'Town05_RearEnd', 'town': 'Town05', 'type': 'rear_end',
        'bg_vehicles': 40, 'bg_peds': 10,
        'ego': {'spawn_point': 98, 'autopilot': {'ignore_vehicles': 80, 'ignore_lights': 100, 'desired_speed': 30, 'path': [129, 90]}},
        'npc1': {'spawn_point': 129, 'autopilot': {'ignore_vehicles': 0, 'ignore_signs': 100, 'ignore_lights': 100, 'desired_speed': 5, 'path': [124, 112]}},
    },
    {
        'name': 'Town01_RearEnd_A', 'town': 'Town01', 'type': 'rear_end',
        'bg_vehicles': 40, 'bg_peds': 10,
        'ego': {'spawn_point': 0, 'autopilot': {'ignore_vehicles': 80, 'ignore_lights': 100, 'desired_speed': 30, 'path': [1, 2, 3]}},
        'npc1': {'spawn_point': 1, 'control': {'throttle': 0.0, 'brake': 1.0}},
    },
    {
        'name': 'Town01_HeadOn', 'town': 'Town01', 'type': 'head_on',
        'bg_vehicles': 40, 'bg_peds': 10,
        'ego': {'spawn_point': 3, 'autopilot': {'ignore_vehicles': 60, 'ignore_lights': 100, 'desired_speed': 25, 'path': [0, 5]}},
        'npc1': {'spawn_point': 0, 'control': {'throttle': 0.8, 'steer': 0.0}},
        'npc2': {'spawn_point': 5, 'control': {'throttle': 0.8, 'steer': 0.0}},
    },
    {
        'name': 'Town01_RearEnd_B', 'town': 'Town01', 'type': 'rear_end',
        'bg_vehicles': 40, 'bg_peds': 10,
        'ego': {'spawn_point': 12, 'autopilot': {'ignore_vehicles': 80, 'ignore_lights': 100, 'desired_speed': 30, 'path': [10, 11]}},
        'npc1': {'spawn_point': 10, 'control': {'throttle': 0.0, 'brake': 1.0}},
    },
    {
        'name': 'Town03_RearEnd', 'town': 'Town03', 'type': 'rear_end',
        'bg_vehicles': 50, 'bg_peds': 10,
        'ego': {'spawn_point': 62, 'autopilot': {'ignore_vehicles': 80, 'ignore_lights': 100, 'desired_speed': 30, 'path': [60, 61]}},
        'npc1': {'spawn_point': 60, 'control': {'throttle': 0.0, 'brake': 1.0}},
    },
    {
        'name': 'Town01_TBone', 'town': 'Town01', 'type': 't_bone',
        'bg_vehicles': 0, 'bg_peds': 0,
        'ego': {'spawn_point': 22, 'autopilot': {'ignore_vehicles': 100, 'ignore_lights': 100, 'desired_speed': 40, 'path': [20, 25]}},
        'npc1': {'spawn_point': 20, 'control': {'brake': 1.0}},
        'npc2': {'spawn_point': 25, 'control': {'brake': 1.0}},
    },
]


class RadarRecorder:
    def __init__(self, vehicle, world, range_m=50.0):
        self.latest = {'distance': range_m, 'relative_velocity': 0.0, 'obstacle_speed': 0.0, 'obstacle_type': 2, 'lateral_offset': 0.0}
        self._ego_speed = 0.0
        self._range = range_m
        bp = world.get_blueprint_library().find('sensor.other.radar')
        bp.set_attribute('horizontal_fov', '30'); bp.set_attribute('vertical_fov', '10')
        bp.set_attribute('range', str(range_m)); bp.set_attribute('points_per_second', '1500')
        tf = carla.Transform(carla.Location(x=2.5, z=0.7), carla.Rotation(pitch=0))
        self.sensor = world.spawn_actor(bp, tf, attach_to=vehicle)
        self.sensor.listen(self._cb)

    def update_ego_speed(self, s): self._ego_speed = s

    def _cb(self, data):
        nd, nv, na = self._range, 0.0, 0.0
        for d in data:
            if abs(d.azimuth) > 0.3 or d.depth < 1.0: continue
            if d.depth < nd: nd, nv, na = d.depth, d.velocity, d.azimuth
        rv = -nv; os_ = max(0, self._ego_speed - rv)
        lo = nd * math.sin(na) if nd < self._range else 0.0
        self.latest = {'distance': nd, 'relative_velocity': rv, 'obstacle_speed': os_, 'obstacle_type': 0 if nd < self._range else 2, 'lateral_offset': lo}

    def get(self): return self.latest.copy()
    def cleanup(self):
        if self.sensor and self.sensor.is_alive: self.sensor.destroy()


class RearRadarRecorder:
    def __init__(self, vehicle, world, range_m=50.0):
        self.latest = {'rear_distance': range_m, 'rear_relative_velocity': 0.0, 'rear_obstacle_speed': 0.0, 'rear_obstacle_type': 2}
        self._ego_speed = 0.0; self._range = range_m
        bp = world.get_blueprint_library().find('sensor.other.radar')
        bp.set_attribute('horizontal_fov', '30'); bp.set_attribute('vertical_fov', '10')
        bp.set_attribute('range', str(range_m)); bp.set_attribute('points_per_second', '1500')
        tf = carla.Transform(carla.Location(x=-2.5, z=0.7), carla.Rotation(pitch=0, yaw=180))
        self.sensor = world.spawn_actor(bp, tf, attach_to=vehicle)
        self.sensor.listen(self._cb)

    def update_ego_speed(self, s): self._ego_speed = s
    def _cb(self, data):
        nd, nv = self._range, 0.0
        for d in data:
            if abs(d.azimuth) > 0.3 or d.depth < 1.0: continue
            if d.depth < nd: nd, nv = d.depth, d.velocity
        rv = -nv; os_ = max(0, rv + self._ego_speed)
        self.latest = {'rear_distance': nd, 'rear_relative_velocity': rv, 'rear_obstacle_speed': os_, 'rear_obstacle_type': 0 if nd < self._range else 2}

    def get(self): return self.latest.copy()
    def cleanup(self):
        if self.sensor and self.sensor.is_alive: self.sensor.destroy()


class CollisionRecorder:
    COOLDOWN = 5.0; MIN_IMPULSE = 300.0
    def __init__(self, vehicle, world):
        self.frames = []; self.details = []; self.counter = [0]; self._last = {}
        bp = world.get_blueprint_library().find('sensor.other.collision')
        self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=vehicle)
        self.sensor.listen(self._cb)

    def _cb(self, event):
        now = time.time(); at = event.other_actor.type_id; imp = event.normal_impulse.length()
        if not (at.startswith('vehicle.') or at.startswith('walker.')): return
        if imp < self.MIN_IMPULSE: return
        aid = event.other_actor.id
        if aid in self._last and now - self._last[aid] < self.COOLDOWN: return
        self._last[aid] = now; f = self.counter[0]
        self.frames.append(f); self.details.append({'frame': f, 'actor': at, 'impulse': imp})
        print(f"\n  💥 COLLISION at frame {f}! {at} ({imp:.0f}N·s)")

    def cleanup(self):
        if self.sensor and self.sensor.is_alive: self.sensor.destroy()


def apply_labels(data, col_frames, lookahead, offset=0):
    for r in data: r['collision_within_2s'] = 0
    for cf in col_frames:
        gcf = cf + offset; start = max(offset, gcf - lookahead)
        for r in data:
            if start <= r['frame_id'] <= gcf: r['collision_within_2s'] = 1


def spawn_npc(world, tm, npc_def, spawn_points):
    bp_lib = world.get_blueprint_library()
    model = random.choice(VEHICLE_POOL)
    bp = bp_lib.find(model)
    if bp.has_attribute('color'):
        bp.set_attribute('color', random.choice(bp.get_attribute('color').recommended_values))

    if 'spawn_point' in npc_def:
        transform = spawn_points[npc_def['spawn_point'] % len(spawn_points)]
    else:
        loc = npc_def['location']; rot = npc_def.get('rotation', (0, 0, 0))
        transform = carla.Transform(carla.Location(x=loc[0], y=loc[1], z=loc[2]),
                                    carla.Rotation(pitch=rot[0], yaw=rot[1], roll=rot[2]))

    vehicle = world.try_spawn_actor(bp, transform)
    if vehicle is None:
        for dx, dy in [(2,0),(-2,0),(0,2),(0,-2),(3,3)]:
            t2 = carla.Transform(transform.location + carla.Location(x=dx, y=dy), transform.rotation)
            vehicle = world.try_spawn_actor(bp, t2)
            if vehicle: break
    if vehicle is None:
        print(f"    ⚠️  Failed to spawn NPC at {transform.location}")
        return None

    if 'autopilot' in npc_def:
        ap = npc_def['autopilot']
        vehicle.set_autopilot(True, 8000)
        tm.ignore_vehicles_percentage(vehicle, ap.get('ignore_vehicles', 0))
        tm.ignore_signs_percentage(vehicle, ap.get('ignore_signs', 0))
        tm.ignore_lights_percentage(vehicle, ap.get('ignore_lights', 0))
        tm.ignore_walkers_percentage(vehicle, ap.get('ignore_walkers', 0))
        if 'desired_speed' in ap:
            tm.set_desired_speed(vehicle, ap['desired_speed'])
        if 'path' in ap:
            path_locs = [spawn_points[sp % len(spawn_points)].location for sp in ap['path']]
            tm.set_path(vehicle, path_locs)
    elif 'control' in npc_def:
        c = npc_def['control']
        vehicle.apply_control(carla.VehicleControl(
            throttle=c.get('throttle', 0), steer=c.get('steer', 0),
            brake=c.get('brake', 0), hand_brake=c.get('hand_brake', False)))

    if 'velocity' in npc_def:
        v = npc_def['velocity']
        vehicle.set_target_velocity(carla.Vector3D(x=v.get('x',0), y=v.get('y',0), z=v.get('z',0)))

    return vehicle


def run_scenario(client, scenario, scenario_id, global_offset, max_secs=120):
    town = scenario['town']
    try:
        world = client.get_world()
        cur = world.get_map().name.split('/')[-1]
    except: cur = ''

    if cur == town:
        print(f"  🗺️  Already on {town}")
        try:
            s = world.get_settings(); s.synchronous_mode = False; s.fixed_delta_seconds = None
            world.apply_settings(s)
        except: pass
        time.sleep(1)
    else:
        print(f"  🗺️  Loading {town}...")
        try:
            world = client.load_world(town); time.sleep(3)
        except Exception as e:
            print(f"  ❌ Failed: {e}"); return [], 0

    settings = world.get_settings()
    settings.synchronous_mode = True; settings.fixed_delta_seconds = 1.0 / FPS
    world.apply_settings(settings)
    tm = client.get_trafficmanager(8000); tm.set_synchronous_mode(True)
    world.tick()

    spawn_points = world.get_map().get_spawn_points()

    ego_cfg = scenario.get('ego', {})
    if 'spawn_point' in ego_cfg:
        ego_sp = spawn_points[ego_cfg['spawn_point'] % len(spawn_points)]
    else:
        ego_sp = random.choice(spawn_points)
    bp = world.get_blueprint_library().find('vehicle.lincoln.mkz_2017')
    ego = world.try_spawn_actor(bp, ego_sp)
    if not ego:
        for sp in spawn_points[:10]:
            ego = world.try_spawn_actor(bp, sp)
            if ego: break
    if not ego:
        print("  ❌ Can't spawn ego"); return [], 0

    ego.set_autopilot(True, 8000)
    ego_ap = ego_cfg.get('autopilot', {})
    tm.auto_lane_change(ego, False)
    tm.distance_to_leading_vehicle(ego, 0.5)
    tm.ignore_vehicles_percentage(ego, ego_ap.get('ignore_vehicles', 60))
    tm.ignore_lights_percentage(ego, ego_ap.get('ignore_lights', 100))
    tm.ignore_signs_percentage(ego, ego_ap.get('ignore_signs', 100))
    if 'desired_speed' in ego_ap:
        tm.set_desired_speed(ego, ego_ap['desired_speed'])
    else:
        tm.vehicle_percentage_speed_difference(ego, -20)
    if 'path' in ego_ap:
        path_locs = [spawn_points[sp % len(spawn_points)].location for sp in ego_ap['path']]
        tm.set_path(ego, path_locs)
    world.tick()

    print(f"  🚗 Ego: {ego.type_id} at ({ego.get_location().x:.0f}, {ego.get_location().y:.0f})")

    bg_vehicles = []
    for _ in range(scenario.get('bg_vehicles', 40)):
        vbp = random.choice(world.get_blueprint_library().filter('vehicle.*'))
        if vbp.has_attribute('color'):
            vbp.set_attribute('color', random.choice(vbp.get_attribute('color').recommended_values))
        sp = random.choice(spawn_points)
        v = world.try_spawn_actor(vbp, sp)
        if v:
            v.set_autopilot(True, 8000)
            bg_vehicles.append(v)
    print(f"  🚦 Background: {len(bg_vehicles)} vehicles")

    npc_actors = []
    for key in ['npc1', 'npc2', 'npc3']:
        if key not in scenario: continue
        npc = spawn_npc(world, tm, scenario[key], spawn_points)
        if npc:
            npc_actors.append(npc)
            print(f"  🎯 {key}: {npc.type_id} at ({npc.get_location().x:.0f}, {npc.get_location().y:.0f})")
    print(f"  📡 Spawned {len(npc_actors)} crash NPCs")

    world.tick()

    rad = RadarRecorder(ego, world, MAX_RADAR_RANGE)
    rrad = RearRadarRecorder(ego, world, MAX_RADAR_RANGE)
    col = CollisionRecorder(ego, world)

    for _ in range(5): world.tick()

    data = []; frame = 0; prev_spd = 0.0; collision_frame = None
    print(f"  🏁 Recording (max {max_secs}s sim-time)...")

    while True:
        col.counter[0] = frame
        world.tick()

        sim_time = frame / FPS
        if sim_time > max_secs:
            print(f"  ⏱️  Timeout ({sim_time:.0f}s sim)"); break

        if collision_frame is not None and frame > collision_frame + FPS * 5:
            print(f"  ✅ Post-crash data collected"); break

        try:
            vel = ego.get_velocity()
            spd = math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)
            acc = (spd - prev_spd) * FPS if frame > 0 else 0.0
            prev_spd = spd; ctrl = ego.get_control()
        except: print("  ⚠️  Ego lost"); break

        rad.update_ego_speed(spd); near = rad.get()
        ttc = min(near['distance'] / near['relative_velocity'] if near['relative_velocity'] > 0.1 else 10.0, 10.0)

        rrad.update_ego_speed(spd); rear = rrad.get()
        rttc = min(rear['rear_distance'] / rear['rear_relative_velocity'] if rear['rear_relative_velocity'] > 0.1 else 10.0, 10.0)

        if collision_frame is None and scenario['type'] == 't_bone':
            ego_loc = ego.get_location()
            for npc in npc_actors:
                if not npc.is_alive: continue
                if getattr(npc, 'has_launched', False): continue
                dist = ego_loc.distance(npc.get_location())
                if 5.0 < dist < 15.0:
                    try:
                        npc.set_autopilot(False, 8000)
                        npc_loc = npc.get_location()
                        dx = ego_loc.x - npc_loc.x
                        dy = ego_loc.y - npc_loc.y
                        mag = math.sqrt(dx**2 + dy**2)
                        speed_m_s = 25.0
                        npc.enable_constant_velocity(carla.Vector3D((dx/mag) * speed_m_s, (dy/mag) * speed_m_s, 0))
                        npc.has_launched = True
                        print(f"  🚀 HEAT-SEEKING MISSILE FIRED! Target acquired at {dist:.1f}m")
                    except Exception:
                        pass

        data.append({
            'frame_id': global_offset + frame, 'scenario_id': scenario_id,
            'timestamp': round(frame / FPS, 3), 'scenario_type': scenario['type'], 'town': town,
            'ego_speed': round(spd, 3), 'ego_acceleration': round(acc, 3),
            'nearest_distance': round(near['distance'], 3), 'relative_velocity': round(near['relative_velocity'], 3),
            'ttc': round(ttc, 3), 'obstacle_speed': round(near['obstacle_speed'], 3),
            'obstacle_type': near['obstacle_type'], 'lateral_offset': round(near['lateral_offset'], 3),
            'ego_steering': round(ctrl.steer, 4),
            'rear_distance': round(rear['rear_distance'], 3), 'rear_relative_velocity': round(rear['rear_relative_velocity'], 3),
            'rear_ttc': round(rttc, 3), 'rear_obstacle_speed': round(rear['rear_obstacle_speed'], 3),
            'rear_obstacle_type': rear['rear_obstacle_type'], 'collision_within_2s': 0,
        })

        try:
            spec = world.get_spectator(); tf = ego.get_transform()
            spec.set_transform(carla.Transform(
                tf.location - tf.get_forward_vector() * 12 + carla.Location(z=6),
                carla.Rotation(pitch=-20, yaw=tf.rotation.yaw)))
        except: pass

        if col.frames and collision_frame is None:
            collision_frame = frame

        if frame % (FPS * 2) == 0 and frame > 0:
            obs = ['VEH','PED','---'][near['obstacle_type']]
            print(f"  [{frame/FPS:5.0f}s] SPD:{spd:5.1f}  DIST:{near['distance']:5.1f}  TTC:{ttc:5.1f}  {obs}  COL:{len(col.frames)}")

        frame += 1

    apply_labels(data, col.frames, LOOKAHEAD_FRAMES, offset=global_offset)

    for r in [rad, rrad, col]:
        try: r.cleanup()
        except: pass
    try: ego.set_autopilot(False, 8000)
    except: pass
    for a in npc_actors:
        try:
            if a.is_alive:
                try: a.set_autopilot(False, 8000)
                except: pass
        except: pass
    for _ in range(10):
        try: world.tick()
        except: break
    for a in npc_actors + bg_vehicles:
        try:
            if a.is_alive: a.destroy()
        except: pass
    for _ in range(5):
        try: world.tick()
        except: break
    try:
        if ego.is_alive: ego.destroy()
    except: pass
    for _ in range(5):
        try: world.tick()
        except: break
    try:
        s = world.get_settings(); s.synchronous_mode = False; s.fixed_delta_seconds = None
        world.apply_settings(s); tm.set_synchronous_mode(False)
    except: pass
    time.sleep(5)

    nc = len(col.frames); np_ = sum(1 for r in data if r['collision_within_2s'] == 1)
    print(f"\n  Result: {len(data)} frames, {nc} collisions, {np_} positive ({100*np_/max(1,len(data)):.1f}%)")
    for d in col.details:
        print(f"    → frame {d['frame']}: {d['actor']} ({d['impulse']:.0f}N·s)")
    return data, nc


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--host', default=CARLA_HOST)
    p.add_argument('--port', type=int, default=CARLA_PORT)
    p.add_argument('--repetitions', type=int, default=2)
    p.add_argument('--timeout', type=int, default=120)
    p.add_argument('--output', default=SAVE_DIR)
    p.add_argument('--town', default=None, help='Run only scenarios for this town (e.g. Town01). Avoids map changes that cause CARLA segfaults.')
    args = p.parse_args()

    active = [s for s in SCENARIOS if args.town is None or s['town'] == args.town]
    if not active:
        print(f"No scenarios for town '{args.town}'. Available: {sorted(set(s['town'] for s in SCENARIOS))}")
        return

    total = len(active) * args.repetitions
    print("=" * 70)
    print("ACCIDENT-STYLE CRASH DATA COLLECTOR")
    print("=" * 70)
    if args.town: print(f"  Town:      {args.town} only (no map changes)")
    print(f"  Scenarios: {len(active)} × {args.repetitions} reps = {total} runs")
    print(f"  Timeout:   {args.timeout}s per run")
    print(f"  Output:    {args.output}/data.csv")
    print("=" * 70)

    os.makedirs(args.output, exist_ok=True)
    csv_path = os.path.join(args.output, 'data.csv')
    client = carla.Client(args.host, args.port); client.set_timeout(30.0)

    all_data = []; g_frames = 0; g_cols = 0; g_pos = 0; rid = 0
    if os.path.exists(csv_path):
        existing = pd.read_csv(csv_path)
        all_data = existing.to_dict('records')
        g_frames = len(all_data)
        g_cols = existing['collision_within_2s'].sum() if 'collision_within_2s' in existing.columns else 0
        g_pos = g_cols
        print(f"  📂 Resuming from {g_frames} existing frames")

    try:
        for rep in range(args.repetitions):
            for sc in active:
                rid += 1
                print(f"\n{'='*70}")
                print(f"RUN {rid}/{total} — {sc['name']} ({sc['type']}) [rep {rep+1}]")
                print(f"{'='*70}")
                try:
                    data, nc = run_scenario(client, sc, rid, g_frames, args.timeout)
                except Exception as e:
                    print(f"  ❌ FAILED: {e}"); traceback.print_exc(); data, nc = [], 0
                np_ = sum(1 for r in data if r.get('collision_within_2s') == 1)
                g_frames += len(data); g_cols += nc; g_pos += np_
                all_data.extend(data)
                if all_data:
                    pd.DataFrame(all_data).to_csv(csv_path, index=False)
                print(f"  💾 Total: {g_frames:,} frames, {g_cols} collisions, {g_pos} positive ({100*g_pos/max(1,g_frames):.1f}%)")
    except KeyboardInterrupt:
        print(f"\n  ⚠️  Interrupted after {rid} runs")

    print(f"\n{'='*70}\nDONE — {rid} runs, {g_frames:,} frames, {g_cols} collisions, {g_pos} positive\n{'='*70}")


if __name__ == '__main__':
    main()
