import carla
import math
import random
import time
import traceback
import xml.etree.ElementTree as ET
from PCLA import PCLA, location_to_waypoint, route_maker


# \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
# CONFIG \u2014 change these as needed
# \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
HOST_IP         = "localhost"
PORT            = 2000
AGENT           = "tfv6_visiononly"    # true camera-only (no LiDAR, no radar)
MAP             = "Town01"
ROUTE_FILE      = "./town01_loop_route.xml"
SPAWN_START_IDX = 0
SPAWN_END_IDX   = 100
FIXED_DELTA     = 0.05                 # 20 FPS simulation
LOG_EVERY_N     = 100                  # print stats every N steps

# ── Route-completion detection ──
MIN_STEPS_BEFORE_COMPLETE = 200        # ignore proximity until car has driven a bit
COMPLETION_PROXIMITY_M    = 15.0       # metres from start to count as "back"

# ── Dynamic obstacle spawn ──
OBSTACLE_SPEED_THRESHOLD  = 40.0       # km/h — spawn obstacle when ego exceeds this
OBSTACLE_SPAWN_DIST       = 40.0       # metres ahead of ego to place the obstacle


# \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
# SPECTATOR \u2014 follows ego vehicle from behind
# \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
def update_spectator(spectator, vehicle):
    transform = vehicle.get_transform()
    forward = transform.get_forward_vector()
    camera_location = (
        transform.location
        - forward * 10.0
        + carla.Location(z=5.0)
    )
    camera_rotation = carla.Rotation(
        pitch=-15.0,
        yaw=transform.rotation.yaw,
        roll=0.0
    )
    spectator.set_transform(
        carla.Transform(camera_location, camera_rotation)
    )


# ──────────────────────────────────────────────────────────────────────────────
# ROUTE — generate if not exists
# ──────────────────────────────────────────────────────────────────────────────
def ensure_route(client, world, route_file, start_idx, end_idx):
    import os
    if os.path.exists(route_file):
        waypoint_count = sum(1 for line in open(route_file) if "waypoint" in line)
        print(f"[Route] Using existing route: {route_file} ({waypoint_count} waypoints)")
        return

    print(f"[Route] Generating new route from spawn {start_idx} to {end_idx}...")
    spawn_points = world.get_map().get_spawn_points()
    start = spawn_points[start_idx].location
    end   = spawn_points[end_idx].location
    waypoints = location_to_waypoint(client, start, end)
    route_maker(waypoints, route_file)
    print(f"[Route] Saved {len(waypoints)} waypoints to {route_file}")


def get_spawn_from_route(route_file):
    """Parse the first waypoint from the route XML and return a CARLA Transform
    so the ego vehicle spawns exactly where the route starts."""
    tree = ET.parse(route_file)
    first_wp = tree.getroot().find("waypoint")
    x     = float(first_wp.get("x"))
    y     = float(first_wp.get("y"))
    z     = float(first_wp.get("z", "0.3"))  # slightly above ground
    yaw   = float(first_wp.get("yaw", "0.0"))
    pitch = float(first_wp.get("pitch", "0.0"))
    # Clamp pitch to sane range (route files sometimes have 360.0)
    if pitch > 180.0:
        pitch -= 360.0
    spawn_transform = carla.Transform(
        carla.Location(x=x, y=y, z=max(z, 0.3)),
        carla.Rotation(pitch=pitch, yaw=yaw, roll=0.0)
    )
    print(f"[Spawn] Route start: x={x:.2f}, y={y:.2f}, yaw={yaw:.1f}°")
    return spawn_transform


# ──────────────────────────────────────────────────────────────────────────────
# LOGGING
# ──────────────────────────────────────────────────────────────────────────────
def log_step(step, vehicle, ego_action, total_dist=0.0):
    v = vehicle.get_velocity()
    speed_kmh = (v.x**2 + v.y**2 + v.z**2) ** 0.5 * 3.6
    loc = vehicle.get_location()
    print(
        f"Step {step:05d} | "
        f"Speed: {speed_kmh:5.1f} km/h | "
        f"Throttle: {ego_action.throttle:.2f} | "
        f"Brake: {ego_action.brake:.2f} | "
        f"Steer: {ego_action.steer:+.3f} | "
        f"Pos: ({loc.x:.1f}, {loc.y:.1f}) | "
        f"Dist: {total_dist:.0f}m"
    )


def check_route_complete(pcla_instance, vehicle_loc, start_loc, step):
    """Return True when the vehicle has completed the full loop."""
    if step < MIN_STEPS_BEFORE_COMPLETE:
        return False

    # 1. Check if any planner flagged route end
    route_ended = False
    try:
        for planner in pcla_instance.agent_instance.gps_waypoint_planners_dict.values():
            if planner.is_last:
                route_ended = True
                break
    except (AttributeError, TypeError):
        pass

    # 2. Check proximity to the starting point
    dx = vehicle_loc.x - start_loc.x
    dy = vehicle_loc.y - start_loc.y
    dist_to_start = math.sqrt(dx * dx + dy * dy)
    near_start = dist_to_start < COMPLETION_PROXIMITY_M

    if route_ended:
        print(f"\n[Route] Route planner reports route end at step {step}")
        print(f"[Route] Distance to start: {dist_to_start:.1f}m")
        return True

    if near_start:
        print(f"\n[Route] Vehicle returned near start at step {step} "
              f"(dist={dist_to_start:.1f}m)")
        return True

    return False


# ──────────────────────────────────────────────────────────────────────────────
# WEATHER — dense fog
# ──────────────────────────────────────────────────────────────────────────────
def set_dense_fog(world):
    """Set weather to dense fog for visibility testing."""
    fog_weather = carla.WeatherParameters(
        cloudiness=100.0,
        precipitation=0.0,
        precipitation_deposits=0.0,
        wind_intensity=10.0,
        sun_azimuth_angle=0.0,
        sun_altitude_angle=45.0,
        fog_density=100.0,
        fog_distance=0.0,
        fog_falloff=0.0,
        wetness=50.0,
    )
    world.set_weather(fog_weather)
    print("[Weather] Dense fog enabled")


# ──────────────────────────────────────────────────────────────────────────────
# OBSTACLE — dynamic spawn ahead of ego when fast
# ──────────────────────────────────────────────────────────────────────────────
def spawn_obstacle_ahead(world, ego_vehicle, dist_ahead=OBSTACLE_SPAWN_DIST):
    """Spawn a stationary vehicle directly ahead of the ego vehicle.

    Uses the ego's current transform + forward vector to place the obstacle
    on the same lane, `dist_ahead` metres in front.

    Returns the spawned obstacle actor (or None on failure).
    """
    ego_tf  = ego_vehicle.get_transform()
    fwd     = ego_tf.get_forward_vector()
    ego_loc = ego_tf.location

    obstacle_location = carla.Location(
        x=ego_loc.x + fwd.x * dist_ahead,
        y=ego_loc.y + fwd.y * dist_ahead,
        z=ego_loc.z + 0.3,              # slightly above ground to avoid clipping
    )
    obstacle_transform = carla.Transform(
        obstacle_location,
        carla.Rotation(pitch=0.0, yaw=ego_tf.rotation.yaw, roll=0.0),
    )

    bp_library  = world.get_blueprint_library()
    vehicle_bps = bp_library.filter("vehicle.*")
    obstacle_bp = random.choice(vehicle_bps)
    if obstacle_bp.has_attribute("color"):
        obstacle_bp.set_attribute("color", "255,0,0")   # red for visibility

    try:
        obstacle = world.spawn_actor(obstacle_bp, obstacle_transform)
        obstacle.apply_control(carla.VehicleControl(hand_brake=True))
        print(f"\n[Obstacle] ⚠️  Spawned {obstacle_bp.id} ~{dist_ahead:.0f}m ahead "
              f"at ({obstacle_location.x:.1f}, {obstacle_location.y:.1f})")
        return obstacle
    except Exception as e:
        print(f"[Obstacle] Spawn failed: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    # Connect to CARLA
    client = carla.Client(HOST_IP, PORT)
    client.set_timeout(60.0)

    print(f"[Init] Loading {MAP}...")
    client.load_world(MAP)
    time.sleep(5)  # wait for map to fully load

    pcla    = None
    vehicle = None
    obstacle_vehicle = None
    settings = None

    try:
        world           = client.get_world()
        spectator       = world.get_spectator()
        traffic_manager = client.get_trafficmanager(8000)

        # Synchronous mode
        settings = world.get_settings()
        traffic_manager.set_synchronous_mode(True)
        settings.synchronous_mode  = True
        settings.fixed_delta_seconds = FIXED_DELTA
        world.apply_settings(settings)
        print("[Init] Synchronous mode enabled")

        # Dense fog weather
        set_dense_fog(world)

        # Generate route if needed
        ensure_route(client, world, ROUTE_FILE, SPAWN_START_IDX, SPAWN_END_IDX)

        # Spawn ego vehicle at exact route start location
        bp_library = world.get_blueprint_library()
        vehicle_bp = bp_library.filter("vehicle.tesla.model3")[0]
        spawn_transform = get_spawn_from_route(ROUTE_FILE)
        vehicle = world.spawn_actor(vehicle_bp, spawn_transform)
        world.tick()
        loc = spawn_transform.location
        print(f"[Init] Ego vehicle spawned at route start ({loc.x:.1f}, {loc.y:.1f})")

        # Init PCLA agent
        pcla = PCLA(AGENT, vehicle, ROUTE_FILE, client)
        print(f"\n[Agent] Running: {AGENT} | Dense fog | obstacle at >{OBSTACLE_SPEED_THRESHOLD:.0f} km/h")
        print(f"        Press Ctrl+C to stop\n")

        step = 0
        total_dist = 0.0
        prev_loc = spawn_transform.location
        start_loc = spawn_transform.location      # remember where we started
        obstacle_spawned = False                   # spawn only once

        while True:
            try:
                # Get action from agent
                ego_action = pcla.get_action()

                # Apply control
                vehicle.apply_control(ego_action)

                # Tick world
                world.tick()

                # Update spectator camera
                update_spectator(spectator, vehicle)

                # Track distance driven
                cur_loc = vehicle.get_location()
                dx = cur_loc.x - prev_loc.x
                dy = cur_loc.y - prev_loc.y
                total_dist += math.sqrt(dx * dx + dy * dy)
                prev_loc = cur_loc

                # Log every N steps
                if step % LOG_EVERY_N == 0:
                    log_step(step, vehicle, ego_action, total_dist)

                # Dynamic obstacle: spawn when ego exceeds speed threshold
                if not obstacle_spawned:
                    v = vehicle.get_velocity()
                    speed_kmh = (v.x**2 + v.y**2 + v.z**2) ** 0.5 * 3.6
                    if speed_kmh > OBSTACLE_SPEED_THRESHOLD:
                        print(f"[Obstacle] Ego speed {speed_kmh:.1f} km/h > "
                              f"{OBSTACLE_SPEED_THRESHOLD:.0f} km/h threshold — spawning!")
                        obstacle_vehicle = spawn_obstacle_ahead(world, vehicle)
                        obstacle_spawned = True

                # Check for route completion
                if check_route_complete(pcla, cur_loc, start_loc, step):
                    log_step(step, vehicle, ego_action, total_dist)
                    print(f"\n{'='*60}")
                    print(f"  ROUTE COMPLETED!")
                    print(f"  Total steps : {step}")
                    print(f"  Total dist  : {total_dist:.0f} m")
                    print(f"{'='*60}\n")
                    break

                step += 1

            except KeyboardInterrupt:
                print("\n[Stop] Ctrl+C received")
                break
            except Exception as e:
                print(f"\n[Error] at step {step}: {type(e).__name__}: {e}")
                traceback.print_exc()
                break

    finally:
        print("\n[Cleanup] Restoring world settings...")
        if settings is not None:
            settings.synchronous_mode    = False
            settings.fixed_delta_seconds = None
            world.apply_settings(settings)

        # Destroy obstacle vehicle
        try:
            if obstacle_vehicle is not None and obstacle_vehicle.is_alive:
                obstacle_vehicle.destroy()
                print("[Cleanup] Obstacle vehicle destroyed")
        except Exception:
            pass

        if pcla is not None:
            pcla.cleanup()
        elif vehicle is not None:
            vehicle.destroy()

        time.sleep(0.5)
        print("[Cleanup] Done.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
    finally:
        print("Done.")