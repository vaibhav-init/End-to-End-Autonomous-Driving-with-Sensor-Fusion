"""
Generate a closed-loop route covering Town01 for PCLA testing.

Run this script on the machine where CARLA is running:
    python generate_town01_loop.py

It connects to CARLA, uses the GlobalRoutePlanner to trace roads between
a set of hand-picked waypoints spread across Town01, and writes the result
to  town01_loop_route.xml  in the same directory.

The route forms a CLOSED LOOP — the last waypoint connects back to the first —
so the ego vehicle can drive the entire map and return to its starting position.
"""

import carla
import sys
import os
import time

# ── Ensure PCLA imports work ──────────────────────────────────────────────────
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from leaderboard_codes.global_route_planner import GlobalRoutePlanner
from leaderboard_codes.global_route_planner_dao import GlobalRoutePlannerDAO

# ── Config ────────────────────────────────────────────────────────────────────
HOST        = "localhost"
PORT        = 2000
MAP         = "Town01"
OUTPUT_FILE = os.path.join(script_dir, "town01_loop_route.xml")
HOP_RES     = 2.0        # metres between interpolated waypoints


# ── Town01 key locations ──────────────────────────────────────────────────────
# These are real road coordinates in Town01, chosen to cover all four quadrants
# and most major roads.  The list forms a loop (last → first is also routed).
#
# Town01 rough layout (CARLA 0.9.x):
#   X range ≈ [-5 .. 400]    Y range ≈ [0 .. 340]
#   Grid-like streets, mostly right-angle intersections.
#
LOOP_WAYPOINTS = [
    # ── Start: centre-east, heading north ──
    carla.Location(x=334.7, y=273.0, z=0.0),   # 0  east side, mid-Y
    # ── North-east corner ──
    carla.Location(x=392.0, y=326.0, z=0.0),   # 1  top-right area
    # ── North-west corner ──
    carla.Location(x=88.0,  y=326.0, z=0.0),   # 2  top-left area
    # ── West side, heading south ──
    carla.Location(x=2.0,   y=192.0, z=0.0),   # 3  far left, mid-Y
    # ── South-west corner ──
    carla.Location(x=88.0,  y=2.5,   z=0.0),   # 4  bottom-left area
    # ── South, centre ──
    carla.Location(x=200.0, y=2.5,   z=0.0),   # 5  bottom-centre
    # ── South-east corner ──
    carla.Location(x=392.0, y=55.0,  z=0.0),   # 6  bottom-right area
    # ── East side mid, heading north back to start ──
    carla.Location(x=392.0, y=192.0, z=0.0),   # 7  right side, mid-Y
    # Loop closure: route planner will connect #7 → #0 automatically
]


def build_route_xml(waypoints, save_path):
    """Write a list of CARLA waypoints (wp objects from trace_route) to XML."""
    from xml.dom import minidom

    root = minidom.Document()
    xml = root.createElement("route")
    xml.setAttribute("id", "_")
    xml.setAttribute("town", MAP)
    root.appendChild(xml)

    for wp in waypoints:
        tf = wp.transform
        el = root.createElement("waypoint")
        el.setAttribute("pitch", str(tf.rotation.pitch))
        el.setAttribute("roll",  str(tf.rotation.roll))
        el.setAttribute("x",     str(tf.location.x))
        el.setAttribute("y",     str(tf.location.y))
        el.setAttribute("yaw",   str(tf.rotation.yaw))
        el.setAttribute("z",     str(tf.location.z))
        xml.appendChild(el)

    with open(save_path, "w") as f:
        f.write(root.toprettyxml(indent="\t"))

    print(f"[Done] Wrote {len(waypoints)} waypoints → {save_path}")


def main():
    # ── Connect ───────────────────────────────────────────────────────────────
    print(f"[Init] Connecting to CARLA at {HOST}:{PORT} …")
    client = carla.Client(HOST, PORT)
    client.set_timeout(60.0)

    print(f"[Init] Loading {MAP} …")
    client.load_world(MAP)
    time.sleep(5)

    world = client.get_world()
    amap  = world.get_map()

    # ── Build the route planner ───────────────────────────────────────────────
    dao = GlobalRoutePlannerDAO(amap, HOP_RES)
    grp = GlobalRoutePlanner(dao)
    grp.setup()

    # ── Trace the full loop ───────────────────────────────────────────────────
    # We connect each pair of consecutive key-points AND close the loop.
    all_waypoints = []
    n = len(LOOP_WAYPOINTS)

    for i in range(n):
        start = LOOP_WAYPOINTS[i]
        end   = LOOP_WAYPOINTS[(i + 1) % n]       # wraps around to 0
        print(f"  Segment {i}→{(i+1)%n}:  "
              f"({start.x:.0f},{start.y:.0f}) → ({end.x:.0f},{end.y:.0f}) … ", end="")

        trace = grp.trace_route(start, end)
        segment_wps = [wp for wp, _road_option in trace]

        # Avoid duplicating the junction waypoint between segments
        if all_waypoints and segment_wps:
            last_loc = all_waypoints[-1].transform.location
            first_loc = segment_wps[0].transform.location
            if last_loc.distance(first_loc) < 1.0:
                segment_wps = segment_wps[1:]

        all_waypoints.extend(segment_wps)
        print(f"{len(segment_wps)} wps")

    print(f"\n[Route] Total waypoints: {len(all_waypoints)}")

    # ── Write XML ─────────────────────────────────────────────────────────────
    build_route_xml(all_waypoints, OUTPUT_FILE)


if __name__ == "__main__":
    main()
