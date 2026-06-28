#!/usr/bin/env python3
"""
PCLA driver — pretrained CARLA Leaderboard agent for longitudinal control.

Wraps the PCLA framework (default agent: tfv6_visiononly, a vision-only
Transfuser-V6 model). The agent is end-to-end, but for this study we keep only
its LONGITUDINAL output (throttle/brake) and delegate steering to BasicAgent, so
lateral handling is identical to the MLP driver.

Runs in the `PCLA` conda env. Lazy-imported by drivers/__init__.py so the carla4
env never has to import PCLA's heavy dependencies.
"""

import os
import sys
import tempfile

import carla

from .base import Driver
from .steering import BasicAgentSteering

# Repo root (three levels up) holds the vendored PCLA framework
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_PCLA_DIR = os.path.join(_REPO_ROOT, "PCLA")
if _PCLA_DIR not in sys.path:
    sys.path.insert(0, _PCLA_DIR)

try:
    from PCLA import PCLA, location_to_waypoint, route_maker
except ImportError as exc:  # pragma: no cover - depends on remote env
    raise RuntimeError(
        f"Could not import PCLA from {_PCLA_DIR}. Run this driver in the PCLA "
        f"conda env (see PCLA/environment.yml). Original error: {exc}"
    )

# Distance ahead of the ego used to build the leaderboard route the agent plans against
ROUTE_AHEAD_M = 200.0

_route_counter = [0]


class PCLADriver(Driver):
    name = "pcla"

    def __init__(self, pcla_agent="tfv6_visiononly", debug_every=20, **_ignored):
        self.pcla_agent = pcla_agent
        self.debug_every = debug_every
        self.pcla = None
        self.steering = None
        self.route_path = None
        self._frame = 0

    def _build_route(self, ego, carla_map, client):
        """Write a leaderboard route XML from the ego spawn forward (~200m)."""
        ego_loc = ego.get_location()
        wp = carla_map.get_waypoint(ego_loc, project_to_road=True,
                                    lane_type=carla.LaneType.Driving)
        end_loc = ego_loc
        if wp is not None:
            for dist in (ROUTE_AHEAD_M, ROUTE_AHEAD_M * 0.5, 50.0):
                ahead = wp.next(dist)
                if ahead:
                    end_loc = ahead[-1].transform.location
                    break

        waypoints = location_to_waypoint(client, ego_loc, end_loc)
        _route_counter[0] += 1
        self.route_path = os.path.join(
            tempfile.gettempdir(),
            f"pcla_route_{os.getpid()}_{_route_counter[0]}.xml",
        )
        route_maker(waypoints, self.route_path)
        return len(waypoints)

    def setup(self, world, ego, carla_map, client):
        n_wp = self._build_route(ego, carla_map, client)
        print(f"  [pcla] route written: {self.route_path} ({n_wp} waypoints)")
        # PCLA attaches the agent's own (vision) sensors and ticks the world once.
        self.pcla = PCLA(self.pcla_agent, ego, self.route_path, client)
        self.steering = BasicAgentSteering(ego, carla_map)
        self._frame = 0
        print(f"  [pcla] agent ready: {self.pcla_agent} "
              f"(longitudinal from agent, steer from BasicAgent)")

    def get_control(self, ego, world):
        action = self.pcla.get_action()
        throttle = float(action.throttle)
        brake = float(action.brake)
        steer = self.steering.get_steer()

        if self.debug_every and self._frame % self.debug_every == 0:
            velocity = ego.get_velocity()
            speed = (velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2) ** 0.5
            print(f"  [pcla] f={self._frame:4d} spd={speed * 3.6:5.1f}km/h "
                  f"thr/brk={throttle:.2f}/{brake:.2f} steer={steer:+.2f} "
                  f"(agent_steer={action.steer:+.2f} discarded)")

        self._frame += 1
        return carla.VehicleControl(throttle=throttle, steer=steer, brake=brake)

    def cleanup(self):
        # NOTE: PCLA.cleanup() destroys the ego vehicle AND every sensor in the
        # world (including the scenario's collision sensor). The scenario's own
        # teardown is guarded against already-dead actors, so this is safe.
        if self.pcla is not None:
            try:
                self.pcla.cleanup()
            finally:
                self.pcla = None
        if self.route_path and os.path.exists(self.route_path):
            try:
                os.remove(self.route_path)
            except OSError:
                pass
            self.route_path = None
