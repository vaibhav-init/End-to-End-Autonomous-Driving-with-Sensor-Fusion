#!/usr/bin/env python3
"""
Shared lateral controller: BasicAgent steering.

Both drivers delegate steering/handling to BasicAgent so lateral behavior is a
controlled constant across the comparison. Only the steer output is used; the
agent's throttle/brake are discarded (the model owns longitudinal control).

This mirrors the BasicAgent setup already used in s1..s4 (ignore lights/stops,
drive a long straight destination) and refreshes the destination when reached so
steering keeps working for the whole scenario.
"""

import os
import sys

import carla

CARLA_ROOT = os.environ.get("CARLA_ROOT", "/opt/carla-simulator")
_agents_path = os.path.join(CARLA_ROOT, "PythonAPI", "carla")
if _agents_path not in sys.path:
    sys.path.insert(0, _agents_path)

try:
    from agents.navigation.basic_agent import BasicAgent
except ImportError:
    BasicAgent = None


class BasicAgentSteering:
    """Lateral-only wrapper around BasicAgent."""

    def __init__(self, ego, carla_map, target_speed=60, lookahead_m=500.0,
                 steer_clamp=0.7):
        if BasicAgent is None:
            raise RuntimeError(
                "BasicAgent not available; check CARLA_ROOT and PythonAPI path"
            )
        self.ego = ego
        self.carla_map = carla_map
        self.lookahead_m = lookahead_m
        self.steer_clamp = steer_clamp
        self.agent = BasicAgent(ego, target_speed=target_speed)
        self.agent.ignore_traffic_lights(True)
        self.agent.ignore_stop_signs(True)
        self._destinations_set = 0
        self._set_forward_destination()
        print(f"  [steering] BasicAgent ready (target_speed={target_speed}, "
              f"lookahead={lookahead_m:.0f}m, destination #{self._destinations_set})")

    def _set_forward_destination(self):
        """Aim at a waypoint far ahead on the current lane."""
        wp = self.carla_map.get_waypoint(
            self.ego.get_location(), project_to_road=True,
            lane_type=carla.LaneType.Driving)
        if wp is None:
            return
        for dist in (self.lookahead_m, self.lookahead_m * 0.4, 50.0):
            nxt = wp.next(dist)
            if nxt:
                self.agent.set_destination(nxt[0].transform.location)
                self._destinations_set += 1
                return

    def get_steer(self):
        """Return clamped steer for this tick; refresh destination if reached."""
        if self.agent.done():
            self._set_forward_destination()
            print(f"  [steering] destination reached, set new "
                  f"(#{self._destinations_set})")
        control = self.agent.run_step()
        return max(-self.steer_clamp, min(self.steer_clamp, control.steer))
