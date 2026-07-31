#!/usr/bin/env python3
"""
Driver interface for scenario evaluation.

A Driver is the *control source* for the ego vehicle inside a scenario. The
scenario owns everything else (spawning, fog, obstacle, ground-truth logging,
termination); the driver only decides throttle/brake/steer each tick.

For this study every driver produces LONGITUDINAL control (throttle/brake) from
its model and delegates LATERAL control (steer) to BasicAgent, so the only thing
that differs between drivers is longitudinal behavior.
"""

from abc import ABC, abstractmethod


class Driver(ABC):
    """Pluggable control source for a scenario ego vehicle."""

    name = "base"

    @abstractmethod
    def setup(self, world, ego, carla_map, client):
        """Attach sensors / load models. Called once after the ego is settled."""

    @abstractmethod
    def get_control(self, ego, world):
        """Return a carla.VehicleControl for this tick."""

    def cleanup(self):
        """Destroy any actors/sensors the driver owns. Safe to call twice."""

    def diagnostics(self):
        """Return optional per-frame driver/sensor diagnostics."""

        return {}
