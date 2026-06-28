#!/usr/bin/env python3
"""
Pluggable scenario drivers.

`make_driver` imports the concrete driver lazily so each conda env only loads
what it can: the `pcla` driver pulls in the PCLA framework, the `mlp` driver
pulls in ultralytics + scikit-learn. Importing this package itself is cheap and
env-agnostic.
"""

from .base import Driver

DRIVER_NAMES = ("pcla", "mlp")


def make_driver(name, model_dir=None, pcla_agent="tfv6_visiononly",
                debug_every=20):
    """Build a Driver by name. Heavy deps are imported only for the chosen one."""
    if name == "pcla":
        from .pcla_driver import PCLADriver
        return PCLADriver(pcla_agent=pcla_agent, debug_every=debug_every)
    if name == "mlp":
        from .mlp_driver import MLPDriver
        if not model_dir:
            raise ValueError("mlp driver requires --model-dir")
        return MLPDriver(model_dir=model_dir, debug_every=debug_every)
    raise ValueError(f"Unknown driver '{name}'. Choose from {DRIVER_NAMES}")


__all__ = ["Driver", "make_driver", "DRIVER_NAMES"]
