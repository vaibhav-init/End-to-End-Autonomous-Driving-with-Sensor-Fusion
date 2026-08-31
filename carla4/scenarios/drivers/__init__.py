#!/usr/bin/env python3
"""
Pluggable scenario drivers.

`make_driver` imports the concrete driver lazily so each conda env only loads
what it can: the `pcla` driver pulls in the PCLA framework, the `mlp` driver
pulls in ultralytics + scikit-learn, and the `idm` driver needs neither.
Importing this package itself is cheap and env-agnostic.
"""

from .base import Driver

DRIVER_NAMES = ("pcla", "mlp", "idm")


def make_driver(
    name,
    model_dir=None,
    pcla_agent="tfv6_visiononly",
    radar_backend=None,
    radar_profile=None,
    radar_config_path=None,
    radar_seed=42,
    radar_ghost_detector=None,
    radar_ghost_threshold=None,
    radar_ghost_device="cpu",
    debug_every=20,
    safety_rules=False,
    cruise_floor=True,
    desired_speed_kmh=None,
    **extra,
):
    """Build a Driver by name. Heavy deps are imported only for the chosen one."""
    if name == "pcla":
        from .pcla_driver import PCLADriver
        return PCLADriver(pcla_agent=pcla_agent, debug_every=debug_every)
    if name == "idm":
        # Reference longitudinal policy: radar -> IDM -> target speed -> the
        # same PID tail the learned driver uses. No camera, no weights.
        from .idm_driver import IDMDriver
        kwargs = dict(
            radar_backend=radar_backend,
            radar_profile=radar_profile,
            radar_config_path=radar_config_path,
            radar_seed=radar_seed,
            radar_ghost_detector=radar_ghost_detector,
            radar_ghost_threshold=radar_ghost_threshold,
            radar_ghost_device=radar_ghost_device,
            debug_every=debug_every,
        )
        if desired_speed_kmh is not None:
            kwargs["desired_speed_kmh"] = desired_speed_kmh
        kwargs.update(extra)
        return IDMDriver(**kwargs)
    if name == "mlp":
        from .mlp_driver import MLPDriver
        if not model_dir:
            raise ValueError("mlp driver requires --model-dir")
        return MLPDriver(
            model_dir=model_dir,
            radar_backend=radar_backend,
            radar_profile=radar_profile,
            radar_config_path=radar_config_path,
            radar_seed=radar_seed,
            radar_ghost_detector=radar_ghost_detector,
            radar_ghost_threshold=radar_ghost_threshold,
            radar_ghost_device=radar_ghost_device,
            debug_every=debug_every,
            safety_rules=safety_rules,
            cruise_floor=cruise_floor,
        )
    raise ValueError(f"Unknown driver '{name}'. Choose from {DRIVER_NAMES}")


__all__ = ["Driver", "make_driver", "DRIVER_NAMES"]
