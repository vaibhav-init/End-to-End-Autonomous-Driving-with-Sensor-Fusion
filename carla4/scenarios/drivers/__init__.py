#!/usr/bin/env python3
"""
Pluggable scenario drivers.

`make_driver` imports the concrete driver lazily so each conda env only loads
what it can: the `pcla` driver pulls in the PCLA framework, the `mlp` and
`transformer` drivers pull in torch (and sklearn for the MLP's scaler).
Importing this package itself is cheap and env-agnostic.

Radar options arrive as the keyword set produced by
`radar.radar_kwargs_from_args`, so a scenario script forwards one dict rather
than threading nine arguments through every call.
"""

from .base import Driver

DRIVER_NAMES = ("pcla", "mlp", "transformer")


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
    radar_ghost_oracle=False,
    radar_overrides=None,
    debug_every=20,
    safety_rules=False,
    cruise_floor=True,
    **extra,
):
    """Build a Driver by name. Heavy deps are imported only for the chosen one."""
    if name == "pcla":
        from .pcla_driver import PCLADriver
        return PCLADriver(pcla_agent=pcla_agent, debug_every=debug_every)
    radar_kwargs = dict(
        radar_backend=radar_backend,
        radar_profile=radar_profile,
        radar_config_path=radar_config_path,
        radar_seed=radar_seed,
        radar_ghost_detector=radar_ghost_detector,
        radar_ghost_threshold=radar_ghost_threshold,
        radar_ghost_device=radar_ghost_device,
        radar_ghost_oracle=radar_ghost_oracle,
        radar_overrides=radar_overrides,
    )
    if name == "mlp":
        from .mlp_driver import MLPDriver
        if not model_dir:
            raise ValueError("mlp driver requires --model-dir")
        return MLPDriver(
            model_dir=model_dir,
            debug_every=debug_every,
            safety_rules=safety_rules,
            cruise_floor=cruise_floor,
            **radar_kwargs,
        )
    if name == "transformer":
        from .transformer_driver import TransformerDriver
        if not model_dir:
            raise ValueError("transformer driver requires --model-dir")
        return TransformerDriver(
            model_dir=model_dir,
            debug_every=debug_every,
            cruise_floor=cruise_floor,
            **radar_kwargs,
        )
    raise ValueError(f"Unknown driver '{name}'. Choose from {DRIVER_NAMES}")


__all__ = ["Driver", "make_driver", "DRIVER_NAMES"]
