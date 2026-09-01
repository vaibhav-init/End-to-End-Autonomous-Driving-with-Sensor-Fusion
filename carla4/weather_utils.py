#!/usr/bin/env python3
"""
Randomized CARLA weather helpers.
"""

import random

import carla


FOG_WEATHER_PRESETS = [
    {
        "name": "mist_light",
        "cloudiness": 25.0,
        "precipitation": 0.0,
        "precipitation_deposits": 0.0,
        "wind_intensity": 10.0,
        "sun_azimuth_angle": 20.0,
        "sun_altitude_angle": 35.0,
        "fog_density": 18.0,
        "fog_distance": 60.0,
        "fog_falloff": 0.35,
        "wetness": 5.0,
    },
    {
        "name": "mist_medium",
        "cloudiness": 45.0,
        "precipitation": 0.0,
        "precipitation_deposits": 0.0,
        "wind_intensity": 15.0,
        "sun_azimuth_angle": 55.0,
        "sun_altitude_angle": 25.0,
        "fog_density": 32.0,
        "fog_distance": 42.0,
        "fog_falloff": 0.55,
        "wetness": 10.0,
    },
    {
        "name": "mist_heavy",
        "cloudiness": 65.0,
        "precipitation": 0.0,
        "precipitation_deposits": 0.0,
        "wind_intensity": 20.0,
        "sun_azimuth_angle": 95.0,
        "sun_altitude_angle": 18.0,
        "fog_density": 48.0,
        "fog_distance": 28.0,
        "fog_falloff": 0.8,
        "wetness": 18.0,
    },
    {
        "name": "fog_dawn",
        "cloudiness": 55.0,
        "precipitation": 0.0,
        "precipitation_deposits": 0.0,
        "wind_intensity": 8.0,
        "sun_azimuth_angle": 130.0,
        "sun_altitude_angle": 8.0,
        "fog_density": 42.0,
        "fog_distance": 24.0,
        "fog_falloff": 1.0,
        "wetness": 22.0,
    },
    {
        "name": "fog_dense",
        "cloudiness": 80.0,
        "precipitation": 0.0,
        "precipitation_deposits": 0.0,
        "wind_intensity": 12.0,
        "sun_azimuth_angle": 180.0,
        "sun_altitude_angle": 12.0,
        "fog_density": 62.0,
        "fog_distance": 18.0,
        "fog_falloff": 1.4,
        "wetness": 28.0,
    },
]


# Radar barely notices fog: at 77 GHz the model charges 0.05 dB per 100 m of
# fog, so the densest preset above costs about 0.045 dB at full range. The fog
# ladder exists for the camera, which it blinds. A radar-only collection pays
# the render cost for no measurable sensor effect, so it should usually run
# clear. Kept alongside the ladder rather than replacing it, because vision is
# reinstatable (see vision.md) and the scenario harness still sweeps fog.
CLEAR_WEATHER_PRESET = {
    "name": "clear",
    "cloudiness": 5.0,
    "precipitation": 0.0,
    "precipitation_deposits": 0.0,
    "wind_intensity": 5.0,
    "sun_azimuth_angle": 45.0,
    "sun_altitude_angle": 60.0,
    "fog_density": 0.0,
    "fog_distance": 0.0,
    "fog_falloff": 0.0,
    "wetness": 0.0,
}

WEATHER_MODES = ("clear", "fog_ladder")


def _to_weather(preset):
    return carla.WeatherParameters(
        **{key: value for key, value in preset.items() if key != "name"}
    )


def apply_weather(world, mode="clear", rng=None):
    """Apply a weather mode by name; returns the preset name applied."""

    if mode not in WEATHER_MODES:
        raise ValueError(
            f"Unknown weather mode '{mode}'. Choose from {WEATHER_MODES}."
        )
    if mode == "clear":
        world.set_weather(_to_weather(CLEAR_WEATHER_PRESET))
        return CLEAR_WEATHER_PRESET["name"]
    return apply_random_fog(world, rng=rng)


def choose_random_fog_weather(rng=None):
    rng = rng or random
    preset = dict(rng.choice(FOG_WEATHER_PRESETS))
    weather = carla.WeatherParameters(**{key: value for key, value in preset.items() if key != "name"})
    return preset["name"], weather


def apply_random_fog(world, rng=None):
    name, weather = choose_random_fog_weather(rng=rng)
    world.set_weather(weather)
    return name
