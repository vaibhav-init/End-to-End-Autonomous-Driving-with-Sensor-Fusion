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


def choose_random_fog_weather(rng=None):
    rng = rng or random
    preset = dict(rng.choice(FOG_WEATHER_PRESETS))
    weather = carla.WeatherParameters(**{key: value for key, value in preset.items() if key != "name"})
    return preset["name"], weather


def apply_random_fog(world, rng=None):
    name, weather = choose_random_fog_weather(rng=rng)
    world.set_weather(weather)
    return name
