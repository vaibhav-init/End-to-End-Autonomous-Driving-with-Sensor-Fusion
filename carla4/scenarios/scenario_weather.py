#!/usr/bin/env python3
"""
Shared weather utility for NHTSA scenarios.

Defines 4 weather presets designed to test camera vs radar robustness:

  Preset 1 — DARK NIGHT:      Pitch black, headlights only. Cameras struggle, radar unaffected.
  Preset 2 — DENSE FOG:       Maximum fog, near-zero visibility. Camera blind, radar works.
  Preset 3 — CLEAR DAY:       Perfect conditions. Both drivers should perform well.
  Preset 4 — NIGHT+FOG+RAIN:  Worst-case combo. Camera nearly useless, radar still works.

Usage:
    from scenario_weather import set_weather_condition
    set_weather_condition(world, preset=1)  # dark night
"""

import carla

# Preset IDs used in FOG_LADDER config
PRESET_DARK_NIGHT = 1
PRESET_DENSE_FOG = 2
PRESET_CLEAR_DAY = 3
PRESET_NIGHT_FOG_RAIN = 4

PRESET_NAMES = {
    1: "Dark Night",
    2: "Dense Fog (0 visibility)",
    3: "Clear Day",
    4: "Night + Fog + Rain",
}


def set_weather_condition(world, density):
    """Apply a weather preset.

    ``density`` is used as a preset ID (1–4) matching FOG_LADDER in config.py.
    """
    weather = world.get_weather()

    if density == PRESET_DARK_NIGHT:
        # ── DARK NIGHT: pitch black, only headlights ──
        weather.sun_altitude_angle = -30.0       # well below horizon
        weather.sun_azimuth_angle = 0.0
        weather.cloudiness = 100.0               # overcast (no moonlight)
        weather.precipitation = 0.0
        weather.precipitation_deposits = 0.0
        weather.fog_density = 0.0
        weather.fog_distance = 100.0
        weather.fog_falloff = 0.0
        weather.wetness = 0.0
        weather.wind_intensity = 5.0

    elif density == PRESET_DENSE_FOG:
        # ── DENSE FOG: maximum fog, near-zero visibility ──
        weather.sun_altitude_angle = 30.0        # daytime but invisible
        weather.sun_azimuth_angle = 60.0
        weather.cloudiness = 100.0
        weather.precipitation = 0.0
        weather.precipitation_deposits = 20.0    # damp roads
        weather.fog_density = 100.0              # MAX fog
        weather.fog_distance = 0.0               # fog starts at camera
        weather.fog_falloff = 0.0                # uniform fog at all heights
        weather.wetness = 40.0                   # damp
        weather.wind_intensity = 0.0

    elif density == PRESET_CLEAR_DAY:
        # ── CLEAR DAY: perfect conditions ──
        weather.sun_altitude_angle = 50.0        # bright midday
        weather.sun_azimuth_angle = 60.0
        weather.cloudiness = 10.0
        weather.precipitation = 0.0
        weather.precipitation_deposits = 0.0
        weather.fog_density = 0.0
        weather.fog_distance = 100.0
        weather.fog_falloff = 0.0
        weather.wetness = 0.0
        weather.wind_intensity = 10.0

    elif density == PRESET_NIGHT_FOG_RAIN:
        # ── NIGHT + FOG + RAIN: worst-case combo ──
        weather.sun_altitude_angle = -25.0       # night
        weather.sun_azimuth_angle = 0.0
        weather.cloudiness = 100.0               # total overcast
        weather.precipitation = 90.0             # heavy rain
        weather.precipitation_deposits = 80.0    # deep puddles
        weather.fog_density = 80.0               # thick fog
        weather.fog_distance = 5.0               # fog very close
        weather.fog_falloff = 0.5                # low-hanging fog
        weather.wetness = 100.0                  # maximum wet
        weather.wind_intensity = 60.0            # strong wind

    else:
        # Fallback: treat as linear density (legacy behavior)
        weather.precipitation = min(100.0, density * 1.0)
        weather.precipitation_deposits = min(100.0, density * 0.9)
        weather.fog_density = min(100.0, density * 0.7)
        weather.fog_distance = max(5.0, 60.0 - density * 0.6) if density > 0 else 100.0
        weather.fog_falloff = min(2.0, 0.2 + density * 0.02) if density > 0 else 0.0
        weather.wetness = min(100.0, density * 1.0)
        weather.cloudiness = min(100.0, density * 1.1)
        weather.wind_intensity = min(100.0, density * 0.5)
        weather.sun_altitude_angle = max(10.0, 45.0 - density * 0.3)
        weather.sun_azimuth_angle = 60.0

    world.set_weather(weather)
