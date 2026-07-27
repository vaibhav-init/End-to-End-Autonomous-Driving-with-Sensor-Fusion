# C-Shenron Radar Adapter

This package provides two interchangeable forward-radar backends:

- `native`: the original `sensor.other.radar` implementation.
- `cshenron`: a CARLA 0.9.16 compatibility port driven by
  `sensor.lidar.ray_cast_semantic`.

Both expose the existing controller contract:

```python
{
    "distance": 32.8,
    "relative_velocity": 4.1,
    "obstacle_speed": 8.3,
}
```

## Usage

Select the backend from supported scripts:

```bash
python3 test_throttle_brake_live.py --radar-backend cshenron
python3 collect_throttle_brake_data.py --radar-backend cshenron
cd scenarios
python3 run_all.py --driver mlp --radar-backend cshenron
```

Or set `CARLA_RADAR_BACKEND=cshenron` for scripts that use the shared factory.
Direct Python integration is:

```python
from radar import create_front_radar

radar = create_front_radar(
    ego, world, range_m=100.0, backend="cshenron", fps=20
)
radar.update_ego_speed(ego_speed_mps)
features = radar.get()
radar.cleanup()
```

No additional package installation is required beyond the repository's
existing CARLA and NumPy dependencies.

## Recommended Migration

Do not mix native and C-Shenron rows in one dataset. Use separate artifact
directories:

```bash
cd carla4
python3 collect_throttle_brake_data.py \
  --radar-backend cshenron --output dataset_throttle_brake_cshenron
python3 collect_scenario_data.py \
  --radar-backend cshenron --output dataset_throttle_brake_cshenron
python3 train_throttle_brake.py \
  --data dataset_throttle_brake_cshenron \
  --config dataset_throttle_brake_cshenron/dataset_config.json \
  --output model_throttle_brake_cshenron
python3 test_throttle_brake_live.py \
  --radar-backend cshenron \
  --model model_throttle_brake_cshenron/target_speed_mlp.pt \
  --scaler model_throttle_brake_cshenron/scaler.pkl \
  --config model_throttle_brake_cshenron/model_config.json
```

For a quick integration smoke test, the old model can be run with the new
backend because the field names and units are unchanged. The scripts print a
distribution-mismatch warning until a matching model is supplied.

## Compatibility Boundary

The upstream public code creates a dense ADC cube and range-angle image using
Open3D, SciPy, Torch, PyNVML, and `mat4py`. This adapter instead preserves its
semantic material classes and surface-scattering equations, applies
signal-to-noise gating, and converts qualified returns to the scalar target
list expected by the current MLP.

It is therefore a C-Shenron-derived target-list port, not a bit-for-bit copy of
the upstream range-angle image pipeline. Existing native-radar weights can run
because the feature schema is unchanged, but C-Shenron data have different
noise and detection statistics. Recollect the dataset and retrain the MLP
before reporting final comparative results.
