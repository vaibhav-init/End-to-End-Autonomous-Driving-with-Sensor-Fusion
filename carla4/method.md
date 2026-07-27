# Method: Camera+Radar MLP Pipeline + PCLA Agent Benchmarking

## The Pipeline

The `carla4/` pipeline trains a target-speed MLP using camera+radar sensor
fusion. The camera (YOLO) handles traffic-light detection while the CARLA
radar sensor provides reliable distance/velocity measurements to obstacles.

### Pipeline Steps

1. **Collect data** — `collect_throttle_brake_data.py` drives on autopilot
   (the "teacher") across 5 NHTSA-inspired scenarios, logging radar distance,
   YOLO traffic-light features, and ego-vehicle state at 20 FPS.

2. **Train MLP** — `train_throttle_brake.py` trains a `TargetSpeedMLP` to
   predict the teacher's future speed from stacked temporal features.

3. **Live test** — `test_throttle_brake_live.py` runs the trained model in
   the CARLA loop: radar + YOLO → features → MLP → target speed → PID →
   throttle/brake. Includes automatic scenario spawning (stopped vehicles,
   sudden brakers, pedestrian crossings) and fog variation.

## Reproducible Native-Radar Workflow

Use one sensor distribution from collection through evaluation. The supported
operating contract is CARLA native radar at 100 m, a 60 km/h maximum teacher
and prediction speed, and Town04:

```bash
python3 collect_throttle_brake_data.py --radar-backend native \
  --town Town04 --duration 900 --vehicles 60 --pedestrians 20 \
  --max-speed-kmh 60 --output dataset_throttle_brake_native
python3 collect_scenario_data.py --radar-backend native --town Town04 \
  --episodes 10 --max-speed-kmh 60 --output dataset_throttle_brake_native
python3 train_throttle_brake.py --data dataset_throttle_brake_native \
  --config dataset_throttle_brake_native/dataset_config.json \
  --output model_throttle_brake_native
python3 test_throttle_brake_live.py --radar-backend native --town Town04 \
  --model model_throttle_brake_native/target_speed_mlp.pt \
  --scaler model_throttle_brake_native/scaler.pkl \
  --config model_throttle_brake_native/model_config.json
```

Collectors attach episode IDs and reset feature history at scenario, weather,
and respawn boundaries. Training holds out complete episodes, caps stopped
frames at 15%, and rejects observed speeds above the configured envelope. Do
not mix native and C-Shenron CSVs or reuse an old scaler after recollection.

## PCLA Integration

For comparison against state-of-the-art agents, we use **PCLA** (Pretrained
CARLA Leaderboard Agents) — a framework that provides 36 pretrained agents
(Transfuser, Interfuser, LAV, etc.) that can be deployed on any CARLA vehicle.

This allows benchmarking our radar MLP against established autonomous driving
models under the same scenarios and weather conditions.

## Why Radar?

| | Camera + Radar |
|---|---|
| **Clear weather** | Works — radar gives precise distance |
| **Heavy fog** | Works — radar sees through fog |
| **Open road cruising** | Model works naturally with hybrid override |
| **Obstacle braking** | Smooth — direct distance measurement |

Radar is cheap (~$50), works in all weather, and makes the control problem
dramatically easier because distance is a direct measurement instead of a
noisy geometric estimate from camera images.
