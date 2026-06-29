# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

CARLA autonomous-driving research with a **camera+radar** longitudinal control pipeline and **PCLA** (Pretrained CARLA Leaderboard Agents) for benchmarking against state-of-the-art autonomous driving agents.

Three largely independent parts:

- **`carla4/`** — Custom ML pipeline (perception → data collection → training → inference). Uses camera (YOLO) for traffic-light detection and CARLA radar for distance/obstacle sensing. This is the main project.
- **`carla4/scenarios/`** — A *separate* NHTSA-aligned evaluation harness that benchmarks CARLA's built-in Traffic Manager **autopilot** (not the ML model) across fog/seeds. Documented in `review.md` (root).
- **`PCLA/`** — Vendored third-party framework (PCLA: Pretrained CARLA Leaderboard Agents, FSE 2025) for deploying 36 pretrained leaderboard agents (Transfuser, Interfuser, LAV…). Has its own conda env and entry points; independent of `carla4/`. Recently converted from a git submodule to a regular directory.

There is no build system, test framework, or `requirements.txt`. Everything is plain `python3` scripts. Dependencies are implicit: `carla` (PythonAPI), `torch`, `ultralytics` (YOLOv8n, auto-downloads `yolov8n.pt`), `opencv` (`cv2`), `numpy`, `pandas`, `scikit-learn`.

## Prerequisites for any run

- CARLA simulator must be running on `127.0.0.1:2000` (`./CarlaUE4.sh` in `/opt/carla-simulator`).
- `CARLA_ROOT` env var (default `/opt/carla-simulator`) locates CARLA's `PythonAPI/carla` agents, imported at runtime by the live/scenario scripts.
- All sims run in **synchronous mode at FPS=20** (`fixed_delta_seconds = 1/20`).

## Common commands (run from `carla4/`)

```bash
# Collect radar+camera data (autopilot teacher), saves to dataset_throttle_brake/
python3 collect_throttle_brake_data.py

# Train the target-speed MLP
python3 train_throttle_brake.py --data dataset_throttle_brake/data.csv \
    --config dataset_throttle_brake/dataset_config.json --output model_throttle_brake

# Live drive with the trained model
python3 test_throttle_brake_live.py

# NHTSA autopilot-baseline evaluation harness (separate; Town04)
cd scenarios && python3 run_all.py                       # all S1–S4, all fog, all seeds
python3 run_all.py --scenarios 1 --fog 0 --seeds 42      # single quick run
python3 s1_lead_vehicle_stopped.py --fog 0 --seeds 42    # one scenario directly
```

## carla4 architecture (the ML pipeline)

The whole pipeline predicts a **target speed** (m/s), which a PID controller then converts to throttle/brake. The model never outputs throttle/brake directly.

**Perception — `yolo_perception.py`**
- `YOLOPerception` runs YOLOv8n, returns traffic-light features (HSV color classification of the largest light box) and obstacle features. Monocular distance comes from bbox height via pinhole geometry (used for visual overlay only — actual distance comes from radar).
- `CameraManager` wraps the RGB sensor (640×480, FOV 90, threaded frame buffer).
- `FrontRadar` (CARLA radar sensor, defined locally in each script) returns `{distance, relative_velocity, obstacle_speed}` for the closest object in a narrow forward cone.

**Feature & model contract — `speed_model.py`** (shared by all scripts)
- `BASE_FEATURE_COLS` = 10 per-frame features (ego speed/accel, distance, rel-vel, ttc, obstacle speed, 4 traffic-light features).
- `flatten_history()` stacks the last N frames into columns named `<feature>_t-<lag>` (lag 0 = current). Default history = 10 frames → model input dim = 10 × 10 = 100.
- `TargetSpeedMLP` = encoder (`encode()`) + linear speed head. The encoder is kept explicit on purpose so a future branch can fuse additional sensors into the head without reworking training/inference.
- The exact feature list and history length live in each model's `model_config.json` and are read back at inference. **Changing the feature schema means re-collecting data and retraining** — models and datasets are tied to their column list.

**Data collection** (`collect_throttle_brake_data.py`)
- Ego drives on CARLA **autopilot** (the "teacher"). Each frame logs stacked history.
- Label = smoothed *future* ego speed: `compute_future_speed_label()` averages ego speed over the next `LABEL_HORIZON` (10) frames. This imitation target is computed at save time, not live.
- Output: `dataset_throttle_brake/data.csv` + `dataset_throttle_brake/dataset_config.json` (records feature cols, history frames, label col, fps).

**Training** (`train_throttle_brake.py`)
- Reads `dataset_config.json` to get the column list. Drops idle frames (stopped + no obstacle) and downsamples stopped frames to ~15%. StandardScaler fit on train split.
- Output: `model_throttle_brake/target_speed_mlp.pt`, `scaler.pkl`, `model_config.json`.

**Inference / control loop** (`test_throttle_brake_live.py`)
- Each frame: perception → radar distance → build feature row → scale → `TargetSpeedMLP` → predicted target speed → PID/`HybridStateMachineController` → throttle/brake.
- **Hybrid override**: when no obstacle is detected and the light isn't red, target speed is floored to a cruise speed (~30 km/h). The ML model handles obstacles/braking; a simple rule handles open-road cruising (the model alone tends to under-drive on empty road). Steering is separate (lane-follow or a route agent), not learned.

## scenarios/ harness (autopilot baseline — do not confuse with the ML pipeline)

- Tests CARLA Traffic Manager autopilot, not the trained model. Runs on **Town04**, shared settings in `scenarios/config.py`.
- `s1`–`s4` = NHTSA cases (lead stopped / decelerating / constant speed / cut-in). `run_all.py` launches each as a fresh subprocess per (scenario, fog, seed). Ground truth logged by `ground_truth_logger.py` → `results_s*/` CSVs + `summary_all.csv`.

## PCLA/ (vendored, separate environment)

- Separate conda env: `conda env create -f PCLA/environment.yml`. Agents registered in `PCLA/agents.json`; main API is `PCLA.PCLA` (`get_action()` → `vehicle.apply_control(...)`). Pretrained weights are downloaded separately (`pcla_functions/download_weights.py`). See `PCLA/README.md`. Don't assume changes here affect `carla4/`.

## Conventions

- Model never emits throttle/brake — always target speed → PID. Keep that separation when adding control logic.
- Keep the radar distance-source dict interface (`distance`, `relative_velocity`, `obstacle_speed`) intact.
- Datasets and trained models are coupled to their `*_config.json` column schema — the 10-col `BASE_FEATURE_COLS` is the canonical schema.
