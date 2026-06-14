# Project Methodology — CARLA Autonomous Driving with MLP + YOLO

## System Overview

An autonomous vehicle controller built in the CARLA simulator. The ego vehicle perceives the world through **sensors only** (front radar + RGB camera with YOLOv8) and uses a trained MLP neural network to control throttle and brake. Steering is handled separately by CARLA's BasicAgent.

```
                    ┌─────────────┐
                    │  RGB Camera  │──→ YOLOv8n ──→ traffic_light_state
                    │  640×480     │             └──→ approaching_intersection
                    └─────────────┘
                    ┌─────────────┐
                    │ Front Radar  │──→ distance, relative_velocity, obstacle_speed
                    └─────────────┘
                    ┌─────────────┐
                    │ Ego Physics  │──→ ego_speed, ego_acceleration, target_speed
                    └─────────────┘
                            │
                            ▼
                    ┌───────────────┐
                    │  StandardScaler│
                    └───────┬───────┘
                            ▼
                    ┌───────────────┐
                    │ MLP 9→64→32→2 │──→ throttle ∈ [0,1]
                    │   (Sigmoid)   │──→ brake    ∈ [0,1]
                    └───────────────┘

        Steering: BasicAgent (pathfinding + lane keeping, separate from MLP)
```

---

## File-by-File Methodology

---

### `yolo_perception.py` — Perception Module

**Purpose:** Real-time visual perception using YOLOv8 + HSV color analysis. This is the "eyes" of the vehicle.

**How it works:**

1. **CameraManager** — Attaches a 640×480 RGB camera sensor to the ego vehicle's roof (x=1.5, z=2.4, pitch=-5°). Runs at 20 FPS matching the simulation tick rate. Stores the latest frame via a threaded callback.

2. **YOLOPerception** — Loads a pre-trained YOLOv8n model (trained on COCO, 80 object classes). Each frame, it runs inference looking for two specific COCO classes:
   - **Class 9: traffic light** — detects the bounding box of traffic lights
   - **Class 11: stop sign** — detects stop signs

3. **Traffic light state detection** (two-step process):
   - **Step 1 — YOLO locates the light:** Finds the bounding box with highest confidence for class 9 (traffic light)
   - **Step 2 — HSV classifies the color:** Crops the bounding box region, converts to HSV color space, filters for bright pixels (V > 150 to isolate the lit bulb from the dark housing), then counts pixels in three hue ranges:
     - Red: H < 10 or H > 170
     - Yellow: H 15–35
     - Green: H 40–85
   - The dominant color wins. Temporal smoothing requires 2 consecutive same-state detections to prevent flicker.
   - Output: 0=none, 1=green, 2=yellow, 3=red

4. **Intersection detection:**
   - If YOLO detects ANY traffic light or stop sign in the frame → `approaching_intersection = 1`
   - This is a visual cue — same way a human driver recognizes an upcoming intersection
   - Uses cached results from the traffic light detection (zero additional compute)

**Key design decision:** No CARLA simulator APIs are used for perception features. Everything comes from the camera sensor, just like a real self-driving car would work.

---

### `collect_throttle_brake_data.py` — Data Collection

**Purpose:** Record driving data to train the MLP. The autopilot drives while sensors observe.

**How it works:**

1. **Autopilot configuration** — CARLA's Traffic Manager drives the ego as a careful, law-abiding driver:
   - Speed: 10% above posted speed limit
   - Following distance: 15m to leading vehicle
   - Global safety margin: 12m
   - Obeys all traffic lights, stop signs, and pedestrians
   - Auto lane change enabled for natural driving

2. **Emergency obstacle injection** — Every ~20 seconds (or on manual ENTER press), a stationary vehicle is spawned 60–80m ahead of the ego on the same lane. This forces the autopilot to perform emergency braking, generating critical braking-transition data. The obstacle is removed after the ego stops for 1 second.

3. **Stuck detection** — If the ego is stationary for 5+ seconds (without an emergency obstacle), it's teleported to a new spawn point to prevent data poisoning from getting stuck behind traffic.

4. **Feature recording** — Each frame (20 FPS), records 9 input features:

   | Feature | Source | Description |
   |---------|--------|-------------|
   | `ego_speed` | Vehicle physics | Current speed in m/s |
   | `target_speed` | Speed limit × 1.10 | Desired cruise speed in m/s |
   | `ego_acceleration` | `(speed - prev_speed) × FPS` | Instantaneous acceleration, clamped to [-20, 20] |
   | `distance` | Front radar | Distance to nearest obstacle in meters |
   | `relative_velocity` | Front radar | Closing speed (positive = getting closer) |
   | `ttc` | Computed | Time-to-collision = distance / relative_velocity, capped at 10s |
   | `obstacle_speed` | Front radar | Speed of the detected obstacle |
   | `approaching_intersection` | YOLO camera | 1 if traffic light or stop sign visible, else 0 |
   | `traffic_light_state` | YOLO camera + HSV | 0=none, 1=green, 2=yellow, 3=red |

5. **Labels** — The autopilot's actual pedal inputs: `autopilot_throttle` and `autopilot_brake` (both ∈ [0, 1]).

6. **Live camera preview** — OpenCV window shows the camera feed with YOLO bounding boxes drawn around detected traffic lights (color-coded by state), plus a HUD overlay with speed, TL state, intersection status, and distance/TTC.

7. **Output:** `dataset_throttle_brake/data.csv`

**Front radar configuration:**
- Range: 50m, horizontal FOV: 10° (narrow beam to avoid adjacent lanes)
- Vertical FOV: 2° with 2° upward pitch (avoids ground returns)
- Filters: azimuth < 0.3 rad, depth > 1m, altitude > -0.02 rad

---

### `train_throttle_brake.py` — MLP Training

**Purpose:** Train the neural network to predict throttle and brake values from the 9 sensor features.

**How it works:**

1. **Data loading and cleaning:**
   - Drops uninformative idle frames (speed=0, no obstacle within 49m)
   - Caps stopped frames (speed < 0.5 km/h) at 15% of the dataset — prevents thousands of identical "stopped behind traffic" rows from drowning out the critical braking-transition data where speed goes from 25 → 0 m/s

2. **Time shift (anticipatory braking):**
   - Default: 10 frames = 0.5 seconds
   - Shifts the label columns forward in time, so the model learns to react 0.5s EARLIER than the autopilot did
   - Example: if the autopilot braked at frame 100, the model sees brake=1.0 at frame 90 — it learns to start braking before the autopilot would have

3. **Feature normalization:**
   - StandardScaler (mean=0, std=1) fitted on training data
   - Scaler saved as `scaler.pkl` — must be loaded at inference time for consistent normalization

4. **MLP architecture:**
   ```
   Input (9)
     → Linear(9, 64) → ReLU → Dropout(0.2)
     → Linear(64, 32) → ReLU → Dropout(0.1)
     → Linear(32, 2) → Sigmoid
   Output: [throttle, brake] each ∈ [0, 1]
   ```
   - Total parameters: ~2,500
   - Sigmoid output ensures predictions stay in valid [0, 1] range for both pedals

5. **Training details:**
   - Loss: MSE (Mean Squared Error) between predicted and actual throttle/brake
   - Optimizer: Adam (lr=0.001, weight_decay=1e-4)
   - Scheduler: ReduceLROnPlateau (halves LR after 15 epochs without improvement)
   - Sequential train/val split (80/20, no shuffle — time-series data would leak future info if shuffled)
   - Best model saved based on validation loss

6. **Output:** `model_throttle_brake/throttle_brake_mlp.pt` + `model_throttle_brake/scaler.pkl`

---

### `test_throttle_brake_live.py` — Live Inference Test

**Purpose:** Deploy the trained model in CARLA and verify it can actually drive safely.

**How it works:**

1. **Control split:**
   - **MLP** controls throttle and brake — the model's predictions are applied directly to the vehicle
   - **BasicAgent** controls steering — handles pathfinding, lane following, and turn execution
   - BasicAgent respects traffic lights and stop signs (so it doesn't steer into an intersection while the MLP is braking)

2. **Sensor setup** — Same as data collection:
   - Front radar (50m range, 10° FOV)
   - RGB camera (640×480) + YOLOv8n
   - Collision recorder (logs any impacts)

3. **Per-frame inference loop:**
   ```
   Get ego speed, acceleration → from vehicle physics
   Get radar data → distance, relative_velocity, obstacle_speed, ttc
   Get camera frame → run YOLO → traffic_light_state, approaching_intersection
   Build 9-feature vector
   Scale with saved scaler
   Feed to MLP → get throttle, brake
   Get steer from BasicAgent
   Apply control to vehicle
   ```

4. **Creep override** — If the model holds the brake while stopped and the nearest obstacle is > 12m away, the override kicks in with throttle=0.35 to prevent the car from getting permanently stuck. (The model sometimes learns to brake indefinitely from training data where the ego was stuck behind traffic.)

5. **Manual obstacle spawning** — Press ENTER to spawn a stationary vehicle 60–80m ahead. Tests whether the model brakes in time. The obstacle auto-removes after the ego stops for 2 seconds.

6. **Background traffic** — Configurable NPC vehicles and pedestrians create realistic urban driving conditions.

7. **Metrics tracked:**
   - Total collisions (with actor type and impulse)
   - Near-miss frames (obstacle < 5m while moving)
   - Minimum distance seen
   - Brake/throttle frame percentages

---

### `scenarios.py` — Structured Scenario Tests

**Purpose:** Repeatable, deterministic test scenarios to validate specific driving behaviors.

**How it works:**

Each scenario follows the same pattern: spawn ego → setup sensors → run phases → measure results → report pass/fail.

#### Scenario 1: Stutter Stop

Tests reaction to a lead vehicle that suddenly brakes.

- **Phase 1 (Warmup, 10s):** Lead vehicle drives ahead on TM autopilot at 20 km/h. Ego follows using MLP control, building speed.
- **Phase 2 (Brake, 8s):** Lead vehicle's autopilot is killed and full brakes applied. The closing distance triggers the ego's MLP to brake.
- **Phase 3 (Resume, 7s):** Lead vehicle is destroyed. Ego should detect clear road and resume driving.
- **Pass condition:** Zero collisions.

#### Scenario 2: Red Light Stop

Tests traffic light detection and response.

- Traffic light at a nearby intersection is forced to red (held for 30s).
- **Phase 1 (Accelerate, 5s):** Manual throttle brings ego up to speed toward the intersection.
- **Phase 2 (Approach, 10s):** MLP takes control. YOLO should detect the red light, and the model should reduce throttle / increase brake.
- **Pass condition:** YOLO detected red light AND ego stopped AND zero collisions.

---

## Pipeline Execution Order

```
Step 1: pip install ultralytics
Step 2: Start CARLA server
Step 3: python collect_throttle_brake_data.py --duration 900
Step 4: python train_throttle_brake.py --epochs 150
Step 5: python test_throttle_brake_live.py --duration 120
Step 6: python scenarios.py
```

---

## Key Design Principles

1. **Sensor-only perception** — No CARLA ground-truth APIs used for input features during collection or inference. The vehicle sees through radar and camera only.

2. **Same features everywhere** — The 9-feature vector is identical in collection, training, and inference. No domain gap.

3. **Anticipatory braking** — The 10-frame time shift teaches the model to brake 0.5s before the autopilot would have, compensating for inference latency.

4. **Dual-output MLP** — Separate throttle and brake outputs (not a single action value) lets the model learn nuanced behaviors like coasting (low throttle, zero brake) vs. trail braking (some throttle, some brake).
