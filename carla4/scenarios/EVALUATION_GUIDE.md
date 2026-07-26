# NHTSA Scenario Evaluation — MLP vs PCLA Driver Comparison

## Overview

This evaluation runs **3 NHTSA-aligned driving scenarios** under **4 extreme weather presets** for **2 autonomous drivers** (MLP and PCLA), then compares their performance using per-tick telemetry.

**Total experiment matrix:** 3 scenarios × 4 weathers × 1 seed × 2 drivers = **24 runs**

---

## Drivers

| Driver | Sensing | Control | Conda Env | Strengths |
|--------|---------|---------|-----------|-----------|
| **MLP** | Radar (100m range) + Camera (YOLO) | MLP model → PID + hardcoded safety rules | `carla4` | Works in darkness, fog, rain — radar is weather-blind |
| **PCLA** | Camera only (TransFuser end-to-end) | TransFuser neural network | `PCLA` | Sophisticated learned behavior, smooth control |

### MLP Safety Rules (hardcoded overrides)
1. If model predicts target speed < 1 km/h → **full brake** (throttle=0, brake=1.0)
2. If obstacle detected within **30m** on radar → **full brake** (throttle=0, brake=1.0)

---

## Scenarios

### S1: Lead Vehicle Stopped
- **What happens:** Ego accelerates to 60 km/h on highway, then a stopped vehicle is placed **25m ahead**.
- **Test:** Can the driver detect and emergency-brake in time?
- **Staging:** SpeedController pushes ego to 60 km/h before obstacle spawns (both drivers start at identical speed)
- **Script:** `s1_lead_vehicle_stopped.py`
- **Duration:** 30 seconds

### S2: Lead Vehicle Decelerating
- **What happens:** Ego follows an NPC at 60 km/h with a **15m gap**. At step 200, the NPC **slams brakes**.
- **Test:** Can the driver react to a sudden deceleration ahead?
- **Staging:** ON by default — GapKeepController maintains 15m gap until NPC brakes, then hands control to the driver
- **Script:** `s2_lead_vehicle_decelerating.py`
- **Duration:** 35 seconds

### S4: Cut-In from Adjacent Lane
- **What happens:** NPC drives in the adjacent lane **5m ahead** at 60 km/h. At step 80, NPC **cuts into ego's lane** and brakes to a full stop.
- **Test:** Can the driver react to a sudden lane intrusion?
- **Staging:** ON by default — GapKeepController holds 8m gap, hands over when both vehicles > 45 km/h
- **Script:** `s4_cut_in.py`
- **Duration:** 35 seconds

---

## Weather Presets

| ID | Name | Camera Impact | Radar Impact | What CARLA Does |
|----|------|---------------|--------------|-----------------|
| **1** | Dark Night | ❌ Nearly blind (headlights only) | ✅ Unaffected | `sun_altitude=-30`, full overcast, no moon |
| **2** | Dense Fog | ❌ Total whiteout | ✅ Unaffected | `fog_density=100`, `fog_distance=0`, zero visibility |
| **3** | Clear Day | ✅ Perfect visibility | ✅ Unaffected | Bright midday sun, no fog/rain |
| **4** | Night+Fog+Rain | ❌ Completely useless | ✅ Unaffected | Night + 90% rain + 80% fog + max wetness |

**Key insight:** 3 out of 4 presets severely degrade cameras but have zero effect on radar. This highlights the radar-based MLP's advantage.

---

## File Structure

```
carla4/scenarios/
├── config.py                  # Seeds, weather presets, scenario geometry
├── scenario_weather.py        # 4 weather preset definitions
├── run_all.py                 # Master runner (calls all scenarios)
├── s1_lead_vehicle_stopped.py # Scenario 1
├── s2_lead_vehicle_decelerating.py  # Scenario 2
├── s4_cut_in.py               # Scenario 4
├── compare_drivers.py         # Human-readable metric comparison
├── analyze_results.py         # CDF plot generator
├── ground_truth_logger.py     # Per-tick CSV logger
├── drivers/
│   ├── __init__.py            # Driver factory (--driver mlp|pcla)
│   ├── mlp_driver.py          # MLP + radar + hardcoded safety
│   └── pcla_driver.py         # TransFuser PCLA wrapper
├── results_mlp/               # MLP output CSVs (created at runtime)
│   ├── results_s1/
│   ├── results_s2/
│   └── results_s4/
└── results_pcla/              # PCLA output CSVs (created at runtime)
    ├── results_s1/
    ├── results_s2/
    └── results_s4/
```

---

## How to Run

### Step 1: Run MLP Driver

```bash
conda activate carla4
cd carla4/scenarios

# All 3 scenarios, all 4 weathers
python run_all.py --driver mlp --scenarios 1 2 4 --output-root results_mlp --model-dir ../model_throttle_brake

# Or individual scenarios:
python s1_lead_vehicle_stopped.py --driver mlp --fog 1 2 3 4 --seeds 42 --output results_mlp/results_s1
python s2_lead_vehicle_decelerating.py --driver mlp --fog 1 2 3 4 --seeds 42 --output results_mlp/results_s2
python s4_cut_in.py --driver mlp --fog 1 2 3 4 --seeds 42 --output results_mlp/results_s4
```

### Step 2: Run PCLA Driver

```bash
conda activate PCLA
cd carla4/scenarios

# All 3 scenarios, all 4 weathers
python run_all.py --driver pcla --scenarios 1 2 4 --output-root results_pcla --pcla-agent tfv6_visiononly

# Or individual scenarios:
python s1_lead_vehicle_stopped.py --driver pcla --fog 1 2 3 4 --seeds 42 --output results_pcla/results_s1
python s2_lead_vehicle_decelerating.py --driver pcla --fog 1 2 3 4 --seeds 42 --output results_pcla/results_s2
python s4_cut_in.py --driver pcla --fog 1 2 3 4 --seeds 42 --output results_pcla/results_s4
```

### Step 3: Compare Results

```bash
# Human-readable table with reaction times, stopping distances, etc.
python compare_drivers.py --runs mlp=results_mlp pcla=results_pcla

# CDF plots (for paper/report figures)
python analyze_results.py --runs pcla=results_pcla mlp=results_mlp --out comparison
```

---

## Metrics Collected (Per Tick, 20 FPS)

| Column in CSV | Description |
|---------------|-------------|
| `gt_ego_speed_kmh` | Ego vehicle speed |
| `gt_npc_speed_kmh` | NPC vehicle speed |
| `gt_distance_to_npc_m` | Distance to NPC/obstacle |
| `gt_relative_velocity` | Closing rate (m/s) |
| `time_to_collision_s` | Time-to-collision at current closing rate |
| `throttle` | Driver's throttle command (0–1) |
| `brake` | Driver's brake command (0–1) |
| `steer` | Driver's steer command (-1 to +1) |
| `ego_accel_mps2` | Longitudinal acceleration (m/s²) |
| `collision_occurred` | 0 or 1 |
| `min_distance_so_far_m` | Running minimum distance (= stopping distance in last row) |

## Derived Metrics (from compare_drivers.py)

| Metric | How Computed | What It Tells You |
|--------|-------------|-------------------|
| **Stopping Distance** | min(`gt_distance_to_npc_m`) | How close the ego got before stopping |
| **Reaction Time** | First obstacle tick → first `brake > 0.3` | How fast the driver started braking |
| **Peak Deceleration** | max(−`ego_accel_mps2`) | Strongest braking force applied |
| **Time to Stop** | First obstacle tick → `ego_speed < 1 km/h` | Total time to reach full stop |
| **Min TTC** | min(`time_to_collision_s`) | Closest moment to impact |
| **Collision Speed** | `gt_ego_speed_kmh` at collision tick | Speed at impact (if crashed) |

---

## Configuration Reference

Current settings in `config.py`:

| Parameter | Value | Effect |
|-----------|-------|--------|
| `S1_OBSTACLE_DISTANCE` | 25m | Obstacle placed 25m ahead at 60 km/h |
| `S2_NPC_INITIAL_GAP` | 15m | NPC starts 15m ahead |
| `S2_BRAKE_TRIGGER_STEP` | 200 | NPC brakes at step 200 (10s at 20 FPS) |
| `S4_NPC_AHEAD_M` | 25m | NPC starts 25m ahead in adjacent lane |
| `S4_CUT_IN_TRIGGER_STEP` | 60 | Cut-in fires at step 60 (3s) |
| `RANDOM_SEEDS` | [42] | 1 seed |
| `FOG_LADDER` | [1, 2, 3, 4] | 4 weather presets |

---

## Expected Outcome

The evaluation is designed to highlight the **radar advantage** of the MLP driver:

- **Dark Night / Dense Fog / Night+Fog+Rain:** PCLA's camera is severely degraded → expect slower reaction or collisions. MLP's radar is unaffected → should maintain consistent performance.
- **Clear Day:** Both drivers should perform well (fair comparison baseline).

This supports the research narrative: *"A lightweight radar-based MLP driver maintains safe braking performance in degraded visibility conditions where camera-based end-to-end systems (PCLA/TransFuser) fail."*
