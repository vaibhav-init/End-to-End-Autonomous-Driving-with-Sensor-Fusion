# Vision-Only Version — Applied Fixes

## Overview

Applied a set of fixes to the vision-only data collection pipeline targeting YOLO-based monocular distance estimation, velocity computation, and feature quality. The changes address noisy relative velocity, detection-gap instability, and asymmetric braking bias.

---

## Files Modified

### 1. `yolo_perception.py`

#### 1a. Primary obstacle selection — tightened lane filter + min area guard

**File:** `YOLOPerception._extract_obstacle_from_detections()`

| Before | After |
|---|---|
| `if 0.25 <= center_x <= 0.75 and est_dist < best_distance:` | `if 0.325 <= center_x <= 0.675 and est_dist < best_distance and area > 0.002:` |

**Why:** The center 50% of the frame is too wide — it picks up cars in adjacent lanes, especially on curves. Tightening to 35% (center 67.5% → 67.5-32.5 = 35%) reduces false positives. The minimum area filter (`area > 0.002`) rejects tiny false-positive detections far away or from noise.

#### 1b. `VisionDistanceTracker.__init__` — new state variables

**Added fields:**
- `self._raw_prev = self.max_range` — previous **raw** distance (pre-smoothing) for velocity computation
- `self._smoothed_velocity = 0.0` — separately smoothed velocity estimate
- `self.vel_alpha = 0.25` — EMA alpha for velocity smoothing (slower than distance)

**Removed:**
- `self._prev_smoothed = self.max_range` — no longer needed (replaced by `_raw_prev`)

**Why:** Previously velocity was computed from EMA-smoothed distance → double smoothing (EMA on distance, then diff → velocity) created ~150ms of lag. Computing velocity from raw distance and smoothing it separately reduces lag while still producing stable readings.

#### 1c. `VisionDistanceTracker.reset()` — match new state

| Before | After |
|---|---|
| `self._smoothed = self.max_range` | `self._smoothed = self.max_range` |
| `self._prev_smoothed = self.max_range` | `self._raw_prev = self.max_range` |
| (missing) | `self._smoothed_velocity = 0.0` |
| `self._missed_frames = 0` | `self._missed_frames = 0` |

#### 1d. `VisionDistanceTracker.update()` — velocity from raw signal, symmetric clamp

**Before:**
1. Store `_prev_smoothed = _smoothed`
2. Update `_smoothed` with EMA of raw_distance
3. `relative_velocity = (prev_smoothed - smoothed) * fps` ← **lagged signal**
4. Clamp `max(-5.0, min(15.0, relative_velocity))` ← **asymmetric, biased toward braking**
5. On miss: distance releases, velocity = 0

**After:**
1. Compute `raw_velocity = (raw_prev - raw_distance) * fps` ← **less lag**
2. Clamp `np.clip(raw_velocity, -8.0, 8.0)` ← **symmetric**
3. Store `raw_prev = raw_distance`
4. EMA smooth distance separately (unchanged)
5. EMA smooth velocity separately with `vel_alpha = 0.25`
6. On miss: distance releases, velocity decays by 0.85× per frame

**Why:** Symmetric clamp (±8 m/s) removes the braking bias from the old -5/+15 clamp. Separate velocity smoothing lets distance react quickly while keeping velocity stable.

---

### 2. `collect_vision_only_data.py`

#### 2a. EMA alpha lowered from 0.4 → 0.15

| Before | After |
|---|---|
| `DISTANCE_EMA_ALPHA = 0.4` | `DISTANCE_EMA_ALPHA = 0.15` |

**Why:** At 20 FPS, an alpha of 0.4 makes the distance estimate jump significantly every frame. 0.15 provides much smoother estimates for longitudinal control.

#### 2b. Detection-gap holdout

**Added constants & state variables:**
```python
MAX_DETECTION_GAP_FRAMES = 8
no_detection_frames = 0
last_valid_obstacle = empty_obstacle_features()
```

**Added logic in the main frame loop (after YOLO inference):**
```python
if obstacle["has_detection"] and obstacle["vision_distance"] < MAX_VISION_RANGE:
    no_detection_frames = 0
    last_valid_obstacle = obstacle
else:
    no_detection_frames += 1

if no_detection_frames < MAX_DETECTION_GAP_FRAMES:
    obstacle = last_valid_obstacle
else:
    obstacle = empty_obstacle_features()
```

**Why:** When YOLO misses a frame, the old code immediately snapped distance to 50m, making the model think the road is clear → lurch forward. The holdout keeps the last valid state for up to 8 frames (~0.4s) before releasing to empty.

#### 2c. New binary feature: `obstacle_detected`

**Added to base_features dict:**
```python
obstacle_detected = float(vision_state["distance"] < MAX_VISION_RANGE * 0.95)
```

**Why:** Previously when no car was ahead, the model saw distance=50, rel_vel=0, ttc=10 — the same as a car 48m away. The binary flag gives the MLP a clean signal to distinguish "open road" from "car at 48m".

---

### 3. `collect_vision_only_data.py` (continued)

#### 3c. Local `VISION_FEATURE_COLS` (instead of modifying shared `BASE_FEATURE_COLS`)

Instead of modifying the shared `BASE_FEATURE_COLS` in `speed_model.py` (which would break the radar collector since it doesn't include `obstacle_detected`), a local extended list is defined:

```python
from speed_model import BASE_FEATURE_COLS as _BASE_FEATURE_COLS, flatten_history

VISION_FEATURE_COLS = _BASE_FEATURE_COLS + ["obstacle_detected"]
```

This is used for `flatten_history()`, `stacked_feature_names()`, and the `dataset_config.json`'s `base_feature_cols` / `stacked_feature_cols` — keeping the radar pipeline completely untouched.

The `speed_model.py` shared module was **not modified**. The 10-column shared `BASE_FEATURE_COLS` remains unchanged.

---

## Feature Distribution Expectations

After these fixes, healthy feature distributions should look like:

| Feature | Open road | Following at 15m | Emergency stop |
|---|---|---|---|
| `distance` | ~50.0 | ~15.0 | ~8.0 |
| `relative_velocity` | 0.0 | ±2–4 m/s | +5–8 m/s |
| `ttc` | 10.0 | 3–7s | <2s |
| `obstacle_detected` | 0 | 1 | 1 |

## Post-Implementation Fixes Applied

### Fix A: File corruption on import line

During the original edit, fragment text ("traffic lights / scene semantics / maybe lane/context") was accidentally merged into the `from ultralytics import YOLO` line, causing a `SyntaxError`. Fixed by restoring the clean import.

### Fix B: Undefined variable `relative_velocity`

The rewritten `VisionDistanceTracker.update()` method used `relative_velocity` in the return dict, but the variable was named `self._smoothed_velocity` in the new code. Fixed `round(relative_velocity, 4)` → `round(self._smoothed_velocity, 4)`.

## Retraining Required

Because `VISION_FEATURE_COLS` now includes `obstacle_detected` (11 columns vs the previous 10), any models trained with the previous schema must be retrained on a new dataset collected with these fixes.
