# Vision (removed — how to put it back)

The camera + YOLO perception path was removed from `carla4/` so the pipeline is
radar-only. Nothing about it was fixed or deprecated; it was taken out to keep
the ghost study on one sensor. This file records exactly what left, so it can
be reinstated deliberately rather than reconstructed from memory.

**Everything below is recoverable from git.** The last commit that still had a
working vision path is the parent of the removal commit; `carla4/yolo_perception.py`
is the single module to restore:

```bash
git log --oneline -- carla4/yolo_perception.py     # find the removal commit
git show <commit>^:carla4/yolo_perception.py > carla4/yolo_perception.py
```

At the time of removal, HEAD was `3e93de3`.

---

## What was removed

### 1. `carla4/yolo_perception.py` (478 lines, deleted)

| Symbol | What it did |
|---|---|
| `CameraManager` | RGB sensor wrapper — 640×480, FOV 90, threaded frame buffer |
| `YOLOPerception` | YOLOv8n (`yolov8n.pt`, auto-downloaded) traffic-light + obstacle detection |
| `classify_traffic_light_color` | HSV heuristic on the largest light box → red/yellow/green |
| `VisionDistanceTracker` | Monocular bbox-height distance estimate, used only by the vision-only driver |
| `empty_visual_features` / `empty_obstacle_features` | Zeroed bundles for "camera present, nothing detected" |
| `TL_NONE/GREEN/YELLOW/RED`, `TL_STATE_NAMES` | Traffic-light state enum |

Implicit dependency dropped with it: **`ultralytics`**, and `opencv` (`cv2`) is
no longer imported anywhere in `carla4/`.

### 2. Four feature columns

`speed_model.BASE_FEATURE_COLS` was 10 columns; it is now the 6 radar/ego ones.
The removed four were:

```
traffic_light_state    # 0 none, 1 green, 2 yellow, 3 red
tl_confidence          # YOLO box confidence
tl_bbox_area           # box area as a fraction of the frame
tl_center_x            # box centre x, normalised 0..1
```

With the default 10-frame history this changes the model input from **100 to 60**.
`speed_model.feature_cols_for(vision_enabled)` and `RADAR_ONLY_FEATURE_COLS` /
`VISION_FEATURE_COLS` are gone — there is one column list again.

**Any checkpoint trained with 100 inputs will not load against the current
schema.** That is intentional, and the same version gate that has always
covered radar config changes: re-collect and retrain, don't force it.

### 3. Per-script surface

| File | What was taken out |
|---|---|
| `collect_throttle_brake_data.py` | `--no-vision` flag, `CameraManager`/`YOLOPerception` setup, `draw_camera_overlay()` (the OpenCV preview window), the `tl=`/`area=` progress fields, `vision_enabled` in `dataset_config.json` |
| `collect_scenario_data.py` | camera/YOLO setup and the visual half of `build_base_features()` |
| `test_throttle_brake_live.py` | camera/YOLO setup, the visual feature row, the red-light guard on the cruise floor |
| `scenarios/drivers/mlp_driver.py` | camera/YOLO setup, `vision_enabled`, `use_radar` (the vision-only inference mode and `VisionDistanceTracker` fallback), `MAX_RANGE` (the 50 m monocular range), the `TL_RED` guard on the cruise floor |
| `train_throttle_brake.py` | `vision_enabled` carried into `model_config.json` |

### 4. Two behaviours that changed, not just code

- **The cruise floor no longer checks for a red light.** It used to be
  "cruise at ~30 km/h when nothing is detected **and** the light is not red".
  It is now just "when nothing is detected". On a route with traffic lights the
  ego will drive through them. `BasicAgent` steering already runs with
  `ignore_traffic_lights(True)`, so this is consistent, but it is a real
  behavioural change to the deployed controller.
- **`mlp_driver` no longer has a vision-only mode.** It selected the distance
  source from the feature schema (`obstacle_detected` in the columns meant
  monocular distance instead of radar). Radar is now the only source.

---

## Reinstating it

1. Restore `carla4/yolo_perception.py` from git (command above).
2. Put the four columns back in `speed_model.py`. Prefer restoring
   `feature_cols_for(vision_enabled)` over hard-coding the 10-column list, so
   radar-only runs stay possible — carrying four zero-pinned columns through a
   10-frame history spends 40 of 100 inputs on constants.
3. Re-add the camera/YOLO setup and the visual half of the feature row to the
   four scripts in the table above.
4. Restore `vision_enabled` through the provenance chain:
   `dataset_config.json` → `model_config.json` → `mlp_driver`. That flag is
   what stops a radar-only model from being deployed with a camera attached.
5. Re-add the red-light guard to the cruise floor in **both**
   `test_throttle_brake_live.py` and `scenarios/drivers/mlp_driver.py` — they
   duplicate that logic and must stay in step.
6. Re-collect and retrain. The feature schema changes, so old datasets and
   checkpoints do not carry over.

Install `ultralytics` again before step 3; `yolov8n.pt` downloads on first use.
