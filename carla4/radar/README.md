# Forward Radar Backends

This package has three interchangeable forward-radar backends:

- `native`: CARLA's low-fidelity `sensor.other.radar`.
- `cshenron`: the existing C-Shenron-derived material/scattering adapter.
- `realistic`: a temporal automotive target-list model built on the
  C-Shenron-derived ideal returns.

All three preserve the controller contract:

```python
{
    "distance": 32.8,
    "relative_velocity": 4.1,  # positive means closing
    "obstacle_speed": 8.3,
}
```

The `realistic` name describes the intended backend, not a completed
sensor-validation claim. The default profile combines published sensor
resolution figures with explicit research priors. It must be fitted and
validated against held-out real sequences before being described as
real-data-calibrated.

## Architecture

```text
CARLA semantic LiDAR
  geometry + occlusion + semantic/instance ID + incidence cosine
        |
        v
C-Shenron-derived material scattering and ideal target extraction
        |
        v
Temporal target-list sensor model
  SNR-conditioned detections and errors
  correlated dropout and colored noise
  rain/fog/dust attenuation priors
  clutter and interference bursts
  persistent multipath-like ghosts
  range/azimuth/Doppler quantization and latency
        |
        v
Nearest-neighbour tracking + M-of-N confirmation + deletion
        |
        v
Ego-path gate -> distance / relative_velocity / obstacle_speed
```

The ideal radial velocity is calculated from CARLA actor motion and then
corrupted and quantized. That mirrors a perception-error-model workflow:
CARLA provides the latent truth, while the sensor model produces the
observable target list. Static world returns without a live CARLA actor use
zero world velocity.

## Built-in Profiles

| Profile | Purpose |
|---|---|
| `ideal_target_list_v1` | No noise, misses, latency, clutter, or ghosts; useful as the upper-bound baseline |
| `gaussian_baseline_v1` | Independent fixed Gaussian error and independent misses; deliberately simple ablation |
| `generic_lrr_v1` | Full temporal model, tracking, path gating, clutter, interference, ghosts, latency, and weather priors |

The default `generic_lrr_v1` envelope follows the public RadarScenes setup
where applicable: 100 m range, approximately ±60° field of view, 0.15 m range
resolution, 0.1 km/h radial-velocity resolution, and angle resolution that
degrades away from boresight. Its cycle is 50 ms instead of RadarScenes'
reported average 60 ms so it stays synchronized with this repository's 20 Hz
control loop.

The probabilities and error-distribution parameters are not RadarScenes
calibration results. They are visible, versioned priors in
`profiles/generic_lrr_v1.json` plus the defaults in `realistic_core.py`.

## CARLA 0.9.16 Accuracy Validator

With a CARLA 0.9.16 server already running, the shortest controlled comparison
is:

```bash
cd carla4
python3 validate_radar_accuracy.py
```

The script verifies both CARLA client and server versions, spawns a same-lane
ego/lead pair, runs `native`, `cshenron`, and `realistic` simultaneously in
synchronous mode, and compares them with CARLA actor/bounding-box ground
truth. Source builds sometimes return a Git build identifier instead of
`0.9.16`; matching client/server identifiers are accepted after the required
radar and semantic-LiDAR blueprint capabilities are audited. It restores world
settings and weather and destroys only the actors and sensors that it created.

It writes a timestamped directory containing:

- `radar_samples.csv`: per-frame ground truth, selected outputs, current and
  latency-aligned errors, frame synchronization, misses, target identity,
  SNR, clutter, ghosts, interference, and tracker state;
- `radar_details.jsonl`: every native return and each extracted, generated,
  delivered, and tracked target;
- `metadata.json`: exact arguments, versions, actor geometry, weather,
  resolved radar configuration, and the CARLA 0.9.16 semantic contract;
- `summary.json`: any-output, miss, lead-recall, and wrong-target rates plus
  range/Doppler bias, MAE, RMSE, median, p95, and maximum absolute errors.

The terminal ends with a block headed `RADAR VALIDATION SUMMARY — COPY THIS
BLOCK BACK TO CODEX`. Send that block first. If semantic-tag or target
association warnings appear, also send `metadata.json` and `summary.json`.
To classify wrong realistic selections without rerunning CARLA, analyze the
existing detailed log:

```bash
python3 analyze_radar_validation.py radar_validation_ideal_v2
```

The forensic report separates missing extraction/detection, latency, tracker
association loss, path-gate rejection, and closer competing targets.

Useful variants:

```bash
# Short smoke run with no rendering
python3 validate_radar_accuracy.py --duration-s 10 --no-rendering

# Adverse-weather sensitivity run
python3 validate_radar_accuracy.py --fog-density 80 \
  --precipitation 80 --wetness 100 --duration-s 30

# A custom calibrated profile
python3 validate_radar_accuracy.py \
  --radar-config /absolute/path/to/my_sensor_profile.json \
  --output radar_validation_my_sensor
```

The validator geometrically associates CARLA native radar's selected hit with
the lead oriented bounding box because the native radar API does not return an
actor ID. The semantic backends expose CARLA truth IDs only for validation;
the realistic tracker itself still associates by range, azimuth, and Doppler.
For scalar ACC scoring, the lead must both be sensor-visible and overlap the
ego-path corridor. A vehicle merely visible in a wide field of view is not
automatically the correct longitudinal target. The terminal reports intrinsic
range/Doppler error only on frames where the selected target is actually the
lead; selected-output error remains available separately to diagnose target
selection.

### CARLA 0.9.16 Compatibility Boundary

The public C-Shenron integration comes from an older CARLA/Leaderboard line,
and its data-conversion material table uses the older semantic ordering. This
port does not copy that numeric table. For CARLA 0.9.16 it explicitly uses:

- the 24-byte semantic-LiDAR record `x, y, z, cos_incidence, uint32 object_id,
  uint32 semantic_tag`;
- UE sensor coordinates `x` forward, `y` right, `z` up, in metres;
- the post-0.9.14 tag table `0..28`, including `12 Pedestrian`, `14 Car`,
  `20 Static`, `21 Dynamic`, `25 Ground`, and `28 GuardRail`;
- current actor IDs for dynamic objects and range/angle cells for static
  surfaces.

Every validation run logs the observed raw tag histogram and the semantic tags
actually associated with the lead actor ID. Unknown IDs, an all-zero raw
semantic stream, client/server version mismatch, and failure to extract the
lead actor are surfaced as terminal warnings.

## Remote Usage

Collect and train in a new artifact directory:

```bash
cd carla4
python3 collect_throttle_brake_data.py \
  --radar-backend realistic \
  --radar-profile generic_lrr_v1 \
  --output dataset_throttle_brake_realistic

python3 collect_scenario_data.py \
  --radar-backend realistic \
  --radar-profile generic_lrr_v1 \
  --output dataset_throttle_brake_realistic

python3 train_throttle_brake.py \
  --data dataset_throttle_brake_realistic \
  --config dataset_throttle_brake_realistic/dataset_config.json \
  --output model_throttle_brake_realistic

python3 test_throttle_brake_live.py \
  --radar-backend realistic \
  --model model_throttle_brake_realistic/target_speed_mlp.pt \
  --scaler model_throttle_brake_realistic/scaler.pkl \
  --config model_throttle_brake_realistic/model_config.json
```

The exact resolved profile is embedded in `dataset_config.json`, copied into
`model_config.json`, and identified by a SHA-256-derived configuration ID.
Collection refuses to append to an incompatible dataset. Live inference and
the scenario MLP driver refuse a configuration that differs from training.

Run scenario evaluation with a deterministic sensor seed:

```bash
cd carla4/scenarios
python3 run_all.py \
  --driver mlp \
  --radar-backend realistic \
  --radar-profile generic_lrr_v1 \
  --radar-seed 42 \
  --model-dir ../model_throttle_brake_realistic \
  --scenarios 1 2 4 --fog 0 50 100 --seeds 42 43 44
```

Normally omit `--radar-seed` during multi-seed evaluation. Each scenario seed
then also seeds the radar, giving paired and reproducible artifact sequences.

## Custom or Real-Data-Calibrated Profile

Pass a partial JSON override:

```json
{
  "profile_name": "my_sensor_heldout_fit_v1",
  "false_alarms_per_scan": 0.14,
  "dropout_enter_probability": 0.021,
  "dropout_exit_probability": 0.27,
  "ghost_start_probability": 0.006,
  "ghost_survival_probability": 0.96,
  "range_noise_floor_m": 0.08
}
```

```bash
python3 collect_throttle_brake_data.py \
  --radar-backend realistic \
  --radar-config /absolute/path/to/my_sensor_heldout_fit_v1.json \
  --output dataset_throttle_brake_my_sensor
```

Unknown fields, invalid probabilities, invalid tracker settings, and
unsupported schema versions fail before CARLA is contacted. The complete
resolved configuration—not just the override—is saved with the dataset.

Fit separate profiles per physical radar or dataset. Split calibration and
validation by complete drive/sequence, never by neighboring frames. At
minimum, fit and validate:

- probability of detection by range, angle, radial speed, class, and aspect;
- conditional range, azimuth, Doppler, and amplitude/SNR errors;
- miss-run lengths and temporal error autocorrelation;
- detections per real object and unstructured clutter per scan;
- interference-burst frequency and duration;
- ghost type, offset, Doppler relation, lifetime, and reflecting surface;
- tracker confirmation time, fragmentation, and false confirmed tracks.

## Diagnostics

`radar.diagnostics()` exposes the current profile and configuration ID,
sensor/frame time, ideal and delivered target counts, misses, direct/ghost/
clutter counts, interference state, active and confirmed tracks, and the
selected track's source, confidence, truth ID, and azimuth.

The collection CSVs and scenario `ground_truth.csv` files record these
diagnostics alongside the controller-facing radar values.

Truth IDs are diagnostic only. Nearest-neighbour tracking does not associate
measurements using CARLA actor IDs.

## What Is and Is Not Modeled

Implemented:

- CARLA ray-cast visibility and occlusion;
- C-Shenron-derived material and incidence-dependent scattering;
- extended-object grouping before target-list generation;
- finite sensor cadence, range/Doppler/angle resolution, Doppler wrapping,
  processing latency, SNR-conditioned misses and errors;
- temporally correlated dropout and measurement error;
- rain/fog/dust attenuation priors and wet-road ghost-rate scaling;
- random clutter, interference bursts, persistent dynamic ghosts;
- multi-frame tracking, confirmation, confidence, coasting, deletion, and
  path-aware longitudinal target selection.

Not implemented:

- raw FMCW chirps, phase noise, ADC saturation, CFAR, antenna calibration, or
  a range-Doppler-angle cube;
- geometrically exact multi-bounce ray tracing and surface-normal-aware ghost
  placement;
- micro-Doppler from wheels, limbs, vibration, or rotating parts;
- polarization, radome/bumper effects, mutual coupling, sidelobe maps, or
  commercial object-list firmware;
- physically simulated spray droplets, snow accumulation, road water films,
  or sensor contamination;
- calibration against a specific production radar.

Use the full upstream C-Shenron, RadaRays, a GPU ray tracer, or hardware/
recording replay if raw-signal realism is required. This backend is designed
for fast, reproducible closed-loop studies of how realistic target-list
artifacts propagate into longitudinal control.

## Existing C-Shenron Compatibility Backend

The `cshenron` backend remains available for compatibility. It applies the
material classes and surface-scattering equations from the public C-Shenron
code, performs SNR gating, and adds fixed range/velocity error before selecting
the nearest target. It does not create the upstream ADC cube or range-angle
image and should be called a C-Shenron-derived target-list port.

Do not mix `native`, `cshenron`, and `realistic` rows in one dataset. Do not
reuse a scaler or model across profiles.

## Primary References

- [CARLA 0.9.16 sensor reference](https://carla.readthedocs.io/en/0.9.16/ref_sensors/)
- [Official C-Shenron implementation](https://github.com/ucsdwcsng/C-Shenron)
- [C-Shenron paper](https://ucsdwcsng.github.io/files/c-shenron-paper.pdf)
- [RadarScenes sensor setup](https://radar-scenes.com/dataset/sensors/)
- [RadarScenes labeling limitations](https://radar-scenes.com/dataset/labeling/)
- [Radar Ghost Dataset](https://github.com/flkraus/ghosts)
- [Perception Error Model for virtual testing](https://arxiv.org/abs/2302.11919)
