# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

CARLA autonomous-driving research with two entangled threads:

1. A **camera+radar longitudinal control pipeline** (perception → data collection → training → live inference → NHTSA scenario evaluation).
2. A **radar multipath-ghost study**: a physics-guided radar sensor model inside CARLA that generates synthetic **ghost** detections, and a detector trained on them.

**Read `work.md` and `handover.md` first — they are the current state** and supersede the older root reports on direction. Two things they record that nothing else does:

- **The sim-to-real thread is closed.** Four controlled experiments (v1–v4) showed CARLA ghost data does not transfer to the real Radar Ghost Dataset; point count, point density and temporal context were each ruled out as the cause. `SIM2REAL_RADAR_RESEARCH_BRIEF.md` is the record of how that was measured, not a live plan — don't restart transfer work unasked.
- **The active question is closed-loop:** does a ghost filter in the radar path reduce phantom braking and jerk, and at what cost to true-obstacle response? Four arms — A clean-trained + no filter, B + oracle (ground-truth) filter, C + learned filter, D trained on ghost-contaminated radar with no filter. All inside CARLA, where ghost labels are exact.

Layout:

- **`carla4/`** — everything above. Top-level scripts are the entry points; `radar/` is the one real package.
- **`carla4/radar/`** — pluggable forward-radar backends + the ghost-detection training/runtime subsystem. Has unit tests.
- **`carla4/scenarios/`** — NHTSA-aligned evaluation harness (Town04) with *pluggable drivers*, so the same scenarios can score the trained MLP, an IDM reference controller, or a PCLA agent.
- **`PCLA/`** — vendored third-party framework (Pretrained CARLA Leaderboard Agents, FSE 2025), own conda env, independent of `carla4/`. Call out any change under it explicitly.

No build system, no `requirements.txt`, no packaging. Plain `python3` scripts run from `carla4/`. Implicit deps: `carla` (PythonAPI), `torch`, `numpy`, `h5py`, `pandas`, `scikit-learn`, `ultralytics` (YOLOv8n, auto-downloads `yolov8n.pt`), `opencv` (`cv2`), `matplotlib`.

`AGENTS.md` holds the style/commit conventions and is the authority on those.

## Prerequisites

- CARLA **0.9.16** server on `127.0.0.1:2000` (`./CarlaUE4.sh` in `/opt/carla-simulator`). Client and server versions are checked by `validate_radar_accuracy.py`.
- `CARLA_ROOT` (default `/opt/carla-simulator`) locates CARLA's `PythonAPI/carla` agents, imported at runtime by live/scenario scripts.
- All sims run **synchronous, FPS=20** (`fixed_delta_seconds = 1/20`). The `rgd_regime_v1` radar profile runs its sensor cycle at 10 Hz inside that loop.
- Much of the code is deliberately CARLA-free (`radar/realistic_core.py`, `radar/multipath.py`, `radar/ghost_detection/*`, the dataset/train/eval scripts) so it can run and be tested on a GPU box with no simulator.

## Checks (there is no CI)

```bash
cd carla4
python3 -m compileall .                                     # syntax gate
python3 -m unittest discover -s radar/tests -p 'test_*.py'  # the only test suite
python3 -m unittest radar.tests.test_ghost_model            # single test module
python3 -m unittest radar.tests.test_ghost_model.RadarGhostModelTest.test_model_output_shapes
```

Torch and `carla` are typically absent on the authoring box, so three modules (`test_ghost_model`, `test_collector_motion`, `test_densify_radar_ghost_dataset`) fail to *import* there and the rest still run — that is expected, not a regression. The full suite (~75 tests) passes on the GPU box.

Simulator smoke test after touching scenario/driver code:

```bash
cd carla4/scenarios && python3 run_all.py --scenarios 1 --fog 0 --seeds 42
```

Record driver, weather/fog preset, seed, town, and collision/result summary when reporting scenario changes.

## Control pipeline (target speed → PID)

The model predicts a **target speed** (m/s); a PID converts it to throttle/brake. The model never emits throttle/brake. Keep that separation.

**Perception — `yolo_perception.py`**: `YOLOPerception` (YOLOv8n) gives traffic-light features (HSV color of the largest light box) and obstacle features; monocular bbox-height distance is *overlay only* — real distance comes from radar. `CameraManager` wraps the RGB sensor (640×480, FOV 90, threaded buffer).

**Feature/model contract — `speed_model.py`**: `BASE_FEATURE_COLS` = `RADAR_ONLY_FEATURE_COLS` (ego speed/accel, distance, rel-vel, ttc, obstacle speed) + `VISION_FEATURE_COLS` (4 traffic-light features) = 10 per-frame features. **A radar-only run uses the 6-column list instead** — `feature_cols_for(vision_enabled)` picks; carrying four zero-pinned columns through a 10-frame history would spend 40 of 100 inputs on constants. `flatten_history()` stacks the last N frames as `<feature>_t-<lag>` (lag 0 = current); 10 frames → input dim 100 with vision, 60 without. `model_config.json` records `vision_enabled` and `base_feature_cols`, and `mlp_driver` reads them, so a radar-only model deploys without a flag. `TargetSpeedMLP` = `encode()` + linear speed head, kept explicit so extra sensors can fuse into the head without reworking training/inference.

**Shared limits — `driving_contract.py`**: `RADAR_RANGE_M`, `MAX_TARGET_SPEED_KMH` (60 — collection labels and runtime predictions must never exceed it), `MAX_STOPPED_FRACTION`, weather segment length. Prefer importing these over re-declaring constants.

**Collection** (`collect_throttle_brake_data.py`, plus `collect_scenario_data.py` for staged NHTSA-like episodes): a teacher policy drives the ego and thereby defines the labels; each frame logs stacked history. `--teacher {autopilot,idm}` **defaults to `idm`**: CARLA autopilot almost never brakes hard, so the shipped model trained on it does not brake at all (measured: 53 km/h target with an obstacle at 8 m closing at 13.6 m/s). `--no-vision` collects from radar alone. Label = smoothed *future* ego speed over `LABEL_HORIZON` (10) frames, computed at save time. Writes `dataset_throttle_brake/data.csv` (+ `data_staged.csv`) and `dataset_config.json`.

**Training** (`train_throttle_brake.py`): reads `dataset_config.json` for the column list; `--data` accepts a CSV *or a directory* (globs all CSVs and assigns episode IDs). Drops idle frames and downsamples stopped frames to ~15%. StandardScaler on the train split. Writes `target_speed_mlp.pt`, `scaler.pkl`, `model_config.json`.

**Inference** (`test_throttle_brake_live.py`): perception → radar → feature row → scale → MLP → target speed → PID/`HybridStateMachineController`. **Hybrid override**: with no obstacle and no red light, target speed is floored to a cruise speed (~30 km/h) — the ML model handles obstacles/braking, a rule handles open road. Steering is separate (lane-follow or route agent), never learned.

**The hardcoded emergency-brake overrides are now off by default** (`--safety-rules` re-enables them as an ablation arm). They were doing *all* of the braking in every result the repo has, including `scenarios/RESULTS_COMPARISON.md` — those numbers describe a rule-based AEB, not a learned controller. Treat pre-existing MLP results as suspect until a model retrained on the IDM teacher passes the acceptance test: **obstacle at 10 m closing fast, predicted target speed must drop below current speed.** Loss curves do not substitute for it.

### The config-provenance chain (important)

Radar backend, profile, and ghost-detector identity are recorded in `dataset_config.json` at collection, copied into `model_config.json` at training (`radar_backend`, `radar_ghost_detector_signature`), and **checked at runtime**: `scenarios/drivers/mlp_driver.py` raises if either the deployed `radar_config_signature` or the `radar_ghost_detector_signature` differs from the one the model was trained under. Feature-schema or radar-config changes therefore mean re-collect → retrain, not just a flag flip.

**Known blocker:** arms A–C need a *clean-trained* model deployed *with* ghosts, which changes the signature and trips this gate. The ghost rate has to become a runtime injection knob **outside** the signature before those arms can run.

## `carla4/radar/` — sensor backends

Three interchangeable backends (`RADAR_BACKENDS = ("native", "cshenron", "realistic")`), all preserving the controller contract dict `{distance, relative_velocity, obstacle_speed}` (positive `relative_velocity` = closing):

- `native` — CARLA's `sensor.other.radar`.
- `cshenron` — C-Shenron-derived material/scattering adapter over semantic LiDAR (`cshenron_core.py`; see `C_SHENRON_NOTICE.md`).
- `realistic` — temporal target-list model (`realistic_core.py`): SNR-conditioned detections/errors, correlated dropout, weather attenuation priors, clutter/interference, geometry multipath ghosts (`multipath.py`), quantization/latency, NN tracking with M-of-N confirmation, and a yaw-rate curved-path extended-target selector.

Profiles are versioned JSON in `radar/profiles/`: `ideal_target_list_v1` (upper bound), `gaussian_baseline_v1` (naive ablation), `generic_lrr_v1` (default full model, **stochastic ghosts on**), `realistic_clean_v1` (full model, multipath off — the clean control arm), `geometry_multipath_v1` (physics ghosts), `rgd_regime_v1` (matches RGD's 10 Hz / ±70° / 1.8° / 0.087 m/s envelope). **These are visible research priors, not calibration results** — don't describe `realistic` as real-data-validated.

Every simulator-facing script shares one CLI surface via `add_radar_arguments(parser)` (`front_radar.py`): `--radar-backend`, `--radar-profile`, `--radar-config`, `--radar-seed`, `--radar-ghost-detector`, `--radar-ghost-threshold`, `--radar-ghost-device`, each with a `CARLA_RADAR_*` env fallback. Add radar options there, not per-script. Construct backends with `create_front_radar(...)`.

`validate_radar_accuracy.py` runs all three backends simultaneously against CARLA actor/bbox ground truth and writes a timestamped `radar_validation_*/`; `analyze_radar_validation.py` explains its wrong target selections without importing CARLA.

## `carla4/radar/ghost_detection/` — the sim-to-real study

Detects multipath ghost points in a radar target list. Detector inputs are only range, azimuth, Doppler, amplitude, point age; CARLA IDs, semantic classes, reflector IDs and bounce labels are supervision/diagnostics only. Models: `PointMLP` (per-point) and `TemporalPointNet` (windowed), via `create_ghost_model`.

**Feature schema is version-gated.** `FEATURE_SCHEMA_VERSION` is currently `radar_ghost_physical_v2` (11 frame-relative features — v1's absolute `signed_log_amplitude` encoded sensor gain and destroyed transfer). Prepared datasets, checkpoints, and `runtime.py` all assert the same version, so v1 artifacts are intentionally rejected. `evaluate_cross_domain.py` is the deliberate bypass for transfer diagnostics only — its numbers are not comparable to gated runs unless schemas match.

Pipeline (full protocol with verified commands: `radar/GHOST_DETECTION.md`; v2 changes and rerun order: `radar/ZERO_SHOT_V2.md`; RGD-matching collection with restart checklist: `radar/RGD_REGIME_COLLECTION.md`):

```bash
cd carla4
python3 download_radar_ghost_dataset.py                 # Zenodo v1.1, resumable, MD5-verified (~5.4 GiB)
python3 prepare_radar_ghost_dataset.py --input data/radar_ghost_v1_1/original \
    --output artifacts/ghost_real_official_v2 --split-mode official   # or scenario_grouped
python3 collect_carla_radar_ghosts.py --target-type pedestrian --split train \
    --sequences 20 --profile rgd_regime_v1 --headless --output artifacts/carla_ghost_zero_shot_v2
python3 train_radar_ghost_detector.py --data <prepared> --output <run> \
    --model temporal_pointnet --window-frames 5 --max-points 1024 --epochs 50 --batch-size 16
python3 evaluate_radar_ghost_detector.py --data <prepared> \
    --checkpoint <run>/best_detector.pt --split test --output <run>/test_metrics.json
```

Notes that are easy to get wrong:
- Use RGD **v1.1**, never v1.0 (radar/LiDAR time-sync bug).
- Under GPU pressure reduce `--batch-size`, not `--max-points` — the data definition must stay comparable across runs.
- Point-cardinality bridge: `radar/ghost_detection/export_expansion.py` (CFAR-emulating extended-target expansion, **default on** in collection, `--no-expand-points` for legacy) supersedes `densify_radar_ghost_dataset.py`, which is kept only for the v1 path.
- Stencils for densification are measured from the RGD **train split only**; val/test are never read.
- `train_and_evaluate_ghost.py` is the one-pass train+test convenience wrapper; `plot_ghost_training.py` and `analyze_ghost_dataset.py` consume its `history.json` / prepared splits.
- `verify_ghost_physics.py` checks the ghost physics (fading, incidence-dependent loss, tangential Doppler) against a **live** CARLA session — the unit tests only cover the maths in isolation. It runs with `capture_debug=True`, which deep-copies every target per frame; do not read its frame timings as sensor cost.
- `decimate_radar_ghost_dataset.py` thins an expanded CARLA export down to real prepared point density, because two of the three v2 features are frame-relative and therefore sensitive to points-per-scan. `collect_carla_radar_dataset.py` is the free-driving (non-RGD-regime) target-list collector.

## `carla4/scenarios/` — evaluation harness

Town04, shared settings in `config.py` (geometry, `FOG_LADDER` weather presets 1–4, seeds, per-scenario durations). `s1`–`s4` are the NHTSA cases (lead stopped / decelerating / constant speed / cut-in); `run_all.py` launches each `(scenario, fog, seed)` as a fresh subprocess. Ground truth via `ground_truth_logger.py` → `results_*/` CSVs + `summary_all.csv`; `analyze_results.py` and `compare_drivers.py` aggregate.

**Drivers** (`scenarios/drivers/`): a `Driver` decides only throttle/brake/steer per tick; the scenario owns spawning, weather, obstacles, logging and termination. Every driver produces **longitudinal** control from its model and delegates **lateral** control to `BasicAgent`, so runs differ only in longitudinal behavior. `make_driver()` imports lazily so each conda env loads only what it can (`pcla` pulls in PCLA, `mlp` pulls in ultralytics + sklearn, `idm` needs neither). Choose with `--driver {mlp,pcla,idm}`; `run_all.py` also forwards the shared radar flags.

`idm` (`drivers/idm_driver.py`) is the **reference longitudinal policy**: radar gap + closing speed → Treiber's IDM → target speed → the same PID tail as the MLP. Deterministic, physical parameters, no camera or weights. It is the measurement instrument that carries the study while the learned model is unreliable; the MLP is the subject. `drivers/longitudinal.py` holds the shared tail (`PIDSpeedController`, `HybridStateMachineController`, `IntelligentDriverModel`) — keeping it identical across drivers is what makes the comparison mean anything.

```bash
cd carla4/scenarios
python3 run_all.py                                       # all S1–S4 × fog × seeds
python3 run_all.py --driver mlp --model-dir ../model_throttle_brake --scenarios 1 --fog 0 --seeds 42
python3 s1_lead_vehicle_stopped.py --fog 0 --seeds 42    # one scenario directly
python3 compare_drivers.py --runs mlp=results_mlp pcla=results_pcla
```

## PCLA/ (vendored)

Separate conda env (`conda env create -f PCLA/environment.yml`); agents in `PCLA/agents.json`; API is `PCLA.PCLA` (`get_action()` → `vehicle.apply_control(...)`); weights via `pcla_functions/download_weights.py`. Changes here don't affect `carla4/`. `scenarios/diagnose_pcla_handover.py` debugs the handover between PCLA control and the harness.

## Conventions

- Model emits target speed only; PID converts to throttle/brake.
- Keep the radar result dict (`distance`, `relative_velocity`, `obstacle_speed`) intact — it is the interface every backend and the ghost filter must satisfy.
- Datasets and models are coupled to their `*_config.json` schema; the 10-col `BASE_FEATURE_COLS` is canonical for the controller, `FEATURE_SCHEMA_VERSION` for the ghost detector.
- Prefer `argparse` options over hard-coded experiment values; new radar options go through `add_radar_arguments`.
- Generated output stays out of commits — `.gitignore` covers `carla4/artifacts/`, `data/`, `dataset_*/`, `model_*/`, `radar_validation_*/`, `scenarios/results*/`. Downloaded RGD data and weights are never committed.
- Be precise in docs and commit messages about what is *measured* vs. what is a *prior*: the radar profiles and ghost-transfer claims are the parts most easily overstated.

## Traps that have cost real time

- **Never run the collector or live inference detached without a TTY.** `collect_throttle_brake_data.py` and `test_throttle_brake_live.py` start a thread calling `input()` for manual obstacle spawns. With no controlling terminal that raises SIGTTIN and **suspends the whole process** — state `T`, zero CPU, no CSV, CARLA frozen waiting for a tick. It looks exactly like a sensor deadlock. Guarded by an `stdin.isatty()` check (commit `367336d`) that is *committed but never verified end to end*. Diagnose with `ps -o stat -p <pid>` before theorising.
- **All three backend constructors end in `**_ignored`**, so a misspelled radar kwarg is silently dropped. `radar_profile=` instead of `profile_name=` once produced an entire run on the wrong profile that still printed confident PASS lines. Assert the config you actually got.
- **`min_detection_probability` puts a ~2% floor under any SNR**, so the sensor emits a trickle of detections regardless of link budget. That is how −118 dB ghosts reached the output.
- **Killed runs leak sensors.** A semantic LiDAR left attached keeps streaming at 240k points/s and crushes throughput; a crashed run also leaves CARLA in synchronous mode with nobody ticking it (the "frozen CARLA" symptom). Sweep stray actors before blaming the code.
- **Don't pipe a long run through `tail`/`grep` or wrap it in `timeout`** — output is hidden and SIGTERM skips the `finally` that writes the CSV, so a working run and a hung one look identical. Use `python3 -u` into a log file.
- **Change one variable at a time.** A session that changed teacher, vision and radar backend together lost hours debugging the combination. A/B against the last known-good configuration first.

## Documentation map

Root, current: `work.md` (phase plan, scope boundary, measured throughput), `handover.md` (session state — what is verified vs. merely committed).
Root, historical background: `SIM2REAL_RADAR_RESEARCH_BRIEF.md` (the closed transfer study, most complete statement of its method and results), `RADAR_FALSE_ALARM_RESEARCH_REPORT.md`, `RESEARCH_DIRECTION_REPORT.md`, `PROFESSOR_RESEARCH_DIRECTION.md`, `deepresearch.md`, `report.md`.
`carla4/radar/`: `README.md` (backend architecture + profiles), `GHOST_DETECTION.md` (execution protocol), `ZERO_SHOT_V2.md`, `RGD_REGIME_COLLECTION.md`, `REALISM_ROADMAP.md`.
`carla4/scenarios/`: `EVALUATION_GUIDE.md`, `RESULTS_COMPARISON.md`.
