# RGD-Regime CARLA Ghost Collection — Session Runbook

This file records why, what, and how for reproducing the **Radar Ghost Dataset
(RGD) v1.1–matching CARLA collection** and restarting the sim-to-real ghost
detector pipeline from this point. Read `GHOST_DETECTION.md` for the overall
pipeline; this file is the focused runbook for the RGD-regime collection step.

**Session status (last verified):** a single 38.5 s pedestrian sequence
collected with `ALL CHECKS PASSED`. See
[Verified Run](#verified-run-3-all-checks-passed) below.

---

## 1. Why this exists: the RGD vs CARLA setup gap

The ghost-detector pipeline trains on the real
[Radar Ghost Dataset v1.1](https://github.com/flkraus/ghosts) (Kraus et al.,
[arXiv 2404.01437](https://arxiv.org/abs/2404.01437)) and pretrains on CARLA.
The two setups are **not interchangeable**:

### Real RGD v1.1 setup
- **Sensor:** 77 GHz chirp-sequence radar, **10 Hz**, range 0.15–153 m,
  azimuth **±70°**, unambiguous Doppler ±44.3 m/s; resolutions Δr 0.15 m,
  Δφ **1.8°**, Δv **0.087 m/s**. Two experimental sensors in the front bumper
  (the `left`/`right` channels; no single front sensor).
- **Ego:** **parked and stationary** for every recording ("to prolong the time
  where multi-path reflections occur").
- **Scenes:** 21 scenarios; the main object is a **pedestrian or cyclist**
  walking away from and back toward the ego near reflective surfaces (plastered/
  marble walls, metal containers, parked cars, curbstone, guardrail).
  111 sequences × ~385 frames (38.5 s) at 10 Hz ≈ 71 min.
- **Data:** ~820 raw CFAR detections/frame (127–1775); ~35M points; only
  ~100k ghost + ~600k real points labeled. Ghosts: type-1 2nd-order, type-2
  2nd/3rd-order, plus "other multipath".

### CARLA side (this repo, before this work)
- Semantic-LiDAR-derived target list + temporal model; 20 Hz; 100 m; ±60° FOV.
- The old ghost collectors moved a **vehicle** near reflectors with a moving
  ego (wrong class and wrong Doppler regime for RGD transfer).

### Ranked mismatches that matter for sim-to-real transfer
1. **Point cardinality:** RGD ~800 raw points/frame vs CARLA's sparse target
   list (tens of points). Biggest gap; monitor, or export denser points later.
2. **Ego motion:** stationary (RGD) vs driving (CARLA). Doppler/clutter
   statistics differ structurally.
3. **Object class:** RGD main objects are pedestrians/cyclists; old CARLA
   ghosts were vehicles.
4. **Cadence/geometry:** 10 vs 20 Hz, ±70° vs ±60°, 2 sensors vs 1.
5. **Amplitude:** CARLA `amp` is a synthetic `10^(SNR_dB/20)` proxy, not
   calibrated to RGD amplitude statistics.
6. **Ghost statistics:** RGD ghosts are measured RF multipath; CARLA ghosts
   come from the planar image-method geometry + priors.

---

## 2. What was implemented

| File | Change |
|---|---|
| `carla4/radar/profiles/rgd_regime_v1.json` | **New profile** matching the RGD sensor envelope + geometry multipath (see below). |
| `carla4/radar/realistic_core.py` | Registered `rgd_regime_v1` in `REALISTIC_RADAR_PROFILES`. |
| `carla4/collect_carla_radar_ghosts.py` | RGD-regime collection: `--target-type`, 10 Hz/38.5 s defaults, walker/motorcycle targets, speed-aware motion, verification block. |
| `carla4/radar/front_radar.py` | Transform-derivative velocity fallback so physics-off kinematic actors report real Doppler. |
| `carla4/radar/GHOST_DETECTION.md` | New section 6.1b documenting the RGD-regime collection. |
| `carla4/radar/README.md` | Added `rgd_regime_v1` row to the built-in profiles table. |

### `rgd_regime_v1` profile
```json
{
  "profile_name": "rgd_regime_v1",
  "multipath_mode": "geometry",
  "ghost_start_probability": 0.0,
  "ghost_survival_probability": 0.0,
  "max_active_ghosts": 0,
  "cycle_time_s": 0.1,
  "horizontal_fov_deg": 140.0,
  "range_resolution_m": 0.15,
  "azimuth_resolution_boresight_deg": 1.8,
  "azimuth_resolution_edge_deg": 1.8,
  "doppler_resolution_mps": 0.087,
  "max_unambiguous_doppler_mps": 44.3
}
```
Multipath mode is `geometry` (deterministic image-method paths from
semantic-LiDAR reflector segments); probabilistic ghosts are disabled.

### `collect_carla_radar_ghosts.py` key behavior
- `--target-type {vehicle, pedestrian, cyclist}` (default `vehicle`):
  - `pedestrian` spawns a **walker** (CARLA tag 12 → RGD class 1).
  - `cyclist` spawns a **two-wheel motorcycle** (tag 18 → RGD class 5
    "motorbike"; CARLA has no cyclist actor — documented mismatch).
  - `vehicle` is the original 4-wheel behavior (tag 14 → class 3).
- Defaults now `--fps 10` and `--duration 38.5` (~385 frames, matching RGD).
- The controlled target is placed by the production image-method solver
  (`_configure_controlled_target`) and **teleported kinematically each tick**
  (physics off) along the reflector tangent, with the oscillation period
  derived from the target's speed (`TARGET_SPEEDS_MPS`:
  vehicle 3.0, pedestrian 1.4, cyclist 4.5 m/s).
- Walkers are spawned with retries over ~24 navigation locations plus a
  fallback next to the ego (single-location spawns are flaky).
- Per-sequence **verification block** is printed (see
  [Verification](#4-verification-block-meaning) below).

### `front_radar.py` velocity fallback
CARLA reports **zero velocity** for physics-off actors (`get_velocity()` ≈ 0),
which killed pedestrian Doppler (and therefore ghost Doppler, since ghosts
inherit it). Fix: `_estimate_kinematic_velocity(actor, timestamp_s)` derives
true motion from successive callback transforms (`Δposition / Δtime`) and is
used in `_target_closing_speed` whenever a dynamic actor's reported velocity
norm is < 1e-3. Real (physics-enabled) actors are unaffected.

---

## 3. Verified runs (chronological)

### Run 1 — teleport + physics off, NO velocity fallback (`artifacts/carla_ghost_rgd`)
Geometry and labels fine, **Doppler dead**:
```
direct target: 363 detections, mean |vr|=0.055 m/s (expected ~1.4), max |vr|=0.348
[FAIL] direct target speed plausible: 0.055 vs 1.400 m/s
```
Cause: physics-off walker reports zero velocity.

### Run 2 — WalkerControl with physics ON (`artifacts/carla_ghost_rgd_v2`)
Crashed during validation — walker not detected at all:
```
RuntimeError: The controlled CARLA vehicle did not produce an observed multipath
target ... last_dynamic_ids=[]. Try a different --seed.
```
Cause: teleported walkers with physics enabled are unstable at 75 m near the
guardrail (fall/settle out of the lidar FOV). Physics-off teleporting is the
stable path.

### Run 3 — teleport + physics off + transform-derivative fallback (`artifacts/carla_ghost_rgd_v3`) ✅
**ALL CHECKS PASSED.** Summary (the JSON printed after the sequence):
```
capture_frames: 385
target_type: pedestrian (tag 12)
ego_speed_mps: 0.0
direct_target_speed_mean_mps: 0.822   (expected ~1.4; radial component only)
direct_target_speed_max_mps: 1.479
real: 486   ghost: 1623
label_class_histogram: {1: 1984, 2: 33, 3: 85, 5: 7}
ghost_family_histogram: {type1-order2: 677, type2-order2: 645, type2-order3: 301}
radar_profile: rgd_regime_v1   radar_fps: 10   radar_fov_deg: 140.0
radar_config_signature: fdc02b5821512e28
reflector: id=-4332483265467969687, tag=28 (GuardRail), length=7.43 m
validated_path_families: [type1-order2, type2-order2]
```
The 0.82 m/s mean is correct physics: the walker moves **along the reflector
tangent**, so the radar only sees the radial component of the 1.4 m/s walking
speed; the 1.48 m/s max confirms full walking speed is present.

---

## 4. Verification block meaning

After each sequence the script prints
`RGD REGIME COLLECTION VERIFICATION — COPY THIS BLOCK BACK` with checks:

| Check | Passes when |
|---|---|
| `ego stationary` | ego speed < 0.5 m/s |
| `radar fps = 10` | collected at 10 Hz |
| `radar fov = 140 deg` | ±70° envelope |
| `ghost points > 0` / `real points > 0` | both classes present |
| `expected class present` | RGD class 1 (ped) / 5 (cyclist) / 3 (vehicle) appears in labels |
| `direct target Doppler alive` | mean \|vr\| > 0.05 m/s |
| `direct target speed plausible` | 0.1× ≤ mean \|vr\| ≤ 1.6× of configured speed (radial-component band) |
| `controlled reflector used` | a reflector id was chosen |

If any check FAILs, copy the block back to the session and debug before
scaling up.

---

## 5. Commands (run in this order)

Start CARLA first (`./CarlaUE4.sh -quality-level=Epic`), then:

```bash
cd carla4
```

**5.1 Smoke test (1 pedestrian sequence):**
```bash
python3 collect_carla_radar_ghosts.py \
  --town Town04 --output artifacts/carla_ghost_rgd \
  --split train --sequences 1 --duration 38.5 \
  --vehicles 30 --walkers 15 --lead-distance 25 \
  --target-type pedestrian --radar-timeout 30
```

**5.2 Full collection (train/val/test):**
```bash
python3 collect_carla_radar_ghosts.py \
  --town Town04 --output artifacts/carla_ghost_rgd \
  --split train --sequences 20 --duration 38.5 \
  --vehicles 30 --walkers 15 --lead-distance 25 \
  --target-type pedestrian --seed 100 --radar-timeout 30 --headless

python3 collect_carla_radar_ghosts.py \
  --town Town04 --output artifacts/carla_ghost_rgd \
  --split val --sequences 4 --duration 38.5 \
  --vehicles 30 --walkers 15 --lead-distance 25 \
  --target-type pedestrian --seed 2000 --radar-timeout 30 --headless

python3 collect_carla_radar_ghosts.py \
  --town Town04 --output artifacts/carla_ghost_rgd \
  --split test --sequences 4 --duration 38.5 \
  --vehicles 30 --walkers 15 --lead-distance 25 \
  --target-type pedestrian --seed 4000 --radar-timeout 30 --headless
```
- Optionally repeat with `--town Town03` for wall/building diversity
  (filenames include the town, so they merge into the same output dir).
- Expect ~1–3 min per sequence (38.5 s sim + world load/teardown).
- **Segfault after ~7 sequences:** many `load_world` cycles in one process
  accumulate native CARLA state and crash. The collector runs each sequence
  in a **fresh worker subprocess** (same pattern as
  `collect_carla_radar_dataset.py`) with `--resume` and
  `--sequence-retries`. If a run dies, rerun the identical command with
  `--resume`; it skips sequences that already have an `.h5` + `.summary.json`
  sidecar.
- Sanity-check a few verification blocks during the run.

**5.3 Prepare H5 for training:**
```bash
python3 prepare_radar_ghost_dataset.py \
  --input artifacts/carla_ghost_rgd \
  --output artifacts/ghost_carla_prepared \
  --split-mode official
```
Open `artifacts/ghost_carla_prepared/manifest.json` and confirm each split has
both `real_points` and `ghost_points`.

**5.4 Train the temporal PointNet on CARLA (needs torch/GPU):**
```bash
python3 train_radar_ghost_detector.py \
  --data artifacts/ghost_carla_prepared \
  --output artifacts/ghost_temporal_carla_pretrain \
  --model temporal_pointnet \
  --window-frames 5 --max-points 1024 \
  --hidden-dim 128 --context-dim 192 \
  --epochs 50 --batch-size 16
```

**5.5 Rest of the pipeline** (from `GHOST_DETECTION.md` steps 7–9):
evaluate zero-shot on real RGD test → fine-tune on RGD → evaluate → deploy
`best_detector.pt` only if it beats real-only and synthetic-only baselines →
recollect/retrain controller data with the filter → closed-loop scenario runs.

---

## 6. Pitfalls & debugging notes

- **Walker spawn flakiness:** `try_spawn_actor` at a single random nav point
  often returns None. The code retries ~24 nav locations + an ego-adjacent
  fallback. If `Unable to spawn the pedestrian radar target` still appears,
  check the town has walker blueprints/nav mesh.
- **Never enable walker physics for the controlled target** (Run 2): the
  teleported walker falls/settles out of the lidar FOV and validation crashes
  with `last_dynamic_ids=[]`. Keep `set_simulate_physics(False)`.
- **Zero Doppler symptom** (Run 1): mean |vr| ≈ 0.05 with max ≈ 0.35 → the
  physics-off actor reports no velocity. This is fixed by the
  transform-derivative fallback in `front_radar.py`; do not "fix" it by
  re-enabling physics.
- **Low mean |vr| is expected:** tangential motion → only the radial component
  is measured (0.82 vs 1.4 m/s above). Use the max, not the mean, to confirm
  the full speed.
- **Ghost Doppler is inherited from the parent target** — a dead parent Doppler
  means dead ghost Doppler, so always check the direct-target checks first.
- **`collect_carla_radar_dataset.py` is a different regime** (moving ego,
  vehicle events) — do not use it for RGD-matching pretraining.
- **Cyclist caveat:** a CARLA motorcycle maps to RGD class 5 (motorbike), not
  class 2 (cyclist).
- Keep datasets/model artifacts out of Git.

---

## 7. Restart checklist (start here in a new session)

1. Read this file + `GHOST_DETECTION.md`.
2. Confirm the code state: `rgd_regime_v1` profile exists and loads
   (`python3 -c "from radar import load_realistic_radar_config as l; print(l('rgd_regime_v1'))"`),
   `collect_carla_radar_ghosts.py` has `--target-type`, `front_radar.py` has
   `_estimate_kinematic_velocity`.
3. Confirm artifacts: `artifacts/carla_ghost_rgd_v3/train/*.h5` (the verified
   single sequence) exists; decide whether to keep or rebuild it as part of
   the full train set.
4. Run 5.2 (full collection) → check verification blocks → run 5.3 (prepare)
   → inspect `manifest.json` → run 5.4 (pretrain).
5. Continue with GHOST_DETECTION.md steps 7–9.
