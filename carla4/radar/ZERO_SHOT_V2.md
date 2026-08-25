# Zero-Shot Pipeline v2 — What Changed and How to Run It

**Date:** August 25, 2026
**Goal:** Level-A zero-shot transfer — train purely on synthetic CARLA ghost
data, test directly on real Radar Ghost Dataset v1.1, no fine-tuning, no
real training data anywhere in the loop.

## What was wrong with v1 (measured)

Zero-shot AUPRC was 0.159 (near random). Root causes addressed here:

1. **Point structure:** CARLA exported ONE row per grouped target while RGD
   has ~800 CFAR surface points per frame. Statistical densification fixed
   cardinality but not spatial/amplitude/Doppler structure.
2. **Absolute amplitude feature:** `signed_log_amplitude` encodes sensor
   gain, which differs by orders of magnitude between the SNR-proxy scale
   (CARLA) and measured echo power (RGD).
3. **No micro-Doppler:** every point of a CARLA actor carried identical
   rigid-body Doppler; real pedestrian/cyclist returns spread ±0.6–1.2 m/s
   around the torso mean.
4. **Threshold saturation:** unsmoothed training let probabilities pin at
   1.0; the ≤1% FPR operating threshold saturated at 0.9995 with a 5% FPR
   floor.

## Changes

| File | Change |
|---|---|
| `radar/ghost_detection/features.py` | **Schema v2** (`radar_ghost_physical_v2`, 11 features). Removes absolute amplitude encoding; adds three frame-relative statistics computed identically in both domains: `relative_log_amplitude` (frame-median-centred log amplitude), `doppler_cluster_residual` (velocity deviation from spatial-cluster median), `local_density_ratio` (neighbours inside a fixed range/azimuth gate vs frame mean). Also adds `log_range_compensated_amplitude` = log1p(amp·R²)/10. New shared helper `frame_context_statistics()` with fixed physical gates (1.5 m / 2 deg density gate, 2 m / 3 deg cluster bins) — never fitted on either domain. |
| `radar/ghost_detection/export_expansion.py` | **New, NumPy-only module**: CFAR-emulating extended-target expansion. Poisson point count per detection, footprint-sampled positions per class (pedestrian 0.45×0.50 m … truck 9×2.5 m), Swerling-like lognormal amplitude fluctuation × radar-equation depth trend, class-scaled micro-Doppler sinusoid + noise, everything quantized onto the exact RGD grid (0.15 m / 1.8° / 0.087 m/s). Labels inherited unchanged. |
| `collect_carla_radar_ghosts.py` | Expansion wired into export (**default ON**, `--no-expand-points` for legacy behaviour, `--points-per-detection` mean count, default 12). With ~40 detections/frame this yields ~400–600 points/frame, matching RGD cardinality *and* structure. Deterministic per-(sequence, frame) RNG. |
| `radar/ghost_detection/dataset.py` | Computes `frame_context_statistics` once per (sensor, frame) at sequence load, over the complete scan, then indexes per sample. |
| `radar/ghost_detection/runtime.py` | Online filter computes identical statistics over its assembled window. |
| `train_radar_ghost_detector.py` | New `--label-smoothing` (default 0.02): targets {0,1}→{s,1−s} during training only; metrics stay on hard labels. Prevents probability saturation so the operating-threshold search can actually find a ≤1% FPR point. |
| `radar/tests/test_zero_shot_pipeline.py` | **New tests (8)**: scale-invariance of relative amplitude, Doppler-outlier detection, cluster-vs-isolated density, v2 shape/range-compensation, legacy call compatibility, expansion label inheritance + resolution grids, ped-vs-vehicle micro-Doppler separation, Poisson mean tracking. |

Densification (`densify_radar_ghost_dataset.py`) is now **obsolete** for this
pipeline — expansion replaces it at the source.

## Remote execution order

```bash
# 0. pull, then verify
cd .../carla4
python3 -m compileall .
python3 -m unittest discover -s radar/tests -p 'test_*.py'

# 1. RE-COLLECT CARLA data (expansion is default-on)
python3 collect_carla_radar_ghosts.py \
  --target-type pedestrian --split train --sequences 20 \
  --profile rgd_regime_v1 --headless \
  --output artifacts/carla_ghost_zero_shot_v2
# (repeat --split val/test as needed; smoke-test 1 sequence first and
#  confirm the ALL CHECKS PASSED block)

# 2. RE-PREPARE both domains under schema v2 (old manifests are rejected)
python3 prepare_radar_ghost_dataset.py \
  --input artifacts/carla_ghost_zero_shot_v2 \
  --output artifacts/ghost_carla_zeroshot_v2 --split-mode official
python3 prepare_radar_ghost_dataset.py \
  --input data/radar_ghost_v1_1/original \
  --output artifacts/ghost_real_official_v2 --split-mode official

# 3. Pretrain on synthetic ONLY
python3 train_radar_ghost_detector.py \
  --data artifacts/ghost_carla_zeroshot_v2 \
  --output artifacts/zeroshot_v2_carla_pretrain \
  --model temporal_pointnet --window-frames 5 --max-points 1024 \
  --hidden-dim 128 --context-dim 192 --epochs 50 --batch-size 16

# 4. Zero-shot evaluation (the number that matters)
python3 evaluate_radar_ghost_detector.py \
  --data artifacts/ghost_real_official_v2 \
  --checkpoint artifacts/zeroshot_v2_carla_pretrain/best_detector.pt \
  --split test \
  --output artifacts/zeroshot_v2_carla_pretrain/real_zero_shot.json

# 5. Reference: retrain real-only baseline under v2 features so the
#    comparison is feature-for-feature fair
python3 train_radar_ghost_detector.py \
  --data artifacts/ghost_real_official_v2 \
  --output artifacts/zeroshot_v2_real_baseline \
  --model temporal_pointnet --epochs 60 --batch-size 16
```

## Interpretation guide

- Compare step 4 against v1's zero-shot AUPRC 0.159 (same protocol).
- Sanity signals that the fixes engaged: per-frame point counts in the
  hundreds; non-zero spread in `relative_log_amplitude`; pedestrian frames
  showing Doppler residuals well above vehicle frames; operating threshold
  strictly below 0.999 with FPR near 1% instead of pinned.
- Old checkpoints/datasets are intentionally incompatible (schema gate).
