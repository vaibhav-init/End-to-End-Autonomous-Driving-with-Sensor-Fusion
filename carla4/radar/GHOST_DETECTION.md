# Physics-Guided Radar Ghost Detection

This is the remote execution guide for the implemented research pipeline. The
scope is deliberately precise: detect multipath ghost objects in an automotive
radar target list. It does not claim to synthesize raw FMCW ADC samples or to
detect every kind of radar false alarm.

The pipeline compares a point-wise MLP, a real-data temporal PointNet,
synthetic CARLA pretraining, and synthetic pretraining followed by real-data
fine-tuning. CARLA IDs, semantic classes, reflector IDs, and bounce labels are
supervision/diagnostics only. Detector inputs are range, azimuth, Doppler,
amplitude, and point age. The controller still receives only `distance`,
`relative_velocity`, and `obstacle_speed`.

## Files to Run

| Order | File | Purpose |
|---:|---|---|
| 1 | `download_radar_ghost_dataset.py` | Fetch and verify official v1.1 data |
| 2 | `prepare_radar_ghost_dataset.py` | Convert official or CARLA H5 sequences |
| 3 | `train_radar_ghost_detector.py` | Train the point or temporal detector |
| 4 | `evaluate_radar_ghost_detector.py` | Evaluate held-out real sequences |
| 5 | `collect_carla_radar_ghosts.py` | Generate path-labeled CARLA multipath |
| 6 | `validate_radar_accuracy.py` | Validate the CARLA target-list sensor |
| 7 | `collect_throttle_brake_data.py` | Recollect final controller data |
| 8 | `collect_scenario_data.py` | Add staged controller-training episodes |
| 9 | `train_throttle_brake.py` | Retrain the target-speed MLP |
| 10 | `test_throttle_brake_live.py` | Run the final controller |
| 11 | `scenarios/run_all.py` | Run scenario evaluation |

## Step 0: Remote Environment

Use the existing `carla4` environment, not the independent PCLA environment.
CARLA server and Python API must both be 0.9.16-compatible.

```bash
cd /path/to/carla-claude/carla4
python3 -m pip install numpy h5py torch
python3 -m compileall .
python3 -m unittest discover -s radar/tests -p 'test_*.py'
```

Do not continue if the source checks fail.

## Step 1: Download the Real Dataset

Download Radar Ghost Dataset **v1.1** `original.zip` from the fixed official
Zenodo record. Version 1.0 has a radar/LiDAR time synchronization problem and
should not be used. The downloader resumes interrupted transfers, verifies the
published 5,818,814,597-byte archive using its MD5 checksum, safely extracts
it, and prints the directory to use as `--input`.

```bash
python3 download_radar_ghost_dataset.py
find data/radar_ghost_v1_1 -type f -name '*.h5' | head
```

The archive is about 5.4 GiB before extraction. Rerun the same command after a
network interruption to resume. Use `--delete-archive` only if disk space is
tight and the completed ZIP is no longer needed.

Use the directory directly above `train/`, `val/`, and `test/` as `--input`.
Background (`0`), ignore (`-1`), noise (`-2`), and sketchy labels are not used
as clean supervised negatives.

## Step 2: Prepare Real Sequences

Create the paper-compatible official split:

```bash
python3 prepare_radar_ghost_dataset.py \
  --input data/radar_ghost_v1_1/original \
  --output artifacts/ghost_real_official \
  --split-mode official
```

Also create a stricter scenario-disjoint split for the main research claim:

```bash
python3 prepare_radar_ghost_dataset.py \
  --input data/radar_ghost_v1_1/original \
  --output artifacts/ghost_real_scenario \
  --split-mode scenario_grouped
```

Open each `manifest.json`. Confirm that train, validation, and test contain
sequences and both `real_points` and `ghost_points`. Use the official split for
published comparisons and the scenario split for the generalization claim.

## Step 3: Train Real-Data Baselines

Independent point MLP:

```bash
python3 train_radar_ghost_detector.py \
  --data artifacts/ghost_real_official \
  --output artifacts/ghost_point_mlp \
  --model point_mlp \
  --window-frames 1 \
  --max-points 1024 \
  --epochs 40 \
  --batch-size 32
```

Temporal PointNet:

```bash
python3 train_radar_ghost_detector.py \
  --data artifacts/ghost_real_official \
  --output artifacts/ghost_temporal_real \
  --model temporal_pointnet \
  --window-frames 5 \
  --max-points 1024 \
  --hidden-dim 128 \
  --context-dim 192 \
  --epochs 60 \
  --batch-size 16
```

If GPU memory is insufficient, reduce `--batch-size`, not `--max-points`, so
the data definition remains comparable.

## Step 4: Evaluate Real Test Data

```bash
python3 evaluate_radar_ghost_detector.py \
  --data artifacts/ghost_real_official \
  --checkpoint artifacts/ghost_point_mlp/best_detector.pt \
  --split test \
  --output artifacts/ghost_point_mlp/test_metrics.json

python3 evaluate_radar_ghost_detector.py \
  --data artifacts/ghost_real_official \
  --checkpoint artifacts/ghost_temporal_real/best_detector.pt \
  --split test \
  --output artifacts/ghost_temporal_real/test_metrics.json
```

Repeat training/evaluation with `ghost_real_scenario`. Never select a threshold
on test data. The checkpoint stores the highest-recall validation threshold
whose real-point false-positive rate is at most 1% (`--max-real-fpr`). Report
AUPRC, AUROC, real false-positive rate, ghost recall, recall by bounce family,
and per-scenario results.

## Step 5: Validate CARLA Geometry

Start CARLA in another terminal:

```bash
cd /path/to/CARLA_0.9.16
./CarlaUE4.sh -quality-level=Epic
```

Validate the new profile without a learned filter:

```bash
cd /path/to/carla-claude/carla4
python3 validate_radar_accuracy.py \
  --radar-profile geometry_multipath_v1 \
  --duration-s 30 \
  --output radar_validation_geometry_v1
```

Inspect `metadata.json`, `summary.json`, and `radar_details.jsonl`. Reflector
and multipath counts should be nonzero near walls, buildings, fences, or
guardrails. If always zero, first increase semantic-LiDAR density or use
Town03/Town04; do not lower geometry-quality gates blindly.

## Step 6: Collect Synthetic Path Labels

First make a short smoke artifact:

```bash
python3 collect_carla_radar_ghosts.py \
  --town Town04 \
  --output artifacts/carla_ghost_smoke \
  --split train \
  --sequences 1 \
  --duration 10 \
  --vehicles 30 \
  --walkers 15 \
  --radar-timeout 30
```

The semantic-LiDAR callback also fits reflectors and generates multipath, so
it can be slower than simulation time. If a frame times out with no callback
error, increase `--radar-timeout`; do not lower the radar density merely to
hide a processing-latency problem.

Inspect printed real/ghost counts. Use a new folder for the full run:

```bash
for town in Town03 Town04; do
  python3 collect_carla_radar_ghosts.py --town "$town" \
    --output artifacts/carla_ghost_full --split train \
    --sequences 30 --duration 45 --vehicles 45 --walkers 25 --seed 100
  python3 collect_carla_radar_ghosts.py --town "$town" \
    --output artifacts/carla_ghost_full --split val \
    --sequences 8 --duration 45 --vehicles 45 --walkers 25 --seed 2000
  python3 collect_carla_radar_ghosts.py --town "$town" \
    --output artifacts/carla_ghost_full --split test \
    --sequences 8 --duration 45 --vehicles 45 --walkers 25 --seed 4000
done
```

Prepare the CARLA H5 files through the identical pipeline:

```bash
python3 prepare_radar_ghost_dataset.py \
  --input artifacts/carla_ghost_full \
  --output artifacts/ghost_carla_prepared \
  --split-mode official
```

Direct road users are labeled real, deterministic reflected paths receive CMTO
bounce labels, and static returns remain unlabeled context. Random synthetic
clutter is noise and is outside the multipath supervision objective.

## Step 7: Synthetic Pretraining and Real Fine-Tuning

```bash
python3 train_radar_ghost_detector.py \
  --data artifacts/ghost_carla_prepared \
  --output artifacts/ghost_temporal_carla_pretrain \
  --model temporal_pointnet \
  --window-frames 5 \
  --max-points 1024 \
  --hidden-dim 128 \
  --context-dim 192 \
  --epochs 50 \
  --batch-size 16

python3 evaluate_radar_ghost_detector.py \
  --data artifacts/ghost_real_official \
  --checkpoint artifacts/ghost_temporal_carla_pretrain/best_detector.pt \
  --split test \
  --output artifacts/ghost_temporal_carla_pretrain/real_zero_shot.json

python3 train_radar_ghost_detector.py \
  --data artifacts/ghost_real_official \
  --output artifacts/ghost_temporal_sim2real \
  --model temporal_pointnet \
  --window-frames 5 \
  --max-points 1024 \
  --hidden-dim 128 \
  --context-dim 192 \
  --pretrained artifacts/ghost_temporal_carla_pretrain/best_detector.pt \
  --epochs 30 \
  --batch-size 16 \
  --learning-rate 0.0003

python3 evaluate_radar_ghost_detector.py \
  --data artifacts/ghost_real_official \
  --checkpoint artifacts/ghost_temporal_sim2real/best_detector.pt \
  --split test \
  --output artifacts/ghost_temporal_sim2real/test_metrics.json
```

Deploy `ghost_temporal_sim2real/best_detector.pt` only if it beats the
real-only and synthetic-only baselines on held-out real data.

## Step 8: Recollect and Retrain the Controller

Filtering changes the radar distribution. Do not reuse an old target-speed
dataset, scaler, or controller model.

```bash
python3 collect_throttle_brake_data.py \
  --radar-backend realistic \
  --radar-profile geometry_multipath_v1 \
  --radar-ghost-detector artifacts/ghost_temporal_sim2real/best_detector.pt \
  --radar-ghost-device cpu \
  --output dataset_throttle_brake_geometry_filtered

python3 collect_scenario_data.py \
  --radar-backend realistic \
  --radar-profile geometry_multipath_v1 \
  --radar-ghost-detector artifacts/ghost_temporal_sim2real/best_detector.pt \
  --radar-ghost-device cpu \
  --output dataset_throttle_brake_geometry_filtered

python3 train_throttle_brake.py \
  --data dataset_throttle_brake_geometry_filtered \
  --config dataset_throttle_brake_geometry_filtered/dataset_config.json \
  --output model_throttle_brake_geometry_filtered
```

The detector hash and validation threshold are embedded in dataset/model
metadata. Runtime refuses a different detector or threshold.

## Step 9: Final Closed-Loop Runs

```bash
python3 test_throttle_brake_live.py \
  --radar-backend realistic \
  --radar-profile geometry_multipath_v1 \
  --radar-ghost-detector artifacts/ghost_temporal_sim2real/best_detector.pt \
  --radar-ghost-device cpu \
  --model model_throttle_brake_geometry_filtered/target_speed_mlp.pt \
  --scaler model_throttle_brake_geometry_filtered/scaler.pkl \
  --config model_throttle_brake_geometry_filtered/model_config.json

cd scenarios
python3 run_all.py \
  --driver mlp \
  --model-dir ../model_throttle_brake_geometry_filtered \
  --radar-backend realistic \
  --radar-profile geometry_multipath_v1 \
  --radar-ghost-detector ../artifacts/ghost_temporal_sim2real/best_detector.pt \
  --radar-ghost-device cpu \
  --scenarios 1 2 3 4 \
  --fog 0 50 100 \
  --seeds 42 43 44
```

Run three separately collected/trained controller ablations: geometry without
filtering, geometry plus point MLP, and geometry plus temporal sim-to-real.
Compare selected-ghost rate, false rejection of direct targets, minimum
distance, TTC, collisions, and intervention. A detector-only gain is not enough
if closed-loop safety worsens.

## Interpretation Boundary

Implemented physics includes planar surface fitting, mirror geometry,
type-1/type-2 second-order and type-2 third-order paths, material/bounce loss,
path-dependent range/azimuth/Doppler, C-Shenron-derived return strength, sensor
errors, latency, and tracking. This remains a target-list simulator. Raw
chirps, antenna arrays, phase, FFTs, CFAR, micro-Doppler, polarization, and
radome effects are outside this implementation.

The official dataset and label definition are at
<https://github.com/flkraus/ghosts>. Keep datasets and model artifacts out of
Git.

## Research Grounding

- [C-Shenron](https://ucsdwcsng.github.io/c-shenron/) motivates the
  CARLA-native material/scattering front end, but its full ADC-cube pipeline is
  not copied into this target-list controller.
- [RadaRays](https://kbs.informatik.uos.de/files/pdfs/ral2025_amock_radarays.pdf)
  motivates explicit multi-bounce geometry rather than random ghost offsets.
- [Radar Ghost Dataset](https://arxiv.org/abs/2404.01437) supplies the real CMTO
  labels and final evaluation domain.
- [Fast Rule-Based Clutter Detection](https://arxiv.org/abs/2108.12224)
  motivates specular guardrail/wall path families and physics-based ablations.
- [Anomaly Detection in Radar Data Using PointNets](https://arxiv.org/abs/2109.09401)
  motivates a point-set detector baseline.
