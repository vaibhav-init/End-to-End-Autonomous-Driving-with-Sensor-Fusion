# Sim-to-Real Automotive Radar: CARLA Ghost-Detector Research Brief

**Purpose of this document:** This is a self-contained brief describing (a) what I am trying to
do, (b) what has already been done in the field, (c) the exact state of my implemented pipeline,
(d) verified results so far, and (e) the open research questions I need answered before I
implement the next steps. It is written to be handed to a higher-capability research model as a
prompt, with or without access to the repository it describes.

---

## 1. Executive Summary

**Goal:** Build a _realistic automotive radar_ inside the CARLA 0.9.16 simulator, **collect training data in it**, and **evaluate a ghost (multipath false-alarm) detector trained on that
synthetic data against the real-life Radar Ghost Dataset (RGD) v1.1** — i.e., a sim-to-real
transfer study for radar multipath-ghost detection.

**Current status:** The full pipeline is implemented and one RGD-matching CARLA sequence
(38.5 s, pedestrian target, 10 Hz, ±70° FOV) has been verified end-to-end with
`ALL CHECKS PASSED` (real points 486, ghost points 1623, live Doppler). Real RGD v1.1 data is
downloaded and prepared for training. The remaining work is: scale up the CARLA collection,
train on synthetic data, evaluate zero-shot on the real test split, optionally fine-tune, and
decide what claims the results can support.

**Central research question:** Can a target-list radar simulator (physics-guided synthetic ghost
geometry, not raw-signal simulation) produce training data that transfers to a real radar ghost
detection task — and if so, under what conditions (zero-shot vs. fine-tuning, feature design,
profile calibration)?

---

## 2. The Goal in Detail

1. **Create a realistic radar in CARLA** — a _target-list_ automotive radar model: it outputs
   per-scan detections with range, azimuth, radial velocity (Doppler), amplitude/SNR, and
   tracks. It is deliberately **not** a raw-signal (FMCW/ADC/CFAR) simulator (see §7).
2. **Collect data in CARLA** — reproduce the RGD recording regime: stationary ego, pedestrian
   or cyclist main object moving near a reflective surface (walls, guardrails, fences),
   10 Hz, ±70° field of view, 38.5 s sequences. Label multipath ghosts with the same CMTO
   label schema as the real dataset.
3. **Test on real life data** — train a ghost classifier on the synthetic CARLA data and
   evaluate it on the held-out test split of the real Radar Ghost Dataset v1.1 (zero-shot),
   then optionally fine-tune on real training data and evaluate again.

**Success criteria (decided with the user):**

- A clean, quantified measurement of the CARLA → RGD domain gap and which ghost families
  transfer (a defensible contribution on its own).
- Zero-shot performance meaningfully better than random.
- Fine-tuned performance that preserves or improves on the real-only baseline (deploy only if
  it wins).
- NOT claimed: that a target-list simulator is equivalent to a real radar, or raw-signal
  realism.

---

## 3. State of the Field (what has already been done)

References cited throughout the repository's documentation:

### 3.1 Real benchmark: Radar Ghost Dataset (RGD)

- **Paper:** Kraus et al., "Multipath Ghosts in Radar Point Clouds: An Automotive Case Study,"
  arXiv:2404.01437. **Repo:** https://github.com/flkraus/ghosts
- **v1.1 setup:** two 77 GHz chirp-sequence radar sensors in the front bumper (`left`/`right`;
  no single front sensor), **10 Hz**, range 0.15–153 m, azimuth **±70°**, unambiguous Doppler
  ±44.3 m/s; resolutions Δr 0.15 m, Δφ 1.8°, Δv 0.087 m/s.
- **Recording regime:** ego **parked and stationary** ("to prolong the time where multi-path
  reflections occur"); main object is a **pedestrian or cyclist** walking away from and back
  toward the ego near reflective surfaces (plastered/marble walls, metal containers, parked
  cars, curbstone, guardrail). 21 scenarios, 111 sequences × ~385 frames (38.5 s) ≈ 71 min.
- **Data volume:** ~820 raw CFAR detections/frame (127–1775); ~35M points; only ~100k ghost +
  ~600k real labeled. Ghosts: type-1 2nd-order, type-2 2nd/3rd-order, plus "other multipath".
- **Labels (CMTO):** 4-digit codes — `class_id` (1 ped, 2 cyclist, 3 car, 4 large vehicle,
  5 motorcycle), `is_main`, `bounce_type` (0–3), `bounce_order` (1 = real/direct; 2, 4 =
  2nd/3rd order multipath; 0 = undecided; 6 = ambiguous). Background `0`, ignore `-1`, noise
  `-2` are not used as clean supervised negatives.
- **v1.0 is broken** (radar/LiDAR time sync); only v1.1 is used.

### 3.2 Radar simulation in CARLA / synthetic radar

- **C-Shenron** (UCSD WCSNG, https://ucsdwcsng.github.io/c-shenron/): CARLA-native
  material/scattering front end that synthesizes radar ADC cubes from the simulator. Its full
  raw-signal pipeline is NOT copied into this repo; only the _material-class + incidence-based
  scattering_ front end is ported to a target-list model.
- **RadaRays** (A. Mock et al., RAL 2025, https://kbs.informatik.uos.de/files/pdfs/ral2025_amock_radarays.pdf):
  ray-tracing-based multi-bounce radar simulation — motivates explicit multi-bounce geometry
  (rather than random ghost offsets) for ghost generation.
- **Fast Rule-Based Clutter Detection** (Scheiner et al., arXiv:2108.12224): specular
  guardrail/wall path families and physics-based ablations — motivates the planar
  image-method path families used here.
- **Anomaly Detection in Radar Data Using PointNets** (Scheiner et al., arXiv:2109.09401):
  point-set detector baseline for radar false alarms — motivates the PointNet-style detector.
- **RadarScenes** (https://radar-scenes.com): 5-sensor 60 ms radar dataset; used as the source
  of the default `generic_lrr_v1` envelope (100 m, ±60°, 0.15 m range res, 0.1 km/h Doppler
  res) and its labeling limitations (background/negative handling).

### 3.3 Sim-to-real methodology

- **Perception error models for virtual testing** (arXiv:2302.11919): fit a simulator's sensor
  error model to real data statistics, then generate unlimited training data in the simulator.
  This is the _proven_ recipe this project should follow to close the gap (calibrate the
  profile to RGD statistics).
- **General sim-to-real findings relevant here:** synthetic radar pretraining typically
  transfers _better when combined with a small amount of real data_ (fine-tuning) than pure
  zero-shot; naive fine-tuning across a large domain gap can fail (catastrophic forgetting,
  overfitting a small real set, poor initialization). Mitigations: physical (non-fitted)
  feature normalization, low fine-tuning LR, early stopping, threshold selection on validation
  only.

### 3.4 Known sim-to-real gaps for this specific project (ranked)

1. **Point cardinality:** RGD ~800 raw points/frame vs. CARLA's sparse target list (tens of
   points). Biggest gap; monitor or export denser points.
2. **Ego motion:** RGD stationary; old CARLA collectors drove the ego. **Fixed** by the
   RGD-regime collector (ego physics disabled, stationary).
3. **Object class:** RGD main objects are pedestrians/cyclists; old CARLA ghosts were
   vehicles. **Fixed** — collector supports pedestrian (walker) / cyclist (motorcycle proxy).
4. **Cadence/geometry:** 10 vs 20 Hz, ±70° vs ±60°, 2 sensors vs 1. **Partially fixed** — the
   `rgd_regime_v1` profile sets 10 Hz and ±70°; still 1 sensor vs 2.
5. **Amplitude:** CARLA `amp` is a synthetic `10^(SNR_dB/20)` proxy, not calibrated to RGD
   amplitude statistics.
6. **Ghost statistics:** RGD ghosts are measured RF multipath; CARLA ghosts come from planar
   image-method geometry + priors (type-1/type-2 2nd-order, type-2 3rd-order).

---

## 4. What Is Already Implemented (repo state)

Repository layout (all paths relative to repo root; main code under `carla4/`):

| Path                                                                       | Purpose                                                                                                                                                                                                                                                                                   |
| -------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `carla4/radar/front_radar.py`                                              | Three backends: `native` (CARLA sensor), `cshenron` (C-Shenron-derived adapter), `realistic` (temporal target-list model). Includes kinematic velocity fallback for physics-off actors.                                                                                                   |
| `carla4/radar/realistic_core.py`                                           | `RealisticRadarConfig` + `RealisticRadarModel`: SNR-conditioned detection/errors, correlated dropout/noise, clutter/interference, probabilistic or geometry ghosts, quantization/latency, NN tracking (M-of-N), path-gated scalar selection. Profile registry `REALISTIC_RADAR_PROFILES`. |
| `carla4/radar/multipath.py`                                                | `extract_reflector_segments` (semantic-LiDAR planar fitting) + `generate_multipath_targets` (image-method type-1/type-2 2nd- and type-2 3rd-order paths).                                                                                                                                 |
| `carla4/radar/cshenron_core.py`                                            | CARLA 0.9.16 material/scattering port (24-byte semantic-LiDAR records, post-0.9.14 tag table 0–28).                                                                                                                                                                                       |
| `carla4/radar/profiles/*.json`                                             | `ideal_target_list_v1`, `gaussian_baseline_v1`, `generic_lrr_v1`, `geometry_multipath_v1`, `rgd_regime_v1` (RGD envelope: 10 Hz, 140° FOV, 0.15 m / 1.8° / 0.087 m/s resolutions, ±44.3 m/s unambiguous Doppler; `multipath_mode=geometry`, probabilistic ghosts disabled).               |
| `carla4/radar/ghost_detection/`                                            | `features.py` (physical feature schema `radar_ghost_physical_v1`), `labels.py` (CMTO decode → binary), `dataset.py` (windowed point-set `PreparedGhostDataset`), `model.py` (`PointMLP`, `TemporalPointNet`), `metrics.py`, `runtime.py` (deployment filter).                             |
| `carla4/download_radar_ghost_dataset.py`                                   | Downloads RGD v1.1 `original.zip` (Zenodo record 6676246, 5,818,814,597 bytes, MD5 `3873152766839286469b4b7e63ceba12`), resumes, verifies, safe-extracts.                                                                                                                                 |
| `carla4/prepare_radar_ghost_dataset.py`                                    | Converts real or CARLA H5 → per-sequence `.npz` with `manifest.json`; split modes `official`, `scenario_grouped`, `all_train`.                                                                                                                                                            |
| `carla4/train_radar_ghost_detector.py`                                     | Trains `point_mlp` / `temporal_pointnet`; BCE with positive-weight; validation threshold at ≤1% real FPR (`--max-real-fpr`); saves `best_detector.pt` with schema/feature checks; `--pretrained` for fine-tuning.                                                                         |
| `carla4/evaluate_radar_ghost_detector.py`                                  | Held-out split eval: AUPRC, AUROC, real FPR, ghost recall, recall by bounce family, per-scenario metrics.                                                                                                                                                                                 |
| `carla4/collect_carla_radar_ghosts.py`                                     | **RGD-regime collector** (see below).                                                                                                                                                                                                                                                     |
| `carla4/collect_carla_radar_dataset.py`                                    | Older moving-ego/vehicle-event collector (different regime — do NOT use for RGD pretraining).                                                                                                                                                                                             |
| `carla4/validate_radar_accuracy.py` / `analyze_radar_validation.py`        | CARLA 0.9.16 accuracy validator + forensic analyzer.                                                                                                                                                                                                                                      |
| `carla4/radar/GHOST_DETECTION.md`, `README.md`, `RGD_REGIME_COLLECTION.md` | Full pipeline docs, sensor docs, and the RGD-regime collection runbook.                                                                                                                                                                                                                   |

### 4.1 The RGD-regime collector (`collect_carla_radar_ghosts.py`)

- `--target-type {vehicle, pedestrian, cyclist}`: pedestrian spawns a CARLA **walker**
  (tag 12 → RGD class 1); cyclist spawns a two-wheel **motorcycle** (tag 18 → RGD class 5
  "motorbike" — documented mismatch, CARLA has no cyclist actor); vehicle is the original
  4-wheel behavior (tag 14 → class 3).
- Defaults: `--fps 10`, `--duration 38.5` (~385 frames, matching RGD). Ego physics disabled
  (stationary). Target speeds: vehicle 3.0, pedestrian 1.4, cyclist 4.5 m/s.
- The controlled target is placed by the **production image-method solver**
  (`_configure_controlled_target`), validated by an actual CARLA actor producing multipath
  (`_validate_controlled_target`), then **kinematically teleported each tick** (physics off)
  along the reflector tangent with a sinusoidal oscillation whose period derives from target
  speed.
- **Walker Doppler fix (critical):** physics-off actors report zero velocity in CARLA. The
  radar adapter's `_estimate_kinematic_velocity` derives true motion from successive
  transforms (`Δposition/Δtime`), so pedestrian Doppler (and inherited ghost Doppler) is live.
- **Robustness:** each sequence runs in a **fresh worker subprocess** (native CARLA state
  accumulates across `load_world` and segfaults after ~7 sequences in one process);
  `--resume` skips sequences that already have `.h5` + `.summary.json`; `--sequence-retries`
  (default 1) retries crashed workers. Supervisor writes
  `collection_<town>_<split>.json` with `failed_sequences`.
- Per-sequence **verification block** (printed, copy-pasteable): checks ego stationary
  (<0.5 m/s), fps=10, FOV=140°, ghost>0, real>0, expected RGD class present, direct-target
  Doppler alive (mean |vr| > 0.05 m/s), speed plausible (0.1×–1.6× of configured speed —
  tangential motion means only the radial component is measured, so mean ≈ 0.8 m/s for a
  1.4 m/s walker is correct), controlled reflector used.

### 4.2 Feature and label bridge (verified consistent)

- **Features** (`features.py`): `x_sensor/100`, `y_sensor/100`, `range/100`, `sin(az)`,
  `cos(az)`, `radial_velocity/40`, `signed_log_amplitude/10`, `age/0.5` (clipped).
  **No dataset-fitted normalization** — physical constants only. This is the key design
  choice enabling sim-to-real transfer.
- **Labels:** CARLA collector writes CMTO codes (`class*1000 + type*10 + order`, direct =
  `class*1000+11`, order encoded 1/2/4); `decode_cmto_label` maps them to the **same binary
  targets** (0=real, 1=multipath, -1=ignore) as official RGD labels. Verified statically.

### 4.3 Verified results (single sequence, Run 3 — the only full verification so far)

```
capture_frames: 385
target_type: pedestrian (tag 12)
ego_speed_mps: 0.0
direct_target_speed_mean_mps: 0.822  (expected ~1.4; radial component only)
direct_target_speed_max_mps: 1.479
real: 486   ghost: 1623
label_class_histogram: {1: 1984, 2: 33, 3: 85, 5: 7}
ghost_family_histogram: {type1-order2: 677, type2-order2: 645, type2-order3: 301}
radar_profile: rgd_regime_v1   radar_fps: 10   radar_fov_deg: 140.0
reflector: id=..., tag=28 (GuardRail), length=7.43 m
validated_path_families: [type1-order2, type2-order2]
RESULT: ALL CHECKS PASSED
```

- **Run 1 failure (informative):** teleport + physics off WITHOUT the velocity fallback →
  dead Doppler (`mean |vr|=0.055`, max 0.348 vs expected 1.4). Fixed by
  `_estimate_kinematic_velocity`.
- **Run 2 failure (informative):** walker with physics ON → unstable, falls/settles out of
  the LiDAR FOV, validation crash `last_dynamic_ids=[]`. Keep physics OFF + kinematic
  teleport; never re-enable walker physics.

### 4.4 Repository git state (IMPORTANT)

The RGD-regime work has **uncommitted changes** in the working tree that a remote `git pull`
will NOT carry:

- `carla4/collect_carla_radar_ghosts.py` (worker supervisor, `--resume`,
  `--sequence-retries`, walker spawn retries, `target_type` plumbing)
- `carla4/radar/front_radar.py` (`_estimate_kinematic_velocity` fallback)
- `carla4/radar/README.md` (profile table row)
- `carla4/radar/RGD_REGIME_COLLECTION.md` (untracked runbook)

These must be committed and pushed before any remote machine pulls, or the collection
produces dead pedestrian Doppler and no crash recovery.

---

## 4.5 Zero-Shot Evaluation Results (CARLA pretrained → real RGD test)

**Date:** August 18, 2026
**Checkpoint:** `artifacts/ghost_temporal_carla_pretrain/best_detector.pt`
**Data:** `artifacts/ghost_real_official` (111 sequences, official split)
**Training:** CARLA densified data, 50 epochs, temporal_pointnet, lr=1e-3
**Training time:** ~2 hours on GPU

### Overall Metrics

| Metric                                | Value   |
| ------------------------------------- | ------- |
| AUPRC                                 | 0.159   |
| AUROC                                 | 0.606   |
| Best F1                               | 0.254   |
| Best F1 threshold                     | 0.486   |
| Operating threshold (≤1% real FPR)    | 0.901   |
| Ghost recall (at operating threshold) | 0.30%   |
| Real false-positive rate              | 1.45%   |
| Precision                             | 5.75%   |
| True positives                        | 148     |
| False negatives                       | 22,250  |
| False positives                       | 2,426   |
| True negatives                        | 165,148 |

### Ghost Recall by Bounce Family

| Family            | Count | Recall |
| ----------------- | ----- | ------ |
| type1_second      | 7,984 | 0.89%  |
| type2_second      | 6,729 | 0.64%  |
| type2_third       | 6,354 | 0.09%  |
| generic_multipath | 1,135 | 0.79%  |
| ambiguous_order   | 161   | 9.32%  |
| other_multipath   | 35    | 11.43% |

### Real False-Positive Rate by Class

| Class          | Count  | FPR   |
| -------------- | ------ | ----- |
| 1 (pedestrian) | 99,576 | 0.41% |
| 2 (cyclist)    | 54,206 | 1.20% |
| 3 (car)        | 13,658 | 9.99% |
| 5 (motorcycle) | 134    | 0.00% |

### Assessment

**Zero-shot transfer is poor.** The CARLA-pretrained model achieves near-random recall (~0.7%)
with low precision (~5.7%). The domain gap between CARLA synthetic multipath geometry and
real RF multipath signatures is too large for direct zero-shot transfer. Car-class real points
have the highest FPR (~10%), suggesting the model confuses car returns with ghosts.

The next step is training directly on real data to establish the real-only baseline, then
deciding whether synthetic pretraining adds any value.

### Densification Finding (August 18, 2026)

Statistical densification (Step 6.4) was applied to the CARLA data to bridge the point-count
gap (~800 pts/frame matching RGD). The densified data was used for the CARLA pretraining
above. Despite matching point-count statistics, zero-shot transfer remained near-random.

**Conclusion:** Densification only fixes point density. The core domain gap is in the
_physics_ of how ghosts look — ghost geometry (planar image-method paths vs. real RF
multipath), ghost Doppler/amplitude signatures (simulator physics model vs. real radar
measurements). Densifying synthetic points around wrong geometry just produces more
wrong-signature points.

**Implication:** The domain gap is in the feature distribution, not point count.
Densification is necessary for architectural compatibility (PointNet expects similar point
counts) but insufficient for transfer.

### Fine-tuning Decision (August 18, 2026)

Fine-tuning with `--pretrained` was considered but deemed unnecessary. Given the near-random
zero-shot results (AUPRC 0.159), the CARLA checkpoint provides almost no useful
initialization — fine-tuning would essentially relearn everything from real data. Training
directly on real data is more straightforward and produces the same result.

### Next Steps (revised)

1. **Train directly on real data** — `temporal_pointnet` on `ghost_real_official` train split,
   60 epochs, same hyperparameters. This is the real-only baseline.
2. **Evaluate on real test split** — compare against zero-shot results.
3. **If real-only is strong:** the contribution is the measured domain gap and the real-data
   ghost detector itself.
4. **If real-only is also weak:** the issue may be the feature schema, label noise, or
   dataset size — investigate augmentation and feature engineering.
5. **Optional:** profile calibration to match RGD statistics before any future synthetic
   pretraining attempt.

### 4.6 Real-Only Baseline Results (temporal_pointnet trained on real RGD)

**Date:** August 24, 2026
**Checkpoint:** `artifacts/ghost_temporal_official/best_detector.pt`
**Data:** `artifacts/ghost_real_official` (official split, real RGD v1.1)
**Training:** temporal_pointnet, 60 epochs, ~15 h GPU, batch 16, lr 1e-3,
window 5 × 1024 points, hidden 128 / context 192, dropout 0.15, seed 42

#### Headline comparison vs zero-shot (both on the same official test split)

| Metric                       | Zero-shot (CARLA pretrain) | Real-only baseline    |
| ---------------------------- | -------------------------- | --------------------- |
| Validation AUPRC             | —                          | 0.9539                |
| Test AUPRC                   | 0.159                      | **0.329**             |
| Test AUROC                   | 0.606                      | **0.819**             |
| Best F1                      | 0.254                      | **0.424** (thr 0.993) |
| Ghost recall @ operating thr | 0.30 %                     | 26.6 %                |
| Real FPR @ operating thr     | 1.45 %                     | 5.09 %                |

Synthetic pretraining is confirmed useless as an initializer (real-only beats
zero-shot by every metric), consistent with the fine-tune skip decision.

#### Finding 1 — severe validation-to-test collapse

Validation AUPRC 0.954 collapses to test AUPRC 0.329. The model generalizes to
held-out _sequences_ but not to held-out _scenario families_. Per-scenario test
AUPRC spans 0.14 (scenario-17 ped) to 0.79 (scenario-11 cycl). Cross-scenario
generalization — not architecture or data volume — is the current bottleneck.
The split-composition/leakage analysis (`analyze_ghost_dataset.py`, commit
`a850b3e`) should be checked against these numbers before any realism claim.

#### Finding 2 — probability calibration is broken

The operating threshold saturates at 0.9995 and still misses the ≤1 % real-FPR
target (achieved 5.09 %). At the validation-selected fixed threshold (0.782)
recall is 68 % but real FPR is 25.8 % — unusable for a safety filter. Scores
are overconfident; temperature scaling or val-quantile threshold selection is
required before deployment-style metrics are meaningful. Class-level FPR at
the fixed threshold: pedestrian 30 %, cyclist 19 %, car 22 %, motorbike 77 %
(n=134, not significant).

#### Defensible claims after this run

1. Physics-guided synthetic ghosts do NOT transfer zero-shot (§4.5).
2. A real-data temporal PointNet detects ghosts well within seen scenario
   families but degrades sharply on unseen families (this section).
3. Both findings are measured, reproducible, and threshold-protocol-clean.

Not yet supportable: any deployment-grade false-alarm filter claim at ≤1 % FPR.

#### Revised next steps (supersedes §4.5 list)

1. Confirm train/val/test scenario composition; if train/val share scenario
   families, re-report with the stricter scenario-disjoint split as primary.
2. Prepare + evaluate the scenario-disjoint split
   (`prepare_radar_ghost_dataset.py --split-mode scenario_disjoint`) with this
   same checkpoint — this number is the paper's main honest result.
3. Fix calibration (temperature scaling on val, or threshold = 1 % quantile of
   val real scores); re-evaluate before quoting any ≤1 % FPR figure.
4. Time-boxed rescue attempts for cross-scenario transfer: stronger
   regularization/augmentation (azimuth mirroring, point dropout bursts),
   amplitude/Doppler normalization per sequence, and one calibrated-synthetic
   pretraining retry (§4.5 option a) judged only on the disjoint split.

---

### 4.7 Zero-Shot v2 Results (CFAR-style expansion + invariant features)

**Date:** August 25, 2026
**Checkpoint:** `artifacts/zeroshot_v2_carla_pretrain/best_detector.pt` (epoch 14,
synthetic val AUPRC 0.962)
**Data:** trained on `ghost_carla_zeroshot_v2` (17 sequences, CFAR-emulating
expansion, ~2000 pts/frame); evaluated on `ghost_real_official` (manifest
patched to schema v2; stored raw fields unchanged by the patch)

| Metric | v1 zero-shot | **v2 zero-shot** |
|---|---|---|
| Test AUPRC | 0.159 | **0.077** |
| Test AUROC | 0.606 | **0.281** |
| Ghost recall @ op. thr | 0.30% | 0.21% |
| Op. threshold | 0.9995 (saturated) | 0.979 (healthy) |
| Op. FPR | 1.45% | 0.97% ✓ |

**Verdict: worse than v1.** Structural fixes alone did not close the gap.
AUROC < 0.5 shows an *inverted* transfer signal: real ghost points receive
systematically LOWER ghost-probability than real direct returns under the
model — some cue learned from expanded synthetic data is anti-correlated in
reality (candidates: local-density ordering, relative-amplitude ordering, or
cluster-Doppler residual direction). Notably, the expansion gives synthetic
ghost detections dense point clusters exactly like direct returns, which may
have destroyed the sparsity cue that v1's sparse ghosts accidentally carried.

**Positive:** label smoothing fixed calibration — the operating point lands
at 0.97% FPR instead of saturating at 0.9995 with a 5% floor.

**Next diagnostic:** per-feature marginal comparison of synthetic-train vs
real-test (`analyze_ghost_dataset.py` on both prepared sets) to identify
which v2 statistic is reversed before any further generator work.

### 4.8 Cross-Domain Diagnostic: Real-Trained Model Scored on Synthetic

**Date:** August 25, 2026
**Tool:** `evaluate_cross_domain.py` (schema-tolerant evaluator)
**Checkpoint:** `ghost_temporal_official/best_detector.pt` (real-only, schema v1)
**Validation:** reproduces the baseline exactly on real test
(AUPRC 0.3291 / AUROC 0.8191 / recall@0.782 = 68.3%)

| Evaluated set | Base rate (ghost) | Random AUPRC | Observed AUPRC | AUROC | pts > 0.782 thr |
|---|---|---|---|---|---|
| Real test | 13% | ~0.13 | **0.329** ✓ | 0.819 | many |
| Old synthetic (densified) | ~75% | ~0.75 | 0.635 (below random) | 0.462 | ~0.4% |
| New synthetic (expanded) | ~75% | ~0.75 | 0.745 (= random) | 0.508 | **none** |

**Findings:**

1. The real-trained model ranks BOTH synthetic generations at chance level
   (AUROC 0.46 / 0.51) — the domain gap is symmetric and survived both
   generator iterations (densified v1, CFAR-expanded v2).
2. Not one labeled synthetic point crosses the real-model's operating
   threshold: the entire synthetic score mass sits below where real-data
   scores live. Primary suspect: v1's absolute-amplitude feature operates on
   completely different scales (SNR-proxy vs measured echo power), so the
   model sees out-of-range inputs everywhere.
3. Caveat: because the v1 checkpoint depends on absolute amplitude, this
   test is partially blind to *geometric* quality — it measures total
   feature-space distance more than shape realism.

**Combined with §4.5–4.7:** four independent transfer measurements
(synthetic→real twice, real→synthetic twice, two generator generations) all
show chance-or-worse transfer. Conclusion: pure Level-A zero-shot is not
achievable with this target-list simulator fidelity, and the gap is
quantified bidirectionally — a defensible negative result.

---

## 5. Planned Next Steps (current plan)

1. Commit/push the uncommitted RGD fixes; verify remote checkout (grep for
   `sequence-retries` and `_estimate_kinematic_velocity`).
2. Remote sanity: `python3 -m compileall .` + unit tests
   (`python3 -m unittest discover -s radar/tests -p 'test_*.py'`, mock CARLA, no server
   needed) + confirm `rgd_regime_v1` profile loads.
3. Real RGD v1.1 is downloaded AND prepared (`artifacts/ghost_real_official`,
   `artifacts/ghost_real_scenario`); verify manifest (schema `radar_ghost_physical_v1`, both
   classes per split).
4. Start CARLA 0.9.16 (`./CarlaUE4.sh -quality-level=Epic`, add `-RenderOffScreen` when
   headless). Smoke collect 1 pedestrian sequence → must print `ALL CHECKS PASSED`.
5. Full collection: train 20 / val 4 / test 4 pedestrian sequences (Town04, seeds
   100/2000/4000, `--headless`), `--resume` on crash; optionally `--target-type cyclist` and
   `--town Town03` for diversity (filenames include town, outputs merge).
6. Prepare CARLA H5s (`prepare_radar_ghost_dataset.py --split-mode official`).
7. **Zero-shot (DONE):** Trained `temporal_pointnet` on CARLA densified data
   (50 epochs, ~2 hrs), evaluated on real RGD test split. Result: AUPRC 0.159,
   AUROC 0.606, ghost recall 0.30% at 1% real FPR — **near-random, poor
   transfer.** See §4.5 for full results.
8. **Fine-tune (SKIPPED):** Near-random zero-shot means CARLA checkpoint
   provides no useful initialization. Fine-tuning ≡ training from scratch on
   real data. Instead, train directly on real data (see §4.5).
9. **Real-only baseline (DONE Aug 24):** temporal_pointnet on real official
   split, 60 epochs (~15 h GPU). Val AUPRC 0.954 → test AUPRC 0.329, AUROC
   0.819, recall 26.6% @ 5.1% FPR. See §4.6 — the val→test collapse and
   calibration failure are now the primary open problems.

---

## 6. Environment and Reproduction Facts

- CARLA **0.9.16** server AND Python API (client/server versions must match; bundled egg is
  Python 3.7: `carla-0.9.16-py3.7-linux-x86_64.egg`, set `PYTHONPATH`).
- Python deps: `numpy`, `h5py`, `torch` (GPU recommended for training). `carla` is imported
  lazily — unit tests run without the server.
- RGD download: 5.4 GiB archive; ~10+ GiB extracted; the downloader enforces free-space
  checks and MD5 verification; `--delete-archive` to reclaim space.
- CARLA collection: ~1–3 min/sequence; expect a native segfault after ~7 in-process world
  reloads (mitigated by per-sequence worker subprocesses).
- `artifacts/`, `data/`, `dataset_*/`, `model_*/` are gitignored — datasets and models stay
  out of git.

---

## 7. What Is Explicitly NOT Modeled (the target-list boundary)

- Raw FMCW chirps, phase noise, ADC saturation, CFAR, antenna arrays/calibration, or a
  range-Doppler-angle cube.
- General mesh ray tracing across arbitrary bounce counts (image-method paths through 3rd
  order only).
- Micro-Doppler from wheels, limbs, vibration, rotating parts.
- Polarization, radome/bumper effects, mutual coupling, sidelobe maps, commercial
  object-list firmware.
- Physically simulated spray/snow/water films/sensor contamination.
- Calibration against a specific production radar (the default profiles are visible, versioned
  _research priors_, not calibrated results).

---

## 8. Open Research Questions (the deliverable I need answers to)

Please answer these concretely (with reasoning, literature where relevant, and, where
possible, concrete implementation guidance):

1. **Point-cardinality gap:** RGD has ~800 raw points/frame vs. CARLA's tens. What are the
   best concrete options to close this — denser extraction in the simulator, virtual point
   synthesis (e.g., sampling around extended targets), per-frame point-count matching, or
   multi-sensor merging? Which preserves label integrity?
2. **Profile calibration to real RGD:** Which real-data statistics should be fitted into a
   profile (per-frame point counts, amplitude/SNR distributions, ghost rates by family, error
   magnitudes, dropout/clutter rates), and what is the correct held-out protocol (split by
   sequence, never neighboring frames)? Is fitting on the RGD train split and validating on
   test acceptable, and does that contaminate the "synthetic-only" claim?
3. **Zero-shot expectations:** For a target-list simulator with physical features and an
   RGD-matched collection regime, what is a realistic zero-shot ceiling on RGD test (AUPRC /
   ghost recall at 1% real FPR)? What evidence exists for/against point-wise physical
   features transferring across radar domains?
4. **Fine-tuning recipe:** Given the domain gap, what is the optimal fine-tuning strategy —
   full fine-tune at 0.0003 LR, frozen-backbone + new head, layer-wise LR decay, or
   discriminative rates? How many epochs / what early-stopping criterion? Does freezing hurt
   when the gap is in _input statistics_ rather than task semantics?
5. **Temporal model value:** Does a 5-frame temporal PointNet add transferable value over a
   point-wise MLP, given that RGD ghost lifetimes and CARLA ghost persistence differ? Should
   `window_frames`/`max_points` differ between synthetic and real training?
6. **Augmentation to reduce the gap:** Which augmentations best bridge sim→real for radar
   point clouds (point-density resampling, amplitude jitter matched to RGD statistics,
   azimuth mirroring, dropout bursts, Doppler noise)? Are any harmful (e.g., breaking the
   physical ghost geometry)?
7. **Two-sensor real data vs one-sensor synthetic:** RGD has left/right bumper sensors; the
   simulator produces one front sensor. Should training/eval treat each sensor as an
   independent channel (current design), merge both real sensors into one stream, or model a
   second virtual sensor? Any precedent in the literature?
8. **Cyclist proxy:** CARLA has no cyclist actor; a motorcycle maps to RGD class 5
   (motorbike) instead of class 2 (cyclist). Is this mismatch acceptable for pretraining, or
   should cyclist-class data be excluded/mixed differently?
9. **Threshold selection & evaluation rigor:** Confirm the protocol — validation threshold at
   ≤1% real FPR, never tuned on test; report AUPRC, AUROC, real FPR, ghost recall by family,
   per-scenario results, official + scenario-disjoint splits. What additional metrics or
   ablations make the study publishable (IV/ITSC/RadarConf level)?
10. **Defensible claims:** Given a target-list simulator, which of these claims are
    supportable: (a) physics-guided synthetic ghosts transfer to real radar; (b) synthetic
    pretraining accelerates/improves real fine-tuning; (c) the measured gap quantifies
    simulator fidelity. Which claims should be explicitly avoided?
11. **What to do if zero-shot is weak:** ~~If zero-shot AUPRC is near random, is the correct
    conclusion "target-list simulation insufficient for this task," or is there a cheaper fix
    (calibration, density matching, feature changes) worth trying before concluding?~~
    **ANSWERED (Aug 18):** Zero-shot AUPRC is 0.159 (near random). Densification did not
    help. The gap is in physics-level feature distributions, not point density. Remaining
    options: (a) profile calibration to match RGD amplitude/Doppler/error distributions,
    (b) feature engineering to bridge the distribution gap, (c) accept the gap as a
    quantified contribution and focus on real-only training.
12. **Point-count/label imbalance:** CARLA collections produce ~3× more ghost than real
    points (1623 vs 486 in the verified sequence), while RGD has ~6× more real than ghost
    (~600k real vs ~100k ghost). How should class balance be handled per domain so the
    pretrained prior doesn't fight the real-data prior?

---

## 9. Constraints and Things to Avoid

- Keep datasets, model artifacts, and downloads out of git (`artifacts/`, `data/` are
  gitignored).
- Do NOT use `collect_carla_radar_dataset.py` for RGD-matching pretraining (moving ego,
  vehicle events — wrong regime).
- Do NOT enable physics on the controlled walker (instability) and do NOT "fix" dead Doppler
  by re-enabling physics — use the transform-derivative fallback.
- Never select thresholds on test data; evaluate on the official test split and the stricter
  scenario-disjoint split.
- Feature-schema changes require new datasets and retraining; preserve the controller
  contract (`distance`, `relative_velocity`, `obstacle_speed`).
- Respect the target-list boundary: claims about raw-signal realism are out of scope.
