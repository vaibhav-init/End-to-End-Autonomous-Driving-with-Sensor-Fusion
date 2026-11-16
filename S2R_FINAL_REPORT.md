# Sim↔Real Radar Ghost Detection — Final Work Report

**Date:** 2026-11-16 · **Scope:** the complete real↔sim radar thread — what was built, which files implement it, every measured result with provenance, what was found, and what remains.
**Companion documents:** `S2R_MATCHED_REPORT.md` (the Nov-1 matched study), `S2R_CLOSING_RESEARCH.md` (literature verification + closing plan + claim-justification matrix), `SIM2REAL_RADAR_RESEARCH_BRIEF.md` (the historical negative-result record), `OCT_end_context.md` (the closed-loop thread).

---

## 1. The research question

**Can a target-list radar simulator inside CARLA, calibrated to real sensor statistics, produce synthetic training data that improves a real-data radar ghost (multipath false-alarm) detector?**

Answer established by this work:

- **Zero-shot: no.** Five generator generations, measured bidirectionally — synthetic-only models are at chance on the real test split, and real-only models are at chance on synthetic data.
- **Matched pretraining + fine-tuning: yes.** With the simulator calibrated to the real dataset and the synthetic collection matched to the real recording regime, pretraining plus two-stage fine-tuning **beats training on real data alone** — now established on **10 paired seeds** (10/10 positive on both AUROC and AUPRC, exact Wilcoxon p = 0.002).
- **The benefit does not survive naive deployment:** at the benchmark operating point, the learned filter suppresses the controller's real obstacles in dense traffic — measured, with the mechanism identified (train/deploy feature-computation mismatch + frame-structure domain gap).

---

## 2. Timeline of the work

| Phase | Period | What |
|---|---|---|
| Radar simulator construction | earlier 2026 | Target-list radar in CARLA 0.9.16: semantic-LiDAR → C-Shenron-derived material SNR → image-method multipath ghosts → temporal error model → tracker |
| Real data + real-only baseline | Jul–Aug 2026 | RGD v1.1 download/preparation; real-only temporal PointNet (test AUROC 0.819 / AUPRC 0.329, v1 schema) |
| Zero-shot failure studies | Aug 2026 | Densified, CFAR-expanded, measured-physics generators — all near-chance zero-shot; bidirectional gap quantified |
| Calibration loop | Sep–Oct 2026 | `calibrate_ghost_profile.py` fits ghost statistics from the RGD **train split only**; nuScenes ARS408 fits core sensor errors; fidelity tooling |
| Matched collection + matched study | Oct–Nov 2026 | RGD-regime collection (tangential motion, 4–10 m, 2 m gap, calibrated); matched study v1 (5 seeds) — headline positive |
| Closing phase | 13–16 Nov 2026 | Literature verification (3 hostile sweeps), arm C enablement + measurement, 10-seed upgrade, soups, MMD fidelity protocol, scenario-disjoint re-check, lever ablation, this report |

---

## 3. System architecture (how the pipeline works)

```text
                ┌─ CARLA 0.9.16 (Town04, sync mode)
                │
collect_carla_radar_ghosts.py        (RGD-regime collector: stationary ego,
 │                                    pedestrian/cyclist teleported along a
 │                                    reflector tangent, 10 Hz, ±70°, 38.5 s)
 ▼
radar/front_radar.py                 (semantic-LiDAR front end)
 ▼
radar/cshenron_core.py               (material + incidence-based SNR)
 ▼
radar/multipath.py                   (planar reflector fitting + image-method
 │                                    type-1/type-2 2nd-order, type-2 3rd-order)
 ▼
radar/realistic_core.py              (SNR-conditioned errors, AR(1) fading,
 │                                    dropout, clutter, NN tracker w/ M-of-N,
 │                                    per-point expanded emission)
 ▼                                    [+ optional ghost filter: oracle or
 │                                     radar/ghost_detection/runtime.py]
prepare_radar_ghost_dataset.py       (H5 → per-sequence npz, CMTO labels →
 │                                    binary, v2 frame-relative features
 │                                    computed PER FRAME)
 ▼
train_radar_ghost_detector.py        (TemporalPointNet / PointMLP, BCE with
 │                                    per-batch positive weighting, label
 │                                    smoothing, val-selected ≤1%-FPR threshold)
 ▼
evaluate_radar_ghost_detector.py     (pooled + per-scenario + per-family
                                     metrics on the RGD v1.1 official test split)
```

Feature contract (`radar/ghost_detection/features.py`, schema `radar_ghost_physical_v2`, 11 features): x, y, range, sin/cos azimuth, radial velocity, age — all scaled by physical constants — plus three **frame-relative** statistics computed identically per scan in every domain: relative log amplitude, Doppler cluster residual, local density ratio. No dataset-fitted normalization.

Calibration chain: `calibrate_rgd_real_data.py` (nuScenes → noise floors, σ-law, detection midpoint) and `calibrate_ghost_profile.py` (RGD train split → bounce losses, fading, points-per-object, amplitude gain) → `rgd_calibration_v7s/calibrated_overrides.json` applied to the collector profile.

---

## 4. Files used in this work

### Collection & simulation
| File | Role |
|---|---|
| `carla4/collect_carla_radar_ghosts.py` | RGD-regime collector (per-sequence worker subprocesses, resume, retries, verification block) |
| `carla4/radar/front_radar.py` | Three radar backends; ghost-filter plumbing; kinematic-velocity fallback for physics-off actors |
| `carla4/radar/realistic_core.py` | Temporal target-list model + tracker; `_apply_point_filter` (learned-filter deployment path) |
| `carla4/radar/multipath.py` | Reflector fitting + image-method ghost generation |
| `carla4/radar/cshenron_core.py` | Material/scattering SNR core |
| `carla4/radar/profiles/*.json` | Versioned sensor profiles (incl. `rgd_regime_v1`) |
| `carla4/radar/ghost_detection/runtime.py` | `RuntimeGhostFilter` — per-point scoring, amplitude gain, per-scan statistics |

### Data & training
| File | Role |
|---|---|
| `carla4/download_radar_ghost_dataset.py` | RGD v1.1 fetch/verify |
| `carla4/prepare_radar_ghost_dataset.py` | H5 → npz preparation (official / scenario_grouped / all_train splits; `scenario_disjoint` accepted alias) |
| `carla4/train_radar_ghost_detector.py` | Main trainer (label smoothing, select-metric, early stop, fine-tune hooks) |
| `carla4/evaluate_radar_ghost_detector.py` | Evaluator (reusable `evaluate_checkpoint`, `--tta`, protocol notes) |
| `carla4/radar/ghost_detection/{features,dataset,model,labels,metrics}.py` | Feature contract, windowed dataset, models, CMTO decode, streaming metrics |
| `carla4/calibrate_ghost_profile.py`, `carla4/calibrate_rgd_real_data.py` | Real-data calibration of the simulator |
| `carla4/calibrate_rgd_real_data.py` | nuScenes-based sensor-core calibration |

### Study tooling (built in the closing phase)
| File | Role |
|---|---|
| `carla4/run_matched_study.sh` | The matched 5-seed study runner |
| `carla4/run_synthetic_to_real.sh` | Fine-tune arms (real_only / ft_low / ft_auprc / ft_twostage / ft_head / ft_reset) + summary tables |
| `carla4/evaluate_ghost_transfer.py` | Cross-domain score dumps, val-calibrated thresholds, ensembles |
| `carla4/ensemble_ghost_scores.py` | Fixed seed-ensemble comparisons + paired bootstrap |
| `carla4/bootstrap_ghost_transfer.py` | Sequence/scenario-level paired bootstrap |
| `carla4/soup_ghost_checkpoints.py` | Uniform / greedy / WiSE-FT weight souping |
| `carla4/fidelity_mmd_table.py` | Per-feature MMD² vs a real-vs-real floor |
| `carla4/paired_seed_stats.py` | Exact Wilcoxon / sign test / paired t / bootstrap CI over seed pairs |
| `carla4/run_lever_ablation.sh` | Leave-one-out lever ablation runner |
| `carla4/analyze_overnight.sh` | One-command overnight-results analysis |
| `carla4/run_final_study.sh` | Closed-loop S5 arms incl. `pcrash_c*` (learned filter) |
| `carla4/calibrate_runtime_threshold.py` | Deployment-threshold curves from score-only sidecars |
| `carla4/evaluate_cross_domain.py` | Schema-tolerant cross-domain checks |

### Result artifacts (committed under `results_s2r/`)
`paired_seed_stats.json` (10-seed paired statistics) · `ensemble_10seeds.json` · `ensemble_recheck.json` (5-seed) · `ensemble_scenario_disjoint.json` · `matched5_ensemble.json` · `bootstrap_matched_vs_real.json` · `mmd_fidelity_matched.json` / `mmd_fidelity_v7s.json` · `threshold_calibration_{real,matched}{,_perscan}.json` · `soup_matched10_{uniform,greedy}_report.json` · `soup_matched5_greedy_report.json` · `soup_matched5_wise90_report.json` · `lever_ablation_summaries.txt` · `arm_c_closed_loop_summary.csv`.

### Documentation
`S2R_MATCHED_REPORT.md`, `S2R_CLOSING_RESEARCH.md`, `SIM2REAL_RADAR_RESEARCH_BRIEF.md`, `carla4/radar/GHOST_DETECTION.md`, `carla4/radar/RGD_REGIME_COLLECTION.md`, `carla4/radar/README.md`, `CLAUDE.md` (working conventions).

---

## 5. How we worked

1. **Authoring locally, execution on the VM.** All code is written and unit-tested locally (`python3 -m compileall` + the CARLA-free tests), committed to GitHub, and pulled on the VM (`iiitd@192.168.21.33`, repo `~/Desktop/vaibhav/carla-claude/End-to-End-Autonomous-Driving-with-Sensor-Fusion`), where CARLA 0.9.16 (`/storage/CARLA_0.9.16`) and the PCLA conda env (torch 2.2.0, h5py, scipy) live. Heavy runs go through staged bash runners inside `tmux` so they survive disconnects.
2. **Everything through the same measurement path.** One evaluator (`evaluate_checkpoint`), one threshold protocol (val-selected ≤1% real FPR, test evaluated once), one stats path (paired seeds + sequence/scenario bootstrap).
3. **Calibration only ever touches train splits.** The fidelity numbers measured on val/test afterwards are honest.
4. **Verification discipline.** Every claimed number was re-run at least twice with the final code; three bugs found during verification (pooled-window statistics, saturated-boundary rejections, an RBF broadcasting bug in the MMD tool) were fixed with regression tests before results were accepted.
5. **Long runs:** CARLA-bound collections (~1–3 min/sequence) and GPU training (~1 h/fine-tune) run unattended overnight; analysis is one idempotent command (`analyze_overnight.sh`).

---

## 6. Measured results (all on RGD v1.1 official test split unless stated)

### 6.1 Headline — matched synthetic pretraining + two-stage fine-tuning vs real-only

**10 paired seeds (5 original: 7, 42, 123, 256, 2024 + 5 added: 5, 11, 77, 99, 300).** Artifact: `paired_seed_stats.json`.

| Metric | matched (mean ± SD) | real-only (mean ± SD) | Δ (matched − real) |
|---|---|---|---|
| AUROC | 0.8834 ± 0.0136 | 0.8611 ± 0.0122 | **+0.0223 ± 0.0144** |
| AUPRC | 0.5417 ± 0.0470 | 0.4481 ± 0.0386 | **+0.0936 ± 0.0461** |

Δ statistics: **10/10 seeds positive on both metrics · exact Wilcoxon p = 0.002 · sign test p = 0.002 · bootstrap 95% CI [+0.0140, +0.0311] (AUROC), [+0.0669, +0.1214] (AUPRC).**
(The 5-seed original: ΔAUROC +0.0239, ΔAUPRC +0.1051, paired t p ≈ 0.017/0.019 — consistent.)

### 6.2 Ensembles and soups

| Artifact | AUROC | AUPRC | note |
|---|---|---|---|
| 5-seed mean ensemble | 0.8922 | 0.6069 | threshold-free |
| **10-seed mean ensemble** | **0.9006** | 0.4891 | paired vs realonly-10: ΔAUROC +0.0228, p = 0.033 |
| 5-seed greedy soup (2 members) | 0.8983 | 0.6246 | val-selected threshold |
| 10-seed greedy soup | 0.8983 | 0.6246 | **same 2 members chosen from 10 seeds — stable selection** |
| 10-seed uniform soup | 0.8964 | 0.5969 | |
| WiSE-FT (α=0.9/0.75/0.5) | 0.8872 / 0.8613 / 0.8208 | 0.533 / 0.353 / 0.278 | monotonically worse — anchor interpolation hurts; negative ablation |

Best deployable artifact: the **2-member greedy soup** (no inference cost, beats the best single seed 0.8947 and every ensemble).

### 6.3 Negative results (measured, publishable)

1. **Zero-shot transfer fails** — 5 generator generations (densified v1, CFAR-expanded v2 [inverted AUROC 0.281], v3, measured-physics, matched [0.7698]), confirmed bidirectionally (real-trained on synthetic ≈ chance). Fenced in the literature: sensor-matched simulators can succeed zero-shot; ours cannot at this fidelity.
2. **Label-scarce regime:** with 5 real sequences the matched prior *hurts* (0.8505 vs real-only 0.8929); with 10 sequences AUROC improves but AUPRC stays below real-only. Literature rescue (untested here): mmSimPrior-style DR curriculum.
3. **Adding a second diverse-but-unmatched synthetic set (matched+v7s) hurt:** 0.8819/0.5039 vs matched-alone 0.8849/0.5536.
4. **WiSE-FT anchor interpolation hurts** (§6.2).
5. **Deployment operating point does not transfer** (§6.5).

### 6.4 Fidelity (synthetic vs real distributions)

Corrected per-feature MMD² vs the real-vs-real floor (artifact `mmd_fidelity_matched.json`; protocol = our contribution, RadarGen-style per-attribute MMD):

| feature | matched ratio | v7s ratio |
|---|---|---|
| radial velocity | **0.76 (at/below floor — Doppler essentially exact)** | 4.25 |
| Doppler cluster residual | 4.94 | 2.63 |
| local density ratio | 9.01 | 10.97 |
| geometry (x/y/range/azimuth) | 6.7–47.0 | 1.3–29.7 |
| relative log amplitude | **81.7** | 104.1 |
| log range-compensated amplitude | 44.6 | 22.5 |
| joint 11-d | 16.6 | 12.8 |

**Mechanism reading:** marginal fidelity does not predict transfer — matched wins on transfer despite a higher joint MMD, and its advantage sits exactly in the two feature families the detector depends on (Doppler, relative amplitude), consistent with RGD's own Doppler ablation (ghost F1 < 5% without Doppler). Amplitude is the quantified residual gap.

Cross-domain check (schema-current): the real-trained detector ranks the matched synthetic test set at **AUROC 0.8313 / AUPRC 0.6964** — calibrated CARLA ghosts are recognizable by a real-data detector.

### 6.5 Split robustness

Scenario-disjoint val (18 sequences from scenarios 12/15/20, disjoint from the 5 test scenarios): threshold-free deltas unchanged (p = 0.039/0.004); the val-selected operating point is threshold-fragile (matched thr 0.799→0.557, recall 0.743→0.888 @ FPR 0.288). Caveat: val *scenarios* appear in training with other sequences — the fully clean version needs retraining (open).

### 6.6 Arm C — the learned filter in the closed loop

1. **Fixed to score on its training distribution** (`runtime.py`: per-point expanded scoring, deployed `amplitude_gain_db` applied, **per-scan frame statistics** — the earlier pooled-window computation was a train/deploy mismatch; training-style context-reserve budget; threshold ≥ 1.0 = annotate-only). RNG parity and accept-all equivalence pinned by tests.
2. **Benchmark operating point in the closed loop (partial 5-run arm):** eliminates ghost selection (0.00 ghost-selected fraction) and cuts phantom braking (14.25/km vs 25.4–30.2) — **but by blinding the controller**: follower validity collapsed to 3/5 runs, gaps to 207 m.
3. **Deployment-threshold calibration on sim-exact labels (n=2 full-length seeds, artifacts `threshold_calibration_*_perscan.json`):** **no usable threshold exists** — real-only filter loses 0.93–0.98 of dynamic clusters at every threshold; matched 0.60–0.95 (far more robust in-domain, still no knee).
4. **Conclusion:** a benchmark-calibrated filter cannot be deployed as-is; the operating point must be recalibrated on the deployment domain (dense-scene retraining with sim labels — open follow-up). Until then, arm D ("train through") remains the only viable single-model answer.
5. **Methodology note:** the S5 closed-loop harness is not bit-deterministic per seed (verified: identical config → 3.9 m/126 rows vs 3.8 m/81 rows; consistent with CARLA TM issues #3839/#4026/#6082) — closed-loop claims rest on multi-seed averages only.

### 6.7 Lever ablation (leave-one-out over the matched levers)

Protocol: identical 18-scene collection per variant (two cyclist scenes crash the worker deterministically and are excluded for all variants), zero-shot-anchored pretrain + two-stage fine-tune, seeds 42/123. Artifact: `lever_ablation_summaries.txt`.

| Variant (lever removed) | AUROC | AUPRC |
|---|---|---|
| **base** (nothing removed) | **0.8507 ± 0.0006** | **0.3477 ± 0.0041** |
| radial_motion (tangential→radial) | 0.8708 ± 0.0227 | 0.4753 ± 0.0194 |
| low_density (336→sparse pts) | 0.8880 ± 0.0079 | 0.4744 ± 0.0417 |
| uncalibrated_amplitude | 0.8735 ± 0.0098 | 0.4434 ± 0.0256 |

**Honest interpretation:** the base variant is the *worst*, and all three removals beat it — the opposite of the expected leave-one-out pattern. The 3-way base/radial/uncalibrated comparison is clean (identical 9-scene prefixes, 17 sequences each), so the signal is real at n=2 but contradicts the matched study's 20-scene result (0.885/0.554). Most defensible reading: **with ~17 scenes the matched levers alone do not reproduce the matched advantage — scene count/composition interacts with (or dominates) the lever effects**, i.e. the recipe's benefit is not decomposable into independently additive levers at this data scale. Status: **pilot result, confound-limited (n=2, scene subsets differ slightly at the margins); requires a controlled re-run (intersection scene set, ≥5 seeds) before any attribution claim.** Two levers (target range, label scope) were not ablated.

---

## 7. Findings and claims (with support status)

| # | Claim | Status |
|---|---|---|
| F1 | Matched synthetic pretraining + two-stage FT beats real-only training (ΔAUROC +0.0223, ΔAUPRC +0.0936; 10/10 seeds; Wilcoxon p = 0.002) | **supported (10 seeds)** |
| F2 | Zero-shot transfer fails at this simulator fidelity, bidirectionally | **supported (5 generations)** |
| F3 | Doppler fidelity is the transfer-critical feature family; amplitude is the residual gap | **supported (MMD + RGD ablation + literature)** |
| F4 | Marginal fidelity does not predict transfer (matched > v7s despite worse joint MMD) | **supported** |
| F5 | A 2-member greedy soup beats single seeds and ensembles at no inference cost, stable across 5→10 seeds | **supported** |
| F6 | Benchmark-calibrated operating points do not survive dense deployment (filter blinds the controller) | **supported (n=2 seeds, closed-loop partial arm)** |
| F7 | Train/deploy feature-computation consistency is load-bearing (pooled-window statistics inflated ghost probabilities) | **supported (fixed + test-pinned)** |
| F8 | Label-scarce pretraining hurts at 5–10 real sequences | **supported** |
| F9 | Adding unmatched diverse synthetic data to a sequential corpus hurts AUPRC | **supported (single configuration)** |
| F10 | Lever effects are not independently additive at ~17-scene scale | **pilot — confound-limited, needs controlled re-run** |
| F11 | CARLA closed-loop runs are not bit-deterministic per seed | **supported (documented TM behavior)** |

Novelty fence (verified 2026-11-14, see `S2R_CLOSING_RESEARCH.md` §2/§9): no published work trains a ghost detector on simulator-generated ghost data and transfers it to the real Radar Ghost Dataset; nobody reports AUROC/AUPRC on the official v1.1 test split; paired-seed significance testing is absent from radar sim2real literature.

---

## 8. Reproduction

```bash
# ── environment ───────────────────────────────────────────────────────────
# VM: iiitd@192.168.21.33, repo ~/Desktop/vaibhav/carla-claude/End-to-End-
# Autonomous-Driving-with-Sensor-Fusion, conda env PCLA, CARLA 0.9.16 at
# /storage/CARLA_0.9.16 (port 2000), TM port 8050.

cd carla4
export PYTHON=$HOME/anaconda3/envs/PCLA/bin/python
python3 -m compileall . && $PYTHON -m unittest discover -s radar/tests -p 'test_*.py'

# ── real data ─────────────────────────────────────────────────────────────
python3 download_radar_ghost_dataset.py
python3 prepare_radar_ghost_dataset.py --input data/radar_ghost_v1_1/original \
    --output artifacts/ghost_real_official_v2 --split-mode official

# ── matched synthetic collection (matched recipe) ────────────────────────
python3 collect_carla_radar_ghosts.py --target-type cyclist --split train \
    --seed 100 --sequences 10 --headless --town Town04 \
    --output artifacts/carla_ghost_matched_v1 --motion-direction tangent \
    --label-scope main --surface-distance-target 2.0 --target-range-min 4 \
    --target-range-max 10 --points-per-second 100000 --points-source sensor \
    --radar-config artifacts/rgd_calibration_v7s/calibrated_overrides.json
#   (repeat for --target-type pedestrian --seed 2000, and val/test splits
#    with distinct seeds; see radar/RGD_REGIME_COLLECTION.md)

# ── matched study (5 seeds) ──────────────────────────────────────────────
S=/storage/vaibhav/s2r_results tmux new -d -s matched './run_matched_study.sh'

# ── 10-seed upgrade (per seed: matched then real-only) ───────────────────
for s in 5 11 77 99 300; do
  PRETRAINED=/storage/vaibhav/s2r_results/ft_matched_v1/best_detector.pt \
  OUT=/storage/vaibhav/s2r_results/s2r_matched EPOCHS=30 PATIENCE=8 \
    ./run_synthetic_to_real.sh train ft_twostage $s
  OUT=/storage/vaibhav/s2r_results/s2r EPOCHS=30 PATIENCE=8 \
    ./run_synthetic_to_real.sh train real_only $s
done

# ── statistics, ensembles, soups, fidelity ───────────────────────────────
python3 paired_seed_stats.py --matched /storage/vaibhav/s2r_results/s2r_matched \
    --real /storage/vaibhav/s2r_results/s2r \
    --seeds 7 42 123 256 2024 5 11 77 99 300 \
    --output /storage/vaibhav/s2r_results/paired_seed_stats.json
python3 fidelity_mmd_table.py --real artifacts/ghost_real_official_v2 \
    --synthetic artifacts/ghost_carla_matched_v1_prepared --split test \
    --output results_s2r/mmd_fidelity_matched.json
#   (dumps + ensembles: evaluate_ghost_transfer.py dump / analyze;
#    soups: soup_ghost_checkpoints.py; see analyze_overnight.sh)

# ── closed-loop arm C (CARLA required) ───────────────────────────────────
tmux new -d -s calib './run_final_study.sh pcrash_c_score_only'   # calibration
C_THRESHOLD=<knee> tmux new -d -s armc './run_final_study.sh pcrash_c'

# ── lever ablation (CARLA required) ──────────────────────────────────────
tmux new -d -s levers './run_lever_ablation.sh base && ./run_lever_ablation.sh radial_motion && ./run_lever_ablation.sh low_density && ./run_lever_ablation.sh uncalibrated_amplitude'
```

Unit suite: **180 tests, OK** on the GPU box (last full run 2026-11-15).

---

## 9. What remains

1. **Controlled lever ablation re-run** (intersection scene set, ≥5 seeds; add the two missing levers: target range, label scope) — required before any lever-attribution claim.
2. **Deployment-domain filter retraining** (dense-scene detector data with sim labels → recalibrate on sim → re-run arm C) — the one experiment that could flip F6.
3. **Fully-clean scenario-disjoint retraining** (val scenarios excluded from training).
4. Optional: LOGO-style held-out-group insurance run (pre-empt the Ali preprint), scaling curve (20→50→100 synthetic sequences), `wide_range`/`all_labels` variants.
5. Paper: RadarConf 2027 hard deadline **Nov 30, 2026** (3–6 pages incl. references; no post-submission updates) or ITSC 2027 (Mar 1, 2027). All justification sentences and the claim-verification matrix are ready in `S2R_CLOSING_RESEARCH.md` §9.

---

## 10. Bottom line

The thread establishes, with a rigor level above the current radar-literature norm (10 paired seeds, exact nonparametric tests, val-only thresholds, bidirectional negatives, per-feature fidelity floors): **calibrated target-list simulation is useful for radar ghost detection exactly when it is matched to the real recording regime and adapted head-first onto real data — and its limits are equally measurable: no zero-shot, no naive deployment, no help in the label-scarce regime.** The remaining open questions are the lever-level mechanism (F10) and deployment-domain recalibration (F6), both scoped and runnable.
