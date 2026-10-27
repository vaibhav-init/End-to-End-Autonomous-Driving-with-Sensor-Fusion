# OCT End Context — radar artefacts, calibration and closed-loop cost

**Generated:** 2026-10-27  
**Repo HEAD:** `b8bf058` (`cal(radar): ghost point yield calibrated to RGD aggregate`) plus the uncommitted session changes listed below.  
**Purpose:** a single handover file for the October 2026 session: what the project is trying to prove, what was changed, what was measured, what the final result is, and what remains.

---

## 1. The goal, at three levels

### Long-term thesis
A **LiDAR-free autonomous driving stack using radar + camera only**, studied in CARLA. The claim is not “eliminate LiDAR”; it is **“quantify the closed-loop cost of removing LiDAR”** on a declared ODD (cost-constrained L2+/L3 longitudinal control, highway/arterial), leading with failure decorrelation rather than sensor cost.

### Current paper
The strongest defensible framing after the literature and experiments:

> **“The False-Positive Budget: what a structured radar artefact costs a driving controller, and what false-positive rate a radar-only controller can afford.”**

Pillars:
- A **calibrated CARLA radar model** (`rgd_regime_v1`) with exact per-point labels.
- **Structured vs rate-matched random artefacts** — the same false-detection rate, different physical origin.
- An **oracle ceiling** — what any perfect filter could recover.
- **`p(crash | phantom brake)`** — the missing denominator in the phantom-braking literature.
- **Requirements arithmetic** — range, latency, and allowable false-positive rate, anchored to FMVSS 127 S5.3 (0.25 g) and ISO 21448-style reasoning.

### Professor’s constraint
**No two-layer patch** (detector → tracker → controller). Mitigation must live inside the temporal model/controller. Arm D (“train through the ghosts”) is the single-model answer; “filter it or train through it?” is the practitioner question.

---

## 2. What the repo contains

- **`carla4/`** — first-party pipeline:
  - `radar/` — `native`, `cshenron`, `realistic` backends; profiles; multipath; ghost detector; realism metrics; RRI; tests.
  - `scenarios/` — S1 stopped lead, S2 decelerating lead, S4 cut-in, S5 ghost exposure; pluggable `pcla`/`mlp`/`transformer`/`cnn` drivers; metrics and ground-truth logging.
  - Top-level collectors, trainers, evaluators, calibration scripts, staged runners.
- **`PCLA/`** — vendored pretrained CARLA Leaderboard agents. After cleanup only `transfuserv6` + `transfuserv6_pretrained/visiononly_resnet34` remain (`tfv6_visiononly`).
- **`research_state/`** — ~110 MB of primary sources and harvested literature (untracked locally).
- **`results_2026-10-09/`** — Stage C realism and fine-tune transfer results (untracked locally).
- **Root Markdown docs** — `CLAUDE.md`, `AGENTS.md`, `ideas.md`, `directions.md`, `2026-10-09.md`, handovers, historical briefs.

---

## 3. How this session worked

1. **Read all first-party documentation**: `directions.md` (all 3,337 lines), `ideas.md` (1,591 lines), both handovers, `2026-10-09.md`, professor briefs, `CLAUDE.md`, `work.md`, `vision.md`, and the conclusions of the historical briefs.
2. **Audited every first-party source area**: radar package, scenario harness, drivers, RRI study, MATLAB comparison, PCLA, research corpus.
3. **Ran five parallel literature sweeps** (ghost detection, radar simulation, closed-loop cost, radar-camera robustness, interference/weather/4D), then verified key IDs directly against arXiv/Crossref/USENIX.
4. **Worked on the VM over SSH**; edited locally, synced specific files.
5. **Fixed the measurement before running more experiments**:
   - standstill HOLD frames excluded
   - staging frames excluded via `driver_in_control`
   - curvature-aware GT corridor
   - per-run phantom rates
   - rear-end + Wilson CI
   - follower-validity filter
6. **Built a parameterised S5 follower** and iterated on it:
   - first version used delayed ego speed as feed-forward → fell 150 m behind
   - second version used current speed but BasicAgent steering → left the lane and stopped
   - final version uses **pure pursuit toward a point ahead of the ego** and holds 15–28 m
7. **Ran the final 10-seed S5 study** in a clean `/tmp/final_run` staging copy because the VM’s absolute path lookup became intermittently stale after heavy I/O.
8. **Generated `FINAL_REPORT.md` from measured artefacts**, not hand-typed numbers.
9. **Synced final code, raw results, and report back into the repo.**

---

## 4. Environment and cleanup state

| Item | State |
|---|---|
| Host | `iiitd-HEXADATA`, user `iiitd` |
| Repo | `~/Desktop/vaibhav/carla-claude/End-to-End-Autonomous-Driving-with-Sensor-Fusion` |
| Work copy for final runs | `/tmp/final_run` |
| CARLA | `/storage/CARLA_0.9.16`, port **2000** (server was running at session end) |
| Traffic Manager port | **8050** |
| Python env | `~/anaconda3/envs/PCLA` (Python 3.10.20, torch 2.2.0+cu121, CUDA available) |
| Disk | `~/Desktop/vaibhav` ~30 GB; root 92% used, **76 GB free** |
| PCLA | 2.1 GB → **848 MB**; only `transfuserv6` + `tfv6_visiononly` |
| Deleted | `C-Shenron` (21 GB), `venv_carla_916` (12 GB), 13 other PCLA agent families |
| Git | HEAD `b8bf058`; ~2,117 working-tree changes (mostly the intentional PCLA deletions + new code/results); **not committed** |

### Important VM quirks
- The absolute path to the repo intermittently returns `ENOENT` after heavy I/O even though `ls`/`find` see it. Use:
  ```bash
  cd ~/Desktop/vaibhav/carla-claude
  cd ./End-to-End-Autonomous-Driving-with-Sensor-Fusion/carla4
  ```
  and stay on relative paths. `/tmp/final_run` is stable.
- `CARLA_ROOT` is unset by default; export `CARLA_ROOT=/storage/CARLA_0.9.16`.
- TM port 8000 is occupied; use 8050.
- `run_all.py` must be invoked with cwd `scenarios` because it launches sibling scripts by relative name.
- The shipped MLP models expect radar signature **`015b654fd7d543fb`** (profile `rgd_regime_v1_rgd_calibrated`, old noise/loss config). Use `carla4/artifacts/final/model_matched_015b.json` as `--radar-config`.
- `model_mlp_clean` and `model_mlp_ghost` both pass the acceptance test.

---

## 5. Code changes made this session

### `carla4/scenarios/metrics.py`
- `phantom_brake_mask` now:
  - ignores standstill frames (`speed >= 0.5 m/s`)
  - ignores frames where `driver_in_control == 0` (staging)
- Added `wilson_interval`.
- Added `rear_end_metrics`:
  - attributes rear-end collisions to phantom vs legitimate brake episodes within 5 s
  - returns `p_crash_given_phantom`, `p_crash_given_legitimate`, Wilson CI
  - adds `follower_max_gap_m`, `follower_valid`
  - a run is **follower-valid only if max gap ≤ 60 m**; otherwise rear-end metrics are NaN and excluded from the pooled denominator.

### `carla4/scenarios/ground_truth_logger.py`
- Curvature-aware path corridor (`path_curvature_per_m`, matching the radar selector).
- New columns:
  - `driver_in_control`
  - `radar_path_curvature_per_m`
  - follower columns: `follower_present`, `follower_gap_m`, `follower_speed_kmh`, `follower_brake`, `follower_target_speed_kmh`, `follower_reaction_s`, `rear_collision`
- `nearest_in_path_actor(..., path_curvature_per_m=...)`.

### `carla4/scenarios/follower.py` (new)
- `spawn_follower`: spawns a four-wheel vehicle in the ego’s lane `gap_m` behind.
- `FollowerController`:
  - delayed ego speed for the braking-reaction lag
  - current ego speed feed-forward when the gap is healthy
  - pure-pursuit steering toward a point 12 m ahead of the ego
  - explicit `hand_brake=False`
  - holds 15–28 m in practice at a 20 m setpoint.

### `carla4/scenarios/s5_ghost_exposure.py`
- Spawns follower before warm-up; drives it during warm-up and the run.
- Follower collision sensor detects rear-end with ego.
- Passes curvature into GT actor-in-path.
- New args: `--follower/--no-follower`, `--follower-gap-m`, `--follower-reaction-s`.
- Logs all follower columns and `rear_collision`.

### `carla4/scenarios/s1_lead_vehicle_stopped.py`
- `random.seed(seed)` so the obstacle blueprint is paired across arms.
- Logs `driver_in_control=obstacle_spawned`.

### `carla4/scenarios/s2_lead_vehicle_decelerating.py`, `s4_cut_in.py`
- Log `driver_in_control` so staging frames can be excluded.

### `carla4/scenarios/analyze_results.py`
- Per-run phantom rates (`mean/std/median`) and pooled rate.
- `p_crash_given_phantom` + Wilson CI.
- `n_follower_valid`, `max_follower_gap_m`.
- `radar_sync_timeouts`, `stale_radar_frames`, `n_runs_with_stale_radar`.
- `--skip-validation` for missing/extra runs.

### `carla4/scenarios/requirements.py` (new)
- `R_req = vτ + v²/(2a)`
- `τ_max = R/v − v/(2a)`
- `allowable FP rate = C_target / p(crash | phantom)`
- Outputs range/latency tables and FP budget at 40–130 km/h.

### `carla4/scenarios/power_analysis.py` (new)
- Reads `per_run_metrics.csv`.
- Bootstrap CI for phantom/km, jerk, collision.
- MDE vs N; runs/arm needed for 1/2/5/10 phantom/km effects.
- Writes `power_summary.csv` + `.md`.

### `carla4/scenarios/make_final_report.py` (new)
- Builds `FINAL_REPORT.md` from Stage C, transfer, structured-vs-random, counterfactual, power, `p(crash)`, requirements.
- Dynamic claims table; no hand-typed result numbers.

### `carla4/run_final_study.sh` (new)
- `smoke|pilot|pcrash|pcrash_close|analyze|all`
- Runs S5 off/on/ghost-trained + close-follower arms, then analysis/report.
- Uses relative paths and `CARLA_ROOT=/storage/CARLA_0.9.16`, TM 8050.

### `carla4/structured_vs_random_artefacts.py`
- Fixed Python 3.10 f-string syntax.

### `carla4/radar/tests/test_closed_loop_metrics.py`
- New tests: standstill HOLD not phantom, staging frames not scored, rear-end attribution.
- Full suite: **154 tests pass**.

---

## 6. Experiments and findings

### 6.1 Sensor calibration — Stage C v8
CARLA 0.9.16, `realistic`, `rgd_regime_v1`, 180 s @ 20 Hz, Town10HD_Opt, seed 7.

| Metric | Real | Sim | Delta |
|---|---:|---:|---:|
| range error std | 0.792 m | **0.836 m** | +0.044 |
| velocity error std | 2.771 m/s | **2.898 m/s** | +0.127 |
| velocity misassignment | 0.527 | 0.573 | +0.046 |
| ghosts / scan | 87.2 | 299.4 | +212.2 |

Selected-target core sigma: range **0.266 m**, velocity **0.438 m/s**. KS: range **0.159**, velocity **0.322**.

**Reading:** matched-pair core errors agree. Ghost density is not comparable because the staged scene has one lead while real streets have dozens of objects.

### 6.2 Sim-to-real transfer (RGD v1.1 official test)

| Fine-tune source | AUROC | AUPRC |
|---|---:|---:|
| v7s fine-tune (project record) | **0.787** | 0.226 |
| real-only ceiling (project record) | **0.819** | 0.329 |
| measured physics, Town10 | 0.650 | 0.194 |
| ratio-matched volume | 0.511 | 0.157 |
| three-town combined | 0.376 | 0.087 |
| Town04 sane-ratio | 0.353 | 0.085 |

**Reading:** label-ratio match is necessary; ghost-event phenomenology is the binding constraint. Zero-shot transfer remains closed.

### 6.3 Structured vs random false positives
5 seeds × 3,000 scans, same sensor/tracker/scene/seed; only the false-detection origin changes.

| Case | False detections / scan | Reached controller |
|---|---:|---:|
| multipath ghost, in-path | 1.13 | **47.27%** |
| random clutter, rate 1.3 | 1.13 | **0.38%** |

**Ratio: 124.4×.** Factorisation: placement **0.38% → 21.21%** (55.8×), coherence **21.21% → 47.27%** (2.2×). Out-of-path ghost ≈2.9%.

**This is the project’s cleanest claim.** The driving cost of a false positive depends on its physical origin, not only its rate. It is a direct refutation of i.i.d.-FPR modelling assumptions (e.g. CODEI-style sensor selection).

### 6.4 Open-loop counterfactual (2,000 windows)

| Model | Ghost windows | Ghost-removal flip | Direct-removal flip | Broke | Fixed | Accuracy change |
|---|---:|---:|---:|---:|---:|---:|
| ghost-trained transformer | 1,524 | 1.25% | 0.59% | 6 | 13 | **+0.46%** (p=0.17) |
| clean-trained transformer | 1,524 | 0.72% | 0.46% | 8 | 3 | **−0.33%** (p=0.23) |

**Reading:** the direction flips with training, but neither is statistically resolvable. Open-loop sensitivity is real but small; it is not a reliable basis for a filtering policy.

### 6.5 Power analysis of the old closed-loop table

The old 10-seed arms cannot resolve the effect they were built to measure.

| Arm | Scenario | Phantom/km mean | SD | Runs/arm for 5/km | Runs/arm for 10/km |
|---|---:|---:|---:|---:|---:|
| clean MLP, no ghosts | S5 | 22.05 | 11.73 | 87 | 22 |
| clean MLP + ghosts | S5 | 32.60 | 22.57 | 320 | 80 |
| clean transformer + ghosts | S5 | 23.02 | 18.44 | 214 | 54 |
| clean CNN + ghosts | S5 | 36.66 | 9.86 | 62 | 16 |

**Conclusion:** the old design is underpowered by roughly an order of magnitude. Do not report a ghost-effect ranking from it.

### 6.6 Final S5 rear-end experiment — the main new result

10 seeds per arm. 20 m follower at 0.8 s reaction for `off`, `on`, `d`; 10 m follower at 1.0 s reaction for `close_off`, `close_on`.

| Arm | Follower | Runs | Phantom brakes | Rear-end crashes | `p(crash | phantom)` | 95% CI |
|---|---|---:|---:|---:|---:|---|
| `close_off` | 10 m / 1.0 s | 10 | 113 | 0 | 0.00% | 0.00–3.29% |
| `close_on` | 10 m / 1.0 s | 10 | 100 | 0 | 0.00% | 0.00–3.70% |
| `d` | 20 m / 0.8 s | 10 | 77 | 0 | 0.00% | 0.00–4.75% |
| `off` | 20 m / 0.8 s | 10 | 79 | 0 | 0.00% | 0.00–4.64% |
| `on` | 20 m / 0.8 s | 10 | 84 | 0 | 0.00% | 0.00–4.37% |
| **pooled valid** | — | — | **453** | **0** | **0.00%** | **0.00–0.84%** |

Phantom brakes/km: `close_off` 32.0, `close_on` 45.3, `d` 28.0, `off` 30.2, `on` 25.4.

**Reading:**
- At these staged conditions, **phantom braking did not produce a rear-end crash in 453 episodes**; the 95% upper bound is **0.84%**.
- Ghosts did **not** measurably increase phantom braking in the 10-seed sample (off 30.2 vs on 25.4/km); the ghost effect remains below resolution.
- The result is a **bounded negative**, not “p=0”. A more aggressive follower, shorter gap, or longer exposure could raise the point estimate.
- Forward ego collisions (2–4 per arm) are counted separately and are **not** rear-end crashes.
- One run in each 20 m arm lost the follower (>60 m); those runs are excluded from the denominator. All runs logged one stale radar frame.

### 6.7 Acceptance and dataset checks
- `model_mlp_clean`: PASS both braking gates.
- `model_mlp_ghost`: PASS both braking gates.
- `dataset_clean`: 11,900 rows, target detected 76.5%, braking-while-moving 7.0%, stopped 8.1%.
- `dataset_ghost`: same rows, ghost selected 0.8%, ghost within 30 m 0.6% (70 frames).

---

## 7. Literature update (Oct 2026)

New work not previously in the repo’s prior-art map:

### Direct threats / must-cite
- **Ali, *Auditing Radar Ghost Labels: Multipath Geometry, Detection, and Risk Control*** — Preprints.org, 7 Oct 2026, DOI `10.20944/preprints202610.0381.v1`. Motion-aware audit of RGD; position-only matching inflates ghost agreement ~20×; Learn-then-Test risk control removes 45% of ghosts at <5% real loss; tracker ghost tracks 11.7→3.3/min. **Owns label auditing and risk-controlled detection.** Adopt, do not claim.
- **Yahia et al., *Seeing is Deceiving: Systematic Vulnerability Analysis of LiDAR-Based Autonomous Driving to Mirror-Induced Perception Failures*** — USENIX VehicleSec ’26, Aug 2026; arXiv:2509.17253. Planar mirrors create phantom LiDAR obstacles → Autoware AEB → follower rear-end in CARLA. **Closest precedent to the p(crash) experiment**; generic causal chain is no longer unclaimed. Our version is radar-multipath-specific with exact labels and a filter ceiling.
- **Ding et al., *Multipath Extended Target Tracking with Labeled Random Finite Sets*** — arXiv:2602.03464. Joint target/reflector RFS tracking; “track through” precedent.
- **Kweon & Monga, *TIGRE*** — arXiv:2602.11441; Asilomar 2025. Physics-regularised DOA/DOD ghost identification.
- **Takahashi & Wang, *GREST*** — ICASSP 2025, DOI `10.1109/ICASSP49660.2025.10889590`. Training-free two-way-angle ghost removal.
- **Li et al., *GCPANet*** — SSRN 2026, DOI `10.2139/ssrn.7243285` (unrefereed). Geometrically constrained point attention ghost recognition.

### Simulation / validation
- **DyRAD** — arXiv:2609.39841. Radar NVS for dynamic driving scenes; closed-loop motivation, not CARLA/control.
- **AI-TwinRadar** — CCECE 2026, DOI `10.1109/CCECE68150.2026.11610239`; **TwinRadar** — Zenodo DOI `10.5281/zenodo.23578728`. Virtual radar ECU + closed-loop ADAS.
- **RadarTwin** — arXiv:2606.28396; **mmDiff** — arXiv:2605.26406; **Simulating Automotive Radar with Lidar and Camera Inputs** — IROS 2025, DOI `10.1109/IROS60139.2025.11247276`.
- **Radar2Plan** — arXiv:2610.06121. 4D radar open-loop planning benchmark.
- **Betschinske et al., *Towards more efficient quantitative safety validation of residual risk*** — arXiv:2506.10363. ISO 21448 C2.2 anchor: AEB unintended braking vs human rear-end-crash rate.
- **NHTSA PE22-002** — reported closed 29 Jun 2026: 25,900 incidents, **zero crashes/injuries**. Strong empirical prior for a small `p(crash | phantom)`.

### Fusion / robustness
- **RFusion4D** — IEEE OJ-ITS 2026, DOI `10.1109/OJITS.2026.3706683`; **RbFT-Net** — arXiv:2608.13102. Temporal radar-camera fusion robust to clutter/multipath.
- **Adversarial Robustness of Camera–Radar Sensor Fusion** — arXiv:2610.16012. Random ghost-point injection; supports structured-vs-random distinction.
- **Akula et al., *Radar-Guided Camera Verification for AEB*** — arXiv:2606.27556. 3/3 no-threat sessions braked at high recall; availability-cost evidence.
- **Gan et al., IWIS 2026** — DOI `10.1109/IWIS70716.2026.11667732`; **Cao et al., IEEE IV 2026** — DOI `10.1109/IV66570.2026.11623995`; **Wu et al., IEEE Sensors 2025** — DOI `10.1109/SENSORS59705.2025.11330396`.
- **Dusza et al., *Radar Sensor Mounting Height*** — *Sensors* 26(16):5060, 2026, DOI `10.3390/s26165060`. 20–40 cm mounting height intensifies multipath.
- **Liu et al., *Automatic Ghost Noise Labeling for 4D mmWave Radar*** — *Remote Sensing* 17(22):3732, 2025, DOI `10.3390/rs17223732`.
- Rain/interference/4D: **Su et al., IET RSN 2025** DOI `10.1049/rsn2.70008`; **Engert et al., RadarConf 2026** DOI `10.1109/RadarConf2663773.2026.11674905`; **López-Valcárcel & García Sánchez, RadarConf25** DOI `10.1109/RadarConf2559087.2025.11205000`; **Kim et al., NLoS pedestrian** arXiv:2508.04033.

### What remains unclaimed
1. Structured, persistent **radar multipath ghosts → closed-loop driving cost**, with exact labels.
2. **`p(crash | phantom brake)` for radar multipath**, with a follower. The LiDAR mirror paper does the generic chain.
3. **Structured vs rate-matched random** under identical tracking/control. Our sensor-level result is still unique.
4. **Oracle-filter ceiling** for radar ghosts.
5. A **CARLA-native physics multipath model calibrated to RGD and used closed-loop**.
6. **Label-audited, risk-controlled closed-loop cost.** Ali audits labels; nobody carries it to control.

---

## 8. Final claims table

| Claim | Status | Evidence |
|---|---|---|
| Calibrated CARLA radar matches real core errors | **supported** | Stage C v8, 0.836 vs 0.792 m range σ |
| Synthetic ghost data can fine-tune to real RGD | **partially supported** | v7s 0.787 vs real ceiling 0.819 |
| Structured false positives cost more than random at matched rate | **supported at sensor level** | 47.27% vs 0.38%, 124× |
| A perfect filter has a measurable closed-loop ceiling | **not established** | oracle ≡ ghosts-off; effect below old noise floor |
| Ghost filtering helps/hurts control | **undecided** | +0.46% (p=0.17) vs −0.33% (p=0.23) |
| `p(crash | phantom brake)` is measurable | **measured (negative)** | 0/453, 95% CI 0–0.84% |
| Phantom braking causes rear-end crashes at calibrated rates | **not observed** | 0/453; upper bound 0.84% |
| Old 10-seed table resolves the ghost effect | **rejected** | 56–320 runs/arm needed for 5/km |

**Can conclude:** calibrated sensor; structured-vs-random; old table underpowered; 453 phantom episodes produced no rear-end crash, upper bound 0.84%; missing regulatory quantity is the allowable FP rate.

**Cannot conclude:** that a learned filter improves driving; that a perfect filter has no value; that ghosts cause crashes at calibrated rates; that CARLA ghost data transfers zero-shot; that the old MLP/transformer/CNN ranking is real.

---

## 9. Key file paths and artifacts

### Local
- `/home/vaibhav/carla-claude/FINAL_REPORT.md`
- `/home/vaibhav/carla-claude/FINAL_PCRASH_SUMMARY.csv`
- `/home/vaibhav/carla-claude/FINAL_REQUIREMENTS.csv`
- `/home/vaibhav/carla-claude/carla4/artifacts/final/` (copied)

### VM repo
- `carla4/artifacts/final/FINAL_REPORT.md`
- `carla4/artifacts/final/comparison_pcrash/summary.csv`
- `carla4/artifacts/final/comparison_pcrash/per_run_metrics.csv`
- `carla4/artifacts/final/comparison_pcrash/cdf_s5.png`
- `carla4/artifacts/final/requirements.csv`
- `carla4/artifacts/final/structured_vs_random_5seeds.txt`
- `carla4/artifacts/final/counterfactual_tf_ghost_2000.json`
- `carla4/artifacts/final/counterfactual_tf_clean_on_ghostdata_2000.json`
- `carla4/artifacts/final/stage_c/`
- `carla4/scenarios/results_final_pcrash_*` (raw CSVs + sidecars)
- `carla4/scenarios/results_final_smoke`, `results_final_pilot_*`
- `carla4/artifacts/final/model_matched_015b.json`
- `/tmp/final_run/` (clean work copy; lost on reboot)

### Code
- `carla4/scenarios/follower.py`
- `carla4/scenarios/metrics.py`
- `carla4/scenarios/ground_truth_logger.py`
- `carla4/scenarios/s5_ghost_exposure.py`
- `carla4/scenarios/analyze_results.py`
- `carla4/scenarios/requirements.py`
- `carla4/scenarios/power_analysis.py`
- `carla4/scenarios/make_final_report.py`
- `carla4/run_final_study.sh`
- `carla4/radar/tests/test_closed_loop_metrics.py`

---

## 10. Reproduce

```bash
cd carla4
python3 -m compileall .
python3 -m unittest discover -s radar/tests -p 'test_*.py'   # 154 pass
python3 structured_vs_random_artefacts.py --scans 3000 --seeds 42 43 44 45 46
python3 acceptance_test.py --model-dir model_mlp_clean
cd scenarios
python3 analyze_results.py --runs old=results_A_mlp_clean_ghost --out comparison_old
python3 power_analysis.py comparison_all/per_run_metrics.csv --out-dir comparison_all/power
python3 requirements.py --out ../artifacts/final/requirements.csv
# CARLA 0.9.16 on :2000, TM :8050
../run_final_study.sh pcrash
../run_final_study.sh pcrash_close
../run_final_study.sh analyze
```

Use `carla4/artifacts/final/model_matched_015b.json` as `--radar-config` for the shipped MLP/transformer/CNN models.

---

## 11. What to do next (if the project continues)

1. **Fix arm C** (learned ghost filter): runtime filter is OOD — it runs on one cluster per object before point expansion and omits `amplitude_gain_db`. Either make it consume expanded points or retrain on clusters. Then run A/C/D.
2. **Strengthen the follower experiment**: shorter gap (5–8 m), slower reaction (1.5–2.0 s), or a deliberately inattentive follower. The current 0/453 is a bounded negative at 10–20 m, not a universal zero.
3. **Power the ghost effect**: 56–320 runs/arm for 5/km; use continuous endpoints, training seeds, pre-registration.
4. **Measure real-world ghost exposure** — the load-bearing unknown.
5. **Camera+radar follow-on**: “is the camera already a free ghost filter?” (2×3 grid), but this is a new project, not the current close.
6. **Commit the current tree** after deciding whether the PCLA agent deletions are permanent. The working tree currently has ~2,117 changes.
7. **Reboot / fsck if the VM’s absolute-path stale-dentry issue persists**; it is a filesystem quirk, not a code issue.

---

## 12. One-paragraph summary

This session converted the project from an underpowered “does filtering help?” study into a defensible measurement paper. The radar model was already calibrated against nuScenes; we added a curvature-aware, staging-aware, follower-aware measurement layer, ran the missing S5 rear-end experiment, and bounded `p(crash | phantom brake)` at **0.84%** from **453 phantom episodes with zero observed rear-end collisions**. The strongest positive result remains sensor-level: a structured multipath ghost reaches the controller **124× more often** than rate-matched random clutter. The old 10-seed closed-loop table is underpowered and should not be used for ranking. The honest close is: calibrated instrument + structured-vs-random result + bounded negative on crash cost + requirements arithmetic; the learned-filter benefit remains undecided.
