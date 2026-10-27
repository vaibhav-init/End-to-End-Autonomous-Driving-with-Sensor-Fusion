# Final report — radar artefacts, calibration and closed-loop cost

Generated from measured artefacts in `carla4/artifacts/final/`. Every table is reproducible from the listed files.

## Bottom line

- The CARLA radar model matches real matched-pair core errors (range σ 0.836 vs 0.792 m; velocity σ 2.898 vs 2.771 m/s).
- A structured multipath ghost reaches the controller **124× more often** than rate-matched random clutter at the same false-detection rate.
- **453 phantom-brake episodes produced 0 rear-end collisions** with a 10–20 m follower; the 95% upper bound on `p(crash | phantom)` is **0.84%**.
- The old 10-seed ghost table is underpowered by roughly an order of magnitude; it cannot support a ghost-effect ranking.
- Whether a learned filter improves driving is still undecided: the open-loop effect is small and not statistically significant in either direction.

## 1. Claims status

| claim | status | evidence |
|---|---|---|
| Calibrated CARLA radar matches real core errors | **supported** | Stage C v8, range σ 0.836 vs 0.792 m, velocity σ 2.898 vs 2.771 m/s |
| Synthetic ghost data can fine-tune to real RGD | **partially supported** | v7s fine-tune AUROC 0.787 vs real ceiling 0.819; measured physics alone 0.650 |
| A structured false positive costs more than random at matched rate | **supported at sensor level** | 47.3% vs 0.38% controller selection, 124x |
| A perfect filter has a measurable closed-loop ceiling | **not established** | oracle ≡ ghosts-off; effect below old harness noise floor |
| Ghost filtering helps/hurts control | **undecided** | open-loop +0.46% (p=0.17) ghost-trained, −0.33% (p=0.23) clean |
| p(crash | phantom brake) is measurable | **measured (negative)** | 0/453 rear-end crashes; 95% CI 0–0.84% |
| Phantom braking causes rear-end crashes at calibrated rates | **not observed** | 0/453 phantom episodes; upper 95% bound 0.84% |
| The old 10-seed table resolves the ghost effect | **rejected** | power analysis requires 56–320 runs/arm for 5/km |

## 2. Sensor-model validation (Stage C v8)

CARLA 0.9.16 `realistic` backend, `rgd_regime_v1`, 180 s at 20 Hz, Town10HD_Opt, seed 7. Matched-pair errors against a lead vehicle; real reference is nuScenes mixed-front.

| metric | real | sim | delta |
|---|---:|---:|---:|
| range error std (m) | 0.792 | 0.836 | 0.044 |
| velocity error std (m/s) | 2.771 | 2.898 | 0.127 |
| velocity misassignment | 0.527 | 0.573 | 0.046 |
| ghosts / scan | 87.2 | 299.4 | 212.2 |

Selected-target core sigma (latency-aligned): range **0.266 m**, velocity **0.438 m/s**; KS range **0.159**, velocity **0.322**.

**Claim:** the calibrated sensor matches the real core error distribution on matched pairs. It does not match real ghost *density* because the staged scene has one lead while nuScenes streets have dozens of annotated objects. That is a scene-composition difference, not a sensor calibration result.

## 3. Sim-to-real transfer (RGD v1.1 official test)

| fine-tune source | AUROC | AUPRC | best F1 |
|---|---:|---:|---:|
| v7s fine-tune (project record) | 0.787 | 0.226 | 0.408 |
| real-only ceiling (project record) | 0.819 | 0.329 | 0.424 |
| ft_meas_combined | 0.376 | 0.087 | 0.211 |
| ft_measloss | 0.650 | 0.194 | 0.277 |
| ft_ratio_matched | 0.511 | 0.157 | 0.225 |
| ft_t04_meas | 0.353 | 0.085 | 0.211 |

Real-data-trained ceiling from the project record: **AUROC 0.819 / AUPRC 0.329**. v7s fine-tuning reaches **0.787** AUROC, i.e. 96% of the real-data ceiling, without a real label. The measured-physics Town10 collection reaches 0.650; ratio-matched volume 0.511; three-town combined 0.376. Label-ratio match is necessary; ghost-event phenomenology is the binding constraint.

## 4. Structured vs random false positives

5 seeds x 3,000 scans, identical sensor, tracker, scene and seed; only the origin of the false detections changes.

| case | false detections / scan | reached controller |
|---|---:|---:|
| multipath ghost, in-path | 1.13 | **47.27%** |
| random clutter, rate 1.3 | 1.13 | **0.38%** |

**At a matched false-detection rate, the structured ghost reaches the controller 124.4x more often.**

The ratio factorises: placement (uniform FOV -> in-path) **0.38% -> 21.21%**, coherence (i.i.d. -> tracks a parent) **21.21% -> 47.27%**.

This is the project's cleanest claim: the driving cost of a false positive depends on its physical origin, not only its rate. The azimuth sweep is the control: the same ghost outside the path gate reaches the controller on ~3% of scans or less.

## 5. Open-loop counterfactual: does removing ghosts help?

2,000 logged windows; each scored full, with ghost-labelled points removed, and with an equal number of random direct points removed. Flips are scored against the teacher's future speed.

| model | ghost windows | ghost-removal flip | direct-removal flip | broke | fixed | accuracy change |
|---|---:|---:|---:|---:|---:|---:|
| ghost-trained transformer | 1524 | 1.25% | 0.59% | 6 | 13 | 0.46% |
| clean-trained transformer | 1524 | 0.72% | 0.46% | 8 | 3 | -0.33% |

For the ghost-trained model, removing ghosts repairs more decisions than it breaks (**+0.46%** brake accuracy) but the paired McNemar test is not significant (**p = 0.17**). For the clean-trained model the direction reverses (**−0.33%**, **p = 0.23**). Neither direction is statistically resolvable at 1,524 ghost windows. Open-loop sensitivity is therefore real but small and not a reliable basis for a filtering policy.

## 6. Power analysis of the existing closed-loop table

The old 10-seed arms cannot resolve the effect they were built to measure. Required runs per arm for a two-sample comparison, using the observed per-run standard deviation:

| arm | scenario | phantom/km mean | sd | runs/arm for 5/km | runs/arm for 10/km |
|---|---:|---:|---:|---:|---:|
| 0_mlp_clean_noghost | 5 | 22.05 | 11.73 | 87 | 22 |
| A_mlp_clean_ghost | 5 | 32.60 | 22.57 | 320 | 80 |
| A_tf_clean_ghost | 5 | 23.02 | 18.44 | 214 | 54 |
| A_cnn_clean_ghost | 5 | 36.66 | 9.86 | 62 | 16 |

**Conclusion:** a 10-seed arm was underpowered by roughly an order of magnitude. The project should not report a ghost-effect ranking from that table. It should report the measured effect, its confidence interval, and the number of seeds a future study would need.

## 7. Closed-loop rear-end risk: p(crash | phantom brake)

S5 free-driving exposure with a following vehicle. Each phantom brake episode is attributed a rear-end collision if the follower strikes the ego within 5 s.

| arm | runs | phantom brakes | rear-end crashes | p(crash | phantom) | 95% CI |
|---|---:|---:|---:|---:|---|
| close_off | 10 | 113 | 0 | 0.00% | 0.00%–3.29% |
| close_on | 10 | 100 | 0 | 0.00% | 0.00%–3.70% |
| d | 10 | 77 | 0 | 0.00% | 0.00%–4.75% |
| off | 10 | 79 | 0 | 0.00% | 0.00%–4.64% |
| on | 10 | 84 | 0 | 0.00% | 0.00%–4.37% |
| **pooled valid arms** | — | **453** | **0** | **0.00%** | **0.00%–0.84%** |

A run enters the denominator only if the follower stayed within 60 m of the ego. Runs where the follower lost the ego are reported but excluded: they cannot measure rear-end risk. Forward ego collisions are counted separately in `n_collisions`; they are not rear-end crashes. Every run logged one stale radar frame; that is recorded in `n_runs_with_stale_radar` and is treated as a caveat, not a reason to discard the run.

This is the missing denominator in the phantom-braking literature: FMVSS 127 limits the *magnitude* of a false activation (0.25 g) but not its *rate*. The measured conditional probability converts a phantom-brake rate into an allowable-rate calculation.

## 8. Requirements arithmetic

Required range `R = v*tau + v^2/(2a)` and allowable false-positive rate `C_target / p(crash | phantom)`, with `a = 6.867 m/s^2` (0.7 g), `C_target = 6.8e-7 /h` (Yu et al. T-ITS 2025).

| p(crash | phantom) | allowable phantom brakes /h | /km at 80 km/h |
|---|---:|---:|
| 0.01 | 0.000068 | 0.00000085 |
| 0.001 | 0.000680 | 0.00000850 |
| 0.0001 | 0.006800 | 0.00008500 |
| 1e-05 | 0.068000 | 0.00085000 |

At a measured 20 phantom brakes/km, the implied p(crash | phantom) needed to meet the target is ~3e-10 — an implausible value. The purpose of reporting the arithmetic is to show that the *rate* is the missing specification, not to claim the simulator meets it.

## 9. What this project can and cannot conclude

**Can conclude.** The CARLA radar model is calibrated on matched-pair core errors; synthetic ghosts can be tuned to transfer to RGD under matched label ratio and ghost-event phenomenology; false positives are not exchangeable — a structured multipath ghost reaches the controller orders of magnitude more often than rate-matched random clutter; the old 10-seed closed-loop table was underpowered; 453 phantom-brake episodes produced **zero rear-end collisions** (95% upper bound 0.84%) with a 10–20 m follower; and the missing regulatory quantity is the allowable false-positive *rate*, not the 0.25 g magnitude limit.

**Cannot conclude.** That a learned ghost filter improves driving; that a perfect filter has no value; that ghosts cause crashes at calibrated rates (the measured upper bound is 0.84%, not zero); that CARLA ghost data transfers zero-shot; that the old MLP/transformer/CNN ranking is real. Those claims require arm C, a properly powered design, and a more aggressive follower than the one staged here.

## 10. Reproduce

```bash
cd carla4
python3 -m compileall .
python3 -m unittest discover -s radar/tests -p 'test_*.py'
python3 structured_vs_random_artefacts.py --scans 3000 --seeds 42 43 44 45 46
python3 acceptance_test.py --model-dir model_mlp_clean
cd scenarios
python3 analyze_results.py --runs old=results_A_mlp_clean_ghost --out comparison_old
python3 power_analysis.py comparison_all/per_run_metrics.csv --out-dir comparison_all/power
python3 requirements.py --out ../artifacts/final/requirements.csv
# after CARLA is up:
../run_final_study.sh pcrash
../run_final_study.sh analyze
```

## 11. Artefacts

- `stage_c/` — Stage C v8 real-vs-sim JSON/PNG
- `structured_vs_random_5seeds.txt` — current sensor-level result
- `counterfactual_tf_ghost_2000.json`, `counterfactual_tf_clean_on_ghostdata_2000.json`
- `comparison_all/power/` — power analysis of the old table
- `requirements.csv` — range/latency/false-positive budget
- `comparison_pcrash/` — S5 follower results, per-run metrics and plots

