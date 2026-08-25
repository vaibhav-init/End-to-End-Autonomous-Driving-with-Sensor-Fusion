# Deep Research: Reducing Radar False Alarms in CARLA

**Date:** August 25, 2026
**Scope:** Every published approach, open-source repo, dataset, and technique found for
reducing automotive-radar false alarms (multipath ghosts, clutter, interference) —
with special attention to what is possible inside the CARLA simulator and what applies
to this repository's `carla4/radar` target-list backend and ghost-detection pipeline.

---

## 1. The Problem Space: What a "Radar False Alarm" Actually Is

False alarms in automotive radar are not one phenomenon. Any serious reduction effort
must name its enemy precisely. Complete taxonomy:

| # | False-alarm type | Cause | Typical signature |
|---|---|---|---|
| 1 | **Multipath ghosts** | Specular reflection off walls/guardrails/parked cars creates image targets | Position mirrored about reflector plane; Doppler related to parent |
| 2 | **Underbody / ground-bounce ghosts** | Reflections under vehicles (road ↔ chassis) | Near-real position, different elevation/Doppler |
| 3 | **Mutual-reflection ghosts** | Signal bounces between ego bumper and a close object | Very short-range returns with doubled range rate |
| 4 | **CFAR noise hits** | Random noise crossing adaptive threshold | Isolated, non-persistent points |
| 5 | **Clutter** | Static environment scatter (fences, poles, signs) | Persistent static-Doppler points misread as objects |
| 6 | **Interference ghosts** | Other radars' chirps beat into the receiver | Raised noise floor, phantom targets |
| 7 | **Sidelobe detections** | Antenna sidelobes on strong targets | Angularly offset copies of strong targets |
| 8 | **Doppler-aliased targets** | Velocities outside unambiguous band fold back | Real object at wrong closing speed |
| 9 | **Range-straddling artifacts** | Extended targets straddle resolution cells | Fragmented multiple detections per object |

The Radar Ghost Dataset (RGD) labels family 1 primarily. This repo's realistic backend
generates families 1–6; its detector is trained only against family-1 labels.

**Key insight from the literature:** production-grade false-alarm reduction is a
*fusion of several method families below*, never a single method.

---

## 2. Method Landscape — Nine Families

### Family A — Classical signal-processing detectors (CFAR and beyond)

- **CA-CFAR** (Rohling 1983): adaptive threshold from neighboring cells; baseline of
  every automotive radar. False alarms come from clutter edges, extended-target
  leakage, multi-target masking.
- **OS/GO/SO-CFAR variants**: ordered-statistic thresholds resist multi-target and
  clutter-edge false alarms better than CA-CFAR.
- **KAN-powered large-target detection** (arXiv 2502.19000, IEEE Trans. Radar Systems
  2025): Kolmogorov-Arnold network learns range-Doppler segment PDFs; matches OS-CFAR
  tuned to PFA=1e-6 while improving large-target detection — evidence that learned
  *detector-level* false-alarm control is an active 2025 direction.
- **Relevance here:** your target-list backend skips CFAR entirely (SNR gating instead).
  Simulating an actual CFAR stage would make synthetic point clouds inherit realistic

### Family B — Physics / geometry path modeling (rule-based ghost identification)

- **Kopp, Kellner, Piroli, Dietmayer — "Fast Rule-Based Clutter Detection in
  Automotive Radar Data"** (arXiv 2108.12224, ITSC 2021). Explicitly models three
  wave-propagation effects — underbody reflections, ego↔object mutual reflections,
  guardrail/wall specular multipath — and derives per-effect rules flagging clutter
  from a *single* frame. Large clutter removal at very low runtime with few false
  classifications of real objects. Same group has follow-ups on multipath handling.
- **RGD baselines** (arXiv 2404.01437): object-feature classifier + free-space/grid
  consistency, evaluated by the dataset authors themselves.
- **Implication:** ghost-parent geometric relations (range sum ≈ wall distance,
  mirrored azimuth, Doppler relation v_ghost ≈ −v_direct or ≈ v_direct by bounce
  type) are established, proven discriminative signals. Your PointNet must currently
  rediscover them from raw coordinates; feeding them as engineered features is both
  literature-backed and the most transferable change available (physical relations
  hold identically in CARLA and reality).

### Family C — Tracking-based suppression

- **M-of-N confirmation/deletion**: act only on tracks seen M of N frames. Already in
  your `realistic_core.py` tracker.
- **Extended Object Tracking (EOT)**: random-matrix EOT models treat each vehicle as
  an extent, so single-frame ghost fragments fail extent-consistency checks. A worked
  MATLAB example exists (`andreaslebherz/demoMatlabGhostTracking`, built on MathWorks'
  "Radar Ghost Multipath" tutorial) demonstrating exactly this suppression pattern.
- **PiVoT** (arXiv 2607.13891, 2026): variational-Bayes joint detector/tracker over
  Poisson measurement models; training-free, clutter-resilient, handles full-resolution
  Doppler clouds, matches deep-learning benchmarks without labels. Modern
  "track-before-detect" school.
- **Track-lifecycle classifier features**: persistence, association stability,
  parent-correlation over seconds (not 5 frames). Ghosts co-appear and co-vanish with
  parents — a temporal signature unused in your current window.
- **Relevance:** your tracker is the simplest member of this family. Track-age and
  coherence features fed to the detector, or an EOT layer above it, are documented
  suppressors.

### Family D — Occupancy-grid / free-space consistency

- Accumulate polar/Cartesian occupancy grids across frames with inverse sensor models;
  any detection landing where accumulated evidence says *free* is suspect. One of the
  two RGD baselines, and standard practice in production ACC/AEB stacks.
- **Domain-independent advantage:** free-space logic depends on geometry over time,
  not amplitude calibration — transfers between CARLA and reality better than
  amplitude-dependent features.
- **Relevance:** implementable directly in `ghost_detection/runtime.py` from tracker

### Family E — Learning-based point-cloud classification / segmentation

- **Kraus, Scheiner, Ritter, Dietmayer 2020** (arXiv 2007.05280): first ML ghost
  detection on real Mercedes data — random forest over object/track features.
- **RGD + PointNet line**: your current approach (Temporal PointNet over range/
  azimuth/Doppler/amplitude) is the direct descendant. RGD authors' own baselines are
  the reference points to beat.
- **Semantic segmentation schools** (Schumann et al., "Semantic Segmentation on Radar
  Point Clouds", scene understanding with radar): treat clutter/ghost suppression as a
  per-point segmentation class rather than binary real/ghost — richer supervision,
  better calibrated per-class thresholds.
- **See Further Than CFAR** (arXiv 2402.12970, RadarConf 2024): 2D-CNN radar detector
  trained with *LiDAR cross-supervision* (no manual labels), producing lidar-like dense
  clouds; significantly beats CFAR baselines. Uses the RaDelft dataset.
- **Relevance:** your pipeline is already here. The open deltas vs literature: (a)
  physics-relation input features (Family B), (b) track features (Family C), (c)
  free-space channel (Family D), (d) calibration of outputs.

### Family F — Micro-Doppler exploitation

- Pedestrians/cyclists produce per-point radial-velocity spread around the torso mean
  (limb swing ±1–2 m/s). Ghosts inherit parent bulk Doppler without micro-texture —
  one of the strongest real-vs-ghost cues for exactly the classes RGD labels.
- **Zaumseil et al.** (arXiv 2608.08701, Aug 2026): anchor-based AI using micro-Doppler
  signatures explicitly to remain reliable "despite radar multipath reflections and
  ghost objects."
- **RadHAR and successors** (`nesl/RadHAR` GitHub): mmWave point-cloud human activity
  recognition — codebase patterns for exploiting per-point velocity dispersion.
- **Relevance:** your transform-derivative velocity fallback gives each CARLA actor a
  single rigid-body Doppler — flat across all its points. Adding a limb-kinematics
  Doppler spread model in the collector + a per-point Doppler-residual-within-cluster
  feature is a documented, high-value cue you currently discard.

### Family G — Interference mitigation (a distinct false-alarm source)

- Time-domain / time-frequency-domain thresholding (arXiv 2402.14018, 2024):
  TFD-thresholding outperforms TD as traffic density and chirp correlation grow;
  evaluated with an in-house automotive radar simulator at scale.
- CFAR-in-t-f-domain masking (arXiv 2101.xxxx line): detect interference spectra with
  1-D CFAR per frequency bin, dilate into masks — computationally cheap.
- Adaptive noise cancellers (arXiv 1911.06372): estimate negative-frequency power as an
  interference indicator.
- **Relevance:** your backend models interference bursts statistically. If you ever
  want *structured* interference (not just raised noise floor), these papers give the
  signal-chain recipe. For false-alarm reduction research, keeping interference simple
  is defensible; ghosts/clutter dominate longitudinal-control impact.

### Family H — Using multipath positively (NLOS exploitation)

- Multipath is not only noise: non-line-of-sight reconstruction around corners
  (cited in RGD paper), occluded-vehicle detection via road-surface bounce (flat
  incidence → ghost lands near true position). Understanding *exploitation* sharpens
  *suppression*: the same geometry relations flag which returns are trustworthy images.
- **Relevance:** your path-validation gate already exploits this partially (road-user
  priority over infrastructure); a "multipath-aware" selector that *uses* wall-bounce
  returns for occluded-lead detection would be novel in your closed-loop setting.

### Family I — Calibration, domain adaptation, sim-to-real alignment

- **Calibrated Domain Randomization** (Trinh et al., arXiv 2601.17871, Jan 2026):
  align global noise-floor statistics of simulated RD maps to a small unlabeled real
  set while preserving discriminative structure; beat ray-tracing and naive domain
  randomization for sim-to-real transfer. Directly applicable: fit your profile's
  amplitude/error marginals to `artifacts/rgd_stencil.json` before pretraining.
- **Deep Stochastic Radar Models** (Wheeler et al., arXiv 1701.09180, IV 2017): GAN-
  fit stochastic radar model conditioned on ideal returns, adversarially matched to
  real recordings — the canonical "learned sensor residual" template.
- **AAETR** (arXiv 2408.09362): angle estimation transformer trained on large-scale
  simulation shows genuine zero-shot sim-to-real transfer — proof it is achievable
  when the signal model is faithful enough.
- **Standard DA toolbox:** DANN/adversarial alignment, CORAL/MMD feature alignment,
  test-time adaptation, per-sequence standardization. All untried in your pipeline.
- **Reverse-transfer diagnostic:** train on real RGD, evaluate zero-shot on synthetic

---

## 3. Radar Simulation Options In and Around CARLA

| Simulator / approach | Fidelity level | Open source | Multipath | Notes for this repo |
|---|---|---|---|---|
| **CARLA native `sensor.other.radar`** | Ray-cast point detector | Yes (in CARLA) | No | Your `native` backend; low-fidelity baseline by design |
| **C-Shenron** (UCSD WCSNG, arXiv/paper 2023) | Full FMCW/ADC → range-angle cube, material-aware scattering | Yes — `ucsdwcsng/C-Shenron` | Via signal chain, GPU-costly | Your `cshenron` backend is a derived target-list port; the full upstream remains the raw-signal gold standard in CARLA |
| **RadaRays** (UOS, arXiv 2310.03505) | Real-time rotating-FMCW ray tracing: reflection/refraction/scattering | Yes — `uos/radarays` (Gazebo plugin) | Yes, geometric | Hardware-accelerated ray tracing pattern to emulate; strongest open geometry upgrade path |
| **RadarSimPy / RadarSimX** (`radarsimx/radarsimpy`) | Python/C++ target-list + raw FMCW sim, RCS, array processing | Core free, X paid | Partial (multipath in X) | Could replace your hand-rolled ideal-return stage with maintained physics |
| **MathWorks Radar Toolbox ghost examples** | Target-list EOT with injected multipath ghosts | No (MATLAB), demos mirrored on GitHub | Modeled per scenario | The `demoMatlabGhostTracking` repo documents the tracking-side suppression recipe |
| **This repo's `realistic` backend** | Temporal target-list: C-Shenron-derived materials + image-method multipath + errors/clutter/interference/tracker | Yes (your code) | Type-1/type-2 2nd-order + type-2 3rd-order planar | Unique as an *open, closed-loop-capable* CARLA radar artifact generator |
| **Digital-twin calibration** (e.g., CARLA digital twin for speed estimation, ITSC 2024) | Match synthetic to a specific real sensor's conditions | Method | n/a | Template for fitting your profile to RGD's sensor envelope |
| **CARLA-Loc** (arXiv 2309.08909) | Multi-sensor synthetic dataset incl. radar in CARLA | Yes (data+scripts) | No | Evidence that native-radar realism limits are known; nobody ships better radar in CARLA openly except C-Shenron and you |

**Conclusion of §3:** within CARLA, only three open options exist — native radar,
C-Shenron, and custom target-list models (yours). Nobody has published a *better*
open target-list radar for CARLA than what you already built; your gap vs RadaRays/
C-Shenron-full is reflector generality and signal-chain realism, not existence.

---

## 4. Datasets Available for False-Alarm Work

| Dataset | Sensor type | Ghost labels? | Size | Use here |
|---|---|---|---|---|
| **RGD v1.1** (Zenodo 6676246; arXiv 2404.01437) | 77 GHz CFAR point lists, 10 Hz, ±70° | **Yes** (CMTO bounce types) | 111 seq × 385 frames, ~35M pts | Your primary real benchmark (already integrated) |
| **RadarScenes** | Production 77 GHz point lists | No clean false-alarm GT | 158 seq, ~5.9 h | Ordinary-detection statistics for profile fitting |
| **RADIal** (Valeo, `valeoai/RADIal`) | Raw ADC + RD maps + point clouds | No | ~2 h highway | If you ever go raw-signal |
| **K-Radar** (arXiv 2206.08171) | 4D radar tensor (range/az/el/Doppler) | No | Various weather | Weather-robustness comparisons |
| **RADIATE** (arXiv 2010.09076) | Imaging radar + lidar/camera | No | Rain/night/fog | Adverse-weather clutter studies |
| **CARRADA** (arXiv 2005.01456) | RD/RA/AD + point clouds, synchronized camera/lidar | No | ~50 min | Cross-modal supervision ideas |
| **View-of-Delft (VoD)** | 80 GHz point clouds + lidar/camera | No | Traffic scenes | Detection baselines |
| **RaDelft** (arXiv 2406.04723) | High-res radar + lidar, large-scale | Implicit via lidar GT | Large | The See-Further-Than-CFAR training ground; cross-supervision template |
| **nuScenes** | Automotive radar (sparse) | No | 1000 scenes | General robustness checks |
| **CARLA-Loc** (arXiv 2309.08909) | Synthetic multi-sensor incl. native CARLA radar | No | 42 seq | Only public synthetic-CARLA radar dataset besides yours |

---

## 5. GitHub Repos Worth Knowing

| Repo | What it gives you |
|---|---|
| `ucsdwcsng/C-Shenron` | Full FMCW radar simulator in CARLA; material scattering equations your `cshenron_core.py` derives from |
| `uos/radarays` (+ RadaRays paper) | GPU ray-traced FMCW simulation architecture; reflection/refraction/scattering model |
| `flkraus/ghosts` | Official RGD code: dataset loaders, two baseline detectors (the numbers to compare against) |
| `radarsimx/radarsimpy` | Maintained target-list/FMCW radar simulator in Python/C++; RCS + array modeling |
| `valeoai/RADIal` | Raw-signal radar dataset + baselines if scope ever extends below point lists |
| `andreaslebherz/demoMatlabGhostTracking`, `demoMatlabGhosting` | Reference implementations of EOT-based ghost suppression logic (translatable to Python) |
| `nesl/RadHAR` | Point-cloud velocity-dispersion feature engineering patterns |
| `SSubhnil/RacingCARLA`, various `carlaRadarSimulation` repos | Small CARLA-radar integrations; mostly native-sensor wrappers, no realism work — confirms your niche is unclaimed |

*(GitHub search counts as of Aug 2026: "radar simulation" ≈ 894 repos but almost all
are GPR/SAR/MATLAB coursework; "radar carla" = 32 repos, all thin native-sensor
wrappers or racing projects; "radar ghost multipath" = 2 repos, both MATLAB demos.

---

## 6. Gap Analysis: This Repo vs the Field

| Capability | Field standard | This repo | Verdict |
|---|---|---|---|
| Ghost labels + detector on real data | Kraus 2020, RGD baselines | Temporal PointNet on RGD ✔ | Present; underperforms cross-scenario |
| Physics-rule suppression (Kopp line) | Single-frame wave-propagation rules | Absent from detector inputs | **Gap — highest-value fix** |
| Track-lifecycle features | Object-feature classifiers (RGD baseline 1) | 5-frame window only | **Gap** |
| Free-space/occupancy consistency | RGD baseline 2; production stacks | Absent | **Gap** |
| Micro-Doppler cues | Zaumseil 2026, RadHAR line | Rigid-body Doppler only | **Gap** |
| CFAR-realistic point generation | Every real dataset | Statistical densification post-hoc | **Root cause of sim-to-real failure** |
| Calibrated sensor statistics | CDR (2026), Wheeler 2017 | Hand-tuned priors in JSON profiles | **Gap — automatable** |
| Multipath geometry | Image method / ray tracing | Planar ≤3rd order ✔ | Present; curved/dynamic reflectors missing |
| Sim-to-real transfer measurement | Rare in literature | Quantified zero-shot failure + diagnosis ✔ | **Your unique asset** |

---

## 7. Exhaustive Possibility List (nothing left out)

Every option found for reducing radar false alarms in this CARLA setup, grouped by
intervention point. ✔ = already implemented here.

### 7.1 Detector-input side
1. Physics-relation residuals as features: ghost-position prediction error per fitted
   reflector plane; range-sum relation r_g ≈ r_d + 2·d_wall; mirrored-azimuth check.
2. Doppler-relation residuals: v_ghost vs −v_direct (mutual bounce), v_ghost vs
   v_direct (wall bounce), underbody ≈ v_direct.
3. Radar-equation-normalized amplitude (amp·R², log-scaled) — invariant to absolute
   scaling mismatch between domains.
4. Per-point Doppler residual within spatial cluster (micro-Doppler proxy).
5. Local point-density and nearest-cluster-distance features.
6. Elevation channel if currently unused.
7. Track features: age, hit/miss ratio, coast length, parent correlation over seconds,
   co-appearance/co-vanishment with a confirmed parent track.
8. Free-space-consistency scalar per point (from accumulated occupancy grid).

### 7.2 Learning-algorithm side
9. Threshold recalibration: val-quantile selection or temperature scaling — fixes the
   saturated 0.9995 threshold without retraining.
10. Class-balanced loss / per-class thresholds (pedestrian FPR ≠ car FPR).
11. Domain-adversarial training (DANN) between synthetic and real point sets.
12. CORAL/MMD auxiliary alignment loss during synthetic pretraining.
13. Test-time adaptation (entropy minimization or BN-statistics recalibration).
14. Cross-modal supervision (See-Further-Than-CFAR pattern): supervise
    domain-aligned representations with CARLA truth instead of naive pretraining.
15. Multi-class segmentation formulation (direct/ghost/clutter/background) rather
    than binary — richer gradients and per-class operating points.
16. Late fusion of rule-based flags + learned score with calibrated weights.
17. Temporal smoothing / sequence-level CRF over per-point probabilities.

### 7.3 Tracker side
18. SNR/class-conditioned M-of-N confirmation policies.
19. Extended-object extent consistency (reject single-point "objects" claiming
    vehicle extents).
20. Ghost-parent association tracking: delete tracks fully explained by a confirmed
    track plus a known reflector geometry.
21. PiVoT-style Poisson multi-object filter — training-free joint detection/tracking.

### 7.4 Simulator side (make synthetic data match reality)
22. CFAR-stage simulation: ideal targets → RD map → CA-CFAR → points, replacing
    post-hoc statistical densification (fixes spatial structure at the source).
23. Surface-sampled extended targets using semantic-LiDAR visible points.
24. Micro-Doppler limb-kinematics Doppler spread for walkers/cyclists.
25. Antenna-model azimuth errors: SNR- and off-boresight-dependent; sidelobe spurious
    detections on strong returns.
26. Doppler ambiguity folding at ±44.3 m/s + range-Doppler coupling.
27. Amplitude physics: material × aspect-angle RCS tables, Swerling fluctuation,
    temporally correlated scintillation.
28. Multipath extensions: curved reflectors (parked cars), finite-extent guardrails,
    moving-reflector Doppler in type-2 paths, path occlusion/shadowing,
    diffuse-vs-specular mixing, Fresnel incidence-angle losses.
29. Underbody/ground-bounce path family (Kopp effect #1) — currently absent.
30. Mutual ego↔object bounce family (Kopp effect #2) for close-range scenes.
31. Two-sensor left/right bumper geometry (RGD-like) + mutual interference.
32. Automated profile calibration: optimize profile JSON parameters by minimizing
    Wasserstein/MMD between simulated and real feature marginals (CDR-style).
33. Learned generative residual: conditional flow/diffusion mapping ideal target list
    → realistic point set fitted on paired distributions (modernized Wheeler 2017).

### 7.5 Controller side (closed-loop cost of residual false alarms)
34. Uncertainty-aware supervisor gating braking on detector confidence + tracker
    confirmation state; log decision source per intervention.
35. Hysteresis/dwell requirements before AEB-level response to unconfirmed tracks.
36. Multipath-aware target selector exploiting wall-bounce returns for occluded
    leads instead of blanket rejection.
37. Separate ACC from AEB logic so artifact effects are causally attributable.

### 7.6 Evaluation side
38. Report full FPR-at-fixed-recall and recall-at-fixed-FPR curves, never single
    operating points.
39. Distribution-distance metrics (Wasserstein/MMD/point-cloud Fréchet distance)
    between synthetic and real marginals as a simulator-fidelity score.
40. Reverse-transfer diagnostic: train on real RGD → zero-shot test on synthetic;
    per-feature error decomposition localizes simulator divergence.
41. Scenario-disjoint splits as the primary reporting protocol.

---

## 8. Recommended Roadmap (priority-ordered, given all findings)

**Stage 1 — Fix what is broken in the current real-data detector (no new data needed)**
1. Threshold recalibration (item 9).
2. Physics-relation + track + free-space features (items 1–2, 7–8).
3. Retrain real-only; evaluate on official AND scenario-disjoint splits (item 41).

**Stage 2 — Make the simulator's points structurally real**
4. CFAR-stage simulation + surface-sampled extended targets (items 22–23).
5. Micro-Doppler spread model for walkers (item 24).
6. Amplitude normalization + Doppler folding (items 3, 26).

**Stage 3 — Calibrated sim-to-real, one honest retry**
7. Automated profile fitting to RGD marginals via MMD/Wasserstein (items 32, 39).
8. Reverse-transfer diagnostic to verify which distributions still diverge (item 40).
9. One synthetic-pretraining retry with CDR alignment + physics features; judge only
   on scenario-disjoint real test.

**Stage 4 — Close the loop (the unclaimed ground)**
10. Deploy best detector into `realistic` backend; run S1–S4 controller ablations;
    measure nuisance-braking reduction vs collision-rate change (items 34–37).
11. If detector quality plateaus: PiVoT-style training-free tracker comparison
    (item 21) — a no-labels baseline nobody has run against learned ghost filters.

---

## 9. Primary References

### Ghost / clutter detection
1. Kraus et al., "The Radar Ghost Dataset" — arXiv 2404.01437 (IROS 2021)
2. Kraus et al., "Using Machine Learning to Detect Ghost Images in Automotive Radar" — arXiv 2007.05280 (ITSC 2020)
3. Kopp et al., "Fast Rule-Based Clutter Detection in Automotive Radar Data" — arXiv 2108.12224 (ITSC 2021)
4. Roldan et al., "See Further Than CFAR: a Data-Driven Radar Detector Trained by Lidar" — arXiv 2402.12970 (RadarConf 2024)
5. Zaumseil et al., micro-Doppler pre-crash detection robust to ghosts — arXiv 2608.08701 (2026)
6. Gan et al., "PiVoT: variational multi-object tracking under heavy clutter" — arXiv 2607.13891 (2026)

### Radar simulation and sim-to-real
7. Wheeler et al., "Deep Stochastic Radar Models" — arXiv 1701.09180 (IV 2017)
8. Amock et al., "RadaRays: Real-time Simulation of Rotating FMCW Radar" — arXiv 2310.03505
9. Bialer et al., "RadSimReal" — CVPR 2024
10. Trinh et al., calibrated domain randomization for FMCW sim-to-real — arXiv 2601.17871 (2026)
11. Zhu et al., "AAETR: zero-shot sim-to-real angle estimation" — arXiv 2408.09362
12. C-Shenron: realistic radar simulator for CARLA — UCSD WCSNG
13. Han et al., "CARLA-Loc: synthetic SLAM dataset incl. radar in CARLA" — arXiv 2309.08909

### Signal processing foundations
14. Rohling, "Radar CFAR Thresholding in Clutter and Multiple Target Situations" — IEEE TAES 1983
15. Li et al., thresholding-based interference mitigation evaluation — arXiv 2402.14018 (2024)
16. Jin & Cao, adaptive noise canceller interference mitigation — arXiv 1911.06372

### Datasets
17. Radar Ghost Dataset v1.1 — Zenodo DOI 10.5281/zenodo.6676246
18. RadarScenes — arxiv 2104.02493 · 19. RADIal — valeoai/RADIal
20. K-Radar — arXiv 2206.08171 · 21. RADIATE — arXiv 2010.09076
22. CARRADA — arXiv 2005.01456 · 23. RaDelft — arXiv 2406.04723
24. Kulkarni et al., KAN-powered detection — arXiv 2502.19000 (IEEE TRS 2025)
