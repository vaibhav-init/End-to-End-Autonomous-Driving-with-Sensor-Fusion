# Report: PCLA vs Custom MLP — Longitudinal Control Comparison on NHTSA Scenarios

A critical review of what this project builds, how it works, where it is weak, what
the state of the art does better, and exactly how to run everything.

---

## 1. Executive summary

The project compares two longitudinal-control "drivers" on four NHTSA-aligned CARLA
scenarios (lead stopped, lead decelerating, lead constant-speed, cut-in):

- **Driver A — PCLA `tfv6_visiononly`**: a pretrained CARLA Leaderboard agent
  (TransFuser V6 / "LEAD" family, vision-only variant) deployed via the PCLA framework.
- **Driver B — Custom MLP**: a small sequence MLP that predicts a *target speed* from
  stacked radar/vision + traffic-light features, converted to throttle/brake by a PID +
  brake-hold state machine. Steering for both is delegated to CARLA's `BasicAgent`.

Evaluation is ground-truth telemetry → CDFs of closest-approach distance, min TTC, peak
deceleration, plus collision rate. To make the dynamic scenarios (S2/S4) testable when the
agent cannot establish the situation itself, a **gap-keep "staging" controller** holds a
fixed follow gap and **hands control to the model at the critical moment**.

**The headline caveat:** the harness almost certainly *understates PCLA's true ability*
(see §5.2). And the overall methodology measures a **narrow, well-controlled slice** of
driving (longitudinal reaction from a staged state), not end-to-end driving — which is what
the modern benchmarks (Bench2Drive, CARLA Leaderboard 2) measure. This report explains
both, and what to do about it.

---

## 2. Goal and research questions

**Primary goal.** Quantify how a heavy pretrained end-to-end agent (PCLA) and a light
custom imitation model compare on *longitudinal safety* in controlled lead-vehicle
scenarios, under fog.

**Underlying thesis** (`carla4/method.md`): camera-only perception degrades in fog → the
vision model fails to brake; radar sees through fog → it brakes in time. Hence the recent
move to a **100 m radar** sensor for the MLP pipeline.

**Concrete research questions:**
1. Does the MLP match/beat PCLA on collision rate and closest-approach distance per scenario/fog?
2. How much does fog degrade each?
3. Is a cheap radar + small MLP competitive with a large vision-only end-to-end model for
   the *longitudinal* sub-problem?

---

## 3. System architecture (current state)

```
                         ┌─────────────────────────────────────────────┐
 DATA (imitation)        │  collect_throttle_brake_data.py (radar)      │
                         │  collect_vision_only_data.py   (YOLO vision) │
                         │  teacher = CARLA TM autopilot                │
                         │  label   = mean future ego speed (10 frames) │
                         └───────────────┬─────────────────────────────┘
                                         │ data.csv + dataset_config.json
                                         ▼
 TRAIN                    train_throttle_brake.py → TargetSpeedMLP
                                         │  model_throttle_brake/ (radar, 10 cols)
                                         │  model_vision_only/    (vision, 11 cols)
                                         ▼
 INFER / EVAL   scenarios/                drivers/
   s1..s4  ──── run_scenario() ────────► make_driver(--driver)
   (NHTSA)        │  spawns NPC,           ├─ pcla_driver  → PCLA.get_action() (throttle/brake)
   Town04         │  fog, GT logging       ├─ mlp_driver   → YOLO|radar → MLP → PID (throttle/brake)
                  │  staging (S2/S4)        └─ steering     → BasicAgent (lateral, both)
                  ▼
            GroundTruthLogger → results_<driver>/results_sN/*.csv
                  ▼
            analyze_results.py → CDFs + collision-rate + summary.csv
```

Key design choices already baked in:

- **Driver is a pluggable control source.** The scenario owns spawning/fog/logging; the
  driver only returns `VehicleControl`. This keeps the comparison apples-to-apples.
- **Lateral is a controlled constant.** Both drivers steer with `BasicAgent`; only
  *longitudinal* differs. PCLA's own steering is discarded.
- **MLP sensor auto-selected from the model's feature schema.** Radar model (10 base cols)
  → 100 m `FrontRadar`; vision model (11 cols incl. `obstacle_detected`) → YOLO depth.
- **Staging** (opt-in `--stage-approach`): a GT-based gap-keeper tailgates the lead, then
  hands longitudinal control to the model exactly when the event fires (S2 brake / S4 cut-in).

---

## 4. What was built this session

| Area | Files | What |
|---|---|---|
| Driver framework | `scenarios/drivers/{base,steering,__init__}.py` | `Driver` interface, shared `BasicAgentSteering`, lazy `make_driver` factory |
| PCLA driver | `scenarios/drivers/pcla_driver.py` | route gen from spawn, `tfv6_visiononly`, throttle/brake only, robust cleanup |
| MLP driver | `scenarios/drivers/mlp_driver.py` | YOLO **or** 100 m radar (auto), MLP→PID, BasicAgent steer |
| Scenario wiring | `scenarios/s1..s4.py`, `run_all.py` | `--driver/--model-dir/--pcla-agent`, driver-specific `--output-root` |
| S1 trigger | `s1`, `config.py` | obstacle spawns only when ego > `S1_SPAWN_SPEED_KMH` (60 km/h) |
| Staging | `scenarios/staging.py`, `s2`, `s4` | `GapKeepController`, `--stage-approach`, `--stage-gap`, handover at event |
| S4 severity | `s4` | `--cutin-stop` (NPC brakes to a full stop after cutting in) |
| Analysis | `scenarios/analyze_results.py` | per-run metrics, CDFs, collision rate, summary CSV |
| Data collector | `collect_throttle_brake_data.py` | Town04, 100 m radar, S2 (`lead_decelerating`) + S4 (`cut_in`) phases, `finally` cleanup |

Design rationale and the live results (e.g. at 30 m gap PCLA stopped at 6.3 m; at 20 m it
was tighter) are in `docs/plans/pcla-vs-mlp-scenario-comparison.md`.

---

## 5. Critical assessment — what is weak and what could be better

### 5.1 The methodology measures a narrow slice (by design, but state it)
Staging + discarding PCLA's steering means we evaluate **"longitudinal reaction from a
clean, staged following state,"** not driving. That is a *legitimate, controlled* ablation
— but it is not what "is PCLA good?" means. The honest framing: *"given an ideal approach,
how well does each model brake/follow?"* Anything about following ability, lane-keeping, or
end-to-end competence is **out of scope** and must not be claimed from these results.

### 5.2 PCLA is very likely being under-measured (most important issue)
PCLA agents are **closed-loop, stateful, ego-history-dependent** systems. The TransFuser V6
/ LEAD agent (the traceback shows `base_agent.tick → kalman_filter.smooth →
_bicycle_model_forward(steers, throttles, brakes)`) **reconstructs its own ego state from
the controls it believes were applied.** In this harness we:
- feed it **BasicAgent steering** and (during staging) **gap-keeper throttle/brake** —
  *not the controls PCLA output*, so its internal Kalman/bicycle ego-state estimate diverges;
- give it a **route generated from the spawn** rather than its expected global plan;
- run a **"visiononly" variant that still executes the LiDAR/RADAR pipeline** internally
  (the ransac ground-removal in the trace), so the sensor rig may be partially mismatched.

The observed **bang-bang throttle/brake** is therefore plausibly a *harness artifact*, not
PCLA's true behavior. **Fair PCLA evaluation requires letting it fully self-drive its own
route** (its native mode) and only *measuring* longitudinal metrics — or accepting that
this is a "PCLA-in-a-foreign-loop" stress test, clearly labelled. This is the single
biggest threat to the validity of any PCLA-vs-MLP claim.

### 5.3 The imitation teacher is weak
The MLP imitates the **CARLA Traffic Manager autopilot** — a privileged but conservative,
jerky rule-based controller. SOTA models (TransFuser++) imitate **PDM-Lite**, a much
stronger rule-based privileged expert designed for Leaderboard 2. *The student cannot
exceed the teacher.* Switching the teacher to PDM-Lite (or an IDM/ACC expert, §6) would
raise the MLP's ceiling far more than more data from TM.

### 5.4 The label and controller are simplistic
- **Label = mean future ego speed over 10 frames.** Averaging *blurs the brake onset* — the
  exact thing S1/S2 test. A model that predicts a **future speed/accel trajectory** (e.g.
  next-N-step profile) or **time-to-collision-aware deceleration** captures sharp braking
  far better than a single smoothed scalar.
- **PID + brake-hold** is a hand-tuned reactive controller. A classical **IDM/ACC** law is
  interpretable, provably gap-stable, and effectively collision-free — a much stronger and
  more honest longitudinal baseline than the current PID, and a better behavioural target.

### 5.5 Statistical and coverage thinness
3 seeds × 4 fog × 4 scenarios, single town (Town04 highway), CDFs without confidence
intervals or significance tests. Modern closed-loop benchmarks (Bench2Drive) use **220
routes across 44 scenarios, 23 weathers, 12 towns** precisely to reduce metric variance.
At minimum: more seeds, report mean ± CI, and add a few non-highway maps.

### 5.6 Perception realism
- **Monocular distance from YOLO bbox height** (pinhole) is noisy and degrades exactly when
  the thesis says it should (fog) — but it is also a *weak* vision baseline; a learned
  monocular-depth or BEV estimator would be a fairer "vision" arm.
- **CARLA's default radar is idealized.** For a credible radar-in-fog claim, a realistic
  radar model such as **C-Shenron** (a physically-based CARLA radar simulator) is the right
  tool — idealized radar overstates the radar advantage.

### 5.7 The staging gap-keeper uses ground truth
The approach controller reads **GT ego↔NPC distance** (privileged). That is fine for
*staging* (it is scaffolding, not under test), but it means the approach phase is not
sensor-driven; don't report anything about the approach from staged runs.

### 5.8 Engineering / reproducibility
- **Run dir ≠ tracked dir.** Git tracks `carla4/scenarios`; the machine runs
  `carla5/Town04/scenarios` (untracked, manually copied). This already caused a
  "nothing changed" incident. **Unify the layout** (or symlink) — manual copy is the single
  biggest source of "stale code" bugs here.
- **No tests, untested CARLA paths.** The collector's `cut_in`/`force_lane_change`,
  `lead_decelerating` timing, and the radar `MLPDriver` were never run against CARLA in this
  session. Treat them as drafts until smoke-tested.
- **Radar/vision auto-detect is a heuristic** (`obstacle_detected not in base_feature_cols`).
  Robust today, but brittle if a radar dataset ever adds that column. A explicit `sensor`
  field in `dataset_config.json` would be safer.
- **Real-time `time.sleep` in scenarios** slows batch sweeps; a `--headless`/no-sleep mode
  would speed full runs.

### 5.9 Missing metrics
No comfort/jerk metric, no infraction taxonomy, no CARLA-style **Driving Score** (route
completion × infraction penalty), no reaction-time distribution as a first-class output.
Bench2Drive's **Multi-Ability** breakdown (Merging, Overtaking, **Emergency Braking**,
Giving Way, Traffic Signs) is essentially the metric design this project is reinventing.

---

## 6. Better work available in this domain

### 6.1 End-to-end driving models (what PCLA already wraps)
- **TransFuser++ / LEAD (TransFuser V6)** — camera+LiDAR end-to-end, 1st/2nd at the CVPR
  2024 CARLA Challenge; the `tfv6_*` agents here are this family.
  [carla_garage](https://github.com/autonomousvision/carla_garage),
  [LEAD](https://github.com/autonomousvision/lead).
- **InterFuser, LAV, TCP, ThinkTwice, PlanT/PlanT 2.0** — all available *inside PCLA*
  already; you can swap `--pcla-agent` to benchmark several SOTA agents with zero new code.
  [PlanT 2.0](https://arxiv.org/pdf/2511.07292).
- **RL planners — CaRL, Roach** (also in PCLA) — learned without imitation; useful contrast
  to your imitation MLP.
- **VLM/LLM drivers — SimLingo, LMDrive** (in PCLA) — language-conditioned; the frontier.

### 6.2 Evaluation frameworks (stronger than hand-rolled S1–S4)
- **Bench2Drive** — the current standard for closed-loop E2E evaluation in CARLA: 220
  short routes, 44 interactive scenarios (incl. **cut-in** and **emergency braking** as
  named skills), 23 weathers, 12 towns, low-variance metrics, per-skill attribution.
  Adopting Bench2Drive's scenario defs + metrics would make results comparable to the field.
  [Bench2Drive](https://arxiv.org/abs/2406.03877).
- **CARLA Leaderboard 2.0** — the hardest public benchmark; Driving Score = route
  completion × infraction multiplier.
- **CARLA ScenarioRunner / OpenSCENARIO** — standardized scenario authoring instead of
  bespoke Python per scenario.
- **NHTSA pre-crash typology / Euro NCAP AEB protocols** — your S1–S4 map onto NHTSA crash
  types; NCAP's CCRs/CCRm/CCRb AEB tests are the *industry* version of exactly what you
  built (lead stopped/moving/braking) and give standardized speeds/criteria.

### 6.3 Longitudinal control specifically (your actual sub-problem)
- **Intelligent Driver Model (IDM) / ACC** — interpretable, gap-stable, near-collision-free
  car-following with a handful of physical parameters; the *right* classical baseline to put
  next to your PID+MLP. [IDM](https://traffic-simulation.de/info/info_IDM.html).
- **MPC / CACC** — optimization-based longitudinal control with explicit constraints
  (comfort, safe distance); stronger than reactive PID.
- An **IDM teacher** (instead of TM autopilot) would also give the imitation MLP a clean,
  collision-free target to learn from.

### 6.4 Radar & adverse weather
- **C-Shenron** — realistic radar simulation for end-to-end driving in CARLA; the credible
  way to support the fog/radar thesis instead of CARLA's idealized radar.
  [C-Shenron](https://wcsng.ucsd.edu/c-shenron/). *(Note: git history shows a `C-shenron`
  dir was removed earlier — worth revisiting.)*
- **Learner–expert asymmetry (LEAD, CVPR'26)** — directly studies why closed-loop IL agents
  fail vs their experts; relevant to your MLP's closed-loop gap.

### 6.5 The one-line takeaway from the literature
For the *longitudinal* question, the strongest, cheapest, most defensible comparison is
**IDM/ACC (classical) vs your MLP (imitation) vs PCLA (end-to-end), evaluated on
Bench2Drive's emergency-braking/cut-in skills with a realistic radar (C-Shenron) and a
PDM-Lite/IDM teacher.** That swaps three weak links (TM teacher, idealized radar, bespoke
scenarios) for field-standard ones.

---

## 7. Prioritized recommendations

1. **Fix PCLA fairness first (§5.2).** Add a "PCLA native" mode that lets PCLA fully
   self-drive its own route, and only *measure* longitudinal metrics. Compare against the
   current "staged/longitudinal-only" mode. If bang-bang persists in native mode, it's real;
   if it disappears, your current PCLA numbers are invalid.
2. **Add an IDM/ACC baseline driver** (`drivers/idm_driver.py`). It is ~40 lines, collision-
   free, interpretable, and instantly contextualizes whether the MLP is any good.
3. **Unify the repo layout** so the run dir == the tracked dir (kill the manual `carla4 →
   carla5/Town04` copy).
4. **Upgrade the teacher** from TM autopilot to IDM or PDM-Lite for the MLP data.
5. **Predict a trajectory/accel, not a smoothed scalar speed**, so braking onset survives.
6. **Adopt Bench2Drive scenarios + metrics** (or at least its emergency-braking/cut-in defs
   and Driving Score) for comparability and lower variance; add seeds + confidence intervals.
7. **Use C-Shenron radar** before making any "radar beats vision in fog" claim.
8. **Smoke-test the untested CARLA paths** (collector `cut_in`/`lead_decelerating`, radar
   MLPDriver) at `--duration 120` and one scenario each.

---

## 8. Complete command reference

> Two conda envs. PCLA driver → `PCLA` env. MLP driver + data collection/training →
> `carla4` env. Analysis → any env with pandas+matplotlib. CARLA must be running
> (`./CarlaUE4.sh`). Paths assume you run from `carla4/` (data/train) or `carla4/scenarios/`
> (scenarios). On your machine that is `carla5/Town04/` and `carla5/Town04/scenarios/`.

### 8.1 Data collection (imitation data)
```bash
conda activate <carla4-env>
cd carla4

# Radar + camera collector — Town04, 100 m radar, all 5 phases (incl. S2/S4)
python collect_throttle_brake_data.py --town Town04 --duration 1800
# quick smoke test (watch for "Lead vehicle braking hard (S2)", "NPC forced cut-in (S4)",
# and a clean map on Ctrl+C):
python collect_throttle_brake_data.py --town Town04 --duration 120

# Vision-only (YOLO) collector
python collect_vision_only_data.py --town Town04 --duration 1800
```

### 8.2 Training
```bash
# Radar model -> model_throttle_brake
python train_throttle_brake.py \
    --data dataset_throttle_brake/data.csv \
    --config dataset_throttle_brake/dataset_config.json \
    --output model_throttle_brake

# Vision model -> model_vision_only
python train_throttle_brake.py \
    --data dataset_vision_only/data.csv \
    --config dataset_vision_only/dataset_config.json \
    --output model_vision_only
```

### 8.3 Single-scenario runs (from `carla4/scenarios/`)
```bash
# --- S1: lead stopped (obstacle spawns only when ego > 60 km/h) ---
python s1_lead_vehicle_stopped.py --driver pcla --fog 0 --seeds 42
python s1_lead_vehicle_stopped.py --driver mlp  --model-dir ../model_throttle_brake --fog 0 --seeds 42

# --- S2: lead decelerating (staging recommended) ---
python s2_lead_vehicle_decelerating.py --driver pcla --fog 0 --seeds 42 --stage-approach
python s2_lead_vehicle_decelerating.py --driver pcla --fog 0 --seeds 42 --stage-approach --stage-gap 30
python s2_lead_vehicle_decelerating.py --driver mlp  --model-dir ../model_throttle_brake --fog 0 --seeds 42 --stage-approach

# --- S3: lead constant speed (NO staging — this is the honest follow test) ---
python s3_lead_vehicle_constant_speed.py --driver pcla --fog 0 --seeds 42
python s3_lead_vehicle_constant_speed.py --driver mlp  --model-dir ../model_throttle_brake --fog 0 --seeds 42

# --- S4: cut-in (staging + optional brake-to-stop) ---
python s4_cut_in.py --driver pcla --fog 0 --seeds 42 --stage-approach
python s4_cut_in.py --driver pcla --fog 0 --seeds 42 --stage-approach --cutin-stop
python s4_cut_in.py --driver pcla --fog 0 --seeds 42 --stage-approach --cutin-stop --stage-gap 12
```

Per-scenario flags:
- `--driver {pcla,mlp}`, `--fog <list>`, `--seeds <list>`, `--town Town04`, `--output <dir>`
- `--model-dir <dir>` (MLP), `--pcla-agent tfv6_visiononly` (PCLA)
- S2/S4 only: `--stage-approach`, `--stage-gap <m>`
- S4 only: `--cutin-stop`

### 8.4 Full sweeps (batch, from `carla4/scenarios/`)
```bash
# PCLA (PCLA env)
conda activate PCLA
python run_all.py --driver pcla --pcla-agent tfv6_visiononly \
    --scenarios 1 2 3 4 --fog 0 40 70 100 --seeds 42 123 256 \
    --town Town04 --output-root results_pcla --timeout 300

# MLP radar (carla4 env)
conda activate <carla4-env>
python run_all.py --driver mlp --model-dir ../model_throttle_brake \
    --scenarios 1 2 3 4 --fog 0 40 70 100 --seeds 42 123 256 \
    --town Town04 --output-root results_mlp --timeout 300
```
> Note: `run_all.py` does **not** currently pass `--stage-approach/--cutin-stop`. For staged
> sweeps, run s2/s4 directly in a shell loop, or ask to add the passthrough.

### 8.5 Analysis / comparison (any env with pandas+matplotlib)
```bash
python analyze_results.py --runs pcla=results_pcla mlp=results_mlp --out comparison
# outputs: comparison/summary.csv, per_run_metrics.csv, cdf_s1..s4.png, collision_rate.png
```

### 8.6 Benchmark a different PCLA agent (zero code change)
```bash
python s2_lead_vehicle_decelerating.py --driver pcla --pcla-agent tfv5_alltowns --fog 0 --seeds 42 --stage-approach
python s2_lead_vehicle_decelerating.py --driver pcla --pcla-agent if_if          --fog 0 --seeds 42 --stage-approach
# (see PCLA/agents.json for the full list)
```

---

## 9. Config knobs (where to tune behavior)

| Knob | Location | Default |
|---|---|---|
| Radar range | `collect_throttle_brake_data.py` `MAX_RADAR_RANGE`; `drivers/mlp_driver.py` `RADAR_RANGE` | 100 m |
| Collection town | `collect_throttle_brake_data.py` `DEFAULT_TOWN` | Town04 |
| Collection phases | `collect_throttle_brake_data.py` `SCENARIOS` | 5 phases |
| S2 lead brake cadence | `collect_throttle_brake_data.py` `LEAD_BRAKE_PERIOD_S/DURATION_S` | 10 s / 2.5 s |
| S4 cut-in cadence | `collect_throttle_brake_data.py` `CUT_IN_PERIOD_S` | 12 s |
| S1 spawn speed gate | `scenarios/config.py` `S1_SPAWN_SPEED_KMH` | 60 km/h |
| S1 obstacle distance | `scenarios/config.py` `S1_OBSTACLE_DISTANCE` | 35 m |
| S2 brake trigger step | `scenarios/config.py` `S2_BRAKE_TRIGGER_STEP` | 300 |
| Staging gap | `--stage-gap` (s2 default 20 m, s4 default 15 m) | 20 / 15 m |
| S4 staging spawn/trigger | `s4_cut_in.py` `STAGE_NPC_AHEAD_M / STAGE_CUT_IN_STEP` | 20 m / 120 |
| MLP cruise floor | `drivers/mlp_driver.py` `CRUISE_SPEED_MPS` | 30 km/h |
| Fog ladder / seeds | `scenarios/config.py` `FOG_LADDER`, `RANDOM_SEEDS` | [0], [42,123,256] |

---

## 10. Known caveats (read before trusting numbers)
- PCLA results may be a harness artifact (§5.2) — fix native-mode eval before claiming.
- Radar/vision models are **not** interchangeable unless `--model-dir` matches the sensor;
  the driver prints `sensor: radar|vision` at startup — verify it.
- Collector `cut_in`/`lead_decelerating` and radar `MLPDriver` are **untested against CARLA**.
- Run dir (`carla5/Town04/...`) ≠ git (`carla4/...`); copy updated files after every pull.

---

## Sources
- [carla_garage — TransFuser++ / Leaderboard 2 starter kit](https://github.com/autonomousvision/carla_garage)
- [LEAD — Minimizing Learner–Expert Asymmetry in End-to-End Driving (CVPR'26)](https://github.com/autonomousvision/lead)
- [PlanT 2.0 — Exposing Biases and Structural Flaws in Closed-Loop Driving](https://arxiv.org/pdf/2511.07292)
- [Bench2Drive — closed-loop multi-ability benchmark](https://arxiv.org/abs/2406.03877)
- [Intelligent Driver Model (IDM) and variants](https://traffic-simulation.de/info/info_IDM.html)
- [C-Shenron — realistic radar simulator for end-to-end driving in CARLA](https://wcsng.ucsd.edu/c-shenron/)
- [Hidden Biases of End-to-End Driving Datasets](https://arxiv.org/html/2412.09602v1)
