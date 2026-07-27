ej# Research Direction Report

## Executive Decision

The strongest research direction for this repository is:

> **Real-data-calibrated radar perception-error modeling for false-alarm-aware longitudinal control in CARLA.**

The central research question should be:

> Can a radar target-list model calibrated on real driving data reproduce meaningful missed detections, clutter, and persistent multipath ghosts in CARLA, and can temporal target validation reduce nuisance braking without degrading safety in genuine hazards?

This is a stronger and fairer contribution than trying to make the TFv6 vision-only policy crash. TFv6/PCLA should remain a secondary external baseline, while the primary comparison isolates improvements to the radar pipeline: current nearest-return processing versus calibrated errors, tracking, path gating, and robust control.

This review covers the main directly relevant primary literature and public datasets available through July 2026. It is a scoped systematic engineering review, not a claim that every radar publication has been examined.

## Diagnosis of the Current Project

### The current radar is not a realistic automotive radar

`carla4/scenarios/drivers/mlp_driver.py` uses CARLA's native radar and selects the nearest gated return. The same basic approach appears in collection and live-testing code. There is no clustering, data association, track management, confidence estimate, multipath model, or temporal false-alarm model.

CARLA describes its radar as a conic point detector, providing azimuth, altitude, depth, and radial velocity. Its implementation casts rays and reports line-trace hits. It does not model an FMCW signal chain, antenna pattern, material-dependent radar cross section, sidelobes, CFAR, or multipath. CARLA itself originally described this sensor as a low-fidelity placeholder. Therefore, increasing `points_per_second` or adding independent Gaussian noise would not make it experimentally realistic.

### A single false point can dominate control

The MLP driver contains hard overrides for stopped targets, closing obstacles, and low TTC. Because the nearest return becomes the target, one spurious close detection can cause full braking. In such trials, the result may measure the override rules rather than the learned MLP. Every intervention must therefore record its decision source: learned speed prediction, PID, TTC override, stopped-target override, or safety supervisor.

### The present PCLA handover does not measure reaction to an event

In S2, the policy runs during staging while another controller applies the longitudinal command. At a 15 m gap and 40–60 km/h, TFv6 can already propose zero throttle or braking because the visible lead vehicle is itself hazardous. At handover, the experiment merely starts applying an existing brake request. It does **not** establish that the policy observed and reacted to the lead vehicle's later deceleration.

There is also a state-validity risk: the LEAD agent propagates its filter using its predicted control, whereas staging applies a different control. Sensor updates partially correct this, but the prior action is inconsistent with the vehicle that actually moved.

Finally, the scenario loop applies ego control, advances the world, and then commands the lead vehicle to brake. Without frame-level event and sensor logging, reaction time is ambiguous by at least one simulation step.

These are experimental-design problems, not weaknesses that should be hidden. At the requested initial state, four requirements cannot all hold simultaneously:

1. The lead vehicle is visible.
2. The gap is fixed at 15 m at road speed.
3. TFv6/PCLA is unmodified.
4. Its first brake after handover is attributed to the later lead deceleration.

## Positioning Against Existing Work

| Area                     | What prior work establishes                                                                                                                 | Consequence for this project                                                                                                                   |
| ------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| CARLA radar              | Native CARLA radar is a simple ray/point sensor. C-Shenron adds substantially richer radar simulation using CARLA scene information.        | “We added Gaussian noise to CARLA” is not a sufficient novelty claim.                                                                          |
| Radar simulation         | RadaRays, full radar ray tracing, and RadSimReal model reflection, multipath, propagation, or learned real-data appearance.                 | A full electromagnetic simulator is too large for the present controller project. Clearly claim target-list fidelity, not raw-signal fidelity. |
| Perception-error models  | Data-driven perception-error models reproduce downstream sensor errors without simulating all physics.                                      | This is the appropriate abstraction and feasible research niche.                                                                               |
| Ghost detection          | Published rule-based and learned methods already detect underbody, double-bounce, wall, and guardrail ghosts.                               | DBSCAN alone or merely injecting ghosts is not a complete contribution.                                                                        |
| Control-aware evaluation | Prior work shows perception metrics alone do not predict driving consequences; AEB false-positive evaluation must include vehicle behavior. | Validate radar distributions, tracks, and closed-loop braking separately.                                                                      |
| End-to-end driving       | LEAD/TFv6 already reports strong Bench2Drive performance and benefits from combining vision and radar.                                      | A four-scenario “MLP beats vision” table is not novel and is not a controlled modality comparison.                                             |

The publishable gap is the connection between **validated radar artifact generation**, **false track formation**, and **closed-loop nuisance braking**, plus a mitigation whose safety cost is measured.

## Recommended Technical Scope

### Use a phenomenological target-list model

Do not begin with raw ADC or FMCW waveform simulation. Build a calibrated generator between CARLA ground truth and the controller's radar input:

```text
CARLA actors and geometry
        ↓
Calibrated radar perception-error model
        ↓
Noisy detections, misses, clutter, and persistent ghosts
        ↓
Clustering and multi-frame tracking
        ↓
In-path threat selection with confidence
        ↓
MLP target-speed prediction and longitudinal controller
```

Condition the model on variables available in both simulation and data:

- target range, azimuth, radial velocity, class, and aspect;
- detection probability and return cardinality;
- conditional range, angle, and Doppler error distributions;
- temporally correlated missed detections and dropouts;
- unstructured background clutter;
- structured underbody, double-bounce, wall, and guardrail ghosts;
- ghost lifetime, motion consistency, and spatial relation to the generating object.

Fit non-Gaussian conditional distributions where supported by data. Preserve temporal sequences instead of sampling every frame independently. This is what separates the work from a generic noise injector.

### Add a target-validation layer

A reasonable first method is:

1. Cluster detections using DBSCAN or HDBSCAN.
2. Track clusters with a Kalman/IMM filter and explicit data association.
3. Confirm tracks with an M-of-N rule and delete them using a separate miss threshold.
4. Estimate track-existence probability.
5. Gate threats against the ego vehicle's predicted path, not only a rectangular region.
6. Separate normal following control from emergency braking.
7. Use TTC together with required deceleration, for example
   \(a*\text{req}=v*\text{rel}^2/[2(d-d_\text{safe})]\), rather than TTC alone.

At 20 Hz, two-frame confirmation adds roughly 50 ms and three-frame confirmation roughly 100 ms. That delay is precisely the safety-versus-false-alarm trade-off to measure, not a detail to ignore.

The MLP should also be trained with artifact domain randomization. Compare clean-only training with calibrated-artifact training, while holding the controller and data split constant.

## Real Dataset Strategy

### Recommended minimum dataset pair

**RadarScenes** is the best starting point for ordinary production-radar detections. It contains more than four hours of driving from four automotive radars, with point-wise semantic labels and track identifiers. Its processed point representation is closer to this repository than raw radar tensors. A limitation is that only moving objects are comprehensively labeled; its static class mixes real static returns with false positives.

**Radar Ghost Dataset** is the necessary complement. It provides labeled real and simulated multipath ghosts across 111 sequences and 21 scenarios. Use it to learn or reproduce ghost geometry, type, duration, and motion consistency.

Split both datasets by complete drive or sequence, never randomly by frame. Frame-level splits leak nearly identical neighboring measurements into training and test sets.

### Optional extensions

- **RADIal:** raw ADC and range–angle/Doppler products; appropriate only if CFAR or signal-level detection becomes part of the contribution.
- **K-Radar:** large 4D radar tensors in fog, rain, and snow; useful for a substantial adverse-weather extension.
- **RADIATE:** diverse weather and lighting, but its rotating 360-degree radar has no automotive Doppler representation matching the present sensor.
- **CARRADA:** useful smaller range–angle/Doppler benchmark under more controlled conditions.
- **nuScenes/View of Delft:** useful external detection checks, but not required for the minimum paper.

Do not pool these sensors as if they were interchangeable. Calibrate per dataset/sensor and report transfer error. “Real-data-calibrated” means the model matches held-out conditional distributions; it does not mean that one parameter set represents every commercial radar.

## Validation at Three Levels

### 1. Sensor/detection fidelity

Compare ideal CARLA, native CARLA, a Gaussian-noise baseline, and the proposed model against held-out real sequences:

- probability of detection by range, angle, relative speed, and aspect;
- false detections per frame or per second;
- return-count distribution;
- conditional range, angle, and Doppler errors;
- ghost category and lifetime distribution;
- temporal autocorrelation of detections and misses;
- OSPA/cardinality error and distribution distances such as sliced Wasserstein.

Do not report cell-level CFAR false-alarm probability unless raw radar cells are actually modeled. At target-list level, false detections per frame, false tracks, and nuisance braking are the honest metrics.

### 2. Tracking fidelity

Measure false confirmed tracks per kilometre/minute, track fragmentation, track purity, identity switches, time to confirmation, and ghost persistence. An isolated false point and a stable false track have very different control consequences.

### 3. Closed-loop consequences

For non-hazard trials, report:

- false braking interventions per kilometre;
- brake duty cycle and intervention duration;
- peak deceleration and jerk;
- unnecessary speed loss and progress delay.

Pre-register a false intervention definition, such as brake above a fixed threshold when no ground-truth in-path actor lies inside a dynamic stopping envelope.

For true hazards, report:

- collision rate and impact speed;
- minimum gap, TTC, and required deceleration;
- first intervention time relative to the first sensor frame containing the hazard;
- stopping margin and comfort.

Reaction latency is undefined or left-censored when a policy is already braking before the event. It must not be reported as a reaction to the later deceleration.

Use paired scenario seeds, bootstrap 95% confidence intervals, and a mixed-effects or paired statistical analysis with method as a fixed effect and scenario/seed as repeated or random effects. Choose run count from a pilot variance or power analysis; the current single seed cannot support a research claim. For mitigation experiments, state a non-inferiority margin for collision rate or impact speed before examining results.

## Correct Handover and Scenario Protocol

Use two complementary protocols.

### Protocol A: Natural closed-loop driving

Give each policy control before the approach and let it create its own gap. Measure route progress, whether the scenario trigger is reached, pre-event braking, collision outcomes, and comfort. If TFv6 stops early, that is a legitimate conservative-policy result. Record “scenario not reached” rather than forcing it into a tailgating state.

### Protocol B: Controlled-state stress testing

Initialize both systems at the same speed, gap, and lead-vehicle state. Warm sensors and model state, record every proposed action, and then enable control. This measures behavior from a standardized state, including immediate pre-event avoidance. It does not automatically measure response to a later lead brake.

For a causal lead-deceleration comparison, sweep speed and time headway and analyze the subset where both policies are below a pre-event brake threshold. Report the excluded/pre-braking trials. A 15 m gap may remain as an explicitly named high-risk stress case, not as the only test point.

Implement frame-defined event sequencing:

1. Log the last pre-event world and sensor frame.
2. Command lead braking for a declared physics frame.
3. Advance the world and record the first frame in which lead deceleration physically occurs.
4. Record the first sensor frame reflecting that event.
5. Record policy inference frame, proposed control, and control application frame.

Log `event_command_frame`, `physical_event_frame`, `sensor_frame`, `inference_frame`, `proposed_control`, `applied_control`, `handover_state`, and `decision_source`. If staging is retained, PCLA's internal prior must receive the actual applied ego command, or its state should be deliberately reset and re-warmed at activation.

Never hide the lead vehicle from PCLA alone. A common occlusion or sudden-appearance experiment is valid only when both systems receive the same physical visibility event.

## Scenario Matrix

Use both true-hazard and non-hazard scenarios.

**True hazards**

- stopped lead vehicle;
- slower moving lead;
- lead vehicle decelerating at multiple rates;
- cut-in and cut-out;
- partial occlusion followed by common reveal.

Sweep ego speed (for example 40, 50, and 60 km/h), time headway (for example 0.8–2.0 s), lead deceleration, weather, and road curvature. If exact NHTSA or Euro NCAP parameters are not reproduced, describe the cases as “inspired by” their car-to-car archetypes rather than compliant tests.

**False-positive exposures**

- adjacent-lane vehicle;
- vehicle leaving the ego path;
- guardrail or concrete wall on a curve;
- overhead sign or bridge;
- truck underbody/double-bounce ghost;
- oncoming vehicle outside the path;
- empty road with unstructured clutter;
- persistent structured ghost with plausible Doppler.

Weather must not be described as making radar invariant. Real radar is often more robust than cameras in poor visibility, but rain, spray, wet surfaces, and multipath can change radar returns. Native CARLA's lack of those effects is a simulator limitation, not evidence of physical immunity.

## Baselines and Ablations

The minimum useful comparison is:

1. Ideal ground-truth target list.
2. Native CARLA radar with current nearest-return processing.
3. Independent Gaussian-noise/dropout baseline.
4. Calibrated model without structured ghosts.
5. Full calibrated model.
6. Full model plus clustering.
7. Full model plus clustering and tracking.
8. Full proposed path/risk gating.
9. Clean-trained versus artifact-randomized MLP.
10. Classical ACC/IDM or MPC longitudinal baseline.
11. TFv6 vision-only in its native full-control configuration.
12. Current longitudinal-only TFv6/PCLA wrapper, explicitly labeled as a non-native sensitivity test.

Also ablate every hard brake override. Otherwise the paper cannot distinguish MLP learning from handcrafted safety logic.

TFv6 is an end-to-end multi-camera driving policy, while the MLP is a small longitudinal controller with radar-derived inputs and BasicAgent steering. This is not a matched architecture or modality ablation. Report it as a system-level reference. Ideally include parameter count, inference latency, compute/energy proxy, and explainability of the radar controller; these can justify an MLP even when the larger policy is safer.

## Research Questions and Expected Claims

**RQ1.** Does the proposed model reproduce held-out real radar detection, clutter, and ghost statistics better than native CARLA and independent noise?

**RQ2.** Which artifact properties—location, persistence, Doppler consistency, or density—most strongly cause false confirmed tracks and nuisance brakin

**RQ4.** How do the radar controller and TFv6 vision-only baseline differ in early intervention, scenario reachability, safety, comfort, and compute under standardized conditions?

A defensible conclusion would be: “The calibrated artifact model better matches held-out target-list statistics; persistent structured ghosts, rather than independent point noise, dominate nuisance braking; the proposed validation layer reduces false interventions while preserving a predeclared safety margin.”

The following conclusions would **not** be supported:

- “The radar is physically realistic” without signal-level validation.
- “Radar is unaffected by weather.”
- “The MLP is better than TFv6” based only on four staged scenarios.
- “PCLA reacted to lead deceleration” when it was braking before the event.
- “The false-alarm rate is X” without defining the radar abstraction and denominator.

## Execution Plan and Stage Gates

### Phase 0 — Repair experimental validity

- Add frame-level event, sensor, inference, proposed-action, and applied-action logging.
- Separate policy proposal from staging command.
- Correct event ordering and PCLA applied-control feedback.
- Add pre-event braking and scenario-reachability outcomes.

**Gate:** deterministic replays must assign every action to an unambiguous sensor and event frame.

### Phase 1 — Establish radar baselines

- Store all native detections rather than only the nearest point.
- Implement ideal and independent-noise baselines.
- Build sequence-level loaders and held-out evaluation for RadarScenes and Radar Ghost.

**Gate:** the proposed calibration must outperform naive baselines on held-out sensor-level statistics. Otherwise do not claim increased realism.

### Phase 2 — Build the artifact model

- Fit conditional detection/error/cardinality models.
- Add temporal dropout, clutter, and structured ghost generators.
- Validate each component separately before combining them.

**Gate:** generated sequences must match both marginal and temporal real-data behavior.

### Phase 3 — Mitigate false alarms

- Add clustering, tracking, track confidence, path gating, and artifact-randomized training.
- Log which component rejects or promotes every candidate threat.

**Gate:** demonstrate a measurable reduction in control-level false interventions without crossing the predeclared safety margin.

### Phase 4 — Full benchmark

- Run the scenario matrix with paired seeds.
- Run the natural and controlled-state protocols.
- Include TFv6/PCLA and classical control baselines.
- Report confidence intervals, failures, scenario non-reachability, and compute.

If Phase 1 fails, pivot to “false-alarm stress testing” without a realism claim. If Phase 3 reduces false braking but materially worsens true-hazard safety, the mitigation is not successful and the threshold trade-off—not superiority—becomes the result.

## Recommended Paper Structure

1. Problem: idealized simulated radar hides nuisance-braking failures.
2. Related work: radar simulation, perception-error models, ghost detection, and control-aware evaluation.
3. Calibrated target-list artifact model.
4. False-alarm-aware tracking and longitudinal control.
5. Three-level validation methodology.
6. Sensor, tracking, and closed-loop results.
7. External TFv6/PCLA comparison and limitations.
8. Domain gap, safety limitations, and reproducibility.

## Core Literature and Resources

### Simulation and modeling

- [CARLA radar sensor documentation](https://carla.readthedocs.io/en/latest/ref_sensors/#radar-sensor)
- [CARLA 0.9.7 radar release description](https://carla.org/2019/12/11/release-0.9.7/)
- [CARLA radar implementation](https://carla.org/Doxygen/html/d5/d99/Radar_8cpp_source.html)
- [C-Shenron: high-fidelity radar in CARLA](https://escholarship.org/uc/item/9d79t73k)
- [RadaRays](https://arxiv.org/abs/2310.03505)
- [Full radar ray-tracing simulation](https://arxiv.org/abs/2305.14176)
- [RadSimReal](https://openaccess.thecvf.com/content/CVPR2024/html/Bialer_RadSimReal_Bridging_the_Gap_Between_Synthetic_and_Real_Data_in_CVPR_2024_paper.html)
- [Survey of automotive radar sensor models](https://www.mdpi.com/1424-8220/22/15/5693)
- [Multi-layer sim-to-real radar validation](https://arxiv.org/abs/2106.08372)
- [Parametric perception-error models](https://arxiv.org/abs/2302.11919)

### Data and radar artifacts

- [RadarScenes](https://arxiv.org/abs/2104.02493)
- [RadarScenes labeling limitations](https://radar-scenes.com/dataset/labeling/)
- [Radar Ghost Dataset paper](https://arxiv.org/abs/2404.01437)
- [Radar Ghost Dataset files](https://zenodo.org/records/6474851)
- [RADIal](https://github.com/valeoai/RADIal)
- [K-Radar](https://arxiv.org/abs/2206.08171)
- [RADIATE](https://arxiv.org/abs/2010.09076)
- [CARRADA](https://arxiv.org/abs/2005.01456)
- [Rule-based automotive radar clutter detection](https://arxiv.org/abs/2108.12224)
- [Learning-based multipath ghost detection](https://arxiv.org/abs/2007.05280)
- [See Further Than CFAR](https://arxiv.org/abs/2402.12970)

### Closed-loop evaluation and baselines

- [Control-aware evaluation of perception errors](https://www.ijcai.org/Proceedings/2020/483)
- [Probabilistic evaluation of AEB false positives](https://www.jstage.jst.go.jp/article/jsaeronbun/53/5/53_20224529/_article/-char/en)
- [PCLA framework](https://arxiv.org/abs/2503.09385)
- [LEAD and TFv6](https://arxiv.org/abs/2512.20563)
- [Bench2Drive](https://arxiv.org/abs/2406.03877)
- [NHTSA automatic emergency braking final rule](https://www.nhtsa.gov/sites/nhtsa.gov/files/2024-04/final-rule-automatic-emergency-braking-systems-light-vehicles_web-version.pdf)
- [Euro NCAP Safety Assist protocols](https://www.euroncap.com/safety-assist/)
