# Recommended Research Direction

## Executive Decision

The strongest direction for this repository is:

> **Real-data-calibrated temporal radar error modeling and false-alarm-aware longitudinal control in CARLA.**

A suitable paper title is:

> **From Radar Artifacts to Braking Decisions: Control-Aware Radar Error Modeling for Robust AEB in CARLA**

The central hypothesis should be:

> Persistent, structured radar errors—missed detections, clutter, and multipath
> ghosts—cause more important closed-loop failures than independent Gaussian
> noise. Temporal track confidence and path-aware risk estimation can reduce
> nuisance braking without materially reducing collision avoidance.

This is a stronger contribution than claiming that a small radar MLP
outperforms a vision-only PCLA agent under fog.

## What the Repository Already Provides

The project has a useful research foundation:

- Two data collectors using Traffic Manager and a privileged ACC teacher.
- Ten-frame histories of ego speed, acceleration, radar distance, relative
  velocity, TTC, obstacle speed, and traffic-light features.
- A lightweight MLP that predicts target speed, followed by PID control.
- Episode-aware training and validation splits.
- Reproducible stopped-lead, braking-lead, constant-lead, and cut-in scenarios.
- Ground-truth telemetry, paired seeds, CDF generation, and PCLA integration.

The MLP is therefore a useful downstream controller, but its architecture
alone is not a strong novelty claim.

## Limitations of the Current Comparison

### Unmatched systems

The MLP system combines native radar, YOLO traffic-light features, an MLP,
PID, BasicAgent steering, launch assistance, and handwritten safety rules.
PCLA is a large pretrained end-to-end policy whose steering is discarded and
whose throttle/brake are applied directly. This is a system comparison, not a
matched modality or architecture ablation.

### Hardcoded braking dominates hazards

The MLP applies full brake when its target is below 1 km/h, when a closing
obstacle is within 30 m, or when TTC is below 3 seconds. Because S1 places a
stopped vehicle at 25 m, the 30 m rule usually determines the response before
the learned target-speed prediction matters.

Experiments must separate these modes:

1. MLP prediction plus PID only.
2. MLP plus the current hardcoded shield.
3. MLP plus the proposed uncertainty-aware supervisor.
4. Classical ACC/AEB baseline.

### Simplified radar representation

The native radar adapter selects one nearest gated return and reduces the
whole radar measurement to distance, relative velocity, and obstacle speed.
It has no clustering, data association, track management, confidence,
multipath handling, or path validation.

The included C-Shenron-derived adapter is not the full upstream simulator. It
uses semantic LiDAR, actor IDs, actor velocity, material gating, and fixed
noise before returning the same three scalars. It should not be presented as
a new physically realistic radar.

### Invalid weather conclusions

CARLA weather changes camera rendering, but native CARLA radar does not model
weather-dependent attenuation, spray, wet-road reflections, or multipath.
Consistency across CARLA fog presets proves simulator invariance, not
real-world radar immunity. Weather should remain secondary unless the radar
error model is calibrated using real adverse-weather data.

### Measurement and statistical issues

- Current distance is actor-center Euclidean distance, not bumper gap.
- TTC is based on scalar speed difference rather than path projection.
- Cut-in latency starts from the command trigger, not necessarily the first
  sensor frame containing an in-path target.
- Very large reported decelerations indicate finite-difference spikes.
- A smaller stopping gap is not automatically a better result.
- Fifteen runs provide a pilot CDF, not strong final statistical evidence.

## Proposed Technical Contribution

Insert a temporal radar perception-error model between CARLA ground truth and
the controller:

```text
CARLA scene and actors
        |
        v
Ideal target list
        |
        v
Real-data-calibrated temporal error model
  - conditional missed detections
  - range, azimuth, and Doppler errors
  - unstructured clutter
  - correlated dropouts
  - persistent structured ghosts
        |
        v
Clustering and multi-frame tracking
        |
        v
Track confidence and predicted-path gating
        |
        v
Risk-aware target-speed/AEB controller
```

This target-list abstraction is computationally feasible and more defensible
than fixed Gaussian noise. RadarScenes can provide production-radar
detection/error statistics, while the Radar Ghost Dataset can provide
multipath categories and temporal persistence. All calibration and evaluation
splits must be made by complete sequence, never by neighboring frames.

Relevant starting resources:

- [RadarScenes](https://arxiv.org/abs/2104.02493)
- [Radar Ghost Dataset](https://arxiv.org/abs/2404.01437)
- [Perception Error Models](https://arxiv.org/abs/2302.11919)
- [C-Shenron](https://wcsng.ucsd.edu/c-shenron/)
- [RadSimReal](https://openaccess.thecvf.com/content/CVPR2024/html/Bialer_RadSimReal_Bridging_the_Gap_Between_Synthetic_and_Real_Data_in_CVPR_2024_paper.html)
- [NHTSA AEB final rule](https://www.nhtsa.gov/press-releases/nhtsa-fmvss-127-automatic-emergency-braking-reduce-crashes)

## Research Questions

**RQ1:** Does the proposed temporal model match held-out real radar
distributions better than native CARLA and independent noise?

**RQ2:** Which artifact properties—persistence, position, Doppler consistency,
or density—most strongly produce false tracks and nuisance braking?

**RQ3:** Can temporal confirmation, track-existence confidence, and path-aware
risk gating reduce false interventions while preserving a predefined
collision-safety margin?

**RQ4:** How does the resulting lightweight radar system compare with TFv6 in
safety, comfort, scenario reachability, inference latency, and compute cost?

## Implementation Roadmap

### Phase 0: Repair experimental validity

- Log sensor, event, inference, and control-application frame numbers.
- Log raw detections, selected target, proposed control, applied control,
  staging state, and decision source.
- Measure bumper-to-bumper path distance and projected relative velocity.
- Filter longitudinal acceleration and calculate jerk.
- Verify the remote `tfv6_visiononly` configuration actually disables radar
  and LiDAR inputs.
- Run TFv6 in native full-control mode as the main external baseline. Retain
  the longitudinal-only wrapper as a controlled sensitivity experiment.

**Gate:** every reported reaction must be traceable to a sensor frame and an
unambiguous physical event.

### Phase 1: Establish sensor baselines

- Preserve every native CARLA radar return instead of only the nearest one.
- Implement an ideal ground-truth target list.
- Add independent Gaussian error/dropout as a deliberately simple baseline.
- Fit conditional detection, error, clutter, and temporal persistence models
  from real sequences.

**Gate:** the calibrated model must outperform simple baselines on held-out
sensor-level distributions before any realism claim is made.

### Phase 2: Develop the mitigation

- Cluster detections and associate them across frames.
- Use Kalman or IMM tracking with M-of-N track confirmation.
- Maintain track-existence probability.
- Gate targets against the ego vehicle's predicted path.
- Replace fixed 30 m/TTC rules with required-deceleration or probabilistic
  collision risk, including explicit comfort and emergency thresholds.
- Optionally train the MLP with calibrated artifact randomization.

### Phase 3: Closed-loop evaluation

Test true hazards:

- stopped and slower leads;
- lead braking at several deceleration rates;
- cut-ins at multiple overlaps and TTC values;
- pedestrian crossings and partial occlusions.

Test false-positive exposures:

- stopped vehicles in adjacent lanes;
- a vehicle leaving the ego path;
- guardrails and concrete walls on curves;
- overhead structures;
- crossing actors that clear the path;
- empty-road clutter;
- persistent structured ghosts with plausible Doppler.

Use these baselines and ablations:

1. Ideal target list.
2. Native CARLA nearest return.
3. Independent noise/dropout.
4. Calibrated model without structured ghosts.
5. Full calibrated model.
6. Full model plus tracking.
7. Full model plus tracking and path/risk gating.
8. Clean-trained versus artifact-randomized MLP.
9. Classical ACC/AEB.
10. Native full-control TFv6.

Report sensor fidelity, false confirmed tracks, collisions, impact speed,
minimum bumper gap, required deceleration, nuisance interventions per
kilometre, speed loss, jerk, inference latency, and scenario reachability.
Use paired seeds, held-out towns, bootstrap confidence intervals, and a pilot
variance or power analysis to determine the final run count.

## Claims to Avoid

Do not claim:

- that native CARLA radar proves real-world weather robustness;
- that the C-Shenron-derived scalar adapter is a full physical simulator;
- that the MLP learned emergency braking while hardcoded rules dominate it;
- that PCLA reacted to a later event if it was already braking;
- that the scenarios are NHTSA/Euro NCAP compliant unless every protocol
  parameter is reproduced;
- that one driver is superior from a 15-run CDF without uncertainty estimates.

## Concise Pitch for the Professor

> Our current platform demonstrates that idealized CARLA radar and hardcoded
> AEB rules can make a lightweight controller appear robust, but they do not
> reproduce important real radar failure modes. We propose a real-data-
> calibrated temporal radar perception-error model for CARLA and an
> uncertainty-aware tracking and braking supervisor. We will study how missed
> detections, clutter, and persistent multipath ghosts propagate into
> collisions and nuisance braking, then quantify the safety-versus-false-alarm
> trade-off using paired closed-loop experiments. TFv6 will be an external
> system baseline rather than the main novelty.

If real-data calibration is infeasible, the fallback topic should be a
**control-aware radar fault-injection and scenario-falsification benchmark**,
without claiming physical realism.

## How to Share Scenario Results

The best handoff is the complete result directories because console summaries
do not contain all per-frame behavior. Keep failed and successful runs; failed
runs are scientifically relevant.

From `carla4/scenarios`, first generate the comparison:

```bash
python3 compare_drivers.py \
  --runs mlp=results_s1_stress_mlp pcla=results_s1_stress_pcla

python3 analyze_results.py \
  --runs mlp=results_s1_stress_mlp pcla=results_s1_stress_pcla \
  --out comparison_s1_stress
```

Then create one transferable archive from the repository's `carla4/scenarios`
directory:

```bash
tar -czf s1_results_bundle.tar.gz \
  results_s1_stress_mlp \
  results_s1_stress_pcla \
  comparison_s1_stress
```

Upload `s1_results_bundle.tar.gz` if the chat interface permits file uploads.
If uploading is not possible, provide:

1. The exact MLP and PCLA commands used.
2. The complete final summary from each `run_all.py` execution.
3. The complete output of `compare_drivers.py`.
4. `comparison_s1_stress/per_run_metrics.csv`.
5. `comparison_s1_stress/summary.csv`.
6. A list of timeouts, crashes, invalid runs, or removed CSV files.
7. CARLA version, town, model checkpoint, radar backend, and git revision.

Do not paste thousands of per-frame terminal lines unless a run failed.
For a failure, paste the first abnormal frame sequence, the traceback, and the
corresponding CSV filename.
