# Radar-to-radar interference study

Answers one question: **when two automotive radars are close enough and
uncoupled enough to affect each other, what does that do to our target list,
and can it make the controller brake for something that is not there?**

## Scope, stated up front

* No waveform is simulated. This is a **structured surrogate**, the same class of
  model as `geometry` multipath in `multipath.py`: artifact *positions* and
  *relative amplitudes* follow from the physics, fine structure does not.
  Getting true interference products needs a waveform simulator.
* CARLA supplies **kinematics only** -- where the two vehicles are, where their
  radar boresights point, how the range closes. CARLA's own radar does not model
  interference and is not consulted.
* Multipath is disabled in every study run, so an RRI phantom is never confused
  with a wall ghost. Phantoms also get their own `"phantom"` source rather than
  being folded into `"clutter"`: a phantom carries coherent position and Doppler,
  so a tracker treats it as a moving object where a clutter cell does not.
* Antenna-to-antenna isolation is **swept, not measured**. The result is a
  boundary, not a single prediction.

## Pipeline

```bash
# 1. analytic sweep, no CARLA required (~2 min)
python3 carla4/rri_study/run_rri_study.py

# 2. CARLA inter-radar geometry (needs CARLA on :2000)
python3 carla4/rri_study/collect_carla_rri_geometry.py \
    --lane-offset-m 3.5 --follow-gap-m 20 --duration 20

# 3. interference physics over that geometry
python3 carla4/rri_study/run_carla_rri_sweep.py

# 4. closed loop: the sensor model actually tracks it (the hazard metric)
python3 carla4/rri_study/run_carla_rri_closed_loop.py

# 5. which thresholds the conclusion rests on
python3 carla4/rri_study/sweep_rri_thresholds.py
```

Stages 3 and 4 need stage 2's output. Stage 1 stands alone and reproduces on a
laptop with no CARLA and no GPU.

## What the scenarios are

| Scenario | Geometry |
|---|---|
| `adjacent_lane/steady` | Neighbour 3.5 m to the side, 1 m ahead, both radars facing along their own lane |
| `adjacent_lane/lane_change` | Same, but the neighbour sweeps laterally into our lane over 4 s |
| `rear_radar_tailgating/closing` | Our **rear-facing** radar; a follower closing at 2 m/s from a 20 m gap down to a 6 m floor |

The tailgating case is expressed in the rear radar's own boresight frame.
Rotating a rear radar's geometry into the ego frame would place it outside a
forward-looking field of view and silently discard every artifact, which is a
property of the wrong sensor model rather than of the interference.

## Result

Interference needs **head-on boresight geometry, a poor mount, and close
proximity**. All three, together:

| Condition | Outcome |
|---|---|
| Neighbour alongside or lane-changing into our path | **No phantom at any isolation**, because its radar points *away* from us -- boresight alignment reaches -1.0 |
| Rear radar + follower, 40 dB mount | No phantom at any distance tested |
| Rear radar + follower, 25 dB mount | Phantom within ~2 m |
| Rear radar + follower, 15 dB mount | Phantom from ~1 m; **15% of scans selected the phantom as the tracked target** |
| Rear radar + follower, 10 dB mount | Phantom from ~2 m; **30% of scans** |

A phantom that wins track association is a phantom brake: the phantom sits at
the interfering radar's true range and azimuth, so a longitudinal selector
picks it as the nearest object. That is the failure mode worth guarding
against, and it is distinct from simply reporting extra clutter.

## What the conclusion rests on

From `sweep_rri_thresholds.py`, varying one parameter at a time around
15 dB isolation / coherence 0.3:

**Dominant -- the result flips entirely inside these ranges:**

| Parameter | Range swept | Phantom selected |
|---|---|---|
| `rri_interference_coherence` | 0.0 → 1.0 | 34% → **0%** |
| `rri_min_inr_db_for_phantom` | 12 → −3 dB | 0% → **33%** |
| `rri_phantom_detection_slope` | 0.05 → 0.8 | 4% → **33%** |
| `rri_antenna_isolation_db` | 30 → 10 dB | 0% → **30%** |

`rri_interference_coherence` is the weakest link in the model. It encodes how
much of our coherent processing gain a phase-locked interferer actually earns,
which depends on the carrier offset between the two radars. Two radars with
matched carriers reject almost everything; a large offset rejects nothing. Real
measured RRI would pin this down and it is the single number most worth
measuring.

**Negligible -- the result is completely insensitive to these:**

| Parameter | Range swept | Phantom selected |
|---|---|---|
| `association_range_gate_m` | 1 → 8 m | flat, 14.7% |
| `association_azimuth_gate_deg` | 1 → 10 deg | flat, 14.7% |
| `detection_snr_midpoint_db` | 4 → 12 dB | flat, 14.7% |
| `rri_max_ghost_orders` | 0 → 3 | flat, 14.7% |

The first three are the important negative result. **The phantom cannot be
gated out.** It lands on top of the real neighbour return, so no association
gate or detection threshold separates them -- this is an identifiability
problem, not a tuning problem. Widening gates to catch a real target also
admits the phantom; tightening them to reject the phantom also loses the target.

`rri_max_ghost_orders` is flat because the range-ambiguity replicas
(`|k·R_unamb − d|`, `R_unamb = c·T_chirp/2` ≈ 150 m at 1 µs) land beyond
`max_range_m`, so extra orders contribute nothing. Halving the chirp period to
0.5 µs puts the first replica inside the envelope and doubles the phantom rate.

## Files

| File | Purpose |
|---|---|
| `../radar/rri.py` | The interference model: link budget, coherent rejection, phantoms, desensitisation, false-alarm inflation |
| `../radar/tests/test_rri.py` | Unit tests, including the invariants that were wrong during development |
| `run_rri_study.py` | Analytic sweep, both scenarios, isolation sweep |
| `collect_carla_rri_geometry.py` | CARLA inter-radar geometry collector |
| `run_carla_rri_sweep.py` | Interference physics over the collected geometry |
| `run_carla_rri_closed_loop.py` | Full sensor model; reports phantom-selected fraction |
| `sweep_rri_thresholds.py` | One-at-a-time threshold sensitivity |

Generated CSVs and the report are gitignored.
