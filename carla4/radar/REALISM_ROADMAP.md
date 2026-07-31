# Radar Realism Options and Roadmap

No single simulator is "realistic radar" at every abstraction level. The
correct choice depends on whether the controller consumes three object-list
values, a detection cloud, a range-Doppler-angle tensor, or raw ADC samples.
This repository consumes a longitudinal target list, so target-list fidelity
is the primary requirement.

## Options Considered

| Approach | Captures well | Main limitation | Fit here |
|---|---|---|---|
| CARLA native radar | Visibility, ideal radial velocity, fast closed loop | Random rays; no material response, noise, signal processing, persistence, or multipath | Keep as the low-fidelity baseline |
| Native radar plus independent Gaussian noise | Simple range/Doppler error ablation | No temporal structure, clutter tracks, or structured ghosts | Included as `gaussian_baseline_v1` |
| Ideal CARLA actor/geometry target list | Exact hazards and upper-bound control | Unrealistically perfect sensing | Included as `ideal_target_list_v1` |
| C-Shenron-derived target list | Material/incidence response with CARLA occlusion | No upstream ADC/range-angle pipeline; no calibrated temporal artifacts | Existing `cshenron` baseline |
| Temporal perception-error model | Fast closed loop; misses, clutter, ghosts, latency, tracking, confidence | Fidelity depends entirely on held-out real-data calibration | Implemented as `realistic` and recommended for this MLP |
| Full official C-Shenron | FMCW/ADC and range-angle synthesis, material-aware scattering | Large GPU/dependency/runtime cost; upstream public noise path is disabled by default; not a drop-in scalar sensor | Use for signal/image experiments, not every control tick |
| Radar ray tracing such as RadaRays/custom Unreal sensor | Geometric reflection and multipath | Engineering and compute cost; material/receiver/post-processing still need validation | Strong future geometry upgrade |
| Learned sim-to-real rendering such as RadSimReal | Can reproduce real tensor appearance | Needs paired or representative real raw radar and changes the controller input | Relevant only if moving beyond scalar features |
| Recorded target-list replay | Exact real sensor statistics for recorded scenes | Cannot react to arbitrary CARLA geometry or closed-loop scene changes | Best sensor-level validation and open-loop regression source |
| Hardware-in-the-loop physical radar | Real RF front end and firmware | Requires RF target simulator/chamber, synchronization, and hardware | Highest-cost final validation |
| Commercial sensor simulation | Vendor-specific models and tool support | Proprietary, expensive, and often opaque | Optional external benchmark |

## Implemented Boundary

The `realistic` backend follows:

```text
CARLA geometry/actor truth
  -> C-Shenron-derived ideal material-qualified target
  -> temporal sensor error model
  -> target-list tracker
  -> path-gated longitudinal target
  -> unchanged MLP feature contract
```

CARLA truth IDs are used to look up ideal motion and to score diagnostics.
They are not used for measurement-to-track association. Tracking is based on
range, azimuth, and Doppler gates.

This is more defensible for the current controller than generating a costly
ADC cube and immediately reducing it to three numbers, but it must be called a
phenomenological target-list model until calibrated.

## Calibration Path

### 1. Ordinary detections

Use RadarScenes for production-radar point-list statistics. Its public sensor
setup reports 100 m range, about ±60° field of view, 0.15 m range resolution,
0.1 km/h radial-velocity resolution, about 0.5° boresight to 2° edge angle
resolution, and an average 60 ms cycle.

Do not treat its `static` class as clean false-alarm ground truth. RadarScenes
only manually labels moving objects; parked vehicles and standing pedestrians
are not comprehensively labeled.

### 2. Ground-truth error

RadarScenes does not by itself provide precise object ground truth for fitting
all measurement errors. Use a radar-plus-RTK dataset such as ViF-GTAD, a
controlled radar/lidar/RTK collection, or a local calibration campaign for
conditional range, azimuth, Doppler, and detection errors.

### 3. Multipath

Use Radar Ghost Dataset v1.1 by complete scenario/sequence. Fit ghost start
rate, bounce class, spatial offset relative to the source/reflector, Doppler
relation, SNR/amplitude loss, and lifetime. Its sensor geometry and mostly
stationary ego setup must not be assumed to transfer unchanged to all driving
scenes.

### 4. Adverse weather

Use controlled paired clear/rain/snow recordings from the same sensor.
CARLA's fog, rain, wetness, and dust values are visual/environment controls,
not RF measurements. Map them to radar parameters only after defining a
calibration experiment. Until then, weather coefficients remain explicit
sensitivity-analysis priors.

### 5. Validation

Split by complete drive or scenario. Compare held-out real and synthetic data
at three levels:

- sensor: detection probability, error distributions, cardinality, clutter,
  miss-run length, autocorrelation, ghost lifetime, and distribution distance;
- tracker: false confirmed tracks, confirmation delay, fragmentation, purity,
  and time-to-delete;
- control: collisions, impact speed, minimum bumper gap, required
  deceleration, nuisance braking per kilometre, speed loss, jerk, and latency.

The full profile should beat both `ideal_target_list_v1` and
`gaussian_baseline_v1` on held-out sensor-distribution metrics before making a
realism claim.

## Highest-Value Next Upgrades

1. Fit ordinary target-list statistics from sequence-level real data and save
   the resolved JSON profile with dataset provenance.
2. Replace the current probabilistic ghost placement with reflector
   surface-normal geometry and labeled bounce-type models.
3. Log every delivered detection and track state to a sidecar Parquet/JSONL
   stream for OSPA, cardinality, and association evaluation.
4. Add curved predicted-path gating based on steering/yaw rate rather than the
   current expanding straight corridor.
5. Separate normal ACC and emergency AEB from the MLP's existing fixed
   30 m/TTC overrides so sensor artifacts can be causally attributed.
6. If raw-radar perception becomes the research target, integrate full
   C-Shenron or a ray-traced/learned range-Doppler-angle backend instead of
   extending this scalar interface.

## References

- [CARLA 0.9.16 sensors](https://carla.readthedocs.io/en/0.9.16/ref_sensors/)
- [Official C-Shenron repository](https://github.com/ucsdwcsng/C-Shenron)
- [RadarScenes](https://radar-scenes.com/)
- [Radar Ghost Dataset](https://github.com/flkraus/ghosts)
- [PEM: Perception Error Model](https://arxiv.org/abs/2302.11919)
- [RadaRays](https://arxiv.org/abs/2310.03505)
- [RadSimReal](https://openaccess.thecvf.com/content/CVPR2024/html/Bialer_RadSimReal_Bridging_the_Gap_Between_Synthetic_and_Real_Data_in_CVPR_2024_paper.html)
- [ViF-GTAD](https://zenodo.org/records/7808255)
