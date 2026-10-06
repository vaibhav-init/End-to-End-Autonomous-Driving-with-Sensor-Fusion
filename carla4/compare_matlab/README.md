# Python vs MATLAB Radar Comparison

Cross-validates this repository's `RealisticRadarModel` (`carla4/radar/realistic_core.py`,
profile `rgd_regime_v1`) against MATLAB Radar Toolbox's statistical radar
(`radarDataGenerator`) on **identical analytic ground truth**. This gives the
simulator an externally-validated fidelity anchor for the paper: measured
differences are error-model differences, not scenario differences.

Both models are scored against the same analytic truth, so the deliverable is
a statement like: *"under a matched envelope, range/azimuth/Doppler error
distributions are statistically consistent with MATLAB's reference model (KS
test), Pd curves agree within X, and ghost geometry matches mirror-method
analytics within Y m."*

## What is compared

| Aspect | Python side | MATLAB side |
|---|---|---|
| Model | `RealisticRadarModel`, `rgd_regime_v1` + 153 m range | `radarDataGenerator`, matched envelope |
| Scenario | analytic truth CSV (this folder) | same CSV → target-pose struct array per frame |
| Ghost | analytic mirror image via `multipath_targets` (geometry mode) | mirror-image pedestrian as a second pose with RCS 10→4 dBsm |
| SNR law | `13.2 + 40·log10(100/r)` dB (Albersheim-equivalent) | same law fed as truth SNR; MATLAB computes its own from Pd/Pfa/RCS |

Envelope (identical both sides): 10 Hz, 140° FOV, 0.15 m / 1.8° / 0.087 m/s
resolutions, ±44.3 m/s Doppler, 153 m range, sensor at origin.

## Pipeline

```
generate_truth_scenario.py  ──▶ truth_scenario.csv
        │                              │
        │                    ┌─────────┴──────────┐
        ▼                    ▼                    ▼
run_python_radar.py   run_matlab_radar.m (MATLAB) ─▶ matlab_detections.csv
        │
        ▼
python_detections.csv
        └──────────────┬───────────┘
                       ▼
              compare_models.py ──▶ comparison_report.md (+ PNG plots)
```

## Step 1 — Install MATLAB on this machine (you, once)

Machine check already done: 100 GB free disk, 15 GB RAM, 8 cores — sufficient
(MATLAB + Radar Toolbox needs ~30 GB disk, 4+ GB RAM).

1. Go to <https://mathworks.com> and **sign in with your college account**
   (the license activation is tied to you — never share the password).
2. **Download MATLAB → Linux** (any release R2023a or newer; R2024a+ recommended).
3. Unzip and run the installer **without `-inputFile`**:
   ```bash
   unzip matlab_R20XXa_glnxa64.zip -d matlab_installer && cd matlab_installer
   ./install
   ```
   Sign in, select your college license (Campus/TAHD), and **select these
   products**: `MATLAB` + `Radar Toolbox` (everything else optional;
   Automated Driving / Sensor Fusion not needed for this comparison).
4. Verify from a terminal:
   ```bash
   matlab -batch "ver, disp(exist('radarDataGenerator','file'))"
   ```
   `ver` must list **Radar Toolbox** and the `exist` call must print `2`.

If your college uses MATLAB Online instead of desktop installs, Option B from
the discussion applies: run the `.m` script there and copy
`matlab_detections.csv` back into this folder.

### Gotcha: do not pass `-inputFile` for an interactive install

Supplying `-inputFile` (even just `destinationFolder`) puts the installer in
unattended mode, which then *requires* a `fileInstallationKey` and exits with
`Exiting with status -2`. To install by signing in interactively, run bare
`./install` and set the destination folder in the GUI.

### Gotcha: FLEXlm segfault on rolling-release distros

On Arch (and other non-RHEL rolling distros) `./install` dies immediately with
`Segmentation fault (core dumped)`, before any UI appears. This is a MathWorks
bug, not a distro packaging problem: their bundled FLEXlm/FLEXnet licensing
library runs the x86 `CPUID` instruction, receives the vendor string in three
32-bit registers (`0x756e6547` "Genu", `0x49656e69` "ineI", `0x6c65746e` "ntel"
on Intel parts), then mistakenly dereferences those raw integers as pointers.
Backtrace:

```
#0 libmwinstall_activationwsclientimpl.so   (crash)
#1 lc_init ()                               (same library)
...
#14 dlopen ()                               (from libc)
#17 install::product_installer::LaunchHelper::launch(...)
```

Workaround: an `LD_PRELOAD` shim that maps those three addresses before the
licensing code runs.

```bash
gcc -shared -fPIC -O2 -o matlab_flexlm_cpuid_fix.so matlab_flexlm_cpuid_fix.c
LD_PRELOAD=$PWD/matlab_flexlm_cpuid_fix.so ./install
```

`matlab_flexlm_cpuid_fix.c` lives in `~/.local/lib/`. The same FLEXlm code runs
inside MATLAB itself, so MATLAB also needs the preload — `~/bin/matlab` is a
wrapper that sets it, and it only applies the shim to MATLAB rather than
exporting `LD_PRELOAD` shell-wide (the shim installs a SIGSEGV handler, which
you do not want active around every other program). Remove the wrapper once
MathWorks ships a fixed build. Set `MATHLAB_FIX_QUIET=1` to silence the
per-fault recovery log.

## Step 2 — Run the pipeline

```bash
cd carla4/compare_matlab
python3 generate_truth_scenario.py                       # analytic truth
python3 run_python_radar.py                              # this repo's model
~/bin/matlab -batch "run('matlab/run_matlab_radar.m')"   # MATLAB reference
python3 compare_models.py                                # report + plots
```

`run_all.sh` in this directory wraps all four steps and prints the headline
table, so the whole comparison is one command:

```bash
./run_all.sh
```

> Use `~/bin/matlab` (the shim wrapper), **not** a bare `matlab`. On a
> rolling-release distro plain `matlab` segfaults on the FLEXlm CPUID bug
> described in Step 1. If `~/bin` is not on your `PATH`, either use the absolute
> path or prefix the call:
> `LD_PRELOAD=$HOME/.local/lib/matlab_flexlm_cpuid_fix.so matlab -batch ...`

Each stage is independent and re-runnable; outputs land next to the scripts and
are gitignored (see `.gitignore`).

## Step 3 — Read the report

`comparison_report.md` contains, per model:

- **Point budget** — mean direct/ghost/clutter points per frame.
- **Geometry error** — bias/RMS/std of range, azimuth, radial velocity vs
  analytic truth, for the direct pedestrian and the ghost.
- **Pd vs range** — per-5 m-bin detection coverage of the pedestrian.
- **Ghost fidelity** — ghost-vs-mirror-image deltas; SNR difference vs the
  −6.0 dB truth bounce loss.
- **KS tests** (needs `scipy`) — Python-vs-MATLAB error-distribution
  equivalence per error type; `p > 0.05` means "statistically consistent".
- **Sanity lines** — Doppler-wrap / FOV / range-envelope violations (must be 0).

Known, expected differences (not bugs — findings for the paper):

- The Python model's logistic Pd (midpoint 8 dB SNR) is typically more
  optimistic at long range than MATLAB's Shnidman-style Pd — the Pd-vs-range
  tables quantify exactly this.
- MATLAB's false alarms come from a Pfa-per-cell model; the Python model uses
  a fixed Poisson clutter rate. Compare measured clutter/frame, not the
  parameters.
- MATLAB's range-rate noise model differs from the
  `floor + scale/sqrt(SNR)` law here.

## Measured results (R2026b, 385 frames, seed 42)

Recorded so the numbers in the paper can be traced to a specific run. Regenerate
with the Step 2 commands.

| Metric | Python | MATLAB |
|---|---|---|
| direct points/frame | 7.72 | 1.00 |
| ghost points/frame | 7.87 | 1.00 |
| clutter points/frame | 0.08 | 80.06 |
| direct range RMS | 0.157 m | 0.010 m |
| direct azimuth RMS | 1.051° | 0.178° |
| direct radial-vel RMS | 0.443 m/s | 0.0044 m/s |
| ghost range RMS | 0.167 m | 0.010 m |
| ghost azimuth RMS | 1.040° | 0.183° |
| ghost radial-vel RMS | 0.444 m/s | 0.0045 m/s |
| ghost−direct SNR | −5.56 dB | −6.45 dB |
| Pd, 12–25 m | 0.90–1.00 | 1.00 |
| Doppler-wrap / FOV / range violations | 0 / 0 / 0 | 0 / 0 / 0 |

What the numbers mean:

- **Both models are unbiased.** Every bias is within a few millimetres / a
  hundredth of a degree / a few mm/s of zero, and both hit the −6.0 dB ghost
  bounce loss to within 0.5 dB. MATLAB reproduces the analytic SNR law exactly
  (49.50 dB measured at 12.3797 m vs 49.50 dB predicted).
- **The KS tests reject** (p ≈ 0 on all six error types), so the two error
  *distributions* are not statistically equivalent. This is expected, not a
  defect: MATLAB's measurement noise collapses onto the declared resolution
  quantisers at high SNR (0.15 m range, 1.8° azimuth, 0.087 m/s range rate),
  giving RMS values 6–100× tighter than the Python model's continuous
  `floor + scale/sqrt(SNR)` law. The comparison is therefore a check that each
  model respects its declared envelope and bias, **not** a demonstration of
  equivalent noise.
- **Point budgets are not comparable as detection rates.** Python emits ~7.7
  points per target per frame (multi-point extended return), MATLAB emits
  exactly one (point target), and MATLAB's clutter count is set by Pfa-per-cell
  over ~8×10⁷ resolution cells. Use the Pd-vs-range table, not points/frame,
  for the detection comparison.
- The 5°/2 m matching gate in `compare_models.py` discards nothing: 0 % of
  points from either model fall outside it, so the error statistics above are
  computed on the full output.

## First-run verification checklist (MATLAB leg)

The `.m` script self-checks three things and prints all of them. Check these
before trusting the report:

1. **`RANGE_RATE_SIGN`** — the script prints the median direct range-rate error
   vs truth. On R2026b this is `0.000 m/s` with the default `-1`; a value near
   ±2× the true velocity means the sign is flipped.
2. **`point budget: direct …, ghost …, clutter …`** — if `direct` and `ghost`
   are 0, every report was classified as clutter, which means the target poses
   are not being associated. See the `RCSAzimuthAngles` and `RCSPattern` notes
   in the script header.
3. **Effective envelope** — the script prints `EffectiveFieldOfView` and the
   azimuth / range-rate resolutions; confirm they match the constants at the
   top of the file.

### radarDataGenerator API differences by release

The script targets the R2026b pose-struct API. Earlier releases documented a
`radarScenario` + `platform` workflow with a `radar(targets, ego, t)` call.
Verified R2026b specifics, all noted inline in `run_matlab_radar.m`:

- Signature is `rdr(targetPoses, simTime)` — pose **struct array**, not
  platforms. `radarScenario`/`platform`/`waypointTrajectory` belong to
  `radarSensor`. There is no `(platforms, sensorPlatform, time)` overload, and
  the old `catch`/`fallback` around that call is gone.
- A pose struct needs `Position` plus one of `PlatformID` / `ActorID` /
  `TruthID`. Build it by **field assignment** — `struct('Position',[x y z], …)`
  silently expands to a struct *array* when values are non-scalar, which passes
  validation but then never detects anything.
- `RCSAzimuthAngles` must span **−180:180**. A −90:90 window makes the
  aspect-angle lookup miss and the target is never detected at all.
- `RCSPattern` on a pose struct is in **dBsm as given**, not m² — do not apply
  `10^(dBsm/10)`. Verified: reported SNR tracks `pattern − ReferenceRCS`
  exactly.
- `rcsSignature` has **no `Unit` property** in R2026b; passing one throws
  `Invalid property/value pair arguments`. It is unused anyway — `radarDataGenerator`
  takes RCS from the pose struct or the `Profiles` property, not `Signatures`.
- `MaxUnambiguousRange` / `MaxUnambiguousRadialSpeed` only apply when
  `HasRangeAmbiguities` / `HasRangeRateAmbiguities` are on and warn
  "not relevant in this configuration" otherwise. Use `RangeLimits` /
  `RangeRateLimits` to bound reports (hard envelope, no wraparound).
- Reports arrive as a cell array of `objectDetection`; `Measurement` is
  `[az; range; range_rate]` when `HasElevation` is false. `ObjectAttributes` is
  a cell — unwrap it. False alarms carry a **negative** `TargetIndex`; real
  targets carry the `PlatformID` of the pose that produced them.

## Conventions (both sides)

- x forward, **y right**; azimuth from boresight, positive right.
- Radial velocity **positive = closing** (MATLAB's opening-positive output is
  flipped in the exporter).
- Ghost = type-1 order-2 mirror image across the guardrail plane `y = +4 m`;
  SNR loss = 5 dB (2nd-order bounce) + 1 dB (guardrail material) = 6 dB.
- Direct pedestrian RCS 10 dBsm; ghost 4 dBsm (MATLAB-only input).
- Truth SNR law anchored to Albersheim (Pd 0.9, Pfa 1e-6, ref 100 m) so both
  models see the same amplitude domain.

## File map

| File | Purpose |
|---|---|
| `run_all.sh` | runs all four stages and prints the headline table |
| `generate_truth_scenario.py` | analytic RGD-regime scenario → `truth_scenario.csv` |
| `run_python_radar.py` | truth → `RealisticRadarModel` (`rgd_regime_v1`, 153 m) → `python_detections.csv` |
| `matlab/run_matlab_radar.m` | truth → `radarDataGenerator` (matched envelope) → `matlab_detections.csv` |
| `compare_models.py` | both CSVs + truth → `comparison_report.md` + plots |
| `truth_scenario.csv` | shared ground truth (regenerated, deterministic) |
| `python_detections.csv` | this repo's model output |
| `matlab_detections.csv` | MATLAB reference output |
| `comparison_report.md` | the comparison result |

All generated outputs are gitignored (`.gitignore`); only the scripts, this
README and `run_all.sh` are meant to be tracked.

Note that the repository's top-level `.gitignore` excludes `*.md` globally, so
`comparison_report.md` (and this README) are **not** committed by design. Copy
the report somewhere tracked if a specific run needs to be cited.

## Scope notes

- This is a **target-list-level** comparison only — it deliberately does not
  validate raw FMCW/CFAR behaviour (the repository's stated §7 boundary). A
  `radarTransceiver` waveform-level extension is a possible follow-up.
- MATLAB's built-in `HasGhosts` models ground/two-ray and target-to-target
  reflections, not vertical walls — hence the mirror-pose construction for
  ghost geometry parity. Flip `ENABLE_MATLAB_MULTIPATH = true` in the `.m`
  script to additionally compare MATLAB's own multipath reference.
- The truth scenario is deterministic (no RNG); stochastic differences come
  from the models. Re-run with different `--seed` (Python) for run-to-run
  spread; MATLAB is seeded internally by `radarDataGenerator`.
