# Handover — 2026-09-01

Read `work.md` first for the phase plan; this file covers session state, what
is trustworthy, and what is not.

---

## 1. Read this before running anything detached

**The collector suspends itself when stdin is not a terminal.**

`collect_throttle_brake_data.py` and `test_throttle_brake_live.py` start a
thread calling `input()` so a key press can force an obstacle spawn. In a
foreground terminal that blocks harmlessly on the TTY. Detached -- tmux,
nohup, ssh without a TTY -- the process sits in a background process group
with no controlling terminal, `input()` raises **SIGTTIN**, and the whole
process is **suspended**: state `T`, zero CPU, no output, no CSV, and CARLA
frozen waiting for a tick that never arrives.

It looks exactly like a sensor deadlock. It is not. Fixed in commit
`367336d` (listener starts only when `stdin.isatty()`), but **the fix is
committed and never verified end to end** -- the verification run was killed
before it finished. Confirming it is the first task.

Diagnostic that would have saved hours: `ps -o stat -p <pid>`. State `T` means
stopped, not hung. Check process state before theorising.

---

## 2. State of the work

### Done and verified

**Phase 1 — ghost physics** (`work.md` has the numbers)
- tangential Doppler now uses the true velocity vector, not a radial
  reconstruction
- incidence-dependent Fresnel reflection loss (ITU-R P.2040 permittivities +
  Rayleigh roughness), replacing a flat per-material table
- AR(1) correlated fading, so ghosts flicker instead of confirming every scan
- verified on live CARLA: 8.2 reflectors/frame, all three path families,
  52/52 paths intermittent, SNR 0.2-40.8 dB
- 7 unit tests in `radar/tests/test_multipath.py`

**Phase 2 — controller**
- hardcoded emergency-brake overrides are **off by default**, behind
  `--safety-rules` as an ablation arm
- IDM added as `--driver idm`: radar -> IDM -> target speed -> the same PID
  tail the learned driver uses
- 9 unit tests in `radar/tests/test_longitudinal.py`
- 75 tests pass on the remote

**Throughput** — collection is comfortably feasible. Geometry multipath runs
at 0.0137 s/frame, 3.7x real time, ~0.3 min per 60 s sequence. An earlier note
claiming it was too slow was wrong: it measured the probe's own
`capture_debug` serialisation, not the sensor.

### Committed but NOT verified

- the SIGTTIN fix (`367336d`)
- IDM as the collection teacher (`--teacher idm`, now the default)
- `--no-vision` radar-only collection
- the radar-only feature schema (6 base columns, 60 inputs instead of 100)
- `--radar-points-per-second`, `--client-timeout`, `--watchdog-s`
- spectator now follows the ego from spawn

**No controller data has been collected this session.** Zero rows.

---

## 3. The finding that changes the plan

**The shipped MLP does not brake.** Measured on S1, same seed, one flag apart:

| | target speed at 8 m, closing +13.6 m/s | throttle/brake | outcome |
|---|---|---|---|
| rules off (model decides) | **53.0 km/h** | 1.00 / 0.00 | collision at 15.5 km/h |
| rules on (ablation) | 32.0 km/h | 0.00 / 1.00 | stopped at 8.5 m |

IDM on the same scenario stopped safely at 7.2 m.

The hardcoded rules were doing **all** of the braking in every result the repo
has, including `scenarios/RESULTS_COMPARISON.md`. Those numbers describe a
rule-based AEB, not a learned controller.

Consequence: **a ghost filter cannot improve a controller that ignores
obstacles**, so Phases 5-7 are unmeasurable until the model is retrained. The
cause is the teacher -- CARLA autopilot almost never brakes hard -- which is
why the collector now defaults to `--teacher idm`.

**Acceptance test for any retrained model:** obstacle at 10 m closing fast,
predicted target speed must drop *below* current speed. Loss curves do not
substitute for this.

---

## 4. Next steps

1. **Verify the SIGTTIN fix.** Run a short detached collection and confirm a
   CSV appears with rows.
2. **Return to the known-good baseline and change one variable at a time.**
   This session changed teacher, vision and radar backend simultaneously and
   then spent hours debugging the combination. Order:
   a. autopilot + vision + native (the configuration that has worked) --
      confirm rows
   b. change only teacher -> IDM; confirm rows and that labels contain braking
   c. change only radar -> realistic; confirm rows
3. **Collect two datasets** into fresh directories (training globs every
   `*.csv` in the folder, so never point it at an existing one):
   - ghosts **off** -> clean baseline for arms A-C
   - ghosts **on** -> arm D, whether a controller learns to ignore ghosts
4. **Train both models**, each gated on the acceptance test above.

---

## 5. Environment

- Remote: `iiitd@192.168.21.33`, key auth, no passphrase
- Repo: `/home/iiitd/Desktop/vaibhav/carla-claude/End-to-End-Autonomous-Driving-with-Sensor-Fusion`
- Env: `conda activate PCLA` (python 3.10, torch 2.2.0+cu121, h5py, sklearn, ultralytics)
- **`export CARLA_ROOT=/storage/CARLA_0.9.16`** -- unset in the login shell, and
  `steering.py` defaults to `/opt/carla-simulator`. Without it scenarios do not
  fail; they warn and fall back to degraded steering, which would silently
  contaminate results.
- GPU: RTX 3060 12 GB, shared with CARLA (~5.8 GB when running)
- Shared office machine: install nothing, never launch CARLA (ask the user)
- Workflow: edit locally, commit, push, `git pull` on the remote. Never author
  on the remote.

**Cleanup script** at `/tmp/cleanup.py` on the remote (recreate if lost):
destroys stray vehicles/sensors and forces async mode. A crashed run leaves
CARLA in synchronous mode with nobody ticking it, which is what a "frozen
CARLA screen" means. Also leaks semantic LiDAR sensors still streaming at
240k points/s -- five accumulated during this session and crushed throughput
until removed by hand.

---

## 6. Open issues in the existing code

- **`RealisticFrontRadar.__init__` ends in `**_ignored`**, so a misnamed radar
  parameter is silently dropped. `radar_profile=` instead of `profile_name=`
  produced an entire run on the wrong sensor profile that still reported four
  confident PASS lines. Anything reading radar config should assert what it
  actually got.
- **`min_detection_probability`** puts a 2% floor under *any* SNR, so the
  sensor emits a trickle of detections regardless of link budget. That is how
  -118 dB ghosts reached the output before the roughness constants were fixed.
- **Config-signature gate blocks arms A-C as designed.** `mlp_driver` refuses
  to run when the deployed radar signature differs from the trained one, but
  those arms need a clean-trained model deployed *with* ghosts. The ghost rate
  must become a runtime knob outside the signature. Required before Phase 5.
- **Sensor teardown leaks.** A killed run leaves the semantic LiDAR attached
  and streaming. A long collection with `--sequence-retries` will leak the same
  way; the collector should sweep stale actors before each sequence.

---

## 7. Process notes for the next session

This session lost several hours to a diagnosable problem. What went wrong:

- **Multiple variables changed at once**, then debugged as a combination.
  Change one thing, verify, then the next.
- **Theories formed from single runs of an intermittent failure.** Five
  successive explanations (the IDM change, autopilot, world reloads, CARLA
  crashing, LiDAR density) were each abandoned when a later run contradicted
  them. Run the A/B against the known-good configuration first.
- **Instrumentation used late.** `faulthandler` cracked the ghost-probe crash
  in one run after six of guessing; `ps -o stat` would have cracked this one
  immediately. Measure before theorising.
- **Piping through `tail`/`grep` hides live output**, and `timeout` sends
  SIGTERM so `finally` never writes the CSV. A working run and a hung one look
  identical under those conditions. Use `python3 -u`, write to a log file, and
  let the job finish.
