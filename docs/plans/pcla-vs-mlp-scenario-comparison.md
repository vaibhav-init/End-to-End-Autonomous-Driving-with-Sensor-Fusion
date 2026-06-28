# Plan: PCLA `tfv6_visiononly` vs custom vision-only MLP — longitudinal comparison

## Status: IMPLEMENTED
Created: `carla4/scenarios/drivers/{base,steering,mlp_driver,pcla_driver,__init__}.py`,
`carla4/scenarios/analyze_results.py`. Modified: `s1..s4` (pluggable `--driver`),
`run_all.py` (`--driver/--model-dir/--pcla-agent/--output-root/--timeout`).
All files syntax-checked; `analyze_results.py` functionally verified on synthetic GT logs.
Not run against CARLA (remote-only). See "Verify on remote" + "Execution sequence" below.



## Goal
Run the **same 4 NHTSA scenarios** (`carla4/scenarios/s1..s4`) with two vision-only drivers,
log ground truth identically for both, then compare **longitudinal** behavior only
(throttle/brake/speed/distance) with CDFs + collision stats.

- **Driver A — PCLA**: pretrained Transfuser-V6 vision-only agent. Key: `tfv6_visiononly`
  (project `tfv6`, variant `visiononly`, weights `pcla_agents/transfuserv6_pretrained/visiononly_resnet34`).
- **Driver B — MLP**: the custom `carla4/model_vision_only/` target-speed model.

No autopilot baseline driver.

## Scope: longitudinal only, steering is a controlled constant
Our system only controls **longitudinal** (throttle/brake). Steering/handling is delegated to
`BasicAgent` for **both** drivers — the scenarios are mostly straight, so BasicAgent steering is a
fixed, neutral lateral controller. This isolates the comparison to longitudinal control: the only
variable between the two runs is which model produces throttle/brake.

Per-driver control each tick:
- **MLP driver**: throttle/brake from the MLP (target-speed → PID); **steer from BasicAgent**.
- **PCLA driver**: throttle/brake from `pcla.get_action()`; **steer from BasicAgent** (PCLA's own
  steer is discarded so lateral handling is identical to the MLP run).

So `BasicAgent` is no longer "a driver" — it is a **shared steering helper** used inside both drivers.

## Core idea: make the scenario *driver-pluggable*
Today each `s1..s4` `run_scenario()` hard-wires `BasicAgent` as the full driver. We replace the inline
`control = agent.run_step()` with `control = driver.get_control(...)`, behind a `--driver {pcla,mlp}`
flag. Everything else (map, seeded spawn, fog, obstacle spawn timing/geometry, `GroundTruthLogger`,
early-termination) stays byte-identical.

### Driver interface
```python
class Driver:
    def setup(self, world, ego, carla_map, client, fog): ...
    def get_control(self, ego, world) -> carla.VehicleControl: ...
    def cleanup(self): ...
```

### Files to CREATE (paths)
```
carla4/scenarios/drivers/__init__.py          # make_driver(name, **kw) factory — LAZY imports
carla4/scenarios/drivers/base.py              # Driver ABC
carla4/scenarios/drivers/steering.py          # shared BasicAgent steering helper (set_destination + steer)
carla4/scenarios/drivers/pcla_driver.py       # PCLA throttle/brake + BasicAgent steer
carla4/scenarios/drivers/mlp_driver.py         # YOLO + VisionDistanceTracker + MLP + PID throttle/brake + BasicAgent steer
carla4/scenarios/analyze_results.py           # longitudinal CDFs, collision rates, comparison plots/tables
```

### Files to MODIFY
- `carla4/scenarios/s1_lead_vehicle_stopped.py` (and `s2`, `s3`, `s4`):
  - Add args: `--driver {pcla,mlp}`, `--pcla-agent` (default `tfv6_visiononly`), `--model-dir` (default `../model_vision_only`).
  - In `run_scenario()`: build the driver via `make_driver(...)`, call `driver.setup(...)` before the loop,
    `driver.get_control(ego, world)` inside it, `driver.cleanup()` in `finally`.
- `carla4/scenarios/run_all.py`:
  - Pass `--driver` / `--pcla-agent` / `--model-dir` through to each scenario subprocess.
  - Write to a driver-specific root: `--output-root results_<driver>` → `results_pcla/`, `results_mlp/`.

## Shared steering helper (`drivers/steering.py`)
Both drivers instantiate one `BasicAgent(ego, target_speed=60)`, `ignore_traffic_lights(True)`,
`ignore_stop_signs(True)`, `set_destination(<~500 m ahead on the same lane>)` — exactly the pattern
already in `s1`. Each tick it returns `run_step().steer`; its throttle/brake are discarded.

## Per-driver design
**MLPDriver** (runs in the `carla4` conda env)
- `setup`: shared steering helper + `CameraManager` + `YOLOPerception` + `VisionDistanceTracker`;
  load `model_vision_only/{target_speed_mlp.pt, scaler.pkl, model_config.json}`.
- `get_control`: perception → 11-col `VISION_FEATURE_COLS` history (use `history_frames` from the
  model config) → scaler → `TargetSpeedMLP` → target speed → `PIDSpeedController` → throttle/brake.
  Keep the `scenario1.py` cruise-floor override (test the model "as deployed"). steer = helper.steer.
- `cleanup`: destroy camera + any sensors it owns.

**PCLADriver** (runs in the `PCLA` conda env)
- `setup`: generate a Leaderboard route XML from the **same seeded ego spawn** (~300–500 m ahead) via
  `location_to_waypoint` + `route_maker`, unique temp file per run; build `PCLA("tfv6_visiononly", ego, route, client)`;
  build the shared steering helper too.
- `get_control`: `pc = pcla.get_action()`; `steer = helper.steer`; return
  `VehicleControl(throttle=pc.throttle, brake=pc.brake, steer=steer)`.
- `cleanup`: `pcla.cleanup()`. **NOTE:** `PCLA.cleanup()` destroys the ego vehicle and its sensors —
  coordinate with the scenario's `finally` (its `cleanup_actor(ego)` already guards on `is_alive`, so
  let PCLA own ego teardown and keep the scenario cleanup defensive).

## Integration details that must be right
1. **PCLA needs a route XML from the same spawn** (its internal planner needs a global plan even though
   BasicAgent steers). Build it from the seeded ego spawn; unique temp file per run (PCLA reuses a stale
   route file if the path already exists).
2. **MLP feature schema:** 11-col `VISION_FEATURE_COLS` (= `BASE_FEATURE_COLS` + `obstacle_detected`)
   + `history_frames` from `model_vision_only/model_config.json`. Never mix with the radar 10-col schema.
   Mirror the exact feature build from `carla4/scenario1.py` (~line 480).
3. **Sensors owned per-driver:** PCLA attaches its own vision rig; MLPDriver attaches its own
   `CameraManager`. The scenario attaches none. Each driver cleans up its own.
4. **Two conda envs → run the suite twice.** `drivers/` uses **lazy imports** (pcla_driver imports
   `from PCLA import PCLA` only when selected; mlp_driver imports ultralytics/sklearn only when selected)
   so importing the package in either env never breaks. `analyze_results.py` is env-agnostic.
5. **Subprocess-per-run isolation:** keep `run_all.py`'s fresh-process-per-(scenario,fog,seed) model —
   essential for PCLA (CUDA/watchdog/agent-module reload). `driver.cleanup()` in `finally`.

## Execution sequence (on the remote, CARLA already running)
```bash
# --- PCLA driver (PCLA conda env) ---
conda activate PCLA
cd carla4/scenarios
python run_all.py --driver pcla --pcla-agent tfv6_visiononly \
    --scenarios 1 2 3 4 --fog 0 40 70 100 --seeds 42 123 256 \
    --town Town04 --output-root results_pcla

# --- MLP driver (carla4 env) ---
conda activate <carla4-env>
cd carla4/scenarios
python run_all.py --driver mlp --model-dir ../model_vision_only \
    --scenarios 1 2 3 4 --fog 0 40 70 100 --seeds 42 123 256 \
    --town Town04 --output-root results_mlp

# --- compare (any env with pandas + matplotlib) ---
python analyze_results.py --runs pcla=results_pcla mlp=results_mlp --out comparison/
```

## Longitudinal metrics / CDFs (`analyze_results.py`) — no GT-logger changes needed
The `GroundTruthLogger` CSV already carries everything we need
(`gt_ego_speed_kmh`, `gt_distance_to_npc_m`, `gt_relative_velocity`, `time_to_collision_s`,
`throttle`, `brake`, `ego_accel_mps2`, `min_distance_so_far_m`, `collision_occurred`).

Per `(driver, scenario, fog)` — **longitudinal only, steer excluded**:
- **Collision rate** = fraction of seeds with any collision (primary safety metric).
- **CDF of closest approach** (`min(gt_distance_to_npc_m)` per run).
- **CDF of TTC** (`time_to_collision_s`, obstacle-present steps only).
- **CDF of peak deceleration / jerk** from `ego_accel_mps2` (smoothness/aggressiveness).
- **Reaction latency** = steps from obstacle spawn to first `brake > 0.3`.
- Overlaid CDF plots PCLA vs MLP, one panel per scenario × fog, + a summary table CSV.

## Debug output (per "print everything for debugging")
- `setup()`: agent/model + config loaded, sensors attached, route waypoint count, steering destination.
- `get_control()` every N steps: target speed, throttle/brake (+ steer from helper), detected distance +
  rel-vel, perception-gap flags (MLP), inference/watchdog timing (PCLA).
- Scenario already logs obstacle spawn + collision; analysis prints the summary table to stdout.

## Verify on remote before a full sweep
- `pcla_agents/transfuserv6_pretrained/visiononly_resnet34` weights present.
- `model_vision_only/{target_speed_mlp.pt,scaler.pkl,model_config.json}` present.
- Smoke run each: `python s1_lead_vehicle_stopped.py --driver pcla --fog 0 --seeds 42`
  and `--driver mlp --fog 0 --seeds 42` — confirm the car drives, obstacle spawns, a CSV is written.
- Confirm `tfv6_visiononly` runs on Town04 (map-agnostic via route; sanity-check the sensor rig).

## Decisions locked in
- **D1** Pluggable `--driver` refactor of `s1..s4`. ✓
- **D2** MLP "as deployed": keep the cruise-floor override. ✓
- **D3** ~~Autopilot baseline~~ — **dropped**. Only PCLA vs MLP. ✓
- **D4** Fog sweep `0/40/70/100`, seeds `42/123/256`, Town04. ✓
- **D5** **Both** drivers: model = longitudinal (throttle/brake), BasicAgent = steering.
  PCLA's own steer is discarded so lateral handling is identical across runs. *(Flag: flip if you'd
  rather PCLA fully self-steer.)*
