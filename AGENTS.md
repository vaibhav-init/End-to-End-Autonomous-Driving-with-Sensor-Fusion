# Repository Guidelines

## Project Structure & Module Organization

`carla4/` is the primary camera-and-radar driving pipeline. Its top-level scripts collect data, train the target-speed MLP, and run live inference; shared feature/model definitions live in `speed_model.py` and perception code in `yolo_perception.py`. `carla4/scenarios/` contains the separate NHTSA-aligned evaluation harness, driver adapters, analysis scripts, and runtime result directories. `PCLA/` is a vendored, independently configured framework for pretrained CARLA agents; avoid coupling its environment or internals to `carla4/`. Root Markdown files contain design and experiment documentation.

Generated datasets, model artifacts, downloaded weights, and scenario CSV/plot outputs should remain out of source changes unless explicitly required.

## Build, Test, and Development Commands

There is no build system or unified dependency file. Use Python 3 and start CARLA at `127.0.0.1:2000` before simulator-backed commands.

```bash
cd carla4
python3 collect_throttle_brake_data.py
python3 train_throttle_brake.py --data dataset_throttle_brake \
  --config dataset_throttle_brake/dataset_config.json --output model_throttle_brake
python3 test_throttle_brake_live.py
cd scenarios
python3 run_all.py --scenarios 1 --fog 0 --seeds 42
python3 compare_drivers.py --runs mlp=results_mlp pcla=results_pcla
```

Create the PCLA environment with `conda env create -f PCLA/environment.yml`; see `PCLA/README.md` for agent-specific setup.

## Coding Style & Naming Conventions

Follow existing Python style: four-space indentation, imports grouped standard-library/third-party/local, `snake_case` functions and modules, `PascalCase` classes, and `UPPER_SNAKE_CASE` constants. Prefer `argparse` options for experiment parameters over hard-coded local values. Keep comments focused on control or experiment intent.

Preserve the model contract: predict target speed, then convert it to throttle/brake through PID. Keep radar results shaped as `distance`, `relative_velocity`, and `obstacle_speed`. Feature-schema changes require new datasets and retraining.

## Testing Guidelines

No automated unit-test framework or coverage threshold is configured. Run `python3 -m compileall carla4` for syntax checks, then use the single-scenario command above as the standard simulator smoke test. For scenario changes, record the driver, weather/fog preset, seed, CARLA town, and collision/result summary.

## Commit & Pull Request Guidelines

Recent history favors short imperative subjects, often with Conventional Commit prefixes such as `feat:`, `docs:`, and `chore:`. Keep each commit focused. Pull requests should describe the behavioral or experimental change, list commands run, link relevant issues, and include plots, screenshots, or metric summaries when results or visual output change. Call out modifications under vendored `PCLA/` explicitly.
