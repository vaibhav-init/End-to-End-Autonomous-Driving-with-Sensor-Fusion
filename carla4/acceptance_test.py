#!/usr/bin/env python3
"""
Acceptance test for a trained target-speed model.

Loss curves do not tell you whether the model learned to brake -- the shipped
model in this repo scored a good validation MAE while predicting 53 km/h with a
stopped car 8 m ahead, because the hardcoded emergency-brake rule was doing all
the braking. This probes the model directly with synthetic radar states and
checks that its predicted target speed actually falls when something is close
and closing.

The gate (from CLAUDE.md): obstacle at 10 m closing fast, predicted target
speed must drop below current speed. The other cases are reported for context
and never fail the run.
"""

import argparse
import json
import os
import pickle  # trusted: loads our own scaler.pkl from train_throttle_brake.py

import numpy as np
import torch

from driving_contract import RADAR_RANGE_M
from speed_model import TargetSpeedMLP, flatten_history


# name, ego speed m/s, gap m, obstacle speed m/s, is a pass/fail gate.
# gap None means nothing detected: the radar reports its range with zero
# relative velocity, exactly as _empty_state does.
CASES = [
    ("stopped car 10 m ahead", 15.0, 10.0, 0.0, True),
    ("stopped car 8 m ahead", 8.3, 8.0, 0.0, True),
    ("stopped car 25 m ahead", 15.0, 25.0, 0.0, False),
    ("slower lead 15 m ahead", 15.0, 15.0, 8.0, False),
    ("matched lead 20 m ahead", 12.0, 20.0, 12.0, False),
    ("open road, nothing detected", 12.0, None, 0.0, False),
]


def build_history(ego_speed, gap, obstacle_speed, history_frames, fps):
    """
    Ten frames of a physically coherent approach ending at `gap`.

    The gap is walked backwards at the closing rate so the history matches what
    the radar would actually have produced, rather than ten frozen copies of
    the current frame -- a model reading closing rate out of the history would
    otherwise see zero.
    """
    if gap is None:
        frame = {
            "ego_speed": ego_speed,
            "ego_acceleration": 0.0,
            "distance": RADAR_RANGE_M,
            "relative_velocity": 0.0,
            "ttc": 10.0,
            "obstacle_speed": 0.0,
        }
        return [dict(frame) for _ in range(history_frames)]

    relative_velocity = ego_speed - obstacle_speed
    frames = []
    for lag in reversed(range(history_frames)):
        past_gap = min(gap + max(0.0, relative_velocity) * lag / fps, RADAR_RANGE_M)
        if relative_velocity > 0.1:
            ttc = min(past_gap / relative_velocity, 10.0)
        else:
            ttc = 10.0
        frames.append(
            {
                "ego_speed": ego_speed,
                "ego_acceleration": 0.0,
                "distance": past_gap,
                "relative_velocity": relative_velocity,
                "ttc": ttc,
                "obstacle_speed": obstacle_speed,
            }
        )
    return frames


def main():
    parser = argparse.ArgumentParser(
        description="Probe a trained model's braking response"
    )
    parser.add_argument("--model-dir", default="model_throttle_brake")
    args = parser.parse_args()

    with open(os.path.join(args.model_dir, "model_config.json"), "r", encoding="utf-8") as fh:
        config = json.load(fh)
    with open(os.path.join(args.model_dir, "scaler.pkl"), "rb") as fh:
        scaler = pickle.load(fh)

    feature_cols = config["feature_cols"]
    base_cols = config["base_feature_cols"]
    history_frames = int(config["history_frames"])
    fps = float(config.get("fps") or 20)

    model = TargetSpeedMLP(input_dim=len(feature_cols))
    model.load_state_dict(
        torch.load(
            os.path.join(args.model_dir, "target_speed_mlp.pt"),
            map_location="cpu",
            weights_only=True,
        )
    )
    model.eval()

    print("=" * 70)
    print("ACCEPTANCE TEST")
    print("=" * 70)
    print(f"  model      {args.model_dir}")
    print(f"  trained on {config.get('radar_backend')} / {config.get('radar_profile')}")
    print(f"  town       {config.get('town')}")
    print()
    print(f"  {'case':<32}{'ego km/h':>10}{'predicted':>11}   verdict")

    failures = []
    for name, ego_speed, gap, obstacle_speed, is_gate in CASES:
        history = build_history(ego_speed, gap, obstacle_speed, history_frames, fps)
        row = flatten_history(history, base_cols)
        vector = scaler.transform(
            np.array([[row[col] for col in feature_cols]], dtype=np.float32)
        )
        with torch.no_grad():
            predicted = float(model(torch.tensor(vector.astype(np.float32)))[0])

        if is_gate:
            passed = predicted < ego_speed
            verdict = ("PASS" if passed else "FAIL") + "  (gate)"
            if not passed:
                failures.append(name)
        else:
            verdict = ""
        print(
            f"  {name:<32}{ego_speed * 3.6:10.1f}{predicted * 3.6:11.1f}   {verdict}"
        )

    print()
    if failures:
        print(
            f"  RESULT: FAIL -- the model does not slow down in "
            f"{len(failures)} gate case(s)."
        )
        print("  It has not learned to brake; do not deploy it as the controller.")
        return 1
    print("  RESULT: PASS -- predicted speed drops below current speed when closing.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
