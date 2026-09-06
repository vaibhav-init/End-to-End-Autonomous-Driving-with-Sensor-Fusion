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

import glob
import argparse
import json
import os
import pickle  # trusted: loads our own scaler.pkl from train_throttle_brake.py

import numpy as np
import torch

from driving_contract import RADAR_RANGE_M
from radar.realistic_core import RadarDetection
from speed_model import TargetSpeedMLP, flatten_history
from transformer_controller import (
    MODEL_TYPE as TRANSFORMER_MODEL_TYPE,
    build_window_tokens,
    load_model as load_transformer,
    predict_target_speed,
)


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


class ProbeBackground:
    """Static background scans sampled from a real collection's sidecar.

    The transformer's point features are frame-relative (amplitude against the
    frame median, local density), so a probe made of one lone point per scan
    is far outside the distribution it trained on. With a collection given,
    each probe scan reuses the non-road-user points of a random logged scan
    and the probe car takes the collection's typical road-user SNR.
    """

    _DYNAMIC_TAGS = frozenset((12, 13, 14, 15, 16, 17, 18, 19, 21))

    def __init__(self, dataset_dir, seed=0):
        from radar.detection_log import detections_by_frame, load_detection_log, sidecar_path

        self.rng = np.random.default_rng(seed)
        self.scans = []
        self.car_snr_db = 30.0
        if not dataset_dir:
            return
        for csv in sorted(glob.glob(os.path.join(dataset_dir, "*.csv"))):
            path = sidecar_path(csv)
            if not os.path.exists(path):
                continue
            log = load_detection_log(path)
            det = log["detections"]
            direct = det["source"] == b"direct"
            dynamic = np.isin(det["semantic_tag"], list(self._DYNAMIC_TAGS))
            road_users = det[direct & dynamic]
            if road_users.size:
                self.car_snr_db = float(np.median(road_users["snr_db"]))
            by_frame = detections_by_frame(det)
            frames = sorted(by_frame)
            for frame in frames[:: max(1, len(frames) // 200)]:
                records = by_frame[frame]
                keep = records[
                    (records["source"] == b"direct")
                    & ~np.isin(records["semantic_tag"], list(self._DYNAMIC_TAGS))
                ]
                if keep.size:
                    self.scans.append(keep)
        print(
            f"  probe background: {len(self.scans)} logged scans from {dataset_dir}, "
            f"road-user SNR {self.car_snr_db:.1f} dB"
        )

    def points(self, ego_speed):
        if not self.scans:
            return []
        scan = self.scans[int(self.rng.integers(len(self.scans)))]
        points = []
        for rec in scan:
            points.append(
                RadarDetection(
                    distance_m=float(rec["distance_m"]),
                    azimuth_rad=float(rec["azimuth_rad"]),
                    # Statics close at the ego speed of the probe, not the
                    # speed the logged ego happened to have.
                    relative_velocity_mps=float(ego_speed),
                    snr_db=float(rec["snr_db"]),
                    source="direct",
                    truth_object_id=int(rec["truth_object_id"]),
                    semantic_tag=int(rec["semantic_tag"]),
                )
            )
        return points


def build_scans(ego_speed, gap, obstacle_speed, window_frames, fps, background=None):
    """The transformer's view of the same approach.

    Oldest first: a ten-point car cluster at the walked-back gap in each scan,
    on top of a logged static background when one is given, so the model sees
    the same closing geometry the scalar model sees inside a realistic scan.
    """
    from radar.extended_target import expand_detection

    frames = build_history(ego_speed, gap, obstacle_speed, window_frames, fps)
    rng = np.random.default_rng(1)
    car_snr = background.car_snr_db if background is not None else 30.0
    scans = []
    for lag, frame in zip(range(window_frames - 1, -1, -1), frames):
        points = list(background.points(ego_speed)) if background is not None else []
        if gap is not None:
            car = RadarDetection(
                distance_m=frame["distance"],
                azimuth_rad=0.0,
                relative_velocity_mps=frame["relative_velocity"],
                snr_db=car_snr,
                source="direct",
                truth_object_id=1,
                semantic_tag=14,
            )
            points.extend(
                expand_detection(
                    car, rng, mean_points=10.0, range_resolution_m=0.15,
                    doppler_resolution_mps=0.087, azimuth_resolution_rad=0.0314,
                    minimum_range_m=1.0, maximum_range_m=RADAR_RANGE_M,
                    footprint_scale=1.3,
                )
            )
        scans.append((lag / fps, tuple(points)))
    return scans


def main():
    parser = argparse.ArgumentParser(
        description="Probe a trained model's braking response"
    )
    parser.add_argument("--model-dir", default="model_throttle_brake")
    parser.add_argument(
        "--background-from", default=None,
        help="collection directory whose logged static scans back the transformer probe",
    )
    args = parser.parse_args()

    with open(os.path.join(args.model_dir, "model_config.json"), "r", encoding="utf-8") as fh:
        config = json.load(fh)
    model_type = config.get("model_type", "mlp")
    fps = float(config.get("fps") or 20)

    if model_type == TRANSFORMER_MODEL_TYPE:
        model, _ = load_transformer(args.model_dir, device="cpu")
        window_frames = int(config["window_frames"])
        max_points = int(config["max_points"])

        background = ProbeBackground(args.background_from) if args.background_from else None

        def predict(ego_speed, gap, obstacle_speed):
            scans = build_scans(ego_speed, gap, obstacle_speed, window_frames, fps, background)
            window = build_window_tokens(scans, ego_speed, 0.0, max_points)
            return predict_target_speed(model, window, device="cpu")
    else:
        with open(os.path.join(args.model_dir, "scaler.pkl"), "rb") as fh:
            scaler = pickle.load(fh)
        feature_cols = config["feature_cols"]
        base_cols = config["base_feature_cols"]
        history_frames = int(config["history_frames"])
        model = TargetSpeedMLP(input_dim=len(feature_cols))
        model.load_state_dict(
            torch.load(
                os.path.join(args.model_dir, "target_speed_mlp.pt"),
                map_location="cpu",
                weights_only=True,
            )
        )
        model.eval()

        def predict(ego_speed, gap, obstacle_speed):
            history = build_history(ego_speed, gap, obstacle_speed, history_frames, fps)
            row = flatten_history(history, base_cols)
            vector = scaler.transform(
                np.array([[row[col] for col in feature_cols]], dtype=np.float32)
            )
            with torch.no_grad():
                return float(model(torch.tensor(vector.astype(np.float32)))[0])

    print("=" * 70)
    print("ACCEPTANCE TEST")
    print("=" * 70)
    print(f"  model      {args.model_dir} ({model_type})")
    print(f"  trained on {config.get('radar_backend')} / {config.get('radar_profile')}")
    injection = config.get("radar_ghost_injection") or {}
    if injection:
        print(f"  multipath  {injection.get('multipath_mode')} during collection")
    print(f"  teacher    {config.get('teacher', 'autopilot')}")
    print(f"  town       {config.get('town')}")
    print()
    print(f"  {'case':<32}{'ego km/h':>10}{'predicted':>11}   verdict")

    failures = []
    for name, ego_speed, gap, obstacle_speed, is_gate in CASES:
        predicted = predict(ego_speed, gap, obstacle_speed)

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
