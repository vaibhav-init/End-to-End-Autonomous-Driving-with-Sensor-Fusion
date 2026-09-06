"""End-to-end smoke of the transformer controller chain on a synthetic collection.

Builds a small realistic-backend collection (CSV + detection sidecar +
dataset_config.json) the way collect_throttle_brake_data.py would, trains the
transformer for two epochs, runs the acceptance probe and the counterfactual
ghost test on the result. Needs torch; skipped on the authoring box.
"""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from radar.detection_log import DetectionLog, sidecar_path  # noqa: E402
from radar.realistic_core import RadarDetection  # noqa: E402
# speed_model imports torch at module level; keep this module importable
# without it so the skip below applies instead of an import error.
BASE_FEATURE_COLS = [
    "ego_speed", "ego_acceleration", "distance", "relative_velocity", "ttc", "obstacle_speed",
]

try:  # noqa: SIM105
    import torch  # noqa: F401

    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False

ROOT = Path(__file__).resolve().parents[2]
FPS = 20
HISTORY = 10


def _episode(rows, log, start_frame, episode, rng, ghosts):
    """One 30 s approach-and-follow episode: ego closes on a slower lead."""

    ego_speed = 12.0
    lead_speed = 6.0
    gap = 60.0
    scan_index = start_frame // 2
    history = []
    speeds = []
    for step in range(30 * FPS):
        frame = start_frame + step
        timestamp = frame / FPS
        # Simple gap-keeping teacher: brake toward the lead speed when close.
        desired = lead_speed if gap < 25.0 else 12.0
        accel = float(np.clip(desired - ego_speed, -3.0, 1.5))
        ego_speed = max(0.0, ego_speed + accel / FPS)
        gap = max(4.0, gap - (ego_speed - lead_speed) / FPS)
        rel = ego_speed - lead_speed
        feature = {
            "ego_speed": ego_speed,
            "ego_acceleration": accel,
            "distance": gap,
            "relative_velocity": rel,
            "ttc": gap / rel if rel > 0.1 else 99.0,
            "obstacle_speed": lead_speed,
        }
        history.append(feature)
        history = history[-HISTORY:]
        speeds.append(ego_speed)
        row = {"frame": frame, "scenario": "lead_follow", "episode_id": episode,
               "ego_speed_now": ego_speed}
        for lag in range(HISTORY):
            src = history[-1 - lag] if lag < len(history) else history[0]
            for name in BASE_FEATURE_COLS:
                row[f"{name}_t-{lag}"] = src[name]
        rows.append(row)
        # The radar runs at 10 Hz inside the 20 Hz loop: a new scan every
        # other frame, the same scan index on the frame in between.
        if step % 2 == 0:
            scan_index += 1
            detections = []
            for _ in range(8):
                detections.append(RadarDetection(
                    distance_m=gap + rng.normal(0.0, 0.3),
                    azimuth_rad=rng.normal(0.0, 0.01),
                    relative_velocity_mps=rel + rng.normal(0.0, 0.1),
                    snr_db=30.0 + rng.normal(0.0, 2.0),
                    source="direct", truth_object_id=7, semantic_tag=14,
                ))
            for _ in range(40):
                detections.append(RadarDetection(
                    distance_m=rng.uniform(5.0, 90.0),
                    azimuth_rad=rng.choice((-1.0, 1.0)) * rng.uniform(0.08, 0.6),
                    relative_velocity_mps=ego_speed + rng.normal(0.0, 0.1),
                    snr_db=22.0 + rng.normal(0.0, 3.0),
                    source="direct", truth_object_id=-1, semantic_tag=28,
                ))
            if ghosts and step % 6 == 0:
                for _ in range(3):
                    detections.append(RadarDetection(
                        distance_m=gap + 1.5 + rng.normal(0.0, 0.2),
                        azimuth_rad=rng.normal(0.0, 0.01),
                        relative_velocity_mps=rel * 0.9,
                        snr_db=26.0, source="ghost", truth_object_id=7,
                        truth_parent_object_id=7, semantic_tag=14,
                        bounce_type="type1", bounce_order=2,
                    ))
            log.append(frame, timestamp, scan_index, detections)
    return speeds


def make_collection(root, ghosts, seed=0):
    import pandas as pd

    rng = np.random.default_rng(seed)
    rows = []
    log = DetectionLog()
    all_speeds = []
    frame = 0
    for episode in range(3):
        speeds = _episode(rows, log, frame, f"ep_{episode}", rng, ghosts)
        all_speeds.extend(speeds)
        frame += 30 * FPS + 50
    df = pd.DataFrame(rows)
    # Label: smoothed future ego speed over the label horizon, per episode.
    label = []
    for _, part in df.groupby("episode_id", sort=False):
        s = part["ego_speed_now"].to_numpy()
        fut = np.array([s[min(i + 1, len(s) - 1): min(i + 11, len(s))].mean() for i in range(len(s))])
        label.append(pd.Series(fut, index=part.index))
    df["teacher_target_speed"] = pd.concat(label).sort_index()
    os.makedirs(root, exist_ok=True)
    csv = os.path.join(root, "data.csv")
    df.to_csv(csv, index=False)
    log.save(sidecar_path(csv))
    config = {
        "label_col": "teacher_target_speed", "fps": FPS, "max_target_speed_kmh": 60.0,
        "town": "Town04", "teacher": "gapkeep", "radar_backend": "realistic",
        "radar_profile": "rgd_regime_v1", "radar_config_signature": "smoke",
        "radar_config": None, "radar_ghost_injection": {"multipath_mode": "geometry" if ghosts else "off"},
        "radar_ghost_oracle": False, "radar_ghost_detector": None,
        "radar_ghost_detector_signature": None, "radar_ghost_threshold": None,
        "radar_ghost_model": None, "radar_ghost_feature_schema": None,
        "radar_range_m": 100.0, "radar_points_per_second": 240000,
        "base_feature_cols": BASE_FEATURE_COLS, "history_frames": HISTORY,
        "feature_cols": [f"{n}_t-{lag}" for lag in range(HISTORY) for n in BASE_FEATURE_COLS],
    }
    with open(os.path.join(root, "dataset_config.json"), "w", encoding="utf-8") as fh:
        json.dump(config, fh)
    return csv


def _run(script, *args):
    result = subprocess.run(
        [sys.executable, str(ROOT / script), *map(str, args)],
        cwd=str(ROOT), capture_output=True, text=True, timeout=900,
    )
    return result


@unittest.skipUnless(HAVE_TORCH, "torch not installed")
class TransformerChainSmokeTest(unittest.TestCase):
    def test_train_accept_counterfactual(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = os.path.join(tmp, "dataset_ghost")
            make_collection(data, ghosts=True)
            model_dir = os.path.join(tmp, "model_tf")
            train = _run(
                "train_target_speed_transformer.py", "--data", data, "--output", model_dir,
                "--epochs", "2", "--batch", "32", "--max-points", "64", "--d-model", "32",
                "--heads", "2", "--layers", "1", "--ff-dim", "64", "--device", "cpu",
                "--num-workers", "0",
            )
            self.assertEqual(train.returncode, 0, train.stdout[-3000:] + train.stderr[-3000:])
            self.assertTrue(os.path.exists(os.path.join(model_dir, "model_config.json")))
            with open(os.path.join(model_dir, "model_config.json"), encoding="utf-8") as fh:
                config = json.load(fh)
            self.assertEqual(config["model_type"], "transformer")
            self.assertEqual(config["radar_backend"], "realistic")
            self.assertEqual(config["max_points"], 64)

            # The probe must run end to end; a two-epoch model need not pass it.
            accept = _run("acceptance_test.py", "--model-dir", model_dir)
            self.assertIn("ACCEPTANCE TEST", accept.stdout, accept.stdout[-2000:] + accept.stderr[-2000:])
            self.assertNotIn("Traceback", accept.stderr)

            report = os.path.join(tmp, "counterfactual.json")
            cf = _run(
                "counterfactual_ghost_test.py", "--model-dir", model_dir, "--data", data,
                "--limit", "64", "--device", "cpu", "--output", report,
            )
            self.assertEqual(cf.returncode, 0, cf.stdout[-3000:] + cf.stderr[-3000:])
            with open(report, encoding="utf-8") as fh:
                summary = json.load(fh)
            self.assertTrue(summary, "counterfactual report is empty")

    def test_mlp_trainer_and_probe_on_the_same_collection(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = os.path.join(tmp, "dataset_clean")
            make_collection(data, ghosts=False, seed=1)
            model_dir = os.path.join(tmp, "model_mlp")
            train = _run(
                "train_throttle_brake.py", "--data", data,
                "--config", os.path.join(data, "dataset_config.json"),
                "--output", model_dir, "--epochs", "3", "--batch", "64",
            )
            self.assertEqual(train.returncode, 0, train.stdout[-3000:] + train.stderr[-3000:])
            for name in ("target_speed_mlp.pt", "scaler.pkl", "model_config.json"):
                self.assertTrue(os.path.exists(os.path.join(model_dir, name)), name)
            with open(os.path.join(model_dir, "model_config.json"), encoding="utf-8") as fh:
                config = json.load(fh)
            self.assertEqual(config.get("radar_backend"), "realistic")
            self.assertEqual(config.get("radar_config_signature"), "smoke")
            accept = _run("acceptance_test.py", "--model-dir", model_dir)
            self.assertIn("ACCEPTANCE TEST", accept.stdout, accept.stdout[-2000:] + accept.stderr[-2000:])
            self.assertNotIn("Traceback", accept.stderr)


if __name__ == "__main__":
    unittest.main()
