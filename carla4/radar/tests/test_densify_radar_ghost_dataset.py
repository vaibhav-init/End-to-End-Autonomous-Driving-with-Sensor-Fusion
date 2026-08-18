"""Unit tests for the statistical densification script (no CARLA/torch)."""

import argparse
import json
import math
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from densify_radar_ghost_dataset import (  # noqa: E402
    DENSIFIED_DTYPE,
    RGD_FOV_RAD,
    RGD_MAX_DOPPLER_MPS,
    RGD_RANGE_MAX_M,
    RGD_RANGE_MIN_M,
    run_densify,
    run_stencil,
)


def _prepared_sequence(point_specs):
    """Build a prepared-style npz dict from (class_id, frame) point specs."""

    points = list(point_specs) or [(0, 0)]
    size = len(points)
    return {
        "frame": np.asarray([frame for _class_id, frame in points], dtype=np.int64),
        "frame_timestamp": np.arange(size, dtype=np.float32) * 0.1,
        "sensor": np.zeros(size, dtype=np.int8),
        "label_id": np.asarray(
            [1000 * class_id + 11 for class_id, _frame in points],
            dtype=np.int32,
        ),
        "target": np.zeros(size, dtype=np.int8),
        "class_id": np.asarray([class_id for class_id, _frame in points], dtype=np.int8),
        "is_main": np.ones(size, dtype=np.int8),
        "bounce_type": np.zeros(size, dtype=np.int8),
        "bounce_order": np.ones(size, dtype=np.int8),
        "sketchy": np.zeros(size, dtype=np.bool_),
    }


def _make_prepared_input(root, split_map):
    """Write a prepared dataset directory; split_map: path -> split."""

    root.mkdir(parents=True, exist_ok=True)
    sequences = []
    for index, (path_text, split) in enumerate(split_map.items()):
        npz_path = root / path_text
        npz_path.parent.mkdir(parents=True, exist_ok=True)
        record = _prepared_sequence([(1, 0)] * 5 + [(1, 1)] * 3)
        # Position class-1 points with a known spread around (10, 2).
        count = len(record["class_id"])
        r = np.full(count, 10.0)
        phi = np.full(count, math.atan2(2.0, 10.0))
        offsets = np.linspace(-0.4, 0.4, count)
        record["x_cc"] = (r * np.cos(phi) + offsets).astype(np.float32)
        record["y_cc"] = (r * np.sin(phi) + 0.3 * offsets).astype(np.float32)
        record["r_sc"] = np.hypot(record["x_cc"], record["y_cc"]).astype(np.float32)
        record["phi_sc"] = np.arctan2(record["y_cc"], record["x_cc"]).astype(np.float32)
        record["vr_sc"] = np.linspace(-1.2, 1.2, count).astype(np.float32)
        record["amp"] = np.full(count, 120.0, dtype=np.float32)
        np.savez_compressed(npz_path, **record)
        frame_key = (int(record["sensor"][0]), int(record["frame"][0]))
        sequences.append(
            {
                "name": f"seq{index}",
                "path": path_text,
                "scenario": f"scenario-{index:02d}",
                "sequence_id": f"scenario-{index:02d}",
                "split": split,
                "points": count,
                "labeled_points": count,
                "real_points": count,
                "ghost_points": 0,
                "sensor_frames": {str(frame_key[0]): [frame_key[1]]},
                "sensor_mapping": {"front": 0},
            }
        )
    with (root / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump({"sequences": sequences, "feature_schema": "radar_ghost_physical_v1"}, handle)


def _make_carla_h5(path, frames=3, points_per_frame=2):
    """Write a minimal CARLA collector-style H5 with pedestrian + background."""

    rows = []
    frame_timestamp = 0.0
    for frame in range(frames):
        for _ in range(points_per_frame):
            label = 1011 if _ % points_per_frame == 0 else 0  # direct ped or background
            rows.append(
                (
                    frame,
                    frame_timestamp,
                    frame_timestamp,
                    b"front",
                    25.0,
                    1.0,
                    25.0208,
                    0.03999,
                    1.4,
                    320.0,
                    f"uuid-{frame}-{_}".encode(),
                    label,
                    7,
                    b"direct",
                    7,
                    0,
                    b"direct",
                    1,
                    50.0,
                )
            )
        frame_timestamp += 0.1
    radar = np.asarray(rows, dtype=DENSIFIED_DTYPE)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("radar", data=radar, compression="gzip")
        handle.attrs["town"] = "Town04"
        handle.attrs["seed"] = 42


class StencilTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = Path(tempfile.mkdtemp(prefix="densify_stencil_"))
        self.addCleanup(shutil.rmtree, self.tempdir, ignore_errors=True)

    def test_stencil_measures_train_split_only(self):
        prepared = self.tempdir / "prepared"
        _make_prepared_input(
            prepared,
            {
                "sequences/train.npz": "train",
                "sequences/val.npz": "val",
                "sequences/test.npz": "test",
            },
        )
        output = self.tempdir / "stencil.json"
        args = argparse.Namespace(
            input=str(prepared),
            output=str(output),
            split="train",
            class_ids=[1, 2],
            radius_m=2.0,
        )
        stencil = run_stencil(args)
        self.assertTrue(output.is_file())
        classes = stencil["classes"]
        self.assertIn("1", classes)
        self.assertTrue(classes["1"]["available"])
        # Train points have a known spatial spread; val/test spreads are
        # identical in this fixture, but the assertion below proves the
        # stencil was built from the train manifest entries only.
        self.assertEqual(classes["1"]["point_count"], 8)
        std_dx = float(np.sqrt(classes["1"]["cov_xy"][0][0]))
        self.assertGreater(std_dx, 0.05)
        self.assertLess(std_dx, 0.6)
        # The fixture's per-cluster Doppler spread is ~0.42 m/s.
        self.assertGreater(classes["1"]["std_dv"], 0.3)
        self.assertAlmostEqual(classes["1"]["log1p_amp_mean"], math.log1p(120.0), places=3)
        # Class 2 has no samples in this fixture.
        self.assertFalse(classes["2"]["available"])

    def test_stencil_val_test_never_contaminate(self):
        prepared = self.tempdir / "prepared"
        # Only train holds class-1 points; val/test hold class-2 points that
        # would shift stats if they were read.
        (prepared / "sequences").mkdir(parents=True, exist_ok=True)
        train_record = _prepared_sequence([(1, 0)] * 5)
        count = len(train_record["class_id"])
        train_record["x_cc"] = np.full(count, 10.0, dtype=np.float32)
        train_record["y_cc"] = np.full(count, 2.0, dtype=np.float32)
        train_record["r_sc"] = np.hypot(train_record["x_cc"], train_record["y_cc"]).astype(np.float32)
        train_record["phi_sc"] = np.arctan2(train_record["y_cc"], train_record["x_cc"]).astype(np.float32)
        train_record["vr_sc"] = np.zeros(count, dtype=np.float32)
        train_record["amp"] = np.full(count, 50.0, dtype=np.float32)
        np.savez_compressed(prepared / "sequences/train.npz", **train_record)
        sequences = [
            {
                "name": "train",
                "path": "sequences/train.npz",
                "scenario": "scenario-01",
                "sequence_id": "scenario-01",
                "split": "train",
                "points": count,
                "labeled_points": count,
                "real_points": count,
                "ghost_points": 0,
                "sensor_frames": {"0": [0]},
                "sensor_mapping": {"front": 0},
            }
        ]
        # val/test entries point to a file that does not contain class 1, so
        # reading them would leave class 1 empty rather than larger.
        for name, split in (("val", "val"), ("test", "test")):
            sequences.append(
                {
                    "name": name,
                    "path": "sequences/missing.npz",
                    "scenario": f"scenario-{name}",
                    "sequence_id": f"scenario-{name}",
                    "split": split,
                    "points": 1,
                    "labeled_points": 1,
                    "real_points": 1,
                    "ghost_points": 0,
                    "sensor_frames": {"0": [0]},
                    "sensor_mapping": {"front": 0},
                }
            )
        with (prepared / "manifest.json").open("w", encoding="utf-8") as handle:
            json.dump({"sequences": sequences, "feature_schema": "radar_ghost_physical_v1"}, handle)
        output = self.tempdir / "stencil.json"
        args = argparse.Namespace(
            input=str(prepared),
            output=str(output),
            split="train",
            class_ids=[1, 2],
            radius_m=2.0,
        )
        stencil = run_stencil(args)
        self.assertTrue(stencil["classes"]["1"]["available"])
        # Only the train sequence contributed points.
        self.assertEqual(stencil["classes"]["1"]["point_count"], 5)


class DensifyTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = Path(tempfile.mkdtemp(prefix="densify_run_"))
        self.addCleanup(shutil.rmtree, self.tempdir, ignore_errors=True)

    def test_densify_scales_to_rdg_density_and_inherits_labels(self):
        prepared = self.tempdir / "prepared"
        _make_prepared_input(
            prepared,
            {"sequences/train.npz": "train"},
        )
        stencil_path = self.tempdir / "stencil.json"
        run_stencil(
            argparse.Namespace(
                input=str(prepared),
                output=str(stencil_path),
                split="train",
                class_ids=[1, 2],
                radius_m=2.0,
            )
        )
        carla_input = self.tempdir / "carla"
        (carla_input / "train").mkdir(parents=True, exist_ok=True)
        _make_carla_h5(carla_input / "train" / "seq.h5", frames=3, points_per_frame=2)
        output = self.tempdir / "densified"
        run_densify(
            argparse.Namespace(
                carla_input=str(carla_input),
                stencil=str(stencil_path),
                output=str(output),
                points_per_frame=800.0,
                seed=7,
            )
        )
        out_path = output / "train" / "seq.h5"
        self.assertTrue(out_path.is_file())
        with h5py.File(out_path, "r") as handle:
            radar = handle["radar"][:]
            self.assertTrue(bool(handle.attrs.get("densified")))
        # 3 frames x ~800 points, with the single background point per frame
        # passed through unmodified.
        self.assertGreaterEqual(len(radar), 3 * 799)
        self.assertLessEqual(len(radar), 3 * 801)
        background = radar[radar["label_id"] == 0]
        self.assertEqual(len(background), 3)
        # Labels inherited: all non-background points carry the parent CMTO.
        ped = radar[radar["label_id"] == 1011]
        self.assertGreater(len(ped), 0)
        self.assertTrue(np.all(radar["label_id"] != 1012))
        # Physics: x_cc/y_cc/r_sc/phi_sc stay mutually consistent, and every
        # point stays inside the RGD envelope.
        r_check = np.hypot(radar["x_cc"], radar["y_cc"])
        np.testing.assert_allclose(r_check, radar["r_sc"], rtol=1.0e-5, atol=1.0e-4)
        self.assertTrue(np.all(radar["r_sc"] >= RGD_RANGE_MIN_M))
        self.assertTrue(np.all(radar["r_sc"] <= RGD_RANGE_MAX_M))
        self.assertTrue(np.all(np.abs(radar["phi_sc"]) <= RGD_FOV_RAD + 1.0e-5))
        self.assertTrue(np.all(np.abs(radar["vr_sc"]) <= RGD_MAX_DOPPLER_MPS + 1.0e-4))
        self.assertTrue(np.all(radar["amp"] > 0.0))
        # Densified Doppler stays around the parent's 1.4 m/s (physics mean).
        ped_doppler = np.abs(ped["vr_sc"])
        self.assertGreater(float(np.mean(ped_doppler)), 0.5)
        self.assertLess(float(np.mean(ped_doppler)), 2.5)

    def test_densify_skips_classes_without_stencil(self):
        prepared = self.tempdir / "prepared"
        _make_prepared_input(
            prepared,
            {"sequences/train.npz": "train"},
        )
        stencil_path = self.tempdir / "stencil.json"
        # Stencil measured for class 2 only; class 1 must pass through.
        run_stencil(
            argparse.Namespace(
                input=str(prepared),
                output=str(stencil_path),
                split="train",
                class_ids=[2],
                radius_m=2.0,
            )
        )
        carla_input = self.tempdir / "carla"
        (carla_input / "train").mkdir(parents=True, exist_ok=True)
        _make_carla_h5(carla_input / "train" / "seq.h5", frames=1, points_per_frame=4)
        output = self.tempdir / "densified"
        run_densify(
            argparse.Namespace(
                carla_input=str(carla_input),
                stencil=str(stencil_path),
                output=str(output),
                points_per_frame=800.0,
                seed=3,
            )
        )
        with h5py.File(output / "train" / "seq.h5", "r") as handle:
            radar = handle["radar"][:]
        # Nothing densifiable -> original 4 points pass through unchanged.
        self.assertEqual(len(radar), 4)


if __name__ == "__main__":
    unittest.main()
