"""Point-cloud statistics shared by profile calibration and fidelity scoring.

Everything here reads a *prepared* sequence (the npz written by
`prepare_radar_ghost_dataset.py`) and returns plain NumPy arrays, so the same
code measures the real Radar Ghost Dataset and the synthetic CARLA export.
That symmetry is the point: a statistic is only a valid fidelity check if it
was computed the same way on both sides.

Ghost-to-parent relations need a parent. When the sequence carries
``instance_id`` and a ghost shares it with real points in the same frame (the
convention both the RGD annotation and the CARLA exporter use), that is the
parent. Otherwise the fallback is the centroid of the same-class real points
in the frame, which is what the RGD recordings, with one main object, make
reasonable. The mode actually used is reported so nobody mistakes one for
the other.
"""

import math

import numpy as np


FAMILY_NAMES = ("type1_second", "type2_second", "type2_third", "other_multipath")
CLASS_NAMES = {1: "pedestrian", 2: "cyclist", 3: "car", 4: "large_vehicle", 5: "motorcycle"}
MIN_PARENT_DOPPLER_MPS = 0.3


def bounce_family(bounce_type, bounce_order):
    """Map RGD CMTO bounce codes to the four reported families."""

    bounce_type = np.asarray(bounce_type, dtype=np.int64)
    bounce_order = np.asarray(bounce_order, dtype=np.int64)
    family = np.full(bounce_type.shape, "other_multipath", dtype="U16")
    family[(bounce_type == 1) & (bounce_order == 2)] = "type1_second"
    family[(bounce_type == 2) & (bounce_order == 2)] = "type2_second"
    family[(bounce_type == 2) & (bounce_order == 4)] = "type2_third"
    return family


def amplitude_to_db(amplitude, mode="auto"):
    """Amplitude in dB. ``auto`` guesses the stored unit and says so.

    A linear amplitude is positive and spans orders of magnitude; a stored dB
    value can be negative and rarely exceeds ~100. The guess is printed by
    the callers because getting it wrong shifts every amplitude statistic by
    a constant that the relative measures then cancel anyway.
    """

    values = np.asarray(amplitude, dtype=np.float64)
    if mode == "auto":
        finite = values[np.isfinite(values)]
        looks_like_db = finite.size > 0 and (finite.min() < 0.0 or finite.max() < 100.0)
        mode = "db" if looks_like_db else "linear"
    if mode == "db":
        return values, mode
    return 20.0 * np.log10(np.maximum(values, 1.0e-6)), "linear"


def frame_groups(sequence):
    """(order, starts, ends) grouping rows by (sensor, frame)."""

    frame = np.asarray(sequence["frame"], dtype=np.int64)
    sensor = np.asarray(sequence["sensor"], dtype=np.int64)
    if frame.size == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    keys = sensor * (int(frame.max()) + 1) + frame
    order = np.argsort(keys, kind="stable")
    sorted_keys = keys[order]
    boundaries = np.flatnonzero(sorted_keys[1:] != sorted_keys[:-1]) + 1
    starts = np.concatenate((np.array((0,), dtype=np.int64), boundaries))
    ends = np.concatenate((boundaries, np.array((len(order),), dtype=np.int64)))
    return order, starts, ends


def wasserstein_1d(a, b, quantiles=256):
    """1-D Wasserstein-1 distance via matched quantiles. NaN when either is empty."""

    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if a.size == 0 or b.size == 0:
        return float("nan")
    grid = (np.arange(quantiles) + 0.5) / quantiles
    return float(np.mean(np.abs(np.quantile(a, grid) - np.quantile(b, grid))))


def sequence_statistics(sequence, amplitude_mode="auto", parent_mode="auto"):
    """All per-sequence statistics as a dict of 1-D arrays (plus metadata).

    ``parent_mode`` is ``auto`` (instance link when it exists, same-class
    centroid otherwise) or ``class`` (always the same-class centroid). The
    fidelity comparison forces ``class`` on both domains when the real data
    has no instance links, so ghost-parent and lifetime statistics are
    computed at the same granularity on each side.
    """

    target = np.asarray(sequence["target"], dtype=np.int64)
    r = np.asarray(sequence["r_sc"], dtype=np.float64)
    phi = np.asarray(sequence["phi_sc"], dtype=np.float64)
    vr = np.asarray(sequence["vr_sc"], dtype=np.float64)
    amp_db, amp_unit = amplitude_to_db(sequence["amp"], amplitude_mode)
    class_id = np.asarray(sequence.get("class_id", np.full(r.shape, -1)), dtype=np.int64)
    family = bounce_family(
        sequence.get("bounce_type", np.zeros(r.shape)),
        sequence.get("bounce_order", np.zeros(r.shape)),
    )
    instance = sequence.get("instance_id")
    instance = None if instance is None else np.asarray(instance, dtype=np.int64)
    sensor = np.asarray(sequence["sensor"], dtype=np.int64)
    frame = np.asarray(sequence["frame"], dtype=np.int64)

    real = target == 0
    ghost = target == 1
    order, starts, ends = frame_groups(sequence)

    stats = {
        "points_per_frame": [],
        "real_per_frame": [],
        "ghost_per_frame": [],
        "ghost_fraction_per_frame": [],
        # Unlabelled points per scan (infrastructure and clutter) and the
        # absolute frame-median amplitude: the two numbers that place the
        # synthetic background where a real-trained detector expects it.
        "background_per_frame": [],
        "frame_median_amp_db": [],
        "real_rel_amp_db": [],
        "ghost_rel_amp_db": [],
        "real_abs_amp_db": amp_db[real],
        "ghost_abs_amp_db": amp_db[ghost],
        "real_range_m": r[real],
        "ghost_range_m": r[ghost],
        "real_abs_doppler_mps": np.abs(vr[real]),
        "ghost_abs_doppler_mps": np.abs(vr[ghost]),
        "object_points": [],
        "object_class": [],
        "object_range_spread_m": [],
        "object_azimuth_spread_rad": [],
        "object_doppler_std_mps": [],
        # Points per ghost cluster (one parent, one family, one scan) and
        # ghost clusters per labelled real object per scan: the two numbers
        # that set how many ghost points a scan carries.
        "ghost_cluster_points": [],
        "ghost_clusters_per_object": [],
    }
    for name in FAMILY_NAMES:
        stats[f"{name}_delta_range_m"] = []
        stats[f"{name}_delta_azimuth_rad"] = []
        stats[f"{name}_delta_amp_db"] = []
        stats[f"{name}_spreading_db"] = []
        stats[f"{name}_doppler_ratio"] = []
    ghost_runs = {}  # (sensor, parent key, family) -> {frame: mean amp dB}
    instance_pairs = 0
    class_pairs = 0

    for start, end in zip(starts, ends):
        idx = order[start:end]
        f_real = idx[real[idx]]
        f_ghost = idx[ghost[idx]]
        stats["points_per_frame"].append(len(idx))
        stats["real_per_frame"].append(len(f_real))
        stats["ghost_per_frame"].append(len(f_ghost))
        labeled = len(f_real) + len(f_ghost)
        stats["ghost_fraction_per_frame"].append(len(f_ghost) / labeled if labeled else np.nan)
        frame_median = float(np.median(amp_db[idx])) if len(idx) else 0.0
        stats["background_per_frame"].append(len(idx) - labeled)
        stats["frame_median_amp_db"].append(frame_median)
        stats["real_rel_amp_db"].extend((amp_db[f_real] - frame_median).tolist())
        stats["ghost_rel_amp_db"].extend((amp_db[f_ghost] - frame_median).tolist())

        # Real objects: per instance when available, otherwise per class.
        real_object_count = 0
        if len(f_real):
            object_key = instance[f_real] if instance is not None else class_id[f_real]
            real_object_count = int(np.unique(object_key).size)
            for key in np.unique(object_key):
                members = f_real[object_key == key]
                if len(members) < 2:
                    continue
                stats["object_points"].append(len(members))
                stats["object_class"].append(int(np.bincount(np.maximum(class_id[members], 0)).argmax()))
                low, high = np.quantile(r[members], (0.10, 0.90))
                stats["object_range_spread_m"].append(float(high - low))
                low, high = np.quantile(phi[members], (0.10, 0.90))
                stats["object_azimuth_spread_rad"].append(float(high - low))
                stats["object_doppler_std_mps"].append(float(np.std(vr[members])))

        # Ghost-to-parent relations.
        frame_ghost_groups = {}
        for g in f_ghost:
            parent = None
            parent_key = None
            if instance is not None and parent_mode != "class":
                same = f_real[instance[f_real] == instance[g]]
                if len(same):
                    parent = same
                    parent_key = ("i", int(instance[g]))
                    instance_pairs += 1
            if parent is None:
                same_class = f_real[class_id[f_real] == class_id[g]]
                if len(same_class):
                    parent = same_class
                    parent_key = ("c", int(class_id[g]))
                    class_pairs += 1
            if parent is None:
                continue
            name = str(family[g])
            r_parent = float(np.mean(r[parent]))
            stats[f"{name}_delta_range_m"].append(float(r[g] - r_parent))
            stats[f"{name}_delta_azimuth_rad"].append(float(phi[g] - np.mean(phi[parent])))
            stats[f"{name}_delta_amp_db"].append(float(amp_db[g] - np.mean(amp_db[parent])))
            stats[f"{name}_spreading_db"].append(
                40.0 * math.log10(max(float(r[g]), 1.0e-3) / max(r_parent, 1.0e-3))
            )
            vr_parent = float(np.mean(vr[parent]))
            if abs(vr_parent) > MIN_PARENT_DOPPLER_MPS:
                stats[f"{name}_doppler_ratio"].append(float(vr[g] / vr_parent))
            run_key = (int(sensor[g]), parent_key, name)
            per_frame = ghost_runs.setdefault(run_key, {})
            per_frame.setdefault(int(frame[g]), []).append(float(amp_db[g]))
            frame_ghost_groups[(parent_key, name)] = frame_ghost_groups.get((parent_key, name), 0) + 1
        stats["ghost_cluster_points"].extend(frame_ghost_groups.values())
        if real_object_count:
            stats["ghost_clusters_per_object"].append(len(frame_ghost_groups) / real_object_count)

    # Ghost persistence: runs of consecutive frames per (parent, family).
    lifetimes, fading_std, lag1 = [], [], []
    for per_frame in ghost_runs.values():
        frames_sorted = np.array(sorted(per_frame))
        means = np.array([np.mean(per_frame[int(f)]) for f in frames_sorted])
        breaks = np.flatnonzero(np.diff(frames_sorted) != 1) + 1
        for chunk_frames, chunk_means in zip(
            np.split(frames_sorted, breaks), np.split(means, breaks)
        ):
            lifetimes.append(len(chunk_frames))
            if len(chunk_means) >= 3:
                fading_std.append(float(np.std(chunk_means)))
            if len(chunk_means) >= 4:
                centred = chunk_means - chunk_means.mean()
                denominator = float(np.dot(centred, centred))
                if denominator > 1.0e-9:
                    lag1.append(float(np.dot(centred[:-1], centred[1:]) / denominator))
    stats["ghost_lifetime_frames"] = lifetimes
    stats["ghost_fading_std_db"] = fading_std
    stats["ghost_fading_lag1"] = lag1

    result = {key: np.asarray(value, dtype=np.float64) for key, value in stats.items()}
    result["object_class"] = np.asarray(stats["object_class"], dtype=np.int64)
    result["_meta"] = {
        "amplitude_unit": amp_unit,
        "parent_mode": parent_mode,
        "instance_pairs": instance_pairs,
        "class_pairs": class_pairs,
        "frames": int(len(starts)),
        "points": int(r.size),
        "real_points": int(real.sum()),
        "ghost_points": int(ghost.sum()),
    }
    return result


def merge_statistics(parts):
    """Concatenate per-sequence statistics; metadata counters are summed."""

    if not parts:
        return {}
    merged = {}
    keys = [key for key in parts[0] if key != "_meta"]
    for key in keys:
        merged[key] = np.concatenate([np.asarray(part[key]) for part in parts if key in part])
    meta = {
        "amplitude_unit": parts[0]["_meta"]["amplitude_unit"],
        "parent_mode": parts[0]["_meta"].get("parent_mode", "auto"),
    }
    for counter in ("instance_pairs", "class_pairs", "frames", "points", "real_points", "ghost_points"):
        meta[counter] = int(sum(part["_meta"][counter] for part in parts))
    merged["_meta"] = meta
    return merged


def summarize(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"count": 0}
    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "median": float(np.median(values)),
        "p10": float(np.percentile(values, 10)),
        "p90": float(np.percentile(values, 90)),
    }


def summarize_statistics(stats):
    return {
        key: summarize(value)
        for key, value in stats.items()
        if key not in ("_meta", "object_class")
    }
