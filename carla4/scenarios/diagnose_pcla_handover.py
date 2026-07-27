#!/usr/bin/env python3
"""
Diagnose whether PCLA braking is caused by handover or by the visible lead.

This is intentionally separate from the research scenario runner. It runs four
controlled cases and records PCLA's proposed control independently from the
control that is actually applied during staging:

  D0  Empty road, PCLA controls naturally.
  D1  Empty road, speed controller stages to the target speed, then handover.
  D2  Constant-speed lead at a safe gap, gap controller stages, then handover.
  D3  Constant-speed lead at the tight experimental gap, then handover.

No lead-vehicle braking event is used. This makes the interpretation direct:
braking in D1 indicates a handover/staging issue; braking in D2/D3 is already
present before any hypothetical lead-deceleration event.

Run from the repository root in the PCLA environment:

  python carla4/scenarios/diagnose_pcla_handover.py \
    --host 127.0.0.1 --port 2000 --town Town04 \
    --cases D0 D1 D2 D3 --output pcla_handover_diagnosis

Paste the terminal block headed "PASTE THIS BLOCK" when asking for analysis.
The output directory also contains the complete console log and per-frame CSVs.
"""

import argparse
import contextlib
import csv
import datetime as dt
import gc
import json
import math
import os
import platform
import random
import subprocess
import sys
import traceback

import carla

from config import CARLA_HOST, CARLA_PORT, DEFAULT_TOWN, FPS
from drivers import make_driver
from spawn_utils import get_highway_spawns, spawn_npc_in_ego_direction
from staging import GapKeepController, SpeedController


BRAKE_THRESHOLD = 0.30
FULL_BRAKE_THRESHOLD = 0.90
CSV_FIELDS = [
    "wall_time_iso",
    "case",
    "case_description",
    "case_step",
    "decision_world_frame",
    "result_world_frame",
    "sim_time_s",
    "driver_frame_before",
    "phase",
    "staging_active",
    "stable_condition",
    "stable_frames",
    "handover_flag",
    "handover_step",
    "handover_reason",
    "policy_throttle",
    "policy_brake",
    "policy_steer",
    "policy_brake_active",
    "policy_full_brake",
    "applied_throttle",
    "applied_brake",
    "applied_steer",
    "applied_brake_active",
    "applied_full_brake",
    "ego_speed_kmh",
    "ego_accel_mps2",
    "ego_x",
    "ego_y",
    "ego_yaw_deg",
    "lead_present",
    "lead_speed_kmh",
    "gap_euclidean_m",
    "gap_longitudinal_m",
    "lateral_offset_m",
    "relative_speed_mps",
    "same_lane",
    "lead_ahead",
    "collision",
]

CASE_DESCRIPTIONS = {
    "D0": "empty road; PCLA controls naturally",
    "D1": "empty road; stage to target speed, then hand over to PCLA",
    "D2": "constant-speed lead at safe gap; stage, then hand over to PCLA",
    "D3": "constant-speed lead at tight gap; stage, then hand over to PCLA",
}


class Tee:
    """Write subprocess/model prints to both the terminal and a log file."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for stream in self.streams:
            stream.write(text)
            stream.flush()
        return len(text)

    def flush(self):
        for stream in self.streams:
            stream.flush()

    def fileno(self):
        """Expose the terminal descriptor required by faulthandler."""
        return self.streams[0].fileno()

    def isatty(self):
        return any(getattr(stream, "isatty", lambda: False)() for stream in self.streams)


class FrameLogger:
    """Write every diagnostic frame to CSV and retain rows for the summary."""

    def __init__(self, path):
        self.path = path
        self.rows = []
        self._file = open(path, "w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=CSV_FIELDS)
        self._writer.writeheader()

    def write(self, row):
        normalized = {name: row.get(name, "") for name in CSV_FIELDS}
        self._writer.writerow(normalized)
        self._file.flush()
        self.rows.append(normalized)

    def close(self):
        if not self._file.closed:
            self._file.close()


def _round(value, places=5):
    if value is None:
        return ""
    return round(float(value), places)


def _speed_kmh(actor):
    velocity = actor.get_velocity()
    return math.sqrt(
        velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2
    ) * 3.6


def _relative_geometry(ego, lead, carla_map):
    if lead is None or not lead.is_alive:
        return {
            "lead_present": 0,
            "lead_speed_kmh": "",
            "gap_euclidean_m": "",
            "gap_longitudinal_m": "",
            "lateral_offset_m": "",
            "relative_speed_mps": "",
            "same_lane": "",
            "lead_ahead": "",
        }

    ego_tf = ego.get_transform()
    lead_tf = lead.get_transform()
    dx = lead_tf.location.x - ego_tf.location.x
    dy = lead_tf.location.y - ego_tf.location.y
    dz = lead_tf.location.z - ego_tf.location.z
    fwd = ego_tf.get_forward_vector()
    right_x = -fwd.y
    right_y = fwd.x
    longitudinal = dx * fwd.x + dy * fwd.y
    lateral = dx * right_x + dy * right_y
    euclidean = math.sqrt(dx * dx + dy * dy + dz * dz)
    ego_speed = _speed_kmh(ego)
    lead_speed = _speed_kmh(lead)

    ego_wp = carla_map.get_waypoint(
        ego_tf.location,
        project_to_road=True,
        lane_type=carla.LaneType.Driving,
    )
    lead_wp = carla_map.get_waypoint(
        lead_tf.location,
        project_to_road=True,
        lane_type=carla.LaneType.Driving,
    )
    same_lane = int(
        ego_wp is not None
        and lead_wp is not None
        and ego_wp.road_id == lead_wp.road_id
        and ego_wp.lane_id == lead_wp.lane_id
    )
    return {
        "lead_present": 1,
        "lead_speed_kmh": _round(lead_speed),
        "gap_euclidean_m": _round(euclidean),
        "gap_longitudinal_m": _round(longitudinal),
        "lateral_offset_m": _round(lateral),
        "relative_speed_mps": _round((ego_speed - lead_speed) / 3.6),
        "same_lane": same_lane,
        "lead_ahead": int(longitudinal > 0.0),
    }


def _cleanup_actor(actor):
    if actor is not None and actor.is_alive:
        try:
            actor.destroy()
        except RuntimeError:
            pass


def _git_revision():
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")),
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def _case_target_gap(case_name, args):
    if case_name == "D2":
        return args.safe_gap
    if case_name == "D3":
        return args.tight_gap
    return None


def _spawn_ego(world, spawn_transform):
    blueprint = world.get_blueprint_library().find("vehicle.tesla.model3")
    ego = world.try_spawn_actor(blueprint, spawn_transform)
    if ego is None:
        raise RuntimeError("Failed to spawn ego at the selected diagnostic transform")
    for _ in range(3):
        world.tick()
    return ego


def _setup_lead(client, world, carla_map, ego, gap_m, speed_kmh):
    lead = spawn_npc_in_ego_direction(world, carla_map, ego, gap_m)
    if lead is None:
        raise RuntimeError(f"Failed to spawn lead vehicle at {gap_m:.1f} m")

    tm = client.get_trafficmanager(8000)
    tm_port = tm.get_port()
    lead.set_autopilot(True, tm_port)
    tm.set_desired_speed(lead, speed_kmh)
    tm.ignore_lights_percentage(lead, 100)
    tm.ignore_signs_percentage(lead, 100)
    tm.auto_lane_change(lead, False)
    return lead


def _mean(rows, field):
    values = [float(row[field]) for row in rows if row.get(field, "") != ""]
    return sum(values) / len(values) if values else None


def _fraction(rows, field):
    values = [int(row[field]) for row in rows if row.get(field, "") != ""]
    return sum(values) / len(values) if values else None


def _first(rows, predicate):
    for row in rows:
        if predicate(row):
            return row
    return None


def summarize_case(case_name, rows, handover_step, handover_reason, stable_reached):
    staging = [row for row in rows if int(row["staging_active"]) == 1]
    policy_control = [
        row for row in rows
        if row["phase"] in ("handover", "policy_control")
    ]
    if case_name == "D0":
        # Ignore the first second of sensor/model bootstrap for the natural run.
        policy_control = [
            row for row in policy_control if int(row["case_step"]) >= FPS
        ]

    pre_window = staging[-FPS:]
    handover_row = _first(rows, lambda row: int(row["handover_flag"]) == 1)
    first_brake = _first(
        policy_control,
        lambda row: float(row["policy_brake"]) > BRAKE_THRESHOLD,
    )
    one_second_row = None
    if handover_step is not None:
        one_second_row = _first(
            rows,
            lambda row: int(row["case_step"]) >= handover_step + FPS,
        )

    summary = {
        "case": case_name,
        "description": CASE_DESCRIPTIONS[case_name],
        "frames_logged": len(rows),
        "stable_reached": bool(stable_reached),
        "handover_step": handover_step,
        "handover_reason": handover_reason,
        "staging_policy_mean_throttle": _mean(staging, "policy_throttle"),
        "staging_policy_mean_brake": _mean(staging, "policy_brake"),
        "staging_policy_zero_throttle_fraction": (
            sum(float(row["policy_throttle"]) <= 0.01 for row in staging) / len(staging)
            if staging else None
        ),
        "staging_policy_brake_fraction": _fraction(staging, "policy_brake_active"),
        "staging_policy_full_brake_fraction": _fraction(staging, "policy_full_brake"),
        "pre_handover_1s_mean_policy_brake": _mean(pre_window, "policy_brake"),
        "pre_handover_1s_full_brake_fraction": _fraction(
            pre_window, "policy_full_brake"
        ),
        "handover_policy_throttle": (
            float(handover_row["policy_throttle"]) if handover_row else None
        ),
        "handover_policy_brake": (
            float(handover_row["policy_brake"]) if handover_row else None
        ),
        "handover_ego_speed_kmh": (
            float(handover_row["ego_speed_kmh"]) if handover_row else None
        ),
        "handover_gap_m": (
            float(handover_row["gap_euclidean_m"])
            if handover_row and handover_row["gap_euclidean_m"] != ""
            else None
        ),
        "policy_control_brake_fraction": _fraction(
            policy_control, "policy_brake_active"
        ),
        "policy_control_full_brake_fraction": _fraction(
            policy_control, "policy_full_brake"
        ),
        "first_policy_brake_step": (
            int(first_brake["case_step"]) if first_brake else None
        ),
        "first_policy_brake_relative_to_handover_s": (
            (int(first_brake["case_step"]) - handover_step) / FPS
            if first_brake is not None and handover_step is not None
            else None
        ),
        "ego_speed_1s_after_handover_kmh": (
            float(one_second_row["ego_speed_kmh"]) if one_second_row else None
        ),
        "collision": any(int(row["collision"]) == 1 for row in rows),
    }
    return summary


def _fmt(value, digits=3):
    if value is None:
        return "NA"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (float, int)):
        return f"{value:.{digits}f}"
    return str(value)


def print_case_summary(summary):
    print(f"\n--- {summary['case']} SUMMARY: {summary['description']} ---")
    print(
        "  stable={stable} handover_step={step} reason={reason} "
        "speed={speed}km/h gap={gap}m".format(
            stable=summary["stable_reached"],
            step=summary["handover_step"],
            reason=summary["handover_reason"],
            speed=_fmt(summary["handover_ego_speed_kmh"], 1),
            gap=_fmt(summary["handover_gap_m"], 1),
        )
    )
    print(
        "  staging proposed: mean throttle={thr}, mean brake={brk}, "
        "zero-throttle={zero}, brake-active={active}, full-brake={full}".format(
            thr=_fmt(summary["staging_policy_mean_throttle"]),
            brk=_fmt(summary["staging_policy_mean_brake"]),
            zero=_fmt(summary["staging_policy_zero_throttle_fraction"]),
            active=_fmt(summary["staging_policy_brake_fraction"]),
            full=_fmt(summary["staging_policy_full_brake_fraction"]),
        )
    )
    print(
        "  final staging second: mean brake={mean}, full-brake={full}".format(
            mean=_fmt(summary["pre_handover_1s_mean_policy_brake"]),
            full=_fmt(summary["pre_handover_1s_full_brake_fraction"]),
        )
    )
    print(
        "  at handover: policy throttle={thr}, policy brake={brk}; "
        "speed after 1s={speed}km/h".format(
            thr=_fmt(summary["handover_policy_throttle"]),
            brk=_fmt(summary["handover_policy_brake"]),
            speed=_fmt(summary["ego_speed_1s_after_handover_kmh"], 1),
        )
    )
    print(
        "  under PCLA control: brake-active={active}, full-brake={full}, "
        "first brake relative to handover={onset}s, collision={collision}".format(
            active=_fmt(summary["policy_control_brake_fraction"]),
            full=_fmt(summary["policy_control_full_brake_fraction"]),
            onset=_fmt(summary["first_policy_brake_relative_to_handover_s"]),
            collision=summary["collision"],
        )
    )


def interpret_summaries(summaries):
    by_case = {summary["case"]: summary for summary in summaries}
    messages = []

    d0 = by_case.get("D0")
    d1 = by_case.get("D1")
    d2 = by_case.get("D2")
    d3 = by_case.get("D3")

    if d0:
        d0_full = d0.get("policy_control_full_brake_fraction")
        if d0_full is not None and d0_full >= 0.5:
            messages.append(
                "D0: PCLA full-brakes on an empty road for at least half the "
                "measured frames. Investigate route/PCLA setup before scenarios."
            )
        else:
            messages.append(
                "D0: PCLA does not predominantly full-brake on the natural "
                "empty-road baseline."
            )

    if d1:
        d1_pre = d1.get("pre_handover_1s_full_brake_fraction")
        d1_handover = d1.get("handover_policy_brake")
        if (
            (d1_pre is not None and d1_pre >= 0.5)
            or (d1_handover is not None and d1_handover >= FULL_BRAKE_THRESHOLD)
        ):
            messages.append(
                "D1: PCLA requests braking during empty-road staging or at "
                "handover. This supports a staging/state-distribution problem."
            )
        else:
            messages.append(
                "D1: Empty-road staging alone does not produce persistent "
                "pre-handover full braking."
            )

    if d2:
        d2_pre = d2.get("pre_handover_1s_full_brake_fraction")
        if d2_pre is not None and d2_pre >= 0.5:
            messages.append(
                "D2: PCLA is already full-braking with the safe-gap lead. "
                "Check achieved gap/speed and route state before blaming 15 m."
            )
        else:
            messages.append(
                "D2: PCLA is not predominantly full-braking during the final "
                "second at the safe lead gap."
            )

    if d3:
        d3_pre = d3.get("pre_handover_1s_full_brake_fraction")
        d3_handover = d3.get("handover_policy_brake")
        if (
            (d3_pre is not None and d3_pre >= 0.5)
            or (d3_handover is not None and d3_handover >= FULL_BRAKE_THRESHOLD)
        ):
            messages.append(
                "D3: PCLA was already requesting full braking before any lead "
                "deceleration. S2 cannot treat the handover brake as reaction latency."
            )
        else:
            messages.append(
                "D3: PCLA was not persistently full-braking before handover at "
                "the tight gap; inspect the first post-handover frames."
            )

    if d2 and d3:
        d2_pre = d2.get("pre_handover_1s_mean_policy_brake")
        d3_pre = d3.get("pre_handover_1s_mean_policy_brake")
        if d2_pre is not None and d3_pre is not None:
            if d3_pre >= d2_pre + 0.25:
                messages.append(
                    "D2 vs D3: braking increases materially at the tight gap, "
                    "which is evidence of a gap-dependent policy response."
                )
            else:
                messages.append(
                    "D2 vs D3: the gap does not explain a large brake difference; "
                    "inspect empty-road staging, achieved states, and CSV traces."
                )

    return messages


def run_case(case_name, args, client, world, spawn_transform, output_dir):
    carla_map = world.get_map()
    target_gap = _case_target_gap(case_name, args)
    csv_path = os.path.join(output_dir, f"{case_name.lower()}_frames.csv")
    frame_logger = FrameLogger(csv_path)

    ego = None
    lead = None
    collision_sensor = None
    driver = None
    collision = [False]
    handover_step = 0 if case_name == "D0" else None
    handover_reason = "natural_control" if case_name == "D0" else "not_reached"
    stable_frames = 0
    stable_reached = case_name == "D0"
    prev_speed_mps = 0.0

    max_stage_steps = int(args.max_stage_seconds * FPS)
    post_handover_steps = int(args.post_handover_seconds * FPS)
    natural_steps = int(args.natural_seconds * FPS)
    required_stable_frames = max(1, int(args.stable_seconds * FPS))
    speed_controller = SpeedController(args.target_speed_kmh / 3.6, dt=1.0 / FPS)
    gap_controller = (
        GapKeepController(target_gap, dt=1.0 / FPS)
        if target_gap is not None
        else None
    )

    print("\n" + "=" * 88)
    print(f"{case_name}: {CASE_DESCRIPTIONS[case_name]}")
    print("=" * 88)

    try:
        random.seed(args.seed)
        ego = _spawn_ego(world, spawn_transform)
        if target_gap is not None:
            lead = _setup_lead(
                client,
                world,
                carla_map,
                ego,
                target_gap,
                args.target_speed_kmh,
            )

        driver = make_driver(
            "pcla",
            pcla_agent=args.pcla_agent,
            debug_every=0,
        )
        driver.setup(world, ego, carla_map, client)

        collision_bp = world.get_blueprint_library().find("sensor.other.collision")
        collision_sensor = world.spawn_actor(
            collision_bp, carla.Transform(), attach_to=ego
        )

        def on_collision(_event):
            collision[0] = True

        collision_sensor.listen(on_collision)

        total_limit = (
            natural_steps
            if case_name == "D0"
            else max_stage_steps + post_handover_steps + 1
        )
        for case_step in range(total_limit):
            decision_snapshot = world.get_snapshot()
            policy = driver.get_control(ego, world)
            driver_frame_before = getattr(driver, "_frame", 0) - 1

            ego_speed_before = _speed_kmh(ego)
            lead_speed_before = _speed_kmh(lead) if lead and lead.is_alive else None
            geometry_before = _relative_geometry(ego, lead, carla_map)

            handover_flag = 0
            stable_condition = False
            staging_active = case_name != "D0" and handover_step is None

            if staging_active:
                if case_name == "D1":
                    stable_condition = abs(ego_speed_before - args.target_speed_kmh) <= 2.0
                else:
                    gap_error = abs(
                        float(geometry_before["gap_euclidean_m"]) - target_gap
                    )
                    relative_speed = abs(float(geometry_before["relative_speed_mps"]))
                    stable_condition = (
                        ego_speed_before >= args.target_speed_kmh - 5.0
                        and lead_speed_before is not None
                        and lead_speed_before >= args.target_speed_kmh - 5.0
                        and gap_error <= args.gap_tolerance
                        and relative_speed <= args.relative_speed_tolerance
                        and int(geometry_before["same_lane"]) == 1
                        and int(geometry_before["lead_ahead"]) == 1
                    )

                stable_frames = stable_frames + 1 if stable_condition else 0
                if stable_frames >= required_stable_frames:
                    handover_step = case_step
                    handover_reason = "stable_target_reached"
                    stable_reached = True
                    handover_flag = 1
                    staging_active = False
                elif case_step >= max_stage_steps - 1:
                    handover_step = case_step
                    handover_reason = "stage_timeout_forced"
                    handover_flag = 1
                    staging_active = False

            if staging_active:
                if case_name == "D1":
                    applied_throttle, applied_brake = speed_controller.run_step(
                        ego_speed_before / 3.6
                    )
                else:
                    applied_throttle, applied_brake = gap_controller.run_step(
                        float(geometry_before["gap_euclidean_m"]),
                        ego_speed_before / 3.6,
                        lead_speed_before / 3.6,
                    )
                applied = carla.VehicleControl(
                    throttle=applied_throttle,
                    brake=applied_brake,
                    steer=policy.steer,
                )
                phase = "staging"
            else:
                applied = policy
                phase = "handover" if handover_flag else "policy_control"

            ego.apply_control(applied)
            result_frame = world.tick()
            result_snapshot = world.get_snapshot()

            ego_speed = _speed_kmh(ego)
            ego_speed_mps = ego_speed / 3.6
            ego_accel = (
                (ego_speed_mps - prev_speed_mps) * FPS if case_step > 0 else 0.0
            )
            prev_speed_mps = ego_speed_mps
            ego_tf = ego.get_transform()
            geometry = _relative_geometry(ego, lead, carla_map)
            actual = ego.get_control()

            row = {
                "wall_time_iso": dt.datetime.now(dt.timezone.utc).isoformat(),
                "case": case_name,
                "case_description": CASE_DESCRIPTIONS[case_name],
                "case_step": case_step,
                "decision_world_frame": decision_snapshot.frame,
                "result_world_frame": result_frame,
                "sim_time_s": _round(result_snapshot.timestamp.elapsed_seconds),
                "driver_frame_before": driver_frame_before,
                "phase": phase,
                "staging_active": int(staging_active),
                "stable_condition": int(stable_condition),
                "stable_frames": stable_frames,
                "handover_flag": handover_flag,
                "handover_step": handover_step if handover_step is not None else "",
                "handover_reason": handover_reason,
                "policy_throttle": _round(policy.throttle),
                "policy_brake": _round(policy.brake),
                "policy_steer": _round(policy.steer),
                "policy_brake_active": int(policy.brake > BRAKE_THRESHOLD),
                "policy_full_brake": int(policy.brake >= FULL_BRAKE_THRESHOLD),
                "applied_throttle": _round(actual.throttle),
                "applied_brake": _round(actual.brake),
                "applied_steer": _round(actual.steer),
                "applied_brake_active": int(actual.brake > BRAKE_THRESHOLD),
                "applied_full_brake": int(actual.brake >= FULL_BRAKE_THRESHOLD),
                "ego_speed_kmh": _round(ego_speed),
                "ego_accel_mps2": _round(ego_accel),
                "ego_x": _round(ego_tf.location.x),
                "ego_y": _round(ego_tf.location.y),
                "ego_yaw_deg": _round(ego_tf.rotation.yaw),
                **geometry,
                "collision": int(collision[0]),
            }
            frame_logger.write(row)

            if (
                case_step % FPS == 0
                or handover_flag
                or policy.brake >= FULL_BRAKE_THRESHOLD
                and case_step % 5 == 0
            ):
                gap_text = (
                    f"{float(geometry['gap_euclidean_m']):5.1f}m"
                    if geometry["gap_euclidean_m"] != ""
                    else "  N/A "
                )
                print(
                    f"[{case_name}] step={case_step:03d} frame={result_frame} "
                    f"phase={phase:14s} speed={ego_speed:5.1f}km/h "
                    f"gap={gap_text} stable={stable_frames:02d} "
                    f"proposed={policy.throttle:.2f}/{policy.brake:.2f} "
                    f"applied={actual.throttle:.2f}/{actual.brake:.2f}"
                )

            if collision[0]:
                print(f"[{case_name}] COLLISION at step {case_step}")
                break
            if (
                case_name != "D0"
                and handover_step is not None
                and case_step >= handover_step + post_handover_steps
            ):
                break

        summary = summarize_case(
            case_name,
            frame_logger.rows,
            handover_step,
            handover_reason,
            stable_reached,
        )
        print_case_summary(summary)
        print(f"  frame log: {csv_path}")
        return summary
    finally:
        frame_logger.close()
        if lead is not None and lead.is_alive:
            try:
                lead.set_autopilot(False)
            except RuntimeError:
                pass
        # PCLA cleanup may destroy the ego and every sensor in the world.
        if driver is not None:
            try:
                driver.cleanup()
            except Exception as exc:
                print(f"[{case_name}] warning during PCLA cleanup: {exc}")
        _cleanup_actor(collision_sensor)
        _cleanup_actor(lead)
        _cleanup_actor(ego)
        gc.collect()
        try:
            for _ in range(3):
                world.tick()
        except RuntimeError:
            pass


def run_diagnostics(args, output_dir):
    client = carla.Client(args.host, args.port)
    client.set_timeout(60.0)
    world = client.get_world()
    current_town = world.get_map().name.split("/")[-1]
    if current_town != args.town:
        print(f"Loading {args.town} (current map: {current_town}) ...")
        world = client.load_world(args.town)

    original_settings = world.get_settings()
    tm = client.get_trafficmanager(8000)
    summaries = []

    try:
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 1.0 / FPS
        world.apply_settings(settings)
        tm.set_synchronous_mode(True)
        world.set_weather(carla.WeatherParameters.ClearNoon)
        world.tick()

        carla_map = world.get_map()
        highway_spawns = get_highway_spawns(carla_map, min_straight_m=150.0)
        if not highway_spawns:
            highway_spawns = get_highway_spawns(carla_map)
        if not highway_spawns:
            raise RuntimeError("No suitable highway spawn points found")
        highway_spawns = sorted(
            highway_spawns,
            key=lambda tf: (tf.location.x, tf.location.y, tf.rotation.yaw),
        )
        spawn_transform = highway_spawns[args.seed % len(highway_spawns)]

        metadata = {
            "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "git_revision": _git_revision(),
            "python": sys.version,
            "platform": platform.platform(),
            "carla_client_version": client.get_client_version(),
            "carla_server_version": client.get_server_version(),
            "map": carla_map.name,
            "fps": FPS,
            "fixed_delta_seconds": 1.0 / FPS,
            "pcla_agent": args.pcla_agent,
            "cases": args.cases,
            "seed": args.seed,
            "target_speed_kmh": args.target_speed_kmh,
            "safe_gap_m": args.safe_gap,
            "tight_gap_m": args.tight_gap,
            "stable_seconds": args.stable_seconds,
            "max_stage_seconds": args.max_stage_seconds,
            "post_handover_seconds": args.post_handover_seconds,
            "natural_seconds": args.natural_seconds,
            "spawn": {
                "x": spawn_transform.location.x,
                "y": spawn_transform.location.y,
                "z": spawn_transform.location.z,
                "yaw": spawn_transform.rotation.yaw,
            },
        }
        metadata_path = os.path.join(output_dir, "metadata.json")
        with open(metadata_path, "w", encoding="utf-8") as file:
            json.dump(metadata, file, indent=2)

        print("=" * 88)
        print("PCLA HANDOVER DIAGNOSTIC")
        print("=" * 88)
        print(f"Output:       {os.path.abspath(output_dir)}")
        print(f"CARLA:        {metadata['carla_server_version']} at {args.host}:{args.port}")
        print(f"Map:          {carla_map.name}")
        print(f"PCLA agent:   {args.pcla_agent}")
        print(f"Cases:        {' '.join(args.cases)}")
        print(f"Target speed: {args.target_speed_kmh:.1f} km/h")
        print(f"Gaps:         safe={args.safe_gap:.1f} m, tight={args.tight_gap:.1f} m")
        print(f"Git revision: {metadata['git_revision']}")
        print("=" * 88)

        for case_name in args.cases:
            try:
                summaries.append(
                    run_case(
                        case_name,
                        args,
                        client,
                        world,
                        spawn_transform,
                        output_dir,
                    )
                )
            except Exception as exc:
                print(f"\n[{case_name}] FAILED: {exc}")
                traceback.print_exc()
                summaries.append({
                    "case": case_name,
                    "description": CASE_DESCRIPTIONS[case_name],
                    "error": repr(exc),
                })

        summary_path = os.path.join(output_dir, "summary.json")
        with open(summary_path, "w", encoding="utf-8") as file:
            json.dump(summaries, file, indent=2)

        print("\n" + "=" * 88)
        print("PASTE THIS BLOCK")
        print("=" * 88)
        print("PCLA_HANDOVER_DIAGNOSTIC_V1")
        print(
            f"server={metadata['carla_server_version']} map={carla_map.name} "
            f"agent={args.pcla_agent} speed={args.target_speed_kmh:.1f} "
            f"safe_gap={args.safe_gap:.1f} tight_gap={args.tight_gap:.1f}"
        )
        for summary in summaries:
            if "error" in summary:
                print(f"{summary['case']}: ERROR {summary['error']}")
                continue
            print(
                "{case}: stable={stable} handover={handover} reason={reason} "
                "speed={speed} gap={gap} pre_brake={pre} pre_full={pre_full} "
                "handover_brake={hbrake} post_brake={post} post_full={post_full} "
                "speed_after_1s={after} collision={collision}".format(
                    case=summary["case"],
                    stable=summary["stable_reached"],
                    handover=summary["handover_step"],
                    reason=summary["handover_reason"],
                    speed=_fmt(summary["handover_ego_speed_kmh"], 1),
                    gap=_fmt(summary["handover_gap_m"], 1),
                    pre=_fmt(summary["pre_handover_1s_mean_policy_brake"]),
                    pre_full=_fmt(
                        summary["pre_handover_1s_full_brake_fraction"]
                    ),
                    hbrake=_fmt(summary["handover_policy_brake"]),
                    post=_fmt(summary["policy_control_brake_fraction"]),
                    post_full=_fmt(summary["policy_control_full_brake_fraction"]),
                    after=_fmt(summary["ego_speed_1s_after_handover_kmh"], 1),
                    collision=summary["collision"],
                )
            )
        for message in interpret_summaries(
            [summary for summary in summaries if "error" not in summary]
        ):
            print(f"INTERPRETATION: {message}")
        print(f"summary_json={os.path.abspath(summary_path)}")
        print(f"complete_log={os.path.abspath(os.path.join(output_dir, 'console.log'))}")
        print("=" * 88)
    finally:
        try:
            world.apply_settings(original_settings)
        except RuntimeError:
            pass
        try:
            tm.set_synchronous_mode(False)
        except RuntimeError:
            pass


def parse_args():
    parser = argparse.ArgumentParser(
        description="Diagnose PCLA staging/handover braking with D0-D3 tests"
    )
    parser.add_argument("--host", default=CARLA_HOST)
    parser.add_argument("--port", type=int, default=CARLA_PORT)
    parser.add_argument("--town", default=DEFAULT_TOWN)
    parser.add_argument(
        "--cases",
        nargs="+",
        choices=sorted(CASE_DESCRIPTIONS),
        default=sorted(CASE_DESCRIPTIONS),
    )
    parser.add_argument("--pcla-agent", default="tfv6_visiononly")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-speed-kmh", type=float, default=50.0)
    parser.add_argument("--safe-gap", type=float, default=40.0)
    parser.add_argument("--tight-gap", type=float, default=15.0)
    parser.add_argument("--gap-tolerance", type=float, default=3.0)
    parser.add_argument("--relative-speed-tolerance", type=float, default=1.0)
    parser.add_argument("--stable-seconds", type=float, default=1.0)
    parser.add_argument("--max-stage-seconds", type=float, default=12.0)
    parser.add_argument("--post-handover-seconds", type=float, default=3.0)
    parser.add_argument("--natural-seconds", type=float, default=8.0)
    parser.add_argument(
        "--output",
        default=None,
        help="Output directory (default: timestamped directory under diagnostics/)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.output:
        output_dir = args.output
    else:
        timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(
            os.path.dirname(__file__),
            "diagnostics",
            f"pcla_handover_{timestamp}",
        )
    os.makedirs(output_dir, exist_ok=True)

    console_path = os.path.join(output_dir, "console.log")
    with open(console_path, "w", encoding="utf-8") as log_file:
        tee_out = Tee(sys.stdout, log_file)
        tee_err = Tee(sys.stderr, log_file)
        with contextlib.redirect_stdout(tee_out), contextlib.redirect_stderr(tee_err):
            run_diagnostics(args, output_dir)


if __name__ == "__main__":
    main()
