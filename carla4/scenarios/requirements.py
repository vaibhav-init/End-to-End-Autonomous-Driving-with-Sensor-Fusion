#!/usr/bin/env python3
"""Sensor-requirement arithmetic for the final report.

Three quantities the driving task fixes, independent of any detector:

  R_req   = v*tau + v^2/(2a)          required range for a stopped lead
  tau_max = R/v - v/(2a)              latency budget at a given range
  lambda  = C_target / p_crash        allowable phantom-brake rate

where ``C_target`` is the accepted collision rate per hour and
``p_crash`` is the conditional probability that one phantom brake produces a
rear-end collision.  The last line is the link FMVSS 127 does not provide:
it limits the *magnitude* of a false activation (0.25 g) but not its rate.

This script is deliberately CARLA-free.  Run it on the VM or locally:

  python3 requirements.py --p-crash 1e-3 1e-4
"""

import argparse
import csv
import os


def required_range_m(speed_mps, latency_s, decel_mps2):
    return speed_mps * latency_s + speed_mps ** 2 / (2.0 * decel_mps2)


def max_latency_s(range_m, speed_mps, decel_mps2):
    return range_m / speed_mps - speed_mps / (2.0 * decel_mps2)


def allowable_rate_per_hour(target_collisions_per_hour, p_crash):
    if p_crash <= 0.0:
        return float("inf")
    return target_collisions_per_hour / p_crash


def rate_per_km_from_per_hour(rate_per_hour, speed_kmh):
    return rate_per_hour / max(speed_kmh, 1.0e-9)


def build_table(
    speeds_kmh=(40, 50, 60, 80, 100, 130),
    latencies_s=(0.05, 0.10, 0.20, 0.50, 1.00),
    decel_mps2=6.867,
    target_collision_rate_per_hour=6.8e-7,
    p_crash_values=(1e-2, 1e-3, 1e-4, 1e-5),
):
    rows = []
    for speed_kmh in speeds_kmh:
        speed_mps = speed_kmh / 3.6
        for latency_s in latencies_s:
            rows.append({
                "kind": "range",
                "speed_kmh": speed_kmh,
                "latency_s": latency_s,
                "decel_mps2": decel_mps2,
                "range_m": required_range_m(speed_mps, latency_s, decel_mps2),
                "tau_max_s": max_latency_s(
                    100.0, speed_mps, decel_mps2
                ),
                "p_crash": "",
                "allowable_phantom_per_hour": "",
                "allowable_phantom_per_km": "",
            })
    for speed_kmh in speeds_kmh:
        for p_crash in p_crash_values:
            rate_h = allowable_rate_per_hour(
                target_collision_rate_per_hour, p_crash
            )
            rows.append({
                "kind": "false_positive_budget",
                "speed_kmh": speed_kmh,
                "latency_s": "",
                "decel_mps2": decel_mps2,
                "range_m": "",
                "tau_max_s": "",
                "p_crash": p_crash,
                "allowable_phantom_per_hour": rate_h,
                "allowable_phantom_per_km": rate_per_km_from_per_hour(
                    rate_h, speed_kmh
                ),
            })
    return rows


def write_csv(rows, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="requirements.csv")
    parser.add_argument(
        "--p-crash", type=float, nargs="+",
        default=[1e-2, 1e-3, 1e-4, 1e-5],
        help="conditional crash probabilities per phantom brake",
    )
    parser.add_argument(
        "--target-collision-rate", type=float, default=6.8e-7,
        help="accepted collisions per hour (Yu et al. T-ITS 2025 target)",
    )
    parser.add_argument("--decel-mps2", type=float, default=6.867)
    args = parser.parse_args()
    rows = build_table(
        p_crash_values=tuple(args.p_crash),
        target_collision_rate_per_hour=args.target_collision_rate,
        decel_mps2=args.decel_mps2,
    )
    write_csv(rows, args.out)
    print(f"wrote {args.out} ({len(rows)} rows)")
    for row in rows:
        if row["kind"] == "false_positive_budget":
            print(
                f"  p={row['p_crash']:.0e}  "
                f"allowable={row['allowable_phantom_per_hour']:.3g}/h  "
                f"({row['allowable_phantom_per_km']:.3g}/km at "
                f"{row['speed_kmh']} km/h)"
            )


if __name__ == "__main__":
    main()
