#!/usr/bin/env python3
"""Export an immutable-position H7.7 dynamic candidate for native replay."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


PRIMITIVE_ORDER: list[Any] = [0, 1000, 1, 1001, 2, 1002, "3A", "3B", 1003, 4]


def load_alpha_plan(raw: str) -> dict[str, float]:
    path = Path(raw)
    value = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else json.loads(raw)
    if isinstance(value, (int, float)):
        value = {str(primitive): float(value) for primitive in PRIMITIVE_ORDER}
    if not isinstance(value, dict):
        raise ValueError("alpha plan must be a number or primitive-to-alpha object")
    plan = {str(key): float(alpha) for key, alpha in value.items()}
    missing = [str(primitive) for primitive in PRIMITIVE_ORDER if str(primitive) not in plan]
    if missing:
        raise ValueError(f"alpha plan is missing primitives: {missing}")
    if any(not math.isfinite(alpha) or not 0.0 < alpha <= 1.0 for alpha in plan.values()):
        raise ValueError("all alpha values must be finite and in (0, 1]")
    return plan


def export(source: Path, destination: Path, plan: dict[str, float]) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    primitive_index = {str(value): index for index, value in enumerate(PRIMITIVE_ORDER)}
    rows = 0
    modified = 0
    zero_breaks_before = 0
    zero_breaks_after = 0
    max_velocity_change = 0.0
    max_acceleration_change = 0.0
    with source.open("r", encoding="utf-8") as input_stream, destination.open("w", encoding="utf-8", newline="") as output_stream:
        writer = csv.writer(output_stream, lineterminator="\n")
        writer.writerow([
            "primitive_index", "local_waypoint", "time",
            *[f"q{joint}" for joint in range(6)],
            *[f"v{joint}" for joint in range(6)],
            *[f"a{joint}" for joint in range(6)],
        ])
        for line in input_stream:
            if not line.strip():
                continue
            row = json.loads(line)
            key = str(row["primitive_id"])
            alpha = plan[key]
            positions = [float(value) for value in row["positions_rad"]]
            velocities = [float(value) for value in row["velocities_rad_s"]]
            accelerations = [float(value) for value in row["accelerations_rad_s2"]]
            scaled_velocity = [alpha * value for value in velocities]
            scaled_acceleration = [alpha * alpha * value for value in accelerations]
            zero_breaks_before += all(value == 0.0 for value in velocities)
            zero_breaks_after += all(value == 0.0 for value in scaled_velocity)
            max_velocity_change = max(max_velocity_change, *(abs(a - b) for a, b in zip(velocities, scaled_velocity)))
            max_acceleration_change = max(max_acceleration_change, *(abs(a - b) for a, b in zip(accelerations, scaled_acceleration)))
            modified += alpha != 1.0 and any(a != b for a, b in zip(velocities + accelerations, scaled_velocity + scaled_acceleration))
            writer.writerow([
                primitive_index[key], int(row["trajectory_index"]), float(row["time_from_start_s"]),
                *positions, *scaled_velocity, *scaled_acceleration,
            ])
            rows += 1
    return {
        "schema_version": "stage3-h7-7-candidate-export-v1",
        "source": str(source.resolve()),
        "destination": str(destination.resolve()),
        "alpha_plan": plan,
        "row_count": rows,
        "modified_dynamic_waypoint_count": modified,
        "position_modification_count": 0,
        "zero_velocity_break_count_before": zero_breaks_before,
        "zero_velocity_break_count_after": zero_breaks_after,
        "new_zero_velocity_breaks": zero_breaks_after - zero_breaks_before,
        "maximum_velocity_change_rad_s": max_velocity_change,
        "maximum_acceleration_change_rad_s2": max_acceleration_change,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--alpha-plan", required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report = export(args.source.resolve(), args.output.resolve(), load_alpha_plan(args.alpha_plan))
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
