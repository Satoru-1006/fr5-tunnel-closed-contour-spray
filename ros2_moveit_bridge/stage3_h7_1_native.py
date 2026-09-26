#!/usr/bin/env python3
"""Native MoveIt2 prefix replay worker for the additive Stage 3 H7.1 audit.

This worker only constructs an in-memory RobotTrajectory and calls the native
MoveIt TOTG API.  It never creates an action client, sends a trajectory goal,
or starts robot motion.  The input is always the frozen H6 JSONL; optional
omissions are explicit diagnostic perturbations and are never written back to
the frozen input.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import rclpy
from moveit.core.robot_state import RobotState
from moveit.core.robot_trajectory import RobotTrajectory
from moveit.planning import MoveItPy

from stage3_h7_native import (
    GROUP_NAME,
    JOINT_NAMES,
    dump_json,
    load_json,
    load_jsonl,
    make_trajectory_message,
    param,
)


def parse_int_set(value: Any) -> set[int]:
    text = str(value or "").strip()
    if not text or text.lower() in {"none", "-"}:
        return set()
    # ROS 2 launch/YAML parameter coercion treats a bare numeric argument as
    # an INTEGER even though the declared launch argument is conceptually a
    # string.  The orchestrator prefixes diagnostic omission lists with `x`.
    if text.startswith("x"):
        text = text[1:]
    return {int(item.strip()) for item in text.split(",") if item.strip()}


def replay_prefix(node: Any, moveit: MoveItPy) -> dict[str, Any]:
    output = Path(str(param(node, "output_dir"))).resolve()
    h6_path = Path(str(param(node, "h6_joint_waypoints"))).resolve()
    totg_path = Path(str(param(node, "totg_configuration"))).resolve()
    prefix_end = int(param(node, "prefix_end", -1))
    omitted = parse_int_set(param(node, "omit_waypoint_indices", ""))
    case_label = str(param(node, "case_label", "prefix_replay"))

    all_rows = load_jsonl(h6_path)
    selected = [
        row for row in all_rows
        if int(row["waypoint_index"]) <= prefix_end
        and int(row["waypoint_index"]) not in omitted
    ]
    raw_q = np.asarray([row["joint_values"] for row in selected], dtype=float)
    result: dict[str, Any] = {
        "schema_version": "stage3-h7-1-native-prefix-replay-v1",
        "case_label": case_label,
        "h6_joint_waypoints": str(h6_path),
        "prefix_end_waypoint": prefix_end,
        "omitted_waypoint_indices": sorted(omitted),
        "selected_waypoint_indices": [int(row["waypoint_index"]) for row in selected],
        "waypoint_count": len(selected),
        "joint_names": list(JOINT_NAMES),
        "totg_api": "RobotTrajectory.apply_totg_time_parameterization",
        "totg_return_value": None,
        "trajectory_generated": False,
        "NEW_FJT_GOALS_SENT": 0,
        "ROBOT_MOTION_STARTED": "NO",
    }
    if raw_q.shape != (len(selected), 6) or len(selected) < 3 or not np.all(np.isfinite(raw_q)):
        result.update({"status": "BLOCKED", "errors": ["invalid_prefix_joint_input"]})
        dump_json(output / "stage3_h7_1_native_prefix_replay.json", result)
        return result

    configuration = load_json(totg_path)
    start_state = RobotState(moveit.get_robot_model())
    trajectory = RobotTrajectory(moveit.get_robot_model())
    trajectory.joint_model_group_name = GROUP_NAME
    trajectory.set_robot_trajectory_msg(start_state, make_trajectory_message(selected))
    trajectory.unwind()
    try:
        succeeded = bool(
            trajectory.apply_totg_time_parameterization(
                float(configuration["velocity_scaling_factor"]),
                float(configuration["acceleration_scaling_factor"]),
                path_tolerance=float(configuration["path_tolerance"]),
                resample_dt=float(configuration["resample_dt"]),
                min_angle_change=float(configuration["min_angle_change"]),
            )
        )
    except Exception as exc:  # native exception is evidence, not a pass
        result.update({
            "status": "BLOCKED",
            "errors": [f"native_totg_exception:{type(exc).__name__}:{exc}"],
        })
        dump_json(output / "stage3_h7_1_native_prefix_replay.json", result)
        return result

    result["totg_return_value"] = succeeded
    result["status"] = "PASSED" if succeeded else "BLOCKED"
    result["errors"] = [] if succeeded else ["native_totg_returned_false"]
    if succeeded:
        message = trajectory.get_robot_trajectory_msg()
        points = message.joint_trajectory.points
        times = [float(point.time_from_start.sec) + float(point.time_from_start.nanosec) * 1e-9 for point in points]
        result.update({
            "trajectory_generated": True,
            "native_output_waypoint_count": len(points),
            "native_duration_s": times[-1] if times else None,
            "timestamps_finite_strictly_increasing": bool(
                len(times) >= 2
                and all(math.isfinite(value) for value in times)
                and all(b > a for a, b in zip(times, times[1:]))
            ),
        })
    dump_json(output / "stage3_h7_1_native_prefix_replay.json", result)
    return result


def main() -> int:
    rclpy.init()
    node = rclpy.create_node("stage3_h7_1_native")
    output = Path(str(param(node, "output_dir"))).resolve()
    output.mkdir(parents=True, exist_ok=True)
    exit_code = 2
    try:
        moveit = MoveItPy(node_name="stage3_h7_1_native")
        result = replay_prefix(node, moveit)
        exit_code = 0 if result.get("status") == "PASSED" else 2
    except Exception as exc:
        dump_json(
            output / "stage3_h7_1_native_prefix_replay.json",
            {
                "schema_version": "stage3-h7-1-native-prefix-replay-v1",
                "status": "BLOCKED",
                "errors": [f"native_worker_exception:{type(exc).__name__}:{exc}"],
                "NEW_FJT_GOALS_SENT": 0,
                "ROBOT_MOTION_STARTED": "NO",
            },
        )
    os._exit(exit_code)


if __name__ == "__main__":
    raise SystemExit(main())
