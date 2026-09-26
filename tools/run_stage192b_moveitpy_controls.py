#!/usr/bin/env python3
"""Stage 1.9.2b MoveItPy control-A/B/C calls using the installed KDL plugin."""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import run_stage192_moveitpy_replay as base


def _matrix_quaternion(m: np.ndarray) -> tuple[float, float, float, float]:
    trace = float(m[0, 0] + m[1, 1] + m[2, 2])
    if trace > 0.0:
        s = (trace + 1.0) ** 0.5 * 2.0
        return ((m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s, 0.25 * s)
    if m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = (1.0 + m[0, 0] - m[1, 1] - m[2, 2]) ** 0.5 * 2.0
        return (0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s, (m[2, 1] - m[1, 2]) / s)
    if m[1, 1] > m[2, 2]:
        s = (1.0 + m[1, 1] - m[0, 0] - m[2, 2]) ** 0.5 * 2.0
        return ((m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s, (m[0, 2] - m[2, 0]) / s)
    s = (1.0 + m[2, 2] - m[0, 0] - m[1, 1]) ** 0.5 * 2.0
    return ((m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s, (m[1, 0] - m[0, 1]) / s)


def main() -> int:  # pragma: no cover - executed under ROS 2 launch
    parser = argparse.ArgumentParser()
    parser.add_argument("--ros-node", action="store_true")
    parser.add_argument("--control-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeat", type=int, default=3)
    args, _ = parser.parse_known_args()
    if not args.ros_node:
        raise SystemExit("real ROS2 launch required")
    import rclpy
    from moveit.core.robot_state import RobotState
    from moveit.planning import MoveItPy
    from geometry_msgs.msg import Pose

    target = np.fromfile(args.control_dir / "target_transform_f64.bin", dtype="<f8").reshape(4, 4)
    seed = np.fromfile(args.control_dir / "seed_joint_vector_f64.bin", dtype="<f8")
    qx, qy, qz, qw = _matrix_quaternion(target)
    pose = Pose(); pose.position.x, pose.position.y, pose.position.z = target[0, 3], target[1, 3], target[2, 3]
    pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = qx, qy, qz, qw
    group, tip = "fairino5_v6_group", "spray_tcp_link"
    rclpy.init()
    moveit = MoveItPy(node_name="stage192b_moveitpy_controls")
    model = moveit.get_robot_model()
    records = []
    for iteration in range(args.repeat):
        state = RobotState(model)
        state.set_joint_group_positions(group, seed.tolist()); state.update()
        started = time.perf_counter()
        solved = bool(state.set_from_ik(group, pose, tip_name=tip, timeout=0.0))
        record = {"control_dir": str(args.control_dir), "iteration": iteration, "process_id": __import__("os").getpid(), "solver_success": solved, "elapsed_time_s": time.perf_counter() - started, "solver": "MoveItPy RobotState.set_from_ik", "group": group, "tip": tip}
        if solved:
            state.update(); q = np.asarray(state.get_joint_group_positions(group), dtype=float)
            record["solution"] = q.tolist(); record["solution_sha256"] = hashlib.sha256(struct.pack("<6d", *q.tolist())).hexdigest()
        records.append(record)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"schema_version": "1.9.2b", "records": records}, indent=2) + "\n", encoding="utf-8")
    try: rclpy.shutdown()
    except Exception: pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
