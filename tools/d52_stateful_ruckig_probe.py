"""Probe whether MoveIt2 native Ruckig accepts a stateful timed trajectory.

This intentionally evaluates one D52 shadow case only.  It is used to decide
whether the local analytic retime can be converted into a direct native
post-Ruckig candidate without changing the protected pipeline.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import rclpy
from moveit.core.robot_state import RobotState
from moveit.core.robot_trajectory import RobotTrajectory
from moveit.planning import MoveItPy
from moveit_msgs.msg import RobotTrajectory as RobotTrajectoryMsg
from trajectory_msgs.msg import JointTrajectoryPoint

ROOT = Path(__file__).resolve().parents[1]
BRIDGE = ROOT / "ros2_moveit_bridge"
if str(BRIDGE) not in sys.path:
    sys.path.insert(0, str(BRIDGE))

from stage3_h7_native import GROUP_NAME, JOINT_NAMES, duration_message  # noqa: E402
from stage3_h7_2_native import message_arrays  # noqa: E402


def read_rows(path: Path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    return [
        (
            float(row["t"]),
            np.asarray([float(row[f"j{i}_q"]) for i in range(1, 7)], dtype=float),
            np.asarray([float(row[f"j{i}_dq"]) for i in range(1, 7)], dtype=float),
            np.asarray([float(row[f"j{i}_ddq"]) for i in range(1, 7)], dtype=float),
        )
        for row in rows
    ]


def stateful_message(rows):
    message = RobotTrajectoryMsg()
    message.joint_trajectory.joint_names = list(JOINT_NAMES)
    for time, q, dq, ddq in rows:
        point = JointTrajectoryPoint()
        point.positions = q.tolist()
        point.velocities = dq.tolist()
        point.accelerations = ddq.tolist()
        point.time_from_start = duration_message(time)
        message.joint_trajectory.points.append(point)
    return message


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--velocity-scaling", type=float, default=0.075)
    parser.add_argument("--acceleration-scaling", type=float, default=0.075)
    args, _ = parser.parse_known_args()
    rows = read_rows(args.trajectory.resolve())
    rclpy.init()
    moveit = MoveItPy(node_name="d52_stateful_ruckig_probe")
    try:
        trajectory = RobotTrajectory(moveit.get_robot_model())
        trajectory.joint_model_group_name = GROUP_NAME
        start = RobotState(moveit.get_robot_model())
        trajectory.set_robot_trajectory_msg(start, stateful_message(rows))
        before = message_arrays(trajectory.get_robot_trajectory_msg())
        result = trajectory.apply_ruckig_smoothing(
            float(args.velocity_scaling),
            float(args.acceleration_scaling),
            mitigate_overshoot=True,
            overshoot_threshold=0.005,
        )
        after = message_arrays(trajectory.get_robot_trajectory_msg())
        print({
            "ruckig_returned_success": bool(result),
            "before_duration_s": float(before[0][-1]),
            "after_duration_s": float(after[0][-1]),
            "before_state_count": int(len(before[0])),
            "after_state_count": int(len(after[0])),
            "after_max_abs_velocity_rad_s": float(np.max(np.abs(after[2]))),
            "after_max_abs_acceleration_rad_s2": float(np.max(np.abs(after[3]))),
        })
        return 0 if result else 2
    finally:
        moveit.shutdown()
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
