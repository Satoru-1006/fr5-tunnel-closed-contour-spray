#!/usr/bin/env python3
"""Fast ROS2/MoveIt2 preflight check for the FR5 tunnel bridge."""

from __future__ import annotations

from pathlib import Path

import rclpy
from moveit.planning import MoveItPy

from plan_closed_contour_moveit import (
    _parameter_bool,
    load_tcp_normals,
    load_tcp_poses,
    preflight_moveit_runtime,
)


def main() -> None:
    rclpy.init()
    node = rclpy.create_node("fr5_closed_contour_moveit_smoke_test")
    tcp_path = Path(node.declare_parameter("tcp_path_csv", "outputs/tcp_poses.csv").value)
    group_name = str(node.declare_parameter("group_name", "fairino5_v6_group").value)
    ee_link = str(node.declare_parameter("ee_link", "spray_tcp_link").value)
    require_ruckig = _parameter_bool(node.declare_parameter("require_ruckig", True).value)

    moveit = MoveItPy(node_name="fr5_closed_contour_moveit_smoke_test")
    preflight_moveit_runtime(moveit, group_name, ee_link, require_ruckig)
    poses = load_tcp_poses(tcp_path)
    normals = load_tcp_normals(tcp_path)
    if len(poses) != len(normals):
        raise RuntimeError(f"Pose/normal count mismatch: {len(poses)} poses vs {len(normals)} normals.")

    robot_model = moveit.get_robot_model()
    group = robot_model.get_joint_model_group(group_name)
    node.get_logger().info("FR5 tunnel MoveIt2 smoke test passed.")
    node.get_logger().info(f"robot_model={robot_model.name}, group={group_name}, ee_link={ee_link}")
    node.get_logger().info(f"active_joints={list(group.active_joint_model_names)}")
    node.get_logger().info(f"tcp_pose_count={len(poses)}, tcp_path_csv={tcp_path}")
    rclpy.shutdown()


if __name__ == "__main__":
    main()
