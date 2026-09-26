#!/usr/bin/env python3
"""Complete ROS2/MoveIt2 launch wrapper for an isolated Stage 1.8 run."""

from __future__ import annotations

import sys
from pathlib import Path


def generate_launch_description():  # pragma: no cover - executed by ROS2 launch
    from launch import LaunchDescription
    from launch.actions import DeclareLaunchArgument
    from launch.substitutions import LaunchConfiguration
    from launch_ros.actions import Node
    from ament_index_python.packages import get_package_share_directory
    from moveit_configs_utils import MoveItConfigsBuilder

    bridge_share = get_package_share_directory("fr5_tunnel_moveit_bridge")
    stage17_script = str(Path(__file__).resolve().parent / "run_stage17_pose_repair.py")
    moveit_config = (
        MoveItConfigsBuilder("fairino5_v6_robot", package_name="fairino5_v6_moveit2_config")
        .robot_description(
            file_path=f"{bridge_share}/config/fairino5_v6_spray_tcp.urdf.xacro",
            mappings={"tool_tcp_xyz": "0 0 0.150", "tool_tcp_rpy": "0 0 0"},
        )
        .robot_description_semantic(file_path=f"{bridge_share}/config/fairino5_v6_spray_tcp.srdf")
        .joint_limits(file_path=f"{bridge_share}/config/joint_limits_with_jerk.yaml")
        .planning_pipelines(pipelines=["ompl"])
        .to_moveit_configs()
    )
    params = moveit_config.to_dict()
    params["planning_pipelines"] = {"pipeline_names": ["ompl"], "ompl": params.get("ompl", {})}
    params["plan_request_params"] = {"planning_attempts": 1, "planning_pipeline": "ompl", "max_velocity_scaling_factor": 0.15, "max_acceleration_scaling_factor": 0.15}
    return LaunchDescription(
        [
            DeclareLaunchArgument("stage17_config"),
            DeclareLaunchArgument("stage18_run_dir"),
            DeclareLaunchArgument("run_id"),
            Node(
                executable=sys.executable,
                name="stage17_pose_repair",
                output="screen",
                emulate_tty=True,
                arguments=[stage17_script, "--ros-node", "--config", LaunchConfiguration("stage17_config"), "--stage18-run-dir", LaunchConfiguration("stage18_run_dir"), "--run-id", LaunchConfiguration("run_id")],
                parameters=[params, {"group_name": "fairino5_v6_group", "ee_link": "spray_tcp_link"}],
            ),
        ]
    )
