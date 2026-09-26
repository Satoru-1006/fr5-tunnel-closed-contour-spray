#!/usr/bin/env python3
"""Launch one Stage 1.9.1 replay with the installed FAIRINO MoveIt config."""

from __future__ import annotations

import sys
from pathlib import Path


def generate_launch_description():  # pragma: no cover - ROS2 only
    from ament_index_python.packages import get_package_share_directory
    from launch import LaunchDescription
    from launch.actions import DeclareLaunchArgument
    from launch.substitutions import LaunchConfiguration
    from launch_ros.actions import Node
    from moveit_configs_utils import MoveItConfigsBuilder

    bridge_share = get_package_share_directory("fr5_tunnel_moveit_bridge")
    script = str(Path(__file__).resolve().parent / "run_stage191_ik_replay.py")
    configs = (MoveItConfigsBuilder("fairino5_v6_robot", package_name="fairino5_v6_moveit2_config")
        .robot_description(file_path=f"{bridge_share}/config/fairino5_v6_spray_tcp.urdf.xacro", mappings={"tool_tcp_xyz": "0 0 0.150", "tool_tcp_rpy": "0 0 0"})
        .robot_description_semantic(file_path=f"{bridge_share}/config/fairino5_v6_spray_tcp.srdf")
        .joint_limits(file_path=f"{bridge_share}/config/joint_limits_with_jerk.yaml")
        .planning_pipelines(pipelines=["ompl"]).to_moveit_configs())
    params = configs.to_dict()
    params["planning_pipelines"] = {"pipeline_names": ["ompl"], "ompl": params.get("ompl", {})}
    return LaunchDescription([
        DeclareLaunchArgument("stage191_config"), DeclareLaunchArgument("stage191_mode", default_value="build"), DeclareLaunchArgument("stage191_process_index", default_value="1"),
        Node(executable=sys.executable, name="stage191_ik_replay", output="screen", emulate_tty=True, arguments=[script, "--ros-node", "--mode", LaunchConfiguration("stage191_mode"), "--process-index", LaunchConfiguration("stage191_process_index"), "--config", LaunchConfiguration("stage191_config")], parameters=[params, {"group_name": "fairino5_v6_group", "ee_link": "spray_tcp_link"}]),
    ])
