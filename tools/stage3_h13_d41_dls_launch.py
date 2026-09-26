#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path


def generate_launch_description():  # pragma: no cover - ROS2-only entry point
    from ament_index_python.packages import get_package_share_directory
    from launch import LaunchDescription
    from launch.actions import DeclareLaunchArgument
    from launch.substitutions import LaunchConfiguration
    from launch_ros.actions import Node
    from moveit_configs_utils import MoveItConfigsBuilder

    repo = Path("/mnt/d/robotfucker")
    bridge_share = get_package_share_directory("fr5_tunnel_moveit_bridge")
    config = (
        MoveItConfigsBuilder("fairino5_v6_robot", package_name="fairino5_v6_moveit2_config")
        .robot_description(file_path=str(repo / "outputs/stage3_h4_5_robot_fixture_geometry_codesign_20260809T000000Z/derived_robot_model.urdf"))
        .robot_description_semantic(file_path=f"{bridge_share}/config/fairino5_v6_spray_tcp.srdf")
        .joint_limits(file_path=f"{bridge_share}/config/joint_limits_with_jerk.yaml")
        .planning_pipelines(pipelines=["ompl"])
        .to_moveit_configs()
    )
    pipeline = {"planning_pipelines": {"pipeline_names": ["ompl"], "ompl": config.to_dict().get("ompl", {})}}
    worker = str(repo / "tools/stage3_h13_d41_dls_batch.py")
    return LaunchDescription([
        DeclareLaunchArgument("config"),
        Node(
            executable=sys.executable,
            name="stage3_h13_d41_dls_batch",
            output="screen",
            emulate_tty=True,
            arguments=[worker, "--config", LaunchConfiguration("config")],
            parameters=[config.to_dict(), pipeline],
        ),
    ])
