"""MoveIt2 launch wrapper for the H4.5 FK link-transform audit."""

from __future__ import annotations

import sys
from pathlib import Path


def generate_launch_description():  # pragma: no cover - requires ROS 2
    from ament_index_python.packages import get_package_share_directory
    from launch import LaunchDescription
    from launch.actions import DeclareLaunchArgument
    from launch.substitutions import LaunchConfiguration
    from launch_ros.actions import Node
    from moveit_configs_utils import MoveItConfigsBuilder

    repo = Path(__file__).resolve().parents[1]
    bridge_share = Path(get_package_share_directory("fr5_tunnel_moveit_bridge"))
    configs = (
        MoveItConfigsBuilder("fairino5_v6_robot", package_name="fairino5_v6_moveit2_config")
        .robot_description(
            file_path=str(bridge_share / "config/fairino5_v6_spray_tcp.urdf.xacro"),
            mappings={"tool_tcp_xyz": "0 0 0.150", "tool_tcp_rpy": "0 0 0"},
        )
        .robot_description_semantic(file_path=str(bridge_share / "config/fairino5_v6_spray_tcp.srdf"))
        .joint_limits(file_path=str(bridge_share / "config/joint_limits_with_jerk.yaml"))
        .planning_pipelines(pipelines=["ompl"])
        .to_moveit_configs()
    )
    params = configs.to_dict()
    params["planning_pipelines"] = {"pipeline_names": ["ompl"], "ompl": params.get("ompl", {})}
    params["robot_description_kinematics"] = {
        "fairino5_v6_group": {
            "kinematics_solver": "kdl_kinematics_plugin/KDLKinematicsPlugin",
            "kinematics_solver_search_resolution": 0.005,
            "position_only_ik": False,
            "orientation_vs_position": 1.0,
            "epsilon": 1.0e-5,
            "max_solver_iterations": 500,
            "joints": ["j1", "j2", "j3", "j4", "j5", "j6"],
        }
    }
    return LaunchDescription([
        DeclareLaunchArgument("input"),
        DeclareLaunchArgument("output"),
        Node(
            executable=sys.executable,
            name="stage3_h4_5_fk_audit_worker",
            output="screen",
            emulate_tty=True,
            arguments=[
                str(repo / "scripts/stage3_h4_5_fk_audit_worker.py"),
                "--input", LaunchConfiguration("input"),
                "--output", LaunchConfiguration("output"),
            ],
            parameters=[params, {"group_name": "fairino5_v6_group", "ee_link": "spray_tcp_link"}],
        ),
    ])
