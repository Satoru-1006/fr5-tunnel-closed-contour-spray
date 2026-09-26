"""Launch the Stage 5B offline-only IK and JTC repair node."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def generate_launch_description():  # pragma: no cover - ROS runtime
    from ament_index_python.packages import get_package_share_directory
    from launch import LaunchDescription
    from launch.actions import DeclareLaunchArgument
    from launch.substitutions import LaunchConfiguration
    from launch_ros.actions import Node
    from launch_ros.parameter_descriptions import ParameterValue
    from moveit_configs_utils import MoveItConfigsBuilder

    share = get_package_share_directory("fr5_tunnel_moveit_bridge")
    configs = (
        MoveItConfigsBuilder("fairino5_v6_robot", package_name="fairino5_v6_moveit2_config")
        .robot_description(file_path=f"{share}/config/stage5_mock.urdf.xacro", mappings={"tool_tcp_xyz": "0 0 0.150", "tool_tcp_rpy": "0 0 0"})
        .robot_description_semantic(file_path=f"{share}/config/fairino5_v6_spray_tcp.srdf")
        .joint_limits(file_path=f"{share}/config/joint_limits_with_jerk.yaml")
        .planning_pipelines(pipelines=["ompl"])
        .to_moveit_configs()
    )
    params = configs.to_dict()
    params["planning_pipelines"] = {"pipeline_names": ["ompl"], "ompl": params.get("ompl", {})}
    return LaunchDescription([
        DeclareLaunchArgument("poses_csv"),
        DeclareLaunchArgument("source_q_csv"),
        DeclareLaunchArgument("environment_json"),
        DeclareLaunchArgument("output_dir"),
        DeclareLaunchArgument("include_collision", default_value="false"),
        DeclareLaunchArgument("ik_timeout_s", default_value="0.02"),
        Node(
            executable=sys.executable,
            name="stage5b_offline_repair",
            output="screen",
            emulate_tty=True,
            arguments=[str(ROOT / "tools/stage5b_offline_repair.py")],
            parameters=[params, {
                "poses_csv": LaunchConfiguration("poses_csv"),
                "source_q_csv": LaunchConfiguration("source_q_csv"),
                "environment_json": LaunchConfiguration("environment_json"),
                "output_dir": LaunchConfiguration("output_dir"),
                "include_collision": ParameterValue(LaunchConfiguration("include_collision"), value_type=bool),
                "ik_timeout_s": ParameterValue(LaunchConfiguration("ik_timeout_s"), value_type=float),
            }],
        ),
    ])
