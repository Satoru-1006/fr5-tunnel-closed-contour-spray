"""Launch the isolated Stage 5A-C sequence-level IK shadow."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def generate_launch_description():  # pragma: no cover - ROS 2 runtime
    from ament_index_python.packages import get_package_share_directory
    from launch import LaunchDescription
    from launch.actions import DeclareLaunchArgument
    from launch.substitutions import LaunchConfiguration
    from launch_ros.actions import Node
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
        DeclareLaunchArgument("seeds_csv"),
        DeclareLaunchArgument("environment_json"),
        DeclareLaunchArgument("output_dir"),
        DeclareLaunchArgument("historical_seed_csv", default_value=""),
        DeclareLaunchArgument("task_slack_m", default_value="0.0"),
        DeclareLaunchArgument("task_offset_x_m", default_value="0.0"),
        DeclareLaunchArgument("task_offset_y_m", default_value="0.0"),
        DeclareLaunchArgument("task_offset_z_m", default_value="0.0"),
        DeclareLaunchArgument("task_offset_start_waypoint", default_value="-1"),
        DeclareLaunchArgument("task_offset_end_waypoint", default_value="-1"),
        DeclareLaunchArgument("beam_width", default_value="12"),
        DeclareLaunchArgument("max_step_deg", default_value="20.0"),
        Node(
            executable=sys.executable,
            name="stage5ac_route_a",
            output="screen",
            emulate_tty=True,
            arguments=[str(ROOT / "tools/stage5ac_route_a.py")],
            parameters=[params, {
                "poses_csv": LaunchConfiguration("poses_csv"),
                "seeds_csv": LaunchConfiguration("seeds_csv"),
                "environment_json": LaunchConfiguration("environment_json"),
                "output_dir": LaunchConfiguration("output_dir"),
                "historical_seed_csv": LaunchConfiguration("historical_seed_csv"),
                "task_slack_m": LaunchConfiguration("task_slack_m"),
                "task_offset_x_m": LaunchConfiguration("task_offset_x_m"),
                "task_offset_y_m": LaunchConfiguration("task_offset_y_m"),
                "task_offset_z_m": LaunchConfiguration("task_offset_z_m"),
                "task_offset_start_waypoint": LaunchConfiguration("task_offset_start_waypoint"),
                "task_offset_end_waypoint": LaunchConfiguration("task_offset_end_waypoint"),
                "beam_width": LaunchConfiguration("beam_width"),
                "max_step_deg": LaunchConfiguration("max_step_deg"),
            }],
        ),
    ])
