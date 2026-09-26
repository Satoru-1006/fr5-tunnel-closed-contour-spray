"""Launch isolated native Stage5AC retiming with the current MoveIt model."""

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
        DeclareLaunchArgument("source_path"),
        DeclareLaunchArgument("output_dir"),
        DeclareLaunchArgument("velocity_scaling", default_value="0.25"),
        DeclareLaunchArgument("acceleration_scaling", default_value="0.25"),
        DeclareLaunchArgument("path_tolerance", default_value="1e-6"),
        DeclareLaunchArgument("resample_dt", default_value="0.01"),
        DeclareLaunchArgument("min_angle_change", default_value="0.0005"),
        DeclareLaunchArgument("manual_time_step_s", default_value="0.0"),
        DeclareLaunchArgument("consistent_fd_time_dilation", default_value="false"),
        DeclareLaunchArgument("fd_time_scale_multiplier", default_value="1.0"),
        DeclareLaunchArgument("zero_target_acceleration", default_value="true"),
        DeclareLaunchArgument("mitigate_overshoot", default_value="true"),
        DeclareLaunchArgument("overshoot_threshold", default_value="0.01"),
        Node(
            executable=sys.executable,
            name="stage5ac_native_retime",
            output="screen",
            emulate_tty=True,
            arguments=[str(ROOT / "tools/stage5ac_native_retime.py")],
            parameters=[params, {
                "source_path": LaunchConfiguration("source_path"),
                "output_dir": LaunchConfiguration("output_dir"),
                "velocity_scaling": ParameterValue(LaunchConfiguration("velocity_scaling"), value_type=float),
                "acceleration_scaling": ParameterValue(LaunchConfiguration("acceleration_scaling"), value_type=float),
                "path_tolerance": ParameterValue(LaunchConfiguration("path_tolerance"), value_type=float),
                "resample_dt": ParameterValue(LaunchConfiguration("resample_dt"), value_type=float),
                "min_angle_change": ParameterValue(LaunchConfiguration("min_angle_change"), value_type=float),
                "manual_time_step_s": ParameterValue(LaunchConfiguration("manual_time_step_s"), value_type=float),
                "consistent_fd_time_dilation": ParameterValue(LaunchConfiguration("consistent_fd_time_dilation"), value_type=bool),
                "fd_time_scale_multiplier": ParameterValue(LaunchConfiguration("fd_time_scale_multiplier"), value_type=float),
                "zero_target_acceleration": ParameterValue(LaunchConfiguration("zero_target_acceleration"), value_type=bool),
                "mitigate_overshoot": ParameterValue(LaunchConfiguration("mitigate_overshoot"), value_type=bool),
                "overshoot_threshold": ParameterValue(LaunchConfiguration("overshoot_threshold"), value_type=float),
            }],
        ),
    ])
