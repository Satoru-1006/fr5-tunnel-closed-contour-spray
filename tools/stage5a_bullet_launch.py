"""Run the maintained native MoveIt/Bullet two-state robot-world probe for Stage 5A."""

from __future__ import annotations

from pathlib import Path
import os


def generate_launch_description():  # pragma: no cover - ROS 2 runtime
    from ament_index_python.packages import get_package_share_directory
    from launch import LaunchDescription
    from launch.actions import DeclareLaunchArgument
    from launch.substitutions import LaunchConfiguration
    from launch_ros.actions import Node
    from launch_ros.parameter_descriptions import ParameterValue
    from moveit_configs_utils import MoveItConfigsBuilder

    share = get_package_share_directory("fr5_tunnel_moveit_bridge")
    # The ROS Jazzy sdformat vendor library is present but is not always added
    # to LD_LIBRARY_PATH by the base setup script. The native probe loads the
    # sdformat URDF plugin during model construction.
    vendor_lib = "/opt/ros/jazzy/opt/sdformat_vendor/lib"
    os.environ["LD_LIBRARY_PATH"] = vendor_lib + ":" + os.environ.get("LD_LIBRARY_PATH", "")
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
        DeclareLaunchArgument("interval_csv"),
        DeclareLaunchArgument("parts_dir"),
        DeclareLaunchArgument("output_dir"),
        DeclareLaunchArgument("backend", default_value="bullet"),
        DeclareLaunchArgument("run_index", default_value="1"),
        DeclareLaunchArgument("capability_only", default_value="false"),
        DeclareLaunchArgument("positive_controls", default_value="false"),
        Node(
            package="stage26_continuous_collision",
            executable="stage26_continuous_probe",
            name="stage5a_bullet_probe",
            output="screen",
            emulate_tty=True,
            parameters=[params, {
                "interval_csv": LaunchConfiguration("interval_csv"),
                "parts_dir": LaunchConfiguration("parts_dir"),
                "output_dir": LaunchConfiguration("output_dir"),
                "backend": LaunchConfiguration("backend"),
                "run_index": ParameterValue(LaunchConfiguration("run_index"), value_type=int),
                "capability_only": ParameterValue(LaunchConfiguration("capability_only"), value_type=bool),
                "positive_controls": ParameterValue(LaunchConfiguration("positive_controls"), value_type=bool),
                "group_name": "fairino5_v6_group",
            }],
        ),
    ])
