"""Run the existing callback-level collision witness with the Stage 5 mock model."""

from __future__ import annotations

from pathlib import Path


def generate_launch_description():  # pragma: no cover - ROS 2 runtime
    from ament_index_python.packages import get_package_share_directory
    from launch import LaunchDescription
    from launch.actions import DeclareLaunchArgument
    from launch.substitutions import LaunchConfiguration
    from launch_ros.actions import Node
    from launch_ros.parameter_descriptions import ParameterValue
    from moveit_configs_utils import MoveItConfigsBuilder

    repo = Path(__file__).resolve().parents[1]
    share = get_package_share_directory("fr5_tunnel_moveit_bridge")
    configs = (
        MoveItConfigsBuilder("fairino5_v6_robot", package_name="fairino5_v6_moveit2_config")
        .robot_description(
            file_path=f"{share}/config/stage5_mock.urdf.xacro",
            mappings={"tool_tcp_xyz": "0 0 0.150", "tool_tcp_rpy": "0 0 0"},
        )
        .robot_description_semantic(file_path=f"{share}/config/fairino5_v6_spray_tcp.srdf")
        .joint_limits(file_path=f"{share}/config/joint_limits_with_jerk.yaml")
        .planning_pipelines(pipelines=["ompl"])
        .to_moveit_configs()
    )
    params = configs.to_dict()
    params["planning_pipelines"] = {"pipeline_names": ["ompl"], "ompl": params.get("ompl", {})}
    return LaunchDescription([
        DeclareLaunchArgument("candidate_csv"),
        DeclareLaunchArgument("parts_dir"),
        DeclareLaunchArgument("output_dir"),
        DeclareLaunchArgument("backend", default_value="bullet"),
        DeclareLaunchArgument("run_index", default_value="1"),
        DeclareLaunchArgument("disputed_manifest", default_value=""),
        DeclareLaunchArgument("only_disputed", default_value="false"),
        DeclareLaunchArgument("repo_root", default_value=str(repo)),
        Node(
            package="stage23a7_runtime_audit",
            executable="stage23a7_runtime_audit_callback_probe",
            name="stage5ar_collision_callback_probe",
            output="screen",
            emulate_tty=True,
            parameters=[
                params,
                {
                    "candidate_csv": LaunchConfiguration("candidate_csv"),
                    "parts_dir": LaunchConfiguration("parts_dir"),
                    "output_dir": LaunchConfiguration("output_dir"),
                    "backend": LaunchConfiguration("backend"),
                    "run_index": ParameterValue(LaunchConfiguration("run_index"), value_type=int),
                    "disputed_manifest": LaunchConfiguration("disputed_manifest"),
                    "only_disputed": ParameterValue(LaunchConfiguration("only_disputed"), value_type=bool),
                    "repo_root": LaunchConfiguration("repo_root"),
                    "group_name": "fairino5_v6_group",
                },
            ],
        ),
    ])
