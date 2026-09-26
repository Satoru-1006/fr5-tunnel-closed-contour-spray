#!/usr/bin/env python3
"""Launch the native MoveIt collision/contact and A/B diagnostic probe."""

from pathlib import Path


def generate_launch_description():  # pragma: no cover - ROS2 only
    from ament_index_python.packages import get_package_share_directory
    from launch import LaunchDescription
    from launch.actions import DeclareLaunchArgument
    from launch.substitutions import LaunchConfiguration
    from launch_ros.actions import Node
    from moveit_configs_utils import MoveItConfigsBuilder

    share = get_package_share_directory("fr5_tunnel_moveit_bridge")
    repo = Path(__file__).resolve().parents[1]
    candidate_csv = str(repo / "outputs" / "ik_graph_stage2_1" / "run_001" / "candidate_provenance.csv")
    pose_csv = str(repo / "outputs" / "tcp_poses_base_link.csv")
    output_dir = str(repo / "outputs" / "ik_graph_stage2_1" / "native_contact_probe_001")
    configs = (MoveItConfigsBuilder("fairino5_v6_robot", package_name="fairino5_v6_moveit2_config")
        .robot_description(file_path=f"{share}/config/fairino5_v6_spray_tcp.urdf.xacro", mappings={"tool_tcp_xyz": "0 0 0.150", "tool_tcp_rpy": "0 0 0"})
        .robot_description_semantic(file_path=f"{share}/config/fairino5_v6_spray_tcp.srdf")
        .joint_limits(file_path=f"{share}/config/joint_limits_with_jerk.yaml")
        .planning_pipelines(pipelines=["ompl"]).to_moveit_configs())
    params = configs.to_dict()
    params["planning_pipelines"] = {"pipeline_names": ["ompl"], "ompl": params.get("ompl", {})}
    return LaunchDescription([
        DeclareLaunchArgument("candidate_csv", default_value=candidate_csv),
        DeclareLaunchArgument("pose_csv", default_value=pose_csv),
        DeclareLaunchArgument("output_dir", default_value=output_dir),
        Node(
            package="stage21_native_diagnostics",
            executable="stage21_native_contact_probe",
            name="stage21_native_contact_probe",
            output="screen",
            emulate_tty=True,
            parameters=[params, {
                "candidate_csv": LaunchConfiguration("candidate_csv"),
                "pose_csv": LaunchConfiguration("pose_csv"),
                "output_dir": LaunchConfiguration("output_dir"),
                "group_name": "fairino5_v6_group",
            }],
        ),
    ])
