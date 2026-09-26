"""ROS 2 launch wrapper for the Stage 3 H4.4 native FCL bridge."""

from __future__ import annotations

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
        DeclareLaunchArgument("executable_path"),
        DeclareLaunchArgument("candidate_csv"),
        DeclareLaunchArgument("curved_mesh_obj"),
        DeclareLaunchArgument("output_dir"),
        DeclareLaunchArgument("group_name", default_value="fairino5_v6_group"),
        DeclareLaunchArgument("max_contacts", default_value="4096"),
        DeclareLaunchArgument("max_contacts_per_pair", default_value="64"),
        Node(
            executable=LaunchConfiguration("executable_path"),
            name="stage3_h4_4_native_bridge",
            output="screen",
            emulate_tty=True,
            parameters=[params, {
                "candidate_csv": LaunchConfiguration("candidate_csv"),
                "curved_mesh_obj": LaunchConfiguration("curved_mesh_obj"),
                "output_dir": LaunchConfiguration("output_dir"),
                "group_name": LaunchConfiguration("group_name"),
                "max_contacts": LaunchConfiguration("max_contacts"),
                "max_contacts_per_pair": LaunchConfiguration("max_contacts_per_pair"),
            }],
        ),
    ])
