"""MoveIt2 launch wrapper for the H4.5 derived robot-model candidate."""

from __future__ import annotations

import sys
from pathlib import Path


def generate_launch_description():  # pragma: no cover - requires ROS 2
    from ament_index_python.packages import get_package_share_directory
    from launch import LaunchDescription
    from launch.actions import DeclareLaunchArgument, RegisterEventHandler
    from launch.event_handlers import OnProcessExit
    from launch.substitutions import LaunchConfiguration
    from launch_ros.actions import Node
    from moveit_configs_utils import MoveItConfigsBuilder

    # The H4.5 orchestrator uses this immutable, additive output namespace.
    # Keeping the path explicit lets MoveItConfigsBuilder resolve the derived
    # xacro before launch substitutions are evaluated.
    output_root = Path("/mnt/d/robotfucker/outputs/stage3_h4_5_robot_fixture_geometry_codesign_20260809T000000Z")
    derived_xacro = output_root / "derived_robot_model.xacro"
    repo = Path("/mnt/d/robotfucker")
    bridge_share = Path(get_package_share_directory("fr5_tunnel_moveit_bridge"))
    configs = (
        MoveItConfigsBuilder("fairino5_v6_robot", package_name="fairino5_v6_moveit2_config")
        .robot_description(
            file_path=str(derived_xacro),
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
    args = [
        str(repo / "scripts/stage3_h4_5_derived_worker.py"),
        "--worker-output", LaunchConfiguration("worker_output"),
        "--replay-index", LaunchConfiguration("replay_index"),
        "--placements", LaunchConfiguration("placements"),
        "--h3-targets", LaunchConfiguration("h3_targets"),
        "--h2-manifest", LaunchConfiguration("h2_manifest"),
        "--kinematics", LaunchConfiguration("kinematics"),
        "--urdf", LaunchConfiguration("urdf"),
        "--srdf", LaunchConfiguration("srdf"),
    ]
    worker = Node(
        executable=sys.executable,
        name="stage3_h4_5_derived_worker",
        output="screen",
        emulate_tty=True,
        arguments=args,
        parameters=[params, {"group_name": "fairino5_v6_group", "ee_link": "spray_tcp_link"}],
    )

    def record_exit(event, context):
        import json

        path = Path(LaunchConfiguration("child_exit_record").perform(context))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "schema_version": "stage3-h4-5-derived-child-exit-v1",
            "pid": int(getattr(event, "pid", -1)),
            "returncode": int(getattr(event, "returncode", -999)),
        }, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        return []

    return LaunchDescription([
        DeclareLaunchArgument("worker_output"),
        DeclareLaunchArgument("replay_index"),
        DeclareLaunchArgument("placements"),
        DeclareLaunchArgument("h3_targets"),
        DeclareLaunchArgument("h2_manifest"),
        DeclareLaunchArgument("kinematics"),
        DeclareLaunchArgument("urdf"),
        DeclareLaunchArgument("srdf"),
        DeclareLaunchArgument("child_exit_record"),
        worker,
        RegisterEventHandler(OnProcessExit(target_action=worker, on_exit=record_exit)),
    ])
