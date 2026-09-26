"""Launch the Stage 5A-R exact-state TF/FK witness with the stage5 mock model."""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def generate_launch_description():  # pragma: no cover - ROS 2 runtime
    from launch import LaunchDescription
    from launch.actions import DeclareLaunchArgument
    from launch.substitutions import LaunchConfiguration
    from launch_ros.actions import Node
    from moveit_configs_utils import MoveItConfigsBuilder
    from ament_index_python.packages import get_package_share_directory

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
        DeclareLaunchArgument("runtime_dir"),
        DeclareLaunchArgument("trajectory"),
        DeclareLaunchArgument("output"),
        Node(
            executable=sys.executable,
            name="stage5ar_exact_tf_fk_witness",
            output="screen",
            emulate_tty=True,
            arguments=[
                str(REPO_ROOT / "tools/stage5ar_tf_fk_witness.py"),
                "--runtime-dir", LaunchConfiguration("runtime_dir"),
                "--trajectory", LaunchConfiguration("trajectory"),
                "--output", LaunchConfiguration("output"),
            ],
            parameters=[params],
        ),
    ])
