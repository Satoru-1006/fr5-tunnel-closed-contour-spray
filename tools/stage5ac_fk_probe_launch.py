"""Launch the read-only Stage5AC FK probe with the current MoveIt model."""

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
        DeclareLaunchArgument("historical_seed_csv"),
        DeclareLaunchArgument("output_json"),
        Node(
            executable=sys.executable,
            name="stage5ac_fk_probe",
            output="screen",
            emulate_tty=True,
            arguments=[str(ROOT / "tools/stage5ac_fk_probe.py")],
            parameters=[params, {
                "poses_csv": LaunchConfiguration("poses_csv"),
                "historical_seed_csv": LaunchConfiguration("historical_seed_csv"),
                "output_json": LaunchConfiguration("output_json"),
            }],
        ),
    ])
