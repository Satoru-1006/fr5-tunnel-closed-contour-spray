"""ROS2 launch wrapper for the isolated Stage 3 H4 offline baseline."""

from __future__ import annotations

import sys
from pathlib import Path


def generate_launch_description():  # pragma: no cover - requires ROS2
    from ament_index_python.packages import get_package_share_directory
    from launch import LaunchDescription
    from launch.actions import DeclareLaunchArgument
    from launch.substitutions import LaunchConfiguration
    from launch_ros.actions import Node
    from moveit_configs_utils import MoveItConfigsBuilder

    repo = Path(__file__).resolve().parents[1]
    bridge_share = Path(get_package_share_directory("fr5_tunnel_moveit_bridge"))
    robot_xacro = bridge_share / "config/fairino5_v6_spray_tcp.urdf.xacro"
    robot_srdf = bridge_share / "config/fairino5_v6_spray_tcp.srdf"
    joint_limits = bridge_share / "config/joint_limits_with_jerk.yaml"
    configs = (
        MoveItConfigsBuilder("fairino5_v6_robot", package_name="fairino5_v6_moveit2_config")
        .robot_description(file_path=str(robot_xacro), mappings={"tool_tcp_xyz": "0 0 0.150", "tool_tcp_rpy": "0 0 0"})
        .robot_description_semantic(file_path=str(robot_srdf))
        .joint_limits(file_path=str(joint_limits))
        # MoveItPy's MoveItCpp initializer requires one registered pipeline,
        # although the H4 node never calls the planning API.
        .planning_pipelines(pipelines=["ompl"])
        .to_moveit_configs()
    )
    params = configs.to_dict()
    params["planning_pipelines"] = {"pipeline_names": ["ompl"], "ompl": params.get("ompl", {})}
    return LaunchDescription(
        [
            DeclareLaunchArgument("output_root", default_value=str(repo / "outputs" / "stage3_h4_reachability_baseline_ros")),
            Node(
                executable=sys.executable,
                name="stage3_h4_offline_reachability",
                output="screen",
                emulate_tty=True,
                arguments=[str(repo / "scripts/stage3_h4_baseline.py"), "--ros-node", "--output-root", LaunchConfiguration("output_root")],
                parameters=[params, {"group_name": "fairino5_v6_group", "ee_link": "spray_tcp_link"}],
            ),
        ]
    )
