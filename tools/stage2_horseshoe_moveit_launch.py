#!/usr/bin/env python3
"""Launch MoveIt2 for the Stage 2 closed-horseshoe ON baseline."""

from pathlib import Path
import sys


def generate_launch_description():  # pragma: no cover - executed by ROS2
    from ament_index_python.packages import get_package_share_directory
    from launch import LaunchDescription
    from launch.actions import DeclareLaunchArgument
    from launch.substitutions import LaunchConfiguration
    from launch_ros.actions import Node
    from moveit_configs_utils import MoveItConfigsBuilder

    bridge_share = get_package_share_directory("fr5_tunnel_moveit_bridge")
    script = str(Path(__file__).resolve().parent / "run_stage2_on_baseline.py")
    config = MoveItConfigsBuilder("fairino5_v6_robot", package_name="fairino5_v6_moveit2_config").robot_description(file_path=f"{bridge_share}/config/fairino5_v6_spray_tcp.urdf.xacro", mappings={"tool_tcp_xyz": "0 0 0.150", "tool_tcp_rpy": "0 0 0"}).robot_description_semantic(file_path=f"{bridge_share}/config/fairino5_v6_spray_tcp.srdf").joint_limits(file_path=f"{bridge_share}/config/joint_limits_with_jerk.yaml").planning_pipelines(pipelines=["ompl"]).to_moveit_configs()
    params = config.to_dict(); params["planning_pipelines"] = {"pipeline_names": ["ompl"], "ompl": params.get("ompl", {})}
    return LaunchDescription([Node(executable=sys.executable, name="stage2_horseshoe_on_baseline", output="screen", emulate_tty=True, arguments=[script], parameters=[params, {"group_name": "fairino5_v6_group", "ee_link": "spray_tcp_link"}])])
