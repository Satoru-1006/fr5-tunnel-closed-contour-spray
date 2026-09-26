"""Load/destroy the H7 controller plugin without MoveItPy or TEM."""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder


ASAN_OPTIONS = "detect_leaks=1:halt_on_error=1:abort_on_error=1:verify_asan_link_order=1"


def generate_launch_description():
    bridge_share = get_package_share_directory("fr5_tunnel_moveit_bridge")
    urdf = Path(
        "/mnt/d/robotfucker/outputs/"
        "stage3_h4_5_robot_fixture_geometry_codesign_20260809T000000Z/derived_robot_model.urdf"
    )
    config = (
        MoveItConfigsBuilder("fairino5_v6_robot", package_name="fairino5_v6_moveit2_config")
        .robot_description(file_path=str(urdf))
        .robot_description_semantic(file_path=f"{bridge_share}/config/fairino5_v6_spray_tcp.srdf")
        .joint_limits(file_path=f"{bridge_share}/config/joint_limits_with_jerk.yaml")
        .planning_pipelines(pipelines=["ompl"])
        .to_moveit_configs()
    )
    return LaunchDescription(
        [
            Node(
                package="stage3_h7_8_controls",
                executable="plugin_controller_control",
                name="stage3_h7_8_plugin_controller_control",
                output="screen",
                additional_env={"ASAN_OPTIONS": ASAN_OPTIONS},
                parameters=[config.to_dict()],
            )
        ]
    )
