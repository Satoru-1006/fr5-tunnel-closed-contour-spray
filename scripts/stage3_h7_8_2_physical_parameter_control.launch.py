"""Full physical-worker imports/parameters without constructing MoveItPy."""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder


ASAN_RUNTIME = "/usr/lib/gcc/x86_64-linux-gnu/13/libasan.so:/usr/lib/x86_64-linux-gnu/libstdc++.so.6"
ASAN_OPTIONS = "detect_leaks=1:halt_on_error=1:abort_on_error=1:verify_asan_link_order=1"


def generate_launch_description():
    share = get_package_share_directory("fr5_tunnel_moveit_bridge")
    urdf = Path(
        "/mnt/d/robotfucker/outputs/"
        "stage3_h4_5_robot_fixture_geometry_codesign_20260809T000000Z/derived_robot_model.urdf"
    )
    config = (
        MoveItConfigsBuilder("fairino5_v6_robot", package_name="fairino5_v6_moveit2_config")
        .robot_description(file_path=str(urdf))
        .robot_description_semantic(file_path=f"{share}/config/fairino5_v6_spray_tcp.srdf")
        .joint_limits(file_path=f"{share}/config/joint_limits_with_jerk.yaml")
        .planning_pipelines(pipelines=["ompl"])
        .to_moveit_configs()
    )
    names = [
        "output_dir", "native_samples", "h6_4_final_validation", "h6_4_segments",
        "process_contract", "fixture_mesh", "shard_index", "shard_count",
    ]
    return LaunchDescription(
        [DeclareLaunchArgument(name, default_value="") for name in names]
        + [
            Node(
                package="stage3_h7_8_mre",
                executable="physical_parameter_control",
                name="stage3_h7_8_physical_parameter_control",
                output="screen",
                additional_env={"LD_PRELOAD": ASAN_RUNTIME, "ASAN_OPTIONS": ASAN_OPTIONS},
                parameters=[config.to_dict(), {name: LaunchConfiguration(name) for name in names}],
            )
        ]
    )
