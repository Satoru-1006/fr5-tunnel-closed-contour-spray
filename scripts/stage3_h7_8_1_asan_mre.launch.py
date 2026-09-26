"""Sanitizer-only launch surface for the unchanged H7.8 MRE worker.

The parent launch process is intentionally not preloaded with ASan.  The
instrumented child alone receives the runtime, avoiding parent-Python
process-global leak noise while exercising the same MRE package and MoveItPy
teardown code path.
"""

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
    bridge_share = get_package_share_directory("fr5_tunnel_moveit_bridge")
    urdf = Path(
        "/mnt/d/robotfucker/outputs/stage3_h4_5_robot_fixture_geometry_codesign_20260809T000000Z/derived_robot_model.urdf"
    )
    config = (
        MoveItConfigsBuilder("fairino5_v6_robot", package_name="fairino5_v6_moveit2_config")
        .robot_description(file_path=str(urdf))
        .robot_description_semantic(file_path=f"{bridge_share}/config/fairino5_v6_spray_tcp.srdf")
        .joint_limits(file_path=f"{bridge_share}/config/joint_limits_with_jerk.yaml")
        .planning_pipelines(pipelines=["ompl"])
        .to_moveit_configs()
    )
    names = ["mre_variant", "teardown_mode", "native_samples"]
    declarations = [DeclareLaunchArgument(name) for name in names]
    parameters = {name: LaunchConfiguration(name) for name in names}
    pipeline = {
        "planning_pipelines": {"pipeline_names": ["ompl"], "ompl": config.to_dict().get("ompl", {})},
        "plan_request_params": {
            "planning_attempts": 1,
            "planning_pipeline": "ompl",
            "max_velocity_scaling_factor": 0.15,
            "max_acceleration_scaling_factor": 0.15,
        },
    }
    return LaunchDescription(
        declarations
        + [
            Node(
                package="stage3_h7_8_mre",
                executable="mre",
                name="stage3_h7_8_mre",
                output="screen",
                additional_env={"LD_PRELOAD": ASAN_RUNTIME, "ASAN_OPTIONS": ASAN_OPTIONS},
                parameters=[config.to_dict(), pipeline, parameters],
            )
        ]
    )
