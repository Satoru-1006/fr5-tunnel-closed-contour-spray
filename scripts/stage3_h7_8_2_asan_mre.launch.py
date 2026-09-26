"""ASan differential launch surface for C4/C5/C6 and MRE A/B/C.

This never sends a trajectory goal.  The only C5 change is an empty controller
manager plugin parameter, preventing TEM's controller action client from being
constructed while retaining MoveItCpp/TEM construction itself.
"""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder


ASAN_RUNTIME = "/usr/lib/gcc/x86_64-linux-gnu/13/libasan.so:/usr/lib/x86_64-linux-gnu/libstdc++.so.6"
ASAN_OPTIONS = "detect_leaks=1:halt_on_error=1:abort_on_error=1:verify_asan_link_order=1"


def make_node(context):
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
    mode = LaunchConfiguration("control_mode").perform(context)
    controller_override = {}
    additional_env = {"LD_PRELOAD": ASAN_RUNTIME, "ASAN_OPTIONS": ASAN_OPTIONS}
    if mode in {"C4", "C5"}:
        additional_env["H78_NO_RCLPY_NODE"] = "1"
        additional_env["H78_MRE_VARIANT"] = "A"
        additional_env["H78_TEARDOWN_MODE"] = "explicit"
    if mode == "C5":
        controller_override["moveit_controller_manager"] = ""
    pipeline = {
        "planning_pipelines": {"pipeline_names": ["ompl"], "ompl": config.to_dict().get("ompl", {})},
        "plan_request_params": {
            "planning_attempts": 1,
            "planning_pipeline": "ompl",
            "max_velocity_scaling_factor": 0.15,
            "max_acceleration_scaling_factor": 0.15,
        },
    }
    parameters = {
        name: LaunchConfiguration(name)
        for name in ("mre_variant", "teardown_mode", "native_samples")
    }
    return [
        Node(
            package="stage3_h7_8_mre",
            executable="mre",
            name="stage3_h7_8_mre",
            output="screen",
            additional_env=additional_env,
            parameters=[config.to_dict(), pipeline, parameters, controller_override],
        )
    ]


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument("control_mode", default_value="C6"),
            DeclareLaunchArgument("mre_variant", default_value="A"),
            DeclareLaunchArgument("teardown_mode", default_value="explicit"),
            DeclareLaunchArgument("native_samples", default_value=""),
            OpaqueFunction(function=make_node),
        ]
    )
