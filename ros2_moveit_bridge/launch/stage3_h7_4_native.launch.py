from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder


def generate_launch_description():
    bridge_share = get_package_share_directory("fr5_tunnel_moveit_bridge")
    derived_urdf = Path(
        "/mnt/d/robotfucker/outputs/stage3_h4_5_robot_fixture_geometry_codesign_20260809T000000Z/derived_robot_model.urdf"
    )
    config = (
        MoveItConfigsBuilder("fairino5_v6_robot", package_name="fairino5_v6_moveit2_config")
        .robot_description(file_path=str(derived_urdf))
        .robot_description_semantic(file_path=f"{bridge_share}/config/fairino5_v6_spray_tcp.srdf")
        .joint_limits(file_path=f"{bridge_share}/config/joint_limits_with_jerk.yaml")
        .planning_pipelines(pipelines=["ompl"])
        .to_moveit_configs()
    )
    names = [
        "output_dir",
        "h6_4_final_validation",
        "h6_4_segments",
        "process_contract",
        "fixture_mesh",
        "totg_parameters",
        "mitigate_overshoot",
    ]
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
                package="fr5_tunnel_moveit_bridge",
                executable="stage3_h7_4_native",
                name="stage3_h7_4_native",
                output="screen",
                parameters=[config.to_dict(), pipeline, parameters],
            )
        ]
    )
