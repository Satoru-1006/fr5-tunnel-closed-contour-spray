from pathlib import Path
import sys

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder


def generate_launch_description():
    bridge_share = get_package_share_directory("fr5_tunnel_moveit_bridge")
    root = Path("/mnt/d/robotfucker")
    derived_urdf = root / "outputs/stage3_h4_5_robot_fixture_geometry_codesign_20260809T000000Z/derived_robot_model.urdf"
    fixture = root / "outputs/stage3_h4_5_robot_fixture_geometry_codesign_20260809T000000Z/curved_fixture_stage3_h4_5_selected.obj"
    config = (
        MoveItConfigsBuilder("fairino5_v6_robot", package_name="fairino5_v6_moveit2_config")
        .robot_description(file_path=str(derived_urdf))
        .robot_description_semantic(file_path=f"{bridge_share}/config/fairino5_v6_spray_tcp.srdf")
        .joint_limits(file_path=f"{bridge_share}/config/joint_limits_with_jerk.yaml")
        .planning_pipelines(pipelines=["ompl"])
        .to_moveit_configs()
    )
    declarations = [
        DeclareLaunchArgument("input_json"),
        DeclareLaunchArgument("output_dir"),
        DeclareLaunchArgument("fixture_mesh", default_value=str(fixture)),
        DeclareLaunchArgument("model_urdf", default_value=str(derived_urdf)),
    ]
    pipeline = {"planning_pipelines": {"pipeline_names": ["ompl"], "ompl": config.to_dict().get("ompl", {})}}
    return LaunchDescription(
        declarations
        + [
            Node(
                executable=sys.executable,
                name="stage3_h10v_fk",
                output="screen",
                emulate_tty=True,
                arguments=[
                    str(root / "ros2_moveit_bridge/stage3_h10v_fk.py"),
                    "--input",
                    LaunchConfiguration("input_json"),
                    "--output",
                    LaunchConfiguration("output_dir"),
                    "--fixture",
                    LaunchConfiguration("fixture_mesh"),
                    "--model-urdf",
                    LaunchConfiguration("model_urdf"),
                ],
                parameters=[config.to_dict(), pipeline],
            )
        ]
    )
