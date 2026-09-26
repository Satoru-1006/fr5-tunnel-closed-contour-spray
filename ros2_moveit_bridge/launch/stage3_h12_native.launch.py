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
    config = (
        MoveItConfigsBuilder("fairino5_v6_robot", package_name="fairino5_v6_moveit2_config")
        .robot_description(file_path=str(derived_urdf))
        .robot_description_semantic(file_path=f"{bridge_share}/config/fairino5_v6_spray_tcp.srdf")
        .joint_limits(file_path=f"{bridge_share}/config/joint_limits_with_jerk.yaml")
        .planning_pipelines(pipelines=["ompl"])
        .to_moveit_configs()
    )
    names = ["input_npz", "window_manifest", "output_json", "h6_validation", "h6_segments", "process_contract", "fixture_mesh", "runtime_limits"]
    declarations = [DeclareLaunchArgument(name) for name in names]
    parameters = {name: LaunchConfiguration(name) for name in names}
    worker_arguments = []
    for name in names:
        worker_arguments.extend([f"--{name.replace('_', '-')}", LaunchConfiguration(name)])
    pipeline = {"planning_pipelines": {"pipeline_names": ["ompl"], "ompl": config.to_dict().get("ompl", {})}}
    return LaunchDescription(declarations + [Node(
        executable=sys.executable,
        name="stage3_h12_native",
        output="screen",
        emulate_tty=True,
        arguments=[str(root / "ros2_moveit_bridge/stage3_h12_native.py"), *worker_arguments],
        parameters=[config.to_dict(), pipeline, parameters],
    )])

