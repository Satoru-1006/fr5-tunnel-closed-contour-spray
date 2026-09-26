from pathlib import Path
import sys

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder


def generate_launch_description():
    root = Path("/mnt/d/robotfucker")
    urdf = root / "outputs/stage3_h4_5_robot_fixture_geometry_codesign_20260809T000000Z/derived_robot_model.urdf"
    srdf = root / "ros2_moveit_bridge/config/fairino5_v6_spray_tcp.srdf"
    limits = root / "ros2_moveit_bridge/config/joint_limits_with_jerk.yaml"
    config = (
        MoveItConfigsBuilder("fairino5_v6_robot", package_name="fairino5_v6_moveit2_config")
        .robot_description(file_path=str(urdf))
        .robot_description_semantic(file_path=str(srdf))
        .joint_limits(file_path=str(limits))
        .planning_pipelines(pipelines=["ompl"])
        .to_moveit_configs()
    )
    trajectory = DeclareLaunchArgument("trajectory")
    velocity = DeclareLaunchArgument("velocity_scaling", default_value="0.075")
    acceleration = DeclareLaunchArgument("acceleration_scaling", default_value="0.075")
    pipeline = {"planning_pipelines": {"pipeline_names": ["ompl"], "ompl": config.to_dict().get("ompl", {})}}
    return LaunchDescription([
        trajectory, velocity, acceleration,
        Node(
            executable=sys.executable,
            name="d52_stateful_ruckig_probe",
            output="screen",
            emulate_tty=True,
            arguments=[
                str(root / "tools/d52_stateful_ruckig_probe.py"),
                "--trajectory", LaunchConfiguration("trajectory"),
                "--velocity-scaling", LaunchConfiguration("velocity_scaling"),
                "--acceleration-scaling", LaunchConfiguration("acceleration_scaling"),
            ],
            parameters=[config.to_dict(), pipeline],
        ),
    ])
