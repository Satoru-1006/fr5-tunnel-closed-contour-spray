import sys

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder


def generate_launch_description():
    manifest = LaunchConfiguration("manifest")
    output_dir = LaunchConfiguration("output_dir")
    root = "/mnt/d/robotfucker"
    robot_description = root + "/outputs/stage3_h4_5_robot_fixture_geometry_codesign_20260809T000000Z/derived_robot_model.urdf"
    robot_description_semantic = root + "/ros2_moveit_bridge/config/fairino5_v6_spray_tcp.srdf"
    limits = root + "/ros2_moveit_bridge/config/joint_limits_with_jerk.yaml"
    config = (
        MoveItConfigsBuilder("fairino5_v6_robot", package_name="fairino5_v6_moveit2_config")
        .robot_description(file_path=robot_description)
        .robot_description_semantic(file_path=robot_description_semantic)
        .joint_limits(file_path=limits)
        .planning_pipelines(pipelines=["ompl"])
        .to_moveit_configs()
    )
    pipeline = {"planning_pipelines": {"pipeline_names": ["ompl"], "ompl": config.to_dict().get("ompl", {})}}
    return LaunchDescription([
        DeclareLaunchArgument("manifest"),
        DeclareLaunchArgument("output_dir"),
        Node(
            executable=sys.executable,
            name="stage4c_self_sweep_native",
            output="screen",
            emulate_tty=True,
            arguments=[root + "/ros2_moveit_bridge/stage4c_self_sweep_native.py", "--manifest", manifest, "--output-dir", output_dir],
            parameters=[config.to_dict(), pipeline, {"manifest": manifest, "output_dir": output_dir}],
        ),
    ])
