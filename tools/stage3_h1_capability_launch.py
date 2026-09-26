"""Launch the isolated Stage 3 H1 offline capability probe."""

from pathlib import Path


def generate_launch_description():  # pragma: no cover - ROS2-only runtime
    from launch import LaunchDescription
    from launch.actions import DeclareLaunchArgument
    from launch.substitutions import LaunchConfiguration
    from launch_ros.actions import Node
    from launch_ros.parameter_descriptions import ParameterValue
    from moveit_configs_utils import MoveItConfigsBuilder

    repo = Path(__file__).resolve().parents[1]
    configs = (
        MoveItConfigsBuilder("fairino5_v6_robot", package_name="fairino5_v6_moveit2_config")
        .robot_description(
            file_path=str(repo / "ros2_moveit_bridge/config/fairino5_v6_spray_tcp.urdf.xacro"),
            mappings={"tool_tcp_xyz": "0 0 0.150", "tool_tcp_rpy": "0 0 0"},
        )
        .robot_description_semantic(file_path=str(repo / "ros2_moveit_bridge/config/fairino5_v6_spray_tcp.srdf"))
        .joint_limits(file_path=str(repo / "ros2_moveit_bridge/config/joint_limits_with_jerk.yaml"))
        .planning_pipelines(pipelines=["ompl"])
        .to_moveit_configs()
    )
    params = configs.to_dict()
    params["planning_pipelines"] = {"pipeline_names": ["ompl"], "ompl": params.get("ompl", {})}
    return LaunchDescription(
        [
            DeclareLaunchArgument("output_dir"),
            Node(
                package="stage3_h1_collision_capability",
                executable="stage3_h1_capability_probe",
                name="stage3_h1_capability_probe",
                output="screen",
                emulate_tty=True,
                parameters=[
                    params,
                    {
                        "output_dir": LaunchConfiguration("output_dir"),
                        "group_name": "fairino5_v6_group",
                    },
                ],
            ),
        ]
    )
