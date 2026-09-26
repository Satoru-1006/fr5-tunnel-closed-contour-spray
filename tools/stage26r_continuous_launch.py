"""ROS 2 launch description for the isolated Stage 2.6R native probe."""

from pathlib import Path


def generate_launch_description():  # pragma: no cover - ROS 2 only
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
    return LaunchDescription([
        DeclareLaunchArgument("interval_csv"),
        DeclareLaunchArgument("parts_dir"),
        DeclareLaunchArgument("output_dir"),
        DeclareLaunchArgument("backend", default_value="bullet"),
        DeclareLaunchArgument("run_index", default_value="1"),
        DeclareLaunchArgument("positive_controls", default_value="false"),
        DeclareLaunchArgument("self_spacing_threshold_rad", default_value="0.004800000000015847"),
        Node(
            package="stage26r_continuous_collision",
            executable="stage26r_continuous_probe",
            name="stage26r_continuous_probe",
            output="screen",
            emulate_tty=True,
            parameters=[params, {
                "interval_csv": LaunchConfiguration("interval_csv"),
                "parts_dir": LaunchConfiguration("parts_dir"),
                "output_dir": LaunchConfiguration("output_dir"),
                "backend": LaunchConfiguration("backend"),
                "run_index": ParameterValue(LaunchConfiguration("run_index"), value_type=int),
                "positive_controls": ParameterValue(LaunchConfiguration("positive_controls"), value_type=bool),
                "self_spacing_threshold_rad": ParameterValue(LaunchConfiguration("self_spacing_threshold_rad"), value_type=float),
                "group_name": "fairino5_v6_group",
            }],
        ),
    ])
