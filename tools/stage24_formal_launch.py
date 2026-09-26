from pathlib import Path


def generate_launch_description():  # pragma: no cover - ROS 2 only
    from launch import LaunchDescription
    from launch.actions import DeclareLaunchArgument
    from launch.substitutions import LaunchConfiguration
    from launch_ros.actions import Node
    from launch_ros.parameter_descriptions import ParameterValue
    from moveit_configs_utils import MoveItConfigsBuilder

    repo = Path(__file__).resolve().parents[1]
    cfg = (
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
    params = cfg.to_dict()
    params["planning_pipelines"] = {"pipeline_names": ["ompl"], "ompl": params.get("ompl", {})}
    return LaunchDescription(
        [
            DeclareLaunchArgument("edge_csv"),
            DeclareLaunchArgument("parts_dir"),
            DeclareLaunchArgument("output_dir"),
            DeclareLaunchArgument("backend", default_value="fcl"),
            DeclareLaunchArgument("run_index", default_value="1"),
            Node(
                package="stage24_closed_loop_graph",
                executable="stage24_formal_probe",
                name="stage24_formal_probe",
                output="screen",
                emulate_tty=True,
                parameters=[
                    params,
                    {
                        "edge_csv": LaunchConfiguration("edge_csv"),
                        "parts_dir": LaunchConfiguration("parts_dir"),
                        "output_dir": LaunchConfiguration("output_dir"),
                        "backend": LaunchConfiguration("backend"),
                        "run_index": ParameterValue(LaunchConfiguration("run_index"), value_type=int),
                        "group_name": "fairino5_v6_group",
                    },
                ],
            ),
        ]
    )
