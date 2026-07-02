from launch import LaunchDescription
from ament_index_python.packages import get_package_share_directory
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder


def moveit_cpp_pipeline_params(moveit_config):
    ompl = moveit_config.to_dict().get("ompl", {})
    return {
        "planning_pipelines": {
            "pipeline_names": ["ompl"],
            "ompl": ompl,
        },
        "plan_request_params": {
            "planning_attempts": 1,
            "planning_pipeline": "ompl",
            "max_velocity_scaling_factor": 0.15,
            "max_acceleration_scaling_factor": 0.15,
        },
    }


def generate_launch_description():
    bridge_share = get_package_share_directory("fr5_tunnel_moveit_bridge")
    robot_xacro = f"{bridge_share}/config/fairino5_v6_spray_tcp.urdf.xacro"
    robot_srdf = f"{bridge_share}/config/fairino5_v6_spray_tcp.srdf"
    joint_limits = f"{bridge_share}/config/joint_limits_with_jerk.yaml"
    tool_xyz = LaunchConfiguration("tool_tcp_xyz")
    tool_rpy = LaunchConfiguration("tool_tcp_rpy")
    tcp_path = LaunchConfiguration("tcp_path_csv")

    moveit_config = (
        MoveItConfigsBuilder("fairino5_v6_robot", package_name="fairino5_v6_moveit2_config")
        .robot_description(
            file_path=robot_xacro,
            mappings={"tool_tcp_xyz": tool_xyz, "tool_tcp_rpy": tool_rpy},
        )
        .robot_description_semantic(file_path=robot_srdf)
        .joint_limits(file_path=joint_limits)
        .planning_pipelines(pipelines=["ompl"])
        .to_moveit_configs()
    )
    return LaunchDescription(
        [
            DeclareLaunchArgument("tool_tcp_xyz"),
            DeclareLaunchArgument("tool_tcp_rpy", default_value="0 0 0"),
            DeclareLaunchArgument("tcp_path_csv"),
            Node(
                package="fr5_tunnel_moveit_bridge",
                executable="smoke_test_moveit_bridge",
                name="fr5_closed_contour_moveit_smoke_test",
                output="screen",
                parameters=[
                    moveit_config.to_dict(),
                    moveit_cpp_pipeline_params(moveit_config),
                    {
                        "tcp_path_csv": tcp_path,
                        "ee_link": "spray_tcp_link",
                        "require_ruckig": True,
                    },
                ],
            ),
        ]
    )
