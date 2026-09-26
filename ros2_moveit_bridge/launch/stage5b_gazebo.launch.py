"""Stage 5B isolated Gazebo Sim + gz_ros2_control launch."""

from __future__ import annotations

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction, SetEnvironmentVariable, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from xacro import process_file


def generate_launch_description():  # pragma: no cover - ROS runtime
    package = "fr5_tunnel_moveit_bridge"
    gazebo_launch = get_package_share_directory("ros_gz_sim") + "/launch/gz_sim.launch.py"
    default_world = get_package_share_directory(package) + "/worlds/stage5b_shadow.sdf"
    default_initial = get_package_share_directory("fairino5_v6_moveit2_config") + "/config/initial_positions.yaml"
    fairino_description_share = get_package_share_directory("fairino_description")
    fairino_model_root = fairino_description_share.rsplit("/", 1)[0]

    def runtime_nodes(context):
        world = LaunchConfiguration("world").perform(context)
        initial = LaunchConfiguration("initial_positions_file").perform(context)
        xacro_path = get_package_share_directory(package) + "/config/stage5b_gazebo.urdf.xacro"
        document = process_file(xacro_path, mappings={"initial_positions_file": initial, "tool_tcp_xyz": "0 0 0.150", "tool_tcp_rpy": "0 0 0"})
        robot_description = ParameterValue(document.toxml(), value_type=str)
        return [
            # Keep the validation backend server-only. The GUI is not part of the
            # evidence path and can starve the controller manager on this host.
            IncludeLaunchDescription(PythonLaunchDescriptionSource(gazebo_launch), launch_arguments={"gz_args": ["-s -r ", world]}.items()),
            Node(package="robot_state_publisher", executable="robot_state_publisher", name="stage5b_robot_state_publisher", parameters=[{"robot_description": robot_description, "use_sim_time": True}], output="screen"),
            Node(package="ros_gz_sim", executable="create", name="stage5b_spawn_robot", arguments=["-topic", "robot_description", "-name", "fairino5", "-allow_renaming", "false"], output="screen"),
            Node(package="ros_gz_bridge", executable="parameter_bridge", name="stage5b_clock_bridge", arguments=["/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock"], output="screen"),
            TimerAction(period=5.0, actions=[
                Node(
                    package="controller_manager",
                    executable="spawner",
                    name="stage5b_spawn_joint_state_broadcaster",
                    arguments=["joint_state_broadcaster", "--controller-manager", "/controller_manager", "--controller-manager-timeout", "60", "--service-call-timeout", "60", "--switch-timeout", "60"],
                    output="screen",
                ),
                TimerAction(period=3.0, actions=[
                    Node(
                        package="controller_manager",
                        executable="spawner",
                        name="stage5b_spawn_fairino5_controller",
                        arguments=["fairino5_controller", "--controller-manager", "/controller_manager", "--controller-manager-timeout", "60", "--service-call-timeout", "60", "--switch-timeout", "60"],
                        output="screen",
                    ),
                ]),
            ]),
        ]

    return LaunchDescription([
        DeclareLaunchArgument("world", default_value=default_world),
        DeclareLaunchArgument("initial_positions_file", default_value=default_initial),
        SetEnvironmentVariable("GZ_SIM_RESOURCE_PATH", fairino_model_root),
        OpaqueFunction(function=runtime_nodes),
    ])
