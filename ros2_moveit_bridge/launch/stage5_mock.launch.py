"""Stage 5A isolated FR5 V6 GenericSystem execution chain."""

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction, RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch.substitutions import LaunchConfiguration
from xacro import process_file


def generate_launch_description():  # pragma: no cover - ROS 2 runtime
    package = "fr5_tunnel_moveit_bridge"
    default_initial = get_package_share_directory("fairino5_v6_moveit2_config") + "/config/initial_positions.yaml"

    def runtime_nodes(context):
        initial_file = LaunchConfiguration("initial_positions_file").perform(context)
        document = process_file(get_package_share_directory(package) + "/config/stage5_mock.urdf.xacro", mappings={
            "initial_positions_file": initial_file,
            "tool_tcp_xyz": "0 0 0.150",
            "tool_tcp_rpy": "0 0 0",
        })
        robot_description = {"robot_description": ParameterValue(document.toxml(), value_type=str), "publish_frequency": 100.0}
        controllers_path = get_package_share_directory(package) + "/config/stage5_mock_controllers.yaml"
        jsb = Node(
            package="controller_manager",
            executable="spawner",
            arguments=["joint_state_broadcaster", "--controller-manager", "/controller_manager", "--controller-manager-timeout", "30"],
            output="screen",
        )
        jtc = Node(
            package="controller_manager",
            executable="spawner",
            arguments=["fairino5_controller", "--controller-manager", "/controller_manager", "--controller-manager-timeout", "30"],
            output="screen",
        )
        return [
            Node(package="robot_state_publisher", executable="robot_state_publisher", name="robot_state_publisher", parameters=[robot_description], output="screen"),
            Node(package="controller_manager", executable="ros2_control_node", name="controller_manager", parameters=[robot_description, controllers_path], output="screen"),
            Node(package="tf2_ros", executable="static_transform_publisher", name="stage5_world_to_base", arguments=["0", "0", "0", "0", "0", "0", "world", "base_link"], output="screen"),
            RegisterEventHandler(OnProcessExit(target_action=jsb, on_exit=[jtc])),
            jsb,
        ]

    return LaunchDescription([
        DeclareLaunchArgument("runtime_tag", default_value="stage5a"),
        DeclareLaunchArgument("initial_positions_file", default_value=default_initial),
        DeclareLaunchArgument("tool_tcp_xyz", default_value="0 0 0.150"),
        DeclareLaunchArgument("tool_tcp_rpy", default_value="0 0 0"),
        OpaqueFunction(function=runtime_nodes),
    ])
