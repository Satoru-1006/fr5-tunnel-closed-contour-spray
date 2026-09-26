"""Launch only the H8-R GenericSystem controller stack."""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.event_handlers import OnProcessExit
from launch.actions import RegisterEventHandler
from launch.substitutions import Command, FindExecutable, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    package_share = get_package_share_directory("fr5_tunnel_moveit_bridge")
    xacro_path = PathJoinSubstitution([FindPackageShare("fr5_tunnel_moveit_bridge"), "config", "stage3_h8_mock.urdf.xacro"])
    controllers_path = PathJoinSubstitution([FindPackageShare("fr5_tunnel_moveit_bridge"), "config", "stage3_h8_mock_controllers.yaml"])
    robot_description = {
        "robot_description": Command([FindExecutable(name="xacro"), " ", xacro_path])
    }
    output_dir = LaunchConfiguration("output_dir")
    joint_state_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["joint_state_broadcaster", "--controller-manager", "/controller_manager"],
        output="screen",
    )
    trajectory_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["fairino5_controller", "--controller-manager", "/controller_manager"],
        output="screen",
    )
    return LaunchDescription([
        DeclareLaunchArgument("h7_trajectory", default_value=""),
        DeclareLaunchArgument("output_dir", default_value="/tmp/stage3_h8_r"),
        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            name="robot_state_publisher",
            output="screen",
            parameters=[robot_description],
        ),
        Node(
            package="controller_manager",
            executable="ros2_control_node",
            name="controller_manager",
            output="screen",
            parameters=[robot_description, controllers_path],
        ),
        joint_state_spawner,
        RegisterEventHandler(
            OnProcessExit(
                target_action=joint_state_spawner,
                on_exit=[trajectory_spawner],
            )
        ),
    ])
