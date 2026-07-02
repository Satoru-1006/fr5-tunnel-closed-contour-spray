from launch import LaunchDescription
from ament_index_python.packages import get_package_share_directory
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder
from moveit_configs_utils.launches import generate_demo_launch


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
    stand_off = LaunchConfiguration("stand_off")
    waypoint_stride = LaunchConfiguration("waypoint_stride")
    validation_stride = LaunchConfiguration("validation_stride")
    max_path_deviation = LaunchConfiguration("max_path_deviation")
    execute_trajectory = LaunchConfiguration("execute_trajectory")
    velocity_scaling = LaunchConfiguration("velocity_scaling")
    acceleration_scaling = LaunchConfiguration("acceleration_scaling")
    max_normal_error_deg = LaunchConfiguration("max_normal_error_deg")
    max_standoff_fraction = LaunchConfiguration("max_standoff_fraction")
    max_speed_fluctuation = LaunchConfiguration("max_speed_fluctuation")
    quality_report_csv = LaunchConfiguration("quality_report_csv")
    fk_trace_csv = LaunchConfiguration("fk_trace_csv")
    trajectory_csv = LaunchConfiguration("trajectory_csv")
    dynamics_report_csv = LaunchConfiguration("dynamics_report_csv")

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
    launch_description = LaunchDescription()
    launch_description.add_action(DeclareLaunchArgument("tool_tcp_xyz"))
    launch_description.add_action(DeclareLaunchArgument("tool_tcp_rpy", default_value="0 0 0"))
    launch_description.add_action(DeclareLaunchArgument("tcp_path_csv"))
    launch_description.add_action(DeclareLaunchArgument("stand_off", default_value="0.18"))
    launch_description.add_action(DeclareLaunchArgument("waypoint_stride", default_value="4"))
    launch_description.add_action(DeclareLaunchArgument("validation_stride", default_value="1"))
    launch_description.add_action(DeclareLaunchArgument("max_path_deviation", default_value="0.005"))
    launch_description.add_action(DeclareLaunchArgument("execute_trajectory", default_value="false"))
    launch_description.add_action(DeclareLaunchArgument("velocity_scaling", default_value="0.15"))
    launch_description.add_action(DeclareLaunchArgument("acceleration_scaling", default_value="0.15"))
    launch_description.add_action(DeclareLaunchArgument("max_normal_error_deg", default_value="10.0"))
    launch_description.add_action(DeclareLaunchArgument("max_standoff_fraction", default_value="0.05"))
    launch_description.add_action(DeclareLaunchArgument("max_speed_fluctuation", default_value="0.05"))
    launch_description.add_action(DeclareLaunchArgument("quality_report_csv", default_value="outputs/moveit_quality_report.csv"))
    launch_description.add_action(DeclareLaunchArgument("fk_trace_csv", default_value="outputs/moveit_fk_tcp_trace.csv"))
    launch_description.add_action(DeclareLaunchArgument("trajectory_csv", default_value="outputs/moveit_smoothed_joint_trajectory.csv"))
    launch_description.add_action(DeclareLaunchArgument("dynamics_report_csv", default_value="outputs/moveit_joint_dynamics_report.csv"))
    for action in generate_demo_launch(moveit_config).entities:
        launch_description.add_action(action)
    launch_description.add_action(
        Node(
            package="fr5_tunnel_moveit_bridge",
            executable="plan_closed_contour_moveit",
            name="fr5_closed_contour_moveit_planner",
            output="screen",
            parameters=[
                moveit_config.to_dict(),
                moveit_cpp_pipeline_params(moveit_config),
                {
                    "tcp_path_csv": tcp_path,
                    "ee_link": "spray_tcp_link",
                    "stand_off": stand_off,
                    "waypoint_stride": waypoint_stride,
                    "validation_stride": validation_stride,
                    "max_path_deviation": max_path_deviation,
                    "velocity_scaling": velocity_scaling,
                    "acceleration_scaling": acceleration_scaling,
                    "max_normal_error_deg": max_normal_error_deg,
                    "max_standoff_fraction": max_standoff_fraction,
                    "max_speed_fluctuation": max_speed_fluctuation,
                    "quality_report_csv": quality_report_csv,
                    "fk_trace_csv": fk_trace_csv,
                    "trajectory_csv": trajectory_csv,
                    "dynamics_report_csv": dynamics_report_csv,
                    "execute_trajectory": execute_trajectory,
                    "validate_post_ruckig_fk": True,
                    "validate_joint_dynamics": True,
                },
            ],
        )
    )
    return launch_description
