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
    stand_off = LaunchConfiguration("stand_off")
    waypoint_stride = LaunchConfiguration("waypoint_stride")
    planning_mode = LaunchConfiguration("planning_mode")
    joint_names = LaunchConfiguration("joint_names")
    ik_timeout = LaunchConfiguration("ik_timeout")
    max_ik_position_error = LaunchConfiguration("max_ik_position_error")
    max_ik_joint_step_deg = LaunchConfiguration("max_ik_joint_step_deg")
    seed_joint_csv = LaunchConfiguration("seed_joint_csv")
    validation_stride = LaunchConfiguration("validation_stride")
    max_path_deviation = LaunchConfiguration("max_path_deviation")
    execute_trajectory = LaunchConfiguration("execute_trajectory")
    velocity_scaling = LaunchConfiguration("velocity_scaling")
    acceleration_scaling = LaunchConfiguration("acceleration_scaling")
    time_parameterization = LaunchConfiguration("time_parameterization")
    target_tcp_speed = LaunchConfiguration("target_tcp_speed")
    zero_boundary_state = LaunchConfiguration("zero_boundary_state")
    max_normal_error_deg = LaunchConfiguration("max_normal_error_deg")
    max_standoff_fraction = LaunchConfiguration("max_standoff_fraction")
    max_speed_fluctuation = LaunchConfiguration("max_speed_fluctuation")
    quality_report_csv = LaunchConfiguration("quality_report_csv")
    fk_trace_csv = LaunchConfiguration("fk_trace_csv")
    trajectory_csv = LaunchConfiguration("trajectory_csv")
    dynamics_report_csv = LaunchConfiguration("dynamics_report_csv")
    collision_report_csv = LaunchConfiguration("collision_report_csv")
    validate_collision = LaunchConfiguration("validate_collision")
    collision_check_stride = LaunchConfiguration("collision_check_stride")
    collision_segment_stride = LaunchConfiguration("collision_segment_stride")
    tunnel_wall_thickness = LaunchConfiguration("tunnel_wall_thickness")
    tunnel_y_thickness = LaunchConfiguration("tunnel_y_thickness")
    fast_exit_after_reports = LaunchConfiguration("fast_exit_after_reports")

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
            DeclareLaunchArgument("stand_off", default_value="0.18"),
            DeclareLaunchArgument("waypoint_stride", default_value="4"),
            DeclareLaunchArgument("planning_mode", default_value="seed_joint_waypoints"),
            DeclareLaunchArgument("joint_names", default_value="j1,j2,j3,j4,j5,j6"),
            DeclareLaunchArgument("ik_timeout", default_value="0.05"),
            DeclareLaunchArgument("max_ik_position_error", default_value="0.002"),
            DeclareLaunchArgument("max_ik_joint_step_deg", default_value="20.0"),
            DeclareLaunchArgument("seed_joint_csv", default_value=""),
            DeclareLaunchArgument("validation_stride", default_value="1"),
            DeclareLaunchArgument("max_path_deviation", default_value="0.005"),
            DeclareLaunchArgument("execute_trajectory", default_value="false"),
            DeclareLaunchArgument("velocity_scaling", default_value="0.15"),
            DeclareLaunchArgument("acceleration_scaling", default_value="0.15"),
            DeclareLaunchArgument("time_parameterization", default_value="tcp_arclength"),
            DeclareLaunchArgument("target_tcp_speed", default_value="0.003"),
            DeclareLaunchArgument("zero_boundary_state", default_value="true"),
            DeclareLaunchArgument("max_normal_error_deg", default_value="10.0"),
            DeclareLaunchArgument("max_standoff_fraction", default_value="0.05"),
            DeclareLaunchArgument("max_speed_fluctuation", default_value="0.05"),
            DeclareLaunchArgument("quality_report_csv", default_value="outputs/moveit_quality_report.csv"),
            DeclareLaunchArgument("fk_trace_csv", default_value="outputs/moveit_fk_tcp_trace.csv"),
            DeclareLaunchArgument("trajectory_csv", default_value="outputs/moveit_smoothed_joint_trajectory.csv"),
            DeclareLaunchArgument("dynamics_report_csv", default_value="outputs/moveit_joint_dynamics_report.csv"),
            DeclareLaunchArgument("collision_report_csv", default_value="outputs/moveit_collision_report.csv"),
            DeclareLaunchArgument("validate_collision", default_value="true"),
            DeclareLaunchArgument("collision_check_stride", default_value="5"),
            DeclareLaunchArgument("collision_segment_stride", default_value="4"),
            DeclareLaunchArgument("tunnel_wall_thickness", default_value="0.025"),
            DeclareLaunchArgument("tunnel_y_thickness", default_value="0.08"),
            DeclareLaunchArgument("fast_exit_after_reports", default_value="true"),
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
                        "planning_mode": planning_mode,
                        "joint_names": joint_names,
                        "ik_timeout": ik_timeout,
                        "max_ik_position_error": max_ik_position_error,
                        "max_ik_joint_step_deg": max_ik_joint_step_deg,
                        "seed_joint_csv": seed_joint_csv,
                        "validation_stride": validation_stride,
                        "max_path_deviation": max_path_deviation,
                        "velocity_scaling": velocity_scaling,
                        "acceleration_scaling": acceleration_scaling,
                        "time_parameterization": time_parameterization,
                        "target_tcp_speed": target_tcp_speed,
                        "zero_boundary_state": zero_boundary_state,
                        "max_normal_error_deg": max_normal_error_deg,
                        "max_standoff_fraction": max_standoff_fraction,
                        "max_speed_fluctuation": max_speed_fluctuation,
                        "quality_report_csv": quality_report_csv,
                        "fk_trace_csv": fk_trace_csv,
                        "trajectory_csv": trajectory_csv,
                        "dynamics_report_csv": dynamics_report_csv,
                        "collision_report_csv": collision_report_csv,
                        "validate_collision": validate_collision,
                        "collision_check_stride": collision_check_stride,
                        "collision_segment_stride": collision_segment_stride,
                        "tunnel_wall_thickness": tunnel_wall_thickness,
                        "tunnel_y_thickness": tunnel_y_thickness,
                        "fast_exit_after_reports": fast_exit_after_reports,
                        "execute_trajectory": execute_trajectory,
                        "validate_post_ruckig_fk": True,
                        "validate_joint_dynamics": True,
                    },
                ],
            ),
        ]
    )
