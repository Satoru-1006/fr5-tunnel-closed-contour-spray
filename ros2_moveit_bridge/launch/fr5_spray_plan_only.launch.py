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
    tool_tcp_source = LaunchConfiguration("tool_tcp_source")
    tool_tcp_measured_by = LaunchConfiguration("tool_tcp_measured_by")
    tool_tcp_measured_date = LaunchConfiguration("tool_tcp_measured_date")
    tool_tcp_calibration_method = LaunchConfiguration("tool_tcp_calibration_method")
    tcp_path = LaunchConfiguration("tcp_path_csv")
    stand_off = LaunchConfiguration("stand_off")
    waypoint_stride = LaunchConfiguration("waypoint_stride")
    planning_mode = LaunchConfiguration("planning_mode")
    joint_names = LaunchConfiguration("joint_names")
    ik_timeout = LaunchConfiguration("ik_timeout")
    max_ik_position_error = LaunchConfiguration("max_ik_position_error")
    max_ik_joint_step_deg = LaunchConfiguration("max_ik_joint_step_deg")
    ik_roll_sample_count = LaunchConfiguration("ik_roll_sample_count")
    ik_beam_width = LaunchConfiguration("ik_beam_width")
    ik_candidates_per_beam = LaunchConfiguration("ik_candidates_per_beam")
    ik_beam_diversity_joint_deg = LaunchConfiguration("ik_beam_diversity_joint_deg")
    ik_search_diagnostics_csv = LaunchConfiguration("ik_search_diagnostics_csv")
    ik_candidate_bank_csv = LaunchConfiguration("ik_candidate_bank_csv")
    ik_candidate_collection_report_json = LaunchConfiguration("ik_candidate_collection_report_json")
    ik_max_solve_seconds = LaunchConfiguration("ik_max_solve_seconds")
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
    production_joint_step_limit_deg = LaunchConfiguration("production_joint_step_limit_deg")
    quality_report_csv = LaunchConfiguration("quality_report_csv")
    fk_trace_csv = LaunchConfiguration("fk_trace_csv")
    trajectory_csv = LaunchConfiguration("trajectory_csv")
    segmented_execution = LaunchConfiguration("segmented_execution")
    segmented_trajectory_csv = LaunchConfiguration("segmented_trajectory_csv")
    transition_step_cap_deg = LaunchConfiguration("transition_step_cap_deg")
    transition_max_speed_deg_s = LaunchConfiguration("transition_max_speed_deg_s")
    transition_min_duration_s = LaunchConfiguration("transition_min_duration_s")
    transition_stop_hold_s = LaunchConfiguration("transition_stop_hold_s")
    waypoint_trajectory_csv = LaunchConfiguration("waypoint_trajectory_csv")
    dynamics_report_csv = LaunchConfiguration("dynamics_report_csv")
    collision_report_csv = LaunchConfiguration("collision_report_csv")
    validate_collision = LaunchConfiguration("validate_collision")
    collision_check_stride = LaunchConfiguration("collision_check_stride")
    collision_segment_stride = LaunchConfiguration("collision_segment_stride")
    collision_interpolation_step_deg = LaunchConfiguration("collision_interpolation_step_deg")
    include_tunnel_floor_collision = LaunchConfiguration("include_tunnel_floor_collision")
    tunnel_floor_z = LaunchConfiguration("tunnel_floor_z")
    tunnel_wall_thickness = LaunchConfiguration("tunnel_wall_thickness")
    tunnel_y_thickness = LaunchConfiguration("tunnel_y_thickness")
    include_bottom_closure_collision = LaunchConfiguration("include_bottom_closure_collision")
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
            DeclareLaunchArgument("tool_tcp_source", default_value=""),
            DeclareLaunchArgument("tool_tcp_measured_by", default_value=""),
            DeclareLaunchArgument("tool_tcp_measured_date", default_value=""),
            DeclareLaunchArgument("tool_tcp_calibration_method", default_value=""),
            DeclareLaunchArgument("tcp_path_csv"),
            DeclareLaunchArgument("stand_off", default_value="0.18"),
            DeclareLaunchArgument("waypoint_stride", default_value="4"),
            DeclareLaunchArgument("planning_mode", default_value="ik_waypoints"),
            DeclareLaunchArgument("joint_names", default_value="j1,j2,j3,j4,j5,j6"),
            DeclareLaunchArgument("ik_timeout", default_value="0.05"),
            DeclareLaunchArgument("max_ik_position_error", default_value="0.002"),
            DeclareLaunchArgument("max_ik_joint_step_deg", default_value="20.0"),
            DeclareLaunchArgument("ik_roll_sample_count", default_value="1"),
            DeclareLaunchArgument("ik_beam_width", default_value="1"),
            DeclareLaunchArgument("ik_candidates_per_beam", default_value="8"),
            DeclareLaunchArgument("ik_beam_diversity_joint_deg", default_value="0.0"),
            DeclareLaunchArgument(
                "ik_search_diagnostics_csv",
                default_value="outputs/moveit_ik_search_diagnostics.csv",
            ),
            DeclareLaunchArgument("ik_candidate_bank_csv", default_value="outputs/ik_candidate_bank.csv"),
            DeclareLaunchArgument(
                "ik_candidate_collection_report_json",
                default_value="outputs/ik_candidate_bank_report.json",
            ),
            DeclareLaunchArgument("ik_max_solve_seconds", default_value="0.0"),
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
            DeclareLaunchArgument("production_joint_step_limit_deg", default_value="20.0"),
            DeclareLaunchArgument("quality_report_csv", default_value="outputs/moveit_quality_report.csv"),
            DeclareLaunchArgument("fk_trace_csv", default_value="outputs/moveit_fk_tcp_trace.csv"),
            DeclareLaunchArgument("trajectory_csv", default_value="outputs/moveit_smoothed_joint_trajectory.csv"),
            DeclareLaunchArgument("segmented_execution", default_value="true"),
            DeclareLaunchArgument(
                "segmented_trajectory_csv",
                default_value="outputs/moveit_executed_segmented_joint_trajectory.csv",
            ),
            DeclareLaunchArgument("transition_step_cap_deg", default_value="5.0"),
            DeclareLaunchArgument("transition_max_speed_deg_s", default_value="12.0"),
            DeclareLaunchArgument("transition_min_duration_s", default_value="2.0"),
            DeclareLaunchArgument("transition_stop_hold_s", default_value="0.20"),
            DeclareLaunchArgument(
                "waypoint_trajectory_csv",
                default_value="outputs/moveit_waypoint_joint_trajectory.csv",
            ),
            DeclareLaunchArgument("dynamics_report_csv", default_value="outputs/moveit_joint_dynamics_report.csv"),
            DeclareLaunchArgument("collision_report_csv", default_value="outputs/moveit_collision_report.csv"),
            DeclareLaunchArgument("validate_collision", default_value="true"),
            DeclareLaunchArgument("collision_check_stride", default_value="5"),
            DeclareLaunchArgument("collision_segment_stride", default_value="4"),
            DeclareLaunchArgument("collision_interpolation_step_deg", default_value="1.0"),
            DeclareLaunchArgument("include_tunnel_floor_collision", default_value="false"),
            DeclareLaunchArgument("tunnel_floor_z", default_value="-0.20"),
            DeclareLaunchArgument("tunnel_wall_thickness", default_value="0.025"),
            DeclareLaunchArgument("tunnel_y_thickness", default_value="0.08"),
            DeclareLaunchArgument("include_bottom_closure_collision", default_value="true"),
            DeclareLaunchArgument("open_path", default_value="false"),
            DeclareLaunchArgument("tcp_points_to_wall", default_value="false"),
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
                        "tool_tcp_xyz": tool_xyz,
                        "tool_tcp_rpy": tool_rpy,
                        "tool_tcp_source": tool_tcp_source,
                        "tool_tcp_measured_by": tool_tcp_measured_by,
                        "tool_tcp_measured_date": tool_tcp_measured_date,
                        "tool_tcp_calibration_method": tool_tcp_calibration_method,
                        "stand_off": stand_off,
                        "waypoint_stride": waypoint_stride,
                        "planning_mode": planning_mode,
                        "joint_names": joint_names,
                        "ik_timeout": ik_timeout,
                        "max_ik_position_error": max_ik_position_error,
                        "max_ik_joint_step_deg": max_ik_joint_step_deg,
                        "ik_roll_sample_count": ik_roll_sample_count,
                        "ik_beam_width": ik_beam_width,
                        "ik_candidates_per_beam": ik_candidates_per_beam,
                        "ik_beam_diversity_joint_deg": ik_beam_diversity_joint_deg,
                        "ik_search_diagnostics_csv": ik_search_diagnostics_csv,
                        "ik_candidate_bank_csv": ik_candidate_bank_csv,
                        "ik_candidate_collection_report_json": ik_candidate_collection_report_json,
                        "ik_max_solve_seconds": ik_max_solve_seconds,
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
                        "production_joint_step_limit_deg": production_joint_step_limit_deg,
                        "quality_report_csv": quality_report_csv,
                        "fk_trace_csv": fk_trace_csv,
                        "trajectory_csv": trajectory_csv,
                        "segmented_execution": segmented_execution,
                        "segmented_trajectory_csv": segmented_trajectory_csv,
                        "transition_step_cap_deg": transition_step_cap_deg,
                        "transition_max_speed_deg_s": transition_max_speed_deg_s,
                        "transition_min_duration_s": transition_min_duration_s,
                        "transition_stop_hold_s": transition_stop_hold_s,
                        "waypoint_trajectory_csv": waypoint_trajectory_csv,
                        "dynamics_report_csv": dynamics_report_csv,
                        "collision_report_csv": collision_report_csv,
                        "validate_collision": validate_collision,
                        "collision_check_stride": collision_check_stride,
                        "collision_segment_stride": collision_segment_stride,
                        "collision_interpolation_step_deg": collision_interpolation_step_deg,
                        "include_tunnel_floor_collision": include_tunnel_floor_collision,
                        "tunnel_floor_z": tunnel_floor_z,
                        "tunnel_wall_thickness": tunnel_wall_thickness,
                        "tunnel_y_thickness": tunnel_y_thickness,
                        "include_bottom_closure_collision": include_bottom_closure_collision,
                        "open_path": LaunchConfiguration("open_path"),
                        "tcp_points_to_wall": LaunchConfiguration("tcp_points_to_wall"),
                        "fast_exit_after_reports": fast_exit_after_reports,
                        "execute_trajectory": execute_trajectory,
                        "validate_post_ruckig_fk": True,
                        "validate_joint_dynamics": True,
                    },
                ],
            ),
        ]
    )
