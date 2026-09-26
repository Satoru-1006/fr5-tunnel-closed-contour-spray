"""Launch the isolated Stage 2.5 MoveIt2 timing runner."""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stage25_contract import (  # noqa: E402
    FORMAL_RUCKIG_PARAMETERS,
    FORMAL_TOTG_PARAMETERS,
)


def generate_launch_description():  # pragma: no cover - ROS2 runtime
    from ament_index_python.packages import get_package_share_directory
    from launch import LaunchDescription
    from launch.actions import DeclareLaunchArgument
    from launch.substitutions import LaunchConfiguration
    from launch_ros.actions import Node
    from moveit_configs_utils import MoveItConfigsBuilder

    repo = REPO_ROOT
    bridge_share = get_package_share_directory("fr5_tunnel_moveit_bridge")
    configs = (
        MoveItConfigsBuilder("fairino5_v6_robot", package_name="fairino5_v6_moveit2_config")
        .robot_description(
            file_path=f"{bridge_share}/config/fairino5_v6_spray_tcp.urdf.xacro",
            mappings={"tool_tcp_xyz": "0 0 0.150", "tool_tcp_rpy": "0 0 0"},
        )
        .robot_description_semantic(file_path=f"{bridge_share}/config/fairino5_v6_spray_tcp.srdf")
        .joint_limits(file_path=f"{bridge_share}/config/joint_limits_with_jerk.yaml")
        .planning_pipelines(pipelines=["ompl"])
        .to_moveit_configs()
    )
    params = configs.to_dict()
    params["planning_pipelines"] = {"pipeline_names": ["ompl"], "ompl": params.get("ompl", {})}
    script = str(repo / "tools" / "stage25_moveit_runner.py")
    return LaunchDescription(
        [
            DeclareLaunchArgument("input_csv"),
            DeclareLaunchArgument("source_reference_json"),
            DeclareLaunchArgument("waypoints_csv"),
            DeclareLaunchArgument("output_dir"),
            DeclareLaunchArgument("group_name", default_value="fairino5_v6_group"),
            DeclareLaunchArgument("ee_link", default_value="spray_tcp_link"),
            DeclareLaunchArgument("velocity_scaling", default_value=str(FORMAL_TOTG_PARAMETERS["velocity_scaling_factor"])),
            DeclareLaunchArgument("acceleration_scaling", default_value=str(FORMAL_TOTG_PARAMETERS["acceleration_scaling_factor"])),
            DeclareLaunchArgument("path_tolerance", default_value=str(FORMAL_TOTG_PARAMETERS["path_tolerance"])),
            DeclareLaunchArgument("resample_dt", default_value=str(FORMAL_TOTG_PARAMETERS["resample_dt"])),
            DeclareLaunchArgument("min_angle_change", default_value=str(FORMAL_TOTG_PARAMETERS["min_angle_change"])),
            DeclareLaunchArgument("ruckig_mitigate_overshoot", default_value=str(FORMAL_RUCKIG_PARAMETERS["mitigate_overshoot"]).lower()),
            DeclareLaunchArgument("ruckig_overshoot_threshold", default_value=str(FORMAL_RUCKIG_PARAMETERS["overshoot_threshold"])),
            # Additive recovery controls.  Defaults preserve the formal runner.
            DeclareLaunchArgument("stage25r_zero_target_acceleration", default_value="true"),
            DeclareLaunchArgument("stage25r_historical_replay", default_value="false"),
            DeclareLaunchArgument("stage25r2_shadow_sweep", default_value="false"),
            Node(
                executable=sys.executable,
                name="stage25_moveit_timing",
                output="screen",
                emulate_tty=True,
                arguments=[script],
                parameters=[
                    params,
                    {
                        "input_csv": LaunchConfiguration("input_csv"),
                        "source_reference_json": LaunchConfiguration("source_reference_json"),
                        "waypoints_csv": LaunchConfiguration("waypoints_csv"),
                        "output_dir": LaunchConfiguration("output_dir"),
                        "group_name": LaunchConfiguration("group_name"),
                        "ee_link": LaunchConfiguration("ee_link"),
                        "velocity_scaling": LaunchConfiguration("velocity_scaling"),
                        "acceleration_scaling": LaunchConfiguration("acceleration_scaling"),
                        "path_tolerance": LaunchConfiguration("path_tolerance"),
                        "resample_dt": LaunchConfiguration("resample_dt"),
                        "min_angle_change": LaunchConfiguration("min_angle_change"),
                        "ruckig_mitigate_overshoot": LaunchConfiguration("ruckig_mitigate_overshoot"),
                        "ruckig_overshoot_threshold": LaunchConfiguration("ruckig_overshoot_threshold"),
                        "stage25r_zero_target_acceleration": LaunchConfiguration("stage25r_zero_target_acceleration"),
                        "stage25r_historical_replay": LaunchConfiguration("stage25r_historical_replay"),
                        "stage25r2_shadow_sweep": LaunchConfiguration("stage25r2_shadow_sweep"),
                    },
                ],
            ),
        ]
    )
