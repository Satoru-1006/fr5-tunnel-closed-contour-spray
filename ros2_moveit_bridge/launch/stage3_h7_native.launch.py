from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
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
    derived_urdf = LaunchConfiguration("derived_urdf")
    # MoveItConfigsBuilder resolves the robot description while the launch
    # description is constructed, so it needs a concrete filesystem path.
    # The worker still receives the declared parameter and verifies the same
    # frozen file before formal computation.
    derived_urdf_path = Path("/mnt/d/robotfucker/outputs/stage3_h4_5_robot_fixture_geometry_codesign_20260809T000000Z/derived_robot_model.urdf")
    srdf = f"{bridge_share}/config/fairino5_v6_spray_tcp.srdf"
    joint_limits = f"{bridge_share}/config/joint_limits_with_jerk.yaml"
    moveit_config = (
        MoveItConfigsBuilder("fairino5_v6_robot", package_name="fairino5_v6_moveit2_config")
        .robot_description(file_path=str(derived_urdf_path))
        .robot_description_semantic(file_path=srdf)
        .joint_limits(file_path=joint_limits)
        .planning_pipelines(pipelines=["ompl"])
        .to_moveit_configs()
    )
    names = [
        "mode", "derived_urdf", "h6_joint_waypoints", "h6_cartesian_waypoints", "fixture_mesh",
        "output_dir", "sequence_manifest", "h6_1_contract", "dynamics_provenance",
        "totg_configuration", "ruckig_configuration", "input_freeze_manifest",
    ]
    declarations = [DeclareLaunchArgument("mode", default_value="audit")]
    declarations += [DeclareLaunchArgument(name) for name in names[1:]]
    parameters = {
        name: LaunchConfiguration(name)
        for name in names
    }
    return LaunchDescription(
        declarations
        + [
            Node(
                package="fr5_tunnel_moveit_bridge",
                executable="stage3_h7_native",
                name="stage3_h7_native",
                output="screen",
                parameters=[moveit_config.to_dict(), moveit_cpp_pipeline_params(moveit_config), parameters],
            )
        ]
    )
from pathlib import Path
