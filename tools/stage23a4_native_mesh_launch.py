from pathlib import Path

def generate_launch_description():  # pragma: no cover - ROS2 only
    from launch import LaunchDescription
    from launch.actions import DeclareLaunchArgument
    from launch.substitutions import LaunchConfiguration
    from launch_ros.actions import Node
    from moveit_configs_utils import MoveItConfigsBuilder
    repo = Path(__file__).resolve().parents[1]
    cfg = MoveItConfigsBuilder("fairino5_v6_robot", package_name="fairino5_v6_moveit2_config").robot_description(file_path=str(repo / "ros2_moveit_bridge/config/fairino5_v6_spray_tcp.urdf.xacro"), mappings={"tool_tcp_xyz":"0 0 0.150","tool_tcp_rpy":"0 0 0"}).robot_description_semantic(file_path=str(repo / "ros2_moveit_bridge/config/fairino5_v6_spray_tcp.srdf")).joint_limits(file_path=str(repo / "ros2_moveit_bridge/config/joint_limits_with_jerk.yaml")).planning_pipelines(pipelines=["ompl"]).to_moveit_configs()
    params=cfg.to_dict(); params["planning_pipelines"]={"pipeline_names":["ompl"],"ompl":params.get("ompl",{})}
    return LaunchDescription([
        DeclareLaunchArgument("candidate_csv"), DeclareLaunchArgument("pose_csv"), DeclareLaunchArgument("mesh_path"), DeclareLaunchArgument("output_dir"), DeclareLaunchArgument("backend", default_value="fcl"), DeclareLaunchArgument("run_index", default_value="1"),
        Node(package="stage23a4_native_mesh_probe", executable="stage23a4_native_mesh_probe", name="stage23a4_native_mesh_probe", output="screen", emulate_tty=True, parameters=[params,{"candidate_csv":LaunchConfiguration("candidate_csv"),"pose_csv":LaunchConfiguration("pose_csv"),"mesh_path":LaunchConfiguration("mesh_path"),"output_dir":LaunchConfiguration("output_dir"),"backend":LaunchConfiguration("backend"),"run_index":LaunchConfiguration("run_index"),"mesh_pose_xyz":[0.32,-0.05,-0.38],"group_name":"fairino5_v6_group"}])
    ])
