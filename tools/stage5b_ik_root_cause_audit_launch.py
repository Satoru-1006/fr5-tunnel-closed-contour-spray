"""Launch the Stage5B wp66->wp67 causal audit in a j6-isolated model."""

from __future__ import annotations

import sys
import copy
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def generate_launch_description():  # pragma: no cover - requires ROS 2 / MoveIt runtime
    from launch import LaunchDescription
    from launch.actions import DeclareLaunchArgument, OpaqueFunction
    from launch.substitutions import LaunchConfiguration

    def configure(context):
        import xacro
        from launch_ros.actions import Node
        from moveit_configs_utils import MoveItConfigsBuilder

        mode = str(LaunchConfiguration("mode").perform(context))
        output_dir = Path(str(LaunchConfiguration("output_dir").perform(context))).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        urdf_path = ROOT / "ros2_moveit_bridge/config/stage5_mock.urdf.xacro"
        srdf_path = ROOT / "ros2_moveit_bridge/config/fairino5_v6_spray_tcp.srdf"
        joint_limits_path = ROOT / "ros2_moveit_bridge/config/joint_limits_with_jerk.yaml"
        document = xacro.process_file(str(urdf_path), mappings={"tool_tcp_xyz": "0 0 0.150", "tool_tcp_rpy": "0 0 0"})
        robot_root = ET.fromstring(document.toxml())
        j6 = robot_root.find(".//joint[@name='j6']")
        if j6 is None:
            raise RuntimeError("j6_missing_from_expanded_model")
        limit = j6.find("limit")
        if limit is None:
            raise RuntimeError("j6_limit_missing_from_expanded_model")
        formal_lower = float(limit.attrib["lower"])
        formal_upper = float(limit.attrib["upper"])
        if mode == "shadow_360":
            # This is the only model limit intentionally changed.  The formal
            # xacro, URDF, SRDF and MoveIt config remain untouched.
            limit.attrib["lower"] = str(-2.0 * 3.141592653589793)
            limit.attrib["upper"] = str(2.0 * 3.141592653589793)
        elif mode != "control_175":
            raise RuntimeError(f"invalid_mode:{mode}")
        model_path = output_dir / "expanded_robot_model.urdf"
        model_xml = ET.tostring(robot_root, encoding="unicode")
        model_path.write_text(model_xml + "\n", encoding="utf-8")
        (output_dir / "j6_limit_change.json").write_text(
            __import__("json").dumps({"mode": mode, "formal_lower_rad": formal_lower, "formal_upper_rad": formal_upper, "model_lower_rad": float(limit.attrib["lower"]), "model_upper_rad": float(limit.attrib["upper"]), "only_changed_field": "j6.limit.lower/upper", "planning_position_override": None if mode != "shadow_360" else {"joint": "j6", "has_position_limits": True, "min_position": -2.0 * 3.141592653589793, "max_position": 2.0 * 3.141592653589793}, "formal_model_untouched": True}, indent=2) + "\n",
            encoding="utf-8",
        )

        configs = (
            MoveItConfigsBuilder("fairino5_v6_robot", package_name="fairino5_v6_moveit2_config")
            .robot_description(file_path=str(urdf_path), mappings={"tool_tcp_xyz": "0 0 0.150", "tool_tcp_rpy": "0 0 0"})
            .robot_description_semantic(file_path=str(srdf_path))
            .joint_limits(file_path=str(joint_limits_path))
            .planning_pipelines(pipelines=["ompl"])
            .to_moveit_configs()
        )
        params = configs.to_dict()
        params["robot_description"] = model_xml
        if mode == "shadow_360":
            # MoveIt may retain the URDF safety-bound as the effective
            # VariableBounds when the planning parameter has no explicit
            # position bound.  Make the same J6-only ablation explicit in the
            # isolated planning parameter so the measured KDL bounds are
            # actually the requested diagnostic bounds.
            planning = copy.deepcopy(params.get("robot_description_planning", {}))
            planning.setdefault("joint_limits", {}).setdefault("j6", {}).update({
                "has_position_limits": True,
                "min_position": -2.0 * 3.141592653589793,
                "max_position": 2.0 * 3.141592653589793,
            })
            params["robot_description_planning"] = planning
        params["planning_pipelines"] = {"pipeline_names": ["ompl"], "ompl": params.get("ompl", {})}
        return [
            Node(
                executable=sys.executable,
                name=f"stage5b_ik_root_cause_{mode}",
                output="screen",
                emulate_tty=True,
                arguments=[str(ROOT / "tools/stage5b_ik_root_cause_audit.py")],
                parameters=[params, {
                    "poses_csv": LaunchConfiguration("poses_csv"),
                    "source_q_csv": LaunchConfiguration("source_q_csv"),
                    "output_dir": str(output_dir),
                    "model_path": str(model_path),
                    "mode": mode,
                    "subdivisions": 512,
                    "ik_timeout_s": 0.005,
                }],
            )
        ]

    return LaunchDescription([
        DeclareLaunchArgument("poses_csv"),
        DeclareLaunchArgument("source_q_csv"),
        DeclareLaunchArgument("output_dir"),
        DeclareLaunchArgument("mode", default_value="control_175"),
        OpaqueFunction(function=configure),
    ])
