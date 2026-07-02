from __future__ import annotations

import ast
from pathlib import Path
import xml.etree.ElementTree as ET


BRIDGE = Path(__file__).resolve().parents[1] / "ros2_moveit_bridge" / "plan_closed_contour_moveit.py"


def _load_bridge_function(name: str):
    tree = ast.parse(BRIDGE.read_text(encoding="utf-8"))
    function_node = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
    module = ast.Module(body=[function_node], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace: dict[str, object] = {}
    if name in {"_nearest_polyline_distance", "_project_to_polyline_with_normals", "_tcp_speed_from_positions"}:
        import numpy as np

        namespace["np"] = np
    exec(compile(module, str(BRIDGE), "exec"), namespace)
    return namespace[name]


def test_parameter_bool_handles_ros_string_values() -> None:
    parameter_bool = _load_bridge_function("_parameter_bool")

    assert parameter_bool(True) is True
    assert parameter_bool(False) is False
    assert parameter_bool("true") is True
    assert parameter_bool("False") is False
    assert parameter_bool("0") is False
    assert parameter_bool("yes") is True


def test_bridge_uses_native_moveit_ruckig_and_preflight() -> None:
    source = BRIDGE.read_text(encoding="utf-8")

    assert "trajectory.apply_ruckig_smoothing" in source
    assert "trajectory.apply_totg_time_parameterization" in source
    build_source = source[source.index("def build_and_smooth_moveit_trajectory") :]
    assert build_source.index("trajectory.apply_totg_time_parameterization") < build_source.index(
        "_apply_ruckig_smoothing_without_known_false_error"
    )
    assert "Ruckig extended the trajectory duration" in source
    assert "preflight_moveit_runtime" in source
    assert "write_joint_trajectory_csv" in source
    assert "validate_joint_dynamics" in source
    assert "moveit_joint_dynamics_report.csv" in source
    assert "moveit_ruckig_smoothing_used" in source
    assert "moveit_fk_tcp_trace.csv" in source
    assert "standoff_error_mm" in source
    assert "tcp_speed_m_s" in source
    assert "execute_trajectory" in source
    assert "Execution requires MoveIt2 native Ruckig smoothing" in source
    assert "smooth_robot_trajectory_with_ruckig" not in source
    assert "jerk limits are missing" in source


def test_bridge_fk_speed_validation_uses_segment_speed_not_arc_gradient() -> None:
    source = BRIDGE.read_text(encoding="utf-8")

    assert "_tcp_speed_from_positions(actual, time)" in source
    assert "_project_to_polyline_with_normals(actual, wall, target_normals)" in source
    assert "np.gradient(arc, time" not in source


def test_bridge_tcp_speed_helper_has_no_endpoint_gradient_spike() -> None:
    tcp_speed = _load_bridge_function("_tcp_speed_from_positions")
    import numpy as np

    t = np.array([0.0, 0.5, 1.0, 1.5])
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.1, 0.0, 0.0],
            [0.2, 0.0, 0.0],
            [0.3, 0.0, 0.0],
        ]
    )

    assert np.allclose(tcp_speed(points, t), 0.2)


def test_smoke_test_console_script_is_installed() -> None:
    setup_py = BRIDGE.parents[0] / "setup.py"
    source = setup_py.read_text(encoding="utf-8")

    assert "smoke_test_moveit_bridge" in source
    assert "smoke_test_moveit_bridge:main" in source
    assert "validate_bridge_inputs" in source


def test_validate_bridge_inputs_accepts_generated_tcp_poses() -> None:
    import importlib.util

    module_path = BRIDGE.parents[0] / "validate_bridge_inputs.py"
    spec = importlib.util.spec_from_file_location("validate_bridge_inputs", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    tcp_csv = BRIDGE.parents[1] / "outputs" / "tcp_poses.csv"
    metrics = module.validate_tcp_pose_csv(tcp_csv, samples_per_loop=240)
    assert metrics["status"] == "pass"
    assert float(metrics["normal_angle_error_max_deg"]) < 0.1
    assert float(metrics["close_position_error"]) == 0.0


def test_generated_offline_reports_include_fk_speed_and_ik_waypoints() -> None:
    import csv

    root = BRIDGE.parents[1]
    quality_csv = root / "outputs" / "quality_report.csv"
    ik_csv = root / "outputs" / "ik_waypoints.csv"

    quality = {row["metric"]: row["value"] for row in csv.DictReader(quality_csv.open(encoding="utf-8"))}
    assert quality["status"] in {"pass", "fail"}
    for metric in [
        "fk_tcp_speed_mean",
        "fk_tcp_speed_min",
        "fk_tcp_speed_max",
        "fk_tcp_speed_fluctuation",
        "fk_tcp_speed_p05_p95_fluctuation",
        "fk_tcp_speed_status",
        "fk_tcp_speed_p05_p95_status",
    ]:
        assert metric in quality
    assert quality["fk_tcp_speed_p05_p95_status"] == "pass"

    with ik_csv.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == ["waypoint", "q1", "q2", "q3", "q4", "q5", "q6"]
        first = next(reader)
    assert int(first["waypoint"]) == 0


def test_spray_tcp_srdf_sets_group_tip_to_tcp_link() -> None:
    srdf = BRIDGE.parents[0] / "config" / "fairino5_v6_spray_tcp.srdf"
    root = ET.parse(srdf).getroot()
    group = root.find("./group[@name='fairino5_v6_group']")
    assert group is not None
    chain = group.find("chain")
    assert chain is not None
    assert chain.attrib["base_link"] == "base_link"
    assert chain.attrib["tip_link"] == "spray_tcp_link"


def test_demo_launch_exposes_quality_gate_parameters() -> None:
    launch_file = BRIDGE.parents[0] / "launch" / "fr5_spray_demo.launch.py"
    source = launch_file.read_text(encoding="utf-8")

    for name in [
        "velocity_scaling",
        "acceleration_scaling",
        "max_normal_error_deg",
        "max_standoff_fraction",
        "max_speed_fluctuation",
        "quality_report_csv",
        "fk_trace_csv",
        "trajectory_csv",
        "dynamics_report_csv",
    ]:
        assert f'DeclareLaunchArgument("{name}"' in source
        assert f'"{name}":' in source


def test_strict_moveit_validation_script_runs_full_audit_chain() -> None:
    script = BRIDGE.parents[1] / "scripts" / "run_moveit_strict_validation.sh"
    source = script.read_text(encoding="utf-8")

    assert "validate_bridge_inputs.py" in source
    assert "ros2 run fr5_tunnel_moveit_bridge validate_bridge_inputs" in source
    assert "ros2 launch fr5_tunnel_moveit_bridge fr5_spray_plan_only.launch.py" in source
    assert "moveit_quality_report.csv" in source
    assert "moveit_joint_dynamics_report.csv" in source
    assert "moveit_fk_tcp_trace.csv" in source
    assert "moveit_runtime.log" in source
    assert "FILTER_KNOWN_RUCKIG_WARNING" in source
    assert "--strict-runtime" in source
    assert "audit_goal_requirements_strict.json" in source
    assert "execute_trajectory:=false" in source
    assert "publish_final_outputs.py" in source


def test_nearest_polyline_distance_uses_segments() -> None:
    nearest_polyline_distance = _load_bridge_function("_nearest_polyline_distance")
    import numpy as np

    square = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
        ]
    )
    points = np.array(
        [
            [0.5, 0.0, 0.0],
            [0.5, 0.0, 0.2],
            [1.2, 0.0, 0.5],
        ]
    )

    distances = nearest_polyline_distance(points, square)
    assert np.allclose(distances, [0.0, 0.2, 0.2])


def test_bridge_standoff_projection_interpolates_normals_on_segments() -> None:
    project = _load_bridge_function("_project_to_polyline_with_normals")
    import numpy as np

    line = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 1.0],
        ]
    )
    normals = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
        ]
    )
    point = np.array([[0.5, 0.0, 0.2]])
    projected, projected_normals, distances = project(point, line, normals)
    expected_normal = np.array([1.0, 0.0, 1.0])
    expected_normal /= np.linalg.norm(expected_normal)

    assert np.allclose(projected, [[0.5, 0.0, 0.0]])
    assert np.allclose(projected_normals[0], expected_normal)
    assert np.allclose(distances, [0.2])


def test_goal_requirement_audit_has_no_failures() -> None:
    import importlib.util
    import sys

    audit_path = BRIDGE.parents[1] / "tools" / "audit_goal_requirements.py"
    spec = importlib.util.spec_from_file_location("audit_goal_requirements", audit_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    items = module.audit_goal_requirements()
    failures = [item for item in items if item.status == "fail"]

    assert not failures


def test_goal_audit_validates_moveit_runtime_reports(tmp_path) -> None:
    import importlib.util
    import sys

    audit_path = BRIDGE.parents[1] / "tools" / "audit_goal_requirements.py"
    spec = importlib.util.spec_from_file_location("audit_goal_requirements_runtime", audit_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    quality = tmp_path / "moveit_quality_report.csv"
    dynamics = tmp_path / "moveit_joint_dynamics_report.csv"
    trace = tmp_path / "moveit_fk_tcp_trace.csv"
    quality.write_text(
        "\n".join(
            [
                "metric,value",
                "fk_normal_error_max_deg,1.0",
                "fk_standoff_error_max_abs_mm,2.0",
                "fk_path_deviation_max_mm,3.0",
                "fk_tcp_speed_p05_p95_fluctuation,0.01",
                "moveit_ruckig_smoothing_used,true",
                "moveit_ee_link,spray_tcp_link",
                "status,pass",
            ]
        ),
        encoding="utf-8",
    )
    dynamics.write_text(
        "\n".join(
            [
                "joint,velocity_ratio,acceleration_ratio,jerk_ratio",
                "j1,0.1,0.2,0.3",
                "j2,1.0,1.01,1.02",
            ]
        ),
        encoding="utf-8",
    )
    trace.write_text(
        "\n".join(
            [
                "t,actual_tcp_x,actual_tcp_y,actual_tcp_z,normal_angle_error_deg,standoff_error_mm,path_deviation_mm,tcp_speed_m_s",
                "0.0,0.1,0.2,0.3,1.0,2.0,3.0,0.08",
            ]
        ),
        encoding="utf-8",
    )

    status, evidence = module._moveit_runtime_report_status(quality, dynamics, trace)
    assert status == "pass"
    assert "FK trace rows=1" in evidence

    dynamics.write_text(
        "\n".join(
            [
                "joint,velocity_ratio,acceleration_ratio,jerk_ratio",
                "j1,0.1,0.2,1.5",
            ]
        ),
        encoding="utf-8",
    )
    status, evidence = module._moveit_runtime_report_status(quality, dynamics, trace)
    assert status == "fail"
    assert "jerk_ratio" in evidence

    quality.write_text(
        "\n".join(
            [
                "metric,value",
                "fk_normal_error_max_deg,1.0",
                "fk_standoff_error_max_abs_mm,2.0",
                "fk_path_deviation_max_mm,3.0",
                "fk_tcp_speed_p05_p95_fluctuation,0.01",
                "moveit_ruckig_smoothing_used,false",
                "moveit_ee_link,spray_tcp_link",
                "status,pass",
            ]
        ),
        encoding="utf-8",
    )
    dynamics.write_text(
        "\n".join(
            [
                "joint,velocity_ratio,acceleration_ratio,jerk_ratio",
                "j1,0.1,0.2,0.3",
            ]
        ),
        encoding="utf-8",
    )
    status, evidence = module._moveit_runtime_report_status(quality, dynamics, trace)
    assert status == "fail"
    assert "Ruckig" in evidence

    quality.write_text(
        "\n".join(
            [
                "metric,value",
                "fk_normal_error_max_deg,1.0",
                "fk_standoff_error_max_abs_mm,2.0",
                "fk_path_deviation_max_mm,3.0",
                "fk_tcp_speed_p05_p95_fluctuation,0.01",
                "moveit_ruckig_smoothing_used,true",
                "moveit_ee_link,wrist3_link",
                "status,pass",
            ]
        ),
        encoding="utf-8",
    )
    status, evidence = module._moveit_runtime_report_status(quality, dynamics, trace)
    assert status == "fail"
    assert "ee_link" in evidence

    trace.write_text(
        "\n".join(
            [
                "t,actual_tcp_x",
                "0.0,0.1",
            ]
        ),
        encoding="utf-8",
    )
    quality.write_text(
        "\n".join(
            [
                "metric,value",
                "fk_normal_error_max_deg,1.0",
                "fk_standoff_error_max_abs_mm,2.0",
                "fk_path_deviation_max_mm,3.0",
                "fk_tcp_speed_p05_p95_fluctuation,0.01",
                "moveit_ruckig_smoothing_used,true",
                "moveit_ee_link,spray_tcp_link",
                "status,pass",
            ]
        ),
        encoding="utf-8",
    )
    status, evidence = module._moveit_runtime_report_status(quality, dynamics, trace)
    assert status == "fail"
    assert "FK trace" in evidence

    trace.write_text(
        "\n".join(
            [
                "t,actual_tcp_x,actual_tcp_y,actual_tcp_z,normal_angle_error_deg,standoff_error_mm,path_deviation_mm,tcp_speed_m_s",
                "0.0,0.1,0.2,0.3,1.0,2.0,3.0,0.08",
                "0.0,0.1,0.2,0.3,1.0,2.0,3.0,0.08",
            ]
        ),
        encoding="utf-8",
    )
    status, evidence = module._moveit_runtime_report_status(quality, dynamics, trace)
    assert status == "fail"
    assert "timestamps" in evidence

    trace.write_text(
        "\n".join(
            [
                "t,actual_tcp_x,actual_tcp_y,actual_tcp_z,normal_angle_error_deg,standoff_error_mm,path_deviation_mm,tcp_speed_m_s",
                "0.0,0.1,0.2,0.3,2.0,2.0,3.0,0.08",
            ]
        ),
        encoding="utf-8",
    )
    status, evidence = module._moveit_runtime_report_status(quality, dynamics, trace)
    assert status == "fail"
    assert "normal-angle" in evidence


def test_publish_final_outputs_merges_moveit_strict_reports(tmp_path) -> None:
    import csv
    import importlib.util
    import json
    import sys

    publisher_path = BRIDGE.parents[1] / "tools" / "publish_final_outputs.py"
    spec = importlib.util.spec_from_file_location("publish_final_outputs", publisher_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    quality = tmp_path / "moveit_quality_report.csv"
    dynamics = tmp_path / "moveit_joint_dynamics_report.csv"
    collision = tmp_path / "moveit_collision_report.csv"
    audit = tmp_path / "audit_goal_requirements_strict.json"
    runtime_log = tmp_path / "moveit_runtime.log"
    out_csv = tmp_path / "final_quality_report.csv"
    out_json = tmp_path / "final_acceptance_summary.json"
    quality.write_text(
        "\n".join(
            [
                "metric,value",
                "fk_normal_error_mean_deg,0.4",
                "fk_normal_error_max_deg,2.0",
                "fk_standoff_error_max_abs_mm,1.2",
                "fk_path_deviation_p95_mm,0.1",
                "fk_path_deviation_max_mm,1.8",
                "fk_tcp_speed_mean_m_s,0.003",
                "fk_tcp_speed_p05_m_s,0.003",
                "fk_tcp_speed_p95_m_s,0.003",
                "fk_tcp_speed_p05_p95_fluctuation,0.001",
                "moveit_ruckig_smoothing_used,true",
                "moveit_time_parameterization,tcp_arclength",
                "moveit_ee_link,spray_tcp_link",
                "status,pass",
            ]
        ),
        encoding="utf-8",
    )
    dynamics.write_text(
        "\n".join(
            [
                "joint,velocity_ratio,acceleration_ratio,jerk_ratio",
                "j1,0.1,0.2,0.3",
                "j2,0.2,0.3,0.4",
            ]
        ),
        encoding="utf-8",
    )
    collision.write_text(
        "\n".join(
            [
                "metric,value",
                "collision_environment_object_count,60",
                "collision_checked_state_count,145",
                "collision_count,0",
                "status,pass",
            ]
        ),
        encoding="utf-8",
    )
    audit.write_text(json.dumps({"overall_status": "pass", "items": []}), encoding="utf-8")
    runtime_log.write_text(
        "Ruckig extended the trajectory duration to its maximum and still did not find a solution\n",
        encoding="utf-8",
    )

    module.write_final_quality_report(quality, dynamics, collision, audit, out_csv, out_json, runtime_log)
    final_quality = {row["metric"]: row["value"] for row in csv.DictReader(out_csv.open(encoding="utf-8"))}
    summary = json.loads(out_json.read_text(encoding="utf-8"))

    assert final_quality["result_source"] == "moveit2_strict_runtime"
    assert final_quality["status"] == "pass"
    assert final_quality["runtime_log_source"] == str(runtime_log)
    assert final_quality["ruckig_known_warning_count_raw_log"] == "1"
    assert final_quality["max_jerk_ratio"] == "0.4"
    assert final_quality["collision_count"] == "0"
    assert summary["overall_status"] == "pass"


def test_goal_audit_strict_runtime_fails_without_runtime_reports(tmp_path) -> None:
    import subprocess
    import sys

    audit_path = BRIDGE.parents[1] / "tools" / "audit_goal_requirements.py"
    normal = subprocess.run([sys.executable, str(audit_path)], cwd=BRIDGE.parents[1], capture_output=True, text=True)
    strict = subprocess.run(
        [
            sys.executable,
            str(audit_path),
            "--strict-runtime",
            "--moveit-quality-report",
            str(tmp_path / "missing_quality.csv"),
            "--moveit-dynamics-report",
            str(tmp_path / "missing_dynamics.csv"),
            "--moveit-fk-trace",
            str(tmp_path / "missing_trace.csv"),
        ],
        cwd=BRIDGE.parents[1],
        capture_output=True,
        text=True,
    )

    assert normal.returncode == 0
    assert strict.returncode != 0
    assert "WARN: ROS2/MoveIt2 runtime execution-chain evidence" in strict.stdout


def test_goal_audit_strict_runtime_accepts_explicit_runtime_reports(tmp_path) -> None:
    import subprocess
    import sys

    quality = tmp_path / "moveit_quality_report.csv"
    dynamics = tmp_path / "moveit_joint_dynamics_report.csv"
    trace = tmp_path / "moveit_fk_tcp_trace.csv"
    quality.write_text(
        "\n".join(
            [
                "metric,value",
                "fk_normal_error_max_deg,1.0",
                "fk_standoff_error_max_abs_mm,2.0",
                "fk_path_deviation_max_mm,3.0",
                "fk_tcp_speed_p05_p95_fluctuation,0.01",
                "moveit_ruckig_smoothing_used,true",
                "moveit_ee_link,spray_tcp_link",
                "status,pass",
            ]
        ),
        encoding="utf-8",
    )
    dynamics.write_text(
        "\n".join(
            [
                "joint,velocity_ratio,acceleration_ratio,jerk_ratio",
                "j1,0.1,0.2,0.3",
            ]
        ),
        encoding="utf-8",
    )
    trace.write_text(
        "\n".join(
            [
                "t,actual_tcp_x,actual_tcp_y,actual_tcp_z,normal_angle_error_deg,standoff_error_mm,path_deviation_mm,tcp_speed_m_s",
                "0.0,0.1,0.2,0.3,1.0,2.0,3.0,0.08",
            ]
        ),
        encoding="utf-8",
    )
    audit_path = BRIDGE.parents[1] / "tools" / "audit_goal_requirements.py"
    result = subprocess.run(
        [
            sys.executable,
            str(audit_path),
            "--strict-runtime",
            "--moveit-quality-report",
            str(quality),
            "--moveit-dynamics-report",
            str(dynamics),
            "--moveit-fk-trace",
            str(trace),
        ],
        cwd=BRIDGE.parents[1],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "PASS: ROS2/MoveIt2 runtime execution-chain evidence" in result.stdout


def test_goal_audit_writes_json_report(tmp_path) -> None:
    import json
    import subprocess
    import sys

    audit_path = BRIDGE.parents[1] / "tools" / "audit_goal_requirements.py"
    report = tmp_path / "audit.json"
    result = subprocess.run(
        [sys.executable, str(audit_path), "--json-report", str(report)],
        cwd=BRIDGE.parents[1],
        capture_output=True,
        text=True,
    )
    payload = json.loads(report.read_text(encoding="utf-8"))

    assert result.returncode == 0
    assert payload["overall_status"] == "pass"
    assert payload["items"]
    assert {"requirement", "status", "evidence"}.issubset(payload["items"][0])
