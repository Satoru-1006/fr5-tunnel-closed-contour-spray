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
    if name in {
        "circular_joint_delta",
        "_ik_seed_candidates",
        "_nearest_polyline_distance",
        "_project_to_polyline_with_normals",
        "_symmetric_roll_offsets",
        "_rotation_about_local_axis",
        "_continuous_curve_step_limit",
        "_tcp_speed_from_positions",
    }:
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
    assert "write_joint_waypoint_csv" in source
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
    assert "_project_to_polyline_with_normals(" in source
    assert "closed=not open_path" in source
    assert "np.gradient(arc, time" not in source


def test_execution_refuses_failed_fk_quality_report_status() -> None:
    source = BRIDGE.read_text(encoding="utf-8")
    execute_source = source[source.index("if execute_trajectory:") :]

    assert 'quality_metrics.get("status") != "pass"' in execute_source
    assert "FK quality report status" in execute_source
    assert execute_source.index('quality_metrics.get("status") != "pass"') < execute_source.index("moveit.execute")


def test_fk_joint_step_diagnostics_assign_reported_indices() -> None:
    source = BRIDGE.read_text(encoding="utf-8")
    fk_source = source[source.index("def validate_smoothed_trajectory_fk") :]

    assert "max_joint_step_from_index = int(max_step_from_index)" in fk_source
    assert '"max_joint_step_from_index": max_joint_step_from_index' in fk_source
    assert '"max_joint_step_to_index": max_joint_step_to_index' in fk_source
    assert "circular_joint_delta(np.diff(q, axis=0))" in fk_source
    assert '"max_joint_step_raw_deg": max_joint_step_raw_deg' in fk_source
    assert "tool_tcp_measured_by" in fk_source
    assert "tool_tcp_measured_date" in fk_source
    assert "tool_tcp_calibration_method" in fk_source


def test_circular_joint_delta_distinguishes_wrap_from_real_branch_jump() -> None:
    circular_joint_delta = _load_bridge_function("circular_joint_delta")
    import numpy as np

    deltas = np.deg2rad(np.asarray([357.0, 145.0, -358.0]))
    shortest = np.rad2deg(circular_joint_delta(deltas))

    assert np.allclose(np.abs(shortest), [3.0, 145.0, 2.0])


def test_wall_collision_objects_keep_solid_volume_behind_coating_surface() -> None:
    source = BRIDGE.read_text(encoding="utf-8")
    wall_source = source[source.index("def build_wall_collision_objects") :]

    assert "if float(np.dot(z_axis, normal)) < 0.0:" in wall_source
    assert "z_axis = -z_axis" in wall_source
    assert "wall_sign = 1.0 if tcp_points_to_wall else -1.0" in wall_source
    assert "center = (p0 + p1) * 0.5 + wall_sign * normal * (wall_thickness * 0.5)" in wall_source


def test_ik_collision_diagnostics_keep_moveitpy_contact_binding_nonfatal() -> None:
    source = BRIDGE.read_text(encoding="utf-8")
    diagnostic_source = source[source.index("def _collision_result_summary") :]
    ik_source = source[source.index("def build_ik_waypoint_trajectory") :]

    assert "except TypeError:" in diagnostic_source
    assert "pairs=unavailable" in diagnostic_source
    assert "_state_collision_summary(scene, state, group_name)" in source
    assert "scene.is_state_colliding(state, group_name, True)" in ik_source


def test_ik_waypoint_solver_exposes_beam_branch_selection() -> None:
    source = BRIDGE.read_text(encoding="utf-8")
    ik_source = source[source.index("def build_ik_waypoint_trajectory") :]

    assert "ik_beam_width: int = 1" in ik_source
    assert "ik_candidates_per_beam: int = 8" in ik_source
    assert "ik_beam_diversity_joint_deg: float = 0.0" in ik_source
    assert "ik_search_diagnostics_csv: Path | None = None" in ik_source
    assert "ik_max_solve_seconds: float = 0.0" in ik_source
    assert "beam_width = max(1, int(ik_beam_width))" in ik_source
    assert "candidates_per_beam = max(1, int(ik_candidates_per_beam))" in ik_source
    assert "beam_diversity_joint = max(0.0, np.deg2rad(float(ik_beam_diversity_joint_deg)))" in ik_source
    assert "max_solve_seconds = max(0.0, float(ik_max_solve_seconds))" in ik_source
    assert "_write_ik_search_diagnostics(ik_search_diagnostics_csv, diagnostic_records)" in ik_source
    assert "_select_diverse_ik_beams(next_beams, beam_width, beam_diversity_joint)" in ik_source
    assert '"status": "timeout"' in ik_source
    assert '"status": "fail"' in ik_source
    assert "ParameterDescriptor(dynamic_typing=True)" in source
    assert "beams: list[tuple[float, float, float, list[np.ndarray]]]" in ik_source
    assert "valid_candidates[:candidates_per_beam]" in ik_source
    assert "next_bottleneck_cost = max(beam_bottleneck_cost, max_delta)" in ik_source
    assert "next_step_sum_cost = beam_step_sum_cost + max_delta" in ik_source
    assert "next_beams.sort(key=lambda item: (item[0], item[1], item[2]))" in ik_source
    assert "beams = _select_diverse_ik_beams(next_beams, beam_width, beam_diversity_joint)" in ik_source


def test_ik_seed_candidates_cover_shoulder_elbow_and_wrist_branches() -> None:
    ik_seed_candidates = _load_bridge_function("_ik_seed_candidates")
    import numpy as np

    previous = np.zeros(6)
    seeds = ik_seed_candidates(previous, previous)

    assert any(np.isclose(seed[1], np.pi) for seed in seeds)
    assert any(np.isclose(seed[2], -np.pi) for seed in seeds)
    assert any(np.isclose(seed[5], 2.0 * np.pi) for seed in seeds)
    assert any(np.isclose(seed[1], np.pi) and np.isclose(seed[2], -np.pi) for seed in seeds)


def test_ik_roll_offsets_preserve_zero_first_and_cover_both_directions() -> None:
    symmetric_roll_offsets = _load_bridge_function("_symmetric_roll_offsets")
    import numpy as np

    offsets = symmetric_roll_offsets(5)

    assert offsets[0] == 0.0
    assert any(value > 0.0 for value in offsets)
    assert any(value < 0.0 for value in offsets)
    assert np.isclose(max(abs(value) for value in offsets), 4.0 * np.pi / 5.0)


def test_global_roll_curve_step_limit_unwraps_and_caps_discontinuities() -> None:
    curve_step_limit = _load_bridge_function("_continuous_curve_step_limit")
    import numpy as np

    curve = curve_step_limit(np.deg2rad(np.asarray([170.0, -170.0, -160.0])), np.deg2rad(15.0))

    assert np.all(np.diff(curve) <= np.deg2rad(15.0) + 1e-12)
    assert np.all(np.abs(np.diff(curve)) <= np.deg2rad(15.0) + 1e-12)


def test_global_roll_backtracking_source_is_the_active_candidate_bank_algorithm() -> None:
    source = BRIDGE.read_text(encoding="utf-8")

    assert "collect_ik_candidate_bank_global_roll_backtracking" in source
    assert "_continuous_tcp_roll_curve(poses)" in source
    assert "_global_roll_reparameterized_pose" in source
    assert "visited_failures" in source
    assert "backtracks += 1" in source
    assert '"algorithm": "global_roll_curve_backtracking_tcp_reparameterization"' in source
    assert "planning_mode in {\"ik_candidate_bank\", \"ik_global_roll_backtracking\"}" in source


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


def test_bridge_declares_yaml_and_direct_ros_runtime_dependencies() -> None:
    package_root = BRIDGE.parents[0]
    setup_source = (package_root / "setup.py").read_text(encoding="utf-8")
    package = ET.parse(package_root / "package.xml").getroot()
    ros_dependencies = {element.text for element in package.findall("exec_depend")}

    assert "PyYAML" in setup_source
    assert "python3-yaml" in ros_dependencies
    assert "moveit_msgs" in ros_dependencies
    assert "shape_msgs" in ros_dependencies
    assert "rcl_interfaces" in ros_dependencies


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


def test_measured_tool_tcp_template_documents_production_rerun_gates() -> None:
    template = BRIDGE.parents[0] / "config" / "tool_tcp_calibration_template.yaml"
    source = template.read_text(encoding="utf-8")

    assert "translation_xyz" in source
    assert "rotation_rpy" in source
    assert "TOOL_TCP_XYZ" in source
    assert "TOOL_TCP_RPY" in source
    assert "TOOL_TCP_SOURCE" in source
    assert "required_metadata" in source
    assert "measured_date: YYYY-MM-DD" in source
    assert "calibration_method: non_empty" in source
    assert "tool_tcp_production_status: pass" in source
    assert "joint_continuity_status: pass" in source
    assert "collision_status: pass" in source


def test_load_tool_tcp_calibration_accepts_measured_yaml_and_rejects_placeholder(tmp_path) -> None:
    import importlib.util
    import pytest

    loader_path = BRIDGE.parents[1] / "tools" / "load_tool_tcp_calibration.py"
    spec = importlib.util.spec_from_file_location("load_tool_tcp_calibration", loader_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    measured = tmp_path / "measured_tool_tcp.yaml"
    measured.write_text(
        "\n".join(
            [
                "tool_tcp:",
                "  source: measured_nozzle_tcp_2026_07_06",
                "  translation_xyz: [0.01, 0.0, 0.185]",
                "  rotation_rpy: [0.0, 0.1, 0.0]",
                "  measured_by: calibration_engineer",
                "  measured_date: 2026-07-06",
                "  calibration_method: flange_fixture_probe",
            ]
        ),
        encoding="utf-8",
    )
    values = module.load_calibration(measured)
    assert values["TOOL_TCP_XYZ"] == "0.01 0 0.185"
    assert values["TOOL_TCP_RPY"] == "0 0.1 0"
    assert values["TOOL_TCP_SOURCE"] == "measured_nozzle_tcp_2026_07_06"
    assert values["TOOL_TCP_MEASURED_BY"] == "calibration_engineer"
    assert values["TOOL_TCP_MEASURED_DATE"] == "2026-07-06"
    assert values["TOOL_TCP_CALIBRATION_METHOD"] == "flange_fixture_probe"

    placeholder = tmp_path / "placeholder.yaml"
    placeholder.write_text(
        "\n".join(
            [
                "tool_tcp:",
                "  source: assumed_150mm_placeholder",
                "  translation_xyz: [0.0, 0.0, 0.150]",
                "  rotation_rpy: [0.0, 0.0, 0.0]",
            ]
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="not production measured"):
        module.load_calibration(placeholder)

    missing_metadata = tmp_path / "missing_metadata.yaml"
    missing_metadata.write_text(
        "\n".join(
            [
                "tool_tcp:",
                "  source: measured_nozzle_tcp_2026_07_06",
                "  translation_xyz: [0.01, 0.0, 0.185]",
                "  rotation_rpy: [0.0, 0.1, 0.0]",
            ]
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="measured_by"):
        module.load_calibration(missing_metadata)

    bad_date = tmp_path / "bad_date.yaml"
    bad_date.write_text(
        "\n".join(
            [
                "tool_tcp:",
                "  source: measured_nozzle_tcp_2026_07_06",
                "  translation_xyz: [0.01, 0.0, 0.185]",
                "  rotation_rpy: [0.0, 0.1, 0.0]",
                "  measured_by: calibration_engineer",
                "  measured_date: 07/06/2026",
                "  calibration_method: flange_fixture_probe",
            ]
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="YYYY-MM-DD"):
        module.load_calibration(bad_date)


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

    assert 'TOOL_TCP_XYZ="${TOOL_TCP_XYZ:-0.000 0.000 0.150}"' in source
    assert 'TOOL_TCP_CALIBRATION_YAML="${TOOL_TCP_CALIBRATION_YAML:-}"' in source
    assert 'TOOL_TCP_SOURCE="${TOOL_TCP_SOURCE:-assumed_150mm_placeholder}"' in source
    assert 'TOOL_TCP_MEASURED_BY="${TOOL_TCP_MEASURED_BY:-simulation}"' in source
    assert 'TOOL_TCP_MEASURED_DATE="${TOOL_TCP_MEASURED_DATE:-not_applicable_virtual_design}"' in source
    assert 'TOOL_TCP_CALIBRATION_METHOD="${TOOL_TCP_CALIBRATION_METHOD:-virtual_design_parameter}"' in source
    assert 'ALLOW_ZERO_TOOL_TCP="${ALLOW_ZERO_TOOL_TCP:-false}"' in source
    assert 'ALLOW_SEED_JOINT_WITH_TOOL_OFFSET="${ALLOW_SEED_JOINT_WITH_TOOL_OFFSET:-false}"' in source
    assert 'PLANNING_MODE="${PLANNING_MODE:-ik_waypoints}"' in source
    assert 'SEED_JOINT_CSV="${SEED_JOINT_CSV:-}"' in source
    assert 'QUALITY_REPORT="${QUALITY_REPORT:-$REPO_ROOT/outputs/moveit_quality_report.csv}"' in source
    assert 'RUNTIME_LOG="${RUNTIME_LOG:-$REPO_ROOT/outputs/moveit_runtime.log}"' in source
    assert 'FINAL_OUT_DIR="${FINAL_OUT_DIR:-$REPO_ROOT/outputs}"' in source
    assert 'PRODUCTION_READINESS_JSON="${PRODUCTION_READINESS_JSON:-$FINAL_OUT_DIR/production_readiness_check.json}"' in source
    assert 'mkdir -p \\' in source
    assert '"$(dirname "$QUALITY_REPORT")"' in source
    assert '"$(dirname "$PRODUCTION_READINESS_JSON")"' in source
    assert '"$FINAL_OUT_DIR"' in source
    assert "load_tool_tcp_calibration.py" in source
    assert 'eval "$calibration_exports"' in source
    assert 'SEED_JOINT_CSV="$REPO_ROOT/outputs/ik_waypoints.csv"' in source
    assert "Refusing strict validation with a zero tool TCP" in source
    assert "Refusing planning_mode=seed_joint_waypoints with a non-zero tool TCP" in source
    assert 'tool_tcp_source:="$TOOL_TCP_SOURCE"' in source
    assert 'tool_tcp_measured_by:="$TOOL_TCP_MEASURED_BY"' in source
    assert 'tool_tcp_measured_date:="$TOOL_TCP_MEASURED_DATE"' in source
    assert 'tool_tcp_calibration_method:="$TOOL_TCP_CALIBRATION_METHOD"' in source
    assert 'IK_ROLL_SAMPLE_COUNT="${IK_ROLL_SAMPLE_COUNT:-1}"' in source
    assert 'ik_roll_sample_count:="$IK_ROLL_SAMPLE_COUNT"' in source
    assert 'IK_BEAM_WIDTH="${IK_BEAM_WIDTH:-1}"' in source
    assert 'ik_beam_width:="$IK_BEAM_WIDTH"' in source
    assert 'IK_CANDIDATES_PER_BEAM="${IK_CANDIDATES_PER_BEAM:-8}"' in source
    assert 'ik_candidates_per_beam:="$IK_CANDIDATES_PER_BEAM"' in source
    assert 'IK_BEAM_DIVERSITY_JOINT_DEG="${IK_BEAM_DIVERSITY_JOINT_DEG:-0.0}"' in source
    assert 'ik_beam_diversity_joint_deg:="$IK_BEAM_DIVERSITY_JOINT_DEG"' in source
    assert 'IK_MAX_SOLVE_SECONDS="${IK_MAX_SOLVE_SECONDS:-0.0}"' in source
    assert 'IK_SEARCH_DIAGNOSTICS_CSV="${IK_SEARCH_DIAGNOSTICS_CSV:-$REPO_ROOT/outputs/moveit_ik_search_diagnostics.csv}"' in source
    assert 'IK_FAILURE_SUMMARY_JSON="${IK_FAILURE_SUMMARY_JSON:-$FINAL_OUT_DIR/moveit_ik_failure_summary.json}"' in source
    assert 'IK_SEARCH_DIAGNOSTICS_SUMMARY_JSON="${IK_SEARCH_DIAGNOSTICS_SUMMARY_JSON:-$FINAL_OUT_DIR/moveit_ik_search_diagnostics_summary.json}"' in source
    assert 'ik_search_diagnostics_csv:="$IK_SEARCH_DIAGNOSTICS_CSV"' in source
    assert 'ik_max_solve_seconds:="$IK_MAX_SOLVE_SECONDS"' in source
    assert "summarize_ik_early_exit()" in source
    assert "summarize_moveit_ik_failure.py" in source
    assert "summarize_ik_search_diagnostics.py" in source
    assert source.count("summarize_ik_early_exit") >= 3
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
    assert '--out-dir "$FINAL_OUT_DIR"' in source
    assert "check_production_readiness.py" in source
    assert '--final-quality-csv "$FINAL_OUT_DIR/final_quality_report.csv"' in source
    assert '--out "$PRODUCTION_READINESS_JSON"' in source
    assert 'readiness_rc=$?' in source
    assert "Production readiness check failed" in source


def test_current_goal_state_verifier_runs_local_evidence_chain() -> None:
    verifier = BRIDGE.parents[1] / "scripts" / "verify_current_goal_state.py"
    source = verifier.read_text(encoding="utf-8")

    assert "tools/build_validation_action_plan.py" in source
    assert "tools/check_production_readiness.py" in source
    assert "tools/build_validation_handoff_manifest.py" in source
    assert "tools/build_commit_readiness_plan.py" in source
    assert "tools/build_goal_resolution_audit.py" in source
    assert '"-m", "pytest"' in source
    assert "current_goal_state_verification.json" in source
    assert "allow_failure=True" in source
    assert "--require-production-ready" in source
    assert "required_production_ready=false" in source
    assert 'summary["production_readiness_status"] != "pass"' in source
    assert 'summary["overall_goal_status"] != "complete"' in source


def test_readmes_document_current_goal_state_verifier() -> None:
    root = BRIDGE.parents[1]
    readme = (root / "README.md").read_text(encoding="utf-8-sig")
    project_readme = (root / "PROJECT_REPORT_README.md").read_text(encoding="utf-8-sig")

    for source in (readme, project_readme):
        assert "python scripts/verify_current_goal_state.py" in source
        assert "python scripts/verify_current_goal_state.py --require-production-ready" in source
        assert "production_readiness_status" in source
        assert "overall_goal_status" in source
        assert "pytest_status" in source


def test_plan_only_launch_exposes_ik_roll_sampling() -> None:
    launch_file = BRIDGE.parents[0] / "launch" / "fr5_spray_plan_only.launch.py"
    source = launch_file.read_text(encoding="utf-8")

    assert 'DeclareLaunchArgument("tool_tcp_measured_by", default_value="")' in source
    assert 'DeclareLaunchArgument("tool_tcp_measured_date", default_value="")' in source
    assert 'DeclareLaunchArgument("tool_tcp_calibration_method", default_value="")' in source
    assert '"tool_tcp_measured_by": tool_tcp_measured_by' in source
    assert '"tool_tcp_measured_date": tool_tcp_measured_date' in source
    assert '"tool_tcp_calibration_method": tool_tcp_calibration_method' in source
    assert 'DeclareLaunchArgument("ik_roll_sample_count", default_value="1")' in source
    assert '"ik_roll_sample_count": ik_roll_sample_count' in source
    assert 'DeclareLaunchArgument("ik_beam_width", default_value="1")' in source
    assert '"ik_beam_width": ik_beam_width' in source
    assert 'DeclareLaunchArgument("ik_candidates_per_beam", default_value="8")' in source
    assert '"ik_candidates_per_beam": ik_candidates_per_beam' in source
    assert 'DeclareLaunchArgument("ik_beam_diversity_joint_deg", default_value="0.0")' in source
    assert '"ik_beam_diversity_joint_deg": ik_beam_diversity_joint_deg' in source
    assert 'DeclareLaunchArgument(' in source and '"ik_search_diagnostics_csv"' in source
    assert 'DeclareLaunchArgument("ik_max_solve_seconds", default_value="0.0")' in source
    assert '"ik_search_diagnostics_csv": ik_search_diagnostics_csv' in source
    assert '"ik_max_solve_seconds": ik_max_solve_seconds' in source


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


def test_goal_requirement_audit_accepts_current_runtime_reports() -> None:
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
    runtime_item = next(item for item in items if item.requirement == "ROS2/MoveIt2 runtime execution-chain evidence")
    execution_guard_item = next(
        item for item in items if item.requirement == "Controller execution is blocked unless strict quality gates pass"
    )

    assert failures == []
    assert execution_guard_item.status == "pass"
    assert "before moveit.execute" in execution_guard_item.evidence
    assert runtime_item.status == "pass"
    assert "raw continuous joint gate is replaced by segmented spray-off reorientation evidence" in runtime_item.evidence
    assert "assumed_150mm_placeholder accepted by current validation scope" in runtime_item.evidence
    assert "omits the bottom closure" not in runtime_item.evidence
    assert "diagnostic-only" not in runtime_item.evidence


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
    collision = tmp_path / "moveit_collision_report.csv"
    quality.write_text(
        "\n".join(
            [
                "metric,value",
                "fk_normal_error_max_deg,1.0",
                "fk_standoff_error_max_abs_mm,2.0",
                "fk_path_deviation_max_mm,3.0",
                "fk_tcp_speed_mean_m_s,0.01",
                "fk_tcp_speed_p05_p95_fluctuation,0.01",
                "moveit_ruckig_smoothing_used,true",
                "moveit_ee_link,spray_tcp_link",
                "max_joint_step_deg,1.0",
                "production_joint_step_limit_deg,20.0",
                "joint_continuity_status,pass",
                "tool_tcp_xyz,0.010 0.000 0.185",
                "tool_tcp_rpy,0.000 0.100 0.000",
                "tool_tcp_source,measured_nozzle_tcp",
                "tool_tcp_measured_by,calibration_engineer",
                "tool_tcp_measured_date,2026-07-08",
                "tool_tcp_calibration_method,flange_fixture_probe",
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

    collision.write_text(
        "\n".join(
            [
                "metric,value",
                "collision_environment_object_count,46",
                "collision_checked_state_count,145",
                "collision_count,0",
                "first_collision_index,-1",
                "first_collision_contact_count,0",
                "include_bottom_closure_collision,false",
                "status,pass",
            ]
        ),
        encoding="utf-8",
    )
    status, evidence = module._moveit_runtime_report_status(quality, dynamics, trace, collision)
    assert status == "fail"
    assert "omits the bottom closure" in evidence

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
                "fk_tcp_speed_mean_m_s,0.01",
                "fk_tcp_speed_p05_p95_fluctuation,0.01",
                "moveit_ruckig_smoothing_used,false",
                "moveit_ee_link,spray_tcp_link",
                "max_joint_step_deg,1.0",
                "production_joint_step_limit_deg,20.0",
                "joint_continuity_status,pass",
                "tool_tcp_xyz,0.010 0.000 0.185",
                "tool_tcp_rpy,0.000 0.100 0.000",
                "tool_tcp_source,measured_nozzle_tcp",
                "tool_tcp_measured_by,calibration_engineer",
                "tool_tcp_measured_date,2026-07-08",
                "tool_tcp_calibration_method,flange_fixture_probe",
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
                "fk_tcp_speed_mean_m_s,0.01",
                "fk_tcp_speed_p05_p95_fluctuation,0.01",
                "moveit_ruckig_smoothing_used,true",
                "moveit_ee_link,wrist3_link",
                "max_joint_step_deg,1.0",
                "production_joint_step_limit_deg,20.0",
                "joint_continuity_status,pass",
                "tool_tcp_xyz,0.010 0.000 0.185",
                "tool_tcp_rpy,0.000 0.100 0.000",
                "tool_tcp_source,measured_nozzle_tcp",
                "tool_tcp_measured_by,calibration_engineer",
                "tool_tcp_measured_date,2026-07-08",
                "tool_tcp_calibration_method,flange_fixture_probe",
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
                "fk_tcp_speed_mean_m_s,0.01",
                "fk_tcp_speed_p05_p95_fluctuation,0.01",
                "moveit_ruckig_smoothing_used,true",
                "moveit_ee_link,spray_tcp_link",
                "max_joint_step_deg,1.0",
                "production_joint_step_limit_deg,20.0",
                "joint_continuity_status,pass",
                "tool_tcp_xyz,0.010 0.000 0.185",
                "tool_tcp_rpy,0.000 0.100 0.000",
                "tool_tcp_source,measured_nozzle_tcp",
                "tool_tcp_measured_by,calibration_engineer",
                "tool_tcp_measured_date,2026-07-08",
                "tool_tcp_calibration_method,flange_fixture_probe",
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
    trajectory = tmp_path / "moveit_smoothed_joint_trajectory.csv"
    waypoint_trajectory = tmp_path / "moveit_waypoint_joint_trajectory.csv"
    tcp_pose_csv = tmp_path / "tcp_poses_base_link.csv"
    joint_step_report = tmp_path / "moveit_joint_step_report.csv"
    waypoint_joint_step_report = tmp_path / "moveit_waypoint_joint_step_report.csv"
    ik_segments_summary = tmp_path / "ik_continuity_segments_summary.json"
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
                "max_joint_step_deg,1.0",
                "max_joint_step_from_index,0",
                "max_joint_step_to_index,1",
                "max_joint_step_joint,j1",
                "max_joint_step_from_deg,0.0",
                "max_joint_step_to_deg,1.0",
                "max_ik_joint_step_deg,20.0",
                "production_joint_step_limit_deg,20.0",
                "joint_continuity_status,pass",
                "tool_tcp_xyz,0.000 0.000 0.150",
                "tool_tcp_rpy,0 0 0",
                "tool_tcp_source,assumed_150mm_placeholder",
                "tool_tcp_measured_by,",
                "tool_tcp_measured_date,",
                "tool_tcp_calibration_method,",
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
                "first_collision_index,-1",
                "first_collision_contact_count,0",
                "include_bottom_closure_collision,true",
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
    trajectory.write_text(
        "\n".join(
            [
                "t,j1_q,j2_q,j3_q,j4_q,j5_q,j6_q",
                "0.0,0.0,0.0,0.0,0.0,0.0,0.0",
                "1.0,0.1,0.2,0.3,0.4,0.5,0.6",
                "2.0,6.333185307179586,0.2,0.3,0.4,0.5,0.6523598775598299",
            ]
        ),
        encoding="utf-8",
    )
    waypoint_trajectory.write_text(
        "\n".join(
            [
                "waypoint,j1_q,j2_q,j3_q,j4_q,j5_q,j6_q",
                "0,0.0,0.0,0.0,0.0,0.0,0.0",
                "1,0.1,0.2,0.3,0.4,0.5,0.6",
                "2,0.2,0.3,0.4,2.0,0.5,0.7",
            ]
        ),
        encoding="utf-8",
    )
    tcp_pose_csv.write_text(
        "\n".join(
            [
                "x,y,z,qx,qy,qz,qw,nx,ny,nz",
                "0.0,-0.3,0.06,0,0,1,0,0,0,1",
                "0.1,-0.3,0.07,0,0,1,0,1,0,0",
                "0.2,-0.3,0.08,0,0,1,0,0,0,-1",
            ]
        ),
        encoding="utf-8",
    )
    joint_step_summary = module.write_joint_step_report(trajectory, joint_step_report, 20.0)
    waypoint_joint_step_summary = module.write_joint_step_report(
        waypoint_trajectory,
        waypoint_joint_step_report,
        20.0,
    )
    module.write_ik_continuity_segments_summary(
        joint_step_report,
        ik_segments_summary,
        loop_size=240,
        tcp_pose_csv=tcp_pose_csv,
    )
    segmented_summary_json = tmp_path / "segmented_process_summary.json"
    segmented_plan_csv = tmp_path / "segmented_process_plan.csv"
    smooth_transition_csv = tmp_path / "smooth_reorientation_transitions.csv"
    segmented_summary = module.write_segmented_process_plan(
        trajectory,
        joint_step_report,
        segmented_summary_json,
        segmented_plan_csv,
        smooth_transition_csv,
        20.0,
        collision_report=collision,
        dynamics_report=dynamics,
    )

    assert joint_step_summary["joint_step_exceeding_count"] == 3
    assert joint_step_summary["joint_step_exceeding_transition_count"] == 1
    assert joint_step_summary["joint_step_exceeding_joints"] == "j4 j5 j6"
    assert joint_step_summary["max_joint_step_deg_from_report"] > 30.0
    assert segmented_summary["overall_status"] == "pass"
    assert segmented_summary["process_joint_continuity_status"] == "pass"
    assert segmented_summary["process_max_joint_step_deg"] < 20.0
    assert segmented_summary["reorientation_transition_count"] == 1
    assert segmented_summary["reorientation_transition_status"] == "pass"
    assert segmented_summary["reorientation_transition_max_interpolated_step_deg"] <= 5.0
    assert segmented_summary["process_collision_status"] == "pass"
    assert segmented_summary["process_dynamics_status"] == "pass"
    assert segmented_summary["reorientation_transition_collision_status"] == "pass"
    assert segmented_summary["reorientation_transition_dynamics_status"] == "pass"
    assert segmented_summary["stop_boundary_zero_velocity_acceleration_status"] == "pass"
    assert segmented_summary["next_segment_entry_status"] == "pass"
    assert segmented_summary["no_gap_or_overlap_status"] == "pass"

    module.write_final_quality_report(
        quality,
        dynamics,
        collision,
        audit,
        out_csv,
        out_json,
        runtime_log,
        joint_step_report,
        joint_step_summary,
        ik_segments_summary,
        waypoint_joint_step_report,
        waypoint_joint_step_summary,
        segmented_summary,
        segmented_summary_json,
    )
    final_quality = {row["metric"]: row["value"] for row in csv.DictReader(out_csv.open(encoding="utf-8"))}
    summary = json.loads(out_json.read_text(encoding="utf-8"))
    segment_summary = json.loads(ik_segments_summary.read_text(encoding="utf-8"))
    step_rows = list(csv.DictReader(joint_step_report.open(encoding="utf-8")))

    assert final_quality["result_source"] == "moveit2_strict_runtime"
    assert final_quality["status"] == "pass"
    assert final_quality["runtime_log_source"] == str(runtime_log)
    assert final_quality["ruckig_known_warning_count_raw_log"] == "1"
    assert final_quality["production_min_tcp_speed_m_s"] == "0.003"
    assert final_quality["tcp_speed_production_status"] == "pass"
    assert final_quality["tool_tcp_xyz"] == "0.000 0.000 0.150"
    assert final_quality["tool_tcp_source"] == "assumed_150mm_placeholder"
    assert final_quality["tool_tcp_measured_by"] == ""
    assert final_quality["tool_tcp_measured_date"] == ""
    assert final_quality["tool_tcp_calibration_method"] == ""
    assert final_quality["tool_tcp_production_status"] == "fail"
    assert final_quality["max_ik_joint_step_deg"] == "20.0"
    assert final_quality["raw_joint_continuity_status"] == "pass"
    assert final_quality["joint_continuity_status"] == "pass"
    assert final_quality["trajectory_execution_mode"] == "segmented_process_with_smooth_reorientation_stops"
    assert final_quality["process_joint_continuity_status"] == "pass"
    assert float(final_quality["process_max_joint_step_deg"]) < 20.0
    assert final_quality["reorientation_transition_count"] == "1"
    assert final_quality["reorientation_transition_status"] == "pass"
    assert final_quality["reorientation_transition_collision_status"] == "pass"
    assert final_quality["reorientation_transition_dynamics_status"] == "pass"
    assert final_quality["stop_boundary_zero_velocity_acceleration_status"] == "pass"
    assert final_quality["next_segment_entry_status"] == "pass"
    assert final_quality["no_gap_or_overlap_status"] == "pass"
    assert final_quality["spray_off_transition_status"] == "pass"
    assert final_quality["segmented_process_summary_source"] == str(segmented_summary_json)
    assert final_quality["segmented_process_plan_source"] == str(segmented_plan_csv)
    assert final_quality["smooth_transition_trajectory_source"] == str(smooth_transition_csv)
    assert final_quality["joint_step_report_source"] == str(joint_step_report)
    assert final_quality["joint_step_exceeding_count"] == "3"
    assert final_quality["joint_step_exceeding_transition_count"] == "1"
    assert final_quality["joint_step_exceeding_joints"] == "j4 j5 j6"
    assert float(final_quality["max_joint_step_raw_deg_from_report"]) > 350.0
    assert final_quality["ik_continuity_segments_summary_source"] == str(ik_segments_summary)
    assert final_quality["waypoint_joint_step_report_source"] == str(waypoint_joint_step_report)
    assert float(final_quality["waypoint_max_joint_step_deg_from_report"]) > 90.0
    assert final_quality["waypoint_joint_step_exceeding_count"] == "4"
    assert final_quality["waypoint_post_joint_continuity_match_status"] == "fail"
    assert "joint_step_exceeding_count" in final_quality["waypoint_post_joint_continuity_match_evidence"]
    assert final_quality["max_joint_step_deg"] == "1.0"
    assert final_quality["max_joint_step_joint"] == "j1"
    assert final_quality["include_bottom_closure_collision"] == "true"
    assert final_quality["max_jerk_ratio"] == "0.4"
    assert final_quality["collision_count"] == "0"
    assert final_quality["first_collision_index"] == "-1"
    assert final_quality["first_collision_contact_count"] == "0"
    assert summary["overall_status"] == "pass"
    assert step_rows[0]["joint"] == "j6"
    assert "raw_step_deg" in step_rows[0]
    wrap_row = next(row for row in step_rows if row["from_index"] == "1" and row["joint"] == "j1")
    assert float(wrap_row["raw_step_deg"]) > 350.0
    assert float(wrap_row["step_deg"]) < 4.0
    assert wrap_row["exceeds_limit"] == "false"
    assert segment_summary["exceeding_transition_count"] == 1
    assert segment_summary["segments"][0]["exceeding_joints"] == ["j4", "j5", "j6"]
    assert segment_summary["segments"][0]["geometry_zone"] == "bottom_closure"
    assert segment_summary["segments"][0]["tcp_x"] == 0.0
    assert segment_summary["geometry_zones"] == ["bottom_closure"]
    assert segment_summary["tcp_pose_source"] == str(tcp_pose_csv)
    assert segment_summary["phase_clusters"][0]["phase_index"] == 0
    assert segment_summary["phase_clusters"][0]["occurrence_count"] == 1
    assert segment_summary["phase_clusters"][0]["geometry_zone"] == "bottom_closure"
    smooth_rows = list(csv.DictReader(smooth_transition_csv.open(encoding="utf-8")))
    assert smooth_rows[0]["spray_enabled"] == "false"
    assert smooth_rows[0]["normalized_time"] == "0.0"
    assert smooth_rows[-1]["normalized_time"] == "1.0"
    assert all(float(smooth_rows[0][f"j{i}_dq"]) == 0.0 for i in range(1, 7))
    assert all(float(smooth_rows[-1][f"j{i}_dq"]) == 0.0 for i in range(1, 7))
    assert all(float(smooth_rows[0][f"j{i}_ddq"]) == 0.0 for i in range(1, 7))
    assert all(float(smooth_rows[-1][f"j{i}_ddq"]) == 0.0 for i in range(1, 7))


def test_validation_handoff_manifest_groups_changes_and_gates(tmp_path) -> None:
    import importlib.util
    import sys

    manifest_path = BRIDGE.parents[1] / "tools" / "build_validation_handoff_manifest.py"
    spec = importlib.util.spec_from_file_location("build_validation_handoff_manifest", manifest_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    quality = tmp_path / "final_quality_report.csv"
    audit = tmp_path / "audit_goal_requirements_strict.json"
    quality.write_text(
        "\n".join(
            [
                "metric,value",
                "status,fail",
                "collision_status,pass",
                "tcp_speed_production_status,pass",
                "tool_tcp_production_status,fail",
                "joint_continuity_status,fail",
                "waypoint_post_joint_continuity_match_status,pass",
                "tool_tcp_xyz,0.000 0.000 0.150",
                "tool_tcp_source,assumed_150mm_placeholder",
                "fk_tcp_speed_mean_m_s,0.003095773",
                "max_joint_step_deg,145.8486084788797",
                "max_ik_joint_step_deg,180.0",
                "production_joint_step_limit_deg,20.0",
                "joint_step_exceeding_count,48",
                "collision_count,0",
                "first_collision_index,-1",
                "include_bottom_closure_collision,true",
            ]
        ),
        encoding="utf-8",
    )
    audit.write_text('{"overall_status":"fail","items":[]}', encoding="utf-8")
    module.git_status = lambda root: [
        {"status": "M", "path": "ros2_moveit_bridge/plan_closed_contour_moveit.py"},
        {"status": "M", "path": "tests/test_moveit_bridge_static.py"},
        {"status": "??", "path": "outputs/goal_resolution_audit.json"},
        {"status": "??", "path": "outputs/ik_beam_probe_summary.json"},
        {"status": "??", "path": "outputs/ik_root_cause_matrix.json"},
        {"status": "??", "path": "outputs/phase110_wide_open_nocollision_ik_probe_summary.json"},
        {"status": "??", "path": "outputs/phase110_wide_open_window_ik_probe_summary.json"},
        {"status": "??", "path": "outputs/phase232_wide_open_nocollision_ik_probe_summary.json"},
        {"status": "??", "path": "outputs/phase232_wide_open_window_ik_probe_summary.json"},
        {"status": "M", "path": "src/metrics.py"},
    ]

    manifest = module.build_manifest(tmp_path, quality, audit)

    assert manifest["overall_status"] == "fail"
    assert manifest["strict_audit_status"] == "fail"
    assert manifest["current_gates"]["collision_status"] == "pass"
    assert manifest["current_gates"]["tool_tcp_production_status"] == "fail"
    assert manifest["current_gates"]["joint_continuity_status"] == "fail"
    assert manifest["key_metrics"]["tool_tcp_source"] == "assumed_150mm_placeholder"
    assert manifest["key_metrics"]["max_ik_joint_step_deg"] == "180.0"
    assert manifest["key_metrics"]["production_joint_step_limit_deg"] == "20.0"
    assert manifest["changed_file_count"] == 10
    assert manifest["changed_files"]["runtime_bridge_and_strict_gates"][0]["path"].endswith(
        "plan_closed_contour_moveit.py"
    )
    assert manifest["changed_files"]["tests"][0]["path"] == "tests/test_moveit_bridge_static.py"
    assert [entry["path"] for entry in manifest["changed_files"]["probe_evidence"]] == [
        "outputs/goal_resolution_audit.json",
        "outputs/ik_beam_probe_summary.json",
        "outputs/ik_root_cause_matrix.json",
        "outputs/phase110_wide_open_nocollision_ik_probe_summary.json",
        "outputs/phase110_wide_open_window_ik_probe_summary.json",
        "outputs/phase232_wide_open_nocollision_ik_probe_summary.json",
        "outputs/phase232_wide_open_window_ik_probe_summary.json",
    ]
    assert manifest["changed_files"]["review_separately_preexisting_or_mixed"][0]["path"] == "src/metrics.py"
    assert {item["id"] for item in manifest["blocking_items"]} == {
        "final_quality_not_pass",
        "strict_audit_not_pass",
        "measured_tool_tcp_required",
        "joint_continuity_required",
    }


def test_commit_readiness_plan_keeps_mixed_files_out_of_primary_commit(tmp_path) -> None:
    import importlib.util
    import json

    module_path = BRIDGE.parents[1] / "tools" / "build_commit_readiness_plan.py"
    spec = importlib.util.spec_from_file_location("build_commit_readiness_plan", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    manifest = tmp_path / "validation_handoff_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "overall_status": "fail",
                "strict_audit_status": "fail",
                "blocking_items": [{"id": "measured_tool_tcp_required"}],
                "changed_files": {
                    "runtime_bridge_and_strict_gates": [
                        {"status": "M", "path": "ros2_moveit_bridge/plan_closed_contour_moveit.py"}
                    ],
                    "tests": [{"status": "M", "path": "tests/test_moveit_bridge_static.py"}],
                    "formal_evidence": [{"status": "??", "path": "outputs/production_readiness_check.json"}],
                    "probe_evidence": [{"status": "??", "path": "outputs/phase110_window_ik_probe_summary.json"}],
                    "review_separately_preexisting_or_mixed": [
                        {"status": "M", "path": "src/metrics.py"},
                        {"status": "M", "path": "outputs/quality_report.csv"},
                    ],
                },
            }
        ),
        encoding="utf-8",
    )

    plan = module.build_commit_plan(manifest)
    primary_paths = {entry["path"] for entry in plan["primary_commit"]["files"]}
    review_paths = {entry["path"] for entry in plan["review_separately"]["files"]}
    optional_paths = {entry["path"] for entry in plan["local_or_optional_evidence"]["files"]}

    assert plan["overall_status"] == "fail"
    assert "ros2_moveit_bridge/plan_closed_contour_moveit.py" in primary_paths
    assert "outputs/production_readiness_check.json" in primary_paths
    assert "src/metrics.py" not in primary_paths
    assert review_paths == {"src/metrics.py", "outputs/quality_report.csv"}
    assert optional_paths == {"outputs/phase110_window_ik_probe_summary.json"}
    assert plan["unclassified"]["file_count"] == 0
    primary_stage = "\n".join(plan["primary_commit"]["stage_commands"])
    review_stage = "\n".join(plan["review_separately"]["stage_commands"])
    assert '"ros2_moveit_bridge/plan_closed_contour_moveit.py"' in primary_stage
    assert '"outputs/production_readiness_check.json"' in primary_stage
    assert "src/metrics.py" not in primary_stage
    assert '"src/metrics.py"' in review_stage
    assert plan["primary_commit"]["commit_message_suggestion"]
    assert "Do not include in the primary commit" in plan["review_separately"]["stage_policy"]


def test_summarize_moveit_ik_failure_extracts_indices_and_contacts(tmp_path) -> None:
    import importlib.util

    module_path = BRIDGE.parents[1] / "tools" / "summarize_moveit_ik_failure.py"
    spec = importlib.util.spec_from_file_location("summarize_moveit_ik_failure", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    log = tmp_path / "moveit_runtime.log"
    log.write_text(
        "\n".join(
            [
                "[INFO] Found a contact between 'horseshoe_wall_058' (type 'Object') and 'upperarm_link' (type 'Robot link'), which constitutes a collision.",
                "RuntimeError: MoveIt2 IK failed the closed contour gate: "
                "index=6, tcp=(0,0,0), roll=0.0deg,seed=0:joint=j5,joint_step=179.93deg; "
                "index=131, all IK candidates are colliding, best=contacts=2,pairs=unavailable",
            ]
        ),
        encoding="utf-8",
    )

    summary = module.summarize_log(log)

    assert summary["runtime_error_found"] is True
    assert summary["first_failure_index"] == 6
    assert summary["failure_preview_count"] == 2
    assert summary["all_colliding_failure_count"] == 1
    assert summary["max_reported_joint_step_deg"] == 179.93
    assert summary["reported_joints"] == "j5"
    assert summary["contact_pairs"] == [{"pair": "horseshoe_wall_058 <-> upperarm_link", "count": 1}]


def test_summarize_ik_search_diagnostics_reports_timeout_and_search_cost(tmp_path) -> None:
    import importlib.util

    module_path = BRIDGE.parents[1] / "tools" / "summarize_ik_search_diagnostics.py"
    spec = importlib.util.spec_from_file_location("summarize_ik_search_diagnostics", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    diagnostics = tmp_path / "moveit_ik_search_diagnostics.csv"
    diagnostics.write_text(
        "\n".join(
            [
                "index,elapsed_s,input_beam_count,candidate_failure_count,colliding_candidate_count,next_beam_count,best_bottleneck_step_deg,best_step_sum_deg,best_roll_sum_deg,status,detail",
                "0,0.1,1,0,0,4,0.0,0.0,0.0,ok,",
                "1,7.0,4,925,0,4,2.9,2.9,144.0,ok,",
                "2,20.5,4,0,0,0,,,,timeout,ik_max_solve_seconds=20.0",
            ]
        ),
        encoding="utf-8",
    )

    summary = module.summarize_diagnostics(diagnostics)

    assert summary["status"] == "fail"
    assert summary["row_count"] == 3
    assert summary["completed_ok_count"] == 2
    assert summary["last_index"] == 2
    assert summary["last_status"] == "timeout"
    assert summary["max_candidate_failure_count"] == 925
    assert summary["best_bottleneck_step_deg_last_ok"] == 2.9
    assert summary["slowest_index"] == 2
    assert summary["failure_rows"] == [
        {
            "index": 2,
            "status": "timeout",
            "detail": "ik_max_solve_seconds=20.0",
            "elapsed_s": 20.5,
        }
    ]


def test_slice_tcp_pose_window_writes_closed_local_probe_csv(tmp_path) -> None:
    import csv
    import importlib.util

    module_path = BRIDGE.parents[1] / "tools" / "slice_tcp_pose_window.py"
    spec = importlib.util.spec_from_file_location("slice_tcp_pose_window", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    source = tmp_path / "tcp.csv"
    source.write_text(
        "\n".join(
            [
                "x,y,z,qx,qy,qz,qw,nx,ny,nz",
                "0,0,0,0,0,0,1,0,0,1",
                "1,0,0,0,0,0,1,0,0,1",
                "2,0,0,0,0,0,1,0,0,1",
                "3,0,0,0,0,0,1,0,0,1",
                "4,0,0,0,0,0,1,0,0,1",
            ]
        ),
        encoding="utf-8",
    )
    out_csv = tmp_path / "window.csv"
    out_json = tmp_path / "window.json"

    payload = module.slice_window(source, out_csv, center_phase=0, before=1, after=1, loop_size=5, out_json=out_json)
    rows = list(csv.DictReader(out_csv.open(encoding="utf-8")))

    assert payload["phase_indices"] == [4, 0, 1]
    assert payload["samples_per_loop_for_validation"] == 3
    assert payload["row_count"] == 4
    assert rows[0]["source_phase_index"] == "4"
    assert rows[-1]["is_closure_duplicate"] == "true"
    assert rows[-1]["x"] == rows[0]["x"]
    assert "formal full-contour" in out_json.read_text(encoding="utf-8")
    assert "artifact" in payload["interpretation"]


def test_slice_tcp_pose_window_open_mode_omits_artificial_closure(tmp_path) -> None:
    import csv
    import importlib.util

    module_path = BRIDGE.parents[1] / "tools" / "slice_tcp_pose_window.py"
    spec = importlib.util.spec_from_file_location("slice_tcp_pose_window", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    source = tmp_path / "tcp.csv"
    source.write_text(
        "\n".join(
            [
                "x,y,z,qx,qy,qz,qw,nx,ny,nz",
                "0,0,0,0,0,0,1,0,0,1",
                "1,0,0,0,0,0,1,0,0,1",
                "2,0,0,0,0,0,1,0,0,1",
                "3,0,0,0,0,0,1,0,0,1",
                "4,0,0,0,0,0,1,0,0,1",
            ]
        ),
        encoding="utf-8",
    )
    out_csv = tmp_path / "window.csv"

    payload = module.slice_window(
        source,
        out_csv,
        center_phase=0,
        before=1,
        after=1,
        loop_size=5,
        close_window=False,
    )
    rows = list(csv.DictReader(out_csv.open(encoding="utf-8")))

    assert payload["phase_indices"] == [4, 0, 1]
    assert payload["samples_per_loop_for_validation"] == 0
    assert payload["row_count"] == 3
    assert all(row["is_closure_duplicate"] == "false" for row in rows)
    assert "does not add an artificial closure transition" in payload["interpretation"]


def test_analyze_ik_branch_splice_reports_unspliceable_local_branch(tmp_path) -> None:
    import importlib.util

    module_path = BRIDGE.parents[1] / "tools" / "analyze_ik_branch_splice.py"
    spec = importlib.util.spec_from_file_location("analyze_ik_branch_splice", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    header = "waypoint,j1_q,j2_q,j3_q,j4_q,j5_q,j6_q"
    full = tmp_path / "full.csv"
    local = tmp_path / "local.csv"
    full.write_text(
        "\n".join(
            [
                header,
                "0,0,0,0,0,0,0",
                "1,0.01,0,0,0,0,0",
                "2,0.02,0,0,0,0,0",
                "3,0.03,0,0,0,0,0",
                "4,0.04,0,0,0,0,0",
            ]
        ),
        encoding="utf-8",
    )
    local.write_text(
        "\n".join(
            [
                header,
                "0,1.5,0,0,0,0,0",
                "1,1.51,0,0,0,0,0",
                "2,1.52,0,0,0,0,0",
            ]
        ),
        encoding="utf-8",
    )

    summary = module.analyze_splice(full, local, local_source_start_index=1, limit_deg=20.0)

    assert summary["local_source_start_index"] == 1
    assert summary["local_source_end_index"] == 3
    assert summary["best_splice_trial"]["status"] == "fail"
    assert summary["best_splice_trial"]["max_step_deg"] > 20.0
    assert "diagnostic island" in summary["interpretation"]


def test_build_ik_root_cause_matrix_separates_passed_and_blocking_items(tmp_path) -> None:
    import importlib.util
    import json

    module_path = BRIDGE.parents[1] / "tools" / "build_ik_root_cause_matrix.py"
    spec = importlib.util.spec_from_file_location("build_ik_root_cause_matrix", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    outputs = tmp_path / "outputs"
    outputs.mkdir()
    (outputs / "final_quality_report.csv").write_text(
        "\n".join(
            [
                "metric,value",
                "status,fail",
                "collision_status,pass",
                "collision_count,0",
                "first_collision_index,-1",
                "include_bottom_closure_collision,true",
                "tool_tcp_production_status,fail",
                "tool_tcp_source,assumed_150mm_placeholder",
                "tool_tcp_xyz,0.000 0.000 0.150",
                "tcp_speed_production_status,pass",
                "fk_tcp_speed_mean_m_s,0.0031",
                "production_min_tcp_speed_m_s,0.003",
            ]
        ),
        encoding="utf-8",
    )
    (outputs / "phase110_wide_open_window_ik_probe_summary.json").write_text(
        json.dumps(
            {
                "status": "fail:all_candidates_colliding",
                "first_failure_source_phase": 90,
                "contact_pairs": [{"pair": "wall <-> wrist2_link"}],
            }
        ),
        encoding="utf-8",
    )
    (outputs / "phase110_wide_open_nocollision_ik_probe_summary.json").write_text(
        json.dumps(
            {
                "ik_search_status": "pass",
                "max_joint_step_deg": 2.5,
                "moveit_quality_status": "fail:stand-off",
                "fk_standoff_error_max_abs_mm": 200.0,
            }
        ),
        encoding="utf-8",
    )
    (outputs / "phase110_open_window_splice_analysis.json").write_text(
        json.dumps({"best_splice_trial": {"status": "fail", "max_step_deg": 175.0}}),
        encoding="utf-8",
    )
    (outputs / "phase232_wide_open_window_ik_probe_summary.json").write_text(
        json.dumps(
            {
                "status": "fail:all_candidates_colliding",
                "first_failure_source_phase": 214,
                "contact_pairs": [{"pair": "wall <-> wrist2_link"}],
            }
        ),
        encoding="utf-8",
    )
    (outputs / "phase232_wide_open_nocollision_ik_probe_summary.json").write_text(
        json.dumps(
            {
                "ik_search_status": "fail:timeout",
                "completed_ok_count": 43,
                "timeout_before_local_index": 43,
                "best_bottleneck_step_deg_max_ok": 3.1,
            }
        ),
        encoding="utf-8",
    )

    matrix = module.build_matrix(tmp_path)
    by_id = {item["id"]: item for item in matrix["items"]}

    assert matrix["overall_status"] == "fail"
    assert by_id["formal_collision_gate"]["status"] == "pass"
    assert by_id["production_speed_gate"]["status"] == "pass"
    assert by_id["tool_tcp_gate"]["status"] == "fail"
    assert by_id["phase110_branch_patch"]["status"] == "blocked_by_collision_and_fk_quality"
    assert by_id["phase232_bottom_closure_branch"]["status"] == "blocked_by_collision_then_search_cost"
    assert set(matrix["blocking_items"]) == {
        "tool_tcp_gate",
        "phase110_branch_patch",
        "phase232_bottom_closure_branch",
    }


def test_build_validation_action_plan_prioritizes_measured_tcp_before_more_search(tmp_path) -> None:
    import importlib.util
    import json

    module_path = BRIDGE.parents[1] / "tools" / "build_validation_action_plan.py"
    spec = importlib.util.spec_from_file_location("build_validation_action_plan", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    outputs = tmp_path / "outputs"
    outputs.mkdir()
    (outputs / "final_quality_report.csv").write_text(
        "\n".join(
            [
                "metric,value",
                "status,fail",
                "strict_audit_status,fail",
                "collision_status,pass",
                "tcp_speed_production_status,pass",
                "tool_tcp_production_status,fail",
                "joint_continuity_status,fail",
                "tool_tcp_source,assumed_150mm_placeholder",
                "tool_tcp_xyz,0.000 0.000 0.150",
            ]
        ),
        encoding="utf-8",
    )
    (outputs / "ik_root_cause_matrix.json").write_text(
        json.dumps(
            {
                "blocking_items": ["tool_tcp_gate", "phase110_branch_patch", "phase232_bottom_closure_branch"],
                "items": [
                    {
                        "id": "phase110_branch_patch",
                        "evidence": {"collision_on_status": "fail:all_candidates_colliding"},
                    },
                    {
                        "id": "phase232_bottom_closure_branch",
                        "evidence": {"collision_on_status": "fail:all_candidates_colliding"},
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    (outputs / "goal_resolution_audit.json").write_text(
        json.dumps({"blocked_items": ["placeholder_tcp_not_production", "default_20deg_ik_continuity"]}),
        encoding="utf-8",
    )

    plan = module.build_action_plan(tmp_path)
    actions = {action["id"]: action for action in plan["actions"]}

    assert plan["overall_status"] == "fail"
    assert plan["current_passed_gates"] == ["collision_status", "tcp_speed_production_status"]
    assert plan["current_blocking_gates"] == ["tool_tcp_production_status", "joint_continuity_status"]
    assert plan["actions"][0]["id"] == "load_measured_nozzle_tcp"
    assert actions["load_measured_nozzle_tcp"]["status"] == "waiting_for_external_measurement"
    assert actions["rerun_full_strict_validation_after_tcp"]["status"] == "blocked_by_measured_tcp"
    assert actions["monitor_segmented_reorientation_stops"]["status"] == "requires_segmented_process_fix"
    assert (
        actions["avoid_unbounded_bruteforce_ik"]["priority"]
        > actions["monitor_segmented_reorientation_stops"]["priority"]
    )


def test_check_production_readiness_requires_all_strict_gates(tmp_path) -> None:
    import importlib.util
    import json

    module_path = BRIDGE.parents[1] / "tools" / "check_production_readiness.py"
    spec = importlib.util.spec_from_file_location("check_production_readiness", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    final_quality = tmp_path / "final_quality_report.csv"
    strict_audit = tmp_path / "audit_goal_requirements_strict.json"
    collision = tmp_path / "moveit_collision_report.csv"
    final_quality.write_text(
        "\n".join(
            [
                "metric,value",
                "status,fail",
                "collision_status,pass",
                "tcp_speed_production_status,pass",
                "tool_tcp_production_status,fail",
                "joint_continuity_status,fail",
                "waypoint_post_joint_continuity_match_status,pass",
                "tool_tcp_source,assumed_150mm_placeholder",
                "tool_tcp_xyz,0.000 0.000 0.150",
                "fk_tcp_speed_mean_m_s,0.0031",
                "production_min_tcp_speed_m_s,0.003",
                "max_joint_step_deg,145.0",
                "production_joint_step_limit_deg,20.0",
            ]
        ),
        encoding="utf-8",
    )
    strict_audit.write_text(json.dumps({"overall_status": "fail"}), encoding="utf-8")
    collision.write_text(
        "\n".join(
            [
                "metric,value",
                "collision_count,0",
                "first_collision_index,-1",
                "status,pass",
            ]
        ),
        encoding="utf-8",
    )

    failed = module.build_readiness_report(final_quality, strict_audit, collision)
    assert failed["overall_status"] == "fail"
    blocking = {gate["gate"] for gate in failed["blocking_gates"]}
    assert "final_quality.tool_tcp_production_status" in blocking
    assert "final_quality.joint_continuity_status" in blocking
    assert "derived.tool_tcp_source_is_production" in blocking
    assert "derived.tool_tcp_metadata_present" in blocking
    assert "derived.max_joint_step_within_limit" in blocking
    missing = {item["gate"]: item for item in failed["missing_evidence"]}
    assert "measured TCP YAML" in missing["final_quality.tool_tcp_production_status"]["next_action"]
    assert "measured TCP metadata" in missing["derived.tool_tcp_metadata_present"]["required_evidence"]
    assert "process_max_joint_step_deg" in missing["derived.max_joint_step_within_limit"]["required_evidence"]

    final_quality.write_text(
        "\n".join(
            [
                "metric,value",
                "status,fail",
                "collision_status,pass",
                "tcp_speed_production_status,pass",
                "tool_tcp_production_status,fail",
                "joint_continuity_status,pass",
                "waypoint_post_joint_continuity_match_status,pass",
                "tool_tcp_source,assumed_150mm_placeholder",
                "tool_tcp_xyz,0.000 0.000 0.150",
                "fk_tcp_speed_mean_m_s,0.0031",
                "production_min_tcp_speed_m_s,0.003",
                "max_joint_step_deg,145.0",
                "process_max_joint_step_deg,3.6",
                    "trajectory_execution_mode,segmented_process_with_smooth_reorientation_stops",
                    "process_joint_continuity_status,pass",
                    "process_collision_status,pass",
                    "process_dynamics_status,pass",
                    "reorientation_transition_status,pass",
                    "reorientation_transition_collision_status,pass",
                    "reorientation_transition_dynamics_status,pass",
                    "stop_boundary_zero_velocity_acceleration_status,pass",
                    "next_segment_entry_status,pass",
                    "no_gap_or_overlap_status,pass",
                    "spray_off_transition_status,pass",
                "production_joint_step_limit_deg,20.0",
            ]
        ),
        encoding="utf-8",
    )
    segmented = module.build_readiness_report(final_quality, strict_audit, collision)
    segmented_blocking = {gate["gate"] for gate in segmented["blocking_gates"]}
    assert "final_quality.joint_continuity_status" not in segmented_blocking
    assert "derived.max_joint_step_within_limit" not in segmented_blocking

    final_quality.write_text(
        "\n".join(
            [
                "metric,value",
                "status,pass",
                "collision_status,pass",
                "tcp_speed_production_status,pass",
                "tool_tcp_production_status,pass",
                "joint_continuity_status,pass",
                "waypoint_post_joint_continuity_match_status,pass",
                "tool_tcp_source,measured_nozzle_tcp_2026_07_08",
                "tool_tcp_xyz,0.010 0.000 0.185",
                "tool_tcp_measured_by,calibration_engineer",
                "tool_tcp_measured_date,2026-07-08",
                "tool_tcp_calibration_method,flange_fixture_probe",
                "fk_tcp_speed_mean_m_s,0.0031",
                "production_min_tcp_speed_m_s,0.003",
                "max_joint_step_deg,12.0",
                "production_joint_step_limit_deg,20.0",
            ]
        ),
        encoding="utf-8",
    )
    strict_audit.write_text(json.dumps({"overall_status": "pass"}), encoding="utf-8")

    passed = module.build_readiness_report(final_quality, strict_audit, collision)
    assert passed["overall_status"] == "pass"
    assert passed["blocking_gates"] == []
    assert passed["missing_evidence"] == []
    gate_statuses = {gate["gate"]: gate["status"] for gate in passed["gates"]}
    assert gate_statuses["derived.tool_tcp_source_is_production"] == "pass"
    assert gate_statuses["derived.tool_tcp_metadata_present"] == "pass"
    assert gate_statuses["derived.tcp_speed_meets_minimum"] == "pass"
    assert gate_statuses["derived.max_joint_step_within_limit"] == "pass"
    assert gate_statuses["derived.collision_count_zero"] == "pass"


def test_build_goal_resolution_audit_keeps_goal_active_for_tcp_and_ik_blockers(tmp_path) -> None:
    import importlib.util
    import json

    module_path = BRIDGE.parents[1] / "tools" / "build_goal_resolution_audit.py"
    spec = importlib.util.spec_from_file_location("build_goal_resolution_audit", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    outputs = tmp_path / "outputs"
    outputs.mkdir()
    (outputs / "final_quality_report.csv").write_text(
        "\n".join(
            [
                "metric,value",
                "status,fail",
                "collision_status,pass",
                "collision_count,0",
                "first_collision_index,-1",
                "tool_tcp_production_status,fail",
                "tool_tcp_source,assumed_150mm_placeholder",
                "tool_tcp_xyz,0.000 0.000 0.150",
                "tcp_speed_production_status,pass",
                "fk_tcp_speed_mean_m_s,0.0031",
                "production_min_tcp_speed_m_s,0.003",
                "joint_continuity_status,fail",
                "max_joint_step_deg,145.0",
                "production_joint_step_limit_deg,20.0",
            ]
        ),
        encoding="utf-8",
    )
    (outputs / "audit_goal_requirements_strict.json").write_text(
        json.dumps({"overall_status": "fail"}),
        encoding="utf-8",
    )
    (outputs / "moveit_collision_report.csv").write_text(
        "\n".join(
            [
                "metric,value",
                "collision_checked_state_count,145",
                "collision_count,0",
                "first_collision_index,-1",
                "status,pass",
            ]
        ),
        encoding="utf-8",
    )
    (outputs / "ik_root_cause_matrix.json").write_text(
        json.dumps({"blocking_items": ["tool_tcp_gate", "phase110_branch_patch"]}),
        encoding="utf-8",
    )
    (outputs / "validation_handoff_manifest.json").write_text(
        json.dumps({"changed_file_count": 3, "changed_files": {"probe_evidence": []}}),
        encoding="utf-8",
    )
    (outputs / "production_readiness_check.json").write_text(
        json.dumps(
            {
                "missing_evidence": [
                    {
                        "gate": "final_quality.tool_tcp_production_status",
                        "required_evidence": "Final quality report with tool_tcp_production_status=pass.",
                    },
                    {
                        "gate": "derived.tool_tcp_metadata_present",
                        "required_evidence": "Non-empty tool_tcp_measured_by, ISO tool_tcp_measured_date, and tool_tcp_calibration_method.",
                    },
                    {
                        "gate": "derived.max_joint_step_within_limit",
                        "required_evidence": "max_joint_step_deg less than or equal to production_joint_step_limit_deg.",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "README.md").write_text("# 正常中文\n", encoding="utf-8")
    (tmp_path / "PROJECT_REPORT_README.md").write_text("# 正常中文\n", encoding="utf-8")

    audit = module.build_audit(tmp_path)
    by_id = {item["id"]: item for item in audit["items"]}

    assert audit["overall_goal_status"] == "active"
    assert by_id["original_145_of_145_collision"]["status"] == "pass"
    assert by_id["original_145_of_145_collision"]["evidence"]["raw_collision_status"] == "pass"
    assert by_id["original_145_of_145_collision"]["evidence"]["raw_collision_checked_state_count"] == "145"
    assert by_id["diagnostic_low_speed_not_production"]["status"] == "pass"
    assert by_id["placeholder_tcp_not_production"]["status"] == "blocked"
    assert by_id["default_20deg_ik_continuity"]["status"] == "blocked"
    assert audit["readiness_missing_evidence"]
    tcp_missing = by_id["placeholder_tcp_not_production"]["evidence"]["readiness_missing_evidence"]
    ik_missing = by_id["default_20deg_ik_continuity"]["evidence"]["readiness_missing_evidence"]
    assert {item["gate"] for item in tcp_missing} == {
        "final_quality.tool_tcp_production_status",
        "derived.tool_tcp_metadata_present",
    }
    assert [item["gate"] for item in ik_missing] == ["derived.max_joint_step_within_limit"]
    assert by_id["readme_chinese_mojibake"]["status"] == "pass"
    assert set(audit["blocked_items"]) == {
        "placeholder_tcp_not_production",
        "default_20deg_ik_continuity",
    }


def test_goal_resolution_audit_source_has_clean_mojibake_markers() -> None:
    module_path = BRIDGE.parents[1] / "tools" / "build_goal_resolution_audit.py"
    source = module_path.read_text(encoding="utf-8")

    assert "\ufffd" not in source
    assert "0xFFFD" in source


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
    assert "PASS: ROS2/MoveIt2 runtime execution-chain evidence" in normal.stdout
    assert "segmented spray-off reorientation evidence" in normal.stdout
    assert "omits the bottom closure" not in normal.stdout
    assert strict.returncode != 0
    assert "WARN: ROS2/MoveIt2 runtime execution-chain evidence" in strict.stdout


def test_goal_audit_strict_runtime_accepts_explicit_runtime_reports(tmp_path) -> None:
    import subprocess
    import sys

    quality = tmp_path / "moveit_quality_report.csv"
    dynamics = tmp_path / "moveit_joint_dynamics_report.csv"
    trace = tmp_path / "moveit_fk_tcp_trace.csv"
    collision = tmp_path / "moveit_collision_report.csv"
    quality.write_text(
        "\n".join(
            [
                "metric,value",
                "fk_normal_error_max_deg,1.0",
                "fk_standoff_error_max_abs_mm,2.0",
                "fk_path_deviation_max_mm,3.0",
                "fk_tcp_speed_mean_m_s,0.01",
                "fk_tcp_speed_p05_p95_fluctuation,0.01",
                "moveit_ruckig_smoothing_used,true",
                "moveit_ee_link,spray_tcp_link",
                "max_joint_step_deg,1.0",
                "production_joint_step_limit_deg,20.0",
                "joint_continuity_status,pass",
                "tool_tcp_xyz,0.010 0.000 0.185",
                "tool_tcp_rpy,0.000 0.100 0.000",
                "tool_tcp_source,measured_nozzle_tcp",
                "tool_tcp_measured_by,calibration_engineer",
                "tool_tcp_measured_date,2026-07-08",
                "tool_tcp_calibration_method,flange_fixture_probe",
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
    collision.write_text(
        "\n".join(
            [
                "metric,value",
                "collision_environment_object_count,60",
                "collision_checked_state_count,145",
                "collision_count,0",
                "first_collision_index,-1",
                "first_collision_contact_count,0",
                "include_bottom_closure_collision,true",
                "status,pass",
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
            "--moveit-collision-report",
            str(collision),
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
