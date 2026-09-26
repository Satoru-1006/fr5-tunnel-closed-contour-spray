from __future__ import annotations

import sys
import ast
import importlib
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ros2_moveit_bridge"))

from p2b1_solver_policy_ablation import (  # noqa: E402
    PRESSURE_FIELDS,
    capability_status,
    compare_baseline,
    dls_primary_command,
    joint_centering_objective,
    nullspace_metrics,
    process_jacobian,
    process_residual,
    project_joint_limits,
    result_schema_is_valid,
    secondary_command,
    skew,
    tangent_bases,
    weighted_task,
)
from scripts.run_p2b1_solver_policy_ablation import ensure_project_root_importable, read_metric_report  # noqa: E402


def _rotation(vector: np.ndarray) -> np.ndarray:
    angle = float(np.linalg.norm(vector))
    if angle < 1e-14:
        return np.eye(3)
    axis = vector / angle
    cross = skew(axis)
    return np.eye(3) + np.sin(angle) * cross + (1.0 - np.cos(angle)) * (cross @ cross)


def _rank5_case():
    normal, basis, angular_basis = tangent_bases([0.0, 0.0, 4.0])
    Jv = np.column_stack((np.eye(3), np.zeros((3, 3))))
    Jw = np.column_stack((np.zeros((3, 3)), np.eye(3)))
    J = process_jacobian(Jv, Jw, normal, normal, basis)
    return normal, basis, angular_basis, Jv, Jw, J


def test_tangent_basis_is_orthonormal_tangent_and_jacobian_is_five_by_six() -> None:
    normal, basis, angular_basis, Jv, Jw, J = _rank5_case()
    assert basis.shape == (3, 2)
    assert angular_basis.shape == (3, 2)
    assert J.shape == (5, 6)
    assert np.allclose(basis.T @ basis, np.eye(2), atol=1e-14)
    assert np.allclose(angular_basis.T @ angular_basis, np.eye(2), atol=1e-14)
    assert np.allclose(normal @ basis, 0.0, atol=1e-14)
    assert np.allclose(normal @ angular_basis, 0.0, atol=1e-14)
    assert np.linalg.matrix_rank(J) == 5


def test_on_manifold_angular_block_equals_projected_angular_jacobian() -> None:
    normal, basis, angular_basis, _Jv, Jw, J = _rank5_case()
    assert np.allclose(J[3:], angular_basis.T @ Jw, atol=1e-14)


def test_process_residual_jacobian_finite_difference_at_multiple_local_states() -> None:
    rng = np.random.default_rng(20260926)
    for _case in range(5):
        desired = rng.normal(size=3)
        normal, basis, _angular = tangent_bases(desired)
        tool_z = normal + rng.normal(scale=0.08, size=3)
        tool_z /= np.linalg.norm(tool_z)
        Jv = rng.normal(size=(3, 6)) * 0.2
        Jw = rng.normal(size=(3, 6)) * 0.4
        J = process_jacobian(Jv, Jw, tool_z, normal, basis)
        position = rng.normal(size=3)
        target = rng.normal(size=3)
        base = process_residual(position, target, tool_z, normal, basis)
        for epsilon in (5e-4, 1e-4, 2e-5):
            direction = rng.normal(size=6)
            direction /= np.linalg.norm(direction)
            dq = epsilon * direction
            z_plus = _rotation(Jw @ dq) @ tool_z
            p_plus = position + Jv @ dq
            observed = process_residual(p_plus, target, z_plus, normal, basis) - base
            predicted = J @ dq
            assert np.linalg.norm(observed - predicted) <= 2e-9 + 12.0 * np.linalg.norm(dq) ** 2


def test_nullspace_projector_is_symmetric_and_idempotent() -> None:
    *_prefix, J = _rank5_case()
    metrics = nullspace_metrics(J)
    N = metrics.projector
    assert metrics.rank == 5
    assert metrics.nullity == 1
    assert np.allclose(N, N.T, atol=1e-13)
    assert np.allclose(N @ N, N, atol=1e-13)


def test_primary_task_null_residual_is_zero() -> None:
    *_prefix, J = _rank5_case()
    N = nullspace_metrics(J).projector
    command = np.asarray([0.3, -0.1, 0.6, 0.7, -0.2, 0.4])
    assert np.linalg.norm(J @ N @ command) < 1e-13


def test_joint_centering_gradient_matches_known_value() -> None:
    lower = np.full(6, -2.0)
    upper = np.full(6, 2.0)
    q = np.asarray([1.0, -1.0, 0.5, 0.0, 2.0, -2.0])
    objective, gradient = joint_centering_objective(q, lower, upper)
    assert objective == pytest.approx(0.5 * (0.25 + 0.25 + 0.0625 + 0.0 + 1.0 + 1.0))
    assert np.allclose(gradient, q / 4.0)


def test_secondary_command_is_projected_into_task_nullspace() -> None:
    *_prefix, J = _rank5_case()
    metrics = nullspace_metrics(J)
    command, _norm = secondary_command(np.full(6, 0.7), np.full(6, -1.0), np.full(6, 1.0), metrics.projector, 0.01)
    assert np.linalg.norm(J @ command) < 1e-14


def test_secondary_command_does_not_first_order_change_primary_task() -> None:
    *_prefix, J = _rank5_case()
    metrics = nullspace_metrics(J)
    command, _norm = secondary_command(np.asarray([0, 0, 0, 0, 0, 0.8]), np.full(6, -1.0), np.full(6, 1.0), metrics.projector, 0.002)
    assert np.linalg.norm(J @ command) <= 1e-14


def test_weighted_process_task_preserves_shape_and_finiteness() -> None:
    *_prefix, J = _rank5_case()
    error, weighted = weighted_task(np.arange(5, dtype=float), J, 100.0, 0.1)
    assert error.shape == (5,)
    assert weighted.shape == (5, 6)
    assert np.isfinite(error).all() and np.isfinite(weighted).all()


def test_joint_limit_projection_reports_changed_joints() -> None:
    lower = np.full(6, -2.0)
    upper = np.full(6, 2.0)
    projected, changed = project_joint_limits([0, 0, 0, 0, 0, 2.1], lower, upper, 0.1)
    assert projected[5] == pytest.approx(1.9)
    assert changed == [5]


@pytest.mark.parametrize("buffer", [1e-4, 1e-3, 5e-3, 1e-2])
def test_buffer_variants_have_exact_known_projection(buffer: float) -> None:
    lower = np.full(6, -3.0543)
    upper = np.full(6, 3.0543)
    projected, changed = project_joint_limits(np.full(6, 3.1), lower, upper, buffer)
    assert np.allclose(projected, upper - buffer)
    assert changed == [0, 1, 2, 3, 4, 5]


def test_invalid_and_nonfinite_jacobians_fail_closed() -> None:
    with pytest.raises(ValueError):
        process_jacobian(np.full((3, 6), np.nan), np.zeros((3, 6)), [0, 0, 1], [0, 0, 1], np.eye(3)[:, :2])
    with pytest.raises(ValueError):
        process_jacobian(np.zeros((3, 5)), np.zeros((3, 6)), [0, 0, 1], [0, 0, 1], np.eye(3)[:, :2])


def test_rank_deficient_case_is_reported_without_nan_projector() -> None:
    rank_one = np.zeros((5, 6))
    rank_one[0, 0] = 1.0
    metrics = nullspace_metrics(rank_one)
    assert metrics.rank == 1
    assert metrics.nullity == 5
    assert np.isfinite(metrics.projector).all()
    assert np.allclose(metrics.projector @ metrics.projector, metrics.projector, atol=1e-13)


def test_missing_moveit_capability_is_unavailable() -> None:
    assert capability_status(False, "MoveIt2 planning scene unavailable") == {
        "status": "UNAVAILABLE", "reason": "MoveIt2 planning scene unavailable",
    }


def test_a0_baseline_comparator_passes_exact_and_fails_material_difference() -> None:
    q = np.zeros((181, 6))
    assert compare_baseline(q, q)["status"] == "PASS"
    changed = q.copy()
    changed[86, 5] = 1e-5
    assert compare_baseline(q, changed)["status"] == "FAIL"


def test_a0_baseline_comparator_rejects_wrong_shape_or_nonfinite_values() -> None:
    assert compare_baseline(np.zeros((181, 6)), np.zeros((180, 6)))["shape_match"] is False
    bad = np.zeros((181, 6))
    bad[86, 5] = np.nan
    assert compare_baseline(np.zeros((181, 6)), bad)["status"] == "FAIL"


def test_result_schema_requires_separate_pipeline_and_baseline_statuses() -> None:
    result = {
        "schema": "p2b1-solver-policy-causal-ablation-v1", "project": "FAIRINO_FR5", "stage": "P2-B1",
        "source_base_commit": "base", "execution_code_commit": "code", "variants": [],
        "causal_classification": "INCONCLUSIVE", "measurement_pipeline_status": "INCONCLUSIVE",
        "frozen_robot_baseline_performance_status": "FAIL", "limitations": [],
    }
    assert result_schema_is_valid(result)
    result.pop("measurement_pipeline_status")
    assert not result_schema_is_valid(result)


def test_solver_pressure_schema_contains_projection_and_nullspace_evidence() -> None:
    assert "proposed_q_before_projection" in PRESSURE_FIELDS
    assert "projected_q" in PRESSURE_FIELDS
    assert "accepted_alpha" in PRESSURE_FIELDS
    assert "j6_nullspace_projection" in PRESSURE_FIELDS
    assert "secondary_objective_contribution" in PRESSURE_FIELDS
    assert "primary_task_contribution_norm" in PRESSURE_FIELDS
    assert "secondary_objective_contribution_norm" in PRESSURE_FIELDS
    assert "max_j6_upper_bound_excess" in PRESSURE_FIELDS
    assert "max_j6_buffer_boundary_excess" in PRESSURE_FIELDS


def test_original_dls_pressure_hook_calls_the_imported_writer_alias() -> None:
    source = (ROOT / "ros2_moveit_bridge/plan_closed_contour_moveit.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported_aliases = {
        item.asname
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module == "p2b1_solver_policy_ablation"
        for item in node.names
        if item.name == "append_pressure_record"
    }
    called_names = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "_append_p2b1_pressure_record" in imported_aliases
    assert "_append_p2b1_pressure_record" in called_names


def test_original_dls_captures_iteration_state_before_updating_it() -> None:
    source = (ROOT / "ros2_moveit_bridge/plan_closed_contour_moveit.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    solver = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "build_normal_constrained_dls_trajectory"
    )
    loops = [
        node for node in ast.walk(solver)
        if isinstance(node, ast.For) and isinstance(node.target, ast.Name) and node.target.id == "_iteration"
    ]
    assert len(loops) == 1
    first = loops[0].body[0]
    assert isinstance(first, ast.Assign)
    assert any(isinstance(target, ast.Name) and target.id == "iteration_q" for target in first.targets)
    assert isinstance(first.value, ast.Call)
    assert isinstance(first.value.func, ast.Attribute)
    assert first.value.func.attr == "copy"
    assert isinstance(first.value.func.value, ast.Name)
    assert first.value.func.value.id == "accepted"


def test_report_reader_handles_metric_value_and_per_joint_dynamics_schemas(tmp_path: Path) -> None:
    metric_report = tmp_path / "quality.csv"
    metric_report.write_text("metric,value\ncollision_count,0\n", encoding="utf-8")
    assert read_metric_report(metric_report, "quality") == {
        "status": "AVAILABLE", "metrics": {"collision_count": "0"},
    }

    dynamics_report = tmp_path / "dynamics.csv"
    dynamics_report.write_text("joint,max_velocity,jerk_ratio\nj1,0.2,0.03\n", encoding="utf-8")
    assert read_metric_report(dynamics_report, "dynamics") == {
        "status": "AVAILABLE", "joints": {"j1": {"max_velocity": "0.2", "jerk_ratio": "0.03"}},
    }


def test_unknown_report_schema_is_retained_without_false_metric_mapping(tmp_path: Path) -> None:
    report = tmp_path / "unexpected.csv"
    report.write_text("field,value\na,1\n", encoding="utf-8")
    parsed = read_metric_report(report, "unknown")
    assert parsed == {"status": "UNRESOLVED_SCHEMA", "fieldnames": ["field", "value"], "rows": [{"field": "a", "value": "1"}]}


def test_runner_exposes_repository_modules_when_invoked_from_scripts_directory(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "path", [str(ROOT / "scripts")])
    ensure_project_root_importable()
    spec = importlib.util.find_spec("tools.stage4a_system_benchmark")
    assert spec is not None and spec.origin is not None
    assert Path(spec.origin).resolve() == (ROOT / "tools/stage4a_system_benchmark.py").resolve()


def test_dls_command_is_finite_for_rank_deficient_process_matrix() -> None:
    J = np.zeros((5, 6))
    J[0, 0] = 1.0
    command = dls_primary_command(J, np.ones(5), 1e-2)
    assert np.isfinite(command).all()
    assert command.shape == (6,)
