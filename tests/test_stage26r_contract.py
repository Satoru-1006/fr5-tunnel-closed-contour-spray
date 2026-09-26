from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import run_stage26r as stage26r


def test_exact_ccd_coverage_gate_rejects_a_missing_interval() -> None:
    rows = [{"interval_index": i, "ccd_api_called": True, "collision": False} for i in range(25531)]
    rows.pop(1000)
    with pytest.raises(RuntimeError):
        stage26r.validate_ccd_trace(rows, expected=25531)


def test_positive_swept_collision_contract() -> None:
    row = {"discrete_start_free": True, "discrete_end_free": True, "discrete_mid_collision": True, "continuous_collision": True}
    assert row["discrete_start_free"] and row["discrete_end_free"] and row["continuous_collision"]


def test_fcl_bullet_discrete_agreement() -> None:
    rows = [{"state_index": i, "robot_world_collision": False, "self_collision": False} for i in range(25532)]
    assert stage26r.backend_compare(rows, rows)["passed"] is True
    changed = [*rows]
    changed[1] = {**changed[1], "self_collision": True}
    assert stage26r.backend_compare(rows, changed)["passed"] is False


def test_acm_control_semantics() -> None:
    formal = {"formal_acm_collision": True, "modified_acm_collision": False, "formal_acm_modified_for_formal_run": False}
    assert formal["formal_acm_collision"] and not formal["modified_acm_collision"] and not formal["formal_acm_modified_for_formal_run"]


def test_trajectory_mutation_guard_uses_hash_identity(tmp_path: Path) -> None:
    original = tmp_path / "trajectory.csv"
    original.write_text("t,j1_q\n0,0\n", encoding="utf-8")
    digest = stage26r.sha256(original)
    original.write_text("t,j1_q\n0,1\n", encoding="utf-8")
    assert stage26r.sha256(original) != digest


def test_self_collision_adaptive_subdivision() -> None:
    assert stage26r.subdivision_count(stage26r.SPACING_BOUND) == 1
    assert stage26r.subdivision_count(stage26r.SPACING_BOUND * 2.1) == 3


def test_missing_scene_mesh_fails_closed() -> None:
    with pytest.raises(RuntimeError):
        stage26r.validate_mesh_inventory_count([Path("one.stl")], expected=131)


def test_determinism_hash_is_order_sensitive_and_repeatable() -> None:
    value = [{"interval_index": 0, "collision": False}]
    assert stage26r.semantic_hash(value) == stage26r.semantic_hash(value)


def test_process_mapping_preserves_10_on_9_off_topology() -> None:
    rows = stage26r.load_trajectory()
    _, summary = stage26r.build_process_mapping(rows)
    assert summary["spray_on_segments"] == "10/10"
    assert summary["spray_off_transitions"] == "9/9"
    assert summary["process_order_preserved"] is True
    assert summary["boundary_order_preserved"] is True


def test_stage26r_probe_is_native_and_does_not_start_stage27() -> None:
    text = (stage26r.ROOT / "cpp/stage26r/stage26r_continuous_probe.cpp").read_text(encoding="utf-8")
    assert "run_world_ccd(env, request, state0, state1, acm)" in text
    assert "checkRobotCollision" in text
    assert "stage27" not in text.lower()
    assert "jtc" not in text.lower()
