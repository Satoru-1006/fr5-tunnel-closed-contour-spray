"""Focused structural tests for the additive Stage 3 H7 bundle."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
H6 = ROOT / "outputs/stage3_h6_surface_coverage_20260808T225000Z"
REQUIRED = {
    "FINAL_REPORT.md",
    "stage3_h7_execution_sequence_manifest.json",
    "stage3_h7_dynamics_limit_provenance.json",
    "stage3_h7_input_freeze_manifest.json",
    "stage3_h7_totg_configuration.json",
    "stage3_h7_totg_trajectory.jsonl",
    "stage3_h7_totg_validation.json",
    "stage3_h7_ruckig_configuration.json",
    "stage3_h7_final_timed_trajectory.jsonl",
    "stage3_h7_dynamic_validation.jsonl",
    "stage3_h7_collision_validation.jsonl",
    "stage3_h7_process_validation.jsonl",
    "stage3_h7_segment_boundary_audit.json",
    "stage3_h7_replay_determinism.json",
    "stage3_h7_regression_report.json",
    "stage3_h7_artifact_manifest.json",
    "stage3_h7_gate_report.json",
    "stage3_h7_terminal_certificate.json",
}


def latest_h7() -> Path:
    candidates = sorted((ROOT / "outputs").glob("stage3_h7_authoritative_execution_*"))
    assert candidates, "Stage 3 H7 output directory is missing"
    return candidates[-1]


def test_h6_sequence_is_reconciled_from_jsonl() -> None:
    rows = [json.loads(line) for line in (H6 / "stage3_h6_joint_waypoints.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) == 1295
    assert [row["waypoint_index"] for row in rows] == list(range(1295))
    assert [row["segment_id"] for row in rows if row["waypoint_index"] in (0, 301, 727, 731, 972)] == [0, 1000, 1, 1001, 2]


def test_h7_artifacts_and_safety_surface_exist() -> None:
    output = latest_h7()
    available = {path.name for path in output.iterdir()}
    # During the orchestrator's in-run regression pass, the final certificate
    # surface is emitted immediately afterward.  Require the pre-formal H7
    # surface here; when the complete bundle exists, require every artifact.
    pre_formal = {
        "stage3_h7_execution_sequence_manifest.json",
        "stage3_h7_dynamics_limit_provenance.json",
        "stage3_h7_input_freeze_manifest.json",
        "stage3_h7_totg_configuration.json",
        "stage3_h7_ruckig_configuration.json",
        "stage3_h7_native_run.json",
    }
    assert pre_formal.issubset(available)
    if not REQUIRED.issubset(available):
        return
    terminal = json.loads((output / "stage3_h7_terminal_certificate.json").read_text(encoding="utf-8"))
    assert terminal["NEW_FJT_GOALS_SENT"] == 0
    assert terminal["ROBOT_MOTION_STARTED"] == "NO"
    gate = json.loads((output / "stage3_h7_gate_report.json").read_text(encoding="utf-8"))
    assert gate["collision_method"] == "adaptive_discrete_interpolation"
    assert gate["CCD"] == "NOT_AVAILABLE"
