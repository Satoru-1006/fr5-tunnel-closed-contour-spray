from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from scripts.publish_p2b3_c1_result import (
    _pressure_waypoint_evidence,
    derive_c1_stage_status,
)


def test_c1_status_separates_fatal_gates_from_limitations() -> None:
    gates = {"strict": "PASS", "lineage": "PASS"}
    assert derive_c1_stage_status(gates, ["local timing diagnostic"]) == "COMPLETE_WITH_LIMITATIONS"
    assert derive_c1_stage_status(gates, []) == "COMPLETE"
    assert derive_c1_stage_status({**gates, "strict": "FAIL"}, []) == "FAIL"
    assert derive_c1_stage_status({**gates, "lineage": "UNKNOWN"}, []) == "BLOCKED"


def test_pressure_evidence_keeps_tight_convergence_separate_from_relaxed_acceptance(tmp_path: Path) -> None:
    path = tmp_path / "solver_pressure.csv"
    fields = ["case_id", "variant", "waypoint", "iteration", "current_q", "accepted_q",
              "line_search_accepted", "projection_triggered", "position_residual_m", "normal_residual"]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for waypoint in range(181):
            position = [1e-5, 0.0, 0.0] if waypoint == 0 else [1e-3, 0.0, 0.0]
            normal = [1e-5, 0.0] if waypoint == 0 else [0.02, 0.0]
            q = [0.0] * 6
            writer.writerow({
                "case_id": "P2B3_C1_R0", "variant": "B0", "waypoint": waypoint, "iteration": 0,
                "current_q": json.dumps(q), "accepted_q": json.dumps(q),
                "line_search_accepted": "true" if waypoint == 0 else "false",
                "projection_triggered": "false", "position_residual_m": json.dumps(position),
                "normal_residual": json.dumps(normal),
            })

    rows, summary = _pressure_waypoint_evidence(path)
    assert rows[0]["tight_converged"] is True
    assert rows[1]["tight_converged"] is False
    assert rows[0]["target_attempted"] and rows[0]["target_state_emitted"]
    assert rows[1]["final_relaxed_geometry_gate"] == "PASS_4MM_5DEG"
    assert summary["tight_converged_count"] == 1
    assert summary["accepted_relaxed_geometry_count"] == 181


def test_pressure_evidence_rejects_missing_target_and_wrong_case(tmp_path: Path) -> None:
    path = tmp_path / "solver_pressure.csv"
    fields = ["case_id", "variant", "waypoint", "iteration", "current_q", "accepted_q",
              "line_search_accepted", "projection_triggered", "position_residual_m", "normal_residual"]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for waypoint in range(180):
            writer.writerow({
                "case_id": "wrong", "variant": "B0", "waypoint": waypoint, "iteration": 0,
                "current_q": json.dumps([0.0] * 6), "accepted_q": json.dumps([0.0] * 6),
                "line_search_accepted": "true", "projection_triggered": "false",
                "position_residual_m": json.dumps([0.0] * 3), "normal_residual": json.dumps([0.0] * 2),
            })
    with pytest.raises(ValueError, match="case_or_variant_mismatch"):
        _pressure_waypoint_evidence(path)
