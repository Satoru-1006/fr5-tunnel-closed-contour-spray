"""Structural and pure-function tests for the additive Stage 3 H7.1 audit."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.stage3_h7_1 import H6_JSONL, load_jsonl, turn_geometry_audit


ROOT = Path(__file__).resolve().parents[1]


def latest_h7_1() -> Path:
    candidates = sorted(
        path for path in (ROOT / "outputs").glob("stage3_h7_1_authoritative_audit_*")
        if (path / "stage3_h7_1_terminal_certificate.json").is_file()
    )
    assert candidates, "Stage 3 H7.1 output directory is missing"
    return candidates[-1]


def test_turn_geometry_uses_native_totg_cosine_criterion() -> None:
    rows = load_jsonl(H6_JSONL)
    audit = turn_geometry_audit(rows)
    assert audit["waypoint_count"] == 1295
    assert audit["exact_or_near_180_turn_centres"] == [40, 260]
    assert audit["moveit_totg_source_semantics"]["criterion"] == "cos_angle <= -1.0 + angle_tolerance"


def test_h7_1_artifacts_are_blocked_and_safe() -> None:
    output = latest_h7_1()
    terminal = json.loads((output / "stage3_h7_1_terminal_certificate.json").read_text(encoding="utf-8"))
    assert terminal["STAGE_3_H7_1"] == "BLOCKED"
    assert terminal["NEW_FJT_GOALS_SENT"] == 0
    assert terminal["ROBOT_MOTION_STARTED"] == "NO"
    assert terminal["READY_FOR_STAGE_3_H8"] == "NO"
