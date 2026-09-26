from __future__ import annotations

import json
from pathlib import Path

from tools.stage3_d45_finalize_release import (
    CANONICAL_H1,
    CANONICAL_H32,
    CANONICAL_SHA256,
    COLLISION_METHOD,
    POINT_COUNT,
    verify_authoritative_inputs,
)


ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / "outputs" / "STAGE3_FINAL_RELEASE"


def test_authoritative_input_contract_is_181_point_open_arch() -> None:
    result = verify_authoritative_inputs()
    assert result["point_count"] == POINT_COUNT
    assert result["future_joint_reads"] == 0
    assert result["legacy_720_point_outputs_in_scope"] is False
    assert result["spray_off_or_reorientation_in_scope"] is False


def test_final_release_manifest_freezes_canonical_identity() -> None:
    manifest = json.loads((RELEASE / "release_manifest.json").read_text(encoding="utf-8"))
    metrics = json.loads((RELEASE / "canonical_metrics_h1_h32.json").read_text(encoding="utf-8"))
    assert manifest["release_status"] == "FROZEN_AND_CLOSED"
    assert manifest["canonical_checkpoint_sha256"] == CANONICAL_SHA256
    assert metrics["H1"] == CANONICAL_H1
    assert metrics["H32"] == CANONICAL_H32
    assert manifest["collision_method"] == COLLISION_METHOD
    assert manifest["ccd"] == "not_available"
    assert manifest["clearance"] is None


def test_final_release_reports_no_unresolved_correctness_blocker() -> None:
    report = (RELEASE / "STAGE3_CLOSURE_REPORT.md").read_text(encoding="utf-8")
    assert "STAGE_3_FINAL_CLOSURE: PASS" in report
    assert "UNRESOLVED_CORRECTNESS_BLOCKERS: `NONE`" in report
    assert "REGRESSIONS_INTRODUCED_BY_ACCEPTED_REPAIRS: `0`" in report
