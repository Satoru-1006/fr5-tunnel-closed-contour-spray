"""Focused D50 regression checks for repaired measurement semantics."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
D50 = ROOT / "outputs" / "D50_STAGE4_INTEGRATED_SHADOW"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_native_profile_oracle_is_authoritative_and_all_cases_pass() -> None:
    report = read_json(D50 / "phase_b_jerk_truth/jerk_truth_resolution.json")
    assert report["authoritative_jerk_oracle"] == "NATIVE_RUCKIG_PROFILE_J"
    assert report["all_analytic_profiles_pass"] is True
    assert report["candidates"]["global_0075"]["finite_difference_limit_violations"] == 24
    assert report["candidates"]["global_0075"]["analytic_successful_profile_count"] == report["candidates"]["global_0075"]["analytic_segment_count"]


def test_known_answer_native_profile_and_clearance_are_fail_closed() -> None:
    known = read_json(D50 / "phase_b_jerk_truth/known_answer/summary.json")
    clearance = read_json(D50 / "phase_d_global_0075_clearance.json")
    assert known["jerk_truth"] == "NATIVE_RUCKIG_PROFILE_J"
    assert known["passed"] is True
    assert clearance["status"] == "UNRESOLVED_NONPOSITIVE_OR_MISSING_BOUND"
    assert clearance["positive_lower_bound_all_cases"] is False


def test_native_d41_replay_has_complete_provenance_and_twelve_cases() -> None:
    root = D50 / "phase_a_d41_native/native_output"
    rows = [json.loads(line) for line in (root / "D41_native_case_summary.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    provenance = read_json(root / "D41_native_provenance.json")
    assert len(rows) == 12
    assert all(row["status"] == "PASS" for row in rows)
    assert provenance["pose_count"] == 181
    assert provenance["source_scope"] == "181-point ON-state open-arch only"
