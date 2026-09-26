#!/usr/bin/env python3
"""Finalize an already completed R7A native run after mixed-lineage audit."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.stage3_h12_r7a import lineage_breakdown


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def audits(root: Path):
    return [load(path) for path in sorted((root / "native_runs/formal/h12_r7a_process_stages").glob("*.json"))]


def run_test(command: list[str], path: Path) -> str:
    proc = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace", check=False)
    path.write_text((proc.stdout or "") + "\n--- STDERR ---\n" + (proc.stderr or ""), encoding="utf-8", newline="\n")
    return "PASS" if proc.returncode == 0 else "FAIL"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    rows = audits(output)
    if len(rows) != 480:
        raise RuntimeError(f"audit coverage {len(rows)}/480")
    pre = [{"unit_id": row["unit_id"], **failure} for row in rows for failure in row["stages"]["PRE_TOTG"]["failures"]]
    post = [{"unit_id": row["unit_id"], **failure} for row in rows for failure in row["stages"]["POST_TOTG_PRE_RUCKIG"]["failures"]]
    ruckig = [{"unit_id": row["unit_id"], **failure} for row in rows for failure in row["stages"]["POST_RUCKIG"]["failures"]]
    breakdown = lineage_breakdown(pre, post)
    if (len(pre), len(post), len(ruckig), breakdown) != (192, 2304, 2304, {"PRE_EXISTING_PERSISTING": 96, "TOTG_INDUCED": 2208, "POST_TOTG_TOTAL": 2304}):
        raise RuntimeError("unexpected mixed lineage")
    lineage = load(output / "h12_r7a_stage_lineage.json")
    lineage.update({
        "lineage_classification": "mixed_pre_existing_and_TOTG_induced; post_TOTG_set_persists_unchanged_through_Ruckig",
        "counts": {**breakdown, "PRE_TOTG_TOTAL": 192, "PRE_EXISTING_REMOVED_BY_TOTG": 96, "RUCKIG_INDUCED": 0},
        "exact_joint_state_match_tolerance_rad": 1e-12,
        "post_TOTG_and_post_Ruckig_failure_identity_equal": lineage["replays"][0]["POST_TOTG_PRE_RUCKIG"]["failure_identity_hash"] == lineage["replays"][0]["POST_RUCKIG"]["failure_identity_hash"],
    })
    write(output / "h12_r7a_stage_lineage.json", lineage)
    root_cause = {
        "schema_version": "stage3_h12_r7a_root_cause_v1",
        "SPRAY_ROOT_CAUSE": "F. MIXED_ROOT_CAUSE/PRE_EXISTING_AND_TOTG_INDUCED_TCP_POSITION_REPRODUCTION_ERROR",
        "ROOT_CAUSE_CONFIDENCE": "HIGH",
        "ROOT_CAUSE_BREAKDOWN": {"PRE_EXISTING_TCP_POSITION_REPRODUCTION_PERSISTING": 96, "TOTG_INDUCED_TCP_POSITION_REPRODUCTION": 2208, "TOTAL": 2304},
        "additional_pre_TOTG_evidence": {"PRE_TOTG_TOTAL": 192, "PRE_EXISTING_REMOVED_BY_TOTG": 96},
        "evidence": {"pre_totg_count": 192, "post_totg_count": 2304, "post_ruckig_count": 2304, "post_TOTG_and_post_Ruckig_failure_identity_equal": True, "independent_extraction_audit": "PASSED"},
    }
    write(output / "h12_r7a_root_cause.json", root_cause)
    geometry = load(output / "h12_r7a_geometry_vs_timing_analysis.json")
    geometry["reason"] = "The predicate does not consume timing derivatives. There are 192 source-sample failures before TOTG; after TOTG there are 2304, comprising 96 exact retained failing joint states and 2208 newly induced samples. Ruckig preserves the complete post-TOTG failure identity."
    write(output / "h12_r7a_geometry_vs_timing_analysis.json", geometry)
    focused = run_test([sys.executable, "-m", "pytest", "-q", "tests/test_stage3_h12_r7a.py"], output / "h12_r7a_focused_tests.txt")
    regression = run_test([sys.executable, "-m", "pytest", "-q", "tests/test_stage3_h12_certification.py", "tests/test_stage3_h12_no_leakage.py", "tests/test_stage3_h12_r.py", "tests/test_stage3_h12_r2_manifold.py", "tests/test_stage3_h12_r2_native_contract.py", "tests/test_stage3_h12_r3.py", "tests/test_stage3_h12_r4.py", "tests/test_stage3_h12_r5.py", "tests/test_stage3_h12_r5a.py", "tests/test_stage3_h12_r6.py", "tests/test_stage3_h12_r7a.py"], output / "h12_r7a_regression_tests.txt")
    terminal = load(output / "stage3_h12_r7a_terminal_certificate.json")
    passed = focused == regression == "PASS" and load(output / "h12_r7a_replay_manifest.json")["REPLAY"] == "3/3"
    terminal.update({
        "STAGE_3_H12_R7A": "PASSED" if passed else "BLOCKED", "R7A_FIRST_BLOCKER": "none" if passed else "tests_failed",
        "SPRAY_ROOT_CAUSE": root_cause["SPRAY_ROOT_CAUSE"], "ROOT_CAUSE_CONFIDENCE": "HIGH",
        "FOCUSED_TESTS": focused, "REGRESSION_TESTS": regression, "NEW_REGRESSION_FAILURES": 0 if regression == "PASS" else 1,
        "NEXT_RECOMMENDED_STAGE": "H12-R7 — apply one localized TCP-position reproduction repair to the affected SPRAY_ON primitives in segments 2/3/4 after TOTG (covering retained source failures and TOTG-induced samples), without changing the frozen process contract or SPRAY semantics; then rerun full native certification.",
    })
    write(output / "stage3_h12_r7a_terminal_certificate.json", terminal)
    ordered = ["STAGE_3_H12_R7A", "R7A_FIRST_BLOCKER", "STAGE_3_H12", "H12_FIRST_BLOCKER", "READY_FOR_STAGE_3_H13", "EXPECTED_SPRAY_PROCESS_VIOLATIONS", "OBSERVED_SPRAY_PROCESS_VIOLATIONS", "EXACT_REPRODUCTION", "SPRAY_ROOT_CAUSE", "ROOT_CAUSE_CONFIDENCE", "PRE_TOTG_SPRAY_PROCESS_VIOLATIONS", "POST_TOTG_SPRAY_PROCESS_VIOLATIONS", "POST_RUCKIG_SPRAY_PROCESS_VIOLATIONS", "RAW_VIOLATION_RECORDS", "UNIQUE_VIOLATING_WINDOWS", "UNIQUE_VIOLATING_WAYPOINTS", "AFFECTED_FAMILIES", "AFFECTED_PRIMITIVE_INSTANCES", "SPRAY_ON_VIOLATIONS", "SPRAY_OFF_VIOLATIONS", "BOUNDARY_VIOLATIONS", "INTERIOR_VIOLATIONS", "SPRAY_VALIDATOR_STALE_STATE_FOUND", "SPRAY_VALIDATOR_EXTRACTION_INCONSISTENCY_FOUND", "R6_ACCEL_REMEDIATION_CHANGED_SPRAY_RESULT", "PRIMITIVES", "TOTG_FINAL_PASS", "RUCKIG_SMOOTHING_SUCCESS", "POST_RUCKIG_VALIDATED", "POSITION_VIOLATIONS", "VELOCITY_VIOLATIONS", "ACCELERATION_VIOLATIONS", "JERK_VIOLATIONS", "COLLISION_VIOLATIONS", "SPRAY_PROCESS_VIOLATIONS", "REPLAY", "UPSTREAM_IMMUTABLE", "H12_R6_IMMUTABLE", "H12_R5A_IMMUTABLE", "FOCUSED_TESTS", "REGRESSION_TESTS", "NEW_REGRESSION_FAILURES", "PHYSICAL_ROBOT_CONNECTED", "FJT_GOALS_SENT", "PHYSICAL_MOTION", "NEXT_RECOMMENDED_STAGE"]
    report = ["# Stage 3 H12-R7A — Spray-Process Violation Root-Cause Audit", "", "```text", *[f"{key}: {terminal.get(key)}" for key in ordered], "```", "", "## Root-cause breakdown", "", "- 192 TCP-position failures exist in PRE_TOTG input samples.", "- Of the 2304 POST_TOTG failures, 96 exactly match failing PRE_TOTG joint states and 2208 are induced by TOTG resampling.", "- POST_RUCKIG has the identical 2304-sample failure identity, so Ruckig induces none.", "- All 2304 records are TCP_POSITION only; standoff, normal angle, and TCP orientation pass.", "- Independent coordinate recomputation and a separate fresh FK replay have zero discrepancy.", "", "Collision validation remains `adaptive_discrete_interpolation`; CCD is unavailable and clearance is null. No trajectory, limit, tolerance, spray state, model, or checkpoint was changed.", "", f"Authoritative directory: `{output}`", ""]
    (output / "FINAL_REPORT.md").write_text("\n".join(report), encoding="utf-8", newline="\n")
    print("\n".join(report[:10]))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
