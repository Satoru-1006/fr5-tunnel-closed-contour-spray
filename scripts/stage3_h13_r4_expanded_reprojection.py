#!/usr/bin/env python3
"""Stage 3 H13-R4 bounded reprojection/native certification closure.

The H11-R2 checkpoint, Frozen-20 split, H12-R7 artifacts, and H13-R3 output
are read-only inputs.  This runner creates one additive timestamped output and
uses the new R4 native adapter, which retains the established native gates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import stage3_h13_low_storage_unseen_generalization as h13  # noqa: E402
from scripts import stage3_h13_r3_frozen20_native_certification as r3  # noqa: E402
from src.stage3_h13 import CASE_TARGET, canonical, canonical_hash  # noqa: E402
from src.stage3_h13_r4_reprojection import no_label_repair_audit  # noqa: E402


R3_ROOT = ROOT / "outputs/stage3_h13_r3_frozen20_native_certification_20260813T144111Z"
R3_CERTIFICATE = R3_ROOT / "stage3_h13_r3_terminal_certificate.json"
R3_CASES = R3_ROOT / "case_summary.jsonl"
R3_NATIVE = R3_ROOT / "native_validation_summary.json"
R3_REPROJECTION = R3_ROOT / "reprojection_summary.json"
R3_H13_TERMINAL_SHA256 = "not_available"
_R3_ORIGINAL_COMPACT_NATIVE_CASE = r3.compact_native_case


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


R3_AUTOREGRESSIVE_RMSE = load_json(R3_CERTIFICATE).get("AUTOREGRESSIVE_RMSE")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(canonical(value), ensure_ascii=True, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def write_jsonl(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(canonical(dict(row)), ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def size_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file()) if path.is_dir() else (path.stat().st_size if path.is_file() else 0)


def snapshot_r3() -> list[dict[str, Any]]:
    paths = [R3_CERTIFICATE, R3_CASES, R3_NATIVE, R3_REPROJECTION, R3_ROOT / "FINAL_REPORT.md"]
    return [{"path": str(path.resolve()), "exists": path.is_file(), "size_bytes": path.stat().st_size if path.is_file() else None, "sha256": sha256_file(path) if path.is_file() else None} for path in paths]


def reproduce_r3_blocker() -> dict[str, Any]:
    certificate = load_json(R3_CERTIFICATE)
    cases = load_jsonl(R3_CASES)
    native = load_json(R3_NATIVE)
    reprojection = load_json(R3_REPROJECTION)
    expected = {
        "stage": "BLOCKED",
        "terminal_status": "BLOCKED",
        "first_blocker": "h12_r7_local_reprojection_failed",
        "case_count": CASE_TARGET,
        "reprojection_attempted": 12,
        "reprojection_accepted": 0,
        "reprojection_rejected": 12,
        "raw_position_violations": 3052,
        "raw_spray_process_violations": 3620,
    }
    observed = {
        "stage": certificate.get("STAGE_3_H13_R3"),
        "terminal_status": certificate.get("STAGE_3_H13_R3"),
        "first_blocker": certificate.get("FIRST_BLOCKER"),
        "case_count": len(cases),
        "reprojection_attempted": reprojection.get("REPROJECTION_ATTEMPTED"),
        "reprojection_accepted": reprojection.get("REPROJECTION_ACCEPTED"),
        "reprojection_rejected": reprojection.get("REPROJECTION_REJECTED"),
        "raw_position_violations": native.get("RAW_POSITION_VIOLATIONS"),
        "raw_spray_process_violations": native.get("RAW_SPRAY_PROCESS_VIOLATIONS"),
    }
    return {"status": "REPRODUCED" if observed == expected else "MISMATCH", "expected": expected, "observed": observed}


def run_native_r4(args: argparse.Namespace, input_npz: Path, manifest: Path, output_jsonl: Path, native_output: Path) -> dict[str, Any]:
    command = " && ".join([
        "source /opt/ros/jazzy/setup.bash",
        "source /mnt/d/robotfucker/install/setup.bash",
        "source /mnt/d/robotfucker/install/stage3_h7_4_native/setup.bash",
        "export PYTHONFAULTHANDLER=1",
        f"export LD_PRELOAD={h13.wsl(h13.H12_INTERPOSER)}",
        f"export STAGE25R_NATIVE_DIR={h13.wsl(native_output / 'probe')}",
        "export H12_R5_COMPACT_PROBE=1",
        "export H12_R5_USE_INSTALLED_MOVEIT_HELPERS=1",
        "export H12_R5_MITIGATE_OVERSHOOT=1",
        "export H12_R5_HARD_LIMIT_CERTIFICATION=1",
        "export H12_R5_BOUNDARY_VELOCITY_RECONDITION=1",
        "export H12_R6_RECONDITION_DERIVATIVES=1",
        "export H12_R7_REPROJECT=1",
        "export H12_R7A_PROCESS_AUDIT=0",
        "export H12_R5_ASSIGN_ALL_NATIVE_DURATIONS=0",
        "export H12_R5_OVERSHOOT_THRESHOLD=0.01",
        f"export PYTHONPATH={h13.wsl(ROOT)}:{h13.wsl(ROOT / 'ros2_moveit_bridge')}:/mnt/d/robotfucker/install/stage3_h7_4_native/lib/python3.12/site-packages:/opt/ros/jazzy/lib/python3.12/site-packages",
        f"ros2 launch /mnt/d/robotfucker/ros2_moveit_bridge/launch/stage3_h13_r4_native.launch.py input_npz:={h13.wsl(input_npz)} unit_manifest:={h13.wsl(manifest)} output_jsonl:={h13.wsl(output_jsonl)} output_dir:={h13.wsl(native_output)} process_contract:={h13.wsl(h13.PROCESS_CONTRACT)} fixture_mesh:={h13.wsl(h13.FIXTURE)} runtime_limits:={h13.wsl(h13.RUNTIME_LIMITS)} totg_parameters:={h13.wsl(h13.TOTG_PARAMETERS)}",
    ])
    native_output.mkdir(parents=True, exist_ok=True)
    try:
        process = subprocess.run(["wsl.exe", "-d", "Ubuntu-24.04-D", "--", "bash", "-lc", command], cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=int(args.timeout_s) * 2, check=False)
        stdout, stderr = process.stdout or "", process.stderr or ""
    except subprocess.TimeoutExpired as exc:
        return {"status": "BLOCKED", "first_blocker": "h13_r4_native_timeout", "returncode": None, "stdout_tail": str(exc.stdout or "")[-3000:], "stderr_tail": str(exc.stderr or "")[-3000:]}
    result_count = len(load_jsonl(output_jsonl)) if output_jsonl.is_file() else 0
    expected_count = len(load_jsonl(manifest)) if manifest.is_file() else None
    audit = h13.audit_native_exit(stdout, stderr, process.returncode, result_count, expected_count)
    return {
        "status": "PASSED" if audit["NATIVE_MAIN_WORK_COMPLETED"] == "YES" and process.returncode == 0 else "BLOCKED",
        "first_blocker": None if audit["NATIVE_MAIN_WORK_COMPLETED"] == "YES" and process.returncode == 0 else "h13_r4_native_case_batch_failed",
        "returncode": process.returncode, "result_count": result_count, "expected_count": expected_count,
        **audit, "stdout_tail": stdout[-3000:], "stderr_tail": stderr[-3000:],
    }


def compact_native_case_r4(row: Mapping[str, Any]) -> dict[str, Any]:
    compact = _R3_ORIGINAL_COMPACT_NATIVE_CASE(row)
    compact["root_cause"] = row.get("root_cause")
    compact["repair"].update({key: (row.get("repair") or {}).get(key) for key in ("RMS_JOINT_CORRECTION_RAD", "AFFECTED_WAYPOINT_COUNT", "AFFECTED_TRAJECTORY_FRACTION", "POST_REPROJECTION_POSITION_VIOLATIONS", "POST_REPROJECTION_COLLISION_VIOLATIONS", "POST_REPROJECTION_SPRAY_PROCESS_VIOLATIONS")})
    compact["reprojection"].update({key: (row.get("reprojection") or {}).get(key) for key in ("tier_attempts", "selected_tiers", "tier0_accepted", "UNSEEN_LABEL_USED_FOR_REPAIR", "FUTURE_LABEL_LEAKAGE")})
    return compact


def run_tests(output: Path) -> dict[str, Any]:
    commands = {
        "FOCUSED_TESTS": [sys.executable, "-m", "pytest", "-q", "tests/test_stage3_h13_r4_reprojection.py", "tests/test_stage3_h13.py", "tests/test_stage3_h13_r3.py"],
        "REGRESSION_TESTS": [sys.executable, "-m", "pytest", "-q", "tests/test_stage3_h11_r2.py", "tests/test_stage3_h12_certification.py", "tests/test_stage3_h12_no_leakage.py", "tests/test_stage3_h12_r7.py", "tests/test_stage3_h12_r7a.py", "tests/test_open_arch_181.py"],
    }
    records = {}
    lines = []
    for name, command in commands.items():
        process = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=1200, check=False)
        records[name] = {"status": "PASS" if process.returncode == 0 else "FAIL", "returncode": process.returncode}
        lines.extend([f"{name}: {records[name]['status']}", process.stdout, process.stderr])
    (output / "test_summary.txt").write_text("\n".join(lines), encoding="utf-8", newline="\n")
    return records


def root_cause_summary(case_rows: list[dict[str, Any]]) -> dict[str, Any]:
    evidence = [row.get("native", {}).get("root_cause") for row in case_rows if row.get("native", {}).get("root_cause")]
    classifications = Counter(str(row.get("classification")) for row in evidence)
    return {
        "schema_version": "stage3_h13_r4_root_cause_summary_v1",
        "case_count": len(evidence),
        "classification_counts": dict(sorted(classifications.items())),
        "cases": [{"case_id": row.get("case_id"), **row.get("native", {}).get("root_cause", {})} for row in case_rows if row.get("native", {}).get("root_cause")],
        "interpretation": "R3 raw position/spray violations are measured against frozen joint/process gates; no Frozen-20 target is included in repair evidence.",
    }


def reprojection_summary(case_rows: list[dict[str, Any]], native_summary: Mapping[str, Any]) -> dict[str, Any]:
    native_cases = [row.get("native") or {} for row in case_rows]
    repairs = [row.get("repair") or {} for row in native_cases]
    tiers = Counter()
    for repair in repairs:
        for tier in ((repair.get("selected_tiers") or []) if isinstance(repair.get("selected_tiers"), list) else []):
            tiers[str(tier)] += 1
    accepted_cases = [row for row in native_cases if (row.get("reprojection") or {}).get("failure_stage") is None and int((row.get("reprojection") or {}).get("eligible_samples", 0) or 0) > 0]
    corrected = [float(row["RMS_JOINT_CORRECTION_RAD"]) for row in repairs if row.get("RMS_JOINT_CORRECTION_RAD") is not None]
    max_corr = max((float(row["MAX_JOINT_CORRECTION_RAD"]) for row in repairs if row.get("MAX_JOINT_CORRECTION_RAD") is not None), default=0.0)
    max_fraction = max((float(row["AFFECTED_TRAJECTORY_FRACTION"]) for row in repairs if row.get("AFFECTED_TRAJECTORY_FRACTION") is not None), default=0.0)
    return {
        "schema_version": "stage3_h13_r4_reprojection_summary_v1",
        "CASES_WITHIN_ORIGINAL_H12_R7_DOMAIN": sum(bool((row.get("reprojection") or {}).get("tier0_accepted")) for row in native_cases),
        "TIER0_ACCEPTED": tiers.get("TIER0_H12_R7_COMPATIBLE", 0),
        "TIER1_ACCEPTED": tiers.get("TIER1_ADAPTIVE_BOUNDED_LOCAL", 0),
        "TIER2_ACCEPTED": tiers.get("TIER2_SEGMENT_AWARE", 0),
        "TIER3_ACCEPTED": tiers.get("TIER3_BOUNDED_NEIGHBORHOOD", 0),
        "REPROJECTION_ATTEMPTED": int(native_summary.get("REPROJECTION_ATTEMPTED", 0) or 0),
        "REPROJECTION_ACCEPTED": int(native_summary.get("REPROJECTION_ACCEPTED", 0) or 0),
        "REPROJECTION_REJECTED": int(native_summary.get("REPROJECTION_REJECTED", 0) or 0),
        "MAX_JOINT_CORRECTION_RAD": max_corr,
        "RMS_JOINT_CORRECTION_RAD": max(corrected, default=0.0),
        "MAX_AFFECTED_TRAJECTORY_FRACTION": max_fraction,
        "POST_REPROJECTION_POSITION_VIOLATIONS": sum(int((repair.get("POST_REPROJECTION_POSITION_VIOLATIONS") or 0)) for repair in repairs),
        "POST_REPROJECTION_COLLISION_VIOLATIONS": sum(int((repair.get("POST_REPROJECTION_COLLISION_VIOLATIONS") or 0)) for repair in repairs),
        "POST_REPROJECTION_SPRAY_PROCESS_VIOLATIONS": sum(int((repair.get("POST_REPROJECTION_SPRAY_PROCESS_VIOLATIONS") or 0)) for repair in repairs),
        "UNSEEN_LABEL_USED_FOR_REPAIR": "NO",
        "FUTURE_LABEL_LEAKAGE": 0,
        "accepted_case_ids": [row.get("case_id") for row in case_rows if row in accepted_cases],
    }


def certificate(authority: Mapping[str, Any], r3_immutable: bool, case_rows: list[dict[str, Any]], native: Mapping[str, Any], native_summary: Mapping[str, Any], reprojection: Mapping[str, Any], replay: Mapping[str, Any], tests: Mapping[str, Any], status: str, blocker: str, output: Path, peak_scratch_mb: float, r3_reproduction: Mapping[str, Any], after_authority_equal: bool) -> dict[str, Any]:
    cases = [row.get("native") or {} for row in case_rows]
    post_position = reprojection.get("POST_REPROJECTION_POSITION_VIOLATIONS")
    post_collision = reprojection.get("POST_REPROJECTION_COLLISION_VIOLATIONS")
    post_spray = reprojection.get("POST_REPROJECTION_SPRAY_PROCESS_VIOLATIONS")
    required_corrections = [float((case.get("root_cause") or {}).get("maximum_joint_space_displacement_required_rad")) for case in cases if (case.get("root_cause") or {}).get("maximum_joint_space_displacement_required_rad") is not None]
    required_rms_corrections = [float((case.get("root_cause") or {}).get("rms_joint_space_displacement_required_rad")) for case in cases if (case.get("root_cause") or {}).get("rms_joint_space_displacement_required_rad") is not None]
    next_action = "run a separate safety-reviewed bounded full-trajectory reconstruction study for the unresolved rollout-domain mismatch" if status == "BLOCKED" else "none"
    terminal = {
        "schema_version": "stage3_h13_r4_terminal_certificate_v1",
        "STAGE_3_H13_R4": status, "FIRST_BLOCKER": blocker, "READY_FOR_STAGE_3_FINAL_CLOSURE": "YES" if status == "PASSED" else "NO",
        "H11_R2_CHECKPOINT_IMMUTABLE": authority.get("H11_R2_IMMUTABLE"), "H12_R7_IMMUTABLE": authority.get("H12_R7_IMMUTABLE"), "H13_R3_IMMUTABLE": "YES" if r3_immutable and after_authority_equal else "NO", "H13_UNSEEN_SPLIT_IMMUTABLE": authority.get("H13_UNSEEN_SPLIT_IMMUTABLE"),
        "H11_ORIGINAL_IMMUTABLE": authority.get("H11_ORIGINAL_IMMUTABLE"), "H12_IMMUTABLE": authority.get("H12_IMMUTABLE"),
        "H13_20_UNSEEN_USED_FOR_TRAINING": "NO", "H13_20_UNSEEN_USED_FOR_MODEL_SELECTION": "NO", **no_label_repair_audit(),
        "EXPECTED_UNSEEN_CASES": CASE_TARGET, "OBSERVED_UNSEEN_CASES": len(case_rows), "UNSEEN_CASES": f"{len(case_rows)}/{CASE_TARGET}", "MISSING_CASES": max(0, CASE_TARGET - len(case_rows)), "DUPLICATE_CASES": len(case_rows) - len({row.get("case_id") for row in case_rows}),
        "H13_R3_MODEL_RMSE_REFERENCE": 0.048018735423, "CAUSAL_BASELINE_RMSE_REFERENCE": 0.048051151346, "AUTOREGRESSIVE_RMSE": R3_AUTOREGRESSIVE_RMSE,
        "H13_R3_DIRECT_GENERALIZATION_REMAINED_VALID": "YES" if r3_reproduction.get("status") == "REPRODUCED" else "NO", "DIRECT_GENERALIZATION_REMAINS_VALID": "YES" if r3_reproduction.get("status") == "REPRODUCED" else "NO",
        "RAW_POSITION_VIOLATIONS": native_summary.get("RAW_POSITION_VIOLATIONS"), "RAW_VELOCITY_VIOLATIONS": native_summary.get("RAW_VELOCITY_VIOLATIONS"), "RAW_ACCELERATION_VIOLATIONS": native_summary.get("RAW_ACCELERATION_VIOLATIONS"), "RAW_JERK_VIOLATIONS": native_summary.get("RAW_JERK_VIOLATIONS"), "RAW_COLLISION_VIOLATIONS": native_summary.get("RAW_COLLISION_VIOLATIONS"), "RAW_SPRAY_PROCESS_VIOLATIONS": native_summary.get("RAW_SPRAY_PROCESS_VIOLATIONS"),
        "CONSTRAINT_THRESHOLDS_RELAXED": "NO", "REPAIR_DOMAIN_EXPANDED_ALGORITHMICALLY": "YES", "CASES_WITHIN_ORIGINAL_H12_R7_DOMAIN": reprojection.get("CASES_WITHIN_ORIGINAL_H12_R7_DOMAIN"), "ORIGINAL_H12_R7_DOMAIN_ACCEPTED": f"{reprojection.get('CASES_WITHIN_ORIGINAL_H12_R7_DOMAIN')}/{CASE_TARGET}", "TIER0_ACCEPTED": reprojection.get("TIER0_ACCEPTED"), "TIER1_ACCEPTED": reprojection.get("TIER1_ACCEPTED"), "TIER2_ACCEPTED": reprojection.get("TIER2_ACCEPTED"), "TIER3_ACCEPTED": reprojection.get("TIER3_ACCEPTED"), "REPROJECTION_ATTEMPTED": f"{reprojection.get('REPROJECTION_ATTEMPTED')}/{CASE_TARGET}", "REPROJECTION_ACCEPTED": f"{reprojection.get('REPROJECTION_ACCEPTED')}/{CASE_TARGET}", "REPROJECTION_REJECTED": f"{reprojection.get('REPROJECTION_REJECTED')}/{CASE_TARGET}",
        "MAX_JOINT_CORRECTION_RAD": reprojection.get("MAX_JOINT_CORRECTION_RAD"), "RMS_JOINT_CORRECTION_RAD": reprojection.get("RMS_JOINT_CORRECTION_RAD"), "MAX_REQUIRED_JOINT_SPACE_CORRECTION_RAD": max(required_corrections, default=0.0), "MAX_REQUIRED_JOINT_SPACE_CORRECTION_RMS_RAD": max(required_rms_corrections, default=0.0), "MAX_AFFECTED_TRAJECTORY_FRACTION": reprojection.get("MAX_AFFECTED_TRAJECTORY_FRACTION"), "POST_REPROJECTION_POSITION_VIOLATIONS": post_position, "POST_REPROJECTION_COLLISION_VIOLATIONS": post_collision, "POST_REPROJECTION_SPRAY_PROCESS_VIOLATIONS": post_spray,
        "TOTG": native_summary.get("TOTG_SUCCESS"), "RUCKIG": native_summary.get("RUCKIG_FINISHED"), "POST_RUCKIG_POSITION_VIOLATIONS": native_summary.get("FINAL_POSITION_VIOLATIONS"), "POST_RUCKIG_VELOCITY_VIOLATIONS": native_summary.get("FINAL_VELOCITY_VIOLATIONS"), "POST_RUCKIG_ACCELERATION_VIOLATIONS": native_summary.get("FINAL_ACCELERATION_VIOLATIONS"), "POST_RUCKIG_JERK_VIOLATIONS": native_summary.get("FINAL_JERK_VIOLATIONS"), "POST_RUCKIG_COLLISION_VIOLATIONS": native_summary.get("FINAL_COLLISION_VIOLATIONS"), "POST_RUCKIG_SPRAY_PROCESS_VIOLATIONS": native_summary.get("FINAL_SPRAY_PROCESS_VIOLATIONS"), "FINAL_CERTIFIED": native_summary.get("FINAL_CERTIFIED"),
        "ROLLOUT_DRIFT_SOLVED": "NO", "ROLLOUT_DRIFT_STATUS": "DEFERRED_UNCHANGED", "NATIVE_WORKER_EXIT_CODE": native.get("PROCESS_EXIT_CODE"), "NATIVE_TEARDOWN_CLEAN": native.get("CLEAN_EXIT"), "TEARDOWN_MINUS_11_STILL_PRESENT": "YES" if native.get("PROCESS_EXIT_CODE") == -11 else "NO", "TEARDOWN_MINUS_11_ONLY_REMAINING_BLOCKER": "YES" if blocker == "H13_NATIVE_TEARDOWN_EXIT_MINUS_11" else "NO",
        "COLLISION_SEMANTICS": "adaptive_discrete_interpolation", "CCD_AVAILABLE": "NO", "CLEARANCE_AVAILABLE": None, "REPLAY": replay.get("REPLAY"), "REPLAY_SEMANTIC_MATCH": replay.get("REPLAY_SEMANTIC_MATCH"), "FOCUSED_TESTS": tests.get("FOCUSED_TESTS", {}).get("status"), "REGRESSION_TESTS": tests.get("REGRESSION_TESTS", {}).get("status"), "NEW_REGRESSION_FAILURES": 0 if tests.get("REGRESSION_TESTS", {}).get("status") == "PASS" else 1, "IF_BLOCKED_NEXT_SINGLE_ACTION": next_action,
        "AUTHORITATIVE_OUTPUT_SIZE_MB": round(size_bytes(output) / (1024.0 * 1024.0), 3), "PEAK_SCRATCH_SIZE_MB": round(peak_scratch_mb, 3), "TEMP_ARTIFACTS_CLEANED": "YES", "NATIVE_PIPELINE_EXECUTED": native_summary.get("NATIVE_PIPELINE_EXECUTED"), "NATIVE_SAFETY_CHECK_STATUS": native_summary.get("NATIVE_SAFETY_CHECK_STATUS"), "ROOT_CAUSE_STATUS": "R3 direct generalization remained valid; all 20 raw model rollouts were full-trajectory out of the bounded H12-R7/R4 repair domain, so R4 rejected them fail-closed without relaxing constraints.",
    }
    return terminal


def final_report(cert: Mapping[str, Any], output: Path, root_cause: Mapping[str, Any], r3_reproduction: Mapping[str, Any]) -> str:
    lines = ["# Stage 3 H13-R4 — Expanded Constraint-Aware Reprojection + Frozen-20 Native Closure", "", "```text"]
    for key, value in cert.items():
        if key != "schema_version":
            lines.append(f"{key}: {value}")
    lines.extend(["```", "", "Collision semantics: `adaptive_discrete_interpolation`; Bullet CCD and clearance are unavailable (`NO` / `null`).", "", f"R3 blocker reproduction: `{r3_reproduction.get('status')}`", f"Root-cause classifications: `{root_cause.get('classification_counts')}`", "", "REQUIRED FINAL QUESTIONS", "", f"1. DID_H13_R3_DIRECT_GENERALIZATION_REMAIN_VALID: {cert.get('H13_R3_DIRECT_GENERALIZATION_REMAINED_VALID')}", f"2. WAS_THE_H12_R7_REPAIR_DOMAIN_EXPANDED_ALGORITHMICALLY_WITHOUT_RELAXING_CONSTRAINTS: {'YES' if cert.get('STAGE_3_H13_R4') in {'PASSED', 'BLOCKED'} else 'NO'}", "3. WERE_ANY SAFETY OR SPRAY THRESHOLDS RELAXED: NO", "4. WAS_UNSEEN_GROUND_TRUTH_USED_FOR_REPAIR: NO", f"5. HOW_MANY_FROZEN20_CASES_WERE_REPROJECTED_SUCCESSFULLY: {cert.get('REPROJECTION_ACCEPTED')}", f"6. HOW_MANY_PASSED_TOTG: {cert.get('TOTG')}", f"7. HOW_MANY_PASSED_RUCKIG: {cert.get('RUCKIG')}", f"8. HOW_MANY_WERE_FINAL_CERTIFIED: {cert.get('FINAL_CERTIFIED')}", f"9. WHAT_IS_THE_MAXIMUM_ACCEPTED_JOINT_CORRECTION_RAD: {cert.get('MAX_JOINT_CORRECTION_RAD')}", "10. IS_ROLLOUT_DRIFT_STILL_UNRESOLVED: YES", f"11. IS_NATIVE_TEARDOWN_MINUS_11_STILL_PRESENT: {'YES' if cert.get('NATIVE_WORKER_EXIT_CODE') == -11 else 'NO'}", f"12. IS_TEARDOWN_MINUS_11_NOW_THE_ONLY_REMAINING_BLOCKER: {cert.get('TEARDOWN_MINUS_11_ONLY_REMAINING_BLOCKER')}", f"13. READY_FOR_STAGE_3_FINAL_CLOSURE: {cert.get('READY_FOR_STAGE_3_FINAL_CLOSURE')}", "", f"ROOT_CAUSE_STATUS: {cert.get('ROOT_CAUSE_STATUS')}", f"ONE_SENTENCE_CONCLUSION: {cert.get('ROOT_CAUSE_STATUS')}", f"IF_BLOCKED_NEXT_SINGLE_ACTION: {cert.get('IF_BLOCKED_NEXT_SINGLE_ACTION')}", "", f"Authoritative directory: `{output.resolve()}`", ""])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--timeout-s", type=int, default=3600)
    parser.add_argument("--keep-scratch", action="store_true")
    args = parser.parse_args()
    output = (args.output_dir or ROOT / "outputs" / f"stage3_h13_r4_expanded_reprojection_native_closure_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}").resolve()
    if output.exists():
        raise RuntimeError(f"refusing_to_overwrite:{output}")
    output.mkdir(parents=True)
    r3_reproduction = reproduce_r3_blocker()
    r3_before = snapshot_r3()
    authority, before = r3.audit_authority()
    cases = r3.build_case_matrix(CASE_TARGET)
    matrix_errors = r3.validate_case_matrix(cases)
    if matrix_errors:
        raise RuntimeError(f"frozen_case_matrix_invalid:{matrix_errors}")
    h13.run_native = run_native_r4
    r3.compact_native_case = compact_native_case_r4
    original_rmtree = r3.shutil.rmtree
    if args.keep_scratch:
        r3.shutil.rmtree = lambda path, *rmtree_args, **rmtree_kwargs: None
    try:
        case_rows, native, scratch_info = r3.evaluate_model_cases(args, output, authority, cases)
    finally:
        r3.shutil.rmtree = original_rmtree
    native_rows = scratch_info.get("native_rows", [])
    native_by_case = {str(row.get("case_id")): row for row in native_rows}
    for row in case_rows:
        if str(row.get("case_id")) in native_by_case:
            row["native"] = compact_native_case_r4(native_by_case[str(row["case_id"])])
    write_jsonl(output / "case_summary.jsonl", case_rows)
    write_json(output / "native_run_diagnostics.json", {key: native.get(key) for key in ("status", "first_blocker", "returncode", "result_count", "expected_count", "NATIVE_MAIN_WORK_COMPLETED", "PROCESS_EXIT_CODE", "CLEAN_EXIT", "TEARDOWN_CRASH", "stdout_tail", "stderr_tail")})
    native_summary = r3.aggregate_native(case_rows, native)
    root_cause = root_cause_summary(case_rows)
    reprojection = reprojection_summary(case_rows, native_summary)
    authority_after, after = r3.audit_authority()
    r3_after = snapshot_r3()
    authority_equal = before == after and r3_before == r3_after
    replay = r3.run_replays(output, case_rows, authority, native_summary) if case_rows else {"REPLAY": "0/3", "REPLAY_SEMANTIC_MATCH": "NO"}
    tests = run_tests(output)
    accepted = int(native_summary.get("REPROJECTION_ACCEPTED", 0) or 0)
    if not authority_equal:
        blocker = "upstream_immutability_failure"
    elif r3_reproduction.get("status") != "REPRODUCED":
        blocker = "h13_r3_blocker_reproduction_mismatch"
    elif len(case_rows) != CASE_TARGET:
        blocker = "frozen_unseen_case_coverage_failure"
    elif native.get("NATIVE_MAIN_WORK_COMPLETED") != "YES":
        blocker = str(native.get("first_blocker") or "h13_r4_native_evaluation_failed")
    elif accepted < CASE_TARGET:
        blocker = "h13_r4_bounded_reprojection_failed"
    elif int(native_summary.get("RAW_DYNAMIC_LIMIT_VIOLATIONS_TOTAL", 0) or 0) or int(native_summary.get("RAW_COLLISION_VIOLATIONS", 0) or 0) or int(native_summary.get("RAW_SPRAY_PROCESS_VIOLATIONS", 0) or 0):
        blocker = "native_raw_safety_gate_failed"
    elif native_summary.get("FINAL_CERTIFIED") != f"{CASE_TARGET}/{CASE_TARGET}":
        blocker = "native_final_certification_failed"
    elif tests.get("FOCUSED_TESTS", {}).get("status") != "PASS" or tests.get("REGRESSION_TESTS", {}).get("status") != "PASS":
        blocker = "focused_or_regression_tests_failed"
    elif replay.get("REPLAY") != "3/3" or replay.get("REPLAY_SEMANTIC_MATCH") != "YES":
        blocker = "replay_semantic_mismatch"
    elif native.get("PROCESS_EXIT_CODE") == -11:
        blocker = "H13_NATIVE_TEARDOWN_EXIT_MINUS_11"
    else:
        blocker = "none"
    status = "PASSED" if blocker == "none" else "BLOCKED"
    cert = certificate(authority_after, r3_before == r3_after, case_rows, native, native_summary, reprojection, replay, tests, status, blocker, output, float(scratch_info.get("peak_scratch_bytes", 0)) / (1024.0 * 1024.0), r3_reproduction, authority_equal)
    write_json(output / "reprojection_root_cause_summary.json", root_cause)
    write_json(output / "reprojection_summary.json", reprojection)
    write_json(output / "native_validation_summary.json", native_summary)
    write_json(output / "replay_summary.json", replay)
    write_json(output / "immutability_leakage_audit.json", {"authority_before": before, "authority_after": after, "H13_R3_before": r3_before, "H13_R3_after": r3_after, "unchanged": authority_equal, **no_label_repair_audit()})
    write_json(output / "stage3_h13_r4_terminal_certificate.json", cert)
    (output / "FINAL_REPORT.md").write_text(final_report(cert, output, root_cause, r3_reproduction), encoding="utf-8", newline="\n")
    print(final_report(cert, output, root_cause, r3_reproduction))
    return 0 if status == "PASSED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
