#!/usr/bin/env python3
"""Finalize an already completed H12 Python-side run with native evidence."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.stage3_h12_certification import (
    ROOT,
    discover_checkpoint,
    discover_h11_root,
    dump,
    freeze_upstream,
    h10_marker,
    load,
    run_native,
    sha256_file,
    write_manifest,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output = Path(args.output_dir).resolve()
    h11_root = discover_h11_root()
    checkpoint, checkpoint_sha, _ = discover_checkpoint(h11_root)
    freeze, metadata = freeze_upstream(h11_root, checkpoint, checkpoint_sha)
    h10_root = metadata["h10_root"]
    dump(output / "upstream_immutability_report.json", freeze)
    native_input = output / "stage3_h12_native_repaired_input.npz"
    native = run_native(output, native_input, h11_root / "h11_window_manifest.json", h10_root, h10_marker(h10_root, "h6_validation"), h10_marker(h10_root, "h6_segments"), h10_marker(h10_root, "process_contract"), h10_marker(h10_root, "fixture_mesh"), h10_marker(h10_root, "runtime_limits"))
    dump(output / "stage3_h12_native_validation.json", native)
    dump(output / "stage3_h12_collision_validation.json", {"schema_version": "stage3_h12_collision_validation_v1", "COLLISION_METHOD": native.get("COLLISION_METHOD", "adaptive_discrete_interpolation"), "CCD_AVAILABLE": native.get("CCD_AVAILABLE", "NO"), "CLEARANCE_AVAILABLE": native.get("CLEARANCE_AVAILABLE"), "POST_REPAIR_SELF_COLLISION_VIOLATIONS": native.get("POST_REPAIR_SELF_COLLISION_VIOLATIONS"), "POST_REPAIR_ENVIRONMENT_COLLISION_VIOLATIONS": native.get("POST_REPAIR_ENVIRONMENT_COLLISION_VIOLATIONS"), "POST_REPAIR_COLLISION_VIOLATIONS": native.get("POST_REPAIR_COLLISION_VIOLATIONS"), "status": native.get("status")})
    dump(output / "stage3_h12_process_validation.json", {"schema_version": "stage3_h12_process_validation_v1", "POST_REPAIR_PROCESS_TOLERANCE_VIOLATIONS": native.get("POST_REPAIR_PROCESS_TOLERANCE_VIOLATIONS"), "native_backend_executed": native.get("native_backend_executed", False), "fk_executed": native.get("fk_executed", False), "scope": native.get("scope"), "status": native.get("status")})
    terminal = load(output / "stage3_h12_terminal_certificate.json")
    repaired = load(output / "stage3_h12_repaired_metrics.json")
    leakage = load(output / "stage3_h12_no_leakage_audit.json")
    determinism = load(output / "stage3_h12_determinism_report.json")
    focused = terminal.get("FOCUSED_TESTS", "BLOCKED")
    regression = terminal.get("REGRESSION", "BLOCKED")
    blockers = []
    if freeze.get("status") != "PASSED":
        blockers.append(str(freeze.get("first_blocker") or "upstream_artifact_mutated"))
    if terminal.get("RAW_BASELINE_REPRODUCED") != "YES":
        blockers.append("h11_raw_baseline_reproduction_failed")
    if leakage.get("REPAIR_LABEL_LEAKAGE_VIOLATIONS") != 0:
        blockers.append("repair_future_label_leakage")
    if not (native.get("status") == "PASSED" and native.get("complete_native_scope") is True and native.get("post_ruckig_executed") is True):
        blockers.append(str(native.get("first_blocker") or "incomplete_native_repair_certification"))
    if repaired.get("REPAIRED_MODEL_BEATS_CONST_VELOCITY_TEST") != "YES":
        blockers.append("repair_destroyed_learned_predictive_advantage")
    if determinism.get("DETERMINISTIC_REPAIR") != "YES":
        blockers.append("deterministic_repair_replay_failed")
    if focused != "PASSED":
        blockers.append("focused_tests_failed")
    if regression != "PASSED":
        blockers.append("relevant_regression_failed")
    blocker = blockers[0] if blockers else "none"
    terminal.update({
        "STAGE_3_H12": "PASSED" if blocker == "none" else "BLOCKED", "FIRST_BLOCKER": blocker,
        "H7_IMMUTABLE": "YES" if freeze["groups"].get("H7_EVIDENCE") else "NO", "H8_R_IMMUTABLE": "YES" if freeze["groups"].get("H8_R_EVIDENCE") else "NO", "HISTORICAL_H8_IMMUTABLE": "YES" if freeze["groups"].get("HISTORICAL_H8_EVIDENCE") else "NO", "H9_IMMUTABLE": "YES" if freeze["groups"].get("H9_EVIDENCE") else "NO", "H10_IMMUTABLE": "YES" if freeze["groups"].get("H10_EVIDENCE") else "NO", "H11_R_IMMUTABLE": "YES" if freeze["groups"].get("H11_R_EVIDENCE") else "NO", "H11_CHECKPOINT_IMMUTABLE": "YES" if freeze["groups"].get("H11_CHECKPOINT") else "NO", "H10_SEMANTIC_HASH_MATCH": freeze.get("H10_SEMANTIC_HASH_MATCH", "NO"),
        "POST_REPAIR_POSITION_LIMIT_VIOLATIONS": native.get("POST_REPAIR_POSITION_LIMIT_VIOLATIONS"), "POST_REPAIR_VELOCITY_LIMIT_VIOLATIONS": native.get("POST_REPAIR_VELOCITY_LIMIT_VIOLATIONS"), "POST_REPAIR_ACCELERATION_LIMIT_VIOLATIONS": native.get("POST_REPAIR_ACCELERATION_LIMIT_VIOLATIONS"), "POST_REPAIR_JERK_LIMIT_VIOLATIONS": native.get("POST_REPAIR_JERK_LIMIT_VIOLATIONS"), "POST_REPAIR_HARD_CONSTRAINT_VIOLATIONS": native.get("POST_REPAIR_HARD_CONSTRAINT_VIOLATIONS"), "POST_REPAIR_SELF_COLLISION_VIOLATIONS": native.get("POST_REPAIR_SELF_COLLISION_VIOLATIONS"), "POST_REPAIR_ENVIRONMENT_COLLISION_VIOLATIONS": native.get("POST_REPAIR_ENVIRONMENT_COLLISION_VIOLATIONS"), "POST_REPAIR_COLLISION_VIOLATIONS": native.get("POST_REPAIR_COLLISION_VIOLATIONS"), "POST_REPAIR_PROCESS_TOLERANCE_VIOLATIONS": native.get("POST_REPAIR_PROCESS_TOLERANCE_VIOLATIONS"), "NATIVE_REPAIRED_TRAJECTORY_VALIDATION": "PASSED" if native.get("status") == "PASSED" and native.get("complete_native_scope") is True and native.get("post_ruckig_executed") is True else "BLOCKED", "READY_FOR_STAGE_3_H13": "YES" if blocker == "none" else "NO",
    })
    gate = load(output / "stage3_h12_gate_report.json")
    gate.update(terminal)
    gate["required"] = {**gate.get("required", {}), "upstream_immutable": freeze.get("status") == "PASSED", "native_full_scope": native.get("status") == "PASSED" and native.get("complete_native_scope") is True and native.get("post_ruckig_executed") is True}
    gate["blockers"] = blockers
    dump(output / "stage3_h12_terminal_certificate.json", terminal)
    dump(output / "stage3_h12_gate_report.json", gate)
    report = ["# Stage 3 H12 — Constraint-Aware Learned Trajectory Repair & Native Certification", "", "```text"]
    report.extend(f"{key}: {terminal.get(key)}" for key in ("STAGE_3_H12", "FIRST_BLOCKER", "H7_IMMUTABLE", "H8_R_IMMUTABLE", "HISTORICAL_H8_IMMUTABLE", "H9_IMMUTABLE", "H10_IMMUTABLE", "H11_R_IMMUTABLE", "H11_CHECKPOINT_IMMUTABLE", "RAW_BASELINE_REPRODUCED", "REPAIR_LABEL_LEAKAGE_VIOLATIONS", "POST_REPAIR_HARD_CONSTRAINT_VIOLATIONS", "POST_REPAIR_COLLISION_VIOLATIONS", "POST_REPAIR_PROCESS_TOLERANCE_VIOLATIONS", "REPAIRED_TEST_RMSE_RAD", "REPAIRED_MODEL_BEATS_CONST_VELOCITY_TEST", "NATIVE_REPAIRED_TRAJECTORY_VALIDATION", "REPAIR_REPLAY", "DETERMINISTIC_REPAIR", "FOCUSED_TESTS", "REGRESSION", "READY_FOR_STAGE_3_H13"))
    report.extend(["```", "", "H12 is software-only. No H13 execution and no physical FollowJointTrajectory goal were attempted.", "", f"Native evidence: `{output / 'stage3_h12_native_validation.json'}`", ""])
    (output / "FINAL_REPORT.md").write_text("\n".join(report), encoding="utf-8", newline="\n")
    write_manifest(output)
    print(__import__("json").dumps(terminal, ensure_ascii=True, sort_keys=True))
    return 0 if blocker == "none" else 2


if __name__ == "__main__":
    raise SystemExit(main())
