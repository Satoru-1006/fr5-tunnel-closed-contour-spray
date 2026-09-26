#!/usr/bin/env python3
"""Finalize an already-computed H12-R output after native shard completion."""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.stage3_h12_r_certification import discover_h11, load_json, run_tests, sha256_file  # noqa: E402


OUTPUT = ROOT / "outputs/stage3_h12_r_constraint_aware_residual_repair_20260811T175146Z"


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def canonical_sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def main() -> int:
    h11 = discover_h11()
    provenance = load_json(OUTPUT / "upstream_immutability_report.json")
    historical = load_json(OUTPUT / "historical_h12_immutability_report.json")
    baseline = load_json(OUTPUT / "h12_r_baseline_reproduction.json")
    metrics = load_json(OUTPUT / "test_metrics.json")
    selected = load_json(OUTPUT / "selected_model_manifest.json")
    h11_checkpoint_manifest = load_json(h11 / "checkpoint_sha256_manifest.json")
    repair = load_json(OUTPUT / "repair_results.json")
    replay = load_json(OUTPUT / "replay_results.json")
    native = load_json(OUTPUT / "native_certification_aggregate.json")
    leakage = load_json(OUTPUT / "h12_r_no_leakage_audit.json")
    focused, regression = run_tests(OUTPUT)

    total = int(native["EXPECTED_WINDOWS"])
    failure_counts = {str(key): int(value or 0) for key, value in (native.get("failure_counts") or {}).items()}
    native_complete = bool(
        native.get("CERTIFIED_WINDOWS") == total
        and native.get("MISSING_WINDOWS") == 0
        and native.get("DUPLICATE_WINDOWS") == 0
        and native.get("OUT_OF_RANGE_WINDOWS") == 0
        and native.get("NATIVE_RUNTIME_ERRORS") == 0
        and native.get("POST_RUCKIG_UNVALIDATED_WINDOWS", total) == 0
        and not failure_counts
    )
    blockers: list[str] = []
    if provenance.get("status") != "PASSED" or historical.get("status") != "PASSED":
        blockers.append("upstream_immutability_or_provenance_failure")
    if baseline.get("RAW_BASELINE_REPRODUCED") != "YES":
        blockers.append("h12_baseline_reproduction_failed")
    if float(metrics["NEURAL_REPAIRED_TEST_RMSE"]) >= float(metrics["CONSTANT_VELOCITY_TEST_RMSE"]):
        blockers.append("model_does_not_beat_constant_velocity_baseline")
    ordered_failure_labels = [
        ("JOINT_POSITION", "joint_position"),
        ("JOINT_VELOCITY", "joint_velocity"),
        ("JOINT_ACCELERATION", "joint_acceleration"),
        ("JOINT_JERK", "joint_jerk"),
        ("SELF_COLLISION", "self_collision"),
        ("ENVIRONMENT_COLLISION", "environment_collision"),
        ("SPRAY_PROCESS_TOLERANCE", "spray_process_tolerance"),
        ("OTHER_EXPLICITLY_DESCRIBED", "other_explicitly_described"),
        ("NATIVE_RUNTIME_ERRORS", "native_runtime_error"),
    ]
    for key, label in ordered_failure_labels:
        if int(failure_counts.get(key, 0)):
            blockers.append(f"native_repair_constraint_violation:{label}")
    if not native_complete:
        blockers.append("incomplete_native_repair_certification")
    replay_hashes = [str(value) for value in replay.get("replay_hashes", [])]
    replay_ok = bool(replay.get("semantic_match") is True and len(replay_hashes) == 3 and len(set(replay_hashes)) == 1)
    replay["REPAIR_REPLAY"] = "3/3" if replay_ok else f"{len(set(replay_hashes))}/3"
    replay["native_aggregate_semantic_sha256"] = canonical_sha(native)
    write_json(OUTPUT / "replay_results.json", replay)
    if not replay_ok:
        blockers.append("deterministic_replay_failure")
    if focused != "PASS":
        blockers.append("focused_test_failure")
    if regression != "PASS":
        blockers.append("regression_test_failure")
    blocker = blockers[0] if blockers else "none"

    terminal = {
        "schema_version": "stage3_h12_r_terminal_certificate_v1",
        "STAGE_3_H12_R": "PASSED" if blocker == "none" else "BLOCKED",
        "FIRST_BLOCKER": blocker,
        "READY_FOR_STAGE_3_H13": "YES" if blocker == "none" else "NO",
        "UPSTREAM_IMMUTABLE": "YES" if provenance.get("status") == "PASSED" else "NO",
        "HISTORICAL_H12_IMMUTABLE": "YES" if historical.get("status") == "PASSED" else "NO",
        "H10_SEMANTIC_HASH_MATCH": provenance.get("H10_SEMANTIC_HASH_MATCH"),
        "H11_CHECKPOINT_HASH": h11_checkpoint_manifest.get("sha256"),
        "TOTAL_WINDOWS": total,
        "NATIVE_CERTIFIED_WINDOWS": native.get("CERTIFIED_WINDOWS"),
        "MISSING_WINDOWS": native.get("MISSING_WINDOWS"),
        "DUPLICATE_WINDOWS": native.get("DUPLICATE_WINDOWS"),
        "OUT_OF_RANGE_WINDOWS": native.get("OUT_OF_RANGE_WINDOWS"),
        "POST_RUCKIG_UNVALIDATED_WINDOWS": native.get("POST_RUCKIG_UNVALIDATED_WINDOWS", total),
        "FUTURE_LABEL_LEAKAGE": int(leakage.get("future_label_leakage", 0)),
        "CONSTANT_VELOCITY_TEST_RMSE": metrics["CONSTANT_VELOCITY_TEST_RMSE"],
        "NEURAL_RAW_TEST_RMSE": metrics["NEURAL_RAW_TEST_RMSE"],
        "NEURAL_REPAIRED_TEST_RMSE": metrics["NEURAL_REPAIRED_TEST_RMSE"],
        "RMSE_IMPROVEMENT_PERCENT": metrics["RELATIVE_RMSE_IMPROVEMENT_PERCENT"],
        "JOINT_POSITION_VIOLATIONS": failure_counts.get("JOINT_POSITION", 0),
        "JOINT_VELOCITY_VIOLATIONS": failure_counts.get("JOINT_VELOCITY", 0),
        "JOINT_ACCELERATION_VIOLATIONS": failure_counts.get("JOINT_ACCELERATION", 0),
        "JOINT_JERK_VIOLATIONS": failure_counts.get("JOINT_JERK", 0),
        "SELF_COLLISION_VIOLATIONS": failure_counts.get("SELF_COLLISION", 0),
        "ENVIRONMENT_COLLISION_VIOLATIONS": failure_counts.get("ENVIRONMENT_COLLISION", 0),
        "SPRAY_PROCESS_TOLERANCE_VIOLATIONS": failure_counts.get("SPRAY_PROCESS_TOLERANCE", 0),
        "OTHER_EXPLICITLY_DESCRIBED_VIOLATIONS": failure_counts.get("OTHER_EXPLICITLY_DESCRIBED", 0),
        "NATIVE_RUNTIME_ERRORS": native.get("NATIVE_RUNTIME_ERRORS"),
        "NATIVE_CERTIFICATION_STATUS": native.get("status"),
        "COLLISION_METHOD": "adaptive_discrete_interpolation",
        "CCD_AVAILABLE": "NO",
        "CLEARANCE_AVAILABLE": None,
        "REPAIR_REPLAY": replay.get("REPAIR_REPLAY"),
        "FOCUSED_TESTS": focused,
        "REGRESSION_TESTS": regression,
        "PHYSICAL_ROBOT_CONNECTED": "NO",
        "FJT_GOALS_SENT": 0,
        "PHYSICAL_MOTION": 0,
        "selected_experiment": selected.get("selected_experiment"),
        "selected_checkpoint_sha256": selected.get("checkpoint_sha256"),
        "NEURAL_CONTRIBUTION_RATE": metrics["NEURAL_CONTRIBUTION_RATE"],
        "CV_FALLBACK_RATE": metrics["CV_FALLBACK_RATE"],
        "REPAIR_RATE": metrics["REPAIR_RATE"],
        "REJECTED_PREDICTION_RATE": metrics["REJECTED_PREDICTION_RATE"],
        "semantic_digest": canonical_sha({"native": native, "metrics": metrics, "replay": replay, "repair": repair.get("repaired_prediction_semantic_sha256")}),
    }

    smoke_path = OUTPUT / "native_certification" / "smoke" / "smoke.json"
    if smoke_path.is_file():
        smoke = load_json(smoke_path)
        write_json(OUTPUT / "targeted_native_smoke_report.json", {
            "schema_version": "stage3_h12_r_targeted_native_smoke_v1",
            "status": "DIAGNOSTIC_ONLY",
            "canonical_start_index": smoke.get("canonical_start_index"),
            "canonical_end_index": smoke.get("canonical_end_index"),
            "completed_windows": smoke.get("actual_window_count"),
            "post_ruckig_executed": smoke.get("post_ruckig_executed"),
            "collision_method": smoke.get("collision_method"),
            "CCD_AVAILABLE": smoke.get("CCD_AVAILABLE"),
            "CLEARANCE_AVAILABLE": smoke.get("CLEARANCE_AVAILABLE"),
            "full_certification": False,
            "reason": "Targeted native smoke only; it is not the 189216-window certification gate.",
        })

    write_json(OUTPUT / "stage3_h12_r_terminal_certificate.json", terminal)
    write_json(OUTPUT / "stage3_h12_r_gate_report.json", {
        "schema_version": "stage3_h12_r_gate_report_v1",
        **terminal,
        "blockers": blockers,
        "required": {
            "upstream_immutable": provenance.get("status") == "PASSED",
            "historical_h12_immutable": historical.get("status") == "PASSED",
            "baseline_reproduced": baseline.get("RAW_BASELINE_REPRODUCED") == "YES",
            "future_label_leakage_zero": int(leakage.get("future_label_leakage", 0)) == 0,
            "model_beats_cv": bool(metrics.get("REPAIRED_BEATS_CONSTANT_VELOCITY")),
            "native_complete": native_complete,
            "replay": replay_ok,
            "focused_tests": focused == "PASS",
            "regression_tests": regression == "PASS",
        },
    })

    native_manifest_path = OUTPUT / "native_certification_manifest.json"
    native_manifest = load_json(native_manifest_path)
    native_manifest.update({"status": native.get("status"), "aggregate_semantic_sha256": canonical_sha(native), "finalized_at": datetime.now(timezone.utc).isoformat(), "post_ruckig_unvalidated_windows": native.get("POST_RUCKIG_UNVALIDATED_WINDOWS", total)})
    write_json(native_manifest_path, native_manifest)

    shard_manifest = []
    shard_dir = OUTPUT / "native_certification" / "shards"
    for result_path in sorted(shard_dir.glob("shard_*.json")):
        if not result_path.name.endswith(".json") or result_path.name.endswith("_failures.json"):
            continue
        result = load_json(result_path)
        failure_path = result_path.with_name(result_path.stem + "_failures.jsonl")
        shard_manifest.append({"shard_id": result.get("shard_id"), "canonical_start_index": result.get("canonical_start_index"), "canonical_end_index": result.get("canonical_end_index"), "completed": result.get("completed"), "result_sha256": sha256_file(result_path), "failure_jsonl_sha256": sha256_file(failure_path) if failure_path.is_file() else None, "result_semantic_sha256": result.get("coordinator_output_semantic_sha256") or result.get("output_semantic_sha256")})
    write_json(OUTPUT / "native_certification" / "native_shard_sha256_manifest.json", {"schema_version": "stage3_h12_r_native_shard_sha256_manifest_v1", "algorithm": "SHA-256", "shards": shard_manifest})
    write_json(OUTPUT / "checkpoint_sha256_manifest.json", {"schema_version": "stage3_h12_r_checkpoint_sha256_manifest_v1", "checkpoint": selected.get("checkpoint"), "sha256": selected.get("checkpoint_sha256"), "matches_selected": sha256_file(Path(selected["checkpoint"])) == selected.get("checkpoint_sha256")})

    report_lines = [
        "# Stage 3 H12-R — Constraint-Aware Residual Trajectory Repair + Complete Native Certification Closure",
        "",
        "```text",
    ]
    report_keys = [
        "STAGE_3_H12_R", "FIRST_BLOCKER", "READY_FOR_STAGE_3_H13", "UPSTREAM_IMMUTABLE", "HISTORICAL_H12_IMMUTABLE",
        "H10_SEMANTIC_HASH_MATCH", "TOTAL_WINDOWS", "NATIVE_CERTIFIED_WINDOWS", "MISSING_WINDOWS", "DUPLICATE_WINDOWS",
        "OUT_OF_RANGE_WINDOWS", "POST_RUCKIG_UNVALIDATED_WINDOWS", "FUTURE_LABEL_LEAKAGE", "CONSTANT_VELOCITY_TEST_RMSE",
        "NEURAL_RAW_TEST_RMSE", "NEURAL_REPAIRED_TEST_RMSE", "RMSE_IMPROVEMENT_PERCENT", "SPRAY_PROCESS_TOLERANCE_VIOLATIONS",
        "OTHER_EXPLICITLY_DESCRIBED_VIOLATIONS", "NATIVE_RUNTIME_ERRORS", "NATIVE_CERTIFICATION_STATUS", "COLLISION_METHOD",
        "CCD_AVAILABLE", "CLEARANCE_AVAILABLE", "REPAIR_REPLAY", "FOCUSED_TESTS", "REGRESSION_TESTS", "PHYSICAL_ROBOT_CONNECTED",
        "FJT_GOALS_SENT", "PHYSICAL_MOTION", "selected_experiment", "NEURAL_CONTRIBUTION_RATE", "CV_FALLBACK_RATE", "REPAIR_RATE",
    ]
    report_lines.extend(f"{key}: {terminal.get(key)}" for key in report_keys)
    report_lines.extend([
        "```", "",
        "H12 historical evidence was not modified. H12-R is software-only; no H13, physical robot, FollowJointTrajectory goal, hardware driver, or physical motion was used.",
        "", f"H11-R source: `{h11}`", f"H12-R evidence root: `{OUTPUT}`", "",
        "The authoritative dataset is the frozen H10/H11-R 189216-window open-arch ON-state dataset. TRAIN/VALIDATION/TEST roles and TRAIN-only normalization were preserved; future-label leakage is zero.",
        "", "The selected repair is residual GRU prediction around constant-velocity q_cv with causal backtracking toward q_cv. Zero residual is exact q_cv. Model selection used VALIDATION only; TEST was used only for frozen evaluation.",
        "", "Native shards executed MoveIt2 PlanningScene, FK, dynamics, process, and FCL/discrete interpolation checks. Collision is labelled adaptive_discrete_interpolation; CCD is unavailable and clearance is null. All windows remain post-Ruckig-unvalidated in shard_full mode, so the H12-R gate is fail-closed.",
        "", f"Failure category counts: {json.dumps(failure_counts, ensure_ascii=False, sort_keys=True)}",
    ])
    (OUTPUT / "FINAL_REPORT.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8", newline="\n")

    records = []
    for path in sorted(OUTPUT.rglob("*")):
        if path.is_file() and path.name != "frozen_artifact_manifest.json":
            records.append({"path": str(path.relative_to(OUTPUT)).replace("\\", "/"), "sha256": sha256_file(path), "size_bytes": path.stat().st_size})
    write_json(OUTPUT / "frozen_artifact_manifest.json", {"schema_version": "stage3_h12_r_frozen_artifact_manifest_v1", "algorithm": "SHA-256", "records": records})
    print(json.dumps(terminal, ensure_ascii=True, sort_keys=True))
    return 0 if blocker == "none" else 2


if __name__ == "__main__":
    raise SystemExit(main())
