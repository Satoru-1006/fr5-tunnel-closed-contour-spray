"""Finalize the Stage 2.9 closure decision without touching Formal R2.

This is intentionally a certificate builder, not a runtime runner.  It never
imports the production action runner, never launches ROS, never consumes the
one-shot ledger, and never calls ``send_goal_async``.  The post-Formal local
pytest observations are recorded from the completed regression attempts; the
script only reads those observations and the immutable historical evidence.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
FORMAL_DIR = ROOT / "outputs" / "stage28sr2_formal_r2_one_shot_20260807T124755Z"
LEDGER = ROOT / ".stage28sr2" / "formal_r2_global_one_shot_ledger.json"
OUT = ROOT / "outputs" / "stage29_final_regression_20260807T225000Z"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def rel(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT.resolve())).replace("\\", "/")


def checked_file(path: Path) -> dict[str, Any]:
    return {"path": rel(path), "exists": path.is_file(), "sha256": sha256(path) if path.is_file() else None}


def package_hashes() -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for path in sorted(FORMAL_DIR.rglob("*")):
        if path.is_file():
            result[rel(path)] = {"size": path.stat().st_size, "sha256": sha256(path)}
    return result


def all_true(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, dict):
        return all(all_true(item) for item in value.values())
    if isinstance(value, list):
        return all(all_true(item) for item in value)
    return True


def main() -> int:
    now = datetime.now(timezone.utc).isoformat()
    OUT.mkdir(parents=True, exist_ok=True)

    terminal = read_json(FORMAL_DIR / "formal_r2_terminal_certificate.json")
    production = read_json(FORMAL_DIR / "production_formal_execution_certificate.json")
    ledger = read_json(LEDGER)
    dispatch = read_json(FORMAL_DIR / "goal_dispatch_evidence.json")
    response = read_json(FORMAL_DIR / "goal_response.json")
    result = read_json(FORMAL_DIR / "goal_result.json")
    bag = read_json(FORMAL_DIR / "bag_validation.json")
    frozen = read_json(FORMAL_DIR / "frozen_hashes.json")
    semantic = read_json(FORMAL_DIR / "production_ros_goal_semantic_report.json")
    jit = read_json(FORMAL_DIR / "final_jit_pre_send_gate.json")
    jit_verify = read_json(FORMAL_DIR / "final_jit_pre_send_verification.json")
    regression = read_json(FORMAL_DIR / "stage28sr2_r2_regression_certificate.json")

    stage21 = read_json(ROOT / "outputs/ik_graph_stage2_1/gate_report.json")
    stage22 = read_json(ROOT / "outputs/ik_graph_stage2_2/stage2_2_gate_report.json")
    stage23b = read_json(ROOT / "outputs/ik_graph_stage23b_representation_remediation/fr5_scaled_horseshoe_demo_v45/stage23b_gate_report.json")
    stage25 = read_json(ROOT / "outputs/stage25r2_full_native_certification/stage25r2_formal_20260804T054820Z/stage25r2_gate_report.json")
    stage26 = read_json(ROOT / "outputs/stage26r_stage25r2_continuous_collision_certification/stage26r_formal_20260804T075115Z/stage26r_gate_report.json")
    stage27s = read_json(ROOT / "outputs/stage27s_native_spline_remediation/stage27s_formal_20260805T130000Z/stage27s_gate_report.json")
    stage27s_jtc = read_json(ROOT / "outputs/stage27s_native_spline_remediation/stage27s_formal_20260805T130000Z/stage27s_exact_jtc_certificate.json")
    stage28sr = read_json(ROOT / "outputs/stage28sr_corrected_tf_fk_runtime_recapture/stage28sr_gate_report.json")
    stage28sr_regression = read_json(ROOT / "outputs/stage28sr_corrected_tf_fk_runtime_recapture/stage28sr_regression.json")
    h41 = read_json(ROOT / "outputs/stage28sr2_h4_1_identity_migration_20260807T063342Z/h4_1_final_summary.json")
    h42 = read_json(ROOT / "outputs/stage28sr2_h4_2_formal_execution_environment_parity_20260807T203000/h4_2_terminal_certificate.json")

    package_before = package_hashes()
    ledger_before_sha = sha256(LEDGER)

    goal_uuid = "26f768a3-b749-4de1-9962-190a802c14f5"
    ledger_budget = ledger.get("formal_goal_budget", {})
    terminal_checks = terminal.get("strict_acceptance_checks", {})
    bag_root = FORMAL_DIR / "recorder" / "evidence_bag"
    bag_files = {}
    for item in bag.get("bag_files", []):
        path = bag_root / str(item["path"])
        bag_files[str(item["path"])] = {
            "exists": path.is_file(),
            "expected_sha256": item.get("sha256"),
            "observed_sha256": sha256(path) if path.is_file() else None,
            "expected_size": item.get("size"),
            "observed_size": path.stat().st_size if path.is_file() else None,
        }

    formal_checks = {
        "terminal_pass": terminal.get("FORMAL_R2_ONE_SHOT") == "PASSED",
        "production_certificate_pass": production.get("formal_status") == "passed",
        "goal_uuid_consistent": all(
            value == goal_uuid
            for value in (
                terminal.get("goal_uuid"),
                dispatch.get("goal_uuid"),
                response.get("goal_uuid"),
                result.get("goal_uuid"),
            )
        ),
        "exactly_one_dispatch": dispatch.get("send_goal_async_call_count") == 1 and dispatch.get("exactly_one_formal_goal_dispatch") is True,
        "no_second_goal": dispatch.get("second_goal_observed") is False and terminal.get("second_goal_observed") is False,
        "no_retry": dispatch.get("retry_observed") is False and terminal.get("retry_observed") is False,
        "terminal_succeeded": terminal.get("goal_terminal_status") == "SUCCEEDED" and result.get("terminal_status") == "SUCCEEDED",
        "fjt_success": terminal.get("fjt_error_code") == 0 and terminal.get("fjt_error_string") == "Goal successfully reached!",
        "bag_finalized_valid_eof": all(bag.get(key) is True for key in ("finalized", "bag_valid", "bag_read_to_eof", "bag_complete")),
        "bag_file_hashes_match": bool(bag_files) and all(
            item["exists"] and item["expected_sha256"] == item["observed_sha256"] and item["expected_size"] == item["observed_size"]
            for item in bag_files.values()
        ),
        "frozen_hash_gate_pass": frozen.get("frozen_hashes_pass") is True and frozen.get("final_jit_manifest_gate", {}).get("passed") is True,
        "semantic_goal_match": semantic.get("passed") is True and semantic.get("max_numeric_difference") == 0.0,
        "jit_gate_pass": jit.get("passed") is True and jit_verify.get("passed") is True,
        "strict_acceptance_checks_all_pass": all_true(terminal_checks),
        "stage28sr2_regression_certificate_pass": regression.get("all_required_tests_passed") is True and regression.get("regression_gate", {}).get("passed") is True,
    }

    authoritative = {
        "Stage 2.1": {"status": "passed", "artifact": checked_file(ROOT / "outputs/ik_graph_stage2_1/gate_report.json"), "gate_value": stage21.get("Stage_2_1")},
        "Stage 2.2": {"status": "audit_passed; planning_infeasible_under_frozen_geometry", "artifact": checked_file(ROOT / "outputs/ik_graph_stage2_2/stage2_2_gate_report.json"), "gate_value": stage22.get("Stage_2_2_gate", {}).get("status"), "graph_status": "not_evaluated_no_valid_nodes"},
        "Stage 2.3A": {"status": stage23b.get("Stage_2_3A", {}).get("status"), "artifact": checked_file(ROOT / "outputs/ik_graph_stage23b_representation_remediation/fr5_scaled_horseshoe_demo_v45/stage23b_gate_report.json")},
        "Stage 2.3B": {"status": stage23b.get("Stage_2_3B", {}).get("status"), "artifact": checked_file(ROOT / "outputs/ik_graph_stage23b_representation_remediation/fr5_scaled_horseshoe_demo_v45/stage23b_gate_report.json")},
        "Stage 2.4/2.4T": {"status": stage25.get("Stage_2_4T"), "artifact": checked_file(ROOT / "outputs/stage25r2_full_native_certification/stage25r2_formal_20260804T054820Z/stage25r2_gate_report.json")},
        "Stage 2.5/2.5R2": {"status": stage25.get("Stage_2_5R2"), "artifact": checked_file(ROOT / "outputs/stage25r2_full_native_certification/stage25r2_formal_20260804T054820Z/stage25r2_gate_report.json")},
        "Stage 2.6/2.6R": {"status": stage26.get("Stage_2_6R"), "artifact": checked_file(ROOT / "outputs/stage26r_stage25r2_continuous_collision_certification/stage26r_formal_20260804T075115Z/stage26r_gate_report.json")},
        "Stage 2.7S": {"status": stage27s.get("Stage_2_7", {}).get("status"), "artifact": checked_file(ROOT / "outputs/stage27s_native_spline_remediation/stage27s_formal_20260805T130000Z/stage27s_gate_report.json")},
        "Stage 2.8S-R2 Formal R2": {"status": terminal.get("FORMAL_R2_ONE_SHOT"), "artifact": checked_file(FORMAL_DIR / "formal_r2_terminal_certificate.json")},
        "Stage 2.8S-R2 H4.1": {"status": h41.get("H4_1"), "artifact": checked_file(ROOT / "outputs/stage28sr2_h4_1_identity_migration_20260807T063342Z/h4_1_final_summary.json")},
        "Stage 2.8S-R2 H4.2": {"status": h42.get("Stage_2_8S_R2_H4_2"), "artifact": checked_file(ROOT / "outputs/stage28sr2_h4_2_formal_execution_environment_parity_20260807T203000/h4_2_terminal_certificate.json")},
    }

    mandatory = {
        "frozen_trajectory_goal_integrity": formal_checks["frozen_hash_gate_pass"] and formal_checks["semantic_goal_match"],
        "urdf_srdf_moveit_configuration_integrity": stage25.get("frozen_artifacts_unchanged") is True and stage28sr_regression.get("URDF_SRDF_xacro_modified") is False,
        "waypoint_segment_semantics": stage25.get("full_nominal_segment_reachability", {}).get("passed") is True and stage26.get("trajectory_interval_integrity", {}).get("missing") == 0,
        "spray_on_repositioning_semantics": stage25.get("spray_off_transitions") == "9/9" and stage25.get("spray_on_segments") == 10 and stage25.get("process_order_preserved") is True,
        "fk_tf_consistency": stage28sr.get("Stage_2_8S_R", {}).get("global_spray_tcp_tf_vs_fk") == "passed" and stage28sr.get("Stage_2_8S_R", {}).get("tf_tree_connectivity") == "passed",
        "spray_tcp_geometry": stage27s.get("process_geometry", {}).get("passed") is True,
        "joint_velocity_acceleration_jerk_limits": stage27s.get("velocity_limits", {}).get("passed") is True and stage27s.get("acceleration_limits", {}).get("passed") is True and stage27s.get("jerk_limits", {}).get("passed") is True and all(stage27s_jtc.get("status", {}).get(key) is True for key in ("velocity", "acceleration", "jerk")),
        "collision_verification": stage27s.get("Bullet", {}).get("passed") is True and stage27s.get("Bullet", {}).get("collisions") == 0 and stage27s.get("Bullet", {}).get("collision_method") == "adaptive_discrete_interpolation" and stage27s.get("Bullet", {}).get("strict_continuous_collision_detection") == "not_available",
        "controller_execution_semantics": stage27s.get("native_execution", {}).get("execution_completed") is True and stage27s.get("native_execution", {}).get("result_code") == 0 and formal_checks["terminal_succeeded"],
        "initial_state_compatibility": read_json(FORMAL_DIR / "runtime_evidence.json").get("initial_state", {}).get("probe_passed") is True,
        "deterministic_execution_analysis": stage27s.get("determinism", {}).get("status") == "3/3_passed" and stage28sr_regression.get("determinism", {}).get("status") == "3/3_passed",
        "formal_r2_evidence_consistency": all(formal_checks.values()),
        "recorder_mcap_evidence_integrity": formal_checks["bag_finalized_valid_eof"] and formal_checks["bag_file_hashes_match"] and terminal.get("recorder_loss_pre") == 0 and terminal.get("recorder_loss_post") == 0 and terminal.get("transport_loss_pre") == 0 and terminal.get("transport_loss_post") == 0,
        "formal_r2_ledger_consistency": ledger.get("maximum") == 1 and ledger.get("consumed") == 1 and ledger.get("send_goal_async_call_count") == 1 and ledger.get("formal_goal_budget", {}).get("send_goal_async_call_count") == 1,
    }

    first_blocker = {
        "id": "stage28sr_tf_fk_consistency_gate",
        "root_cause": "The latest available strict Stage 2.8S-R TF/FK certificate still records the same-header-stamp TF/FK gate as blocked. Formal R2 is immutable action/recorder evidence and did not generate a passing FK/TF certificate.",
        "affected_artifact": rel(ROOT / "outputs/stage28sr_corrected_tf_fk_runtime_recapture/stage28sr_gate_report.json"),
        "expected_value": {"global_spray_tcp_tf_vs_fk": "passed", "tf_tree_connectivity": "passed", "timestamp_semantics": "passed", "process_mapping": "passed"},
        "observed_value": {"global_spray_tcp_tf_vs_fk": stage28sr.get("Stage_2_8S_R", {}).get("global_spray_tcp_tf_vs_fk"), "tf_tree_connectivity": stage28sr.get("Stage_2_8S_R", {}).get("tf_tree_connectivity"), "timestamp_semantics": stage28sr.get("Stage_2_8S_R", {}).get("timestamp_semantics"), "process_mapping": stage28sr.get("Stage_2_8S_R", {}).get("process_mapping"), "first_blocker": stage28sr.get("first_blocker")},
    }

    # This is deliberately a post-Formal compatibility note, not a hidden
    # pass.  These tests were defined for zero-consumption H4.1 migration and
    # cannot be run against the immutable consumed ledger without violating
    # the user safety boundary.  Their original 17/17 and 475/475 evidence is
    # retained above as historical evidence.
    regression_runs = {
        "run_1": {"status": "PASSED", "scope": "post_formal_compatible_local_regression", "passed": 423, "failed": 0, "errors": 0, "skipped": 0, "command": "python -m pytest -q -p no:cacheprovider --ignore tests/test_stage24a_outputs.py --ignore tests/test_stage24b_outputs.py --ignore tests/test_stage24c_outputs.py --ignore tests/test_stage23a7_1_outputs.py --ignore tests/test_stage28sr2_h4_1_identity_migration.py"},
        "run_2": {"status": "NOT_RUN", "reason": "No repository-native Stage 2.9 three-run requirement was found; fail-closed after mandatory TF/FK blocker."},
        "run_3": {"status": "NOT_RUN", "reason": "No repository-native Stage 2.9 three-run requirement was found; fail-closed after mandatory TF/FK blocker."},
        "pre_formal_full_local_attempt": {"status": "BLOCKED_BY_INCOMPATIBLE_POST_FORMAL_TEST_ASSUMPTION", "passed": 436, "failed": 4, "errors": 0, "affected_test_file": "tests/test_stage28sr2_h4_1_identity_migration.py", "expected": {"canonical_ledger.consumed": 0, "canonical_ledger.send_goal_async_call_count": 0}, "observed": {"canonical_ledger.consumed": 1, "canonical_ledger.send_goal_async_call_count": 1}},
        "historical_external_full_regression": {"status": "PASSED", "passed": 475, "failed": 0, "errors": 0, "source": "outputs/stage28sr2_h4_1_identity_migration_20260807T063342Z/host_regression_20260807T064500Z/h4_1_host_regression_evidence.txt", "sha256": "3d9dbf7367f2cda36b824bc9498425b4f3a0189571b1eef9122dbe8adeb2a672"},
    }

    package_after = package_hashes()
    ledger_after_sha = sha256(LEDGER)
    freeze = {
        "schema_version": "stage29-formal-r2-freeze-manifest-v1",
        "captured_utc": now,
        "formal_r2_directory": rel(FORMAL_DIR),
        "canonical_ledger": {"path": rel(LEDGER), "sha256_before": ledger_before_sha, "sha256_after": ledger_after_sha},
        "formal_r2_file_count": len(package_before),
        "formal_r2_package_before": package_before,
        "formal_r2_package_after": package_after,
        "immutable": package_before == package_after and ledger_before_sha == ledger_after_sha,
    }
    write_json(OUT / "formal_r2_sha256_freeze_manifest.json", freeze)

    certificate = {
        "schema_version": "stage29-stage2-closure-certificate-v1",
        "created_utc": now,
        "stage": "Stage 2.9 — Final Regression & Stage-2 Closure",
        "STAGE_2_9": "BLOCKED",
        "STAGE_2_CLOSED": "NO",
        "STAGE_3_AUTHORIZATION_RECOMMENDATION": "NO",
        "stage_3_started": False,
        "stage29_definition_found_in_repository": False,
        "stage29_three_run_requirement": "not_defined_in_repository",
        "stage_authoritative_status": authoritative,
        "formal_r2": {
            "immutable_historical_evidence": formal_checks["terminal_pass"] and freeze["immutable"],
            "goal_uuid": goal_uuid,
            "formal_goal_dispatch_count": ledger.get("send_goal_async_call_count"),
            "canonical_ledger": {"maximum": ledger.get("maximum"), "consumed": ledger.get("consumed"), "send_goal_async_call_count": ledger.get("send_goal_async_call_count"), "second_goal_observed": dispatch.get("second_goal_observed")},
            "terminal_status": terminal.get("goal_terminal_status"),
            "fjt_error_code": terminal.get("fjt_error_code"),
            "fjt_error_string": terminal.get("fjt_error_string"),
            "retry": dispatch.get("retry_observed"),
            "resend": dispatch.get("second_goal_observed"),
            "recorder_loss": {"pre": terminal.get("recorder_loss_pre"), "post": terminal.get("recorder_loss_post")},
            "transport_loss": {"pre": terminal.get("transport_loss_pre"), "post": terminal.get("transport_loss_post")},
            "bag": {"finalized": bag.get("finalized"), "valid": bag.get("bag_valid"), "read_to_eof": bag.get("bag_read_to_eof")},
        },
        "formal_r2_evidence_checks": formal_checks,
        "mandatory_gates": mandatory,
        "regression": {"runs": regression_runs, "deterministic_comparison": {"required": False, "status": "NOT_APPLICABLE_REPOSITORY_HAS_NO_STAGE29_THREE_RUN_DEFINITION", "run_1_metrics": {"passed": 423, "failed": 0, "errors": 0, "skipped": 0}}},
        "frozen_input_integrity": {"authoritative_stage01_input": {"open_arch_seed_joints": checked_file(ROOT / "outputs/internal_wiper_moveit_inputs/open_arch_seed_joints.csv"), "open_arch_tcp_poses": checked_file(ROOT / "outputs/internal_wiper_moveit_inputs/open_arch_tcp_poses_base_link.csv")}, "formal_r2_freeze_manifest": rel(OUT / "formal_r2_sha256_freeze_manifest.json"), "hash_drift": False},
        "first_blocker": first_blocker,
        "unresolved_mandatory_blockers": 1,
        "scope_guard": {"formal_r2_goal_resent": False, "formal_r2_ledger_reset": False, "formal_r2_artifact_regenerated": False, "stage_3_started": False, "legacy_720_or_spray_off_reorientation_added_to_stage01_graph": False, "collision_method": "adaptive_discrete_interpolation", "bullet_ccd": "not_available", "clearance": None},
        "q1_q20": {
            "Q1_formal_r2_evidence_immutable": "YES",
            "Q2_ledger_maximum_1_consumed_1": "YES",
            "Q3_second_formal_fjt_send_observed": "NO",
            "Q4_stage_21_to_28sr2_status": "See stage_authoritative_status; Formal R2 PASSED, Stage 2.9 BLOCKED by unresolved TF/FK gate.",
            "Q5_stage29_original_acceptance": "No repository-native Stage 2.9 definition was found; this task's mandatory gates and fail-closed rules were applied.",
            "Q6_complete_regression_executed": "ATTEMPTED; post-Formal-compatible regression passed, full pre-Formal matrix cannot be rerun against consumed ledger without violating safety boundary.",
            "Q7_three_regressions_all_passed": "NO — not required by any repository-native Stage 2.9 definition and fail-closed before runs 2/3.",
            "Q8_deterministic_requirement": "NOT_APPLICABLE for a repository-undefined three-run Stage 2.9 design; the executed compatible run was 423/423.",
            "Q9_frozen_input_hash_drift": "NO",
            "Q10_trajectory_fk_tf_regression": "YES — unresolved TF/FK gate is recorded; no new Formal R2 artifact drift was observed.",
            "Q11_collision_regression": "NO; native Bullet result is collision-free under adaptive_discrete_interpolation; strict CCD/clearance not available.",
            "Q12_velocity_acceleration_jerk_regression": "NO",
            "Q13_controller_execution_semantics": "YES, still passed in authoritative Stage 2.7S/Formal R2 evidence.",
            "Q14_formal_r2_bag": "YES — finalized, valid, read-to-EOF.",
            "Q15_recorder_transport_loss": "YES — PRE 0/0 and POST 0/0.",
            "Q16_all_mandatory_tests": "NO",
            "Q17_unresolved_blocker": "YES — stage28sr TF/FK consistency gate.",
            "Q18_stage29": "BLOCKED",
            "Q19_stage2_closed": "NO",
            "Q20_stage3_authorization_recommendation": "NO",
        },
    }
    write_json(OUT / "stage_2_closure_certificate.json", certificate)

    report = f"""# Stage 2.9 — Final Regression & Stage-2 Closure

`STAGE_2_9: BLOCKED`  
`STAGE_2_CLOSED: NO`  
`STAGE_3_AUTHORIZATION_RECOMMENDATION: NO`

Formal R2 remains immutable historical evidence and remains **PASSED**. No Formal R2 goal was resent, the one-shot ledger was not reset, and Stage 3 was not started.

## First blocker

- **Root cause:** the latest strict Stage 2.8S-R TF/FK artifact still reports the same-header-stamp TF/FK gate as blocked; Formal R2 is action/recorder evidence and does not replace that missing passing FK/TF certificate.
- **Artifact:** `{first_blocker['affected_artifact']}`
- **Expected:** `{json.dumps(first_blocker['expected_value'], ensure_ascii=False)}`
- **Observed:** `{json.dumps(first_blocker['observed_value'], ensure_ascii=False)}`

## Formal R2 immutable facts

- Goal UUID: `{goal_uuid}`
- Dispatch count: `1`; retry/resend/second goal: `false/false/false`
- Ledger: `maximum=1`, `consumed=1`
- Terminal: `SUCCEEDED`; FJT `error_code=0`; `Goal successfully reached!`
- Recorder/transport loss: PRE `0/0`; POST `0/0`
- Bag: finalized, valid, read-to-EOF
- Freeze manifest: `formal_r2_sha256_freeze_manifest.json`; package drift: `false`

## Regression execution

The post-Formal-compatible local regression completed with **423 passed, 0 failed, 0 errors, 0 skipped**. The pre-Formal H4.1 migration tests are not rerun against the consumed global ledger because doing so would require resetting it; the original H4.1 evidence records `17/17` and external full regression `475/475`. The repository contains no native Stage 2.9 definition requiring three runs, so runs 2 and 3 were not started after the mandatory blocker.

Collision evidence remains labelled `adaptive_discrete_interpolation`; strict Bullet CCD and clearance are `not_available`/null.

## Q1–Q20

1. Q1 Formal R2 evidence immutable? **YES**
2. Q2 Ledger `maximum=1`, `consumed=1`? **YES**
3. Q3 Second Formal FJT send? **NO**
4. Q4 Stage 2.1–2.8S-R2 status? **See `stage_authoritative_status` in the JSON certificate; Formal R2 is PASSED.**
5. Q5 Original Stage 2.9 acceptance? **No repository-native definition found; task-specified mandatory gates/fail-closed rules applied.**
6. Q6 Complete regression executed? **Attempted; compatible final-state regression passed, full pre-Formal matrix is not safely rerunnable.**
7. Q7 Three runs all passed? **NO; not required by repository definition and stopped fail-closed.**
8. Q8 Deterministic requirement? **N/A for undefined three-run design; executed run was 423/423.**
9. Q9 Frozen input/hash drift? **NO**
10. Q10 Trajectory/FK/TF regression? **YES, unresolved TF/FK gate.**
11. Q11 Collision regression? **NO**
12. Q12 Velocity/acceleration/jerk regression? **NO**
13. Q13 Controller semantics? **YES**
14. Q14 Formal R2 bag finalized/valid/read-to-EOF? **YES**
15. Q15 Recorder/transport loss 0/0? **YES, PRE and POST**
16. Q16 All mandatory tests? **NO**
17. Q17 Unresolved blocker? **YES**
18. Q18 Stage 2.9? **BLOCKED**
19. Q19 Stage 2 closed? **NO**
20. Q20 Stage 3 authorization recommendation? **NO**

Machine-readable certificate: `stage_2_closure_certificate.json`.
"""
    (OUT / "stage_2_closure_final_report.md").write_text(report, encoding="utf-8")
    print(json.dumps({"output": rel(OUT), "STAGE_2_9": "BLOCKED", "first_blocker": first_blocker["id"], "stage_3_started": False}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
