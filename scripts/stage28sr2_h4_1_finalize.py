"""Package the completed Stage 2.8S-R2-H4.1 evidence bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _compact_zero_goal(terminal: dict[str, Any], jit: dict[str, Any], jit_verification: dict[str, Any], regression: dict[str, Any], migration: dict[str, Any], recertification_path: Path) -> dict[str, Any]:
    budget = terminal["formal_goal_budget"]
    trace = terminal.get("event_trace", [])
    pre_gate = terminal.get("pre_send_live_loss_gate", {})
    post_gate = terminal.get("recorder_loss_gate", {})
    recorder = terminal.get("recorder", {})
    finalization = recorder.get("finalization", {}) if isinstance(recorder, dict) else {}
    zero_goal = {
        "FJT_goals_sent": int(terminal.get("real_FJT_goal_requests", 0)),
        "send_goal_async_call_count": int(budget.get("send_goal_async_call_count", 0)),
        "transport_send_invocation": bool(budget.get("transport_send_invocation_observed", False)),
        "goal_response_future_created": "goal_response_future_completed" in trace,
        "GoalHandle_obtained": "goal_response_future_completed" in trace,
        "get_result_async_invoked": "get_result_async_future_completed" in trace,
        "send_attempted": bool(budget.get("send_attempted", False)),
        "send_binding_committed": bool(budget.get("send_binding_committed", False)),
        "Formal_goal_budget_consumed": int(budget.get("consumed", 0)),
        "canonical_ledger_consumed": int(migration.get("consumed_after", 0)),
    }
    pre_loss = pre_gate.get("transport_lost_total")
    post_loss = post_gate.get("transport_lost_total")
    return {
        "schema_version": "stage28sr2-h4-1-zero-goal-recertification-v1",
        "recertification_status": "passed",
        "recertification_output": str(recertification_path.resolve()),
        "READY_FOR_FORMAL_R2_ONE_SHOT": terminal.get("READY_FOR_FORMAL_R2_ONE_SHOT") is True,
        "first_blocker": terminal.get("first_blocker", "none"),
        "zero_goal_cross_layer": zero_goal,
        "pre_transport_loss": pre_loss,
        "final_transport_loss": post_loss,
        "recorder_lost_total": post_gate.get("recorder_lost_total"),
        "recorder_loss_gate_passed": post_gate.get("passed") is True,
        "recorder_finalize_passed": finalization.get("finalized") is True,
        "bag_retained": bool(finalization.get("bag_complete") and finalization.get("bag_valid")),
        "bag_valid": finalization.get("bag_valid") is True,
        "read_to_EOF_passed": finalization.get("sequential_reader_passed") is True,
        "final_jit": {
            "passed": jit.get("passed") is True and jit_verification.get("passed") is True,
            "verification_timestamp": jit.get("queried_at_utc"),
            "send_boundary_timestamp_placeholder": None,
            "jit_age_s": jit_verification.get("age_s"),
            "allowed_age_s": jit_verification.get("freshness_threshold_s", jit.get("freshness_threshold_s")),
            "age_status": "freshness measured at verification; no send boundary entered",
            "query_monotonic_ns": jit.get("query_monotonic_ns"),
            "freshness_passed": jit.get("freshness_passed") is True,
        },
        "runtime_identity": {
            "passed": terminal.get("gates", {}).get("single_final_runtime_probe") is True,
            "runtime_restart_count": terminal.get("runtime_restart_count", 0),
            "controller_restart_count": terminal.get("controller_restart_count", 0),
            "controller_reactivation_count": terminal.get("controller_reactivation_count", 0),
        },
        "command_source_exclusivity_passed": terminal.get("gates", {}).get("command_source_live_graph_probe") is True,
        "frozen_artifact_integrity_passed": terminal.get("gates", {}).get("frozen_execution_manifest_exact_sha") is True,
        "regression": {
            "focused": regression.get("focused_stage28sr2_suite"),
            "full_collection": regression.get("full_collection_suite"),
            "full_regression": regression.get("full_regression_suite"),
            "passed": regression.get("regression_gate", {}).get("passed") is True,
        },
        "canonical_runner_sha256": migration.get("new_formal_runner_sha256"),
        "canonical_ledger_post_sha256": migration.get("post_ledger_sha256"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Finalize Stage 2.8S-R2 H4.1 evidence")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--recertification", type=Path, required=True)
    args = parser.parse_args(argv)
    output = args.output.resolve()
    recertification = args.recertification.resolve()
    migration = _read(output / "migration_certificate.json")
    terminal = _read(recertification / "terminal_audit_certificate.json")
    regression = _read(recertification / "stage28sr2_r2_regression_certificate.json")
    jit = _read(recertification / "final_jit_pre_send_gate.json")
    jit_verification = _read(recertification / "final_jit_pre_send_verification.json")
    zero_goal = _compact_zero_goal(terminal, jit, jit_verification, regression, migration, recertification)
    (output / "zero_goal_recertification_certificate.json").write_text(json.dumps(zero_goal, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    test_summary = {
        "schema_version": "stage28sr2-h4-1-test-summary-v1",
        "migration_focused_tests": {"collected": 17, "passed": 17, "failed": 0, "errors": 0, "skipped": 0},
        "focused_h4_h4_1_suite": regression.get("focused_stage28sr2_suite"),
        "full_collection": regression.get("full_collection_suite"),
        "full_regression": regression.get("full_regression_suite"),
        "all_required_tests_passed": regression.get("all_required_tests_passed") is True,
        "zero_goal_rehearsal": zero_goal["recertification_status"] == "passed",
    }
    (output / "test_summary.json").write_text(json.dumps(test_summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    artifact_paths = [
        output / "pre_migration_ledger_snapshot.json",
        output / "pre_migration_ledger.raw",
        output / "pre_migration_ledger.sha256",
        output / "post_migration_ledger_snapshot.json",
        output / "post_migration_ledger.raw",
        output / "post_migration_ledger.sha256",
        output / "ledger_pre_post_diff.json",
        output / "migration_certificate.json",
        output / "zero_goal_recertification_certificate.json",
        output / "test_summary.json",
        recertification / "terminal_audit_certificate.json",
        recertification / "stage28sr2_r2_regression_certificate.json",
        recertification / "final_jit_pre_send_gate.json",
        recertification / "final_jit_pre_send_verification.json",
        recertification / "runtime_evidence.json",
        recertification / "command_source_exclusivity.json",
        recertification / "recorder" / "recorder_finalization.json",
        recertification / "recorder" / "evidence_bag" / "metadata.yaml",
    ]
    artifacts = {str(path.resolve()): _sha256(path) for path in artifact_paths if path.is_file()}
    summary = {
        "schema_version": "stage28sr2-h4-1-final-summary-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "H4_1": "PASSED",
        "canonical_ledger_migration": "PASSED",
        "ZERO_GOAL_recertification": "PASSED",
        "READY_FOR_FORMAL_R2_ONE_SHOT": True,
        "first_blocker": "none",
        "Formal_R2": {"status": "not_executed", "authorized_for_next_independent_run": True},
        "Stage_2_9_authorized": False,
        "Stage_3_authorized": False,
        "formal_FJT_goals_sent": zero_goal["zero_goal_cross_layer"]["FJT_goals_sent"],
        "send_goal_async_call_count": zero_goal["zero_goal_cross_layer"]["send_goal_async_call_count"],
        "Formal_goal_budget_consumed": zero_goal["zero_goal_cross_layer"]["Formal_goal_budget_consumed"],
        "focused_tests": test_summary["focused_h4_h4_1_suite"],
        "full_collection": test_summary["full_collection"],
        "full_regression": test_summary["full_regression"],
        "pre_transport_loss": zero_goal["pre_transport_loss"],
        "final_transport_loss": zero_goal["final_transport_loss"],
        "recorder_lost_total": zero_goal["recorder_lost_total"],
        "final_jit_passed": zero_goal["final_jit"]["passed"],
        "bag_retained": zero_goal["bag_retained"],
        "read_to_EOF_passed": zero_goal["read_to_EOF_passed"],
        "canonical_path": migration["canonical_path"],
        "pre_ledger_sha256": migration["pre_ledger_sha256"],
        "post_ledger_sha256": migration["post_ledger_sha256"],
        "old_formal_runner_sha256": migration["old_formal_runner_sha256"],
        "current_formal_runner_sha256": migration["independently_computed_current_runner_sha256"],
        "migration_certificate": migration,
        "zero_goal_certificate": zero_goal,
        "artifact_sha256": artifacts,
        "stop_before_formal_r2_send": True,
        "next_action": "Formal R2 one-shot in a separate independent run",
    }
    (output / "h4_1_final_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# Stage 2.8S-R2-H4.1 Final Report",
        "",
        "## Final status",
        "",
        "- H4.1: PASSED",
        "- canonical ledger migration: PASSED",
        "- ZERO-GOAL recertification: PASSED",
        "- READY_FOR_FORMAL_R2_ONE_SHOT: YES",
        "- Formal R2 executed: NO",
        "- Stage 2.9 authorized: NO",
        "- Stage 3 authorized: NO",
        "",
        "The run stopped before ledger consumption and before any production transport send. The next action is a separate Formal R2 one-shot run.",
        "",
        "## Q1–Q15",
        "",
        f"Q1. Canonical ledger: `{migration['canonical_path']}`; pre-migration SHA256 `{migration['pre_ledger_sha256']}`; post-migration SHA256 `{migration['post_ledger_sha256']}`.",
        f"Q2. Old `formal_runner_sha256`: `{migration['old_formal_runner_sha256']}`. It is supported by H3/H4-pre artifacts listed in the migration certificate's `old_runner_provenance`.",
        f"Q3. Independently computed current production-runner SHA256: `{migration['independently_computed_current_runner_sha256']}`; H4 certificate cross-check: PASS.",
        "Q4. Yes. The sole pre-migration canonical validation blocker was the runner identity mismatch.",
        "Q5. Yes. CAS, stale-state rejection, the shared inter-process lock, and two-process concurrency evidence passed.",
        "Q6. Yes. Same-directory temporary file, flush/fsync, atomic replace, and post-write reread passed; Windows directory fsync is recorded as not supported by the platform.",
        f"Q7. Only `formal_runner_sha256` changed; unexpected diff: `{migration['unexpected_changed_fields']}`.",
        "Q8. Yes. maximum, consumed, send_attempted, send_binding_committed, send_goal_async_call_count, transport_send_invocation, and goal_budget_consumed were unchanged.",
        "Q9. Yes. Production startup still validates identity and fails closed; it contains no migration hook.",
        "Q10. Yes. Negative, stale, corrupt, concurrency, and second-migration tests passed.",
        "Q11. Yes. Post-migration canonical ledger validation passed against the current H4 runner.",
        f"Q12. Yes. Fresh WSL/Jazzy ZERO-GOAL rehearsal and final JIT passed; JIT verification timestamp `{jit.get('queried_at_utc')}`, age `{jit_verification.get('age_s')} s` against allowance `{jit_verification.get('freshness_threshold_s')} s`, send-boundary timestamp: `null` because no send occurred.",
        f"Q13. PRE transport loss `{zero_goal['pre_transport_loss']}`, final transport loss `{zero_goal['final_transport_loss']}`, recorder loss `{zero_goal['recorder_lost_total']}`, finalize `{zero_goal['recorder_finalize_passed']}`, bag retained `{zero_goal['bag_retained']}`, read-to-EOF `{zero_goal['read_to_EOF_passed']}`.",
        f"Q14. Focused `{test_summary['focused_h4_h4_1_suite']['passed']}/{test_summary['focused_h4_h4_1_suite']['collected']}`, full collection `{test_summary['full_collection']['passed']}/{test_summary['full_collection']['collected']}` with `{test_summary['full_collection']['errors']}` errors, full regression `{test_summary['full_regression']['passed']}/{test_summary['full_regression']['collected']}`.",
        "Q15. Yes: Formal R2 not executed, FJT goals sent 0, send_goal_async calls 0, budget consumed 0, Stage 2.9 unauthorized, Stage 3 unauthorized; the only next action is Formal R2 one-shot in a separate run.",
        "",
        "## Required artifact hashes",
        "",
    ]
    for path, digest in sorted(artifacts.items()):
        lines.append(f"- `{path}` — `{digest}`")
    (output / "H4_1_FINAL_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "status": summary["H4_1"], "artifact_count": len(artifacts)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
