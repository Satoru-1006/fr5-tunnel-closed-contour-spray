"""Compose the independent zero-goal Stage 2.8S-R2 readiness certificate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _first_false_gate(gates: Mapping[str, Any], evidence: Mapping[str, Any]) -> str | None:
    for name, passed in gates.items():
        if passed is True:
            continue
        if name == "recorder_loss_gate":
            loss_gate = evidence.get("recorder_loss_gate", {})
            return str(loss_gate.get("first_blocker") or "recorder_message_loss_gate_failed")
        return f"production_gate_failed:{name}"
    return None


def compose(runtime_certificate_path: Path, regression_certificate_path: Path, output_path: Path) -> dict[str, Any]:
    runtime = read_json(runtime_certificate_path)
    regression = read_json(regression_certificate_path)
    runtime_gates = dict(runtime.get("gates", {}))
    runtime_gates.pop("regression_gate", None)
    runtime_gate_passed = bool(runtime_gates) and all(value is True for value in runtime_gates.values())
    regression_gate = regression.get("regression_gate", {})
    regression_passed = regression_gate.get("passed") is True and regression.get("all_required_tests_passed") is True
    budget = dict(runtime.get("formal_goal_budget", {}))
    budget_safe = (
        budget.get("maximum") == 1
        and budget.get("consumed") == 0
        and budget.get("send_attempted") is False
        and budget.get("send_goal_async_call_count") == 0
    )
    production_send_count = int(runtime.get("real_FJT_goal_requests", 0))
    live_gate = runtime.get("pre_send_live_loss_gate", {})
    pre_send_live_zero_proven = bool(
        runtime.get("pre_send_live_zero_proven") is True
        and isinstance(live_gate, Mapping)
        and live_gate.get("pre_send_live_zero_proven") is True
        and live_gate.get("recorder_bound") is True
        and live_gate.get("transport_lost_total") == 0
    )
    current_gates_passed = runtime_gate_passed and regression_passed
    ready = bool(current_gates_passed and pre_send_live_zero_proven and budget_safe and production_send_count == 0)
    first_blocker = _first_false_gate(runtime_gates, runtime)
    if first_blocker is None and not regression_passed:
        first_blocker = str(regression_gate.get("first_blocker") or "regression_gate_failed")
    if first_blocker is None and not pre_send_live_zero_proven:
        first_blocker = "pre_send_recorder_bound_live_zero_not_proven"
    if first_blocker is None and not budget_safe:
        first_blocker = "formal_goal_budget_not_zero_before_formal"
    if first_blocker is None and production_send_count != 0:
        first_blocker = "production_send_goal_async_call_count_nonzero"
    if first_blocker is None:
        first_blocker = "none"
    status = "preformal_ready" if ready else f"blocked_{first_blocker}"
    certificate = {
        "schema_version": "stage28sr2-r2-final-formal-readiness-certificate-v1",
        "runtime_certificate": {
            "path": str(runtime_certificate_path.resolve()),
            "sha256": sha256(runtime_certificate_path),
        },
        "regression_certificate": {
            "path": str(regression_certificate_path.resolve()),
            "sha256": sha256(regression_certificate_path),
        },
        "production_runtime_gates": runtime_gates,
        "production_runtime_gate_passed": runtime_gate_passed,
        "recorder_loss_gate": runtime.get("recorder_loss_gate", {}),
        "pre_send_live_loss_gate": live_gate,
        "pre_send_live_zero_proven": pre_send_live_zero_proven,
        "pytest": {
            "focused": regression.get("focused_stage28sr2_suite"),
            "full_collection": regression.get("full_collection_suite"),
            "full_regression": regression.get("full_regression_suite"),
            "all_required_tests_passed": regression.get("all_required_tests_passed") is True,
        },
        "regression_gate": regression_gate,
        "formal_goal_budget": budget,
        "production_send_goal_async_call_count": production_send_count,
        "real_FJT_goals_sent": production_send_count,
        "Stage_2_8S_R2_PreFormal_Production_Integration": {"status": status},
        "Stage_2_8S_R2_PreFormal_Final_Readiness": {"status": status},
        "Stage_2_8S_R2": {"status": status},
        "Stage_2_8S_R2_Formal": {"status": "not_started", "started": False},
        "READY_FOR_FORMAL_R2_ONE_SHOT": ready,
        "Can_we_safely_consume_the_one_and_only_formal_R2_FJT_goal_next": ready,
        "Stage_2_9_authorized": False,
        "Stage_3_authorized": False,
        "same_runtime_identity_initial_to_final": runtime.get("single_runtime_identity", {}).get("passed") is True,
        "production_ActionClient_created": runtime.get("production_ActionClient_created") is True,
        "production_action_server_available": runtime.get("production_action_server_available") is True,
        "frozen_goal_semantic_gate": runtime.get("production_ROS_goal_constructed") is True and runtime.get("production_goal_semantic_gate", {}).get("passed") is True,
        "event_trace": runtime.get("event_trace", []),
        "first_blocker": first_blocker,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(certificate, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return certificate


def main() -> int:
    parser = argparse.ArgumentParser(description="Compose Stage 2.8S-R2 zero-goal readiness evidence")
    parser.add_argument("--runtime-certificate", type=Path, required=True)
    parser.add_argument("--regression-certificate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    certificate = compose(args.runtime_certificate, args.regression_certificate, args.output)
    print(json.dumps({
        "output": str(args.output.resolve()),
        "status": certificate["Stage_2_8S_R2_PreFormal_Final_Readiness"]["status"],
        "READY_FOR_FORMAL_R2_ONE_SHOT": certificate["READY_FOR_FORMAL_R2_ONE_SHOT"],
        "formal_goal_budget": certificate["formal_goal_budget"],
        "first_blocker": certificate["first_blocker"],
    }, ensure_ascii=False, indent=2))
    return 0 if certificate["READY_FOR_FORMAL_R2_ONE_SHOT"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
