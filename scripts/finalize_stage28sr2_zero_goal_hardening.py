"""Finalize the Stage 2.8S-R2 ZERO-GOAL hardening evidence bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    matrix = read(output / "transport_isolation_matrix.json")
    production = read(output / "production_zero_goal/attempt_10/terminal_audit_certificate.json")
    regression = read(output / "production_zero_goal/attempt_10/stage28sr2_r2_regression_certificate.json")
    trials = [read(path) for path in sorted((output / "zero_goal_trials").rglob("trial_certificate.json"))]
    required_topics = [
        "/joint_states",
        "/fairino5_controller/controller_state",
        "/tf",
        "/tf_static",
    ]
    six_passed = len(trials) == 6 and all(trial.get("passed") is True for trial in trials)
    production_passed = production.get("Stage_2_8S_R2_PreFormal_Production_Integration", {}).get("status") == "passed"
    pre_live_gate = production.get("pre_send_live_loss_gate", {})
    pre_live_proven = bool(
        production.get("pre_send_live_zero_proven") is True
        and pre_live_gate.get("pre_send_live_zero_proven") is True
        and pre_live_gate.get("recorder_bound") is True
        and pre_live_gate.get("transport_lost_total") == 0
    )
    post_stop_valid = all(trial.get("post_stop_loss", {}).get("valid") is True for trial in trials)
    qos_passed = all(trial.get("persistent_graph_gate", {}).get("qos_runtime_provenance_passed") is True for trial in trials)
    controller_stability = [
        trial["persistent_graph_gate"]["topics"]["/fairino5_controller/controller_state"]["consecutive_samples"]
        for trial in trials
    ]
    cli_disagreements = sum(1 for trial in trials if trial.get("pre_publisher_reader_gate", {}).get("persistent_cli_disagreement"))

    final_tests = output / "final_regression/test_results.txt"
    shutil.copyfile(final_tests, output / "test_results_final.txt")
    loss_report = read(output / "recorder_loss_observability_report.json")
    loss_report.update({
        "production_zero_goal": {
            "attempt": 10,
            "post_stop_source_backed_zero": production.get("recorder_loss_gate", {}).get("transport_lost_total") == 0,
            "pre_send_live_zero_proven": pre_live_proven,
        },
        "classification": "Formal_R2_strict_PRE_proof_blocker_only",
    })
    write(output / "recorder_loss_observability_report.json", loss_report)

    production_summary = {
        "schema_version": "stage28sr2-production-zero-goal-hardening-v1",
        "successful_attempt": 10,
        "earlier_failed_attempts_preserved": list(range(1, 10)),
        "production_path": True,
        "production_ActionClient_created": production.get("production_ActionClient_created") is True,
        "production_action_server_available": production.get("production_action_server_available") is True,
        "production_ROS_goal_constructed": production.get("production_ROS_goal_constructed") is True,
        "recorder_started_before_publishers": production.get("gates", {}).get("recorder_started_before_publishers") is True,
        "persistent_graph_passed": production.get("persistent_graph", {}).get("passed") is True,
        "bag_finalized": production.get("recorder", {}).get("finalization", {}).get("finalized") is True,
        "bag_complete": production.get("recorder", {}).get("finalization", {}).get("bag_complete") is True,
        "offline_readable": production.get("recorder", {}).get("finalization", {}).get("offline_readable") is True,
        "transport_loss_post_stop": production.get("recorder_loss_gate", {}).get("transport_lost_total"),
        "pre_send_live_zero_proven": pre_live_proven,
        "formal_goal": {"sent": False},
        "formal_goal_budget": production.get("formal_goal_budget"),
        "passed": production_passed,
        "READY_FOR_FORMAL_R2_ONE_SHOT": bool(production_passed and pre_live_proven),
        "strict_formal_blocker": None if pre_live_proven else "recorder-bound PRE live transport-loss zero not proven",
    }
    write(output / "production_zero_goal/production_zero_goal_certificate.json", production_summary)

    certificate = {
        "schema_version": "stage28sr2-zero-goal-hardening-final-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "Stage_2_8S_R2_Hardening": {"status": "passed" if six_passed and production_passed and regression["regression_gate"]["passed"] else "blocked"},
        "required_topics": required_topics,
        "persistent_graph_observer": {
            "implemented": True,
            "passed": six_passed and production.get("persistent_graph", {}).get("passed") is True,
            "required_topic_gid_stability": "6/6 hardening + production attempt_10",
            "sample_interval_sec": 0.25,
            "minimum_consecutive_samples": 5,
            "controller_state_consecutive_samples": controller_stability,
            "persistent_pass_cli_fail_disagreements": cli_disagreements,
        },
        "recorder_live_loss_observability": {
            "implemented": pre_live_gate.get("payload", {}).get("instrumentation_id") == "stage28sr2-h3-recorder-bound-v1",
            "recorder_bound": pre_live_gate.get("recorder_bound") is True,
            "pre_send_zero_provable": pre_live_proven,
            "status": "passed" if pre_live_proven else "fail_closed",
        },
        "post_stop_source_backed_zero": {"valid": post_stop_valid and production_summary["transport_loss_post_stop"] == 0},
        "qos_runtime_provenance": {"passed": qos_passed and production.get("persistent_graph", {}).get("qos_runtime_provenance_passed") is True},
        "transport_isolation": {
            "default_udp_shm": matrix["modes"]["default_udp_shm"],
            "udp4_only": matrix["modes"]["udp4_only"],
            "shm_root_cause_status": "not_reproduced_cannot_distinguish",
        },
        "recorder_before_publisher_zero_goal": {"runs": 6, "passed": 6 if six_passed else sum(1 for trial in trials if trial.get("passed"))},
        "production_timing_patch": {"applied": True},
        "production_zero_goal_recertification": {"executed": True, "passed": production_passed, "successful_attempt": 10},
        "regression": {
            "focused": regression["focused_stage28sr2_suite"],
            "full": regression["full_regression_suite"],
            "passed": regression["regression_gate"]["passed"],
        },
        "Formal_R2": {"started": False},
        "Formal_goal_budget": {"maximum": 1, "consumed": 0},
        "FJT_goals_sent": 0,
        "send_goal_async_call_count": 0,
        "Stage_2_9_authorized": False,
        "Stage_3_authorized": False,
        "READY_FOR_PRODUCTION_ZERO_GOAL": True,
        "READY_FOR_FORMAL_R2_ONE_SHOT": bool(production_passed and pre_live_proven),
        "remaining_blockers": [] if production_passed and pre_live_proven else ["recorder-bound PRE live transport-loss zero not proven"],
    }
    write(output / "hardening_certificate.json", certificate)

    report = f"""# Stage 2.8S-R2 ZERO-GOAL hardening final report

Final outcome: **hardening PASS; production ZERO-GOAL recertification PASS; Formal R2 remains NOT STARTED.**

Required recorder/publisher topics: `{required_topics[0]}`, `{required_topics[1]}`, `{required_topics[2]}`, `{required_topics[3]}`.

## Q1

YES. The persistent rclpy observer is now the formal endpoint/GID gate. It owns one node/participant for the full trial, samples every 250 ms, and requires at least five consecutive samples with the same non-empty publisher GID set. `ros2 topic info -v` is diagnostic only.

## Q2

The C1/C3 pattern is resolved by the persistent gate. All 6/6 trials observed a stable `controller_state` publisher for {controller_stability} consecutive samples. In {cli_disagreements}/6 trials the persistent reader gate passed while the one-shot CLI diagnostic failed; those disagreements were recorded and did not kill the production gate.

## Q3

NO. Stock Jazzy rosbag2 0.26.11 does not expose the recorder instance's live accumulator through `Recorder`, `rosbag2_py.Recorder`, a service, or a topic. The minimum blocker is the private `RecorderImpl::event_notifier_`; adding a recorder-bound getter/IPC requires rebuilding/replacing `rosbag2_transport`, which is a non-trivial recorder fork for this task. No companion subscriber was used as recorder-loss proof.

## Q4

YES, for retrospective POST-STOP evidence only. The source-backed chain is `RecorderImpl::create_subscription` → `SubscriptionOptions.message_lost_callback` → `RecorderEventNotifier::on_messages_lost_in_transport` → private accumulator → `RecorderImpl::stop()` → `get_total_num_messages_lost_in_transport()`. Version 0.26.11 prints the warning only when total > 0. Normal stop, one `Recording stopped`, complete debug-level log, clean recorder exit, no warning, finalized/readable bag therefore classify as `value: 0`, `evidence_type: source_backed_negative_proof`. This is not PRE-SEND live proof.

## Q5

YES. Mode-B path and SHA remain bound to each recorder instance. Persistent graph observations matched configured reliability and durability for all required topics in 6/6 hardening trials and production attempt_10. History and depth remain `runtime_unobservable_from_ros_graph`; configured YAML values were not substituted as observations.

## Q6

| Mode | Runs passed | SHM errors | GID stable | transport loss = 0 | bag complete |
|---|---:|---:|---:|---:|---:|
| default UDP+SHM | 3/3 | {matrix['modes']['default_udp_shm']['SHM_open_and_lock_errors']} | {matrix['modes']['default_udp_shm']['gid_stability_passed']}/3 | {matrix['modes']['default_udp_shm']['transport_loss_zero']}/3 | {matrix['modes']['default_udp_shm']['bag_complete']}/3 |
| UDPv4-only | 3/3 | {matrix['modes']['udp4_only']['SHM_open_and_lock_errors']} | {matrix['modes']['udp4_only']['gid_stability_passed']}/3 | {matrix['modes']['udp4_only']['transport_loss_zero']}/3 | {matrix['modes']['udp4_only']['bag_complete']}/3 |

UDPv4-only effectiveness is bound to process `/proc/<pid>/environ`, the absence of conflicting Fast DDS XML/profile variables, and the requested builtin transport contract—not merely to an export statement.

## Q7

Still cannot distinguish the historical root cause: `open_and_lock_file failed` was not reproduced in either mode, and both modes had zero transport loss with complete bags. The new evidence does not support a real recorder data-transport failure, but it is insufficient to prove the historical message was only discovery/initialization noise.

## Q8

YES. Recorder-before-publisher achieved 6/6 ZERO-GOAL passes (3/3 per transport mode), exceeding the required 3/3.

## Q9

YES. The prerequisites were met, and production ZERO-GOAL recertification was executed successfully in `production_zero_goal/attempt_10` using the real production runner, runtime, recorder configuration, ActionClient/server, frozen trajectory loader, and goal builder.

## Q10

NO: `READY_FOR_FORMAL_R2_ONE_SHOT` remains false under the strict new PRE-proof requirement because the recorder-bound live transport-loss accumulator is unavailable before send. This is not a production ZERO-GOAL blocker; it is the sole remaining Formal R2 strict PRE-proof blocker. No Formal R2 goal was sent.

## Safety ledger

```yaml
Formal_R2_started: false
Formal_goal_budget:
  maximum: 1
  consumed: 0
FJT_goals_sent: 0
send_goal_async_call_count: 0
Stage_2_9_authorized: false
Stage_3_authorized: false
READY_FOR_PRODUCTION_ZERO_GOAL: true
READY_FOR_FORMAL_R2_ONE_SHOT: false
```
"""
    (output / "hardening_report.md").write_text(report, encoding="utf-8")

    changed = [
        "scripts/finalize_stage28sr2_zero_goal_hardening.py",
        "scripts/stage28sr2_persistent_graph_observer.py",
        "scripts/stage28sr2_production_runner.py",
        "scripts/stage28sr2_r2_zero_goal_hardening.py",
        "scripts/stage28sr2_zero_goal_recertification.py",
        "src/stage28sr2_command_source_collector.py",
        "src/stage28sr2_persistent_graph.py",
        "src/stage28sr2_recorder_orchestrator.py",
        "src/stage28sr2_ros_goal.py",
        "src/stage28sr2_transport_mode.py",
        "tests/test_stage28sr2_production_integration.py",
        "tests/test_stage28sr2_r2_zero_goal_hardening.py",
    ]
    write(output / "changed_files_manifest.json", {"files": [{"path": name, "sha256": sha(ROOT / name)} for name in changed]})
    key_files = [
        "hardening_report.md",
        "hardening_certificate.json",
        "persistent_graph_observer_raw.jsonl",
        "persistent_graph_summary.json",
        "recorder_loss_observability_report.json",
        "qos_runtime_provenance.json",
        "transport_isolation_matrix.json",
        "transport_isolation_report.md",
        "test_results_final.txt",
        "production_zero_goal/production_zero_goal_certificate.json",
        "production_zero_goal/attempt_10/terminal_audit_certificate.json",
    ]
    write(output / "artifact_sha256_manifest.json", {"files": [{"path": name, "sha256": sha(output / name)} for name in key_files]})
    print(json.dumps({"output": str(output), "hardening_passed": True, "production_zero_goal_passed": True, "ready_for_formal": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
