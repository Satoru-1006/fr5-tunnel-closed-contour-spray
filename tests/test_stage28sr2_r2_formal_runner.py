"""Zero-goal and fake-transport regressions for the hardened R2 runner."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.stage28sr2_formal_runner import (
    ACTION_NAME,
    EXPECTED_JOINTS,
    FROZEN_GOAL,
    FakeActionTransport,
    FakeTransportOutcome,
    FormalRunner,
    FormalState,
    frozen_goal_gate,
    verify_atomic_ledger_binding_static,
    verify_command_source_exclusivity,
    verify_point0_no_extra_interpolation,
    verify_recorder_readiness,
    verify_runtime_identity,
    verify_state_machine_static,
    verify_terminal_certificate_static,
)


def _frozen_point0() -> list[float]:
    return [2.0285338324812976, 1.2221671180677403, -1.4714285950888577, -1.3215542746099105, -1.5707769341011972, 0.0]


def valid_runtime_evidence() -> dict:
    identity = "single-final-runtime-test-instance"
    return {
        "evidence_kind": "live_final_runtime_instance",
        "same_final_runtime_identity": True,
        "runtime_instance_id": identity,
        "verified_runtime_instance_id": identity,
        "recorder_runtime_instance_id": identity,
        "recheck_runtime_instance_id": identity,
        "runtime_restart_count": 0,
        "controller_restart_count": 0,
        "controller_reactivation_count": 0,
        "controller_pid_or_process_identity": "pid=1234/start=2026-08-06T00:00:00Z",
        "ROS_distribution": "jazzy",
        "JTC_package_version": "4.40.1",
        "controller_name": "fairino5_controller",
        "controller_type": "joint_trajectory_controller/JointTrajectoryController",
        "controller_state": "active",
        "interpolation_method": "splines",
        "hardware_component": "mock_components/GenericSystem",
        "simulation_only": True,
        "configured_joints": list(EXPECTED_JOINTS),
        "command_interfaces": ["position"],
        "state_interfaces": ["position"],
        "robot_description_sha256": "a" * 64,
        "controllers_yaml_sha256": "b" * 64,
        "initial_positions_sha256": "c" * 64,
        "initial_joint_state": _frozen_point0(),
        "frozen_point0": _frozen_point0(),
        "initial_state_max_abs_error_rad": 0.0,
        "graph_audit": {"observed_from_runtime_graph": True, "controller_state": "active"},
        "command_source_exclusivity": {
            "observed_from_runtime_graph": True,
            "authorized_formal_runner": 0,
            "other_action_clients": 0,
            "joint_trajectory_publishers": 0,
            "MoveIt_execution_nodes": 0,
            "known_old_runtime_clients": 0,
            "conflicting_project_processes": 0,
        },
        "recorder_readiness": {
            "runtime_instance_id": identity,
            "storage_backend": "mcap",
            "recorder_process_alive": True,
            "rosbag2_version": "0.26.11",
            "max_cache_size": 0,
            "cache_enabled": False,
            "direct_write_verified": True,
            "rosbag2_argv": ["ros2", "bag", "record", "--max-cache-size", "0"],
            "capability": {
                "rosbag2_version": "0.26.11",
                "max_cache_size_zero_direct_write_supported": True,
            },
            "native_filesystem_verified": True,
            "endpoint_gate_passed": True,
            "subscribed_topics": [
                "/joint_states",
                "/fairino5_controller/controller_state",
                "/tf",
                "/tf_static",
            ],
            "output_directory_writable": True,
            "disk_space_sufficient": True,
            "sequential_reader": {"passed": True},
            "recorder_loss_gate": {
                "source": "structured_json:message_loss_statistics.json",
                "transport_lost_total": 0,
                "recorder_lost_total": None,
                "recorder_counter_applicable": False,
                "per_topic": {},
                "passed": True,
                "first_blocker": None,
            },
            "pre_send_live_loss_gate": {
                "schema_version": "stage28sr2-recorder-bound-pre-live-loss-gate-v1",
                "pre_send_live_zero_proven": True,
                "recorder_bound": True,
                "transport_lost_total": 0,
                "passed": True,
                "first_blocker": None,
            },
        },
    }


def runner(tmp_path: Path, *, audit_only: bool = True, outcome: FakeTransportOutcome | None = None) -> tuple[FormalRunner, FakeActionTransport | None]:
    transport = None if audit_only else FakeActionTransport(outcome)
    evidence = valid_runtime_evidence()
    if not audit_only:
        evidence["command_source_exclusivity"]["authorized_formal_runner"] = 1
    return FormalRunner(output=tmp_path, action_transport=transport, runtime_evidence=evidence, audit_only=audit_only, ledger_path=tmp_path / "test_ledger.json"), transport


def test_frozen_goal_gate_and_point0_contract() -> None:
    result = frozen_goal_gate(FROZEN_GOAL)
    assert result["passed"]
    assert result["joint_names"] == EXPECTED_JOINTS
    assert result["point_count"] == 25532
    assert result["first_time_from_start"] == 0.0
    assert result["last_time_from_start"] == 765.97968192
    assert result["positions_present"]
    assert result["velocities_present"]
    assert result["accelerations_present"]
    assert not result["effort_present"]
    assert result["point0_exact_passed"]
    assert result["round_trip"]["max_numeric_difference"] == 0
    assert verify_point0_no_extra_interpolation()["passed"]


def test_single_runtime_identity_requires_recheck_and_zero_restart() -> None:
    assert verify_runtime_identity(valid_runtime_evidence())["passed"]
    changed = valid_runtime_evidence()
    changed["recheck_runtime_instance_id"] = "different-runtime"
    result = verify_runtime_identity(changed)
    assert not result["passed"]
    assert result["first_blocker"] == "runtime_identity_mismatch:same_final_runtime_identity"


def test_recorder_readiness_and_command_source_gates_are_fail_closed() -> None:
    evidence = valid_runtime_evidence()
    assert verify_recorder_readiness(evidence["recorder_readiness"])["passed"]
    assert verify_command_source_exclusivity(evidence["command_source_exclusivity"], audit_only=True)["passed"]
    competing = dict(evidence["command_source_exclusivity"])
    competing["other_action_clients"] = 1
    assert not verify_command_source_exclusivity(competing, audit_only=True)["passed"]


def test_static_state_machine_and_certificate_contracts() -> None:
    assert verify_state_machine_static()["passed"]
    assert verify_atomic_ledger_binding_static()["passed"]
    assert verify_terminal_certificate_static()["passed"]


def test_audit_only_stops_before_ledger_and_never_calls_transport(tmp_path: Path) -> None:
    runner_instance, _ = runner(tmp_path)
    certificate = runner_instance.audit_only_run()
    assert certificate["Stage_2_8S_R2"]["status"] == "preformal_ready"
    assert certificate["READY_FOR_FORMAL_R2_ONE_SHOT"]
    assert certificate["formal_goal_budget"]["maximum"] == 1
    assert certificate["formal_goal_budget"]["consumed"] == 0
    assert certificate["formal_goal_budget"]["send_attempted"] is False
    assert certificate["formal_goal_budget"]["send_goal_async_call_count"] == 0
    assert certificate["real_FJT_goals_sent"] == 0
    assert "ledger_consumed" not in certificate["event_trace"]
    assert certificate["event_trace"][-1] == "audit_stopped_before_ledger_consumption"
    assert runner_instance.machine.current == FormalState.TERMINAL


def test_ledger_is_consumed_before_fake_send_and_second_send_is_impossible(tmp_path: Path) -> None:
    runner_instance, transport = runner(tmp_path, audit_only=False)
    certificate = runner_instance.run_formal_with_fake_transport()
    assert transport is not None
    assert transport.send_call_count == 1
    assert transport.call_order == ["fake_send_goal_async"]
    assert certificate["formal_goal_budget"]["consumed"] == 1
    assert certificate["formal_goal_budget"]["send_attempted"]
    assert certificate["formal_goal_budget"]["send_goal_async_call_count"] == 1
    assert certificate["formal_state"] == FormalState.TERMINAL.value
    assert certificate["Stage_2_9"]["authorized"] is False
    assert certificate["execution_mode"] == "fake_rehearsal"
    assert certificate["eligible_for_formal_certification"] is False
    assert certificate["event_trace"].index("ledger_consumed") < certificate["event_trace"].index("fake_send_called")
    second = runner_instance.run_formal_with_fake_transport()
    assert transport.send_call_count == 1
    assert second["formal_goal_budget"]["consumed"] == 1
    assert second["retry_allowed"] is False


def test_crash_after_ledger_consume_is_terminal_and_not_retryable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runner_instance, transport = runner(tmp_path, audit_only=False)
    original = transport.send_goal_async

    def crash(_goal):
        raise RuntimeError("simulated process termination after binding")

    transport.send_goal_async = crash  # type: ignore[method-assign]
    certificate = runner_instance.run_formal_with_fake_transport()
    assert certificate["formal_goal_budget"]["consumed"] == 1
    assert certificate["formal_goal_budget"]["send_goal_async_call_count"] == 1
    assert certificate["goal_send_attempted"]
    assert certificate["retry_allowed"] is False
    assert certificate["first_blocker"].startswith("runner_exception:")
    transport.send_goal_async = original  # retain a useful explicit assertion that the fake was injected


@pytest.mark.parametrize(
    ("outcome", "expected_state"),
    [
        (FakeTransportOutcome(accepted=None, action_terminal_status=None, fjt_error_code=None, fjt_error_string="ack timeout"), FormalState.CLIENT_TIMEOUT),
        (FakeTransportOutcome(accepted=False, action_terminal_status=None, fjt_error_code=-1, fjt_error_string="server rejected"), FormalState.REJECTED),
        (FakeTransportOutcome(accepted=True, action_terminal_status="ABORTED", fjt_error_code=-4, fjt_error_string="abort"), FormalState.ABORTED),
        (FakeTransportOutcome(accepted=True, action_terminal_status="CANCELED", fjt_error_code=-5, fjt_error_string="cancel", cancel_requested=True), FormalState.CANCELED),
        (FakeTransportOutcome(accepted=True, action_terminal_status="SUCCEEDED", fjt_error_code=0, recorder_alive_during_execution=False, recorder_finalized=False, bag_complete=False, bag_valid=False, partial_bag_retained=True), FormalState.RECORDER_FAILED),
        (FakeTransportOutcome(accepted=True, action_terminal_status="SUCCEEDED", fjt_error_code=0, analysis_passed=False), FormalState.ANALYSIS_FAILED),
    ],
)
def test_all_post_consume_outcomes_are_terminal_and_no_retry(tmp_path: Path, outcome: FakeTransportOutcome, expected_state: FormalState) -> None:
    runner_instance, transport = runner(tmp_path, audit_only=False, outcome=outcome)
    certificate = runner_instance.run_formal_with_fake_transport()
    assert transport is not None and transport.send_call_count == 1
    assert certificate["formal_goal_budget"]["consumed"] == 1
    assert certificate["retry_allowed"] is False
    assert any(item["to"] == expected_state.value for item in json.loads((tmp_path / "stage28sr2_r2_formal_state.json").read_text())["history"])
    assert certificate["Stage_2_9"]["authorized"] is False


def test_default_runtime_evidence_blocks_without_fabricating_a_pass(tmp_path: Path) -> None:
    runner_instance = FormalRunner(output=tmp_path, audit_only=True, ledger_path=tmp_path / "test_ledger.json")
    certificate = runner_instance.audit_only_run()
    assert certificate["Stage_2_8S_R2"]["status"].startswith("blocked_")
    assert "runtime_identity" in certificate["first_blocker"]
    assert certificate["formal_goal_budget"]["consumed"] == 0
    assert certificate["real_FJT_goals_sent"] == 0


def test_action_name_and_frozen_goal_are_not_retimed() -> None:
    assert ACTION_NAME == "/fairino5_controller/follow_joint_trajectory"
    source = json.loads(FROZEN_GOAL.read_text(encoding="utf-8"))
    assert source["points"][0]["time_from_start"]["sec"] == 0
    assert source["points"][0]["time_from_start"]["nanosec"] == 0
