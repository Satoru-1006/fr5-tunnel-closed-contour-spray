"""H3 formal plumbing tests.  Every ledger is temporary; no real ROS send occurs."""

from __future__ import annotations

import json
import multiprocessing
from pathlib import Path
from typing import Any

import pytest

from scripts.stage28sr2_final_readiness_certification import compose
from scripts.stage28sr2_formal_runner import FormalRunner, FormalState, PersistentStateMachine
from scripts.stage28sr2_production_runner import ProductionFormalExecutor
from src.stage28sr2_formal_ledger import FormalGoalLedger, LEDGER_SCHEMA_VERSION
from src.stage28sr2_final_jit_gate import build_final_jit_pre_send_gate
from src.stage28sr2_recorder_orchestrator import (
    ROSBAG2_0_26_11_COMMIT,
    Rosbag2RecorderOrchestrator,
    post_stop_source_backed_zero,
)
from src.stage28sr2_ros_action_transport import GoalResult


IDENTITY = {
    "certification_identity": "h3-test-certification",
    "frozen_goal_sha256": "a" * 64,
    "formal_runner_sha256": "b" * 64,
}


def _ledger(path: Path) -> FormalGoalLedger:
    return FormalGoalLedger(path, **IDENTITY)


def _final_jit_gate() -> dict[str, Any]:
    query = {
        "passed": True,
        "pre_send_live_zero_proven": True,
        "recorder_bound": True,
        "transport_lost_total": 0,
        "observation_timestamp": "2026-08-07T00:00:00+00:00",
        "observation_timestamp_ns": 1,
        "recorder_pid": 123,
        "recorder_node_identity": "/stage28sr2_recorder_test",
        "runtime_instance_id": "runtime-test",
        "rosbag2_source_commit": ROSBAG2_0_26_11_COMMIT,
        "identity_checks": {"counter_well_formed": True},
        "patched_binary_or_library": {
            "loaded_library_path": "/mnt/d/robotfucker/ros2_overlay/install_h3/lib/librosbag2_transport.so",
            "expected_library_path": "/mnt/d/robotfucker/ros2_overlay/install_h3/lib/librosbag2_transport.so",
            "loaded_library_sha256": "a" * 64,
            "expected_library_sha256": "a" * 64,
        },
    }
    return build_final_jit_pre_send_gate(
        initial_pre_gate=query,
        final_query=query,
        recorder_alive=True,
        controller_active=True,
        action_server_available=True,
        command_source_report={
            "passed": True,
            "unauthorized_action_clients": [],
            "unauthorized_publishers": [],
            "moveit_execution_nodes": [],
            "old_runtime_clients": [],
            "conflicting_project_processes": [],
        },
        manifest_gate={"passed": True},
    )


def _consume_worker(path: str, queue: Any) -> None:
    try:
        _ledger(Path(path)).consume_before_send()
    except Exception as error:
        queue.put((False, type(error).__name__))
    else:
        queue.put((True, None))


def _ready_machine(path: Path) -> PersistentStateMachine:
    machine = PersistentStateMachine(path)
    machine.transition(FormalState.PREFLIGHT_RUNNING)
    machine.transition(FormalState.PREFLIGHT_PASSED)
    machine.transition(FormalState.READY_FOR_LEDGER)
    return machine


def test_ledger_reopen_restart_and_new_output_share_irreversible_state(tmp_path: Path) -> None:
    path = tmp_path / "global" / "ledger.json"
    first = _ledger(path)
    assert first.snapshot()["schema_version"] == LEDGER_SCHEMA_VERSION
    assert first.snapshot()["maximum"] == 1
    first.consume_before_send()
    restarted = _ledger(path)
    assert restarted.snapshot()["consumed"] == 1
    assert restarted.snapshot()["formal_goal_budget"]["goal_budget_consumed"] is True
    assert restarted.snapshot()["consumed_utc"]
    with pytest.raises(RuntimeError, match="already consumed"):
        _ledger(path).consume_before_send()


def test_different_runner_outputs_use_one_injected_global_ledger(tmp_path: Path) -> None:
    path = tmp_path / "global-ledger.json"
    first = FormalRunner(output=tmp_path / "output-a", ledger_path=path, audit_only=True)
    second = FormalRunner(output=tmp_path / "output-b", ledger_path=path, audit_only=True)
    assert first.ledger.path == second.ledger.path == path.resolve()
    first.ledger.consume_before_send()
    assert second.ledger.reread()["formal_goal_budget"]["consumed"] == 1


def test_two_process_consume_race_has_exactly_one_winner(tmp_path: Path) -> None:
    path = tmp_path / "race" / "ledger.json"
    _ledger(path)
    context = multiprocessing.get_context("spawn")
    queue = context.Queue()
    processes = [context.Process(target=_consume_worker, args=(str(path), queue)) for _ in range(2)]
    for process in processes:
        process.start()
    for process in processes:
        process.join(20)
        assert process.exitcode == 0
    results = [queue.get(timeout=5) for _ in processes]
    assert sum(1 for won, _ in results if won) == 1
    assert _ledger(path).snapshot()["consumed"] == 1


@pytest.mark.parametrize("stage", ["consume", "bind", "invoke"])
def test_crash_boundaries_remain_permanently_consumed(tmp_path: Path, stage: str) -> None:
    path = tmp_path / stage / "ledger.json"
    ledger = _ledger(path)
    ledger.consume_before_send()
    if stage in {"bind", "invoke"}:
        ledger.bind_send_goal_async_call()
    if stage == "invoke":
        ledger.record_transport_send_invocation()
    reopened = _ledger(path)
    assert reopened.snapshot()["consumed"] == 1
    with pytest.raises(RuntimeError, match="already consumed"):
        reopened.consume_before_send()


def _live_envelope(*, counter: int = 0, pid: int = 123, runtime: str = "runtime-test") -> str:
    payload = {
        "schema_version": "stage28sr2-recorder-live-loss-v1",
        "recorder_pid": pid,
        "recorder_node_identity": "/stage28sr2_recorder_test",
        "runtime_instance_id": runtime,
        "transport_lost_total": counter,
        "observation_timestamp_ns": 123456,
        "rosbag2_source_commit": ROSBAG2_0_26_11_COMMIT,
        "instrumentation_id": "stage28sr2-h3-recorder-bound-v1",
    }
    envelope = {"passed": True, "payload": payload}
    return "STAGE28SR2_JSON_START\n" + json.dumps(envelope) + "\nSTAGE28SR2_JSON_END\n"


def _recorder_for_live_query(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stdout: str) -> Rosbag2RecorderOrchestrator:
    recorder = Rosbag2RecorderOrchestrator(
        tmp_path,
        runtime_instance_id="runtime-test",
        run_id="test",
        command_runner=lambda command: (0, stdout, ""),
    )
    monkeypatch.setattr(recorder, "_recorder_pid_from_graph", lambda: 123)
    monkeypatch.setattr(recorder, "_loaded_library_provenance", lambda pid: {"passed": True, "loaded_library_sha256": "c" * 64})
    return recorder


def test_patched_recorder_bound_live_zero_passes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    result = _recorder_for_live_query(tmp_path, monkeypatch, _live_envelope()).query_live_transport_loss()
    assert result["passed"]
    assert result["pre_send_live_zero_proven"] is True
    assert result["recorder_bound"] is True
    assert result["transport_lost_total"] == 0


@pytest.mark.parametrize(
    ("stdout", "blocker"),
    [
        (_live_envelope(counter=2), "transport_message_loss_nonzero"),
        (_live_envelope(pid=999), "recorder_live_loss_identity_mismatch_or_malformed"),
        ("", "recorder_live_loss_query_malformed_or_timeout"),
    ],
)
def test_live_loss_nonzero_identity_mismatch_and_unavailable_fail_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stdout: str, blocker: str) -> None:
    result = _recorder_for_live_query(tmp_path, monkeypatch, stdout).query_live_transport_loss()
    assert result["passed"] is False
    assert result["pre_send_live_zero_proven"] is False
    assert result["first_blocker"] == blocker


def test_post_stop_zero_never_becomes_pre_authorization() -> None:
    report = post_stop_source_backed_zero(
        "Recording stopped\n",
        finalized=True,
        process_alive_after_stop=False,
        rosbag2_version="0.26.11",
        recorder_argv=["--log-level", "debug"],
    )
    assert report["valid"] is True
    assert report["pre_send_live_zero_proven"] is False


class _ProductionShapedTransport:
    is_fake_transport = False

    def __init__(self, ledger: FormalGoalLedger) -> None:
        self.ledger = ledger
        self.calls: list[str] = []

    def send_goal_async(self, goal: Any) -> object:
        budget = self.ledger.reread()["formal_goal_budget"]
        assert budget["consumed"] == 1
        assert budget["send_binding_committed"] is True
        assert budget["transport_send_invocation_observed"] is True
        self.calls.append("send_goal_async")
        return object()

    def wait_for_goal_response(self, future: Any, *, timeout_sec: float) -> GoalResult:
        self.calls.append("wait_for_goal_response")
        handle = type("Handle", (), {"accepted": True})()
        return GoalResult(True, "ACCEPTED", None, None, goal_handle=handle)

    def wait_for_result(self, goal_handle: Any, *, timeout_sec: float) -> GoalResult:
        self.calls.append("get_result_async")
        return GoalResult(True, "SUCCEEDED", 0, "", goal_handle=goal_handle)


class _RecorderResult:
    def finalize(self) -> dict[str, Any]:
        return {
            "finalized": True,
            "bag_complete": True,
            "sequential_reader_passed": True,
            "bag_valid": True,
            "recorder_loss_gate": {"passed": True, "transport_lost_total": 0},
        }


def test_production_executor_models_full_ros_future_lifecycle_without_real_send(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path / "ledger.json")
    transport = _ProductionShapedTransport(ledger)
    executor = ProductionFormalExecutor(
        output=tmp_path,
        transport=transport,
        recorder=_RecorderResult(),
        ledger=ledger,
        machine=_ready_machine(tmp_path / "machine.json"),
        goal=object(),
        pre_send_live_loss_gate={"pre_send_live_zero_proven": True, "recorder_bound": True, "transport_lost_total": 0},
        final_jit_gate_factory=_final_jit_gate,
        analysis=lambda result, recorder: {"passed": True},
        test_mode=True,
    )
    certificate = executor.run()
    assert transport.calls == ["send_goal_async", "wait_for_goal_response", "get_result_async"]
    assert certificate["send_goal_async_call_count"] == 1
    assert certificate["execution_mode"] == "production_future_mock_rehearsal"
    assert certificate["real_ros_transport_used"] is False
    assert certificate["Stage_2_9_authorized"] is False


def test_final_readiness_requires_pre_send_live_zero(tmp_path: Path) -> None:
    runtime = {
        "gates": {"x": True, "regression_gate": True},
        "formal_goal_budget": {"maximum": 1, "consumed": 0, "send_attempted": False, "send_goal_async_call_count": 0},
        "real_FJT_goal_requests": 0,
        "pre_send_live_zero_proven": False,
        "pre_send_live_loss_gate": {"pre_send_live_zero_proven": False, "recorder_bound": True, "transport_lost_total": 0},
    }
    regression = {"regression_gate": {"passed": True}, "all_required_tests_passed": True}
    runtime_path, regression_path = tmp_path / "runtime.json", tmp_path / "regression.json"
    runtime_path.write_text(json.dumps(runtime), encoding="utf-8")
    regression_path.write_text(json.dumps(regression), encoding="utf-8")
    certificate = compose(runtime_path, regression_path, tmp_path / "certificate.json")
    assert certificate["READY_FOR_FORMAL_R2_ONE_SHOT"] is False
    assert certificate["Can_we_safely_consume_the_one_and_only_formal_R2_FJT_goal_next"] is False
