"""H4 final-boundary and production-shaped failure-path contracts."""

from __future__ import annotations

import hashlib
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.stage28sr2_formal_runner import FormalState, PersistentStateMachine
from scripts.stage28sr2_production_runner import ProductionFormalExecutor
from src.stage28sr2_final_jit_gate import (
    FINAL_JIT_PRE_SEND_GATE,
    build_final_jit_pre_send_gate,
    production_dispatch_static_audit,
    revalidate_frozen_execution_manifest,
    verify_final_jit_pre_send_gate,
)
from src.stage28sr2_ros_action_transport import GoalResult, ProductionROSActionTransport
from src.stage28sr2_formal_ledger import FormalGoalLedger


def _query(*, pid: int = 42, runtime: str = "runtime-1", library_sha: str = "a" * 64) -> dict:
    return {
        "schema_version": "stage28sr2-recorder-bound-pre-live-loss-gate-v1",
        "passed": True,
        "pre_send_live_zero_proven": True,
        "recorder_bound": True,
        "transport_lost_total": 0,
        "observation_timestamp": "2026-08-07T00:00:00+00:00",
        "observation_timestamp_ns": 123,
        "recorder_pid": pid,
        "recorder_node_identity": "/stage28sr2_recorder_test",
        "runtime_instance_id": runtime,
        "rosbag2_source_commit": "commit-1",
        "identity_checks": {"counter_well_formed": True},
        "patched_binary_or_library": {
            "loaded_library_path": "/mnt/d/robotfucker/ros2_overlay/install_h3/lib/librosbag2_transport.so",
            "expected_library_path": "/mnt/d/robotfucker/ros2_overlay/install_h3/lib/librosbag2_transport.so",
            "loaded_library_sha256": library_sha,
            "expected_library_sha256": library_sha,
        },
    }


def _command_sources() -> dict:
    return {
        "passed": True,
        "unauthorized_action_clients": [],
        "unauthorized_publishers": [],
        "moveit_execution_nodes": [],
        "old_runtime_clients": [],
        "conflicting_project_processes": [],
    }


def _jit_gate() -> dict:
    initial = _query()
    final = _query()
    gate = build_final_jit_pre_send_gate(
        initial_pre_gate=initial,
        final_query=final,
        recorder_alive=True,
        controller_active=True,
        action_server_available=True,
        command_source_report=_command_sources(),
        manifest_gate={"passed": True},
        query_monotonic_ns=time.monotonic_ns(),
    )
    assert gate["gate_name"] == FINAL_JIT_PRE_SEND_GATE
    assert gate["passed"]
    return gate


def test_final_jit_gate_rechecks_identity_and_freshness() -> None:
    gate = _jit_gate()
    assert verify_final_jit_pre_send_gate(gate)["passed"]
    stale = verify_final_jit_pre_send_gate(gate, now_monotonic_ns=gate["query_monotonic_ns"] + 3_000_000_001)
    assert not stale["passed"]
    assert stale["first_blocker"] == "freshness_within_threshold"
    changed = _query(pid=99)
    mismatch = build_final_jit_pre_send_gate(
        initial_pre_gate=_query(),
        final_query=changed,
        recorder_alive=True,
        controller_active=True,
        action_server_available=True,
        command_source_report=_command_sources(),
        manifest_gate={"passed": True},
    )
    assert not mismatch["passed"]
    assert mismatch["first_blocker"] == "same_recorder_pid"


def test_frozen_manifest_revalidates_untracked_sources_and_loaded_library(tmp_path: Path) -> None:
    library = tmp_path / "librosbag2_transport.so"
    source = tmp_path / "source.py"
    library.write_bytes(b"patched")
    source.write_text("source", encoding="utf-8")
    library_sha = hashlib.sha256(library.read_bytes()).hexdigest()
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    manifest = {
        "schema_version": "stage28sr2-frozen-formal-execution-manifest-v1",
        "all_component_hashes_available": True,
        "components": {
            "patched_rosbag2_library": {"path": str(library), "sha256": library_sha, "git_tracked": False},
            "execution_source": {"path": str(source), "sha256": source_sha, "git_tracked": False},
        },
    }
    result = revalidate_frozen_execution_manifest(
        manifest,
        loaded_library_sha256=library_sha,
        loaded_library_path=str(library),
    )
    assert result["passed"]
    source.write_text("tampered", encoding="utf-8")
    changed = revalidate_frozen_execution_manifest(
        manifest,
        loaded_library_sha256=library_sha,
        loaded_library_path=str(library),
    )
    assert not changed["passed"]
    assert changed["first_blocker"] == "all_components_sha_revalidated"


def test_production_dispatch_static_audit_has_one_real_chain() -> None:
    from scripts import stage28sr2_production_runner
    from src import stage28sr2_ros_action_transport

    report = production_dispatch_static_audit(
        {
            "production_runner": Path(stage28sr2_production_runner.__file__),
            "transport": Path(stage28sr2_ros_action_transport.__file__),
        }
    )
    assert report["passed"]


class _FakeProductionTransport(ProductionROSActionTransport):
    is_fake_transport = False

    def __init__(self, *, response: GoalResult | None = None, send_error: Exception | None = None, response_error: Exception | None = None, result_error: Exception | None = None):
        self.response = response or GoalResult(True, "SUCCEEDED", 0, "")
        self.send_error = send_error
        self.response_error = response_error
        self.result_error = result_error
        self.send_count = 0
        self.action_server_available = True
        self.production_ActionClient_created = True

    def bind_formal_send_authorization(self, ledger, pre_send_live_loss_gate):
        return None

    def send_goal_async(self, goal):
        self.send_count += 1
        if self.send_error:
            raise self.send_error
        return object()

    def wait_for_goal_response(self, send_future, *, timeout_sec=None):
        if self.response_error:
            raise self.response_error
        return self.response

    def wait_for_result(self, goal_handle, *, timeout_sec=None):
        if self.result_error:
            raise self.result_error
        return self.response


class _FakeRecorder:
    def __init__(self, *, fail: bool = False):
        self.fail = fail
        self.finalize_count = 0
        self.state = {"metadata_exists_after_finalize": True, "partial_bag_retained": True}

    def finalize(self):
        self.finalize_count += 1
        if self.fail:
            raise RuntimeError("finalize failure")
        self.state.update(
            {
                "finalized": True,
                "bag_complete": True,
                "sequential_reader_passed": True,
                "bag_valid": True,
                "recorder_loss_gate": {"passed": True, "transport_lost_total": 0},
            }
        )
        return dict(self.state)


def _executor(tmp_path: Path, *, transport: _FakeProductionTransport | None = None, recorder: _FakeRecorder | None = None, analysis_passed: bool = True):
    machine = PersistentStateMachine(tmp_path / "state.json")
    machine.transition(FormalState.PREFLIGHT_RUNNING)
    machine.transition(FormalState.PREFLIGHT_PASSED)
    machine.transition(FormalState.READY_FOR_LEDGER)
    recorder = recorder or _FakeRecorder()
    transport = transport or _FakeProductionTransport()
    ledger = FormalGoalLedger(tmp_path / "ledger.json")
    return ProductionFormalExecutor(
        output=tmp_path,
        transport=transport,
        recorder=recorder,
        ledger=ledger,
        machine=machine,
        goal=object(),
        pre_send_live_loss_gate=_query(),
        final_jit_gate_factory=lambda: _jit_gate(),
        frozen_execution_manifest={"components": {}},
        analysis=lambda result, state: {"passed": analysis_passed, "first_blocker": "analysis_failed" if not analysis_passed else None},
    ), ledger, recorder, transport


def test_formal_success_sets_started_passed_and_finalizes_once(tmp_path: Path) -> None:
    executor, ledger, recorder, transport = _executor(tmp_path)
    certificate = executor.run()
    assert certificate["formal_started"]
    assert certificate["formal_status"] == "passed"
    assert certificate["Stage_2_9_authorized"]
    assert ledger.reread()["formal_goal_budget"]["consumed"] == 1
    assert transport.send_count == 1
    assert recorder.finalize_count == 1


@pytest.mark.parametrize(
    "transport,recorder,analysis_passed,expected_blocker",
    [
        (_FakeProductionTransport(response=GoalResult(False, "REJECTED", -1, "rejected")), _FakeRecorder(), True, "goal_rejected"),
        (_FakeProductionTransport(response=GoalResult(None, "CLIENT_TIMEOUT", None, "timeout")), _FakeRecorder(), True, "goal_response_timeout"),
        (_FakeProductionTransport(response=GoalResult(True, "CLIENT_TIMEOUT", None, "timeout")), _FakeRecorder(), True, "action_terminal_status:CLIENT_TIMEOUT"),
        (_FakeProductionTransport(response=GoalResult(True, "ABORTED", -4, "abort")), _FakeRecorder(), True, "action_terminal_status:ABORTED"),
        (_FakeProductionTransport(response=GoalResult(True, "SUCCEEDED", 1, "bad code")), _FakeRecorder(), True, "fjt_error_code_nonzero"),
        (_FakeProductionTransport(), _FakeRecorder(), False, "analysis_failed"),
        (_FakeProductionTransport(), _FakeRecorder(fail=True), True, "recorder_cleanup_exception"),
        (_FakeProductionTransport(send_error=RuntimeError("send")), _FakeRecorder(), True, "production_executor_exception:RuntimeError"),
        (_FakeProductionTransport(response_error=RuntimeError("ack")), _FakeRecorder(), True, "production_executor_exception:RuntimeError"),
        (_FakeProductionTransport(result_error=RuntimeError("result")), _FakeRecorder(), True, "production_executor_exception:RuntimeError"),
    ],
)
def test_formal_failure_paths_consume_once_and_cleanup_once(tmp_path: Path, transport, recorder, analysis_passed, expected_blocker) -> None:
    executor, ledger, recorder, _ = _executor(tmp_path, transport=transport, recorder=recorder, analysis_passed=analysis_passed)
    certificate = executor.run()
    budget = ledger.reread()["formal_goal_budget"]
    assert certificate["formal_started"]
    assert budget["consumed"] == 1
    assert budget["send_attempted"]
    assert certificate["retry_allowed"] is False
    assert recorder.finalize_count == 1
    assert expected_blocker in certificate["first_blocker"] or expected_blocker in str(certificate["formal_analysis"])
    assert certificate["Stage_2_9_authorized"] is False
