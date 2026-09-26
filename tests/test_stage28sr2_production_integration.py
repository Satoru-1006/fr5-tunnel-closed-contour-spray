"""Production-integration contracts; every send path here is fake or blocked."""

from __future__ import annotations

import json
import io
import types
from pathlib import Path

import pytest

from scripts.stage28sr2_formal_runner import FakeTransportOutcome, formal_success_positive_whitelist
from src.stage28sr2_command_source_collector import verify_command_source_report
from src.stage28sr2_formal_ledger import FormalGoalLedger
from src.stage28sr2_live_runtime_evidence import LiveRuntimeEvidenceCollector, RuntimeIdentity, validate_machine_provenance
from src.stage28sr2_recorder_orchestrator import (
    Rosbag2RecorderOrchestrator,
    inspect_live_recorder_endpoints,
    parse_loss_statistics,
)
from src.stage28sr2_ros_action_transport import ProductionROSActionTransport, transport_static_audit
from src.stage28sr2_ros_goal import build_ros_goal, verify_goal_semantics


POINT0 = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]


def small_goal() -> dict:
    return {
        "header": {"stamp": {"sec": 3, "nanosec": 4}},
        "joint_names": ["j1", "j2"],
        "points": [{
            "positions": [0.1, 0.2],
            "velocities": [0.0, 0.1],
            "accelerations": [0.0, 0.0],
            "time_from_start": {"sec": 0, "nanosec": 0},
        }],
        "path_tolerance": [{"name": "j1", "position": 0.01}],
        "goal_tolerance": [],
        "goal_time_tolerance": {"sec": 1, "nanosec": 2},
    }


def test_production_goal_conversion_is_exact_and_digestable() -> None:
    source = small_goal()
    goal = build_ros_goal(source)
    report = verify_goal_semantics(source, goal)
    assert report["passed"]
    assert report["joint_order_equal"]
    assert report["positions_equal"]
    assert report["velocities_equal"]
    assert report["accelerations_equal"]
    assert report["time_from_start_equal"]
    assert report["trajectory_header_semantics_verified"]
    assert report["path_tolerance_semantics_verified"]
    assert report["goal_tolerance_semantics_verified"]
    assert report["goal_time_tolerance_semantics_verified"]
    assert report["max_numeric_difference"] == 0
    assert len(report["canonical_semantic_digest"]) == 64


def test_machine_provenance_rejects_user_fixture(tmp_path: Path) -> None:
    fixture = {"source": "user_fixture", "runtime_instance_id": "fixture"}
    path = tmp_path / "runtime_evidence.json"
    path.write_text(json.dumps(fixture), encoding="utf-8")
    result = validate_machine_provenance(fixture)
    assert not result["passed"]
    assert result["first_blocker"] == "source_machine_collected_live"


def _probe_runner(name: str, command: str, timeout_s: float):
    from src.stage28sr2_live_runtime_evidence import ProbeResult

    if name == "ros_environment":
        stdout = "ROS_DISTRO=jazzy\nROS_VERSION=2\nROS_DOMAIN_ID=7\n"
    elif name == "ros2_pkg_prefix":
        stdout = "/opt/ros/jazzy"
    elif name == "ros2_pkg_version":
        stdout = "4.40.1"
    elif name == "list_controllers":
        stdout = "fairino5_controller joint_trajectory_controller/JointTrajectoryController active\n"
    elif name == "list_hardware_components":
        stdout = "name: mock_components/GenericSystem\nstate: active\n"
    elif name == "controller_parameters":
        stdout = "j1 j2 j3 j4 j5 j6\ncommand_interfaces: [position]\nstate_interfaces: [position]\ninterpolation_method: splines\n"
    elif name == "joint_state_names":
        stdout = "- j1\n- j2\n- j3\n- j4\n- j5\n- j6\n"
    elif name == "joint_state_positions":
        stdout = "- 0.1\n- 0.2\n- 0.3\n- 0.4\n- 0.5\n- 0.6\n"
    else:
        stdout = "/stage28sr2_hardened_runner\n"
    return ProbeResult(name, command, 0, stdout, "")


def test_live_collector_persists_raw_artifacts_and_provenance(tmp_path: Path) -> None:
    identity = RuntimeIdentity("runtime-test", 123, "2026-08-06T00:00:00Z")
    collector = LiveRuntimeEvidenceCollector(tmp_path, identity, probe_runner=_probe_runner)
    collector.collect(phase="initial", frozen_point0=POINT0)
    collector.collect(phase="final", frozen_point0=POINT0)
    evidence = collector.finalize(frozen_point0=POINT0)
    assert evidence["source"] == "machine_collected_live"
    assert evidence["runtime_identity"]["same_runtime_instance"]
    assert validate_machine_provenance(evidence)["passed"]
    assert (tmp_path / "runtime_evidence.json").is_file()
    assert (tmp_path / "raw_live_evidence_manifest.json").is_file()


class _FakeNode:
    def get_fully_qualified_name(self):
        return "/stage28sr2_hardened_runner"

    def destroy_node(self):
        return None


class _FakeFuture:
    def add_done_callback(self, callback):
        self.callback = callback


class _DoneFuture:
    def __init__(self, value):
        self.value = value

    def done(self):
        return True

    def result(self):
        return self.value

    def exception(self):
        return None


class _AcceptedGoalHandle:
    accepted = True

    def get_result_async(self):
        result = types.SimpleNamespace(error_code=0, error_string="")
        return _DoneFuture(types.SimpleNamespace(status=4, result=result))


class _FakeActionClient:
    def __init__(self, node, action_type, name):
        self.name = name
        self.send_count = 0

    def wait_for_server(self, timeout_sec):
        return True

    def send_goal_async(self, goal, feedback_callback=None):
        self.send_count += 1
        return _FakeFuture()


def test_production_action_client_constructs_but_audit_send_is_forbidden(monkeypatch: pytest.MonkeyPatch) -> None:
    rclpy = types.ModuleType("rclpy")
    rclpy.ok = lambda: True
    rclpy.init = lambda args=None: None
    rclpy.create_node = lambda name: _FakeNode()
    action_module = types.ModuleType("rclpy.action")
    action_module.ActionClient = _FakeActionClient
    control_module = types.ModuleType("control_msgs.action")
    class _Goal:
        pass
    class _FJT:
        Goal = _Goal
    control_module.FollowJointTrajectory = _FJT
    monkeypatch.setitem(__import__("sys").modules, "rclpy", rclpy)
    monkeypatch.setitem(__import__("sys").modules, "rclpy.action", action_module)
    monkeypatch.setitem(__import__("sys").modules, "control_msgs.action", control_module)
    transport = ProductionROSActionTransport(audit_only=True)
    assert transport.production_ActionClient_created
    assert transport.wait_for_server()
    with pytest.raises(RuntimeError, match="audit-only"):
        transport.send_goal_async(object())
    assert transport.send_goal_async_call_count == 0
    transport.close()


def test_production_transport_rejects_direct_noncanonical_send_bypass(monkeypatch: pytest.MonkeyPatch) -> None:
    rclpy = types.ModuleType("rclpy")
    rclpy.ok = lambda: True
    rclpy.init = lambda args=None: None
    rclpy.create_node = lambda name: _FakeNode()
    action_module = types.ModuleType("rclpy.action")
    action_module.ActionClient = _FakeActionClient
    control_module = types.ModuleType("control_msgs.action")
    control_module.FollowJointTrajectory = type("FJT", (), {"Goal": object})
    monkeypatch.setitem(__import__("sys").modules, "rclpy", rclpy)
    monkeypatch.setitem(__import__("sys").modules, "rclpy.action", action_module)
    monkeypatch.setitem(__import__("sys").modules, "control_msgs.action", control_module)
    transport = ProductionROSActionTransport(audit_only=False)
    assert transport.wait_for_server()
    with pytest.raises(RuntimeError, match="lacks canonical-ledger PRE authorization"):
        transport.send_goal_async(object())
    assert transport.send_goal_async_call_count == 0
    transport.close()


def test_production_transport_waits_goal_handle_and_result_futures(monkeypatch: pytest.MonkeyPatch) -> None:
    rclpy = types.ModuleType("rclpy")
    rclpy.ok = lambda: True
    rclpy.init = lambda args=None: None
    rclpy.create_node = lambda name: _FakeNode()
    rclpy.spin_once = lambda node, timeout_sec=0.1: None
    action_module = types.ModuleType("rclpy.action")
    action_module.ActionClient = _FakeActionClient
    control_module = types.ModuleType("control_msgs.action")
    control_module.FollowJointTrajectory = type("FJT", (), {"Goal": object})
    monkeypatch.setitem(__import__("sys").modules, "rclpy", rclpy)
    monkeypatch.setitem(__import__("sys").modules, "rclpy.action", action_module)
    monkeypatch.setitem(__import__("sys").modules, "control_msgs.action", control_module)
    transport = ProductionROSActionTransport(audit_only=True)
    handle = _AcceptedGoalHandle()
    response = transport.wait_for_goal_response(_DoneFuture(handle), timeout_sec=1.0)
    assert response.accepted is True
    assert response.terminal_status == "ACCEPTED"
    result = transport.wait_for_result(response.goal_handle, timeout_sec=1.0)
    assert result.terminal_status == "SUCCEEDED"
    assert result.fjt_error_code == 0
    transport.close()


def test_ledger_separates_budget_binding_and_transport_invocation(tmp_path: Path) -> None:
    ledger = FormalGoalLedger(tmp_path / "ledger.json")
    ledger.consume_before_send()
    consumed = ledger.reread()["formal_goal_budget"]
    assert consumed["consumed"] == 1
    assert consumed["send_goal_async_call_count"] == 0
    ledger.bind_send_goal_async_call()
    bound = ledger.reread()["formal_goal_budget"]
    assert bound["send_attempted"] is True
    assert bound["send_binding_committed"] is True
    assert bound["send_goal_async_call_count"] == 0
    ledger.record_transport_send_invocation()
    ledger.record_goal_response(acknowledged=False, accepted=None)
    sent = ledger.reread()["formal_goal_budget"]
    assert sent["transport_send_invocation_observed"] is True
    assert sent["goal_request_acknowledged"] is False
    assert sent["goal_accepted"] is False


@pytest.mark.parametrize(
    "outcome",
    [
        FakeTransportOutcome(accepted=None, action_terminal_status=None, fjt_error_code=None),
        FakeTransportOutcome(accepted=True, action_terminal_status="UNKNOWN", fjt_error_code=0),
        FakeTransportOutcome(accepted=True, action_terminal_status="ACCEPTED", fjt_error_code=0),
        FakeTransportOutcome(accepted=True, action_terminal_status="EXECUTING", fjt_error_code=0),
        FakeTransportOutcome(accepted=True, action_terminal_status="SUCCEEDED", fjt_error_code=None),
        FakeTransportOutcome(accepted=True, action_terminal_status="SUCCEEDED", fjt_error_code=1),
    ],
)
def test_formal_success_whitelist_rejects_adversarial_statuses(outcome: FakeTransportOutcome) -> None:
    assert not formal_success_positive_whitelist(outcome)["passed"]


def test_formal_success_whitelist_accepts_only_succeeded_code_zero() -> None:
    result = formal_success_positive_whitelist(FakeTransportOutcome())
    assert result["passed"]
    assert result["accepted_status_whitelist"] == ["SUCCEEDED"]
    assert result["accepted_fjt_error_codes"] == [0]


def test_command_source_authorized_client_alone_passes() -> None:
    report = {
        "observed_from_runtime_graph": True,
        "action_clients": ["/stage28sr2_hardened_runner"],
        "unauthorized_action_clients": [],
        "unauthorized_publishers": [],
        "moveit_execution_nodes": [],
        "old_runtime_clients": [],
        "conflicting_project_processes": [],
    }
    assert verify_command_source_report(report, authorized_node="/stage28sr2_hardened_runner")["passed"]
    report["action_clients"].append("/old_sender")
    assert not verify_command_source_report(report, authorized_node="/stage28sr2_hardened_runner")["passed"]


def test_command_source_collector_authorizes_persistent_observer_pid(tmp_path: Path) -> None:
    from src.stage28sr2_command_source_collector import LiveCommandSourceCollector

    def runner(command: str):
        if command == "ps -ef":
            return 0, "robot 4242 1 0 00:00 ? 00:00:00 python3 stage28sr2_persistent_graph_observer.py\n", ""
        return 0, "/stage28sr2_hardened_runner\nAction clients: 1\n /stage28sr2_hardened_runner\nAction servers: 1\n /fairino5_controller\nPublisher count: 0\n", ""

    report = LiveCommandSourceCollector(tmp_path, runner=runner).collect(
        authorized_action_client_nodes={"/stage28sr2_hardened_runner"},
        authorized_process_ids={4242},
    )
    assert report["conflicting_project_processes"] == []
    assert report["authorized_process_ids"] == [4242]


class _FakeProcess:
    pid = 456

    def __init__(self):
        self.alive = True
        self.stdout = io.StringIO(
            "Number of messages lost on the transport layer: 0\n"
            "Number of messages lost on the recorder layer: 0\n"
        )

    def poll(self):
        return None if self.alive else 0

    def send_signal(self, signal):
        self.alive = False

    def wait(self, timeout=None):
        self.alive = False
        return 0


def test_recorder_readiness_and_safe_finalize(tmp_path: Path) -> None:
    process = _FakeProcess()
    capability = {
        "rosbag2_version": "0.26.11",
        "max_cache_size_zero_direct_write_supported": True,
        "stock_recorder_side_machine_readable_zero_counter": False,
        "passed": True,
    }

    def command_runner(command: str):
        if command.startswith("df -T"):
            return 0, "Filesystem Type\n/dev/sda ext4\next4\n", ""
        if command.startswith("df -Pk"):
            return 0, "/dev/sda 1000000 1 999999 1% /\n", ""
        if command.startswith("test -f"):
            return 0, "", ""
        if command.startswith("python3"):
            payload = {"passed": True, "reader_api": "rosbag2_py.SequentialReader", "read_to_eof": True, "required_topics_present": True, "required_topic_counts_positive": True, "counts_match_metadata": True, "total_count_matches_metadata": True, "reader_counts": {}, "metadata_counts": {}}
            return 0, "STAGE28SR2_JSON_START\n" + json.dumps(payload) + "\nSTAGE28SR2_JSON_END\n", ""
        return 0, "", ""

    recorder = Rosbag2RecorderOrchestrator(
        tmp_path,
        runtime_instance_id="runtime-test",
        process_factory=lambda *args, **kwargs: process,
        command_runner=command_runner,
        capability=capability,
    )
    started = recorder.start()
    assert started["process_alive"]
    ready = recorder.verify_readiness(minimum_free_bytes=1)
    assert ready["ready"]
    finalized = recorder.finalize()
    assert finalized["finalized"]
    assert finalized["offline_readable"]
    assert finalized["recorder_loss_gate"]["passed"]


def test_endpoint_qos_provenance_is_bound_to_each_recorder_file(tmp_path: Path) -> None:
    topic_blocks = []
    for topic, type_name in (
        ("/joint_states", "sensor_msgs/msg/JointState"),
        ("/fairino5_controller/controller_state", "control_msgs/msg/JointTrajectoryControllerState"),
        ("/tf", "tf2_msgs/msg/TFMessage"),
        ("/tf_static", "tf2_msgs/msg/TFMessage"),
    ):
        topic_blocks.append(
            f"Type: {type_name}\nPublisher count: 0\nSubscription count: 1\n"
            "Node name: recorder_test\nNode namespace: /\nEndpoint type: SUBSCRIPTION\n"
            "GID: 01.02.03.04\nQoS profile:\n  Reliability: RELIABLE\n"
            "  History (Depth): UNKNOWN\n  Durability: VOLATILE\n"
        )

    def runner(_command: str):
        chunks = []
        for index, block in enumerate(topic_blocks):
            chunks.append(f"STAGE28SR2_TOPIC_BEGIN_{index}\n{block}\nSTAGE28SR2_TOPIC_END_{index}:0")
        return 0, "\n".join(chunks), ""

    def write_qos(path: Path, history: str, depth: int) -> None:
        entries = []
        for topic in ("/joint_states", "/fairino5_controller/controller_state", "/tf", "/tf_static"):
            entries.extend(
                [
                    f"{topic}:",
                    "  reliability: reliable",
                    f"  history: {history}",
                    f"  depth: {depth}",
                    "  durability: volatile",
                ]
            )
        path.write_text("\n".join(entries) + "\n", encoding="utf-8")

    qos_a = tmp_path / "qos_A.yaml"
    qos_b = tmp_path / "qos_B.yaml"
    write_qos(qos_a, "keep_all", 1000)
    write_qos(qos_b, "keep_last", 256)
    reports = [
        inspect_live_recorder_endpoints(
            run_id=label,
            recorder_node_fqn="/recorder_test",
            runtime_instance_id="runtime-test",
            recorder_pid=123,
            qos_override_path=path,
            runner=runner,
        )
        for label, path in (("A", qos_a), ("B", qos_b))
    ]
    assert reports[0]["endpoints"]["/joint_states"]["requested_qos_configured"] == {
        "reliability": "reliable",
        "history": "keep_all",
        "depth": 1000,
        "durability": "volatile",
    }
    assert reports[1]["endpoints"]["/joint_states"]["requested_qos_configured"]["history"] == "keep_last"
    assert reports[1]["endpoints"]["/joint_states"]["requested_qos_configured"]["depth"] == 256
    assert reports[0]["qos_override_sha256"] != reports[1]["qos_override_sha256"]
    assert reports[0]["endpoints"]["/joint_states"]["requested_qos_source"]["qos_override_path"] == str(qos_a.resolve())
    assert reports[1]["endpoints"]["/joint_states"]["requested_qos_source"]["qos_override_path"] == str(qos_b.resolve())


@pytest.mark.parametrize(
    ("label", "process_output", "passed", "first_blocker"),
    [
        ("zero_loss", "Number of messages lost on the transport layer: 0\nNumber of messages lost on the recorder layer: 0\n", True, None),
        ("transport_loss", "Number of messages lost on the transport layer: 1\nNumber of messages lost on the recorder layer: 0\n", False, "transport_message_loss_nonzero"),
        ("recorder_loss", "Number of messages lost on the transport layer: 0\nNumber of messages lost on the recorder layer: 1\n", False, "recorder_message_loss_nonzero"),
        ("unavailable", "", False, "recorder_message_loss_statistics_unavailable"),
        ("mcap_readable_but_loss", "storage_identifier: mcap\nNumber of messages lost on the transport layer: 2\nNumber of messages lost on the recorder layer: 0\n", False, "transport_message_loss_nonzero"),
        ("metadata_without_loss", "storage_identifier: mcap\n", False, "recorder_message_loss_statistics_parse_failure"),
    ],
)
def test_recorder_loss_gate_is_machine_bound_and_fail_closed(label: str, process_output: str, passed: bool, first_blocker: str | None) -> None:
    result = parse_loss_statistics(process_output)
    assert result["passed"] is passed, label
    assert result["first_blocker"] == first_blocker, label


def test_structured_per_topic_zero_loss_is_persistable(tmp_path: Path) -> None:
    report = tmp_path / "message_loss_statistics.json"
    report.write_text(
        json.dumps(
            {
                "topics": {
                    "/joint_states": {"messages_lost_in_transport": 0, "messages_lost_in_recorder": 0},
                    "/tf": {"messages_lost_in_transport": 0, "messages_lost_in_recorder": 0},
                }
            }
        ),
        encoding="utf-8",
    )
    result = parse_loss_statistics("", structured_reports=(report,), cache_enabled=True)
    assert result["passed"]
    assert result["source"] == "structured_json:message_loss_statistics.json"
    assert set(result["per_topic"]) == {"/joint_states", "/tf"}


@pytest.mark.parametrize(
    ("transport", "recorder", "cache_enabled", "passed", "first_blocker"),
    [
        (2, None, False, False, "transport_message_loss_nonzero"),
        (0, None, True, False, "recorder_message_loss_statistics_incomplete"),
        (0, None, False, True, None),
        (None, None, False, False, "transport_message_loss_statistics_incomplete"),
        (0, 1, True, False, "recorder_message_loss_nonzero"),
        (0, 0, True, True, None),
    ],
)
def test_loss_gate_transport_first_precedence_and_zero_cache_contract(transport, recorder, cache_enabled, passed, first_blocker) -> None:
    payload = {}
    if transport is not None:
        payload["messages_lost_in_transport"] = transport
    if recorder is not None:
        payload["messages_lost_in_recorder"] = recorder
    result = parse_loss_statistics(json.dumps(payload), cache_enabled=cache_enabled)
    assert result["passed"] is passed
    assert result["first_blocker"] == first_blocker
    assert result["transport_lost_total"] == transport
    assert result["recorder_lost_total"] == recorder


def test_static_transport_contract_is_complete() -> None:
    assert transport_static_audit()["passed"]
