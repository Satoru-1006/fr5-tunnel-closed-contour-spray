from __future__ import annotations

from src.stage28sr2_persistent_graph import compare_runtime_qos, summarize_persistent_graph
from src.stage28sr2_recorder_orchestrator import post_stop_source_backed_zero
from src.stage28sr2_recorder_orchestrator import Rosbag2RecorderOrchestrator
from src.stage28sr2_transport_mode import DEFAULT_UDP_SHM, UDP4_ONLY, build_transport_environment
from scripts.stage28sr2_production_runner import run_regression_gates


TOPIC = "/joint_states"
RECORDER = "/stage28sr2_recorder_test"


def endpoint(gid: str, endpoint_type: str = "publisher", *, reliability: str | None = "reliable", durability: str | None = "volatile"):
    return {
        "topic_name": TOPIC,
        "endpoint_type": endpoint_type,
        "node_name": "stage28sr2_recorder_test" if endpoint_type == "subscription" else "publisher",
        "node_namespace": "/",
        "gid": gid,
        "reliability": reliability,
        "durability": durability,
        "runtime_observable": {
            "gid": True,
            "reliability": reliability is not None,
            "durability": durability is not None,
            "history": False,
            "depth": False,
        },
    }


def samples(gids: list[str | None], *, include_recorder: bool = True):
    result = []
    for index, gid in enumerate(gids):
        endpoints = [] if gid is None else [endpoint(gid)]
        if include_recorder:
            endpoints.append(endpoint("reader", "subscription"))
        result.append({"timestamp_monotonic": index * 0.25, "endpoints": endpoints})
    return result


def summarize(values):
    return summarize_persistent_graph(
        values,
        required_topics=[TOPIC],
        recorder_node_fqn=RECORDER,
        configured_qos={TOPIC: {"reliability": "reliable", "durability": "volatile"}},
        observer_started_before_runtime=True,
        observer_pid=123,
    )


def test_persistent_observer_uses_one_participant_and_stable_gid() -> None:
    report = summarize(samples(["writer"] * 5))
    assert report["observer_single_participant"]
    assert report["topics"][TOPIC]["consecutive_samples"] == 5
    assert report["topics"][TOPIC]["stable_gid"]
    assert report["passed"]


def test_gid_change_fails_until_five_new_consecutive_samples() -> None:
    report = summarize(samples(["a", "a", "a", "b", "b", "b", "b"]))
    assert not report["topics"][TOPIC]["stable_gid"]
    assert not report["passed"]


def test_temporary_endpoint_absence_resets_consecutive_window() -> None:
    report = summarize(samples(["a", "a", None, "a", "a", "a", "a"]))
    assert report["topics"][TOPIC]["consecutive_samples"] == 4
    assert not report["passed"]


def test_qos_match_mismatch_and_unknown_are_not_fabricated() -> None:
    assert compare_runtime_qos(endpoint("x"), {"reliability": "reliable", "durability": "volatile"})["passed"]
    mismatch = compare_runtime_qos(endpoint("x", reliability="best_effort"), {"reliability": "reliable", "durability": "volatile"})
    assert mismatch["reliability"]["status"] == "fail"
    unknown = compare_runtime_qos(endpoint("x", reliability=None), {"reliability": "reliable", "durability": "volatile"})
    assert unknown["reliability"] == {"status": "unobservable", "configured": "reliable", "observed": None}
    assert unknown["history"]["status"] == "runtime_unobservable_from_ros_graph"
    assert unknown["depth"]["status"] == "runtime_unobservable_from_ros_graph"


def test_post_stop_zero_requires_complete_normal_stop_and_unfiltered_warning() -> None:
    report = post_stop_source_backed_zero(
        "[INFO] Recording stopped\n",
        finalized=True,
        process_alive_after_stop=False,
        rosbag2_version="0.26.11",
        recorder_argv=["ros2", "bag", "record", "--log-level", "debug"],
    )
    assert report["valid"]
    assert report["value"] == 0
    assert report["evidence_type"] == "source_backed_negative_proof"
    assert report["pre_send_live_zero_proven"] is False
    nonzero = post_stop_source_backed_zero(
        "Recording stopped\nNumber of messages lost on the transport layer: 1\n",
        finalized=True,
        process_alive_after_stop=False,
        rosbag2_version="0.26.11",
        recorder_argv=["--log-level", "debug"],
    )
    assert not nonzero["valid"]


def test_transport_modes_and_configuration_conflicts_fail_closed() -> None:
    default_env, default = build_transport_environment(DEFAULT_UDP_SHM, {})
    assert default["passed"] and "FASTDDS_BUILTIN_TRANSPORTS" not in default_env
    udp_env, udp = build_transport_environment(UDP4_ONLY, {})
    assert udp["passed"] and udp_env["FASTDDS_BUILTIN_TRANSPORTS"] == "UDPv4"
    _, conflict = build_transport_environment(UDP4_ONLY, {"FASTDDS_DEFAULT_PROFILES_FILE": "/tmp/other.xml"})
    assert not conflict["passed"]
    assert conflict["first_blocker"] == "fastdds_transport_configuration_conflict"


def test_recorder_explicitly_includes_unpublished_topics(tmp_path) -> None:
    recorder = Rosbag2RecorderOrchestrator(tmp_path, runtime_instance_id="test")
    argv = recorder._rosbag_argv()
    assert "--include-unpublished-topics" in argv


def test_external_regression_evidence_is_sha_bound(tmp_path) -> None:
    evidence = tmp_path / "test_results.txt"
    evidence.write_text("===== focused =====\n52 passed in 1.0s\n===== full =====\n428 passed in 2.0s\n", encoding="utf-8")
    result = run_regression_gates(tmp_path / "certificate", skip_tests=False, external_evidence=evidence)
    assert result["regression_gate"]["passed"]
    assert result["focused_stage28sr2_suite"]["collected"] == 52
    assert result["full_regression_suite"]["collected"] == 428
    assert len(result["external_regression_evidence"]["sha256"]) == 64
