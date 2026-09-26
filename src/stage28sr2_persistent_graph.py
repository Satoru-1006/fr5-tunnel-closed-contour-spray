"""Persistent ROS graph evidence for Stage 2.8S-R2 ZERO-GOAL gates.

The runtime process that writes the raw JSONL owns one rclpy node for its
entire lifetime.  This module contains the ROS-independent aggregation and
QoS comparison logic so the certification policy can be regression tested on
hosts without ROS installed.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SAMPLE_INTERVAL_SEC = 0.25
MIN_CONSECUTIVE_SAMPLES = 5


def _normalise_qos(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    for prefix in ("qospolicyreliability.", "qospolicydurability.", "reliabilitypolicy.", "durabilitypolicy."):
        if text.startswith(prefix):
            text = text[len(prefix) :]
    aliases = {
        "reliable": "reliable",
        "best_effort": "best_effort",
        "besteffort": "best_effort",
        "volatile": "volatile",
        "transient_local": "transient_local",
        "transientlocal": "transient_local",
    }
    return aliases.get(text)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    result: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and isinstance(value.get("endpoints"), list):
            result.append(value)
    return result


def _topic_publisher_gids(sample: Mapping[str, Any], topic: str) -> tuple[str, ...]:
    return tuple(sorted({
        str(endpoint.get("gid"))
        for endpoint in sample.get("endpoints", [])
        if isinstance(endpoint, Mapping)
        and endpoint.get("topic_name") == topic
        and endpoint.get("endpoint_type") == "publisher"
        and isinstance(endpoint.get("gid"), str)
        and endpoint.get("gid")
    }))


def _publisher_stability(samples: Sequence[Mapping[str, Any]], topic: str, minimum: int) -> dict[str, Any]:
    best_count = 0
    best_gids: tuple[str, ...] = ()
    best_start: float | None = None
    best_end: float | None = None
    current_gids: tuple[str, ...] = ()
    current_count = 0
    current_start: float | None = None
    for sample in samples:
        timestamp = float(sample.get("timestamp_monotonic", 0.0))
        gids = _topic_publisher_gids(sample, topic)
        if not gids:
            current_gids, current_count, current_start = (), 0, None
            continue
        if gids == current_gids:
            current_count += 1
        else:
            current_gids, current_count, current_start = gids, 1, timestamp
        if current_count > best_count:
            best_count, best_gids, best_start, best_end = current_count, gids, current_start, timestamp
    duration = max(0.0, float(best_end or 0.0) - float(best_start or 0.0)) if best_count else 0.0
    value: str | list[str] | None = None
    if len(best_gids) == 1:
        value = best_gids[0]
    elif best_gids:
        value = list(best_gids)
    return {
        "publisher_seen": bool(best_gids),
        "stable_gid": bool(best_gids and best_count >= minimum),
        "stable_gid_value": value,
        "consecutive_samples": best_count,
        "stability_duration_sec": duration,
        "required_consecutive_samples": minimum,
    }


def _recorder_endpoint(
    samples: Sequence[Mapping[str, Any]], topic: str, recorder_node_fqn: str
) -> dict[str, Any] | None:
    for sample in reversed(samples):
        for endpoint in sample.get("endpoints", []):
            if not isinstance(endpoint, Mapping):
                continue
            node_fqn = f"{str(endpoint.get('node_namespace', '')).rstrip('/')}/{endpoint.get('node_name', '')}"
            node_fqn = "/" + node_fqn.strip("/")
            if (
                endpoint.get("topic_name") == topic
                and endpoint.get("endpoint_type") == "subscription"
                and node_fqn == recorder_node_fqn
                and endpoint.get("gid")
            ):
                return dict(endpoint)
    return None


def compare_runtime_qos(observed: Mapping[str, Any] | None, configured: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key in ("reliability", "durability"):
        actual = _normalise_qos(observed.get(key) if observed else None)
        expected = _normalise_qos(configured.get(key))
        if actual is None:
            status = "unobservable"
        else:
            status = "pass" if actual == expected else "fail"
        result[key] = {"status": status, "configured": expected, "observed": actual}
    for key in ("history", "depth"):
        result[key] = {"status": "runtime_unobservable_from_ros_graph"}
    result["passed"] = all(result[key]["status"] == "pass" for key in ("reliability", "durability"))
    return result


def summarize_persistent_graph(
    samples: Sequence[Mapping[str, Any]],
    *,
    required_topics: Sequence[str],
    recorder_node_fqn: str | None = None,
    configured_qos: Mapping[str, Mapping[str, Any]] | None = None,
    observer_started_before_runtime: bool,
    observer_pid: int | None = None,
    minimum_consecutive_samples: int = MIN_CONSECUTIVE_SAMPLES,
    require_publishers: bool = True,
) -> dict[str, Any]:
    topics: dict[str, Any] = {}
    for topic in required_topics:
        item = _publisher_stability(samples, topic, minimum_consecutive_samples)
        if recorder_node_fqn is not None:
            endpoint = _recorder_endpoint(samples, topic, recorder_node_fqn)
            item["recorder_subscription_seen"] = endpoint is not None
            item["recorder_subscription"] = endpoint
            item["qos_runtime_match"] = compare_runtime_qos(
                endpoint, (configured_qos or {}).get(topic, {})
            )
        topics[topic] = item
    publisher_gate = all(item["stable_gid"] for item in topics.values())
    subscription_gate = recorder_node_fqn is None or all(
        item.get("recorder_subscription_seen") is True for item in topics.values()
    )
    qos_gate = configured_qos is None or all(
        item.get("qos_runtime_match", {}).get("passed") is True for item in topics.values()
    )
    return {
        "schema_version": "stage28sr2-persistent-graph-summary-v1",
        "sample_interval_sec": SAMPLE_INTERVAL_SEC,
        "minimum_consecutive_samples": minimum_consecutive_samples,
        "sample_count": len(samples),
        "observer_started_before_runtime": observer_started_before_runtime,
        "observer_single_participant": observer_pid is not None,
        "observer_pid": observer_pid,
        "topics": topics,
        "required_publishers_seen": all(item["publisher_seen"] for item in topics.values()),
        "required_publisher_gid_stability": publisher_gate,
        "required_recorder_subscriptions_seen": subscription_gate,
        "qos_runtime_provenance_passed": qos_gate,
        "passed": bool(
            observer_started_before_runtime
            and observer_pid is not None
            and (publisher_gate or not require_publishers)
            and subscription_gate
            and qos_gate
        ),
    }


def wait_for_persistent_gate(
    raw_path: Path,
    *,
    required_topics: Sequence[str],
    recorder_node_fqn: str | None,
    configured_qos: Mapping[str, Mapping[str, Any]] | None,
    observer_started_before_runtime: bool,
    observer_pid: int | None,
    timeout_sec: float = 30.0,
    require_publishers: bool = True,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_sec
    result: dict[str, Any] = {"passed": False, "first_blocker": "persistent_graph_observation_pending"}
    while time.monotonic() < deadline:
        result = summarize_persistent_graph(
            load_jsonl(raw_path),
            required_topics=required_topics,
            recorder_node_fqn=recorder_node_fqn,
            configured_qos=configured_qos,
            observer_started_before_runtime=observer_started_before_runtime,
            observer_pid=observer_pid,
            require_publishers=require_publishers,
        )
        if result["passed"]:
            return result
        time.sleep(SAMPLE_INTERVAL_SEC)
    if require_publishers:
        result["first_blocker"] = next(
            (f"persistent_graph_topic_failed:{topic}" for topic, value in result.get("topics", {}).items() if not value.get("stable_gid")),
            "persistent_graph_gate_failed",
        )
    else:
        result["first_blocker"] = next(
            (f"persistent_recorder_subscription_failed:{topic}" for topic, value in result.get("topics", {}).items() if not value.get("recorder_subscription_seen")),
            "persistent_graph_gate_failed",
        )
    return result
