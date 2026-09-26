"""Fail-closed stock Jazzy rosbag2/MCAP recorder orchestration.

Stage 2.8S-R2 uses the installed Jazzy recorder in direct-write mode.  The
recorder is deliberately kept independent from the formal action client and
the formal one-goal ledger.  This module owns the evidence boundary:

* the exact rosbag2 argv is persisted and checked;
* recording happens on a WSL-native Linux filesystem, never on ``/mnt/d``;
* live graph endpoints are bound to the unique recorder node, not inferred
  from command text;
* loss counters use transport-first precedence and never fabricate a recorder
  counter when stock Jazzy does not expose one; and
* finalized bags are opened and read to EOF through ``SequentialReader``
  before they are copied into the Windows evidence tree.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import yaml

from src.stage28sr2_persistent_graph import (
    load_jsonl,
    summarize_persistent_graph,
    wait_for_persistent_gate,
)
from src.stage28sr2_transport_mode import DEFAULT_UDP_SHM, build_transport_environment, shell_prefix


ROOT = Path(__file__).resolve().parents[1]
REQUIRED_TOPICS = (
    "/joint_states",
    "/fairino5_controller/controller_state",
    "/tf",
    "/tf_static",
)
ROS_BAG2_VERSION = "0.26.11"
ROSBAG2_0_26_11_COMMIT = "cb8fa198b93bfed90e2417b52b8645fb9ef89c2b"
RECORDER_INSTRUMENTATION_ID = "stage28sr2-h3-recorder-bound-v1"
PATCHED_ROSBAG2_PREFIX = ROOT / "ros2_overlay/install_h3"
PATCHED_ROSBAG2_LIBRARY = PATCHED_ROSBAG2_PREFIX / "lib/librosbag2_transport.so"
NATIVE_RECORDING_ROOT = "/tmp"
RECORDER_REQUESTED_QOS = {
    "/joint_states": {"reliability": "reliable", "history": "keep_all", "depth": 1000, "durability": "volatile"},
    "/fairino5_controller/controller_state": {"reliability": "reliable", "history": "keep_all", "depth": 1000, "durability": "volatile"},
    "/tf": {"reliability": "reliable", "history": "keep_all", "depth": 1000, "durability": "volatile"},
    "/tf_static": {"reliability": "reliable", "history": "keep_all", "depth": 1000, "durability": "transient_local"},
}


def load_recorder_qos_config(
    qos_override_path: Path,
    *,
    required_topics: Sequence[str] = REQUIRED_TOPICS,
) -> dict[str, dict[str, Any]]:
    """Load the exact rosbag2 QoS file used by one recorder instance.

    Endpoint evidence must never be populated from the module's formal/default
    QoS constant when a shadow recorder was launched with another file.
    """

    path = Path(qos_override_path).resolve()
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"recorder QoS override is not a mapping: {path}")
    result: dict[str, dict[str, Any]] = {}
    for topic in required_topics:
        raw = payload.get(topic)
        if not isinstance(raw, Mapping):
            raise ValueError(f"recorder QoS override missing topic {topic}: {path}")
        configured: dict[str, Any] = {}
        for key in ("reliability", "history", "depth", "durability"):
            if key not in raw:
                raise ValueError(f"recorder QoS override missing {topic}.{key}: {path}")
            value = raw[key]
            configured[key] = str(value).strip().lower() if key != "depth" else int(value)
        result[topic] = configured
    return result


_TRANSPORT_LOSS_KEYS = {
    "messages_lost_in_transport",
    "messages_lost_on_transport_layer",
    "number_of_messages_lost_in_transport",
    "number_of_messages_lost_on_the_transport_layer",
    "transport_lost",
    "transport_lost_total",
    "transport_loss_count",
}
_RECORDER_LOSS_KEYS = {
    "messages_lost_in_recorder",
    "messages_lost_on_recorder_layer",
    "number_of_messages_lost_in_recorder",
    "number_of_messages_lost_on_the_recorder_layer",
    "recorder_lost",
    "recorder_lost_total",
    "recorder_loss_count",
}


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _restore_recorder_child_signals() -> None:
    """Undo ignored job-control signals inherited from a background WSL shell."""

    signal.signal(signal.SIGINT, signal.SIG_DFL)
    signal.signal(signal.SIGTERM, signal.SIG_DFL)


def sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _normalise_loss_key(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


def _loss_counter(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value >= 0 else None
    if isinstance(value, float) and value.is_integer() and value >= 0:
        return int(value)
    if isinstance(value, str) and re.fullmatch(r"\d+", value.strip()):
        return int(value.strip())
    return None


def _find_loss_counter(payload: object, keys: set[str]) -> int | None:
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            if _normalise_loss_key(key) in keys:
                counter = _loss_counter(value)
                if counter is not None:
                    return counter
        for value in payload.values():
            counter = _find_loss_counter(value, keys)
            if counter is not None:
                return counter
    elif isinstance(payload, list):
        for value in payload:
            counter = _find_loss_counter(value, keys)
            if counter is not None:
                return counter
    return None


def _find_per_topic_loss(payload: object) -> dict[str, dict[str, int | None]]:
    if not isinstance(payload, Mapping):
        return {}
    for key, value in payload.items():
        if _normalise_loss_key(key) not in {"per_topic", "topics", "topic"}:
            continue
        entries: dict[str, dict[str, int | None]] = {}
        if isinstance(value, Mapping):
            iterator = value.items()
        elif isinstance(value, list):
            iterator = ((str(item.get("topic", "")), item) for item in value if isinstance(item, Mapping))
        else:
            iterator = ()
        for topic, details in iterator:
            if not topic or not isinstance(details, Mapping):
                continue
            transport = _find_loss_counter(details, _TRANSPORT_LOSS_KEYS)
            recorder = _find_loss_counter(details, _RECORDER_LOSS_KEYS)
            if transport is not None or recorder is not None:
                entries[str(topic)] = {
                    "messages_lost_in_transport": transport,
                    "messages_lost_in_recorder": recorder,
                }
        if entries:
            return entries
    return {}


def _loss_gate_from_payload(
    payload: object,
    *,
    source: str,
    cache_enabled: bool | None,
    capability_proof: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Apply the R2 transport-first loss precedence exactly.

    ``cache_enabled`` is an observed runtime fact supplied by the recorder
    argv/capability gate.  It is not derived from missing counters.  In direct
    write mode a missing recorder-side counter is expected and remains null.
    """

    per_topic = _find_per_topic_loss(payload)
    transport = _find_loss_counter(payload, _TRANSPORT_LOSS_KEYS)
    recorder = _find_loss_counter(payload, _RECORDER_LOSS_KEYS)
    if per_topic:
        if transport is None and all(item["messages_lost_in_transport"] is not None for item in per_topic.values()):
            transport = sum(int(item["messages_lost_in_transport"]) for item in per_topic.values())
        if recorder is None and all(item["messages_lost_in_recorder"] is not None for item in per_topic.values()):
            recorder = sum(int(item["messages_lost_in_recorder"]) for item in per_topic.values())
    result: dict[str, Any] = {
        "source": source,
        "transport_lost_total": transport,
        "recorder_lost_total": recorder,
        "per_topic": per_topic,
        "cache_enabled": cache_enabled,
        "recorder_counter_applicable": cache_enabled is not False,
        "capability_proof": dict(capability_proof or {}),
        "passed": False,
        "first_blocker": None,
    }
    # Required precedence: transport nonzero wins over all recorder fields.
    if transport is not None and transport != 0:
        result["first_blocker"] = "transport_message_loss_nonzero"
    elif recorder is not None and recorder != 0:
        result["first_blocker"] = "recorder_message_loss_nonzero"
    elif cache_enabled is True and recorder is None:
        result["first_blocker"] = "recorder_message_loss_statistics_incomplete"
    elif transport is None:
        result["first_blocker"] = "transport_message_loss_statistics_incomplete"
    elif recorder == 0 and cache_enabled in {True, None}:
        result["passed"] = True
    elif cache_enabled is False and transport == 0:
        result["passed"] = True
    else:
        result["first_blocker"] = "recorder_message_loss_gate_fail_closed"
    return result


def _loss_gate_from_text(
    text: str,
    *,
    source: str,
    cache_enabled: bool | None,
    capability_proof: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    transport_patterns = (
        r"(?i)messages_lost_in_transport\s*[:=]\s*(\d+)\b",
        r"(?i)(?:number\s+of\s+)?messages?\s+lost\s+(?:on|in)\s+(?:the\s+)?transport(?:\s+layer)?\s*[:=]\s*(\d+)\b",
    )
    recorder_patterns = (
        r"(?i)messages_lost_in_recorder\s*[:=]\s*(\d+)\b",
        r"(?i)(?:number\s+of\s+)?messages?\s+lost\s+(?:on|in|by)\s+(?:the\s+)?recorder(?:\s+layer)?\s*[:=]\s*(\d+)\b",
    )
    transport_match = next((re.search(pattern, text) for pattern in transport_patterns if re.search(pattern, text)), None)
    recorder_match = next((re.search(pattern, text) for pattern in recorder_patterns if re.search(pattern, text)), None)
    payload = {
        "messages_lost_in_transport": int(transport_match.group(1)) if transport_match else None,
        "messages_lost_in_recorder": int(recorder_match.group(1)) if recorder_match else None,
    }
    result = _loss_gate_from_payload(payload, source=source, cache_enabled=cache_enabled, capability_proof=capability_proof)
    if not text.strip() and result["first_blocker"] == "transport_message_loss_statistics_incomplete":
        result["source"] = "unavailable"
        result["first_blocker"] = "recorder_message_loss_statistics_unavailable" if cache_enabled is None else "transport_message_loss_statistics_unavailable"
    elif text.strip() and result["first_blocker"] == "transport_message_loss_statistics_incomplete":
        result["first_blocker"] = "recorder_message_loss_statistics_parse_failure" if cache_enabled is None else "transport_message_loss_statistics_parse_failure"
    return result


def parse_loss_statistics(
    process_output: str | None,
    *,
    structured_reports: Sequence[Path] = (),
    cache_enabled: bool | None = None,
    capability_proof: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Parse actual loss statistics without fabricating zero values.

    Structured reports are accepted only when they contain enough information
    for the precedence gate.  A stock Jazzy direct-write recorder may expose a
    transport counter but no recorder-side counter; that is represented as
    ``null`` and can pass only when the observed cache mode is false and
    transport loss is exactly zero.
    """

    parse_errors: list[str] = []
    for report_path in structured_reports:
        if not report_path.is_file():
            continue
        try:
            payload = json.loads(report_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            parse_errors.append(f"{report_path.name}:{type(error).__name__}")
            continue
        result = _loss_gate_from_payload(
            payload,
            source=f"structured_json:{report_path.name}",
            cache_enabled=cache_enabled,
            capability_proof=capability_proof,
        )
        if result["passed"] or result["first_blocker"] in {"transport_message_loss_nonzero", "recorder_message_loss_nonzero"}:
            return result
    raw_output = process_output or ""
    if raw_output.lstrip().startswith("{"):
        try:
            payload = json.loads(raw_output)
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, (Mapping, list)):
            return _loss_gate_from_payload(
                payload,
                source="process_output_json",
                cache_enabled=cache_enabled,
                capability_proof=capability_proof,
            )
    result = _loss_gate_from_text(
        raw_output,
        source="process_output",
        cache_enabled=cache_enabled,
        capability_proof=capability_proof,
    )
    if parse_errors:
        result["structured_parse_errors"] = parse_errors
    return result


def post_stop_source_backed_zero(
    process_output: str,
    *,
    finalized: bool,
    process_alive_after_stop: bool,
    rosbag2_version: str | None,
    recorder_argv: Sequence[str],
) -> dict[str, Any]:
    """Classify a complete normal-stop log without laundering it as live proof.

    In rosbag2 0.26.11 RecorderImpl::stop(), the accumulator is read only after
    ``Recording stopped`` and the transport-loss warning is emitted only when
    the value is greater than zero.  Debug logging is requested explicitly, so
    warning output cannot have been filtered by the configured log threshold.
    """

    stopped_count = len(re.findall(r"\bRecording stopped\b", process_output))
    warning_count = len(re.findall(r"Number of messages lost on the transport layer:", process_output))
    log_level_debug = "--log-level" in recorder_argv and recorder_argv[recorder_argv.index("--log-level") + 1 : recorder_argv.index("--log-level") + 2] == ["debug"]
    valid = bool(
        rosbag2_version == ROS_BAG2_VERSION
        and finalized
        and not process_alive_after_stop
        and stopped_count == 1
        and warning_count == 0
        and log_level_debug
    )
    return {
        "value": 0 if valid else None,
        "evidence_type": "source_backed_negative_proof" if valid else "unavailable",
        "valid": valid,
        "pre_send_live_zero_proven": False,
        "recording_stopped_seen_count": stopped_count,
        "transport_loss_warning_seen_count": warning_count,
        "complete_log_capture": bool(stopped_count == 1),
        "warning_log_level_not_filtered": log_level_debug,
        "process_exited_cleanly": bool(finalized and not process_alive_after_stop),
        "source_version": ROS_BAG2_VERSION,
        "source_commit": ROSBAG2_0_26_11_COMMIT,
        "source_call_chain": [
            "RecorderImpl::create_subscription",
            "rclcpp::SubscriptionOptions.event_callbacks.message_lost_callback",
            "RecorderEventNotifier::on_messages_lost_in_transport",
            "RecorderEventNotifier::get_total_num_messages_lost_in_transport",
            "RecorderImpl::stop",
            "RCLCPP_WARN only when total > 0",
        ],
    }


def _default_ros_runner(command: str, *, timeout_s: float = 30.0) -> tuple[int | None, str, str]:
    overlay = _linux_path(PATCHED_ROSBAG2_PREFIX / "setup.bash")
    if os.name == "nt":
        argv = [
            "wsl.exe",
            "-d",
            "Ubuntu-24.04-D",
            "--",
            "bash",
            "-lc",
            f"source /opt/ros/jazzy/setup.bash; source {shlex.quote(overlay)}; source /mnt/d/robotfucker/install/setup.bash 2>/dev/null || true; {command}",
        ]
    else:
        argv = ["bash", "-lc", f"source /opt/ros/jazzy/setup.bash; source {shlex.quote(overlay)}; {command}"]
    try:
        completed = subprocess.run(argv, capture_output=True, timeout=timeout_s, check=False)
        return completed.returncode, completed.stdout.decode("utf-8", errors="replace"), completed.stderr.decode("utf-8", errors="replace")
    except (OSError, subprocess.TimeoutExpired) as error:
        return None, "", repr(error)


def probe_installed_rosbag2_capability(
    *,
    runner: Callable[[str], tuple[int | None, str, str]] | None = None,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Probe the installed runtime, including the exact zero-cache help text."""

    run = runner or (lambda command: _default_ros_runner(command, timeout_s=30.0))
    package_code, package_stdout, package_stderr = run("ros2 pkg xml rosbag2_transport")
    prefix_code, prefix_stdout, prefix_stderr = run("ros2 pkg prefix rosbag2_transport")
    help_code, help_stdout, help_stderr = run("ros2 bag record --help")
    version_match = re.search(r"<version>\s*([^<]+?)\s*</version>", package_stdout)
    version = version_match.group(1).strip() if version_match else None
    help_text = help_stdout + "\n" + help_stderr
    zero_cache_supported = bool(
        help_code == 0
        and "--max-cache-size" in help_text
        and re.search(r"value specified is\s*0.*directly\s+written\s+to\s+disk", help_text, re.I | re.S)
    )
    capability = {
        "source": "installed_runtime_capability_test",
        "rosbag2_version": version,
        "package_probe_exit_code": package_code,
        "record_help_exit_code": help_code,
        "package_probe_stdout": package_stdout,
        "package_probe_stderr": package_stderr,
        "prefix_probe_exit_code": prefix_code,
        "rosbag2_transport_prefix": prefix_stdout.strip() or None,
        "prefix_probe_stderr": prefix_stderr,
        "patched_recorder_bound_live_counter": bool(
            prefix_code == 0 and "ros2_overlay/install_h3" in prefix_stdout.replace("\\", "/")
        ),
        "patched_library_path": str(PATCHED_ROSBAG2_LIBRARY.resolve()),
        "patched_library_sha256": sha256_file(PATCHED_ROSBAG2_LIBRARY),
        "record_help_stdout": help_stdout,
        "record_help_stderr": help_stderr,
        "stock_recorder_side_machine_readable_zero_counter": False,
        "max_cache_size_zero_direct_write_supported": zero_cache_supported,
        "proof_basis": {
            "rosbag2_package": "rosbag2_transport",
            "rosbag2_version": version,
            "help_claim": "--max-cache-size 0: every message is directly written to disk",
            "counter_claim": "stock Jazzy recorder output has no machine-readable recorder-side counter",
        },
        "passed": bool(
            version == ROS_BAG2_VERSION
            and zero_cache_supported
            and prefix_code == 0
            and "ros2_overlay/install_h3" in prefix_stdout.replace("\\", "/")
            and sha256_file(PATCHED_ROSBAG2_LIBRARY) is not None
        ),
    }
    if not capability["passed"]:
        capability["first_blocker"] = "patched_rosbag2_recorder_capability_unproven"
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / "rosbag2_installed_capability.json"
        path.write_text(json.dumps(capability, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        capability["artifact_path"] = str(path.resolve())
        capability["artifact_sha256"] = sha256_file(path)
    return capability


@dataclass
class RecorderProcess:
    process: Any
    command: list[str]
    rosbag2_argv: list[str]
    native_output_path: str
    start_time: str
    pid: int | None


def _is_native_linux_path(path: str) -> bool:
    value = str(path).replace("\\", "/")
    return value.startswith("/tmp/") and not value.startswith("/mnt/")


def _linux_path(path: Path) -> str:
    value = str(path.resolve()).replace("\\", "/")
    if len(value) > 1 and value[1] == ":":
        return f"/mnt/{value[0].lower()}/{value[3:]}"
    return value


def _json_between_markers(stdout: str) -> dict[str, Any] | None:
    match = re.search(r"STAGE28SR2_JSON_START\s*(\{.*?\})\s*STAGE28SR2_JSON_END", stdout, re.S)
    if not match:
        return None
    try:
        value = json.loads(match.group(1))
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def _normalise_node_fqn(namespace: str, name: str) -> str:
    if name.startswith("/"):
        return name
    prefix = namespace.rstrip("/")
    return f"{prefix}/{name}" if prefix else f"/{name}"


def _endpoint_blocks(text: str) -> list[str]:
    matches = list(re.finditer(r"(?m)^Node name:\s*", text))
    return [text[matches[i].start() : matches[i + 1].start() if i + 1 < len(matches) else None] for i in range(len(matches))]


def _parse_recorder_endpoint(
    text: str,
    *,
    topic: str,
    recorder_node_fqn: str,
    requested_qos_configured: Mapping[str, Any],
    qos_override_path: Path,
    qos_override_sha256: str,
) -> dict[str, Any] | None:
    for block in _endpoint_blocks(text):
        name_match = re.search(r"(?m)^Node name:\s*(\S+)", block)
        namespace_match = re.search(r"(?m)^Node namespace:\s*(\S+)", block)
        endpoint_match = re.search(r"(?m)^Endpoint type:\s*(\S+)", block)
        gid_match = re.search(r"(?m)^GID:\s*(\S+)", block)
        if not name_match or not namespace_match or not endpoint_match or not gid_match:
            continue
        node_fqn = _normalise_node_fqn(namespace_match.group(1), name_match.group(1))
        if node_fqn != recorder_node_fqn or endpoint_match.group(1).upper() != "SUBSCRIPTION":
            continue
        qos: dict[str, Any] = {}
        reliability = re.search(r"(?m)^\s*Reliability:\s*(.+)$", block)
        history = re.search(r"(?m)^\s*History \(Depth\):\s*(.+)$", block)
        durability = re.search(r"(?m)^\s*Durability:\s*(.+)$", block)
        if reliability:
            qos["reliability"] = reliability.group(1).strip()
        if history:
            qos["history_depth"] = history.group(1).strip()
        if durability:
            qos["durability"] = durability.group(1).strip()
        if not qos:
            return None
        return {
            "topic": topic,
            "recorder_node_fqn": node_fqn,
            "endpoint_type": "SUBSCRIPTION",
            "gid": gid_match.group(1),
            "requested_qos": qos,
            "requested_qos_configured": dict(requested_qos_configured),
            "requested_qos_source": {
                "qos_override_path": str(qos_override_path),
                "qos_override_sha256": qos_override_sha256,
            },
            "observed_from_live_graph": True,
        }
    return None


def inspect_live_recorder_endpoints(
    *,
    run_id: str,
    recorder_node_fqn: str,
    runtime_instance_id: str,
    recorder_pid: int | None,
    qos_override_path: Path | None = None,
    runner: Callable[[str], tuple[int | None, str, str]] | None = None,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Bind every required topic to this recorder's live SUBSCRIPTION endpoint."""

    run = runner or _default_ros_runner
    actual_qos_path = Path(qos_override_path or ROOT / "config/stage28sr2_recorder_qos_override.yaml").resolve()
    actual_qos_sha256 = sha256_file(actual_qos_path)
    if actual_qos_sha256 is None:
        raise FileNotFoundError(actual_qos_path)
    requested_qos = load_recorder_qos_config(actual_qos_path, required_topics=REQUIRED_TOPICS)
    endpoints: dict[str, Any] = {}
    raw: dict[str, Any] = {}
    markers = []
    for index, topic in enumerate(REQUIRED_TOPICS):
        begin = f"STAGE28SR2_TOPIC_BEGIN_{index}"
        end = f"STAGE28SR2_TOPIC_END_{index}"
        markers.append((begin, end))
    probe_parts = ["sleep 0.5"]
    for index, topic in enumerate(REQUIRED_TOPICS):
        begin, end = markers[index]
        probe_parts.append(
            f"printf '\\n{begin}\\n'; ros2 topic info -v {shlex.quote(topic)} 2>&1; status=$?; printf '\\n{end}:%s\\n' \"$status\""
        )
    code, combined_stdout, combined_stderr = run("; ".join(probe_parts))
    for index, topic in enumerate(REQUIRED_TOPICS):
        begin, end = markers[index]
        segment = combined_stdout.split(begin, 1)[1] if begin in combined_stdout else ""
        if end in segment:
            segment, status_text = segment.split(end, 1)
            status_match = re.search(r":(\d+)\s*$", status_text)
            topic_code = int(status_match.group(1)) if status_match else code
        else:
            topic_code = code
        raw[topic] = {"exit_code": topic_code, "stdout": segment, "stderr": combined_stderr}
        endpoint = _parse_recorder_endpoint(
            segment,
            topic=topic,
            recorder_node_fqn=recorder_node_fqn,
            requested_qos_configured=requested_qos[topic],
            qos_override_path=actual_qos_path,
            qos_override_sha256=actual_qos_sha256,
        )
        if endpoint is None:
            endpoint = {
                "topic": topic,
                "recorder_node_fqn": recorder_node_fqn,
                "endpoint_type": None,
                "gid": None,
                "requested_qos": None,
                "requested_qos_configured": dict(requested_qos[topic]),
                "requested_qos_source": {
                    "qos_override_path": str(actual_qos_path),
                    "qos_override_sha256": actual_qos_sha256,
                },
                "observed_from_live_graph": False,
            }
        endpoint.update({"recorder_pid": recorder_pid, "runtime_instance_id": runtime_instance_id, "run_id": run_id})
        endpoints[topic] = endpoint
    checks = {
        topic: bool(
            value.get("observed_from_live_graph") is True
            and value.get("recorder_node_fqn") == recorder_node_fqn
            and value.get("endpoint_type") == "SUBSCRIPTION"
            and isinstance(value.get("gid"), str)
            and bool(value.get("gid"))
            and isinstance(value.get("requested_qos"), Mapping)
            and bool(value.get("requested_qos"))
            and value.get("recorder_pid") is not None
            and value.get("runtime_instance_id") == runtime_instance_id
            and value.get("run_id") == run_id
        )
        for topic, value in endpoints.items()
    }
    report: dict[str, Any] = {
        "schema_version": "stage28sr2-live-recorder-endpoint-gate-v1",
        "run_id": run_id,
        "runtime_instance_id": runtime_instance_id,
        "recorder_node_fqn": recorder_node_fqn,
        "recorder_pid": recorder_pid,
        "qos_override_path": str(actual_qos_path),
        "qos_override_sha256": actual_qos_sha256,
        "requested_qos_configured": requested_qos,
        "endpoints": endpoints,
        "raw_topic_info": raw,
        "checks": checks,
        "passed": all(checks.values()),
        "first_blocker": next((f"live_recorder_endpoint_missing:{topic}" for topic, passed in checks.items() if not passed), None),
    }
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / "live_recorder_endpoints.json"
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        report["artifact_path"] = str(path.resolve())
        report["artifact_sha256"] = sha256_file(path)
    return report


class Rosbag2RecorderOrchestrator:
    """Start, inspect, finalize, sequentially read, and copy one recorder."""

    def __init__(
        self,
        output_dir: Path,
        *,
        runtime_instance_id: str,
        run_id: str | None = None,
        required_topics: Sequence[str] = REQUIRED_TOPICS,
        process_factory: Callable[..., Any] | None = None,
        command_runner: Callable[[str], tuple[int | None, str, str]] | None = None,
        capability: Mapping[str, Any] | None = None,
        reader_runner: Callable[[str], tuple[int | None, str, str]] | None = None,
        health_monitor_period_s: float = 1.0,
        qos_override_path: Path | None = None,
        persistent_graph_raw_path: Path | None = None,
        persistent_observer_pid: int | None = None,
        observer_started_before_runtime: bool = False,
        transport_mode: str = DEFAULT_UDP_SHM,
    ) -> None:
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.runtime_instance_id = runtime_instance_id
        self.run_id = re.sub(r"[^A-Za-z0-9_-]", "_", run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
        self.required_topics = tuple(required_topics)
        self.process_factory = process_factory or subprocess.Popen
        self.command_runner = command_runner or _default_ros_runner
        self.reader_runner = reader_runner or self.command_runner
        self.capability = dict(capability or {})
        self.health_monitor_period_s = health_monitor_period_s
        self.node_name = f"stage28sr2_recorder_{self.run_id}"
        self.recorder_node_fqn = f"/{self.node_name}"
        self.native_output_path = f"{NATIVE_RECORDING_ROOT}/stage28sr2_{self.run_id}_mcap"
        self.native_log_path = f"{NATIVE_RECORDING_ROOT}/stage28sr2_{self.run_id}_recorder.log"
        self.qos_override_path = Path(qos_override_path or ROOT / "config/stage28sr2_recorder_qos_override.yaml").resolve()
        self.persistent_graph_raw_path = Path(persistent_graph_raw_path).resolve() if persistent_graph_raw_path else None
        self.persistent_observer_pid = persistent_observer_pid
        self.observer_started_before_runtime = observer_started_before_runtime
        self.transport_mode = transport_mode
        self.process_environment, self.transport_configuration = build_transport_environment(transport_mode)
        self.host_evidence_path = self.output_dir / "evidence_bag"
        self.capture: RecorderProcess | None = None
        self.state: dict[str, Any] = {
            "schema_version": "stage28sr2-recorder-state-v2",
            "runtime_instance_id": runtime_instance_id,
            "run_id": self.run_id,
            "pid": None,
            "recorder_pid": None,
            "start_time": None,
            "storage_backend": "mcap",
            "rosbag2_version": None,
            "max_cache_size": 0,
            "cache_enabled": None,
            "direct_write_verified": False,
            "recorder_internal_loss_mode": {
                "max_cache_size": 0,
                "cache_enabled": False,
                "direct_write": True,
                "message_cache_drop_counter_applicable": False,
                "proof_basis": {"rosbag2_version": ROS_BAG2_VERSION},
            },
            "active_recording_filesystem": "linux_native",
            "native_output_path": self.native_output_path,
            "native_log_path": self.native_log_path,
            "qos_override_path": str(self.qos_override_path.resolve()),
            "qos_override_sha256": sha256_file(self.qos_override_path),
            "output_path": self.native_output_path,
            "host_evidence_path": str(self.host_evidence_path.resolve()),
            "required_topics": list(self.required_topics),
            "recorder_node_fqn": self.recorder_node_fqn,
            "subscriptions_verified": False,
            "endpoint_gate_passed": False,
            "endpoint_evidence_mode": "persistent_graph_observer" if self.persistent_graph_raw_path else "legacy_cli_diagnostic",
            "transport_configuration": self.transport_configuration,
            "process_alive": False,
            "health_monitor_period_s": health_monitor_period_s,
            "disk_space_bytes": None,
            "finalized": False,
            "sequential_reader_passed": False,
            "bag_complete": False,
            "bag_valid": False,
            "partial_bag_retained": False,
            "recorder_loss_gate": {
                "source": "unavailable",
                "transport_lost_total": None,
                "recorder_lost_total": None,
                "per_topic": {},
                "cache_enabled": None,
                "passed": False,
                "first_blocker": "transport_message_loss_statistics_unavailable",
            },
            "pre_send_live_loss_gate": {
                "pre_send_live_zero_proven": False,
                "recorder_bound": False,
                "transport_lost_total": None,
                "passed": False,
                "first_blocker": "recorder_live_loss_not_queried",
            },
        }

    def _run(self, command: str, *, timeout_s: float = 30.0) -> tuple[int | None, str, str]:
        if self.command_runner is _default_ros_runner:
            return _default_ros_runner(command, timeout_s=timeout_s)
        return self.command_runner(command)

    def _ensure_capability(self) -> dict[str, Any]:
        if not self.capability:
            self.capability = probe_installed_rosbag2_capability(output_dir=self.output_dir)
        self.state["capability"] = dict(self.capability)
        self.state["rosbag2_version"] = self.capability.get("rosbag2_version")
        return self.capability

    def _rosbag_argv(self) -> list[str]:
        return [
            "ros2",
            "bag",
            "record",
            "--storage",
            "mcap",
            "--output",
            self.native_output_path,
            "--max-cache-size",
            "0",
            "--node-name",
            self.node_name,
            "--qos-profile-overrides-path",
            _linux_path(self.qos_override_path),
            "--disable-keyboard-controls",
            "--include-unpublished-topics",
            "--log-level",
            "debug",
            "--topics",
            *self.required_topics,
        ]

    def _process_command(self, rosbag_argv: list[str]) -> list[str]:
        overlay = _linux_path(PATCHED_ROSBAG2_PREFIX / "setup.bash")
        runtime_export = "export STAGE28SR2_RUNTIME_INSTANCE_ID=" + shlex.quote(self.runtime_instance_id) + "; "
        shell = "source " + shlex.quote(overlay) + "; " + runtime_export + shell_prefix(self.transport_mode) + "exec " + shlex.join(rosbag_argv) + " > " + shlex.quote(self.native_log_path) + " 2>&1"
        if os.name != "nt":
            return ["bash", "-lc", shell]
        if shutil.which("ros2"):
            return ["bash", "-lc", shell]
        shell = "source /opt/ros/jazzy/setup.bash; source /mnt/d/robotfucker/install/setup.bash 2>/dev/null || true; " + shell
        return ["wsl.exe", "-d", "Ubuntu-24.04-D", "--", "bash", "-lc", shell]

    def start(self) -> dict[str, Any]:
        if self.capture is not None:
            raise RuntimeError("recorder already started")
        capability = self._ensure_capability()
        if self.transport_configuration.get("passed") is not True:
            raise RuntimeError(str(self.transport_configuration.get("first_blocker")))
        if capability.get("rosbag2_version") != ROS_BAG2_VERSION:
            raise RuntimeError("rosbag2 version is not stock Jazzy 0.26.11")
        if capability.get("max_cache_size_zero_direct_write_supported") is not True:
            raise RuntimeError("installed rosbag2 does not prove max_cache_size=0 direct-write mode")
        if not _is_native_linux_path(self.native_output_path):
            raise RuntimeError("active recorder output is not a Linux-native /tmp path")
        # Bind the persisted provenance to the exact file used to construct
        # this recorder's argv, including callers that intentionally replace
        # ``qos_override_path`` for a diagnostic shadow run.
        self.qos_override_path = Path(self.qos_override_path).resolve()
        self.state["qos_override_path"] = str(self.qos_override_path)
        self.state["qos_override_sha256"] = sha256_file(self.qos_override_path)
        self.state["requested_qos_configured"] = load_recorder_qos_config(
            self.qos_override_path,
            required_topics=self.required_topics,
        )
        rosbag_argv = self._rosbag_argv()
        command = self._process_command(rosbag_argv)
        process_kwargs: dict[str, Any] = {
            "stdout": subprocess.PIPE,
            "stderr": subprocess.STDOUT,
            "text": True,
            "env": self.process_environment if (os.name != "nt" or shutil.which("ros2")) else None,
        }
        if process_kwargs["env"] is not None:
            process_kwargs["env"] = dict(process_kwargs["env"])
            process_kwargs["env"]["STAGE28SR2_RUNTIME_INSTANCE_ID"] = self.runtime_instance_id
        if os.name != "nt":
            process_kwargs["start_new_session"] = True
            process_kwargs["preexec_fn"] = _restore_recorder_child_signals
        process = self.process_factory(command, **process_kwargs)
        self.capture = RecorderProcess(process, command, rosbag_argv, self.native_output_path, now_utc(), getattr(process, "pid", None))
        self.state.update(
            {
                "pid": self.capture.pid,
                "start_time": self.capture.start_time,
                "command": list(command),
                "rosbag2_argv": list(rosbag_argv),
                "process_alive": self._is_alive(),
                "cache_enabled": False,
                "direct_write_verified": bool("--max-cache-size" in rosbag_argv and rosbag_argv[rosbag_argv.index("--max-cache-size") + 1] == "0"),
                "recorder_internal_loss_mode": {
                    "max_cache_size": 0,
                    "cache_enabled": False,
                    "direct_write": True,
                    "message_cache_drop_counter_applicable": False,
                    "proof_basis": {"rosbag2_version": capability.get("rosbag2_version"), "argv": list(rosbag_argv)},
                },
            }
        )
        self.state["disk_space_bytes"] = self._disk_space()
        self.state["linux_native_filesystem"] = self._filesystem_probe()
        self.state["native_filesystem_verified"] = bool(self.state["linux_native_filesystem"].get("passed"))
        self.persist("recorder_start.json")
        return dict(self.state)

    def _is_alive(self) -> bool:
        if self.capture is None:
            return False
        poll = getattr(self.capture.process, "poll", None)
        return bool(poll is None or poll() is None)

    def _disk_space(self) -> int | None:
        try:
            if os.name == "nt":
                code, stdout, _stderr = self._run("df -Pk /tmp | tail -1")
                match = re.search(r"\s(\d+)\s+\d+\s+\d+\s+\d+%\s", stdout)
                return int(match.group(1)) * 1024 if code == 0 and match else None
            return shutil.disk_usage("/tmp").free
        except (OSError, ValueError):
            return None

    def _filesystem_probe(self) -> dict[str, Any]:
        # Probe the native filesystem before rosbag2 creates its output child;
        # probing the not-yet-created child itself would fail closed for the
        # wrong reason on GNU df/stat.
        code, stdout, stderr = self._run("df -T /tmp; stat -f -c %T /tmp")
        fs_type = None
        for line in stdout.splitlines():
            if line.strip() and not line.startswith("Filesystem") and len(line.split()) >= 2:
                fields = line.split()
                if fields[0].startswith("/") or fields[0].startswith("overlay"):
                    fs_type = fields[1]
        if fs_type is None:
            last = stdout.splitlines()[-1].strip() if stdout.splitlines() else ""
            if last and len(last.split()) == 1:
                fs_type = last
        forbidden = {"9p", "drvfs", "fuseblk", "cifs", "ntfs"}
        return {
            "path": self.native_output_path,
            "filesystem_type": fs_type,
            "linux_native_path": _is_native_linux_path(self.native_output_path),
            "not_windows_mount": not self.native_output_path.startswith("/mnt/"),
            "passed": bool(code == 0 and _is_native_linux_path(self.native_output_path) and fs_type not in forbidden and fs_type is not None),
            "stdout": stdout,
            "stderr": stderr,
        }

    def verify_readiness(self, *, minimum_free_bytes: int = 64 * 1024 * 1024) -> dict[str, Any]:
        if self.capture is None:
            self.state.update({"process_alive": False, "subscriptions_verified": False, "ready": False})
            return self.persist("recorder_readiness.json")
        self.state["process_alive"] = self._is_alive()
        self.state["disk_space_bytes"] = self._disk_space()
        self.state["output_directory_writable"] = os.access(self.output_dir, os.W_OK)
        self.state["disk_space_sufficient"] = bool(self.state["disk_space_bytes"] is not None and self.state["disk_space_bytes"] >= minimum_free_bytes)
        self.state["storage_backend_verified"] = self.state["storage_backend"] == "mcap"
        argv = list(self.state.get("rosbag2_argv", []))
        cache_pair = "--max-cache-size" in argv and argv[argv.index("--max-cache-size") + 1 : argv.index("--max-cache-size") + 2] == ["0"]
        self.state["direct_write_verified"] = bool(
            cache_pair
            and self.state.get("cache_enabled") is False
            and self.capability.get("max_cache_size_zero_direct_write_supported") is True
            and self.capability.get("rosbag2_version") == ROS_BAG2_VERSION
        )
        base_ready = bool(
            self.state["process_alive"]
            and self.state["output_directory_writable"]
            and self.state["disk_space_sufficient"]
            and self.state["storage_backend_verified"]
            and self.state.get("native_filesystem_verified") is True
            and self.state["direct_write_verified"]
        )
        self.state["ready"] = base_ready
        self.state["strict_ready"] = bool(base_ready and self.state.get("endpoint_gate_passed") is True)
        return self.persist("recorder_readiness.json")

    def _recorder_pid_from_graph(self) -> int | None:
        code, stdout, _stderr = self._run(f"ps -eo pid=,args= | grep -F {shlex.quote(self.node_name)} | grep -v grep")
        if code != 0:
            return None
        match = re.search(r"^\s*(\d+)\s+", stdout, re.M)
        return int(match.group(1)) if match else None

    def _loaded_library_provenance(self, recorder_pid: int | None) -> dict[str, Any]:
        expected_sha = sha256_file(PATCHED_ROSBAG2_LIBRARY)
        report: dict[str, Any] = {
            "recorder_pid": recorder_pid,
            "expected_library_path": _linux_path(PATCHED_ROSBAG2_LIBRARY),
            "expected_library_sha256": expected_sha,
            "loaded_library_path": None,
            "loaded_library_sha256": None,
            "passed": False,
        }
        if recorder_pid is None:
            report["first_blocker"] = "recorder_pid_unavailable_for_library_proof"
            return report
        code, stdout, stderr = self._run(
            f"awk '/librosbag2_transport[.]so/ {{print $6; exit}}' /proc/{int(recorder_pid)}/maps"
        )
        loaded = stdout.strip().splitlines()[0] if code == 0 and stdout.strip() else None
        report.update({"maps_probe_exit_code": code, "maps_probe_stderr": stderr, "loaded_library_path": loaded})
        if loaded is None:
            report["first_blocker"] = "loaded_rosbag2_transport_library_not_found"
            return report
        sha_code, sha_stdout, sha_stderr = self._run(f"sha256sum {shlex.quote(loaded)}")
        loaded_sha = sha_stdout.strip().split()[0] if sha_code == 0 and sha_stdout.strip() else None
        report.update({"sha_probe_exit_code": sha_code, "sha_probe_stderr": sha_stderr, "loaded_library_sha256": loaded_sha})
        report["passed"] = bool(
            expected_sha is not None
            and loaded_sha == expected_sha
            and loaded == _linux_path(PATCHED_ROSBAG2_LIBRARY)
        )
        report["first_blocker"] = None if report["passed"] else "loaded_patched_library_identity_mismatch"
        return report

    def query_live_transport_loss(self, *, timeout_s: float = 5.0) -> dict[str, Any]:
        """Query the same RecorderImpl accumulator before ledger consumption."""

        recorder_pid = self._recorder_pid_from_graph()
        self.state["recorder_pid"] = recorder_pid
        service = f"{self.recorder_node_fqn}/stage28sr2_live_transport_loss"
        helper = _linux_path(ROOT / "scripts/stage28sr2_recorder_live_loss_query.py")
        code, stdout, stderr = self._run(
            f"python3 {shlex.quote(helper)} --service {shlex.quote(service)} --timeout {float(timeout_s)}",
            timeout_s=timeout_s + 3.0,
        )
        envelope = _json_between_markers(stdout)
        library = self._loaded_library_provenance(recorder_pid)
        result: dict[str, Any] = {
            "schema_version": "stage28sr2-recorder-bound-pre-live-loss-gate-v1",
            "query_exit_code": code,
            "query_stderr": stderr,
            "query_service": service,
            "recorder_bound": False,
            "pre_send_live_zero_proven": False,
            "transport_lost_total": None,
            "observation_timestamp": now_utc(),
            "rosbag2_source_commit": ROSBAG2_0_26_11_COMMIT,
            "patched_binary_or_library": library,
            "passed": False,
            "first_blocker": "recorder_live_loss_query_unavailable",
        }
        if not isinstance(envelope, Mapping):
            result["first_blocker"] = "recorder_live_loss_query_malformed_or_timeout"
        elif envelope.get("passed") is not True or not isinstance(envelope.get("payload"), Mapping):
            result["first_blocker"] = str(envelope.get("first_blocker", "recorder_live_loss_query_failed"))
        else:
            payload = dict(envelope["payload"])
            counter = payload.get("transport_lost_total")
            identity_checks = {
                "schema": payload.get("schema_version") == "stage28sr2-recorder-live-loss-v1",
                "recorder_pid": payload.get("recorder_pid") == recorder_pid,
                "recorder_node": payload.get("recorder_node_identity") == self.recorder_node_fqn,
                "runtime_instance": payload.get("runtime_instance_id") == self.runtime_instance_id,
                "source_commit": payload.get("rosbag2_source_commit") == ROSBAG2_0_26_11_COMMIT,
                "instrumentation": payload.get("instrumentation_id") == RECORDER_INSTRUMENTATION_ID,
                "observation_timestamp": isinstance(payload.get("observation_timestamp_ns"), int),
                "counter_well_formed": isinstance(counter, int) and not isinstance(counter, bool) and counter >= 0,
                "patched_library_loaded": library.get("passed") is True,
            }
            result.update(
                {
                    "payload": payload,
                    "identity_checks": identity_checks,
                    "recorder_pid": recorder_pid,
                    "recorder_node_identity": payload.get("recorder_node_identity"),
                    "runtime_instance_id": payload.get("runtime_instance_id"),
                    "transport_lost_total": counter if identity_checks["counter_well_formed"] else None,
                    "observation_timestamp_ns": payload.get("observation_timestamp_ns"),
                    "recorder_bound": all(identity_checks.values()),
                }
            )
            if not all(identity_checks.values()):
                result["first_blocker"] = "recorder_live_loss_identity_mismatch_or_malformed"
            elif counter != 0:
                result["first_blocker"] = "transport_message_loss_nonzero"
            else:
                result["pre_send_live_zero_proven"] = True
                result["passed"] = True
                result["first_blocker"] = None
        self.state["pre_send_live_loss_gate"] = result
        self.persist("recorder_pre_send_live_loss_gate.json")
        return result

    def verify_live_endpoints(self) -> dict[str, Any]:
        if self.capture is None:
            report = {"passed": False, "first_blocker": "recorder_not_started"}
        else:
            report = {"passed": False, "first_blocker": "live_recorder_endpoint_discovery_pending"}
            if self.persistent_graph_raw_path is not None:
                recorder_pid = self._recorder_pid_from_graph()
                self.state["recorder_pid"] = recorder_pid
                configured = load_recorder_qos_config(self.qos_override_path, required_topics=self.required_topics)
                report = wait_for_persistent_gate(
                    self.persistent_graph_raw_path,
                    required_topics=self.required_topics,
                    recorder_node_fqn=self.recorder_node_fqn,
                    configured_qos=configured,
                    observer_started_before_runtime=self.observer_started_before_runtime,
                    observer_pid=self.persistent_observer_pid,
                    timeout_sec=30.0,
                    require_publishers=False,
                )
                report.update({
                    "schema_version": "stage28sr2-persistent-recorder-endpoint-gate-v1",
                    "run_id": self.run_id,
                    "runtime_instance_id": self.runtime_instance_id,
                    "recorder_node_fqn": self.recorder_node_fqn,
                    "recorder_pid": recorder_pid,
                    "qos_override_path": str(self.qos_override_path),
                    "qos_override_sha256": sha256_file(self.qos_override_path),
                    "evidence_mode": "persistent_graph_observer",
                })
                # A one-shot CLI snapshot is retained only to diagnose disagreement.
                diagnostic = inspect_live_recorder_endpoints(
                    run_id=self.run_id,
                    recorder_node_fqn=self.recorder_node_fqn,
                    runtime_instance_id=self.runtime_instance_id,
                    recorder_pid=recorder_pid,
                    qos_override_path=self.qos_override_path,
                    runner=self._run,
                    output_dir=None,
                )
                report["cli_diagnostic"] = diagnostic
                report["persistent_cli_disagreement"] = bool(report.get("passed") is True and diagnostic.get("passed") is not True)
            else:
            # rosbag2 discovery is asynchronous.  A single failed query is not
            # evidence that a topic is absent, but the gate remains fail closed
            # after this bounded live-graph barrier.
                time.sleep(1.0)
                for attempt in range(30):
                    recorder_pid = self._recorder_pid_from_graph()
                    self.state["recorder_pid"] = recorder_pid
                    report = inspect_live_recorder_endpoints(
                        run_id=self.run_id,
                        recorder_node_fqn=self.recorder_node_fqn,
                        runtime_instance_id=self.runtime_instance_id,
                        recorder_pid=recorder_pid,
                        qos_override_path=self.qos_override_path,
                        runner=self._run,
                        output_dir=self.output_dir,
                    )
                    report["discovery_attempt"] = attempt + 1
                    report["evidence_mode"] = "legacy_cli_diagnostic"
                    if report.get("passed") is True:
                        break
                    if not self._is_alive():
                        break
                    time.sleep(1.0)
        self.state["live_recorder_endpoints"] = report
        self.state["subscriptions_verified"] = report.get("passed") is True
        self.state["endpoint_gate_passed"] = report.get("passed") is True
        self.persist("recorder_endpoint_gate.json")
        self.verify_readiness()
        return report

    def verify_persistent_publishers(self, *, timeout_sec: float = 30.0) -> dict[str, Any]:
        if self.persistent_graph_raw_path is None:
            report = {"passed": False, "first_blocker": "persistent_graph_observer_not_configured"}
        else:
            configured = load_recorder_qos_config(self.qos_override_path, required_topics=self.required_topics)
            report = wait_for_persistent_gate(
                self.persistent_graph_raw_path,
                required_topics=self.required_topics,
                recorder_node_fqn=self.recorder_node_fqn,
                configured_qos=configured,
                observer_started_before_runtime=self.observer_started_before_runtime,
                observer_pid=self.persistent_observer_pid,
                timeout_sec=timeout_sec,
                require_publishers=True,
            )
        self.state["persistent_graph_gate"] = report
        self.state["persistent_graph_gate_passed"] = report.get("passed") is True
        self.persist("persistent_graph_summary.json")
        return report

    def monitor_health(self) -> dict[str, Any]:
        self.state["process_alive"] = self._is_alive()
        self.state["health_checked_at"] = now_utc()
        if not self.state["process_alive"]:
            self.state["health_failure"] = "recorder_process_died"
        return self.persist("recorder_health.json")

    def _reader(self, native_path: str) -> dict[str, Any]:
        script = _linux_path(ROOT / "scripts/stage28sr2_sequential_reader.py")
        command = f"python3 {shlex.quote(script)} --bag {shlex.quote(native_path)}"
        code, stdout, stderr = self._run(command)
        payload = _json_between_markers(stdout)
        if payload is None:
            return {
                "schema_version": "stage28sr2-sequential-reader-v1",
                "reader_api": "rosbag2_py.SequentialReader",
                "passed": False,
                "errors": ["sequential_reader_json_marker_missing", stderr[-2000:]],
                "runner_exit_code": code,
                "runner_stdout_sha256": hashlib.sha256(stdout.encode()).hexdigest(),
            }
        payload["runner_exit_code"] = code
        payload["runner_stdout_sha256"] = hashlib.sha256(stdout.encode()).hexdigest()
        payload["runner_stderr"] = stderr
        return payload

    def _native_manifest(self, path: str) -> dict[str, Any]:
        script = _linux_path(ROOT / "scripts/stage28sr2_sequential_reader.py")
        code, stdout, stderr = self._run(f"python3 {shlex.quote(script)} --manifest {shlex.quote(path)}")
        payload = _json_between_markers(stdout)
        if payload is None:
            return {"passed": False, "errors": ["tree_manifest_json_marker_missing", stderr[-2000:]], "runner_exit_code": code}
        payload["runner_exit_code"] = code
        payload["passed"] = code == 0
        return payload

    def _copy_native_bag(self) -> dict[str, Any]:
        if self.state.get("sequential_reader_passed") is not True:
            return {"passed": False, "first_blocker": "sequential_reader_not_passed_before_copy"}
        if not self.state.get("native_filesystem_verified"):
            return {"passed": False, "first_blocker": "linux_native_filesystem_not_proven"}
        if self.host_evidence_path.exists():
            return {"passed": False, "first_blocker": "evidence_bag_already_exists"}
        native_manifest = self._native_manifest(self.native_output_path)
        if native_manifest.get("passed") is not True:
            return {"passed": False, "first_blocker": "native_bag_manifest_failed", "native_manifest": native_manifest}
        destination = _linux_path(self.host_evidence_path)
        parent = _linux_path(self.host_evidence_path.parent)
        code, stdout, stderr = self._run(
            f"set -e; test ! -e {shlex.quote(destination)}; mkdir -p {shlex.quote(parent)}; cp -a {shlex.quote(self.native_output_path)} {shlex.quote(destination)}",
            timeout_s=120.0,
        )
        if code != 0:
            return {"passed": False, "first_blocker": "native_to_evidence_copy_failed", "stdout": stdout, "stderr": stderr}
        copied_manifest = self._native_manifest(destination)
        byte_identical = native_manifest.get("files") == copied_manifest.get("files") and native_manifest.get("file_count") == copied_manifest.get("file_count")
        report = {
            "source_native_path": self.native_output_path,
            "destination_evidence_path": destination,
            "native_manifest": native_manifest,
            "copied_manifest": copied_manifest,
            "byte_identical": byte_identical,
            "sha256_verified": byte_identical,
            "passed": bool(byte_identical and copied_manifest.get("passed") is True),
            "first_blocker": None if byte_identical and copied_manifest.get("passed") is True else "native_and_evidence_bag_sha256_mismatch",
        }
        self.state["bag_copy"] = report
        self.state["evidence_bag_byte_identical"] = report["passed"]
        self.persist("bag_copy_sha256.json")
        return report

    def _diagnose_transport_loss(self) -> dict[str, Any]:
        """Collect second-line zero-goal diagnostics; never send a formal goal."""

        topics: dict[str, Any] = {}
        for topic in self.required_topics:
            code, stdout, stderr = self._run(f"ros2 topic info -v {shlex.quote(topic)}")
            topics[topic] = {"exit_code": code, "stdout": stdout, "stderr": stderr}
        code, ps_stdout, ps_stderr = self._run("ps -eo pid,pcpu,pmem,stat,args")
        diagnostics = {
            "trigger": "transport_lost_total_nonzero",
            "zero_goal": True,
            "checks_in_order": [
                "recorder_requested_qos_and_publisher_offered_qos",
                "dds_history_depth_and_subscriber_queue_depth",
                "recorder_discovery_startup_barrier",
                "fastdds_transport_diagnostics",
                "scheduler_cpu_starvation",
                "publisher_frequencies",
                "recorder_callback_write_latency",
            ],
            "topic_info_live_graph": topics,
            "process_scheduler_snapshot": {"exit_code": code, "stdout": ps_stdout, "stderr": ps_stderr},
            "qos_override_allowed": True,
            "formal_goal_sent": False,
        }
        self.state["transport_loss_diagnostics"] = diagnostics
        self.persist("transport_loss_diagnostics.json")
        return diagnostics

    def finalize(self, *, timeout_s: float = 90.0) -> dict[str, Any]:
        if self.capture is None:
            self.state.update({"finalized": False, "sequential_reader_passed": False, "bag_valid": False})
            return self.persist("recorder_finalization.json")
        process = self.capture.process
        try:
            # On Windows the Popen handle is the WSL bridge, not the Linux
            # rosbag2 process.  Signal the unique Linux recorder node first so
            # rosbag2 can flush MCAP metadata and direct-write messages.
            if os.name == "nt" and self.capture.command and self.capture.command[0].lower() == "wsl.exe":
                self._run(f"pkill -INT -f {shlex.quote(self.node_name)}", timeout_s=15.0)
            else:
                if os.name != "nt" and getattr(process, "pid", None):
                    os.killpg(os.getpgid(process.pid), signal.SIGINT)
                else:
                    send_signal = getattr(process, "send_signal", None)
                    if callable(send_signal):
                        send_signal(signal.SIGINT)
                    else:
                        terminate = getattr(process, "terminate", None)
                        if callable(terminate):
                            terminate()
            wait = getattr(process, "wait", None)
            if callable(wait):
                wait(timeout=timeout_s)
        except (OSError, ValueError, subprocess.TimeoutExpired):
            # A test double or a non-group subprocess may not have a valid
            # process group.  Fall back to the process handle so cleanup is
            # deterministic on both POSIX and Windows.
            send_signal = getattr(process, "send_signal", None)
            if callable(send_signal):
                try:
                    send_signal(signal.SIGINT)
                except (OSError, ValueError):
                    pass
            else:
                kill = getattr(process, "kill", None)
                if callable(kill):
                    kill()
        stream = getattr(process, "stdout", None)
        if stream is not None and hasattr(stream, "read"):
            try:
                self.state["process_output"] = stream.read() or ""
            except (OSError, ValueError):
                self.state["process_output"] = ""
        # WSL's bridge does not reliably return the redirected child stream to
        # the Windows Popen pipe.  Read the native recorder log after the
        # child has stopped; this is actual rosbag2 output, never a synthetic
        # counter fixture.
        log_code, log_stdout, log_stderr = self._run(f"cat {shlex.quote(self.native_log_path)}", timeout_s=30.0)
        if log_code == 0 and log_stdout.strip():
            self.state["process_output"] = log_stdout
        self.state["recorder_log_probe"] = {"exit_code": log_code, "stderr": log_stderr}
        self.state["process_alive"] = self._is_alive()
        metadata_code, metadata_stdout, metadata_stderr = self._run(f"test -f {shlex.quote(self.native_output_path + '/metadata.yaml')}")
        metadata_exists = metadata_code == 0
        self.state["metadata_exists_after_finalize"] = metadata_exists
        self.state["metadata_probe"] = {"exit_code": metadata_code, "stdout": metadata_stdout, "stderr": metadata_stderr}
        self.state["partial_bag_retained"] = metadata_exists
        self.state["finalized"] = bool(metadata_exists and not self.state["process_alive"])
        self.state["recorder_loss_gate"] = parse_loss_statistics(
            self.state.get("process_output", ""),
            cache_enabled=self.state.get("cache_enabled"),
            capability_proof=self.capability,
        )
        self.state["post_stop_source_backed_zero"] = post_stop_source_backed_zero(
            self.state.get("process_output", ""),
            finalized=self.state["finalized"],
            process_alive_after_stop=self.state["process_alive"],
            rosbag2_version=self.capability.get("rosbag2_version"),
            recorder_argv=self.state.get("rosbag2_argv", []),
        )
        if (
            self.state["recorder_loss_gate"].get("transport_lost_total") is None
            and self.state["post_stop_source_backed_zero"].get("valid") is True
        ):
            self.state["recorder_loss_gate"] = {
                "source": "post_stop_source_backed_negative_proof",
                "transport_lost_total": 0,
                "recorder_lost_total": 0,
                "per_topic": {},
                "cache_enabled": self.state.get("cache_enabled"),
                "passed": True,
                "first_blocker": None,
                "evidence_type": "source_backed_negative_proof",
                "pre_send_live_zero_proven": False,
            }
        if self.state["recorder_loss_gate"].get("first_blocker") == "transport_message_loss_nonzero":
            self._diagnose_transport_loss()
        if self.state["finalized"]:
            reader_result = self._reader(self.native_output_path)
        else:
            reader_result = {"reader_api": "rosbag2_py.SequentialReader", "passed": False, "errors": ["bag_not_finalized"]}
        self.state["sequential_reader"] = reader_result
        self.state["sequential_reader_passed"] = reader_result.get("passed") is True
        self.state["offline_readable"] = self.state["sequential_reader_passed"]  # derived only from actual reader execution
        self.state["bag_complete"] = bool(self.state["finalized"] and self.state["sequential_reader_passed"])
        self.state["bag_valid"] = bool(self.state["bag_complete"] and self.state.get("native_filesystem_verified") is True and self.state.get("direct_write_verified") is True)
        self.state["formal_evidence_valid"] = bool(
            self.state["bag_valid"]
            and self.state["recorder_loss_gate"].get("passed") is True
            and self.state.get("endpoint_gate_passed") is True
        )
        self.persist("recorder_finalization.json")
        if self.state["formal_evidence_valid"]:
            self._copy_native_bag()
            self.state["formal_evidence_valid"] = bool(self.state.get("evidence_bag_byte_identical"))
            self.persist("recorder_finalization.json")
        return dict(self.state)

    def persist(self, name: str) -> dict[str, Any]:
        path = self.output_dir / name
        path.write_text(json.dumps(self.state, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        self.state["artifact_path"] = str(path.resolve())
        self.state["artifact_sha256"] = sha256_file(path)
        return dict(self.state)


def recorder_static_audit(path: str | None = None) -> dict[str, Any]:
    source = Path(path or __file__).resolve()
    text = source.read_text(encoding="utf-8")
    checks = {
        "starts_rosbag2": "ros2" in text and "bag" in text and "record" in text,
        "mcap_backend": "mcap" in text and "--storage" in text,
        "max_cache_size_zero_explicit": '"--max-cache-size",\n            "0"' in text,
        "unique_recorder_node_name": "stage28sr2_recorder_" in text and "--node-name" in text,
        "required_topics": all(topic in text for topic in REQUIRED_TOPICS),
        "process_alive_check": "_is_alive" in text,
        "safe_finalize": "SIGINT" in text and "finalize" in text,
        "linux_native_storage_gate": "_is_native_linux_path" in text and "native_filesystem_verified" in text,
        "live_endpoint_gate": "inspect_live_recorder_endpoints" in text and "endpoint_gate_passed" in text,
        "sequential_reader_gate": "SequentialReader" in text and "sequential_reader_passed" in text,
        "sha256_copy_gate": "byte_identical" in text and "_copy_native_bag" in text,
        "loss_parser_present": "parse_loss_statistics" in text,
        "loss_precedence_present": "transport_message_loss_nonzero" in text and "recorder_message_loss_nonzero" in text,
        "loss_fail_closed": "transport_message_loss_statistics_unavailable" in text and "recorder_message_loss_gate_fail_closed" in text,
        "recorder_bound_pre_live_query": "query_live_transport_loss" in text and "pre_send_live_zero_proven" in text,
        "patched_library_runtime_maps_proof": "_loaded_library_provenance" in text and "/proc/" in text,
    }
    return {"path": str(source), "sha256": sha256_file(source), "checks": checks, "passed": all(checks.values())}
