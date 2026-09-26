"""Fail-closed final just-in-time authorization evidence for Formal R2.

The ordinary PRE recorder-loss observation is intentionally not sufficient to
authorize a send.  This module contains the pure, testable part of the final
boundary: the last live query must be fresh, identity-stable, loss-free, and
must agree with the frozen execution manifest before the one-shot ledger can
be consumed.
"""

from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


FINAL_JIT_PRE_SEND_GATE = "FINAL_JIT_PRE_SEND_GATE"
FINAL_JIT_FRESHNESS_THRESHOLD_S = 2.0


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _normalise_runtime_path(value: object) -> str:
    text = str(value or "").replace("\\", "/")
    if len(text) > 1 and text[1] == ":":
        return f"/mnt/{text[0].lower()}/{text[3:]}"
    return text


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _library_sha(query: Mapping[str, Any]) -> str | None:
    library = _mapping(query.get("patched_binary_or_library"))
    return library.get("loaded_library_sha256") or library.get("expected_library_sha256")


def _library_path(query: Mapping[str, Any]) -> str | None:
    library = _mapping(query.get("patched_binary_or_library"))
    return library.get("loaded_library_path") or library.get("expected_library_path")


def _identity_snapshot(query: Mapping[str, Any]) -> dict[str, Any]:
    checks = _mapping(query.get("identity_checks"))
    return {
        "recorder_pid": query.get("recorder_pid"),
        "recorder_node_fqn": query.get("recorder_node_identity"),
        "runtime_instance_id": query.get("runtime_instance_id"),
        "patched_library_sha256": _library_sha(query),
        "patched_library_path": _library_path(query),
        "rosbag2_source_commit": query.get("rosbag2_source_commit"),
        "counter_well_formed": checks.get("counter_well_formed") is True
        or (
            isinstance(query.get("transport_lost_total"), int)
            and not isinstance(query.get("transport_lost_total"), bool)
            and query.get("transport_lost_total") >= 0
        ),
    }


def _identity_equal(initial: Mapping[str, Any], final: Mapping[str, Any]) -> dict[str, bool]:
    initial_id = _identity_snapshot(initial)
    final_id = _identity_snapshot(final)
    return {
        "same_recorder_pid": initial_id["recorder_pid"] is not None and initial_id["recorder_pid"] == final_id["recorder_pid"],
        "same_recorder_node_fqn": bool(initial_id["recorder_node_fqn"]) and initial_id["recorder_node_fqn"] == final_id["recorder_node_fqn"],
        "same_runtime_instance_id": bool(initial_id["runtime_instance_id"]) and initial_id["runtime_instance_id"] == final_id["runtime_instance_id"],
        "same_patched_library_sha256": bool(initial_id["patched_library_sha256"]) and initial_id["patched_library_sha256"] == final_id["patched_library_sha256"],
        "same_patched_library_path": bool(initial_id["patched_library_path"]) and initial_id["patched_library_path"] == final_id["patched_library_path"],
        "same_rosbag2_source_commit": bool(initial_id["rosbag2_source_commit"]) and initial_id["rosbag2_source_commit"] == final_id["rosbag2_source_commit"],
        "final_counter_well_formed": final_id["counter_well_formed"],
    }


def build_final_jit_pre_send_gate(
    *,
    initial_pre_gate: Mapping[str, Any],
    final_query: Mapping[str, Any],
    recorder_alive: bool,
    controller_active: bool,
    action_server_available: bool,
    command_source_report: Mapping[str, Any],
    manifest_gate: Mapping[str, Any],
    query_monotonic_ns: int | None = None,
    queried_at_utc: str | None = None,
    freshness_threshold_s: float = FINAL_JIT_FRESHNESS_THRESHOLD_S,
) -> dict[str, Any]:
    """Create the one proof object accepted by the production transport."""

    query_clock = int(query_monotonic_ns if query_monotonic_ns is not None else time.monotonic_ns())
    identity = _identity_equal(initial_pre_gate, final_query)
    command_checks = {
        "report_passed": command_source_report.get("passed") is True,
        "unauthorized_action_clients_empty": not command_source_report.get("unauthorized_action_clients"),
        "unauthorized_publishers_empty": not command_source_report.get("unauthorized_publishers"),
        "moveit_execution_nodes_empty": not command_source_report.get("moveit_execution_nodes"),
        "old_runtime_clients_empty": not command_source_report.get("old_runtime_clients"),
        "conflicting_processes_empty": not command_source_report.get("conflicting_project_processes"),
    }
    final_counter_checks = {
        "query_passed": final_query.get("passed") is True,
        "recorder_bound": final_query.get("recorder_bound") is True,
        "pre_send_live_zero_proven": final_query.get("pre_send_live_zero_proven") is True,
        "transport_lost_total_zero": final_query.get("transport_lost_total") == 0,
        "counter_well_formed": identity["final_counter_well_formed"],
    }
    checks = {
        **identity,
        **final_counter_checks,
        "recorder_still_alive": bool(recorder_alive),
        "controller_still_active": bool(controller_active),
        "action_server_still_available": bool(action_server_available),
        "authorized_command_source_set_remains_exclusive": all(command_checks.values()),
        "manifest_exact_sha_revalidated": manifest_gate.get("passed") is True,
    }
    return {
        "schema_version": "stage28sr2-final-jit-pre-send-gate-v1",
        "gate_name": FINAL_JIT_PRE_SEND_GATE,
        "proof_kind": "final_jit_query",
        "queried_at_utc": queried_at_utc or now_utc(),
        "query_monotonic_ns": query_clock,
        "freshness_threshold_s": float(freshness_threshold_s),
        "freshness_passed": True,
        "initial_pre_counter_observation": {
            "observed_at": initial_pre_gate.get("observation_timestamp"),
            "transport_lost_total": initial_pre_gate.get("transport_lost_total"),
            "recorder_pid": initial_pre_gate.get("recorder_pid"),
            "recorder_node_fqn": initial_pre_gate.get("recorder_node_identity"),
            "runtime_instance_id": initial_pre_gate.get("runtime_instance_id"),
        },
        "final_jit_counter_observation": {
            "observed_at": final_query.get("observation_timestamp"),
            "observation_timestamp_ns": final_query.get("observation_timestamp_ns"),
            "transport_lost_total": final_query.get("transport_lost_total"),
            "recorder_pid": final_query.get("recorder_pid"),
            "recorder_node_fqn": final_query.get("recorder_node_identity"),
            "runtime_instance_id": final_query.get("runtime_instance_id"),
            "patched_library_sha256": _library_sha(final_query),
            "rosbag2_source_commit": final_query.get("rosbag2_source_commit"),
        },
        "initial_final_identity_equality": identity,
        "command_source_checks": command_checks,
        "manifest_gate": dict(manifest_gate),
        "checks": checks,
        "recorder_alive": bool(recorder_alive),
        "controller_active": bool(controller_active),
        "action_server_available": bool(action_server_available),
        "authorized_command_source_set": dict(command_source_report),
        "passed": all(checks.values()),
        "first_blocker": next((name for name, passed in checks.items() if not passed), None),
    }


def verify_final_jit_pre_send_gate(
    gate: Mapping[str, Any],
    *,
    now_monotonic_ns: int | None = None,
) -> dict[str, Any]:
    """Recheck the proof at the transport boundary, including freshness."""

    query_ns = gate.get("query_monotonic_ns")
    threshold = gate.get("freshness_threshold_s")
    age_s: float | None = None
    freshness = False
    if isinstance(query_ns, int) and not isinstance(query_ns, bool) and isinstance(threshold, (int, float)) and threshold >= 0:
        current_ns = int(now_monotonic_ns if now_monotonic_ns is not None else time.monotonic_ns())
        age_s = (current_ns - query_ns) / 1_000_000_000
        freshness = 0.0 <= age_s <= float(threshold)
    checks = dict(_mapping(gate.get("checks")))
    checks["gate_name_exact"] = gate.get("gate_name") == FINAL_JIT_PRE_SEND_GATE
    checks["proof_kind_exact"] = gate.get("proof_kind") == "final_jit_query"
    checks["query_timestamp_present"] = isinstance(gate.get("queried_at_utc"), str) and bool(gate.get("queried_at_utc"))
    checks["query_monotonic_timestamp_present"] = isinstance(query_ns, int) and not isinstance(query_ns, bool)
    checks["freshness_within_threshold"] = freshness and gate.get("freshness_passed") is True
    checks["proof_passed"] = gate.get("passed") is True
    return {
        "schema_version": "stage28sr2-final-jit-pre-send-verification-v1",
        "age_s": age_s,
        "freshness_threshold_s": threshold,
        "checks": checks,
        "passed": all(checks.values()),
        "first_blocker": next((name for name, passed in checks.items() if not passed), None),
    }


def revalidate_frozen_execution_manifest(
    manifest: Mapping[str, Any],
    *,
    loaded_library_sha256: str | None = None,
    loaded_library_path: str | None = None,
) -> dict[str, Any]:
    """Verify every frozen source hash immediately before ledger consumption."""

    components = manifest.get("components")
    component_checks: dict[str, Any] = {}
    if not isinstance(components, Mapping):
        return {"schema_version": "stage28sr2-frozen-manifest-revalidation-v1", "passed": False, "first_blocker": "manifest_components_missing", "components": {}}
    for name, raw in components.items():
        item = _mapping(raw)
        path = Path(str(item.get("path", ""))).resolve() if item.get("path") else Path(".").resolve() / "__missing__"
        expected = item.get("sha256")
        actual = _sha256(path)
        tracked_state = item.get("git_tracked")
        component_checks[str(name)] = {
            "path": str(path),
            "expected_sha256": expected,
            "actual_sha256": actual,
            "exists": path.is_file(),
            "sha_matches": bool(expected and actual and actual == expected),
            "explicit_sha_bound": isinstance(tracked_state, bool),
            "git_tracked_frozen": tracked_state,
            "passed": bool(expected and actual and actual == expected and path.is_file() and isinstance(tracked_state, bool)),
        }
    library = next((value for name, value in component_checks.items() if name == "patched_rosbag2_library"), None)
    loaded_checks = {
        "loaded_library_sha256_present": loaded_library_sha256 is not None,
        "loaded_library_path_present": loaded_library_path is not None,
        "loaded_library_sha256_matches_frozen": bool(library and loaded_library_sha256 and loaded_library_sha256 == library.get("expected_sha256")),
        "loaded_library_path_matches_frozen": bool(library and loaded_library_path and _normalise_runtime_path(loaded_library_path) == _normalise_runtime_path(library.get("path"))),
    }
    checks = {
        "manifest_schema_present": str(manifest.get("schema_version", "")).startswith("stage28sr2-frozen-formal-execution-manifest-"),
        "all_component_hashes_available": manifest.get("all_component_hashes_available") is True,
        "all_components_sha_revalidated": bool(component_checks) and all(item["passed"] for item in component_checks.values()),
        "untracked_components_sha_bound": all(item["explicit_sha_bound"] for item in component_checks.values()),
        **loaded_checks,
    }
    return {
        "schema_version": "stage28sr2-frozen-manifest-revalidation-v1",
        "checks": checks,
        "components": component_checks,
        "loaded_library": {"sha256": loaded_library_sha256, "path": loaded_library_path},
        "passed": all(checks.values()),
        "first_blocker": next((name for name, passed in checks.items() if not passed), None),
    }


def production_dispatch_static_audit(paths: Mapping[str, Path]) -> dict[str, Any]:
    """Audit the production files for one real FJT dispatch chain."""

    texts = {name: path.read_text(encoding="utf-8", errors="replace") if path.is_file() else "" for name, path in paths.items()}
    transport = texts.get("transport", "")
    executor = texts.get("production_runner", "")
    checks = {
        "action_client_only_in_transport": "ActionClient(" in transport and all("ActionClient(" not in text for name, text in texts.items() if name != "transport"),
        "exact_fjt_action_name_in_transport": "/fairino5_controller/follow_joint_trajectory" in transport,
        "sole_underlying_client_send": transport.count("self.client.send_goal_async(") == 1,
        "sole_executor_transport_send": executor.count("self.transport.send_goal_async(") == 1,
        "goal_response_future_chain": "wait_for_goal_response" in transport and "goal_response_future_completed" in executor,
        "accepted_goal_handle_chain": "goal_handle" in transport and "accepted" in transport,
        "result_future_chain": "get_result_async" in transport and "get_result_async_future_completed" in executor,
        "post_roll_and_finalize_chain": "recorder_post_roll_completed" in executor and "recorder_stopped_finalized_and_read_to_eof" in executor,
        "no_cli_send_goal_bypass": "ros2 action send_goal" not in "\n".join(texts.values()),
    }
    return {"checks": checks, "passed": all(checks.values()), "first_blocker": next((name for name, passed in checks.items() if not passed), None)}


def json_copy(value: Any) -> Any:
    """Return a JSON-shaped copy for certificates without shared mutability."""

    return json.loads(json.dumps(value, ensure_ascii=False))
