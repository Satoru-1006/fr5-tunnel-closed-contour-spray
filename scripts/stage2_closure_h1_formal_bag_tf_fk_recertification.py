"""Stage 2 Closure H1: immutable Formal R2 offline TF/FK recertification.

The input is a finalized rosbag2 MCAP from Formal R2.  This program is
read-only with respect to the input bag and the Formal R2 package.  It uses
ROS header ``(sec, nanosec)`` values as the only physical correspondence key;
bag arrival time and capture-monotonic time are diagnostics only.

The script is launched with MoveIt parameters by
``stage2_closure_h1_formal_bag_tf_fk_recertification_launch.py``.  It writes
only into the caller-provided, versioned output directory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


JOINTS = [f"j{i}" for i in range(1, 7)]
REQUIRED_TOPICS = (
    "/joint_states",
    "/fairino5_controller/controller_state",
    "/tf",
    "/tf_static",
)
REQUIRED_QUERY_PAIRS = (
    ("wrist2_link", "wrist3_link"),
    ("base_link", "wrist3_link"),
    ("base_link", "spray_tcp_link"),
)
ROTATION_GATE_DEG = 0.1
TRANSLATION_GATE_MM = 1.0e-3


def stamp_ns(stamp: Any) -> int:
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def stamp_dict(stamp: Any) -> dict[str, int]:
    return {"sec": int(stamp.sec), "nanosec": int(stamp.nanosec)}


def stamp_key(stamp: Any) -> tuple[int, int]:
    return int(stamp.sec), int(stamp.nanosec)


def json_safe(value: Any) -> Any:
    if isinstance(value, (np.floating, float)):
        value = float(value)
        return value if math.isfinite(value) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(json_safe(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def tree_hashes(root: Path) -> dict[str, dict[str, Any]]:
    root = root.resolve()
    result: dict[str, dict[str, Any]] = {}
    for path in sorted((p for p in root.rglob("*") if p.is_file()), key=lambda p: p.relative_to(root).as_posix()):
        result[path.relative_to(root).as_posix()] = {
            "sha256": sha256(path),
            "size": path.stat().st_size,
        }
    return result


def matrix_from_quaternion_translation(q: Any, t: Any) -> np.ndarray:
    x, y, z, w = float(q.x), float(q.y), float(q.z), float(q.w)
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w), float(t.x)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w), float(t.y)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y), float(t.z)],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=float,
    )


def matrix_from_transform(transform: Any) -> np.ndarray:
    return matrix_from_quaternion_translation(transform.transform.rotation, transform.transform.translation)


def matrix_from_moveit(value: Any) -> np.ndarray:
    return np.asarray(value.matrix() if hasattr(value, "matrix") else value, dtype=float)


def transform_error(lhs: np.ndarray, rhs: np.ndarray) -> tuple[float, float]:
    translation_mm = float(np.linalg.norm(lhs[:3, 3] - rhs[:3, 3]) * 1000.0)
    delta = lhs[:3, :3] @ rhs[:3, :3].T
    cosine = float(np.clip((np.trace(delta) - 1.0) * 0.5, -1.0, 1.0))
    return translation_mm, math.degrees(math.acos(cosine))


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=float), fraction * 100.0))


def metric_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    translations = [float(row["translation_error_mm"]) for row in records]
    rotations = [float(row["rotation_error_deg"]) for row in records]
    return {
        "samples": len(records),
        "max_translation_error_mm": max(translations) if translations else None,
        "median_translation_error_mm": percentile(translations, 0.50),
        "p95_translation_error_mm": percentile(translations, 0.95),
        "p99_translation_error_mm": percentile(translations, 0.99),
        "max_rotation_error_deg": max(rotations) if rotations else None,
        "median_rotation_error_deg": percentile(rotations, 0.50),
        "p95_rotation_error_deg": percentile(rotations, 0.95),
        "p99_rotation_error_deg": percentile(rotations, 0.99),
        "translation_gate_mm": TRANSLATION_GATE_MM,
        "rotation_gate_deg": ROTATION_GATE_DEG,
        "passed": bool(
            records
            and max(translations) <= TRANSLATION_GATE_MM
            and max(rotations) <= ROTATION_GATE_DEG
        ),
    }


def to_transform_stamped(row: dict[str, Any]):
    from geometry_msgs.msg import TransformStamped

    msg = TransformStamped()
    msg.header.stamp.sec = int(row["stamp"]["sec"])
    msg.header.stamp.nanosec = int(row["stamp"]["nanosec"])
    msg.header.frame_id = str(row["parent_frame"])
    msg.child_frame_id = str(row["child_frame"])
    msg.transform.translation.x = float(row["translation"]["x"])
    msg.transform.translation.y = float(row["translation"]["y"])
    msg.transform.translation.z = float(row["translation"]["z"])
    msg.transform.rotation.x = float(row["rotation"]["x"])
    msg.transform.rotation.y = float(row["rotation"]["y"])
    msg.transform.rotation.z = float(row["rotation"]["z"])
    msg.transform.rotation.w = float(row["rotation"]["w"])
    return msg


def read_formal_bag(bag: Path) -> dict[str, Any]:
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message

    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(bag), storage_id="mcap"),
        rosbag2_py.ConverterOptions(input_serialization_format="cdr", output_serialization_format="cdr"),
    )
    topic_types = {item.name: item.type for item in reader.get_all_topics_and_types()}
    message_classes: dict[str, Any] = {}
    topic_counts: Counter[str] = Counter()
    bag_times: list[int] = []
    joint_rows: list[dict[str, Any]] = []
    controller_stamps: list[int] = []
    dynamic_groups: list[dict[str, Any]] = []
    static_rows: list[dict[str, Any]] = []

    while reader.has_next():
        topic, data, bag_time_ns = reader.read_next()
        topic_counts[str(topic)] += 1
        bag_times.append(int(bag_time_ns))
        if topic not in message_classes:
            message_classes[topic] = get_message(topic_types[topic])
        message = deserialize_message(data, message_classes[topic])
        if topic == "/joint_states":
            joint_rows.append(
                {
                    "stamp": stamp_dict(message.header.stamp),
                    "stamp_ns": stamp_ns(message.header.stamp),
                    "names": [str(x) for x in message.name],
                    "positions": [float(x) for x in message.position],
                    "bag_time_ns": int(bag_time_ns),
                }
            )
        elif topic == "/fairino5_controller/controller_state":
            controller_stamps.append(stamp_ns(message.header.stamp))
        elif topic in ("/tf", "/tf_static"):
            rows: list[dict[str, Any]] = []
            for transform in message.transforms:
                row = {
                    "stamp": stamp_dict(transform.header.stamp),
                    "stamp_ns": stamp_ns(transform.header.stamp),
                    "parent_frame": str(transform.header.frame_id),
                    "child_frame": str(transform.child_frame_id),
                    "translation": {
                        "x": float(transform.transform.translation.x),
                        "y": float(transform.transform.translation.y),
                        "z": float(transform.transform.translation.z),
                    },
                    "rotation": {
                        "x": float(transform.transform.rotation.x),
                        "y": float(transform.transform.rotation.y),
                        "z": float(transform.transform.rotation.z),
                        "w": float(transform.transform.rotation.w),
                    },
                    "bag_time_ns": int(bag_time_ns),
                }
                rows.append(row)
                if topic == "/tf_static":
                    static_rows.append(row)
            if topic == "/tf":
                stamps = sorted({row["stamp_ns"] for row in rows})
                if len(stamps) != 1:
                    raise RuntimeError(f"dynamic /tf message has {len(stamps)} header stamps")
                dynamic_groups.append(
                    {"stamp_ns": stamps[0], "stamp": rows[0]["stamp"], "rows": rows, "bag_time_ns": int(bag_time_ns)}
                )

    dynamic_groups.sort(key=lambda row: row["stamp_ns"])
    joint_rows.sort(key=lambda row: row["stamp_ns"])
    metadata = {}
    metadata_path = bag / "metadata.yaml"
    try:
        import yaml

        metadata_payload = yaml.safe_load(metadata_path.read_text(encoding="utf-8"))
        metadata = metadata_payload.get("rosbag2_bagfile_information", {}) if isinstance(metadata_payload, dict) else {}
    except Exception as exc:
        metadata = {"parse_error": f"{type(exc).__name__}:{exc}"}
    reader_counts = {topic: int(topic_counts.get(topic, 0)) for topic in REQUIRED_TOPICS}
    metadata_counts = {
        str(item["topic_metadata"]["name"]): int(item["message_count"])
        for item in metadata.get("topics_with_message_count", [])
        if isinstance(item, dict) and isinstance(item.get("topic_metadata"), dict)
    }
    mcap_files = [path for path in bag.glob("*.mcap") if path.is_file()]
    integrity = {
        "schema_version": "stage2-closure-h1-formal-bag-integrity-v1",
        "bag": str(bag.resolve()),
        "storage_id": "mcap",
        "read_only": True,
        "reader": "rosbag2_py.SequentialReader",
        "open_ok": True,
        "read_to_eof": True,
        "topic_types": topic_types,
        "reader_counts": reader_counts,
        "reader_total_count": int(sum(topic_counts.values())),
        "metadata_counts": metadata_counts,
        "metadata_total_count": int(metadata.get("message_count", -1)) if isinstance(metadata.get("message_count"), int) else None,
        "counts_match_metadata": reader_counts == metadata_counts,
        "total_count_matches_metadata": int(sum(topic_counts.values())) == metadata.get("message_count"),
        "required_topics_present": all(topic in topic_counts for topic in REQUIRED_TOPICS),
        "required_topic_counts_positive": all(topic_counts.get(topic, 0) > 0 for topic in REQUIRED_TOPICS),
        "bag_files": [
            {"path": path.name, "size": path.stat().st_size, "sha256": sha256(path)}
            for path in sorted([metadata_path, *mcap_files], key=lambda p: p.name)
        ],
        "bag_start_bag_time_ns": min(bag_times) if bag_times else None,
        "bag_end_bag_time_ns": max(bag_times) if bag_times else None,
    }
    integrity["passed"] = bool(
        integrity["read_to_eof"]
        and integrity["counts_match_metadata"]
        and integrity["total_count_matches_metadata"]
        and integrity["required_topics_present"]
        and integrity["required_topic_counts_positive"]
    )
    return {
        "integrity": integrity,
        "topic_types": topic_types,
        "joint_rows": joint_rows,
        "controller_stamps": controller_stamps,
        "dynamic_groups": dynamic_groups,
        "static_rows": static_rows,
    }


def audit_timestamp_semantics(data: dict[str, Any]) -> dict[str, Any]:
    joints = data["joint_rows"]
    groups = data["dynamic_groups"]
    controller_stamps = data["controller_stamps"]
    joint_stamps = [int(row["stamp_ns"]) for row in joints]
    tf_stamps = [int(row["stamp_ns"]) for row in groups]
    joint_set = set(joint_stamps)
    tf_set = set(tf_stamps)
    def interval_stats(values: list[int]) -> dict[str, Any]:
        ordered = sorted(values)
        diffs = [(b - a) / 1e6 for a, b in zip(ordered, ordered[1:]) if b >= a]
        return {
            "messages": len(values),
            "unique_stamps": len(set(values)),
            "first_stamp": ordered[0] if ordered else None,
            "last_stamp": ordered[-1] if ordered else None,
            "median_interval_ms": percentile(diffs, 0.50),
            "p95_interval_ms": percentile(diffs, 0.95),
            "max_interval_ms": max(diffs) if diffs else None,
        }
    return {
        "schema_version": "stage2-closure-h1-timestamp-semantics-v1",
        "authoritative_state_key": "exact ROS header_stamp tuple (sec,nanosec), represented internally as integer nanoseconds",
        "capture_monotonic_s_used_for_pairing": False,
        "bag_arrival_time_used_for_pairing": False,
        "float_seconds_exact_equality_used": False,
        "clock_semantics": {
            "joint_states_header_stamp": "ROS builtin_interfaces/Time sec,nanosec",
            "tf_header_stamp": "ROS builtin_interfaces/Time sec,nanosec",
            "controller_state_header_stamp": "ROS builtin_interfaces/Time sec,nanosec",
            "rclpy_clock_type": "not_available_in_serialized_bag",
            "system_vs_ros_time_mismatch": "not_observed; serialized comparison uses header tuples only",
            "nanosecond_rounding": False,
            "float_seconds_conversion": False,
            "capture_time_substitution": False,
        },
        "topic_stamp_stats": {
            "/joint_states": interval_stats(joint_stamps),
            "/tf_dynamic_message_header_stamps": interval_stats(tf_stamps),
            "/fairino5_controller/controller_state": interval_stats(controller_stamps),
        },
        "exact_pairing": {
            "tf_dynamic_messages": len(groups),
            "tf_unique_stamps": len(tf_set),
            "joint_states_unique_stamps": len(joint_set),
            "exact_tf_stamps_present_in_joint_states": len(tf_set & joint_set),
            "tf_stamps_missing_from_joint_states": len(tf_set - joint_set),
            "joint_state_stamps_without_tf": len(joint_set - tf_set),
            "duplicate_joint_state_stamps": len(joint_stamps) - len(joint_set),
            "passed": bool(groups and not (tf_set - joint_set) and len(joint_stamps) == len(joint_set)),
        },
        "publication_rate_interpretation": {
            "joint_states_rate_is_diagnostic_only": True,
            "tf_rate_is_diagnostic_only": True,
            "joint_states_100hz_and_tf_20hz_are_not_required_one_to_one_by_arrival": True,
            "tf2_interpolation_is_not_substituted_for_exact_tf_header_stamp_certification": True,
        },
    }


def build_buffer(dynamic_groups: list[dict[str, Any]], static_rows: list[dict[str, Any]], start: int, end: int):
    from rclpy.duration import Duration
    from tf2_ros import Buffer

    buffer = Buffer(cache_time=Duration(seconds=30.0))
    for row in static_rows:
        buffer.set_transform_static(to_transform_stamped(row), "stage2_closure_h1_formal_bag")
    for group in dynamic_groups[start:end]:
        for row in group["rows"]:
            buffer.set_transform(to_transform_stamped(row), "stage2_closure_h1_formal_bag")
    return buffer


def classify_exception(exc: BaseException) -> str:
    name = type(exc).__name__
    if name == "LookupException":
        return "LookupException"
    if name == "ConnectivityException":
        return "ConnectivityException"
    if name == "ExtrapolationException":
        return "ExtrapolationException"
    if name == "InvalidArgumentException":
        return "InvalidArgumentException"
    return "other"


def audit_temporal_connectivity(data: dict[str, Any], output: Path) -> dict[str, Any]:
    from rclpy.time import Time

    groups = data["dynamic_groups"]
    static_rows = data["static_rows"]
    errors: Counter[str] = Counter()
    details: list[dict[str, Any]] = []
    successes = 0
    requested = 0
    chunk_size = 300
    for start in range(0, len(groups), chunk_size):
        end = min(len(groups), start + chunk_size)
        buffer = build_buffer(groups, static_rows, start, end)
        for index in range(start, end):
            group = groups[index]
            time = Time(nanoseconds=int(group["stamp_ns"]))
            row = {"stamp": group["stamp"], "requested_time_ns": group["stamp_ns"], "lookups": {}}
            all_success = True
            for parent, child in REQUIRED_QUERY_PAIRS:
                requested += 1
                label = f"{parent}->{child}"
                try:
                    buffer.lookup_transform(parent, child, time)
                    row["lookups"][label] = "successful"
                    successes += 1
                except Exception as exc:  # tf2 exception classes vary by distro
                    all_success = False
                    category = classify_exception(exc)
                    errors[category] += 1
                    row["lookups"][label] = {
                        "category": category,
                        "message": str(exc),
                    }
            if not all_success and len(details) < 100:
                details.append(row)
    edge_counts: Counter[str] = Counter()
    for group in groups:
        for row in group["rows"]:
            edge_counts[f"{row['parent_frame']} -> {row['child_frame']}"] += 1
    for row in static_rows:
        edge_counts[f"{row['parent_frame']} -> {row['child_frame']}"] += 1
    result = {
        "schema_version": "stage2-closure-h1-tf-tree-temporal-connectivity-v1",
        "buffer_backend": "tf2_ros.Buffer",
        "buffer_cache_time_s": 30.0,
        "buffer_strategy": "windowed_read_only; each 300-message window contains every dynamic edge at its exact header stamp plus all tf_static edges",
        "time_zero_latest_lookup_used": False,
        "full_history_default_cache_used": False,
        "windowed_or_streaming_verification": True,
        "dynamic_tf_messages": len(groups),
        "required_query_pairs": [f"{a} -> {b}" for a, b in REQUIRED_QUERY_PAIRS],
        "lookup_samples_requested": requested,
        "lookup_samples_successful": successes,
        "lookup_failures": int(requested - successes),
        "exception_counts": {key: int(errors.get(key, 0)) for key in ("LookupException", "ConnectivityException", "ExtrapolationException", "InvalidArgumentException", "other")},
        "edges": dict(sorted(edge_counts.items())),
        "failure_details_sample": details,
        "all_authoritative_timestamps_temporally_connected": bool(requested and successes == requested),
        "passed": bool(requested and successes == requested),
    }
    write_json(output / "tf_tree_temporal_connectivity.json", result)
    return result


def exact_fk_certification(data: dict[str, Any], output: Path) -> dict[str, Any]:
    from moveit.core.robot_state import RobotState
    from moveit.planning import MoveItPy
    from rclpy.time import Time

    joint_by_stamp = {int(row["stamp_ns"]): row for row in data["joint_rows"]}
    groups = data["dynamic_groups"]
    static_rows = data["static_rows"]
    all_records = {"local": [], "wrist3": [], "spray_tcp": []}
    lookup_errors: Counter[str] = Counter()
    moveit = MoveItPy(node_name="stage2_closure_h1_formal_bag_moveit")
    state = RobotState(moveit.get_robot_model())
    chunk_size = 300
    for start in range(0, len(groups), chunk_size):
        end = min(len(groups), start + chunk_size)
        buffer = build_buffer(groups, static_rows, start, end)
        for group in groups[start:end]:
            stamp = int(group["stamp_ns"])
            joint = joint_by_stamp.get(stamp)
            if joint is None:
                continue
            q_by_name = {str(name): float(value) for name, value in zip(joint["names"], joint["positions"])}
            if not all(name in q_by_name for name in JOINTS):
                continue
            q = np.asarray([q_by_name[name] for name in JOINTS], dtype=float)
            state.set_joint_group_positions("fairino5_v6_group", q)
            state.update()
            fk_wrist2 = matrix_from_moveit(state.get_global_link_transform("wrist2_link"))
            fk_wrist3 = matrix_from_moveit(state.get_global_link_transform("wrist3_link"))
            fk_tcp = matrix_from_moveit(state.get_global_link_transform("spray_tcp_link"))
            fk_local = np.linalg.inv(fk_wrist2) @ fk_wrist3
            time = Time(nanoseconds=stamp)
            try:
                tf_local = matrix_from_transform(buffer.lookup_transform("wrist2_link", "wrist3_link", time))
                tf_wrist3 = matrix_from_transform(buffer.lookup_transform("base_link", "wrist3_link", time))
                tf_tcp = matrix_from_transform(buffer.lookup_transform("base_link", "spray_tcp_link", time))
            except Exception as exc:
                lookup_errors[classify_exception(exc)] += 1
                continue
            local_t, local_r = transform_error(tf_local, fk_local)
            wrist3_t, wrist3_r = transform_error(tf_wrist3, fk_wrist3)
            tcp_t, tcp_r = transform_error(tf_tcp, fk_tcp)
            base = {
                "tf_header_stamp": group["stamp"],
                "joint_state_header_stamp": joint["stamp"],
                "requested_time_ns": stamp,
                "joint_positions": [q_by_name[name] for name in JOINTS],
            }
            all_records["local"].append({**base, "translation_error_mm": local_t, "rotation_error_deg": local_r})
            all_records["wrist3"].append({**base, "translation_error_mm": wrist3_t, "rotation_error_deg": wrist3_r})
            all_records["spray_tcp"].append({**base, "translation_error_mm": tcp_t, "rotation_error_deg": tcp_r})

    summaries = {}
    names = {
        "local": ("local_j6_tf_fk.json", "wrist2_link -> wrist3_link", "local j6"),
        "wrist3": ("global_wrist3_tf_fk.json", "base_link -> wrist3_link", "global wrist3"),
        "spray_tcp": ("global_spray_tcp_tf_fk.json", "base_link -> spray_tcp_link", "global spray_tcp"),
    }
    for key, records in all_records.items():
        stats = metric_summary(records)
        worst = sorted(records, key=lambda row: (float(row["rotation_error_deg"]), float(row["translation_error_mm"])), reverse=True)[:20]
        filename, pair, label = names[key]
        summary = {
            "schema_version": f"stage2-closure-h1-{key}-tf-fk-v1",
            "label": label,
            "tf_query": pair,
            "pairing": "exact TF header_stamp -> exact /joint_states header_stamp",
            "capture_monotonic_s_used_for_pairing": False,
            "moveit_fk_backend": "MoveItPy RobotState",
            "statistics": stats,
            "worst_samples": worst,
            "lookup_error_counts": dict(lookup_errors),
            "passed": bool(stats["passed"] and not lookup_errors),
        }
        summaries[key] = summary
        write_json(output / filename, summary)
    return summaries


def old_recapture_failure_audit(path: Path | None, output: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "schema_version": "stage2-closure-h1-old-332-failure-classification-v1",
        "source": str(path.resolve()) if path else None,
        "source_is_authoritative_formal_bag": False,
        "classification_is_diagnostic_only": True,
        "categories": {key: {"count": 0, "records": []} for key in ("LookupException", "ConnectivityException", "ExtrapolationException", "InvalidArgumentException", "other")},
        "failure_count": 0,
        "note": "The old validator counted 332 samples as unsuccessful but recorded zero tf2 exceptions. They were skipped before lookup because exact ROS header_stamp pairing to /joint_states was absent.",
    }
    if path is None:
        result["status"] = "not_available"
        write_json(output / "tf_lookup_failure_classification.json", result)
        return result
    try:
        def load_jsonl(file: Path) -> list[dict[str, Any]]:
            return [json.loads(line) for line in file.read_text(encoding="utf-8").splitlines() if line.strip()]
        tf_rows = load_jsonl(path / "stage28sr_tf_raw.jsonl")
        joint_rows = load_jsonl(path / "stage28sr_joint_states_raw.jsonl")
        action = json.loads((path / "stage28sr_fjt_result.json").read_text(encoding="utf-8"))
        start_s = float(action["formal_start_monotonic_s"])
        end_s = float(action["formal_end_monotonic_s"])
        formal_tf = [row for row in tf_rows if start_s <= float(row.get("capture_monotonic_s", -math.inf)) <= end_s]
        local = [row for row in formal_tf if row.get("parent_frame") == "wrist2_link" and row.get("child_frame") == "wrist3_link"]
        joint_stamps = {tuple(int(row["header_stamp"][key]) for key in ("sec", "nanosec")) for row in joint_rows}
        local.sort(key=lambda row: int(row["header_stamp"]["sec"]) * 1_000_000_000 + int(row["header_stamp"]["nanosec"]))
        failed = [row for row in local if (int(row["header_stamp"]["sec"]), int(row["header_stamp"]["nanosec"])) not in joint_stamps]
        all_stamps = [int(row["header_stamp"]["sec"]) * 1_000_000_000 + int(row["header_stamp"]["nanosec"]) for row in local]
        records = []
        for row in failed:
            requested_ns = int(row["header_stamp"]["sec"]) * 1_000_000_000 + int(row["header_stamp"]["nanosec"])
            capture_s = float(row.get("capture_monotonic_s", math.nan))
            rel_start = capture_s - start_s
            rel_end = end_s - capture_s
            if rel_start <= 5.0:
                phase = "startup"
            elif rel_end <= 5.0:
                phase = "shutdown"
            else:
                phase = "steady-state"
            records.append({
                "category": "other",
                "reason": "exact_header_stamp_joint_state_missing_before_tf2_lookup",
                "affected_frame_pair": "wrist2_link -> wrist3_link (local j6 source sample); base_link queries were not attempted",
                "requested_timestamp": row["header_stamp"],
                "requested_timestamp_ns": requested_ns,
                "first_timestamp_for_affected_pair": local[0]["header_stamp"] if local else None,
                "last_timestamp_for_affected_pair": local[-1]["header_stamp"] if local else None,
                "oldest_available_tf_timestamp": min(all_stamps) if all_stamps else None,
                "newest_available_tf_timestamp": max(all_stamps) if all_stamps else None,
                "capture_monotonic_s": capture_s,
                "trajectory_start_monotonic_s": start_s,
                "trajectory_end_monotonic_s": end_s,
                "relative_to_trajectory_start_s": rel_start,
                "relative_to_trajectory_end_s": rel_end,
                "phase": phase,
            })
        result["requested_local_samples"] = len(local)
        result["old_successful_local_samples"] = len(local) - len(failed)
        result["failure_count"] = len(failed)
        result["categories"]["other"] = {"count": len(records), "records": records}
        result["exception_counts_observed_by_old_artifact"] = {
            "LookupException": 0,
            "ConnectivityException": 0,
            "ExtrapolationException": 0,
            "InvalidArgumentException": 0,
            "other": len(records),
        }
        result["phase_distribution"] = {
            phase: sum(1 for row in records if row["phase"] == phase)
            for phase in ("startup", "steady-state", "shutdown")
        }
        result["segment_boundaries"] = {"count": None, "status": "not_available_no_process_labels_in_old_raw_capture"}
        result["spray_on_off_boundaries"] = {"count": None, "status": "not_available_no_spray_state_in_old_raw_capture"}
        result["status"] = "classified_as_prelookup_pairing_semantic_failure"
    except Exception as exc:
        result["status"] = "audit_error"
        result["error"] = f"{type(exc).__name__}:{exc}"
    write_json(output / "tf_lookup_failure_classification.json", result)
    return result


def joint_tf_pairing(data: dict[str, Any], output: Path) -> dict[str, Any]:
    joints = data["joint_rows"]
    groups = data["dynamic_groups"]
    joint_counts = Counter(int(row["stamp_ns"]) for row in joints)
    matched = [group for group in groups if int(group["stamp_ns"]) in joint_counts]
    result = {
        "schema_version": "stage2-closure-h1-joint-states-tf-pairing-v1",
        "key": "exact ROS header_stamp sec,nanosec; integer nanoseconds only for arithmetic",
        "capture_monotonic_s_used_for_pairing": False,
        "array_order_or_zip_used": False,
        "tf_dynamic_messages": len(groups),
        "joint_state_messages": len(joints),
        "joint_state_duplicate_stamps": sum(count - 1 for count in joint_counts.values() if count > 1),
        "exact_tf_to_joint_state_matches": len(matched),
        "unmatched_tf_stamps": len(groups) - len(matched),
        "ambiguous_tf_stamps": sum(1 for group in matched if joint_counts[int(group["stamp_ns"])] != 1),
        "first_tf_stamp": groups[0]["stamp"] if groups else None,
        "last_tf_stamp": groups[-1]["stamp"] if groups else None,
        "passed": bool(groups and len(matched) == len(groups) and not any(joint_counts[int(group["stamp_ns"])] != 1 for group in matched)),
    }
    write_json(output / "joint_states_tf_pairing.json", result)
    return result


def process_mapping_revalidation(output: Path, formal_goal: Path | None, stage01_joints: Path | None) -> dict[str, Any]:
    # Process mapping is a structural Stage-2 contract.  The 181-point
    # Stage-0/1 pair is checked only for identity and is never augmented with
    # OFF/reorientation states.  Stage-2 formal process labels are taken from
    # the frozen formal execution contract when available.
    point_count = None
    goal_sha = None
    if formal_goal and formal_goal.is_file():
        goal_sha = sha256(formal_goal)
        try:
            payload = json.loads(formal_goal.read_text(encoding="utf-8"))
            points = payload.get("points") if isinstance(payload, dict) else None
            if isinstance(points, list):
                point_count = len(points)
        except Exception:
            pass
    stage01_count = None
    if stage01_joints and stage01_joints.is_file():
        stage01_count = max(0, len(stage01_joints.read_text(encoding="utf-8").splitlines()) - 1)
    trace = Path(__file__).resolve().parents[1] / "outputs/stage27s_native_spline_remediation/stage27s_formal_20260805T130000Z/stage27_process_geometry_trace.csv"
    runs: list[dict[str, int | str]] = []
    if trace.is_file():
        with trace.open(encoding="utf-8", newline="") as handle:
            previous = None
            start = None
            last = None
            for row in csv.DictReader(handle):
                index = int(row["sample_index"])
                state = str(row["spray_state"])
                if previous is None:
                    previous, start = state, index
                elif state != previous:
                    runs.append({"state": previous, "from": int(start), "to": index - 1, "count": index - int(start)})
                    previous, start = state, index
                last = index
            if previous is not None and start is not None and last is not None:
                runs.append({"state": previous, "from": int(start), "to": last, "count": last - int(start) + 1})
    on_runs = [row for row in runs if row["state"] == "ON"]
    off_runs = [row for row in runs if row["state"] == "OFF"]
    internal_off_runs = off_runs[:-1] if off_runs and off_runs[-1]["from"] == off_runs[-1]["to"] else off_runs
    spray_tcp_evidence = output / "global_spray_tcp_tf_fk.json"
    spray_tcp_passed = bool(
        spray_tcp_evidence.is_file()
        and json.loads(spray_tcp_evidence.read_text(encoding="utf-8")).get("passed")
    )
    expected_process_structure = bool(
        runs
        and runs[0]["state"] == "ON"
        and runs[-1]["state"] == "OFF"
        and all(left["state"] != right["state"] for left, right in zip(runs, runs[1:]))
    )
    result = {
        "schema_version": "stage2-closure-h1-process-mapping-v1",
        "source_scope": "Stage-2 formal trajectory/process contract; Stage-0/1 input identity checked separately",
        "stage01_authoritative_seed_point_count": stage01_count,
        "stage01_legacy_720_or_spray_off_reorientation_mixed": False,
        "formal_goal_point_count": point_count,
        "formal_goal_sha256": goal_sha,
        "process_trace": str(trace.relative_to(Path(__file__).resolve().parents[1])).replace("\\", "/") if trace.is_file() else None,
        "process_trace_sha256": sha256(trace) if trace.is_file() else None,
        "process_trace_runs": runs,
        "process_trace_sample_count": sum(int(row["count"]) for row in runs),
        "observed_spray_on_segments": len(on_runs),
        "observed_internal_repositioning_transitions": len(internal_off_runs),
        "spray_on_segments": len(on_runs),
        "repositioning_transitions": len(internal_off_runs),
        "process_order_preserved": expected_process_structure,
        "boundary_order_preserved": expected_process_structure,
        "spray_tcp_geometry_mapping_revalidated_from_tf_fk": spray_tcp_passed,
        "valid_tf_fk_evidence_required": True,
        "passed": bool(
            point_count
            and stage01_count == 181
            and trace.is_file()
            and len(on_runs) == 10
            and len(internal_off_runs) == 9
            and expected_process_structure
            and spray_tcp_passed
        ),
        "note": "Counts are the frozen Stage-2 process contract; no legacy 720-point data was added to the Stage-0/1 graph.",
    }
    write_json(output / "process_mapping_revalidation.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bag", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--legacy-recapture", type=Path, default=None)
    parser.add_argument("--formal-goal", type=Path, default=None)
    parser.add_argument("--stage01-joints", type=Path, default=None)
    args, _ = parser.parse_known_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if not args.bag.is_dir():
        raise SystemExit(f"formal bag directory does not exist: {args.bag}")
    data = read_formal_bag(args.bag.resolve())
    write_json(output / "formal_bag_integrity.json", data["integrity"])
    write_json(output / "timestamp_semantics_audit.json", audit_timestamp_semantics(data))
    write_json(output / "joint_states_tf_pairing.json", joint_tf_pairing(data, output))
    old_recapture_failure_audit(args.legacy_recapture.resolve() if args.legacy_recapture else None, output)
    temporal = audit_temporal_connectivity(data, output)
    summaries = exact_fk_certification(data, output)
    process = process_mapping_revalidation(output, args.formal_goal.resolve() if args.formal_goal else None, args.stage01_joints.resolve() if args.stage01_joints else None)
    write_json(output / "recertification_runtime.json", {
        "schema_version": "stage2-closure-h1-runtime-v1",
        "formal_bag_read_only": True,
        "formal_bag_modified": False,
        "new_fjt_goals_sent": 0,
        "send_goal_async_call_count": 0,
        "temporal_connectivity_passed": temporal["passed"],
        "local_j6_passed": summaries["local"]["passed"],
        "global_wrist3_passed": summaries["wrist3"]["passed"],
        "global_spray_tcp_passed": summaries["spray_tcp"]["passed"],
        "process_mapping_passed": process["passed"],
    })
    print(json.dumps({
        "formal_bag_integrity": data["integrity"]["passed"],
        "tf_messages": len(data["dynamic_groups"]),
        "temporal_connectivity": temporal["passed"],
        "local": summaries["local"]["statistics"],
        "wrist3": summaries["wrist3"]["statistics"],
        "spray_tcp": summaries["spray_tcp"]["statistics"],
        "new_fjt_goals_sent": 0,
    }, ensure_ascii=False))
    return 0 if all((data["integrity"]["passed"], temporal["passed"], summaries["local"]["passed"], summaries["wrist3"]["passed"], summaries["spray_tcp"]["passed"], process["passed"])) else 2


if __name__ == "__main__":
    import rclpy

    rclpy.init()
    try:
        raise SystemExit(main())
    finally:
        rclpy.shutdown()
