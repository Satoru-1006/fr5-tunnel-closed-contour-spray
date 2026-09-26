"""Strict Stage 2.8S-R TF/FK and simulation-state certification.

This file runs inside the ROS 2 environment.  It reconstructs the complete
recorded TF graph in a real tf2 Buffer, then compares TF2 and MoveIt FK at the
same dynamic TF header stamp.  Capture monotonic time is never a physical
state key.
"""

from __future__ import annotations

import argparse
import bisect
import json
import math
import re
from pathlib import Path

import numpy as np
import rclpy
from geometry_msgs.msg import TransformStamped
from moveit.core.robot_state import RobotState
from moveit.planning import MoveItPy
from rclpy.duration import Duration
from rclpy.time import Time
from tf2_ros import Buffer, ConnectivityException, ExtrapolationException, LookupException, TransformException


JOINTS = [f"j{i}" for i in range(1, 7)]
ROTATION_GATE_DEG = 0.1
TRANSLATION_GATE_MM = 1.0e-3


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def stamp_key(value: dict) -> tuple[int, int]:
    return int(value["sec"]), int(value["nanosec"])


def stamp_ns(value: dict) -> int:
    sec, nsec = stamp_key(value)
    return sec * 1_000_000_000 + nsec


def stamp_time(value: dict) -> Time:
    return Time(nanoseconds=stamp_ns(value))


def matrix_from_moveit(value) -> np.ndarray:
    return np.asarray(value.matrix() if hasattr(value, "matrix") else value, dtype=float)


def matrix_from_row(row: dict) -> np.ndarray:
    q = row.get("quaternion_xyzw", row.get("rotation"))
    x, y, z, w = (float(q[key]) for key in ("x", "y", "z", "w"))
    tx, ty, tz = (float(row["translation"][key]) for key in ("x", "y", "z"))
    return np.asarray([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w), tx],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w), ty],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y), tz],
        [0.0, 0.0, 0.0, 1.0],
    ], dtype=float)


def matrix_from_transform(value: TransformStamped) -> np.ndarray:
    return matrix_from_row({
        "translation": {"x": value.transform.translation.x, "y": value.transform.translation.y, "z": value.transform.translation.z},
        "quaternion_xyzw": {"x": value.transform.rotation.x, "y": value.transform.rotation.y, "z": value.transform.rotation.z, "w": value.transform.rotation.w},
    })


def matrix_to_quat(matrix: np.ndarray) -> list[float]:
    r = np.asarray(matrix[:3, :3], dtype=float)
    trace = float(np.trace(r))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (r[2, 1] - r[1, 2]) / s
        y = (r[0, 2] - r[2, 0]) / s
        z = (r[1, 0] - r[0, 1]) / s
    elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
        s = math.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2.0
        w = (r[2, 1] - r[1, 2]) / s
        x = 0.25 * s
        y = (r[0, 1] + r[1, 0]) / s
        z = (r[0, 2] + r[2, 0]) / s
    elif r[1, 1] > r[2, 2]:
        s = math.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2.0
        w = (r[0, 2] - r[2, 0]) / s
        x = (r[0, 1] + r[1, 0]) / s
        y = 0.25 * s
        z = (r[1, 2] + r[2, 1]) / s
    else:
        s = math.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2.0
        w = (r[1, 0] - r[0, 1]) / s
        x = (r[0, 2] + r[2, 0]) / s
        y = (r[1, 2] + r[2, 1]) / s
        z = 0.25 * s
    result = np.asarray([x, y, z, w], dtype=float)
    result /= max(float(np.linalg.norm(result)), 1.0e-30)
    return [float(x) for x in result]


def transform_error(lhs: np.ndarray, rhs: np.ndarray) -> tuple[float, float]:
    translation_mm = float(np.linalg.norm(lhs[:3, 3] - rhs[:3, 3]) * 1000.0)
    delta = lhs[:3, :3] @ rhs[:3, :3].T
    cosine = float(np.clip((np.trace(delta) - 1.0) * 0.5, -1.0, 1.0))
    return translation_mm, math.degrees(math.acos(cosine))


def stats(records: list[dict], translation_key: str = "translation_error_mm", rotation_key: str = "rotation_error_deg") -> dict:
    translations = np.asarray([float(row[translation_key]) for row in records], dtype=float) if records else np.asarray([], dtype=float)
    rotations = np.asarray([float(row[rotation_key]) for row in records], dtype=float) if records else np.asarray([], dtype=float)
    return {
        "samples": len(records),
        "max_translation_error_mm": float(np.max(translations)) if len(translations) else None,
        "median_translation_error_mm": float(np.median(translations)) if len(translations) else None,
        "p95_translation_error_mm": float(np.percentile(translations, 95)) if len(translations) else None,
        "p99_translation_error_mm": float(np.percentile(translations, 99)) if len(translations) else None,
        "max_rotation_error_deg": float(np.max(rotations)) if len(rotations) else None,
        "median_rotation_error_deg": float(np.median(rotations)) if len(rotations) else None,
        "p95_rotation_error_deg": float(np.percentile(rotations, 95)) if len(rotations) else None,
        "p99_rotation_error_deg": float(np.percentile(rotations, 99)) if len(rotations) else None,
        "translation_gate_mm": TRANSLATION_GATE_MM,
        "rotation_gate_deg": ROTATION_GATE_DEG,
        "passed": bool(records and np.max(translations) <= TRANSLATION_GATE_MM and np.max(rotations) <= ROTATION_GATE_DEG),
    }


def json_safe(value):
    if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        return None
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    return value


def to_tf(row: dict) -> TransformStamped:
    message = TransformStamped()
    message.header.stamp.sec = int(row["header_stamp"]["sec"])
    message.header.stamp.nanosec = int(row["header_stamp"]["nanosec"])
    message.header.frame_id = str(row["parent_frame"])
    message.child_frame_id = str(row["child_frame"])
    message.transform.translation.x = float(row["translation"]["x"])
    message.transform.translation.y = float(row["translation"]["y"])
    message.transform.translation.z = float(row["translation"]["z"])
    q = row.get("quaternion_xyzw", row.get("rotation"))
    message.transform.rotation.x = float(q["x"])
    message.transform.rotation.y = float(q["y"])
    message.transform.rotation.z = float(q["z"])
    message.transform.rotation.w = float(q["w"])
    return message


def nearest_controller_state(rows: list[dict], target_ns: int, *, ordered: list[dict] | None = None, keys: list[int] | None = None) -> dict | None:
    # The caller may provide one pre-sorted header-time index.  Re-sorting a
    # 75k-row capture for every callback/TF row is only a diagnostic-audit
    # performance bug and must never be confused with physical pairing.
    ordered = ordered if ordered is not None else sorted(rows, key=lambda row: stamp_ns(row["header_stamp"]))
    keys = keys if keys is not None else [stamp_ns(row["header_stamp"]) for row in ordered]
    index = bisect.bisect_left(keys, target_ns)
    candidates = ordered[max(0, index - 1):min(len(ordered), index + 1)]
    if not candidates:
        return None
    return min(candidates, key=lambda row: abs(stamp_ns(row["header_stamp"]) - target_ns))


def write(path: Path, value) -> None:
    path.write_text(json.dumps(json_safe(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    # ros2 launch appends --ros-args/--params-file to the process command
    # line.  The MoveIt model parameters are intentionally supplied by the
    # launch action; the validator only owns its --output argument.
    args, _ros_args = parser.parse_known_args()
    output = args.output.resolve()
    joint_rows = read_jsonl(output / "stage28sr_joint_states_raw.jsonl")
    controller_rows = read_jsonl(output / "stage28sr_controller_state_raw.jsonl")
    tf_rows_all = read_jsonl(output / "stage28sr_tf_raw.jsonl")
    static_rows = read_jsonl(output / "stage28sr_tf_static_raw.jsonl")
    action = read_json(output / "stage28sr_fjt_result.json")
    formal_start = float(action.get("formal_start_monotonic_s", -float("inf")))
    formal_end = float(action.get("formal_end_monotonic_s", float("inf")))
    tf_rows = [row for row in tf_rows_all if formal_start <= float(row["capture_monotonic_s"]) <= formal_end]
    joint_by_stamp: dict[tuple[int, int], list[dict]] = {}
    for row in joint_rows:
        joint_by_stamp.setdefault(stamp_key(row["header_stamp"]), []).append(row)

    dynamic_matches = 0
    dynamic_unmatched = []
    for row in tf_rows:
        if stamp_key(row["header_stamp"]) in joint_by_stamp and len(joint_by_stamp[stamp_key(row["header_stamp"])]) == 1:
            dynamic_matches += 1
        else:
            dynamic_unmatched.append(row)
    dynamic_message_indices = {int(row.get("tf_message_index", row.get("message_index", -1))) for row in tf_rows}
    timestamp_mapping = {
        "schema_version": "stage28sr-tf-joint-state-stamp-mapping-v1",
        "physical_pairing_key": "exact ROS header_stamp sec,nanosec",
        "capture_monotonic_s_used_for_pairing": False,
        "formal_capture_window": {"start_monotonic_s": formal_start, "end_monotonic_s": formal_end},
        "dynamic_tf_messages": len(dynamic_message_indices),
        "dynamic_tf_transform_records": len(tf_rows),
        "exact_stamp_match_count": dynamic_matches,
        "exact_stamp_match_ratio": float(dynamic_matches / len(tf_rows)) if tf_rows else 0.0,
        "unmatched_count": len(dynamic_unmatched),
        "unmatched_examples": dynamic_unmatched[:20],
        "exact_state_correspondence": "proven" if tf_rows and not dynamic_unmatched else "not_proven",
        "status": "passed" if tf_rows and not dynamic_unmatched else "blocked_exact_stamp_correspondence",
    }
    write(output / "stage28sr_tf_joint_state_stamp_mapping.json", timestamp_mapping)

    # The independent controller-feedback audit uses ROS header time only.  It
    # may use a one-to-one header-time correspondence for the independently
    # published controller state, but never substitutes it for formal TF/FK.
    joint_rows_by_header_time = sorted(joint_rows, key=lambda row: stamp_ns(row["header_stamp"]))
    joint_header_times = [stamp_ns(row["header_stamp"]) for row in joint_rows_by_header_time]
    controller_matches = []
    for row in controller_rows:
        candidate = nearest_controller_state(joint_rows, stamp_ns(row["header_stamp"]), ordered=joint_rows_by_header_time, keys=joint_header_times)
        if candidate is None:
            continue
        delta_s = abs(stamp_ns(candidate["header_stamp"]) - stamp_ns(row["header_stamp"])) * 1.0e-9
        if delta_s > 0.002:
            continue
        js_by_name = {str(name): float(value) for name, value in zip(candidate.get("names", candidate.get("name", [])), candidate.get("positions", candidate.get("position", [])))}
        feedback = row.get("feedback", {}).get("positions", [])
        by_controller_name = {str(name): float(value) for name, value in zip(row.get("joint_names", []), feedback)}
        if not all(joint in js_by_name and joint in by_controller_name for joint in JOINTS):
            continue
        errors = [abs(js_by_name[joint] - by_controller_name[joint]) for joint in JOINTS]
        controller_matches.append({"controller_header_stamp": row["header_stamp"], "joint_state_header_stamp": candidate["header_stamp"], "header_stamp_delta_s": delta_s, "max_joint_position_error_rad": max(errors)})
    controller_errors = np.asarray([row["max_joint_position_error_rad"] for row in controller_matches], dtype=float)
    joint_feedback = {
        "schema_version": "stage28sr-joint-states-vs-controller-feedback-v1",
        "samples": len(controller_matches),
        "max_joint_position_error_rad": float(np.max(controller_errors)) if len(controller_errors) else None,
        "p99_joint_position_error_rad": float(np.percentile(controller_errors, 99)) if len(controller_errors) else None,
        "timestamp_method": "ROS header_stamp one-to-one nearest correspondence for independent audit; max_delta_s<=0.002; not used for TF/FK",
        "common_ros_timestamp_exact_count": sum(stamp_key(row["header_stamp"]) in joint_by_stamp for row in controller_rows),
        "max_header_stamp_delta_s": max((row["header_stamp_delta_s"] for row in controller_matches), default=None),
        "passed": bool(controller_matches and np.max(controller_errors) <= 1.0e-12),
    }
    write(output / "stage28sr_joint_state_vs_controller_feedback.json", joint_feedback)

    # The raw capture is complete, and the formal lookup is performed with a
    # real tf2 Buffer.  A single Buffer containing the full 765 s history is
    # prohibitively expensive in this tf2 build because each TimeCache insert
    # scans the retained history.  Partitioning by formal time window keeps
    # the graph complete at every queried T: every dynamic transform record
    # for every timestamp in the window is inserted using its own header.stamp,
    # together with every static transform.  No capture-time pairing or
    # transform composition outside tf2 is used for the formal lookup.
    edge_counts: dict[str, dict[str, int]] = {}
    for row in tf_rows_all:
        key = f"{row['parent_frame']} -> {row['child_frame']}"
        edge_counts.setdefault(key, {"dynamic_records": 0, "static_records": 0})["dynamic_records"] += 1
    for row in static_rows:
        key = f"{row['parent_frame']} -> {row['child_frame']}"
        edge_counts.setdefault(key, {"dynamic_records": 0, "static_records": 0})["static_records"] += 1
    write(output / "stage28sr_tf_tree.json", {
        "schema_version": "stage28sr-tf-tree-v1",
        "buffer_backend": "tf2_ros.Buffer",
        "all_raw_dynamic_transform_records_loaded": len(tf_rows_all),
        "all_raw_static_transform_records_loaded": len(static_rows),
        "formal_dynamic_transform_records_loaded": len(tf_rows),
        "buffer_strategy": "complete_per_timestamp_window; each window loads every dynamic record in its own header_stamp window plus all tf_static records",
        "buffer_window_samples": 100,
        "edges": edge_counts,
        "raw_sources": {"tf": str((output / "stage28sr_tf_raw.jsonl").resolve()), "tf_static": str((output / "stage28sr_tf_static_raw.jsonl").resolve())},
        "required_query_edges": ["base_link -> wrist2_link", "base_link -> wrist3_link", "base_link -> spray_tcp_link"],
    })

    local_rows = [row for row in tf_rows if row.get("parent_frame") == "wrist2_link" and row.get("child_frame") == "wrist3_link"]
    moveit = MoveItPy(node_name="stage28sr_tf_fk_validator_moveit")
    state = RobotState(moveit.get_robot_model())
    local_records = []
    wrist3_records = []
    tcp_records = []
    lookup_exceptions = {"lookup_exception": 0, "connectivity_exception": 0, "extrapolation_exception": 0}
    lookup_samples = []
    controller_for_worst = sorted(controller_rows, key=lambda row: stamp_ns(row["header_stamp"]))
    controller_for_worst_times = [stamp_ns(row["header_stamp"]) for row in controller_for_worst]
    tf_by_stamp: dict[tuple[int, int], list[dict]] = {}
    for row in tf_rows:
        tf_by_stamp.setdefault(stamp_key(row["header_stamp"]), []).append(row)
    window_size = 100
    buffer = None
    for local_index, local in enumerate(local_rows):
        if local_index % window_size == 0:
            window_rows = local_rows[local_index:local_index + window_size]
            window_stamps = {stamp_key(row["header_stamp"]) for row in window_rows}
            buffer = Buffer(cache_time=Duration(seconds=30.0))
            for window_stamp in window_stamps:
                for row in tf_by_stamp.get(window_stamp, []):
                    buffer.set_transform(to_tf(row), "stage28sr_recorded_tf")
            for row in static_rows:
                buffer.set_transform_static(to_tf(row), "stage28sr_recorded_tf_static")
        key = stamp_key(local["header_stamp"])
        matches = joint_by_stamp.get(key, [])
        if len(matches) != 1:
            continue
        joint = matches[0]
        q_by_name = {str(name): float(value) for name, value in zip(joint.get("names", joint.get("name", [])), joint.get("positions", joint.get("position", [])))}
        if not all(joint_name in q_by_name for joint_name in JOINTS):
            continue
        q = np.asarray([q_by_name[joint_name] for joint_name in JOINTS], dtype=float)
        state.set_joint_group_positions("fairino5_v6_group", q)
        state.update()
        fk_wrist2 = matrix_from_moveit(state.get_global_link_transform("wrist2_link"))
        fk_wrist3 = matrix_from_moveit(state.get_global_link_transform("wrist3_link"))
        fk_tcp = matrix_from_moveit(state.get_global_link_transform("spray_tcp_link"))
        fk_local = np.linalg.inv(fk_wrist2) @ fk_wrist3
        local_translation, local_rotation = transform_error(matrix_from_row(local), fk_local)
        controller = nearest_controller_state(controller_for_worst, stamp_ns(local["header_stamp"]), ordered=controller_for_worst, keys=controller_for_worst_times)
        base_record = {
            "tf_header_stamp": local["header_stamp"],
            "joint_state_header_stamp": joint["header_stamp"],
            "controller_state_header_stamp": controller["header_stamp"] if controller else None,
            "capture_monotonic_times": {
                "tf": local.get("capture_monotonic_s"),
                "joint_state": joint.get("capture_monotonic_s"),
                "controller_state": controller.get("capture_monotonic_s") if controller else None,
            },
            "joint_positions": [q_by_name[joint_name] for joint_name in JOINTS],
            "joint_velocities": joint.get("velocities", joint.get("velocity", [])),
            "tf_local_quaternion_xyzw": local.get("quaternion_xyzw", local.get("rotation")),
            "fk_local_quaternion_xyzw": matrix_to_quat(fk_local),
            "translation_error_mm": local_translation,
            "rotation_error_deg": local_rotation,
        }
        local_records.append(base_record)

        query_record = {"tf_header_stamp": local["header_stamp"], "requested_time_ns": stamp_ns(local["header_stamp"]), "lookups": {}}
        lookup_transforms = {}
        for label, target, source in (("base_to_wrist2", "base_link", "wrist2_link"), ("base_to_wrist3", "base_link", "wrist3_link"), ("base_to_spray_tcp", "base_link", "spray_tcp_link")):
            try:
                value = buffer.lookup_transform(target, source, stamp_time(local["header_stamp"]))
                lookup_transforms[label] = value
                query_record["lookups"][label] = "successful"
            except ConnectivityException as exc:
                lookup_exceptions["connectivity_exception"] += 1
                query_record["lookups"][label] = f"connectivity_exception: {exc}"
            except ExtrapolationException as exc:
                lookup_exceptions["extrapolation_exception"] += 1
                query_record["lookups"][label] = f"extrapolation_exception: {exc}"
            except LookupException as exc:
                lookup_exceptions["lookup_exception"] += 1
                query_record["lookups"][label] = f"lookup_exception: {exc}"
            except TransformException as exc:
                lookup_exceptions["lookup_exception"] += 1
                query_record["lookups"][label] = f"lookup_exception: {exc}"
        lookup_samples.append(query_record)
        if "base_to_wrist3" in lookup_transforms:
            tf_wrist3 = matrix_from_transform(lookup_transforms["base_to_wrist3"])
            translation, rotation = transform_error(tf_wrist3, fk_wrist3)
            wrist3_records.append({**base_record, "tf_global_wrist3_quaternion_xyzw": [float(lookup_transforms["base_to_wrist3"].transform.rotation.x), float(lookup_transforms["base_to_wrist3"].transform.rotation.y), float(lookup_transforms["base_to_wrist3"].transform.rotation.z), float(lookup_transforms["base_to_wrist3"].transform.rotation.w)], "fk_global_wrist3_quaternion_xyzw": matrix_to_quat(fk_wrist3), "translation_error_mm": translation, "rotation_error_deg": rotation})
        if "base_to_spray_tcp" in lookup_transforms:
            tf_tcp = matrix_from_transform(lookup_transforms["base_to_spray_tcp"])
            translation, rotation = transform_error(tf_tcp, fk_tcp)
            tcp_records.append({**base_record, "tf_global_tcp_quaternion_xyzw": [float(lookup_transforms["base_to_spray_tcp"].transform.rotation.x), float(lookup_transforms["base_to_spray_tcp"].transform.rotation.y), float(lookup_transforms["base_to_spray_tcp"].transform.rotation.z), float(lookup_transforms["base_to_spray_tcp"].transform.rotation.w)], "fk_global_tcp_quaternion_xyzw": matrix_to_quat(fk_tcp), "translation_error_mm": translation, "rotation_error_deg": rotation})

    lookup_validation = {
        "schema_version": "stage28sr-tf2-lookup-validation-v1",
        "buffer_backend": "tf2_ros.Buffer",
        "lookup_semantics": "lookupTransform(target_frame='base_link', source_frame=..., time=T)",
        "samples_requested": len(local_rows),
        "successful": sum(1 for row in lookup_samples if all(value == "successful" for value in row["lookups"].values())),
        "lookup_exception": lookup_exceptions["lookup_exception"],
        "connectivity_exception": lookup_exceptions["connectivity_exception"],
        "extrapolation_exception": lookup_exceptions["extrapolation_exception"],
        "required_queries": {"base_link -> wrist2_link": True, "base_link -> wrist3_link": True, "base_link -> spray_tcp_link": True},
        "sample_details": lookup_samples[:20],
        "passed": bool(local_rows and len(lookup_samples) == len(local_rows) and lookup_exceptions == {"lookup_exception": 0, "connectivity_exception": 0, "extrapolation_exception": 0}),
    }
    write(output / "stage28sr_tf2_lookup_validation.json", lookup_validation)

    local_summary = {"schema_version": "stage28sr-local-j6-tf-fk-v1", "pairing": "TF wrist2_link->wrist3_link header_stamp exact /joint_states", "statistics": stats(local_records), "records_for_audit": len(local_records), "passed": bool(local_records and stats(local_records)["passed"] and not dynamic_unmatched)}
    wrist3_stats = stats(wrist3_records)
    wrist3_summary = {"schema_version": "stage28sr-global-wrist3-tf-fk-v1", "pairing": "tf2 base_link->wrist3_link and MoveIt FK at exact TF header_stamp /joint_states", "statistics": wrist3_stats, "records_for_audit": len(wrist3_records), "passed": bool(wrist3_records and wrist3_stats["passed"] and lookup_validation["passed"])}
    tcp_stats = stats(tcp_records)
    tcp_summary = {"schema_version": "stage28sr-global-spray-tcp-tf-fk-v1", "pairing": "tf2 base_link->spray_tcp_link and MoveIt FK at exact TF header_stamp /joint_states", "statistics": tcp_stats, "records_for_audit": len(tcp_records), "passed": bool(tcp_records and tcp_stats["passed"] and lookup_validation["passed"]), "complete_se3_comparison": True}
    write(output / "stage28sr_local_j6_tf_fk.json", local_summary)
    write(output / "stage28sr_global_wrist3_tf_fk.json", wrist3_summary)
    write(output / "stage28sr_global_spray_tcp_tf_fk.json", tcp_summary)

    fresh_max = local_summary["statistics"].get("max_rotation_error_deg")
    if fresh_max is not None and fresh_max <= ROTATION_GATE_DEG:
        root_cause = "A"
        interpretation = "Fresh strict same-header-stamp capture removes the old corrected residual; it was a recorder/verifier capture-time pairing artifact."
    elif dynamic_unmatched:
        root_cause = "B"
        interpretation = "Residual remains with incomplete exact TF-to-JointState state correspondence; timing semantics are not proven."
    elif fresh_max is not None:
        root_cause = "C"
        interpretation = "Residual remains despite exact header-stamp correspondence; this is a model/runtime mismatch."
    else:
        root_cause = "D"
        interpretation = "Fresh formal residual is unavailable or has an unclassified failure mode."
    worst = sorted(local_records, key=lambda row: float(row["rotation_error_deg"]), reverse=True)[:20]
    residual = {
        "schema_version": "stage28sr-residual-6p9277-root-cause-v1",
        "old_artifact_corrected_residual": {"max_deg": 6.9277, "source": "pre-remediation Q&A supplied formal evidence"},
        "classification": root_cause,
        "interpretation": interpretation,
        "fresh_local_j6_max_rotation_error_deg": fresh_max,
        "fresh_exact_state_correspondence": timestamp_mapping["exact_state_correspondence"],
        "fresh_tf2_lookup_passed": lookup_validation["passed"],
        "gate_threshold_deg": ROTATION_GATE_DEG,
        "worst_20_samples_if_over_gate": worst if fresh_max is not None and fresh_max > ROTATION_GATE_DEG else [],
        "fail_closed": bool(fresh_max is None or fresh_max > ROTATION_GATE_DEG or dynamic_unmatched or not lookup_validation["passed"]),
    }
    write(output / "stage28sr_residual_6p9277_root_cause.json", residual)
    write(output / "stage28sr_tf_fk_validator_runtime.json", {"schema_version": "stage28sr-tf-fk-validator-runtime-v1", "formal_tf_rows": len(tf_rows), "formal_joint_state_rows": len(joint_rows), "formal_controller_state_rows": len(controller_rows), "old_nearest_capture_pairing_used": False, "old_zip_sample_raw_pairing_used": False})
    print(json.dumps({"local": local_summary["statistics"], "wrist3": wrist3_summary["statistics"], "tcp": tcp_summary["statistics"], "root_cause": root_cause, "timestamp": timestamp_mapping}, ensure_ascii=False))
    return 0 if local_summary["passed"] and wrist3_summary["passed"] and tcp_summary["passed"] and timestamp_mapping["status"] == "passed" else 2


if __name__ == "__main__":
    rclpy.init()
    try:
        raise SystemExit(main())
    finally:
        rclpy.shutdown()
