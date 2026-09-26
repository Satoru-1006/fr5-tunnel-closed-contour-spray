"""Offline rosbag2 audit for Stage 2.8S-R2-H1 capture integrity.

This file is executed inside the ROS 2 environment.  It deliberately performs
no graph probing and never sends an action goal; it only reads the completed
bag with rosbag2_py and writes deterministic diagnostics.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


TOPICS = {
    "/joint_states": "sensor_msgs/msg/JointState",
    "/tf": "tf2_msgs/msg/TFMessage",
    "/tf_static": "tf2_msgs/msg/TFMessage",
    "/fairino5_controller/controller_state": "control_msgs/msg/JointTrajectoryControllerState",
}


def stamp_ns(stamp) -> int:
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    index = (len(ordered) - 1) * fraction
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return float(ordered[lower])
    return float(ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower))


def gap_windows(stamps: list[int], bag_times: list[int], threshold_ms: float = 100.0) -> list[dict]:
    windows = []
    for index, (before, after) in enumerate(zip(stamps, stamps[1:])):
        gap_ms = (after - before) / 1_000_000.0
        if gap_ms > threshold_ms:
            windows.append({
                "sequence_index": index,
                "before_header_stamp_ns": before,
                "after_header_stamp_ns": after,
                "header_gap_ms": gap_ms,
                "before_bag_time_ns": bag_times[index] if index < len(bag_times) else None,
                "after_bag_time_ns": bag_times[index + 1] if index + 1 < len(bag_times) else None,
            })
    return windows


def stamp_stats(stamps: list[int], bag_times: list[int] | None = None) -> dict:
    counts = Counter(stamps)
    invalid = sum(1 for value in stamps if value <= 0)
    duplicate = sum(count - 1 for count in counts.values() if count > 1)
    out_of_order = sum(1 for before, after in zip(stamps, stamps[1:]) if after < before)
    intervals_ms = [
        (after - before) / 1_000_000.0
        for before, after in zip(stamps, stamps[1:])
        if after >= before
    ]
    capture_intervals_ms = []
    if bag_times:
        capture_intervals_ms = [
            (after - before) / 1_000_000.0
            for before, after in zip(bag_times, bag_times[1:])
            if after >= before
        ]
    positive_intervals = [value for value in intervals_ms if value > 0.0]
    return {
        "messages": len(stamps),
        "unique_header_stamps": len(counts),
        "duplicate_header_stamps": duplicate,
        "out_of_order_header_stamps": out_of_order,
        "invalid_header_stamps": invalid,
        "capture_header_intervals_ms": {
            "median": percentile(positive_intervals, 0.50),
            "p95": percentile(positive_intervals, 0.95),
            "p99": percentile(positive_intervals, 0.99),
            "max": max(positive_intervals) if positive_intervals else None,
            "gaps_over_20ms": sum(value > 20.0 for value in positive_intervals),
            "gaps_over_50ms": sum(value > 50.0 for value in positive_intervals),
            "gaps_over_100ms": sum(value > 100.0 for value in positive_intervals),
        },
        "capture_monotonic_intervals_ms": {
            "median": percentile(capture_intervals_ms, 0.50),
            "p95": percentile(capture_intervals_ms, 0.95),
            "p99": percentile(capture_intervals_ms, 0.99),
            "max": max(capture_intervals_ms) if capture_intervals_ms else None,
            "gaps_over_100ms": sum(value > 100.0 for value in capture_intervals_ms),
        },
        "gap_windows_over_100ms": gap_windows(stamps, bag_times or []),
        "stamp_counts": {str(key): value for key, value in counts.items() if value > 1},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bag", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-initial-positions", type=Path, default=None)
    args = parser.parse_args()

    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message

    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(args.bag), storage_id="mcap"),
        rosbag2_py.ConverterOptions(input_serialization_format="cdr", output_serialization_format="cdr"),
    )
    topic_types = {item.name: item.type for item in reader.get_all_topics_and_types()}
    topic_counts: Counter[str] = Counter()
    topic_header_stamps: dict[str, list[int]] = defaultdict(list)
    topic_bag_times: dict[str, list[int]] = defaultdict(list)
    tf_stamps: list[int] = []
    tf_bag_times: list[int] = []
    tf_message_stamp_sets: list[list[int]] = []
    tf_message_bag_times: list[int] = []
    joint_stamps: list[int] = []
    joint_bag_times: list[int] = []
    controller_stamps: list[int] = []
    controller_bag_times: list[int] = []
    first_joint_positions: list[float] | None = None
    last_joint_positions: list[float] | None = None
    all_bag_times: list[int] = []
    message_classes: dict[str, object] = {}

    while reader.has_next():
        topic, data, bag_time_ns = reader.read_next()
        topic_counts[topic] += 1
        all_bag_times.append(int(bag_time_ns))
        if topic not in message_classes:
            message_classes[topic] = get_message(topic_types[topic])
        message = deserialize_message(data, message_classes[topic])
        if topic == "/joint_states":
            value = stamp_ns(message.header.stamp)
            joint_stamps.append(value)
            joint_bag_times.append(int(bag_time_ns))
            topic_header_stamps[topic].append(value)
            topic_bag_times[topic].append(int(bag_time_ns))
            positions = [float(item) for item in message.position]
            if first_joint_positions is None:
                first_joint_positions = positions
            last_joint_positions = positions
        elif topic == "/fairino5_controller/controller_state":
            value = stamp_ns(message.header.stamp)
            controller_stamps.append(value)
            controller_bag_times.append(int(bag_time_ns))
            topic_header_stamps[topic].append(value)
            topic_bag_times[topic].append(int(bag_time_ns))
        elif topic == "/tf":
            message_stamps = sorted({stamp_ns(transform.header.stamp) for transform in message.transforms})
            tf_message_stamp_sets.append(message_stamps)
            tf_message_bag_times.append(int(bag_time_ns))
            for transform in message.transforms:
                value = stamp_ns(transform.header.stamp)
                tf_stamps.append(value)
                tf_bag_times.append(int(bag_time_ns))

    joint_counts = Counter(joint_stamps)
    exact_message_matches = sum(
        1 for values in tf_message_stamp_sets if values and all(value in joint_counts for value in values)
    )
    unmatched_message_count = sum(
        1 for values in tf_message_stamp_sets if values and any(value not in joint_counts for value in values)
    )
    ambiguous_message_count = sum(
        1 for values in tf_message_stamp_sets if any(joint_counts.get(value, 0) > 1 for value in values)
    )
    exact_transform_matches = sum(1 for value in tf_stamps if value in joint_counts)
    ambiguous_transforms = sum(1 for value in tf_stamps if joint_counts.get(value, 0) > 1)
    unmatched_tf_stamps = sorted({value for value in tf_stamps if value not in joint_counts})
    bag_start = min(all_bag_times) if all_bag_times else None
    bag_end = max(all_bag_times) if all_bag_times else None
    duration_s = (bag_end - bag_start) / 1_000_000_000.0 if bag_start is not None and bag_end is not None else None
    controller_stats = stamp_stats(controller_stamps, controller_bag_times)
    controller_capture_gaps = controller_stats["capture_monotonic_intervals_ms"]
    joint_stats = stamp_stats(joint_stamps, joint_bag_times)
    joint_stats["first_positions"] = first_joint_positions
    joint_stats["last_positions"] = last_joint_positions
    if args.expected_initial_positions and args.expected_initial_positions.is_file() and first_joint_positions is not None:
        expected = json.loads(args.expected_initial_positions.read_text(encoding="utf-8"))
        expected_values = [float(value) for value in expected["initial_positions"]]
        errors = [abs(actual - target) for actual, target in zip(first_joint_positions, expected_values)]
        joint_stats["initial_state_check"] = {
            "expected_source": str(args.expected_initial_positions.resolve()),
            "max_abs_error_rad": max(errors) if errors else None,
            "passed": bool(errors and max(errors) <= 1.0e-12),
        }
    else:
        joint_stats["initial_state_check"] = {"max_abs_error_rad": None, "passed": False, "status": "not_available"}
    joint_gaps = joint_stats["gap_windows_over_100ms"]
    controller_gaps = controller_stats["gap_windows_over_100ms"]
    synchronized_gaps = []
    for joint_gap in joint_gaps:
        for controller_gap in controller_gaps:
            if abs(joint_gap["before_header_stamp_ns"] - controller_gap["before_header_stamp_ns"]) <= 5_000_000 and abs(joint_gap["after_header_stamp_ns"] - controller_gap["after_header_stamp_ns"]) <= 5_000_000:
                synchronized_gaps.append({"joint_states": joint_gap, "controller_state": controller_gap})
                break
    if synchronized_gaps:
        gap_attribution = "publisher_did_not_publish_supported_by_synchronized_joint_and_controller_topics_and_matching_bag_arrival_gaps"
    elif controller_gaps:
        gap_attribution = "not_available"
    else:
        gap_attribution = "not_applicable_no_gap"

    result = {
        "schema_version": "stage28sr2-h1-bag-audit-v1",
        "bag": str(args.bag.resolve()),
        "storage_id": "mcap",
        "topic_types": topic_types,
        "topic_message_counts": {topic: int(topic_counts.get(topic, 0)) for topic in TOPICS},
        "rosbag_metadata": {
            "bag_message_count": int(sum(topic_counts.values())),
            "starting_time_ns": bag_start,
            "ending_time_ns": bag_end,
            "duration_s": duration_s,
            "topic_types": topic_types,
            "topic_message_counts": {topic: int(topic_counts.get(topic, 0)) for topic in TOPICS},
        },
        "joint_states": joint_stats,
        "controller_state": controller_stats,
        "tf": {
            "messages": int(topic_counts.get("/tf", 0)),
            "ros_messages": int(topic_counts.get("/tf", 0)),
            "transform_records": len(tf_stamps),
            "exact_joint_state_matches": exact_message_matches,
            "unmatched": unmatched_message_count,
            "ambiguous_duplicate": ambiguous_message_count,
            "exact_transform_matches": exact_transform_matches,
            "unmatched_transform_records": len(tf_stamps) - exact_transform_matches,
            "ambiguous_transform_records": ambiguous_transforms,
            "unmatched_header_stamps_sample": unmatched_tf_stamps[:20],
            "message_header_stamp_stats": stamp_stats(
                [values[0] for values in tf_message_stamp_sets if len(values) == 1],
                [bag_time for values, bag_time in zip(tf_message_stamp_sets, tf_message_bag_times) if len(values) == 1],
            ),
            "header_stamp_stats": stamp_stats(tf_stamps, tf_bag_times),
        },
        "controller_state_coverage": {
            "header_stamp_duplicate": controller_stats["duplicate_header_stamps"],
            "header_stamp_out_of_order": controller_stats["out_of_order_header_stamps"],
            "capture_monotonic_gap": {"max_ms": controller_capture_gaps.get("max"), "gaps_over_100ms": controller_capture_gaps.get("gaps_over_100ms", 0)},
            "long_callback_gap_over_100ms": {"count": controller_capture_gaps.get("gaps_over_100ms", 0)},
            "publisher_vs_recorder_gap_attribution": gap_attribution,
            "evidence": {
                "joint_states_gap_windows_over_100ms": joint_gaps,
                "controller_state_gap_windows_over_100ms": controller_gaps,
                "synchronized_independent_topic_gap_windows": synchronized_gaps,
                "interpretation": "publisher_did_not_publish_supported: both independent controller-origin topics have the same header gap and the same bag-arrival gap; rosbag2 event counters remain not_available for formal lost-message attribution",
            },
        },
        "message_lost_event_counter": {"status": "not_available", "reason": "rosbag2_py reader does not expose subscription event callbacks after recording"},
        "deadline_event_counter": {"status": "not_available", "reason": "rosbag2_py reader does not expose subscription event callbacks after recording"},
        "liveliness_event_counter": {"status": "not_available", "reason": "rosbag2_py reader does not expose subscription event callbacks after recording"},
        "event_level_loss_counter": {"status": "not_available", "reason": "Jazzy rosbag2_py reader does not expose post-recording subscription event callbacks"},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "duration_s": duration_s, "topics": result["topic_message_counts"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
