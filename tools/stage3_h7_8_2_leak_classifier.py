#!/usr/bin/env python3
"""Deterministically parse and compare LeakSanitizer records.

The parser deliberately preserves complete symbolized call paths while removing
run-specific prefixes, PIDs, addresses, BuildIds, timestamps, and transient
build roots.  It does not suppress, ignore, or reclassify any LSan record.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


RECORD_RE = re.compile(
    r"^(Direct|Indirect) leak of (\d+) byte\(s\) in (\d+) object\(s\) allocated from:$"
)
FRAME_RE = re.compile(r"^#(\d+)\s+(.*)$")
SUMMARY_RE = re.compile(
    r"SUMMARY: AddressSanitizer: (\d+) byte\(s\) leaked in (\d+) allocation\(s\)\."
)
PREFIX_RE = re.compile(r"^\[[^\]]+\]\s*")


def strip_launch_prefix(line: str) -> str:
    line = line.rstrip("\r\n")
    while True:
        updated = PREFIX_RE.sub("", line, count=1)
        if updated == line:
            return line.strip()
        line = updated


def normalize_frame(frame: str) -> str:
    value = frame.strip()
    value = re.sub(r"^0x[0-9a-fA-F]+\s+", "<ADDR> ", value)
    value = re.sub(r"\s+\(BuildId: [0-9a-fA-F]+\)", "", value)
    value = re.sub(r"==\d+==", "==<PID>==", value)
    value = re.sub(r"0x[0-9a-fA-F]{6,}", "<ADDR>", value)
    value = re.sub(
        r"/home/robot/stage3_h7_8_[^/\s]+_sanitizers/(?:asan|ubsan)",
        "<SANITIZER_ROOT>",
        value,
    )
    value = re.sub(
        r"/home/robot/stage3_h7_8_1_sanitizers/(?:asan|ubsan)",
        "<SANITIZER_ROOT>",
        value,
    )
    value = value.replace("/mnt/d/robotfucker", "<REPO>")
    value = re.sub(r"/tmp/(?:tmp\.[^/\s]+|[^/\s]*stage3_h7_8[^/\s]*)", "<TMP>", value)
    value = re.sub(r"202\d-\d\d-\d\d-\d\d-\d\d-\d\d-[^/\s]+", "<RUNSTAMP>", value)
    return re.sub(r"\s+", " ", value).strip()


def first_matching(frames: list[str], predicates: Iterable[str]) -> str | None:
    lowered = tuple(item.lower() for item in predicates)
    for frame in frames:
        text = frame.lower()
        if any(token in text for token in lowered):
            return frame
    return None


def parse_log(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = [strip_launch_prefix(line) for line in text.splitlines()]
    records: list[dict[str, Any]] = []
    index = 0
    while index < len(lines):
        match = RECORD_RE.match(lines[index])
        if not match:
            index += 1
            continue
        kind, byte_text, object_text = match.groups()
        index += 1
        frames: list[str] = []
        while index < len(lines):
            frame_match = FRAME_RE.match(lines[index])
            if frame_match:
                frames.append(normalize_frame(frame_match.group(2)))
                index += 1
                continue
            if not lines[index]:
                break
            index += 1
        normalized_stack = "\n".join(frames)
        stack_hash = hashlib.sha256(normalized_stack.encode("utf-8")).hexdigest()
        owner_end = next(
            (position + 1 for position, frame in enumerate(frames) if "rclcpp_action::ClientBase::ClientBase" in frame),
            len(frames),
        )
        owner_frames = frames[:owner_end]
        owner_hash = hashlib.sha256("\n".join(owner_frames).encode("utf-8")).hexdigest()
        component_frames = [
            frame
            for frame in owner_frames
            if any(
                token in frame
                for token in (
                    "asan_malloc_linux.cpp",
                    "asan_new_delete.cpp",
                    "rcutils_strndup",
                    "rcutils_load_shared_library",
                    "rcpputils::SharedLibrary::SharedLibrary",
                    "librosidl_typesupport_cpp.so",
                    "rcl_client_init",
                    "rcl_subscription_init",
                    "rcl_action_client_init",
                    "rclcpp_action::ClientBase::ClientBase",
                )
            )
        ]
        component_hash = hashlib.sha256("\n".join(component_frames).encode("utf-8")).hexdigest()
        non_asan = next(
            (
                frame
                for frame in frames
                if not any(
                    token in frame.lower()
                    for token in ("libasan", "asan_", "sanitizer/asan", "asan_interceptors")
                )
            ),
            None,
        )
        records.append(
            {
                "leak_kind": kind.lower(),
                "bytes": int(byte_text),
                "objects": int(object_text),
                "top_allocator": frames[0] if frames else None,
                "first_symbolized_non_asan_frame": non_asan,
                "first_python_frame": first_matching(
                    frames,
                    ("/python3", "pyobject_", "pyinit_", "_pyeval", "pyimport_", "pyunicode_"),
                ),
                "first_pybind11_frame": first_matching(frames, ("pybind11",)),
                "first_ros_frame": first_matching(
                    frames,
                    ("rclcpp", "rclpy", "rcl_", "rmw_", "rosidl", "fastrtps", "fastdds"),
                ),
                "first_moveit_frame": first_matching(
                    frames,
                    ("moveit", "trajectory_execution_manager", "planning_scene_monitor"),
                ),
                "first_local_patched_source_frame": first_matching(
                    frames,
                    (
                        "moveit_py/src/moveit/moveit_ros/moveit_cpp/moveit_cpp.cpp",
                        "trajectory_execution_manager/src/trajectory_execution_manager.cpp",
                    ),
                ),
                "normalized_stack_hash": stack_hash,
                "allocation_owner_hash": owner_hash,
                "allocation_owner_frames": owner_frames,
                "allocation_component_hash": component_hash,
                "allocation_component_frames": component_frames,
                "normalized_frames": frames,
            }
        )
    summaries = [SUMMARY_RE.search(line) for line in lines]
    summary = next((match for match in reversed(summaries) if match), None)
    totals = {
        "bytes": int(summary.group(1)) if summary else sum(record["bytes"] for record in records),
        "allocations": int(summary.group(2)) if summary else sum(record["objects"] for record in records),
        "direct_bytes": sum(record["bytes"] for record in records if record["leak_kind"] == "direct"),
        "direct_allocations": sum(
            record["objects"] for record in records if record["leak_kind"] == "direct"
        ),
        "indirect_bytes": sum(record["bytes"] for record in records if record["leak_kind"] == "indirect"),
        "indirect_allocations": sum(
            record["objects"] for record in records if record["leak_kind"] == "indirect"
        ),
        "record_count": len(records),
    }
    return {
        "schema_version": "stage3-h7-8-2-leak-log-v1",
        "raw_log": str(path.resolve()),
        "raw_log_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "raw_log_size_bytes": path.stat().st_size,
        "leak_sanitizer_reported": "ERROR: LeakSanitizer" in text,
        "fatal_memory_markers": {
            "heap_use_after_free": text.count("heap-use-after-free"),
            "stack_use_after_free": text.count("stack-use-after-free"),
            "double_free": text.count("double-free"),
            "invalid_free": text.count("invalid-free"),
            "buffer_overflow": text.count("buffer-overflow"),
            "segv": text.count("AddressSanitizer:DEADLYSIGNAL") + text.count("SIGSEGV"),
        },
        "totals": totals,
        "records": records,
    }


def aggregate(parsed: list[dict[str, Any]]) -> dict[str, Any]:
    signatures: dict[str, dict[str, Any]] = {}
    occurrence: dict[str, list[dict[str, int]]] = defaultdict(list)
    for run_index, run in enumerate(parsed):
        per_run: dict[str, dict[str, int]] = defaultdict(lambda: {"bytes": 0, "objects": 0, "records": 0})
        for record in run["records"]:
            signature = record["normalized_stack_hash"]
            signatures.setdefault(signature, record)
            per_run[signature]["bytes"] += record["bytes"]
            per_run[signature]["objects"] += record["objects"]
            per_run[signature]["records"] += 1
        for signature, counts in per_run.items():
            occurrence[signature].append({"run_index": run_index, **counts})
    return {
        "run_count": len(parsed),
        "unique_signature_count": len(signatures),
        "signatures": [
            {**signatures[signature], "occurrences": occurrence[signature]}
            for signature in sorted(signatures)
        ],
    }


def compare(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    left_map = {item["normalized_stack_hash"]: item for item in left["signatures"]}
    right_map = {item["normalized_stack_hash"]: item for item in right["signatures"]}
    left_set, right_set = set(left_map), set(right_map)
    return {
        "left_unique": sorted(left_set - right_set),
        "right_unique": sorted(right_set - left_set),
        "shared": sorted(left_set & right_set),
        "counts": {
            "left_unique": len(left_set - right_set),
            "right_unique": len(right_set - left_set),
            "shared": len(left_set & right_set),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("logs", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--compare-left", type=Path)
    parser.add_argument("--compare-right", type=Path)
    args = parser.parse_args()
    parsed = [parse_log(path) for path in args.logs]
    result: dict[str, Any] = {
        "schema_version": "stage3-h7-8-2-leak-signature-catalog-v1",
        "normalization": {
            "removed": [
                "launch prefixes",
                "PIDs",
                "instruction/heap addresses",
                "ELF BuildIds",
                "timestamped ROS log directories",
                "temporary sanitizer/build roots",
            ],
            "retained": "complete normalized symbolized stack",
        },
        "runs": parsed,
        "aggregate": aggregate(parsed),
    }
    if args.compare_left and args.compare_right:
        left = json.loads(args.compare_left.read_text(encoding="utf-8"))["aggregate"]
        right = json.loads(args.compare_right.read_text(encoding="utf-8"))["aggregate"]
        result["comparison"] = compare(left, right)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
