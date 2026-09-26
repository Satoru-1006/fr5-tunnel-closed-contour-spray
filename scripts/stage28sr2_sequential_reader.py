"""Run the installed Jazzy rosbag2_py SequentialReader against one final bag.

This file is intentionally executable inside the WSL ROS environment.  The
host-side orchestrator invokes it only after rosbag2 has been stopped and the
native Linux bag has been finalized.  It never treats metadata alone as proof
of readability: the reader opens the storage plugin, enumerates topics, reads
until EOF, and reconciles every count with metadata.yaml.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any


REQUIRED_TOPICS = (
    "/joint_states",
    "/fairino5_controller/controller_state",
    "/tf",
    "/tf_static",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def tree_manifest(root: Path) -> dict[str, Any]:
    """Return a deterministic byte manifest for a finalized bag directory."""

    root = root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    files: list[dict[str, Any]] = []
    for path in sorted((candidate for candidate in root.rglob("*") if candidate.is_file()), key=lambda p: p.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix()
        files.append({"path": relative, "size": path.stat().st_size, "sha256": _sha256(path)})
    return {"root": str(root), "file_count": len(files), "files": files}


def _metadata_counts(metadata_path: Path) -> tuple[dict[str, int], int]:
    try:
        import yaml  # type: ignore
    except ImportError as error:  # pragma: no cover - ROS image dependency
        raise RuntimeError("PyYAML is required to reconcile rosbag2 metadata counts") from error
    payload = yaml.safe_load(metadata_path.read_text(encoding="utf-8"))
    info = payload.get("rosbag2_bagfile_information") if isinstance(payload, dict) else None
    if not isinstance(info, dict):
        raise ValueError("metadata.yaml has no rosbag2_bagfile_information")
    entries = info.get("topics_with_message_count")
    if not isinstance(entries, list):
        raise ValueError("metadata.yaml has no topics_with_message_count")
    counts: dict[str, int] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("invalid metadata topic entry")
        topic_metadata = entry.get("topic_metadata")
        if not isinstance(topic_metadata, dict) or not isinstance(topic_metadata.get("name"), str):
            raise ValueError("metadata topic entry has no name")
        count = entry.get("message_count")
        if not isinstance(count, int) or isinstance(count, bool) or count < 0:
            raise ValueError(f"invalid metadata count for {topic_metadata['name']}")
        counts[topic_metadata["name"]] = count
    total = info.get("message_count")
    if not isinstance(total, int) or isinstance(total, bool) or total < 0:
        raise ValueError("metadata.yaml has no valid message_count")
    return counts, total


def read_bag(bag_path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "schema_version": "stage28sr2-sequential-reader-v1",
        "bag_path": str(bag_path.resolve()),
        "metadata_path": str((bag_path / "metadata.yaml").resolve()),
        "reader_api": "rosbag2_py.SequentialReader",
        "open_ok": False,
        "topics_and_types": [],
        "reader_counts": {},
        "metadata_counts": {},
        "reader_total_count": None,
        "metadata_total_count": None,
        "read_to_eof": False,
        "required_topics_present": False,
        "required_topic_counts_positive": False,
        "counts_match_metadata": False,
        "total_count_matches_metadata": False,
        "errors": [],
        "passed": False,
    }
    try:
        metadata_counts, metadata_total = _metadata_counts(bag_path / "metadata.yaml")
        result["metadata_counts"] = metadata_counts
        result["metadata_total_count"] = metadata_total
        from rosbag2_py import ConverterOptions, SequentialReader, StorageOptions  # type: ignore

        reader = SequentialReader()
        reader.open(StorageOptions(uri=str(bag_path), storage_id="mcap"), ConverterOptions("", ""))
        result["open_ok"] = True
        topic_types = reader.get_all_topics_and_types()
        result["topics_and_types"] = [
            {"name": str(getattr(item, "name", "")), "type": str(getattr(item, "type", ""))}
            for item in topic_types
        ]
        reader_counts: dict[str, int] = {}
        while reader.has_next():
            topic, _serialized, _timestamp = reader.read_next()
            topic_name = str(topic)
            reader_counts[topic_name] = reader_counts.get(topic_name, 0) + 1
        result["reader_counts"] = reader_counts
        result["reader_total_count"] = sum(reader_counts.values())
        result["read_to_eof"] = True
        result["required_topics_present"] = all(topic in reader_counts for topic in REQUIRED_TOPICS)
        result["required_topic_counts_positive"] = all(reader_counts.get(topic, 0) > 0 for topic in REQUIRED_TOPICS)
        result["counts_match_metadata"] = reader_counts == metadata_counts
        result["total_count_matches_metadata"] = result["reader_total_count"] == metadata_total
        result["passed"] = bool(
            result["open_ok"]
            and result["read_to_eof"]
            and result["required_topics_present"]
            and result["required_topic_counts_positive"]
            and result["counts_match_metadata"]
            and result["total_count_matches_metadata"]
        )
        if not result["passed"]:
            result["errors"].append("sequential_reader_reconciliation_failed")
    except Exception as error:  # fail closed and preserve exact exception
        result["errors"].append(f"{type(error).__name__}:{error}")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bag", type=Path)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    try:
        if args.manifest is not None:
            payload = tree_manifest(args.manifest)
        elif args.bag is not None:
            payload = read_bag(args.bag)
        else:
            raise SystemExit("one of --bag or --manifest is required")
    except Exception as error:
        payload = {"passed": False, "errors": [f"{type(error).__name__}:{error}"]}
    print("STAGE28SR2_JSON_START")
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    print("STAGE28SR2_JSON_END")
    return 0 if payload.get("passed") is True or args.manifest is not None else 2


if __name__ == "__main__":
    raise SystemExit(main())
