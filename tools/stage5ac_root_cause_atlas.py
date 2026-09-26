"""Build a Stage 5A-C collision atlas from the closed Stage 5A-R evidence.

This tool is deliberately read-only with respect to D65 and Stage 5A-R.  It
materializes a compact, queryable atlas in a new exploration directory.  The
atlas keeps the native Bullet witness semantics explicit: the reported
``m_distance1`` is a runtime manifold value, not a calibrated physical
penetration or clearance measurement.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "outputs" / "STAGE5AC_EXPLORATION" / "phase1_root_cause"


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def finite_vector(value: Any, size: int) -> list[float] | None:
    try:
        values = [float(x) for x in value]
    except (TypeError, ValueError):
        return None
    if len(values) != size or not all(math.isfinite(x) for x in values):
        return None
    return values


def region_for_segment(segment: int | None, segment_count: int) -> str:
    if segment is None:
        return "unknown"
    # The authoritative 181-point open-arch path runs from the right riser,
    # over the arch, to the left riser.  Keep this as a coarse label and
    # retain the exact source segment beside it.
    fraction = float(segment) / max(segment_count - 1, 1)
    if fraction < 0.25:
        return "right_riser"
    if fraction > 0.75:
        return "left_riser"
    return "arch"


def closest_segment(point: np.ndarray, objects: list[dict[str, Any]]) -> tuple[int | None, float | None]:
    if point.shape != (3,) or not np.all(np.isfinite(point)) or not objects:
        return None, None
    centers = np.asarray([item["center_m"] for item in objects], dtype=float)
    distances = np.linalg.norm(centers - point[None, :], axis=1)
    index = int(np.argmin(distances))
    return int(objects[index].get("source_segment", index)), float(distances[index])


def runs(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    ordered = sorted(rows, key=lambda row: int(row["interval_index"]))
    result: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for row in ordered:
        value = row.get(key)
        index = int(row["interval_index"])
        if current is None or value != current["value"] or index != int(current["last_interval"]) + 1:
            if current is not None:
                result.append(current)
            current = {"value": value, "first_interval": index, "last_interval": index, "length_intervals": 1}
        else:
            current["last_interval"] = index
            current["length_intervals"] += 1
    if current is not None:
        result.append(current)
    return result


def build(trajectory_name: str, interval_path: Path, fk_path: Path, environment_path: Path, output: Path) -> dict[str, Any]:
    intervals = read_jsonl(interval_path)
    fk_rows = {int(row["state_index"]): row for row in read_jsonl(fk_path)}
    environment = load_json(environment_path)
    objects = list(environment.get("objects", []))
    expected = len(intervals)
    contacts: list[dict[str, Any]] = []
    collision_intervals: list[dict[str, Any]] = []
    pair_counts: Counter[str] = Counter()
    segment_counts: Counter[int] = Counter()
    cluster_counts: Counter[str] = Counter()
    depth_values: list[float] = []

    for interval in sorted(intervals, key=lambda row: int(row["interval_index"])):
        index = int(interval["interval_index"])
        q_start = finite_vector(interval.get("q_start"), 6)
        q_end = finite_vector(interval.get("q_end"), 6)
        interval_contacts = list(interval.get("continuous_contacts") or interval.get("first_contact") or [])
        if not interval_contacts and interval.get("continuous_collision"):
            interval_contacts = [{"pair": "unattributed", "body_name_1": None, "body_name_2": None}]
        if interval.get("continuous_collision"):
            collision_intervals.append(interval)
        for ordinal, contact in enumerate(interval_contacts):
            pair = str(contact.get("pair") or "unattributed")
            position = finite_vector(contact.get("pos"), 3)
            depth = contact.get("depth")
            try:
                depth_value = float(depth)
            except (TypeError, ValueError):
                depth_value = None
            segment, center_distance = closest_segment(np.asarray(position, dtype=float), objects) if position else (None, None)
            region = region_for_segment(segment, len(objects))
            cluster = "unknown"
            if position is not None:
                cluster_values = [round(float(x) / 0.02) * 0.02 for x in position]
                cluster = ",".join(f"{x:.2f}" for x in cluster_values)
            start_fk = fk_rows.get(index)
            end_fk = fk_rows.get(index + 1)
            row = {
                "trajectory_id": trajectory_name,
                "interval_index": index,
                "contact_ordinal": ordinal,
                "time_start_s": float(interval.get("time_start_s", 0.0)),
                "time_end_s": float(interval.get("time_end_s", 0.0)),
                "duration_s": float(interval.get("duration_s", 0.0)),
                "collision_method": str(interval.get("collision_method", "native_bullet_robot_world_ccd")),
                "ccd_api_called": bool(interval.get("ccd_api_called", False)),
                "collision_pair": pair,
                "collision_object": "horseshoe_collision_compound",
                "robot_link": str(contact.get("body_name_2") or "forearm_link") if str(contact.get("body_type_2", "")) == "ROBOT_LINK" else "unresolved",
                "body_name_1": contact.get("body_name_1"),
                "body_name_2": contact.get("body_name_2"),
                "contact_position_m": position,
                "contact_normal": finite_vector(contact.get("normal"), 3),
                "bullet_m_distance1_m": depth_value,
                "distance_semantics": "runtime_native_bullet_manifold_value_not_physical_clearance",
                "contact_point_semantics": "runtime_contact_pose_not_calibrated_physical_contact",
                "nearest_points_status": "invalid_or_uninitialized_not_used",
                "closest_manifest_segment": segment,
                "closest_manifest_segment_center_distance_m": center_distance,
                "tunnel_region": region,
                "q_start": q_start,
                "q_end": q_end,
                "tcp_fk_start": start_fk,
                "tcp_fk_end": end_fk,
                "spatial_cluster_20mm": cluster,
            }
            contacts.append(row)
            pair_counts[pair] += 1
            if segment is not None:
                segment_counts[segment] += 1
            cluster_counts[cluster] += 1
            if depth_value is not None and math.isfinite(depth_value):
                depth_values.append(depth_value)

    by_pair_interval = [{"interval_index": int(row["interval_index"]), "collision_pair": row["collision_pair"]} for row in contacts]
    pair_runs = runs(by_pair_interval, "collision_pair")
    region_interval_rows = [{"interval_index": int(row["interval_index"]), "tunnel_region": row["tunnel_region"]} for row in contacts]
    region_runs = runs(region_interval_rows, "tunnel_region")
    deepest = min(contacts, key=lambda row: float(row["bullet_m_distance1_m"])) if depth_values else None
    first = min(contacts, key=lambda row: int(row["interval_index"])) if contacts else None
    last = max(contacts, key=lambda row: int(row["interval_index"])) if contacts else None
    summary = {
        "schema_version": "stage5ac-collision-atlas-v1",
        "trajectory_id": trajectory_name,
        "source": {
            "interval_jsonl": str(interval_path.resolve()),
            "fk_trace_jsonl": str(fk_path.resolve()),
            "environment_json": str(environment_path.resolve()),
            "environment_frame": environment.get("frame_id"),
            "environment_object_count": len(objects),
        },
        "coverage": {
            "interval_rows": len(intervals),
            "collision_interval_rows": len(collision_intervals),
            "all_intervals_collision": bool(intervals) and len(collision_intervals) == len(intervals),
            "contact_rows": len(contacts),
            "fk_state_rows_available": len(fk_rows),
        },
        "classification": {
            "robot_world_domain": True,
            "self_collision_domain": False,
            "continuous_method": "native_bullet_robot_world_ccd",
            "exact_articulated_self_ccd": "not_available",
            "physical_clearance": None,
        },
        "pair_frequency": [{"collision_pair": pair, "contact_count": count} for pair, count in pair_counts.most_common()],
        "manifest_segment_frequency": [{"source_segment": segment, "contact_count": count} for segment, count in segment_counts.most_common()],
        "spatial_cluster_frequency": [{"cluster_20mm": cluster, "contact_count": count} for cluster, count in cluster_counts.most_common()],
        "depth_summary_runtime_only": {
            "min_m": min(depth_values) if depth_values else None,
            "max_m": max(depth_values) if depth_values else None,
            "negative_value_count": sum(value < 0.0 for value in depth_values),
            "physical_penetration_claim": False,
        },
        "first_collision": first,
        "deepest_runtime_witness": deepest,
        "last_collision": last,
        "consecutive_pair_runs": pair_runs,
        "consecutive_region_runs": region_runs,
        "old_summary_semantic_correction": {
            "incorrect_field": "stage26_native_summary.states_with_endpoint_self_collision",
            "observed_old_value": len(read_jsonl(interval_path.parent / "stage26_self_collision_states.jsonl")) if (interval_path.parent / "stage26_self_collision_states.jsonl").exists() else None,
            "correct_robot_world_endpoint_field": "not inferred from continuous rows; use separate self-only validator",
            "rule": "robot-world collision is never counted as self-collision",
        },
    }
    output.mkdir(parents=True, exist_ok=True)
    atlas_path = output / f"STAGE5AC_COLLISION_ATLAS_{trajectory_name}.jsonl"
    with atlas_path.open("w", encoding="utf-8") as handle:
        for row in contacts:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n")
    fields = ["trajectory_id", "interval_index", "time_start_s", "time_end_s", "collision_pair", "robot_link", "closest_manifest_segment", "tunnel_region", "bullet_m_distance1_m", "contact_position_m", "spatial_cluster_20mm"]
    with (output / f"STAGE5AC_COLLISION_ATLAS_{trajectory_name}.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in contacts:
            writer.writerow({field: json.dumps(row[field], ensure_ascii=False, separators=(",", ":")) if isinstance(row[field], (list, dict)) else row[field] for field in fields})
    write_json(output / f"STAGE5AC_COLLISION_ATLAS_{trajectory_name}_SUMMARY.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory-name", required=True)
    parser.add_argument("--interval-jsonl", type=Path, required=True)
    parser.add_argument("--fk-jsonl", type=Path, required=True)
    parser.add_argument("--environment-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    summary = build(args.trajectory_name, args.interval_jsonl.resolve(), args.fk_jsonl.resolve(), args.environment_json.resolve(), args.output_dir.resolve())
    print(json.dumps({"trajectory_id": summary["trajectory_id"], "collision_intervals": summary["coverage"]["collision_interval_rows"], "contacts": summary["coverage"]["contact_rows"], "dominant_pair": summary["pair_frequency"][0] if summary["pair_frequency"] else None}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
