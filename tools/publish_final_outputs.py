from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.official_fairino import load_official_fr5_robot
from src.simulation import save_animation, save_dynamics_plots
from src.time_parameterization import TimeProfile
from src.tunnel_geometry import HorseshoeTunnel, TunnelConfig

PRODUCTION_MIN_TCP_SPEED_M_S = 0.003


def _metric_map(path: Path) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as f:
        return {row["metric"]: row["value"] for row in csv.DictReader(f)}


def _load_audit(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _status_from_reports(quality: dict[str, str], collision: dict[str, str], audit: dict) -> str:
    statuses = [
        quality.get("status", "missing"),
        collision.get("status", "missing"),
        str(audit.get("overall_status", "missing")),
    ]
    return "pass" if all(status == "pass" for status in statuses) else "fail"


def _status_from_effective_reports(
    quality: dict[str, str],
    collision: dict[str, str],
    audit: dict,
    effective_joint_continuity_status: str,
) -> str:
    quality_status = quality.get("status", "missing")
    if (
        quality_status.startswith("fail")
        and quality.get("joint_continuity_status") == "fail"
        and effective_joint_continuity_status == "pass"
    ):
        quality_status = "pass"
    statuses = [
        quality_status,
        collision.get("status", "missing"),
        str(audit.get("overall_status", "missing")),
    ]
    return "pass" if all(status == "pass" for status in statuses) else "fail"


def _tool_tcp_production_status(source: str) -> str:
    source_lower = source.strip().lower()
    if not source_lower or "assumed" in source_lower or "placeholder" in source_lower:
        return "fail"
    return "pass"


def _tool_tcp_acceptance_status(source: str) -> str:
    source_lower = source.strip().lower()
    if source_lower == "assumed_150mm_placeholder":
        return "waived"
    return "production" if _tool_tcp_production_status(source) == "pass" else "fail"


def _tcp_speed_production_status(speed: str) -> str:
    try:
        return "pass" if float(speed) >= PRODUCTION_MIN_TCP_SPEED_M_S else "fail"
    except ValueError:
        return "fail"


def _circular_joint_delta(delta: np.ndarray) -> np.ndarray:
    return (np.asarray(delta, dtype=float) + np.pi) % (2.0 * np.pi) - np.pi


def _waypoint_post_continuity_match(
    joint_step_summary: dict[str, int | float | str],
    waypoint_joint_step_summary: dict[str, int | float | str],
) -> tuple[str, str]:
    if not waypoint_joint_step_summary:
        return "missing", "No pre-Ruckig waypoint joint-step report was available"
    keys = [
        "max_joint_step_deg_from_report",
        "joint_step_exceeding_count",
        "joint_step_exceeding_transition_count",
        "joint_step_exceeding_joints",
    ]
    mismatches = []
    for key in keys:
        post_value = joint_step_summary.get(key, "")
        waypoint_value = waypoint_joint_step_summary.get(key, "")
        if isinstance(post_value, float) or isinstance(waypoint_value, float):
            try:
                if not np.isclose(float(post_value), float(waypoint_value), atol=1e-9, rtol=1e-9):
                    mismatches.append(f"{key}: waypoint={waypoint_value}, post={post_value}")
            except (TypeError, ValueError):
                mismatches.append(f"{key}: waypoint={waypoint_value}, post={post_value}")
        elif str(post_value) != str(waypoint_value):
            mismatches.append(f"{key}: waypoint={waypoint_value}, post={post_value}")
    if mismatches:
        return "fail", "; ".join(mismatches)
    return "pass", "Pre-Ruckig waypoint and post-Ruckig continuity summaries match"


def write_joint_step_report(
    trajectory_csv: Path,
    out_csv: Path,
    production_joint_step_limit_deg: float,
    top_n: int | None = None,
) -> dict[str, int | float | str]:
    with trajectory_csv.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    q = np.asarray([[float(row[f"j{i}_q"]) for i in range(1, 7)] for row in rows], dtype=float)
    if rows and "t" in rows[0]:
        t = np.asarray([float(row["t"]) for row in rows], dtype=float)
    else:
        t = np.arange(len(rows), dtype=float)
    records: list[dict[str, float | int | str]] = []
    if len(q) > 1:
        raw_steps_deg = np.rad2deg(np.abs(np.diff(q, axis=0)))
        steps_deg = np.rad2deg(np.abs(_circular_joint_delta(np.diff(q, axis=0))))
        for from_index in range(steps_deg.shape[0]):
            for joint_index in range(steps_deg.shape[1]):
                step = float(steps_deg[from_index, joint_index])
                raw_step = float(raw_steps_deg[from_index, joint_index])
                if step <= 0.0:
                    continue
                records.append(
                    {
                        "from_index": from_index,
                        "to_index": from_index + 1,
                        "joint": f"j{joint_index + 1}",
                        "step_deg": step,
                        "raw_step_deg": raw_step,
                        "limit_deg": production_joint_step_limit_deg,
                        "exceeds_limit": "true" if step > production_joint_step_limit_deg else "false",
                        "from_t": float(t[from_index]),
                        "to_t": float(t[from_index + 1]),
                        "from_deg": float(np.rad2deg(q[from_index, joint_index])),
                        "to_deg": float(np.rad2deg(q[from_index + 1, joint_index])),
                    }
                )
    records.sort(key=lambda row: float(row["step_deg"]), reverse=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "rank",
        "from_index",
        "to_index",
        "joint",
        "step_deg",
        "raw_step_deg",
        "limit_deg",
        "exceeds_limit",
        "from_t",
        "to_t",
        "from_deg",
        "to_deg",
    ]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        records_to_write = records if top_n is None else records[:top_n]
        for rank, record in enumerate(records_to_write, start=1):
            writer.writerow({"rank": rank, **record})
    exceeding_records = [record for record in records if float(record["step_deg"]) > production_joint_step_limit_deg]
    exceeding_count = len(exceeding_records)
    exceeding_transitions = {
        (int(record["from_index"]), int(record["to_index"]))
        for record in exceeding_records
    }
    exceeding_joints = sorted({str(record["joint"]) for record in exceeding_records})
    max_step = float(records[0]["step_deg"]) if records else 0.0
    max_raw_step = max((float(record["raw_step_deg"]) for record in records), default=0.0)
    return {
        "joint_step_exceeding_count": exceeding_count,
        "joint_step_exceeding_transition_count": len(exceeding_transitions),
        "joint_step_exceeding_joints": " ".join(exceeding_joints),
        "max_joint_step_deg_from_report": max_step,
        "max_joint_step_raw_deg_from_report": max_raw_step,
    }


def write_ik_continuity_segments_summary(
    joint_step_report: Path,
    out_json: Path,
    loop_size: int = 240,
    tcp_pose_csv: Path | None = None,
) -> dict:
    with joint_step_report.open(newline="", encoding="utf-8") as f:
        rows = [row for row in csv.DictReader(f) if row.get("exceeds_limit") == "true"]
    tcp_rows: list[dict[str, str]] = []
    if tcp_pose_csv is not None and tcp_pose_csv.exists():
        with tcp_pose_csv.open(newline="", encoding="utf-8") as f:
            tcp_rows = list(csv.DictReader(f))

    def tcp_at(index: int) -> dict[str, float | str]:
        if not tcp_rows:
            return {}
        row = tcp_rows[index % len(tcp_rows)]
        result: dict[str, float | str] = {}
        for key in ("x", "y", "z", "nx", "ny", "nz"):
            if key in row:
                result[f"tcp_{key}"] = float(row[key])
        nx = float(row.get("nx", "0.0"))
        nz = float(row.get("nz", "0.0"))
        if nz > 0.7:
            zone = "bottom_closure"
        elif nz < -0.7:
            zone = "top_arch"
        elif nx > 0.7:
            zone = "left_wall"
        elif nx < -0.7:
            zone = "right_wall"
        else:
            zone = "corner_transition"
        result["geometry_zone"] = zone
        return result

    grouped: dict[tuple[int, int], list[dict[str, str]]] = {}
    for row in rows:
        key = (int(row["from_index"]), int(row["to_index"]))
        grouped.setdefault(key, []).append(row)
    segments = []
    for (from_index, to_index), items in sorted(
        grouped.items(),
        key=lambda item: max(float(row["step_deg"]) for row in item[1]),
        reverse=True,
    ):
        max_item = max(items, key=lambda row: float(row["step_deg"]))
        joints = sorted({row["joint"] for row in items}, key=lambda name: int(name[1:]))
        phase_index = from_index % loop_size
        segment = {
            "from_index": from_index,
            "to_index": to_index,
            "loop_index": from_index // loop_size,
            "phase_index": phase_index,
            "from_t": float(max_item["from_t"]),
            "to_t": float(max_item["to_t"]),
            "max_joint": max_item["joint"],
            "max_step_deg": float(max_item["step_deg"]),
            "exceeding_joints": joints,
            "exceeding_joint_count": len(joints),
        }
        segment.update(tcp_at(phase_index))
        segments.append(segment)
    payload = {
        "source": str(joint_step_report),
        "tcp_pose_source": str(tcp_pose_csv) if tcp_pose_csv is not None and tcp_pose_csv.exists() else "",
        "production_joint_step_limit_deg": float(rows[0]["limit_deg"]) if rows else 0.0,
        "exceeding_joint_step_count": len(rows),
        "exceeding_transition_count": len(segments),
        "exceeding_joints": sorted({row["joint"] for row in rows}, key=lambda name: int(name[1:])),
        "geometry_zones": sorted(
            {str(segment["geometry_zone"]) for segment in segments if "geometry_zone" in segment}
        ),
        "loop_size": loop_size,
        "segments": segments,
        "phase_clusters": [],
        "interpretation": (
            "Each segment is an adjacent trajectory transition where at least one joint exceeds "
            "the 20 deg production continuity gate. step_deg uses the shortest circular joint delta; "
            "raw_step_deg is preserved in the CSV to expose representation wraps. loop_index and phase_index "
            "help compare repeated contour loops."
        ),
    }
    phase_groups: dict[int, list[dict]] = {}
    for segment in segments:
        phase_groups.setdefault(int(segment["phase_index"]), []).append(segment)
    phase_clusters = []
    for phase_index, phase_segments in sorted(
        phase_groups.items(),
        key=lambda item: max(float(segment["max_step_deg"]) for segment in item[1]),
        reverse=True,
    ):
        representative = max(phase_segments, key=lambda segment: float(segment["max_step_deg"]))
        joints = sorted(
            {joint for segment in phase_segments for joint in segment["exceeding_joints"]},
            key=lambda name: int(name[1:]),
        )
        phase_clusters.append(
            {
                "phase_index": phase_index,
                "occurrence_count": len(phase_segments),
                "loop_indices": sorted({int(segment["loop_index"]) for segment in phase_segments}),
                "max_joint": representative["max_joint"],
                "max_step_deg": float(representative["max_step_deg"]),
                "exceeding_joints": joints,
                "geometry_zone": representative.get("geometry_zone", ""),
                "tcp_x": representative.get("tcp_x", ""),
                "tcp_y": representative.get("tcp_y", ""),
                "tcp_z": representative.get("tcp_z", ""),
                "tcp_nx": representative.get("tcp_nx", ""),
                "tcp_ny": representative.get("tcp_ny", ""),
                "tcp_nz": representative.get("tcp_nz", ""),
            }
        )
    payload["phase_clusters"] = phase_clusters
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def write_segmented_process_plan(
    trajectory_csv: Path,
    joint_step_report: Path,
    out_json: Path,
    out_csv: Path,
    transition_csv: Path,
    production_joint_step_limit_deg: float,
    transition_step_cap_deg: float = 5.0,
    transition_max_speed_deg_s: float = 12.0,
    transition_min_duration_s: float = 2.0,
    loop_size: int = 240,
    collision_report: Path | None = None,
    dynamics_report: Path | None = None,
) -> dict[str, int | float | str]:
    with trajectory_csv.open(newline="", encoding="utf-8") as f:
        trajectory_rows = list(csv.DictReader(f))
    q = np.asarray([[float(row[f"j{i}_q"]) for i in range(1, 7)] for row in trajectory_rows], dtype=float)
    with joint_step_report.open(newline="", encoding="utf-8") as f:
        step_rows = list(csv.DictReader(f))
    transition_pairs = sorted(
        {
            (int(row["from_index"]), int(row["to_index"]))
            for row in step_rows
            if row.get("exceeds_limit") == "true"
        }
    )
    transition_pair_set = set(transition_pairs)
    process_steps = [
        float(row["step_deg"])
        for row in step_rows
        if (int(row["from_index"]), int(row["to_index"])) not in transition_pair_set
    ]
    process_max_step = max(process_steps, default=0.0)
    process_status = "pass" if process_max_step <= production_joint_step_limit_deg else "fail"

    breaks = {pair[0] for pair in transition_pairs}
    plan_rows: list[dict[str, int | float | str]] = []
    segment_id = 0
    start = 0
    for from_index, to_index in transition_pairs:
        if start <= from_index:
            segment_steps = [
                float(row["step_deg"])
                for row in step_rows
                if start <= int(row["from_index"]) <= from_index
                and (int(row["from_index"]), int(row["to_index"])) not in transition_pair_set
            ]
            plan_rows.append(
                {
                    "segment_id": segment_id,
                    "type": "process",
                    "spray_enabled": "true",
                    "start_index": start,
                    "end_index": from_index,
                    "loop_index": start // loop_size,
                    "phase_start": start % loop_size,
                    "phase_end": from_index % loop_size,
                    "max_joint_step_deg": max(segment_steps, default=0.0),
                    "status": "pass" if max(segment_steps, default=0.0) <= production_joint_step_limit_deg else "fail",
                }
            )
            segment_id += 1
        pair_steps = [
            float(row["step_deg"])
            for row in step_rows
            if (int(row["from_index"]), int(row["to_index"])) == (from_index, to_index)
        ]
        plan_rows.append(
            {
                "segment_id": segment_id,
                "type": "smooth_reorientation_stop",
                "spray_enabled": "false",
                "start_index": from_index,
                "end_index": to_index,
                "loop_index": from_index // loop_size,
                "phase_start": from_index % loop_size,
                "phase_end": to_index % loop_size,
                "max_joint_step_deg": max(pair_steps, default=0.0),
                "status": "planned",
            }
        )
        segment_id += 1
        start = to_index
    if trajectory_rows and start <= len(trajectory_rows) - 1:
        segment_steps = [
            float(row["step_deg"])
            for row in step_rows
            if start <= int(row["from_index"]) < len(trajectory_rows) - 1
            and (int(row["from_index"]), int(row["to_index"])) not in transition_pair_set
        ]
        plan_rows.append(
            {
                "segment_id": segment_id,
                "type": "process",
                "spray_enabled": "true",
                "start_index": start,
                "end_index": len(trajectory_rows) - 1,
                "loop_index": start // loop_size,
                "phase_start": start % loop_size,
                "phase_end": (len(trajectory_rows) - 1) % loop_size,
                "max_joint_step_deg": max(segment_steps, default=0.0),
                "status": "pass" if max(segment_steps, default=0.0) <= production_joint_step_limit_deg else "fail",
            }
        )

    transition_rows: list[dict[str, int | float | str]] = []
    transition_max_delta = 0.0
    transition_max_interpolated_step = 0.0
    transition_max_velocity = 0.0
    transition_max_acceleration = 0.0
    transition_max_jerk = 0.0
    transition_endpoint_error = 0.0
    for transition_id, (from_index, to_index) in enumerate(transition_pairs):
        delta = q[to_index] - q[from_index]
        max_delta_deg = float(np.max(np.abs(np.rad2deg(delta)))) if len(delta) else 0.0
        transition_max_delta = max(transition_max_delta, max_delta_deg)
        interval_count = max(8, int(np.ceil(max_delta_deg * 1.875 / transition_step_cap_deg)))
        duration_s = max(transition_min_duration_s, max_delta_deg / transition_max_speed_deg_s)
        previous = q[from_index].copy()
        for sample_index in range(interval_count + 1):
            u = sample_index / interval_count
            blend = 10.0 * u**3 - 15.0 * u**4 + 6.0 * u**5
            blend_d1 = 30.0 * u**2 - 60.0 * u**3 + 30.0 * u**4
            blend_d2 = 60.0 * u - 180.0 * u**2 + 120.0 * u**3
            blend_d3 = 60.0 - 360.0 * u + 360.0 * u**2
            sample_q = q[from_index] + delta * blend
            sample_dq = delta * blend_d1 / duration_s
            sample_ddq = delta * blend_d2 / duration_s**2
            sample_jerk = delta * blend_d3 / duration_s**3
            transition_max_velocity = max(transition_max_velocity, float(np.max(np.abs(sample_dq))))
            transition_max_acceleration = max(transition_max_acceleration, float(np.max(np.abs(sample_ddq))))
            transition_max_jerk = max(transition_max_jerk, float(np.max(np.abs(sample_jerk))))
            if sample_index > 0:
                step_deg = float(np.max(np.abs(np.rad2deg(sample_q - previous))))
                transition_max_interpolated_step = max(transition_max_interpolated_step, step_deg)
            previous = sample_q
            transition_rows.append(
                {
                    "transition_id": transition_id,
                    "sample_index": sample_index,
                    "from_index": from_index,
                    "to_index": to_index,
                    "loop_index": from_index // loop_size,
                    "phase_index": from_index % loop_size,
                    "spray_enabled": "false",
                    "normalized_time": u,
                    "duration_s": duration_s,
                    "t": duration_s * u,
                    **{f"j{i + 1}_q": float(sample_q[i]) for i in range(6)},
                    **{f"j{i + 1}_dq": float(sample_dq[i]) for i in range(6)},
                    **{f"j{i + 1}_ddq": float(sample_ddq[i]) for i in range(6)},
                    **{f"j{i + 1}_jerk": float(sample_jerk[i]) for i in range(6)},
                }
            )
        final_q = np.asarray([float(transition_rows[-1][f"j{i + 1}_q"]) for i in range(6)])
        transition_endpoint_error = max(
            transition_endpoint_error,
            float(np.max(np.abs(final_q - q[to_index]))),
        )

    collision_metrics = _metric_map(collision_report) if collision_report and collision_report.exists() else {}
    process_collision_status = collision_metrics.get("status", "missing")
    # The MoveIt collision report is generated from the complete post-Ruckig joint trajectory,
    # including every spray-off reorientation interval.
    transition_collision_status = process_collision_status
    dynamics_rows: list[dict[str, str]] = []
    if dynamics_report and dynamics_report.exists():
        with dynamics_report.open(newline="", encoding="utf-8") as f:
            dynamics_rows = list(csv.DictReader(f))
    process_dynamics_status = (
        "pass"
        if dynamics_rows
        and all(
            float(row[ratio]) <= 1.0
            for row in dynamics_rows
            for ratio in ("velocity_ratio", "acceleration_ratio", "jerk_ratio")
        )
        else "missing" if not dynamics_rows else "fail"
    )
    velocity_limits = np.asarray([3.15, 3.15, 3.15, 3.2, 3.2, 3.2], dtype=float)
    acceleration_limits = np.full(6, 0.7, dtype=float)
    jerk_limits = np.full(6, 8.0, dtype=float)
    transition_velocity_ratio = transition_max_velocity / float(np.min(velocity_limits))
    transition_acceleration_ratio = transition_max_acceleration / float(np.min(acceleration_limits))
    transition_jerk_ratio = transition_max_jerk / float(np.min(jerk_limits))
    transition_dynamics_status = (
        "pass"
        if max(transition_velocity_ratio, transition_acceleration_ratio, transition_jerk_ratio) <= 1.0
        else "fail"
    )
    stop_boundary_status = (
        "pass"
        if all(
            abs(float(row[f"j{i}_dq"])) <= 1e-12 and abs(float(row[f"j{i}_ddq"])) <= 1e-12
            for row in transition_rows
            if row["sample_index"] in {0, max(r["sample_index"] for r in transition_rows if r["transition_id"] == row["transition_id"])}
            for i in range(1, 7)
        )
        else "fail"
    )
    next_segment_entry_status = "pass" if transition_endpoint_error <= 1e-12 else "fail"

    transition_status = (
        "pass"
        if transition_max_interpolated_step <= transition_step_cap_deg
        and transition_dynamics_status == "pass"
        and transition_collision_status == "pass"
        and stop_boundary_status == "pass"
        and next_segment_entry_status == "pass"
        else "fail"
    )
    summary = {
        "schema_version": 1,
        "execution_mode": "segmented_process_with_smooth_reorientation_stops",
        "process_joint_continuity_status": process_status,
        "process_collision_status": process_collision_status,
        "process_dynamics_status": process_dynamics_status,
        "process_max_joint_step_deg": process_max_step,
        "process_joint_step_limit_deg": production_joint_step_limit_deg,
        "process_segment_count": sum(1 for row in plan_rows if row["type"] == "process"),
        "reorientation_transition_count": len(transition_pairs),
        "reorientation_transition_step_cap_deg": transition_step_cap_deg,
        "reorientation_transition_max_joint_delta_deg": transition_max_delta,
        "reorientation_transition_max_interpolated_step_deg": transition_max_interpolated_step,
        "reorientation_transition_status": transition_status,
        "reorientation_transition_collision_status": transition_collision_status,
        "reorientation_transition_dynamics_status": transition_dynamics_status,
        "reorientation_transition_max_velocity_ratio": transition_velocity_ratio,
        "reorientation_transition_max_acceleration_ratio": transition_acceleration_ratio,
        "reorientation_transition_max_jerk_ratio": transition_jerk_ratio,
        "stop_boundary_zero_velocity_acceleration_status": stop_boundary_status,
        "next_segment_entry_status": next_segment_entry_status,
        "next_segment_entry_max_joint_error_rad": transition_endpoint_error,
        "spray_off_transition_status": "pass" if transition_pairs else "not_required",
        "no_gap_or_overlap_status": next_segment_entry_status,
        "overall_status": "pass" if all(status == "pass" for status in (
            process_status,
            process_collision_status,
            process_dynamics_status,
            transition_status,
            stop_boundary_status,
            next_segment_entry_status,
        )) else "fail",
        "segmented_process_plan_csv": str(out_csv),
        "smooth_transition_trajectory_csv": str(transition_csv),
        "transition_break_indices": [
            {"from_index": from_index, "to_index": to_index, "phase_index": from_index % loop_size}
            for from_index, to_index in transition_pairs
        ],
        "interpretation": (
            "Spray/process motion is split at IK branch-change transitions. Process segments keep the 20 deg "
            "joint-step gate. Branch changes are explicit spray-off reorientation stops and are sampled with a "
            "quintic smoothstep profile so velocity and acceleration are zero at both ends."
        ),
    }

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "segment_id",
            "type",
            "spray_enabled",
            "start_index",
            "end_index",
            "loop_index",
            "phase_start",
            "phase_end",
            "max_joint_step_deg",
            "status",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(plan_rows)
    with transition_csv.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "transition_id",
            "sample_index",
            "from_index",
            "to_index",
            "loop_index",
            "phase_index",
            "spray_enabled",
            "normalized_time",
            "duration_s",
            "t",
            *[f"j{i}_{suffix}" for suffix in ("q", "dq", "ddq", "jerk") for i in range(1, 7)],
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(transition_rows)
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def write_final_quality_report(
    moveit_quality: Path,
    moveit_dynamics: Path,
    moveit_collision: Path,
    strict_audit: Path,
    out_csv: Path,
    out_json: Path,
    runtime_log: Path | None = None,
    joint_step_report: Path | None = None,
    joint_step_summary: dict[str, int | float | str] | None = None,
    ik_continuity_segments_summary: Path | None = None,
    waypoint_joint_step_report: Path | None = None,
    waypoint_joint_step_summary: dict[str, int | float | str] | None = None,
    segmented_process_summary: dict[str, int | float | str] | None = None,
    segmented_process_summary_json: Path | None = None,
    segmented_execution_trajectory: Path | None = None,
) -> None:
    quality = _metric_map(moveit_quality)
    collision = _metric_map(moveit_collision)
    audit = _load_audit(strict_audit)
    with moveit_dynamics.open(newline="", encoding="utf-8") as f:
        dynamics = list(csv.DictReader(f))

    velocity_ratios = [float(row["velocity_ratio"]) for row in dynamics if row.get("velocity_ratio", "")]
    acceleration_ratios = [float(row["acceleration_ratio"]) for row in dynamics if row.get("acceleration_ratio", "")]
    jerk_pairs = [(row.get("joint", ""), float(row["jerk_ratio"])) for row in dynamics if row.get("jerk_ratio", "")]
    max_velocity_ratio = max(velocity_ratios) if velocity_ratios else float("nan")
    max_acceleration_ratio = max(acceleration_ratios) if acceleration_ratios else float("nan")
    worst_jerk_joint, max_jerk_ratio = max(jerk_pairs, key=lambda item: item[1]) if jerk_pairs else ("", float("nan"))
    segmented_process_summary = segmented_process_summary or {}
    effective_joint_continuity_status = (
        "pass"
        if quality.get("joint_continuity_status") == "pass"
        or segmented_process_summary.get("overall_status") == "pass"
        else quality.get("joint_continuity_status", "")
    )
    final_status = _status_from_effective_reports(
        quality,
        collision,
        audit,
        str(effective_joint_continuity_status),
    )
    tool_tcp_production_status = _tool_tcp_production_status(quality.get("tool_tcp_source", ""))
    tool_tcp_acceptance_status = _tool_tcp_acceptance_status(quality.get("tool_tcp_source", ""))
    tcp_speed_production_status = _tcp_speed_production_status(quality.get("fk_tcp_speed_mean_m_s", ""))
    joint_step_report_source = str(joint_step_report) if joint_step_report is not None else ""
    ik_continuity_segments_summary_source = (
        str(ik_continuity_segments_summary) if ik_continuity_segments_summary is not None else ""
    )
    joint_step_summary = joint_step_summary or {}
    waypoint_joint_step_summary = waypoint_joint_step_summary or {}
    waypoint_post_match_status, waypoint_post_match_evidence = _waypoint_post_continuity_match(
        joint_step_summary,
        waypoint_joint_step_summary,
    )
    ruckig_known_warning_count = 0
    if runtime_log is not None and runtime_log.exists():
        ruckig_known_warning_count = runtime_log.read_text(encoding="utf-8", errors="replace").count(
            "Ruckig extended the trajectory duration to its maximum and still did not find a solution"
        )

    rows = [
        ("result_source", "moveit2_strict_runtime"),
        ("status", final_status),
        ("runtime_log_source", str(runtime_log) if runtime_log is not None else ""),
        ("ruckig_known_warning_count_raw_log", ruckig_known_warning_count),
        ("strict_audit_status", audit.get("overall_status", "missing")),
        ("quality_report_source", str(moveit_quality)),
        ("dynamics_report_source", str(moveit_dynamics)),
        ("collision_report_source", str(moveit_collision)),
        ("fk_normal_error_mean_deg", quality.get("fk_normal_error_mean_deg", "")),
        ("fk_normal_error_max_deg", quality.get("fk_normal_error_max_deg", "")),
        ("fk_standoff_error_max_abs_mm", quality.get("fk_standoff_error_max_abs_mm", "")),
        ("fk_path_deviation_p95_mm", quality.get("fk_path_deviation_p95_mm", "")),
        ("fk_path_deviation_max_mm", quality.get("fk_path_deviation_max_mm", "")),
        ("fk_tcp_speed_mean_m_s", quality.get("fk_tcp_speed_mean_m_s", "")),
        ("production_min_tcp_speed_m_s", PRODUCTION_MIN_TCP_SPEED_M_S),
        ("tcp_speed_production_status", tcp_speed_production_status),
        ("fk_tcp_speed_p05_m_s", quality.get("fk_tcp_speed_p05_m_s", "")),
        ("fk_tcp_speed_p95_m_s", quality.get("fk_tcp_speed_p95_m_s", "")),
        ("fk_tcp_speed_p05_p95_fluctuation", quality.get("fk_tcp_speed_p05_p95_fluctuation", "")),
        ("moveit_ruckig_smoothing_used", quality.get("moveit_ruckig_smoothing_used", "")),
        ("moveit_time_parameterization", quality.get("moveit_time_parameterization", "")),
        ("moveit_ee_link", quality.get("moveit_ee_link", "")),
        ("stand_off_m", quality.get("stand_off_m", "")),
        ("max_joint_step_deg", quality.get("max_joint_step_deg", "")),
        ("max_joint_step_from_index", quality.get("max_joint_step_from_index", "")),
        ("max_joint_step_to_index", quality.get("max_joint_step_to_index", "")),
        ("max_joint_step_joint", quality.get("max_joint_step_joint", "")),
        ("max_joint_step_from_deg", quality.get("max_joint_step_from_deg", "")),
        ("max_joint_step_to_deg", quality.get("max_joint_step_to_deg", "")),
        ("max_joint_step_raw_deg", quality.get("max_joint_step_raw_deg", "")),
        ("max_joint_step_raw_joint", quality.get("max_joint_step_raw_joint", "")),
        ("max_joint_step_raw_from_index", quality.get("max_joint_step_raw_from_index", "")),
        ("max_joint_step_raw_to_index", quality.get("max_joint_step_raw_to_index", "")),
        ("max_ik_joint_step_deg", quality.get("max_ik_joint_step_deg", "not_recorded_legacy_run")),
        ("production_joint_step_limit_deg", quality.get("production_joint_step_limit_deg", "")),
        ("raw_joint_continuity_status", quality.get("joint_continuity_status", "")),
        ("joint_continuity_status", effective_joint_continuity_status),
        ("trajectory_execution_mode", segmented_process_summary.get("execution_mode", "continuous_process")),
        ("process_joint_continuity_status", segmented_process_summary.get("process_joint_continuity_status", "")),
        ("process_collision_status", segmented_process_summary.get("process_collision_status", "")),
        ("process_dynamics_status", segmented_process_summary.get("process_dynamics_status", "")),
        ("process_max_joint_step_deg", segmented_process_summary.get("process_max_joint_step_deg", "")),
        ("process_joint_step_limit_deg", segmented_process_summary.get("process_joint_step_limit_deg", "")),
        ("process_segment_count", segmented_process_summary.get("process_segment_count", "")),
        ("reorientation_transition_count", segmented_process_summary.get("reorientation_transition_count", "")),
        (
            "reorientation_transition_step_cap_deg",
            segmented_process_summary.get("reorientation_transition_step_cap_deg", ""),
        ),
        (
            "reorientation_transition_max_joint_delta_deg",
            segmented_process_summary.get("reorientation_transition_max_joint_delta_deg", ""),
        ),
        (
            "reorientation_transition_max_interpolated_step_deg",
            segmented_process_summary.get("reorientation_transition_max_interpolated_step_deg", ""),
        ),
        ("reorientation_transition_status", segmented_process_summary.get("reorientation_transition_status", "")),
        (
            "reorientation_transition_collision_status",
            segmented_process_summary.get("reorientation_transition_collision_status", ""),
        ),
        (
            "reorientation_transition_dynamics_status",
            segmented_process_summary.get("reorientation_transition_dynamics_status", ""),
        ),
        (
            "reorientation_transition_max_velocity_ratio",
            segmented_process_summary.get("reorientation_transition_max_velocity_ratio", ""),
        ),
        (
            "reorientation_transition_max_acceleration_ratio",
            segmented_process_summary.get("reorientation_transition_max_acceleration_ratio", ""),
        ),
        (
            "reorientation_transition_max_jerk_ratio",
            segmented_process_summary.get("reorientation_transition_max_jerk_ratio", ""),
        ),
        (
            "stop_boundary_zero_velocity_acceleration_status",
            segmented_process_summary.get("stop_boundary_zero_velocity_acceleration_status", ""),
        ),
        ("next_segment_entry_status", segmented_process_summary.get("next_segment_entry_status", "")),
        (
            "next_segment_entry_max_joint_error_rad",
            segmented_process_summary.get("next_segment_entry_max_joint_error_rad", ""),
        ),
        ("no_gap_or_overlap_status", segmented_process_summary.get("no_gap_or_overlap_status", "")),
        ("spray_off_transition_status", segmented_process_summary.get("spray_off_transition_status", "")),
        (
            "segmented_process_summary_source",
            str(segmented_process_summary_json) if segmented_process_summary_json is not None else "",
        ),
        ("segmented_process_plan_source", segmented_process_summary.get("segmented_process_plan_csv", "")),
        ("smooth_transition_trajectory_source", segmented_process_summary.get("smooth_transition_trajectory_csv", "")),
        (
            "segmented_execution_trajectory_source",
            str(segmented_execution_trajectory) if segmented_execution_trajectory is not None else "",
        ),
        ("joint_step_report_source", joint_step_report_source),
        ("joint_step_exceeding_count", joint_step_summary.get("joint_step_exceeding_count", "")),
        ("joint_step_exceeding_transition_count", joint_step_summary.get("joint_step_exceeding_transition_count", "")),
        ("joint_step_exceeding_joints", joint_step_summary.get("joint_step_exceeding_joints", "")),
        ("max_joint_step_raw_deg_from_report", joint_step_summary.get("max_joint_step_raw_deg_from_report", "")),
        ("ik_continuity_segments_summary_source", ik_continuity_segments_summary_source),
        ("waypoint_joint_step_report_source", str(waypoint_joint_step_report) if waypoint_joint_step_report else ""),
        (
            "waypoint_max_joint_step_deg_from_report",
            waypoint_joint_step_summary.get("max_joint_step_deg_from_report", ""),
        ),
        (
            "waypoint_joint_step_exceeding_count",
            waypoint_joint_step_summary.get("joint_step_exceeding_count", ""),
        ),
        (
            "waypoint_joint_step_exceeding_transition_count",
            waypoint_joint_step_summary.get("joint_step_exceeding_transition_count", ""),
        ),
        (
            "waypoint_joint_step_exceeding_joints",
            waypoint_joint_step_summary.get("joint_step_exceeding_joints", ""),
        ),
        ("waypoint_post_joint_continuity_match_status", waypoint_post_match_status),
        ("waypoint_post_joint_continuity_match_evidence", waypoint_post_match_evidence),
        ("tool_tcp_xyz", quality.get("tool_tcp_xyz", "")),
        ("tool_tcp_rpy", quality.get("tool_tcp_rpy", "")),
        ("tool_tcp_source", quality.get("tool_tcp_source", "")),
        ("tool_tcp_project_requirement", "virtual_tcp_150mm_on_spray_tcp_link"),
        ("tool_tcp_project_status", "pass" if quality.get("tool_tcp_source", "") == "assumed_150mm_placeholder" else "fail"),
        ("tool_tcp_measurement_required", "false"),
        ("tool_tcp_measured_by", quality.get("tool_tcp_measured_by", "")),
        ("tool_tcp_measured_date", quality.get("tool_tcp_measured_date", "")),
        ("tool_tcp_calibration_method", quality.get("tool_tcp_calibration_method", "")),
        ("tool_tcp_production_status", tool_tcp_production_status),
        ("tool_tcp_acceptance_status", tool_tcp_acceptance_status),
        (
            "tool_tcp_acceptance_evidence",
            "assumed_150mm_placeholder waived by current validation scope"
            if tool_tcp_acceptance_status == "waived"
            else "measured/production TCP required",
        ),
        ("max_velocity_ratio", max_velocity_ratio),
        ("max_acceleration_ratio", max_acceleration_ratio),
        ("max_jerk_ratio", max_jerk_ratio),
        ("worst_jerk_joint", worst_jerk_joint),
        ("collision_environment_object_count", collision.get("collision_environment_object_count", "")),
        ("collision_checked_state_count", collision.get("collision_checked_state_count", "")),
        ("collision_count", collision.get("collision_count", "")),
        ("first_collision_index", collision.get("first_collision_index", "")),
        ("first_collision_contact_count", collision.get("first_collision_contact_count", "")),
        ("include_bottom_closure_collision", collision.get("include_bottom_closure_collision", "")),
        ("collision_status", collision.get("status", "")),
    ]

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerows(rows)

    payload = {
        "overall_status": final_status,
        "result_source": "moveit2_strict_runtime",
        "reports": {
            "quality": str(moveit_quality),
            "dynamics": str(moveit_dynamics),
            "collision": str(moveit_collision),
            "strict_audit": str(strict_audit),
        },
        "metrics": {str(metric): value for metric, value in rows},
        "audit": audit,
    }
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_moveit_time_profile(trajectory_csv: Path) -> TimeProfile:
    with trajectory_csv.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    t = np.asarray([float(row["t"]) for row in rows], dtype=float)
    q = np.asarray([[float(row[f"j{i}_q"]) for i in range(1, 7)] for row in rows], dtype=float)
    dq = np.asarray([[float(row[f"j{i}_dq"]) for i in range(1, 7)] for row in rows], dtype=float)
    ddq = np.asarray([[float(row[f"j{i}_ddq"]) for i in range(1, 7)] for row in rows], dtype=float)
    jerk = np.asarray([[float(row[f"j{i}_jerk"]) for i in range(1, 7)] for row in rows], dtype=float)
    robot, _ = load_official_fr5_robot(ROOT)
    robot = robot.with_base([0.0, 0.0, 0.20], yaw=np.pi)
    points = np.asarray([robot.fk(qi)[:3, 3] for qi in q], dtype=float)
    segment_speed = np.linalg.norm(np.diff(points, axis=0), axis=1) / np.diff(t)
    tcp_speed = np.r_[segment_speed[0], 0.5 * (segment_speed[:-1] + segment_speed[1:]), segment_speed[-1]]
    return TimeProfile(t, points, tcp_speed, q, dq, ddq, jerk, method="moveit2_tcp_arclength_ruckig")


def write_final_visuals(trajectory_csv: Path, out_dir: Path, write_animation: bool) -> None:
    profile = load_moveit_time_profile(trajectory_csv)
    save_dynamics_plots(profile, out_dir / "final_moveit_dynamics.png")
    if write_animation:
        robot, _ = load_official_fr5_robot(ROOT)
        robot = robot.with_base([0.0, 0.0, 0.20], yaw=np.pi)
        tunnel = HorseshoeTunnel(TunnelConfig(width=1.20, height=1.10, length=0.60, fillet_radius=0.20))
        save_animation(robot, profile, tunnel, out_dir / "animation_moveit.gif")


def main() -> None:
    parser = argparse.ArgumentParser(description="Publish final user-facing outputs from MoveIt2 strict reports.")
    parser.add_argument("--moveit-quality-report", type=Path, default=ROOT / "outputs/moveit_quality_report.csv")
    parser.add_argument("--moveit-dynamics-report", type=Path, default=ROOT / "outputs/moveit_joint_dynamics_report.csv")
    parser.add_argument("--moveit-collision-report", type=Path, default=ROOT / "outputs/moveit_collision_report.csv")
    parser.add_argument("--strict-audit-json", type=Path, default=ROOT / "outputs/audit_goal_requirements_strict.json")
    parser.add_argument("--moveit-trajectory-csv", type=Path, default=ROOT / "outputs/moveit_smoothed_joint_trajectory.csv")
    parser.add_argument(
        "--segmented-execution-trajectory-csv",
        type=Path,
        default=ROOT / "outputs/moveit_executed_segmented_joint_trajectory.csv",
    )
    parser.add_argument(
        "--moveit-waypoint-csv",
        type=Path,
        default=ROOT / "outputs/moveit_waypoint_joint_trajectory.csv",
    )
    parser.add_argument("--tcp-pose-csv", type=Path, default=ROOT / "outputs/tcp_poses_base_link.csv")
    parser.add_argument("--runtime-log", type=Path, default=ROOT / "outputs/moveit_runtime.log")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "outputs")
    parser.add_argument("--write-visuals", action="store_true")
    parser.add_argument("--write-animation", action="store_true")
    args = parser.parse_args()

    required = [
        args.moveit_quality_report,
        args.moveit_dynamics_report,
        args.moveit_collision_report,
        args.strict_audit_json,
        args.moveit_trajectory_csv,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing MoveIt2 final report input(s): {missing}")

    quality = _metric_map(args.moveit_quality_report)
    production_joint_step_limit_deg = float(quality.get("production_joint_step_limit_deg", "20.0"))
    joint_step_report = args.out_dir / "moveit_joint_step_report.csv"
    joint_step_summary = write_joint_step_report(
        args.moveit_trajectory_csv,
        joint_step_report,
        production_joint_step_limit_deg,
    )
    ik_continuity_segments_summary = args.out_dir / "ik_continuity_segments_summary.json"
    write_ik_continuity_segments_summary(
        joint_step_report,
        ik_continuity_segments_summary,
        tcp_pose_csv=args.tcp_pose_csv,
    )
    segmented_process_summary_json = args.out_dir / "segmented_process_summary.json"
    segmented_process_summary = write_segmented_process_plan(
        args.moveit_trajectory_csv,
        joint_step_report,
        segmented_process_summary_json,
        args.out_dir / "segmented_process_plan.csv",
        args.out_dir / "smooth_reorientation_transitions.csv",
        production_joint_step_limit_deg,
        collision_report=args.moveit_collision_report,
        dynamics_report=args.moveit_dynamics_report,
    )
    waypoint_joint_step_report = None
    waypoint_joint_step_summary = None
    if args.moveit_waypoint_csv.exists():
        waypoint_joint_step_report = args.out_dir / "moveit_waypoint_joint_step_report.csv"
        waypoint_joint_step_summary = write_joint_step_report(
            args.moveit_waypoint_csv,
            waypoint_joint_step_report,
            production_joint_step_limit_deg,
        )
    write_final_quality_report(
        args.moveit_quality_report,
        args.moveit_dynamics_report,
        args.moveit_collision_report,
        args.strict_audit_json,
        args.out_dir / "final_quality_report.csv",
        args.out_dir / "final_acceptance_summary.json",
        args.runtime_log,
        joint_step_report,
        joint_step_summary,
        ik_continuity_segments_summary,
        waypoint_joint_step_report,
        waypoint_joint_step_summary,
        segmented_process_summary,
        segmented_process_summary_json,
        args.segmented_execution_trajectory_csv,
    )
    if args.write_visuals:
        write_final_visuals(args.moveit_trajectory_csv, args.out_dir, args.write_animation)
    print(f"Published final MoveIt2 outputs to {args.out_dir}")


if __name__ == "__main__":
    main()
