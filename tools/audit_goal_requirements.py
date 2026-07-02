from __future__ import annotations

import csv
import argparse
import json
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class AuditItem:
    requirement: str
    status: str
    evidence: str


def _read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def _metric_map(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as f:
        return {row["metric"]: row["value"] for row in csv.DictReader(f)}


def _moveit_runtime_report_status(
    quality_path: Path,
    dynamics_path: Path,
    fk_trace_path: Path,
    collision_path: Path | None = None,
) -> tuple[str, str]:
    if not quality_path.exists() or not dynamics_path.exists() or not fk_trace_path.exists():
        return (
            "warn",
            "No local ROS2/MoveIt2 runtime reports were found; this Windows run can only prove static bridge wiring and generated inputs",
        )
    quality = _metric_map(quality_path)
    if quality.get("status") != "pass":
        return ("fail", f"MoveIt FK quality report status is {quality.get('status', 'missing')!r}")
    required_quality = [
        "fk_normal_error_max_deg",
        "fk_standoff_error_max_abs_mm",
        "fk_path_deviation_max_mm",
        "fk_tcp_speed_p05_p95_fluctuation",
        "moveit_ruckig_smoothing_used",
        "moveit_ee_link",
    ]
    missing_quality = [key for key in required_quality if key not in quality]
    if missing_quality:
        return ("fail", "MoveIt FK quality report is missing: " + ", ".join(missing_quality))
    if quality.get("moveit_ruckig_smoothing_used", "").strip().lower() != "true":
        return ("fail", "MoveIt FK quality report does not prove native Ruckig smoothing was used")
    if quality.get("moveit_ee_link", "").strip() != "spray_tcp_link":
        return ("fail", f"MoveIt FK quality report used unexpected ee_link={quality.get('moveit_ee_link', 'missing')!r}")

    required_trace = {
        "t",
        "actual_tcp_x",
        "actual_tcp_y",
        "actual_tcp_z",
        "normal_angle_error_deg",
        "standoff_error_mm",
        "path_deviation_mm",
        "tcp_speed_m_s",
    }
    with fk_trace_path.open(newline="", encoding="utf-8") as f:
        trace_reader = csv.DictReader(f)
        if trace_reader.fieldnames is None:
            return ("fail", "MoveIt FK trace report has no header")
        missing_trace = sorted(required_trace - set(trace_reader.fieldnames))
        if missing_trace:
            return ("fail", "MoveIt FK trace report is missing: " + ", ".join(missing_trace))
        trace_data = list(trace_reader)
        trace_rows = len(trace_data)
    if trace_rows == 0:
        return ("fail", "MoveIt FK trace report is empty")
    trace_numeric: dict[str, np.ndarray] = {}
    for key in required_trace:
        try:
            values = np.asarray([float(row[key]) for row in trace_data], dtype=float)
        except ValueError:
            return ("fail", f"MoveIt FK trace report has non-numeric values in {key}")
        if not np.all(np.isfinite(values)):
            return ("fail", f"MoveIt FK trace report has non-finite values in {key}")
        trace_numeric[key] = values
    if np.any(np.diff(trace_numeric["t"]) <= 0.0):
        return ("fail", "MoveIt FK trace timestamps are not strictly increasing")
    if np.any(trace_numeric["tcp_speed_m_s"] < -1e-9):
        return ("fail", "MoveIt FK trace contains negative TCP speed")
    tolerance = 1.0e-6
    trace_normal_max = float(np.max(np.abs(trace_numeric["normal_angle_error_deg"])))
    trace_standoff_max = float(np.max(np.abs(trace_numeric["standoff_error_mm"])))
    trace_path_max = float(np.max(np.abs(trace_numeric["path_deviation_mm"])))
    if trace_normal_max > float(quality["fk_normal_error_max_deg"]) + tolerance:
        return ("fail", "MoveIt FK trace normal-angle errors exceed the summary report")
    if trace_standoff_max > float(quality["fk_standoff_error_max_abs_mm"]) + tolerance:
        return ("fail", "MoveIt FK trace stand-off errors exceed the summary report")
    if trace_path_max > float(quality["fk_path_deviation_max_mm"]) + tolerance:
        return ("fail", "MoveIt FK trace path deviations exceed the summary report")

    rows = []
    with dynamics_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return ("fail", "MoveIt joint dynamics report is empty")
    for row in rows:
        for key in ("velocity_ratio", "acceleration_ratio", "jerk_ratio"):
            if key not in row:
                return ("fail", f"MoveIt joint dynamics report missing {key}")
            if float(row[key]) > 1.02:
                return (
                    "fail",
                    f"MoveIt joint dynamics report exceeds {key}: joint={row.get('joint', '?')} value={row[key]}",
                )
    max_jerk_ratio = max(float(row["jerk_ratio"]) for row in rows)

    collision_suffix = ""
    if collision_path is not None and collision_path.exists():
        collision = _metric_map(collision_path)
        if collision.get("status") != "pass":
            return ("fail", f"MoveIt collision report status is {collision.get('status', 'missing')!r}")
        if float(collision.get("collision_count", "nan")) != 0.0:
            return ("fail", f"MoveIt collision report found collision_count={collision.get('collision_count')}")
        collision_states = int(float(collision.get("collision_checked_state_count", "0")))
        collision_objects = int(float(collision.get("collision_environment_object_count", "0")))
        collision_suffix = f", collision states={collision_states}, environment objects={collision_objects}"
    return (
        "pass",
        f"MoveIt FK quality report passed, FK trace rows={trace_rows}, joint dynamics ratios are within limits; "
        f"max_jerk_ratio={max_jerk_ratio:.4g}{collision_suffix}",
    )


def _tcp_pose_normal_error(csv_path: Path) -> tuple[float, int]:
    max_error = 0.0
    count = 0
    with csv_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            qx, qy, qz, qw = (float(row[key]) for key in ("qx", "qy", "qz", "qw"))
            quat_norm = max(float(np.linalg.norm([qx, qy, qz, qw])), 1e-12)
            qx, qy, qz, qw = qx / quat_norm, qy / quat_norm, qz / quat_norm, qw / quat_norm
            tool_z = np.array(
                [
                    2.0 * (qx * qz + qy * qw),
                    2.0 * (qy * qz - qx * qw),
                    1.0 - 2.0 * (qx * qx + qy * qy),
                ],
                dtype=float,
            )
            normal = np.array([float(row["nx"]), float(row["ny"]), float(row["nz"])], dtype=float)
            normal /= max(float(np.linalg.norm(normal)), 1e-12)
            angle = float(np.rad2deg(np.arccos(np.clip(np.dot(tool_z, normal), -1.0, 1.0))))
            max_error = max(max_error, angle)
            count += 1
    return max_error, count


def audit_goal_requirements(
    moveit_quality_report: Path | None = None,
    moveit_dynamics_report: Path | None = None,
    moveit_fk_trace: Path | None = None,
    moveit_collision_report: Path | None = None,
) -> list[AuditItem]:
    items: list[AuditItem] = []
    example = _read("examples/run_fr5_tunnel_spray.py")
    metrics_source = _read("src/metrics.py")
    robot_source = _read("src/robot_model.py")
    bridge = _read("ros2_moveit_bridge/plan_closed_contour_moveit.py")
    srdf = _read("ros2_moveit_bridge/config/fairino5_v6_spray_tcp.srdf")
    report = _metric_map(ROOT / "outputs/quality_report.csv")

    fk_code_ok = all(
        needle in example + metrics_source
        for needle in [
            "fk_tcp_profile",
            "tcp_speed_from_positions(actual_positions, profile.t)",
            "project_points_to_polyline(points, wall_points, wall_normals",
        ]
    )
    fk_report_ok = all(
        report.get(key) == "pass"
        for key in ["fk_path_deviation_status", "normal_angle_status", "standoff_status"]
    )
    items.append(
        AuditItem(
            "Real TCP speed and stand-off are recomputed from FK(q(t))",
            "pass" if fk_code_ok and fk_report_ok else "fail",
            "example FK profile uses tcp_speed_from_positions; final strict validation uses MoveIt2 FK reports for TCP speed, normal, and stand-off gates"
            if fk_code_ok and fk_report_ok
            else "Missing FK-derived speed/distance code or current FK quality report gates do not pass",
        )
    )

    tcp_csv = ROOT / "outputs/tcp_poses.csv"
    if tcp_csv.exists():
        max_normal_error, pose_count = _tcp_pose_normal_error(tcp_csv)
        pose_csv_ok = pose_count > 0 and max_normal_error < 0.1
    else:
        max_normal_error, pose_count, pose_csv_ok = float("nan"), 0, False
    moveit_normal_ok = all(
        needle in bridge
        for needle in [
            "load_tcp_poses",
            "TCP quaternion is not aligned with the wall normal",
            "planning_component.set_goal_state",
            "pose_link=ee_link",
        ]
    ) and 'tip_link="spray_tcp_link"' in srdf
    items.append(
        AuditItem(
            "MoveIt2 uses wall-normal TCP pose goals",
            "pass" if moveit_normal_ok and pose_csv_ok else "fail",
            f"tcp_poses pose_count={pose_count}, quaternion-vs-normal max error={max_normal_error:.6g} deg; MoveIt goal link is spray_tcp_link"
            if moveit_normal_ok and pose_csv_ok
            else "MoveIt bridge pose handling, SRDF TCP tip, or tcp_poses normal alignment is missing",
        )
    )

    ik_orientation_ok = all(
        needle in robot_source
        for needle in [
            "orientation_weight: float = 0.1",
            "target_tool_z = target[:3, 2]",
            "orientation_weight * axis_err",
            "orientation_weight=orientation_weight",
        ]
    ) and float(report.get("normal_angle_error_max_deg", "999")) < 10.0
    items.append(
        AuditItem(
            "IK includes nozzle wall-normal orientation constraints",
            "pass" if ik_orientation_ok else "fail",
            f"IK residual includes tool-Z axis error; current FK normal_angle_error_max_deg={report.get('normal_angle_error_max_deg', 'missing')}"
            if ik_orientation_ok
            else "IK orientation residual or current normal-angle report is missing/failing",
        )
    )

    totg = "trajectory.apply_totg_time_parameterization"
    ruckig = "_apply_ruckig_smoothing_without_known_false_error"
    build_fn = bridge[bridge.find("def build_and_smooth_moveit_trajectory") :]
    ruckig_order_ok = (
        totg in build_fn
        and ruckig in build_fn
        and build_fn.index(totg) < build_fn.index(ruckig)
        and "trajectory.apply_ruckig_smoothing" in bridge
        and "Execution requires MoveIt2 native Ruckig smoothing" in bridge
    )
    items.append(
        AuditItem(
            "Ruckig smoothing runs after MoveIt2 trajectory planning/time-parameterization",
            "pass" if ruckig_order_ok else "fail",
            "bridge builds the MoveIt trajectory, applies configured time-parameterization, then native RobotTrajectory.apply_ruckig_smoothing; execution refuses use_ruckig_smoothing:=false"
            if ruckig_order_ok
            else "MoveIt2 TOTG/Ruckig order or execution guard is missing",
        )
    )

    runtime_status, runtime_evidence = _moveit_runtime_report_status(
        moveit_quality_report or (ROOT / "outputs/moveit_quality_report.csv"),
        moveit_dynamics_report or (ROOT / "outputs/moveit_joint_dynamics_report.csv"),
        moveit_fk_trace or (ROOT / "outputs/moveit_fk_tcp_trace.csv"),
        moveit_collision_report or (ROOT / "outputs/moveit_collision_report.csv"),
    )
    items.append(
        AuditItem(
            "ROS2/MoveIt2 runtime execution-chain evidence",
            runtime_status,
            runtime_evidence,
        )
    )
    return items


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit FR5 closed-contour execution-chain requirements.")
    parser.add_argument(
        "--strict-runtime",
        action="store_true",
        help="Treat missing ROS2/MoveIt2 runtime reports as a failure instead of a local-development warning.",
    )
    parser.add_argument(
        "--moveit-quality-report",
        type=Path,
        default=ROOT / "outputs/moveit_quality_report.csv",
        help="Path to the MoveIt FK quality report generated by ros2_moveit_bridge.",
    )
    parser.add_argument(
        "--moveit-dynamics-report",
        type=Path,
        default=ROOT / "outputs/moveit_joint_dynamics_report.csv",
        help="Path to the MoveIt joint dynamics report generated by ros2_moveit_bridge.",
    )
    parser.add_argument(
        "--moveit-fk-trace",
        type=Path,
        default=ROOT / "outputs/moveit_fk_tcp_trace.csv",
        help="Path to the per-sample MoveIt FK TCP trace generated by ros2_moveit_bridge.",
    )
    parser.add_argument(
        "--moveit-collision-report",
        type=Path,
        default=ROOT / "outputs/moveit_collision_report.csv",
        help="Path to the MoveIt collision report generated by ros2_moveit_bridge.",
    )
    parser.add_argument(
        "--json-report",
        type=Path,
        default=None,
        help="Optional path to write a machine-readable audit report.",
    )
    args = parser.parse_args()
    items = audit_goal_requirements(
        args.moveit_quality_report,
        args.moveit_dynamics_report,
        args.moveit_fk_trace,
        args.moveit_collision_report,
    )
    for item in items:
        print(f"{item.status.upper()}: {item.requirement}")
        print(f"  {item.evidence}")
    has_failure = any(item.status == "fail" for item in items)
    has_runtime_warning = any(
        item.status == "warn" and item.requirement == "ROS2/MoveIt2 runtime execution-chain evidence"
        for item in items
    )
    exit_code = 1 if has_failure or (args.strict_runtime and has_runtime_warning) else 0
    if args.json_report is not None:
        args.json_report.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "overall_status": "fail" if exit_code else "pass",
            "strict_runtime": bool(args.strict_runtime),
            "items": [asdict(item) for item in items],
        }
        args.json_report.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
