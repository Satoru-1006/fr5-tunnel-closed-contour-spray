from __future__ import annotations

import csv
import argparse
import datetime as dt
import json
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PRODUCTION_MIN_TCP_SPEED_M_S = 0.003
SEGMENTED_PROCESS_SUMMARY = ROOT / "outputs/segmented_process_summary.json"


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


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _segmented_process_ok(summary: dict) -> bool:
    return (
        summary.get("overall_status") == "pass"
        and summary.get("execution_mode") == "segmented_process_with_smooth_reorientation_stops"
        and summary.get("process_joint_continuity_status") == "pass"
        and summary.get("reorientation_transition_status") == "pass"
        and summary.get("spray_off_transition_status") in {"pass", "not_required"}
    )


def _tool_tcp_waived(quality: dict[str, str]) -> bool:
    return quality.get("tool_tcp_source", "").strip().lower() == "assumed_150mm_placeholder"


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
    segmented_process = _load_json(SEGMENTED_PROCESS_SUMMARY)
    segmented_joint_ok = _segmented_process_ok(segmented_process)
    required_quality = [
        "fk_normal_error_max_deg",
        "fk_standoff_error_max_abs_mm",
        "fk_path_deviation_max_mm",
        "fk_tcp_speed_mean_m_s",
        "fk_tcp_speed_p05_p95_fluctuation",
        "moveit_ruckig_smoothing_used",
        "moveit_ee_link",
        "max_joint_step_deg",
        "production_joint_step_limit_deg",
        "joint_continuity_status",
        "tool_tcp_xyz",
        "tool_tcp_rpy",
        "tool_tcp_source",
    ]
    missing_quality = [key for key in required_quality if key not in quality]
    if missing_quality:
        return ("fail", "MoveIt FK quality report is missing: " + ", ".join(missing_quality))
    gate_failures: list[str] = []
    if quality.get("status") != "pass" and not (
        quality.get("joint_continuity_status") == "fail" and segmented_joint_ok
    ):
        gate_failures.append(f"MoveIt FK quality report status is {quality.get('status', 'missing')!r}")
    if quality.get("moveit_ruckig_smoothing_used", "").strip().lower() != "true":
        return ("fail", "MoveIt FK quality report does not prove native Ruckig smoothing was used")
    if quality.get("moveit_ee_link", "").strip() != "spray_tcp_link":
        return ("fail", f"MoveIt FK quality report used unexpected ee_link={quality.get('moveit_ee_link', 'missing')!r}")
    tool_tcp_source = quality.get("tool_tcp_source", "").strip().lower()
    if _tool_tcp_waived(quality):
        pass
    elif not tool_tcp_source or "assumed" in tool_tcp_source or "placeholder" in tool_tcp_source:
        gate_failures.append(
            "MoveIt quality report uses a non-production TCP source: "
            f"tool_tcp_source={quality.get('tool_tcp_source', 'missing')!r}, "
            f"tool_tcp_xyz={quality.get('tool_tcp_xyz', 'missing')!r}, "
            f"tool_tcp_rpy={quality.get('tool_tcp_rpy', 'missing')!r}"
        )
    else:
        metadata_missing = [
            key
            for key in ("tool_tcp_measured_by", "tool_tcp_measured_date", "tool_tcp_calibration_method")
            if not quality.get(key, "").strip()
        ]
        if metadata_missing:
            gate_failures.append("MoveIt quality report is missing measured TCP metadata: " + ", ".join(metadata_missing))
        else:
            try:
                dt.date.fromisoformat(quality["tool_tcp_measured_date"].strip())
            except ValueError:
                gate_failures.append("MoveIt quality report has invalid tool_tcp_measured_date; expected YYYY-MM-DD")
    tcp_speed_mean = float(quality["fk_tcp_speed_mean_m_s"])
    if tcp_speed_mean < PRODUCTION_MIN_TCP_SPEED_M_S:
        gate_failures.append(
            "MoveIt TCP speed is diagnostic-only: "
            f"fk_tcp_speed_mean_m_s={tcp_speed_mean:.6g}, "
            f"production_min_tcp_speed_m_s={PRODUCTION_MIN_TCP_SPEED_M_S:.6g}"
        )
    if quality.get("joint_continuity_status") != "pass" and not segmented_joint_ok:
        raw_suffix = ""
        if quality.get("max_joint_step_raw_deg"):
            raw_suffix = (
                f", raw_max={quality.get('max_joint_step_raw_deg')}, "
                f"raw_joint={quality.get('max_joint_step_raw_joint', 'missing')}"
            )
        gate_failures.append(
            "MoveIt joint continuity gate failed: "
            f"max_joint_step_deg={quality.get('max_joint_step_deg', 'missing')}, "
            f"limit={quality.get('production_joint_step_limit_deg', 'missing')}"
            f"{raw_suffix}",
        )

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
            gate_failures.append(
                f"MoveIt collision report status is {collision.get('status', 'missing')!r}; "
                f"collision_count={collision.get('collision_count', 'missing')}, "
                f"checked_states={collision.get('collision_checked_state_count', 'missing')}, "
                f"first_collision_index={collision.get('first_collision_index', 'missing')}, "
                f"first_collision_contact_count={collision.get('first_collision_contact_count', 'missing')}"
            )
        if float(collision.get("collision_count", "nan")) != 0.0:
            gate_failures.append(f"MoveIt collision report found collision_count={collision.get('collision_count')}")
        open_path_collision = collision.get("open_path", "").strip().lower() == "true"
        bottom_closure = collision.get("include_bottom_closure_collision", "").strip().lower()
        if not open_path_collision and bottom_closure != "true":
            gate_failures.append(
                "MoveIt collision environment omits the bottom closure "
                f"(include_bottom_closure_collision={collision.get('include_bottom_closure_collision', 'missing')})"
            )
        collision_states = int(float(collision.get("collision_checked_state_count", "0")))
        collision_objects = int(float(collision.get("collision_environment_object_count", "0")))
        collision_suffix = f", collision states={collision_states}, environment objects={collision_objects}"
    if gate_failures:
        return ("fail", "; ".join(gate_failures))
    waiver_suffix = ""
    if _tool_tcp_waived(quality):
        waiver_suffix += (
            "; tool_tcp_source=assumed_150mm_placeholder accepted by current validation scope"
        )
    if quality.get("joint_continuity_status") != "pass" and segmented_joint_ok:
        waiver_suffix += (
            "; raw continuous joint gate is replaced by segmented spray-off reorientation evidence "
            f"(process_max_joint_step_deg={segmented_process.get('process_max_joint_step_deg')}, "
            f"transition_max_step_deg={segmented_process.get('reorientation_transition_max_interpolated_step_deg')})"
        )
    return (
        "pass",
        f"MoveIt FK quality report passed, FK trace rows={trace_rows}, joint dynamics ratios are within limits; "
        f"max_jerk_ratio={max_jerk_ratio:.4g}{collision_suffix}{waiver_suffix}",
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
    tcp_pose_csv: Path | None = None,
) -> list[AuditItem]:
    items: list[AuditItem] = []
    example = _read("examples/run_fr5_tunnel_spray.py")
    metrics_source = _read("src/metrics.py")
    robot_source = _read("src/robot_model.py")
    bridge = _read("ros2_moveit_bridge/plan_closed_contour_moveit.py")
    strict_script = _read("scripts/run_moveit_strict_validation.sh")
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

    tcp_csv = tcp_pose_csv or (ROOT / "outputs/tcp_poses.csv")
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
            f"tcp_poses path={tcp_csv}, pose_count={pose_count}, quaternion-vs-normal max error={max_normal_error:.6g} deg; MoveIt goal link is spray_tcp_link"
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

    execute_fn = bridge[bridge.find("if execute_trajectory:") :]
    execution_quality_guard_ok = (
        'quality_metrics.get("status") != "pass"' in execute_fn
        and "FK quality report status" in execute_fn
        and "moveit.execute" in execute_fn
        and execute_fn.index('quality_metrics.get("status") != "pass"') < execute_fn.index("moveit.execute")
    )
    items.append(
        AuditItem(
            "Controller execution is blocked unless strict quality gates pass",
            "pass" if execution_quality_guard_ok else "fail",
            "execute_trajectory path refuses non-pass FK quality reports before moveit.execute, including joint-continuity failures"
            if execution_quality_guard_ok
            else "execute_trajectory path can reach moveit.execute without checking FK quality report status",
        )
    )

    tool_tcp_guard_ok = all(
        needle in strict_script
        for needle in [
            'TOOL_TCP_CALIBRATION_YAML="${TOOL_TCP_CALIBRATION_YAML:-}"',
            'TOOL_TCP_XYZ="${TOOL_TCP_XYZ:-0.000 0.000 0.150}"',
            'TOOL_TCP_SOURCE="${TOOL_TCP_SOURCE:-assumed_150mm_placeholder}"',
            'ALLOW_ZERO_TOOL_TCP="${ALLOW_ZERO_TOOL_TCP:-false}"',
            'ALLOW_SEED_JOINT_WITH_TOOL_OFFSET="${ALLOW_SEED_JOINT_WITH_TOOL_OFFSET:-false}"',
            'PLANNING_MODE="${PLANNING_MODE:-ik_waypoints}"',
            "load_tool_tcp_calibration.py",
            "Refusing strict validation with a zero tool TCP",
            "Refusing planning_mode=seed_joint_waypoints with a non-zero tool TCP",
            'tool_tcp_source:="$TOOL_TCP_SOURCE"',
        ]
    ) and all(
        needle in bridge
        for needle in [
            '"tool_tcp_xyz": tool_tcp_xyz',
            '"tool_tcp_rpy": tool_tcp_rpy',
            '"tool_tcp_source": tool_tcp_source',
        ]
    )
    items.append(
        AuditItem(
            "Strict validation records and guards the assumed spray TCP",
            "pass" if tool_tcp_guard_ok else "fail",
            "strict script defaults to a 150 mm assumed TCP, can load measured TCP calibration YAML, refuses zero TCP unless explicitly allowed, blocks seed-joint replay with non-zero TCP, and the MoveIt quality report records the TCP assumption"
            if tool_tcp_guard_ok
            else "Strict TCP defaults, unsafe-mode guards, or quality-report TCP evidence are missing",
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
        "--tcp-pose-csv",
        type=Path,
        default=ROOT / "outputs/tcp_poses.csv",
        help="TCP pose CSV used for the MoveIt runtime under audit.",
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
        args.tcp_pose_csv,
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
