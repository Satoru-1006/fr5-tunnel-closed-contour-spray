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


def write_final_quality_report(
    moveit_quality: Path,
    moveit_dynamics: Path,
    moveit_collision: Path,
    strict_audit: Path,
    out_csv: Path,
    out_json: Path,
    runtime_log: Path | None = None,
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
    final_status = _status_from_reports(quality, collision, audit)
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
        ("fk_tcp_speed_p05_m_s", quality.get("fk_tcp_speed_p05_m_s", "")),
        ("fk_tcp_speed_p95_m_s", quality.get("fk_tcp_speed_p95_m_s", "")),
        ("fk_tcp_speed_p05_p95_fluctuation", quality.get("fk_tcp_speed_p05_p95_fluctuation", "")),
        ("moveit_ruckig_smoothing_used", quality.get("moveit_ruckig_smoothing_used", "")),
        ("moveit_time_parameterization", quality.get("moveit_time_parameterization", "")),
        ("moveit_ee_link", quality.get("moveit_ee_link", "")),
        ("max_velocity_ratio", max_velocity_ratio),
        ("max_acceleration_ratio", max_acceleration_ratio),
        ("max_jerk_ratio", max_jerk_ratio),
        ("worst_jerk_joint", worst_jerk_joint),
        ("collision_environment_object_count", collision.get("collision_environment_object_count", "")),
        ("collision_checked_state_count", collision.get("collision_checked_state_count", "")),
        ("collision_count", collision.get("collision_count", "")),
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
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing MoveIt2 final report input(s): {missing}")

    write_final_quality_report(
        args.moveit_quality_report,
        args.moveit_dynamics_report,
        args.moveit_collision_report,
        args.strict_audit_json,
        args.out_dir / "final_quality_report.csv",
        args.out_dir / "final_acceptance_summary.json",
        args.runtime_log,
    )
    if args.write_visuals:
        write_final_visuals(args.moveit_trajectory_csv, args.out_dir, args.write_animation)
    print(f"Published final MoveIt2 outputs to {args.out_dir}")


if __name__ == "__main__":
    main()
