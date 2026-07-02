from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.metrics import (
    build_quality_report,
    closure_errors,
    coverage_metrics,
    polyline_distance,
    project_points_to_polyline,
    tcp_speed_from_positions,
)
from src.official_fairino import load_official_fr5_robot
from src.path_planner import (
    generate_closed_horseshoe_contour_path,
    generate_closed_horseshoe_path,
    generate_helical_path,
    generate_serpentine_path,
)
from src.robot_model import FR5Robot, make_tool_frame
from src.simulation import save_animation, save_closed_section_plot, save_coverage_heatmap, save_dynamics_plots, save_path_plot
from src.smoothing import cubic_arclength_resample, periodic_bspline_arclength_resample, smooth_and_resample, smooth_vectors
from src.time_parameterization import make_adaptive_ruckig_time_profile, make_jerk_limited_time_profile
from src.time_parameterization import TimeProfile
from src.tool_models import ContactTool, SprayTool
from src.tunnel_geometry import HorseshoeTunnel, TunnelConfig


def nearest_normals(points: np.ndarray, reference_points: np.ndarray, reference_normals: np.ndarray) -> np.ndarray:
    d2 = np.sum((points[:, None, :] - reference_points[None, :, :]) ** 2, axis=2)
    return reference_normals[np.argmin(d2, axis=1)]


def nearest_wall_and_normals(points: np.ndarray, wall_points: np.ndarray, wall_normals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Project points onto a closed wall polyline and interpolate its normals."""

    projected, projected_normals, _ = project_points_to_polyline(points, wall_points, wall_normals, closed=True)
    if projected_normals is None:
        raise RuntimeError("wall normal interpolation unexpectedly returned no normals")
    return projected, projected_normals


def fk_tcp_profile(robot: FR5Robot, profile: TimeProfile) -> tuple[TimeProfile, np.ndarray]:
    actual_positions = []
    actual_z = []
    for qi in profile.q:
        T = robot.fk(qi)
        actual_positions.append(T[:3, 3])
        actual_z.append(T[:3, 2])
    actual_positions = np.asarray(actual_positions)
    actual_z = np.asarray(actual_z)
    actual_speed = tcp_speed_from_positions(actual_positions, profile.t)
    actual_profile = TimeProfile(
        profile.t,
        actual_positions,
        actual_speed,
        profile.q,
        profile.dq,
        profile.ddq,
        profile.jerk,
        method=profile.method + "+fk",
    )
    return actual_profile, actual_z


def tool_quality(
    actual_positions: np.ndarray,
    actual_tool_z: np.ndarray,
    wall_points: np.ndarray,
    wall_normals: np.ndarray,
    stand_off: float,
) -> tuple[np.ndarray, np.ndarray]:
    nearest_wall, normals = nearest_wall_and_normals(actual_positions, wall_points, wall_normals)
    standoff_actual = np.sum((actual_positions - nearest_wall) * normals, axis=1)
    standoff_error = standoff_actual - stand_off
    dots = np.clip(np.sum(actual_tool_z * normals, axis=1), -1.0, 1.0)
    normal_angle_error_deg = np.rad2deg(np.arccos(dots))
    return normal_angle_error_deg, standoff_error


def nearest_distance_errors(tcp: np.ndarray, wall: np.ndarray, stand_off: float) -> tuple[float, float, float]:
    d = np.sqrt(np.sum((tcp[:, None, :] - wall[None, :, :]) ** 2, axis=2))
    err = np.min(d, axis=1) - stand_off
    return float(np.mean(err)), float(np.max(np.abs(err))), float(np.percentile(np.abs(err), 95))


def _segments_intersect(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> bool:
    def orient(p, q, r):
        return np.sign((q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0]))

    o1 = orient(a, b, c)
    o2 = orient(a, b, d)
    o3 = orient(c, d, a)
    o4 = orient(c, d, b)
    return bool(o1 != o2 and o3 != o4)


def contour_self_intersects(path: np.ndarray) -> bool:
    pts = path[:, [0, 2]]
    n = len(pts)
    for i in range(n):
        a, b = pts[i], pts[(i + 1) % n]
        for j in range(i + 2, n):
            if j == i or (j + 1) % n == i:
                continue
            c, d = pts[j], pts[(j + 1) % n]
            if _segments_intersect(a, b, c, d):
                return True
    return False


def tcp_inside_standard_horseshoe(tcp: np.ndarray, R: float, H: float) -> bool:
    x = tcp[:, 0]
    z = tcp[:, 2]
    in_box = (x >= -R - 1e-9) & (x <= R + 1e-9) & (z >= -1e-9)
    under_arch = np.where(np.abs(x) <= R, z <= H + np.sqrt(np.maximum(R * R - x * x, 0.0)) + 1e-9, False)
    return bool(np.all(in_box & under_arch))


def main() -> None:
    parser = argparse.ArgumentParser(description="FR5 horseshoe tunnel spray/contact simulation.")
    parser.add_argument("--tool", choices=["spray", "contact"], default="spray")
    parser.add_argument("--robot-model", choices=["official", "placeholder"], default="official")
    parser.add_argument("--path-mode", choices=["closed_horseshoe", "helical", "serpentine"], default="closed_horseshoe")
    parser.add_argument("--station-y", type=float, default=None)
    parser.add_argument("--loops", type=int, default=3)
    parser.add_argument("--samples-per-loop", type=int, default=240)
    parser.add_argument("--fillet-radius", type=float, default=0.20)
    parser.add_argument("--width", type=float, default=1.20)
    parser.add_argument("--height", type=float, default=1.10)
    parser.add_argument("--length", type=float, default=0.60)
    parser.add_argument("--speed", type=float, default=0.08)
    parser.add_argument("--overlap", type=float, default=0.50)
    parser.add_argument("--spray-distance", type=float, default=0.18)
    parser.add_argument("--spray-angle", type=float, default=35.0)
    parser.add_argument("--orientation-weight", type=float, default=0.05)
    parser.add_argument("--full-orientation-weight", type=float, default=0.0)
    parser.add_argument("--continuity-weight", type=float, default=0.018)
    parser.add_argument("--max-ik-step-deg", type=float, default=180.0)
    parser.add_argument("--max-path-deviation", type=float, default=0.005)
    parser.add_argument(
        "--joint-smoothness",
        type=float,
        default=1e-6,
        help="Periodic joint B-spline smoothing used before closed-contour Ruckig timing.",
    )
    parser.add_argument("--no-auto-overlap", action="store_true")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs")
    parser.add_argument("--no-animation", action="store_true")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    tunnel = HorseshoeTunnel(TunnelConfig(width=args.width, height=args.height, length=args.length, fillet_radius=args.fillet_radius))
    tool = SprayTool(spray_distance=args.spray_distance, spray_angle_deg=args.spray_angle) if args.tool == "spray" else ContactTool()
    robot_source = "placeholder"
    if args.robot_model == "official":
        try:
            robot, assets = load_official_fr5_robot(ROOT)
            robot = robot.with_base([0.0, 0.0, 0.20], yaw=np.pi)
            robot_source = f"official:{assets.urdf}"
        except Exception as exc:
            raise RuntimeError(
                "Official FR5 assets could not be loaded. Refusing to silently use placeholder DH; "
                "repair external/frcobot_ros2 or explicitly pass --robot-model placeholder for demo-only use."
            ) from exc
    else:
        robot = FR5Robot.placeholder().with_base([0.0, 0.0, 0.20], yaw=np.pi)

    warning = tunnel.warn_if_real_tunnel_exceeds_fr5()
    if warning:
        print(f"WARNING: {warning}")

    selected_overlap = args.overlap
    scan_rows = []
    if args.path_mode == "helical" and not args.no_auto_overlap:
        tool_candidates = [tool]
        if args.tool == "spray":
            tool_candidates = [
                SprayTool(spray_distance=float(distance), spray_angle_deg=float(angle))
                for angle in np.arange(30.0, 56.0, 5.0)
                for distance in (0.16, 0.18, 0.20, 0.22)
            ]
        for candidate_tool in tool_candidates:
            for overlap in np.arange(0.20, 0.56, 0.05):
                candidate = generate_helical_path(tunnel, candidate_tool, pass_overlap=float(overlap))
                cov, repeat, _ = coverage_metrics(tunnel, candidate.surface_points, candidate_tool.footprint_width)
                scan_rows.append(
                    {
                        "spray_width": float(candidate_tool.footprint_width),
                        "spray_distance": float(candidate_tool.spray_distance) if isinstance(candidate_tool, SprayTool) else 0.0,
                        "spray_angle": float(candidate_tool.spray_angle_deg) if isinstance(candidate_tool, SprayTool) else 0.0,
                        "overlap_ratio": float(overlap),
                        "pitch": float(candidate.pass_spacing),
                        "coverage_rate": cov,
                        "repeat_rate": repeat,
                    }
                )
        feasible = [row for row in scan_rows if row["coverage_rate"] >= 0.98]
        if feasible:
            selected = sorted(feasible, key=lambda row: (row["repeat_rate"], -row["coverage_rate"]))[0]
        else:
            selected = sorted(scan_rows, key=lambda row: (-row["coverage_rate"], row["repeat_rate"]))[0]
        selected_overlap = float(selected["overlap_ratio"])
        if args.tool == "spray":
            tool = SprayTool(spray_distance=float(selected["spray_distance"]), spray_angle_deg=float(selected["spray_angle"]))
        pd.DataFrame(scan_rows).to_csv(args.out / "overlap_scan.csv", index=False)

    if args.path_mode == "closed_horseshoe":
        R = args.width / 2.0
        H = args.height - R
        y0 = args.length * 0.5 if args.station_y is None else args.station_y
        stand_off_for_path = tool.spray_distance if isinstance(tool, SprayTool) else tool.stand_off
        raw_path = generate_closed_horseshoe_contour_path(
            R=R,
            H=H,
            y0=y0,
            stand_off=stand_off_for_path,
            n_loops=max(1, args.loops),
            samples_per_loop=args.samples_per_loop,
            fillet_radius=args.fillet_radius,
        )
        path_smoothness = 0.0
    elif args.path_mode == "helical":
        raw_path = generate_helical_path(tunnel, tool, pass_overlap=selected_overlap)
        path_smoothness = 0.0002
    else:
        raw_path = generate_serpentine_path(tunnel, tool, pass_overlap=selected_overlap)
        path_smoothness = 0.0
    if args.path_mode == "closed_horseshoe":
        smooth_tcp = raw_path.tcp_points
        normals = raw_path.normals
    elif args.path_mode == "helical":
        smooth_tcp, _ = cubic_arclength_resample(raw_path.tcp_points, spacing=0.018)
        normals = nearest_normals(smooth_tcp, raw_path.tcp_points, raw_path.normals)
        normals = smooth_vectors(normals)
    else:
        smooth_tcp, _ = smooth_and_resample(raw_path.tcp_points, spacing=0.020, smoothness=path_smoothness)
        normals = nearest_normals(smooth_tcp, raw_path.tcp_points, raw_path.normals)
        normals = smooth_vectors(normals)
    stand_off = tool.spray_distance if isinstance(tool, SprayTool) else tool.stand_off
    smooth_surface = smooth_tcp - stand_off * normals
    poses = [make_tool_frame(p, n) for p, n in zip(smooth_tcp, normals)]

    pose_rows = []
    base_pose_rows = []
    world_to_base = robot.base_inv
    R_world_to_base = world_to_base[:3, :3]
    for p, n, T in zip(smooth_tcp, normals, poses):
        quat = Rotation.from_matrix(T[:3, :3]).as_quat()
        T_base = world_to_base @ T
        n_base = R_world_to_base @ n
        n_base /= max(float(np.linalg.norm(n_base)), 1e-12)
        quat_base = Rotation.from_matrix(T_base[:3, :3]).as_quat()
        pose_rows.append(
            {
                "x": p[0],
                "y": p[1],
                "z": p[2],
                "qx": quat[0],
                "qy": quat[1],
                "qz": quat[2],
                "qw": quat[3],
                "nx": n[0],
                "ny": n[1],
                "nz": n[2],
            }
        )
        base_pose_rows.append(
            {
                "x": T_base[0, 3],
                "y": T_base[1, 3],
                "z": T_base[2, 3],
                "qx": quat_base[0],
                "qy": quat_base[1],
                "qz": quat_base[2],
                "qw": quat_base[3],
                "nx": n_base[0],
                "ny": n_base[1],
                "nz": n_base[2],
            }
        )
    pd.DataFrame(pose_rows).to_csv(args.out / "tcp_poses.csv", index=False)
    pd.DataFrame(base_pose_rows).to_csv(args.out / "tcp_poses_base_link.csv", index=False)

    q, ik_errors, ik_success = robot.solve_trajectory_ik(
        poses,
        max_step_deg=args.max_ik_step_deg,
        orientation_weight=args.orientation_weight,
        continuity_weight=args.continuity_weight,
        full_orientation_weight=args.full_orientation_weight,
    )
    time_profile_note = ""
    try:
        profile = make_adaptive_ruckig_time_profile(
            smooth_tcp,
            q,
            robot.limits,
            target_tcp_speed=args.speed,
            closed_loop=args.path_mode == "closed_horseshoe",
            fk_position=lambda qi: robot.fk(qi)[:3, 3],
            joint_smoothness=args.joint_smoothness,
        )
    except Exception as exc:
        time_profile_note = f"Ruckig failed, fallback used: {exc}"
        print(f"WARNING: {time_profile_note}")
        profile = make_jerk_limited_time_profile(smooth_tcp, q, robot.limits, target_tcp_speed=args.speed)
    actual_profile, actual_tool_z = fk_tcp_profile(robot, profile)
    normal_angle_error_deg, standoff_error = tool_quality(
        actual_profile.path_points,
        actual_tool_z,
        smooth_surface,
        normals,
        stand_off,
    )
    loop_count = max(1, args.loops) if args.path_mode == "closed_horseshoe" else 1
    actual_samples_per_loop = len(smooth_tcp) // loop_count
    one_loop_tcp = smooth_tcp[:actual_samples_per_loop]
    one_loop_wall = smooth_surface[:actual_samples_per_loop]
    contour_standoff_mean, contour_standoff_max, contour_standoff_p95 = nearest_distance_errors(one_loop_tcp, one_loop_wall, stand_off)
    tcp_self_intersects = contour_self_intersects(one_loop_tcp) if args.path_mode == "closed_horseshoe" else False
    tcp_inside = (
        tcp_inside_standard_horseshoe(one_loop_tcp, args.width / 2.0, args.height - args.width / 2.0)
        if args.path_mode == "closed_horseshoe"
        else True
    )
    report = build_quality_report(
        tunnel,
        smooth_surface,
        tool.footprint_width,
        actual_profile.tcp_speed,
        profile.q,
        profile.ddq,
        profile.jerk,
        ik_success,
        ik_errors,
        t=profile.t,
        section_only=args.path_mode == "closed_horseshoe",
        jerk_limits=robot.limits.jerk_max,
    )
    speed_p05 = float(np.percentile(actual_profile.tcp_speed, 5))
    speed_p95 = float(np.percentile(actual_profile.tcp_speed, 95))
    speed_robust_mean = max(float(np.mean(actual_profile.tcp_speed)), 1e-12)
    speed_robust_fluctuation = float((speed_p95 - speed_p05) / speed_robust_mean)
    speed_minmax_fluctuation = float(
        (np.max(actual_profile.tcp_speed) - np.min(actual_profile.tcp_speed)) / speed_robust_mean
    )
    target_tcp_for_distance = one_loop_tcp if args.path_mode == "closed_horseshoe" else smooth_tcp
    fk_path_deviation = polyline_distance(
        actual_profile.path_points,
        target_tcp_for_distance,
        closed=args.path_mode == "closed_horseshoe",
    )
    fk_path_deviation_p95_mm = float(np.percentile(fk_path_deviation, 95) * 1000.0)
    fk_path_deviation_max_mm = float(np.max(fk_path_deviation) * 1000.0)
    fk_path_deviation_limit_mm = float(args.max_path_deviation * 1000.0)
    fk_path_deviation_status = "pass" if fk_path_deviation_max_mm <= fk_path_deviation_limit_mm else "fail"
    normal_angle_mean = float(np.mean(normal_angle_error_deg))
    normal_angle_max = float(np.max(normal_angle_error_deg))
    normal_angle_p95 = float(np.percentile(normal_angle_error_deg, 95))
    standoff_p95_abs_mm = float(np.percentile(np.abs(standoff_error) * 1000.0, 95))
    standoff_max_abs_mm = float(np.max(np.abs(standoff_error)) * 1000.0)
    standoff_limit_mm = stand_off * 0.05 * 1000.0
    speed_status = "pass" if report.tcp_speed_fluctuation <= 0.05 else "fail"
    speed_p95_status = "pass" if speed_robust_fluctuation <= 0.05 else "fail"
    standoff_status = "pass" if standoff_max_abs_mm <= standoff_limit_mm else "fail"
    normal_status = "pass" if normal_angle_mean <= 5.0 and normal_angle_max <= 10.0 else "fail"
    jerk_ratio = report.jerk_abs_max / np.maximum(report.jerk_p95, 1e-12)
    jerk_status = "pass" if len(report.spike_indices) == 0 else "fail"
    fk_speed_mean = float(np.mean(actual_profile.tcp_speed))
    fk_speed_min = float(np.min(actual_profile.tcp_speed))
    fk_speed_max = float(np.max(actual_profile.tcp_speed))
    fk_speed_p05 = float(np.percentile(actual_profile.tcp_speed, 5))
    fk_speed_p95 = float(np.percentile(actual_profile.tcp_speed, 95))
    fk_speed_minmax_fluctuation = (
        (fk_speed_max - fk_speed_min) / max(fk_speed_mean, 1e-12) if fk_speed_mean > 0.0 else 0.0
    )
    fk_speed_p05_p95_fluctuation = (
        (fk_speed_p95 - fk_speed_p05) / max(fk_speed_mean, 1e-12)
        if fk_speed_mean > 0.0
        else 0.0
    )
    fk_speed_status = "pass" if fk_speed_p05_p95_fluctuation <= 0.05 else "fail"
    fk_speed_p05_p95_status = "pass" if fk_speed_p05_p95_fluctuation <= 0.05 else "fail"
    overall_status = (
        "pass"
        if all(
            status == "pass"
            for status in (
                speed_p95_status,
                fk_speed_p05_p95_status,
                standoff_status,
                normal_status,
                fk_path_deviation_status,
                jerk_status,
            )
        )
        else "fail"
    )
    for detail in report.spike_details:
        sample_index = int(detail["path_index"])
        actual_point = actual_profile.path_points[sample_index]
        detail["path_index"] = int(np.argmin(np.linalg.norm(smooth_tcp - actual_point, axis=1)))

    pd.DataFrame(smooth_surface, columns=["x", "y", "z"]).to_csv(args.out / "surface_path.csv", index=False)
    pd.DataFrame(smooth_tcp, columns=["x", "y", "z"]).to_csv(args.out / "tcp_path.csv", index=False)
    ik_columns: dict[str, np.ndarray] = {"waypoint": np.arange(len(q), dtype=int)}
    for joint in range(6):
        ik_columns[f"q{joint + 1}"] = q[:, joint]
    pd.DataFrame(ik_columns).to_csv(args.out / "ik_waypoints.csv", index=False)
    pd.DataFrame(actual_profile.path_points, columns=["x", "y", "z"]).assign(
        t=actual_profile.t,
        tcp_speed=actual_profile.tcp_speed,
    ).to_csv(
        args.out / "actual_tcp_path_fk.csv", index=False
    )
    joint_columns: dict[str, np.ndarray] = {"t": profile.t}
    for joint in range(6):
        joint_columns[f"q{joint + 1}"] = profile.q[:, joint]
        joint_columns[f"dq{joint + 1}"] = profile.dq[:, joint]
        joint_columns[f"ddq{joint + 1}"] = profile.ddq[:, joint]
        joint_columns[f"jerk{joint + 1}"] = profile.jerk[:, joint]
    pd.DataFrame(joint_columns).to_csv(args.out / "joint_trajectory.csv", index=False)
    pd.DataFrame(
        {
            "metric": [
                "coverage_rate",
                "repeat_rate",
                "status",
                "tcp_speed_fluctuation",
                "tcp_speed_minmax_fluctuation",
                "ik_success_rate",
                "max_ik_error",
                "joint_jump_count",
                "spike_count",
                "jerk_status",
                "max_jerk_to_p95_ratio",
                "path_pitch_or_pass_spacing",
                "selected_overlap_ratio",
                "recommended_spray_width",
                "recommended_spray_distance",
                "recommended_spray_angle",
                "mean_speed",
                "min_speed",
                "max_speed",
                "speed_p05",
                "speed_p95",
                "speed_p05_p95_fluctuation",
                "tcp_speed_status",
                "tcp_speed_p05_p95_status",
                "fk_tcp_speed_mean",
                "fk_tcp_speed_min",
                "fk_tcp_speed_max",
                "fk_tcp_speed_fluctuation",
                "fk_tcp_speed_minmax_fluctuation",
                "fk_tcp_speed_p05",
                "fk_tcp_speed_p95",
                "fk_tcp_speed_p05_p95_fluctuation",
                "fk_tcp_speed_status",
                "fk_tcp_speed_p05_p95_status",
                "fk_path_deviation_p95_mm",
                "fk_path_deviation_max_mm",
                "fk_path_deviation_limit_mm",
                "fk_path_deviation_status",
                "requested_speed",
                "time_profile_method",
                "time_profile_note",
                "robot_model_source",
                "ik_orientation_weight",
                "ik_full_orientation_weight",
                "ik_continuity_weight",
                "ik_max_step_deg",
                "joint_smoothness",
                "loops",
                "station_y_std",
                "normal_angle_error_mean_deg",
                "normal_angle_error_p95_deg",
                "normal_angle_error_max_deg",
                "normal_angle_status",
                "standoff_error_mean_mm",
                "standoff_error_p95_abs_mm",
                "standoff_error_max_abs_mm",
                "standoff_error_limit_mm",
                "standoff_status",
                "samples_per_loop",
                "actual_samples_per_loop",
                "fillet_radius",
                "contour_standoff_error_mean_mm",
                "contour_standoff_error_max_abs_mm",
                "contour_standoff_error_p95_abs_mm",
                "tcp_self_intersects",
                "tcp_inside_wall_contour",
            ],
            "value": [
                report.coverage_rate,
                report.repeat_rate,
                overall_status,
                report.tcp_speed_fluctuation,
                speed_minmax_fluctuation,
                report.ik_success_rate,
                report.max_ik_error,
                len(report.jump_indices),
                len(report.spike_indices),
                jerk_status,
                float(np.max(jerk_ratio)),
                raw_path.pass_spacing,
                selected_overlap,
                tool.footprint_width,
                tool.spray_distance if isinstance(tool, SprayTool) else 0.0,
                tool.spray_angle_deg if isinstance(tool, SprayTool) else 0.0,
                report.speed_mean,
                report.speed_min,
                report.speed_max,
                speed_p05,
                speed_p95,
                speed_robust_fluctuation,
                speed_status,
                speed_p95_status,
                fk_speed_mean,
                fk_speed_min,
                fk_speed_max,
                fk_speed_p05_p95_fluctuation,
                fk_speed_minmax_fluctuation,
                fk_speed_p05,
                fk_speed_p95,
                fk_speed_p05_p95_fluctuation,
                fk_speed_status,
                fk_speed_p05_p95_status,
                fk_path_deviation_p95_mm,
                fk_path_deviation_max_mm,
                fk_path_deviation_limit_mm,
                fk_path_deviation_status,
                args.speed,
                profile.method,
                time_profile_note,
                robot_source,
                args.orientation_weight,
                args.full_orientation_weight,
                args.continuity_weight,
                args.max_ik_step_deg,
                args.joint_smoothness,
                loop_count,
                float(np.std(smooth_tcp[:, 1])),
                normal_angle_mean,
                normal_angle_p95,
                normal_angle_max,
                normal_status,
                float(np.mean(standoff_error) * 1000.0),
                standoff_p95_abs_mm,
                standoff_max_abs_mm,
                standoff_limit_mm,
                standoff_status,
                args.samples_per_loop if args.path_mode == "closed_horseshoe" else 0,
                actual_samples_per_loop if args.path_mode == "closed_horseshoe" else 0,
                args.fillet_radius if args.path_mode == "closed_horseshoe" else 0.0,
                contour_standoff_mean * 1000.0,
                contour_standoff_max * 1000.0,
                contour_standoff_p95 * 1000.0,
                int(tcp_self_intersects),
                int(tcp_inside),
            ],
        }
    ).to_csv(args.out / "quality_report.csv", index=False)
    close = closure_errors(smooth_tcp, actual_samples_per_loop if args.path_mode == "closed_horseshoe" else len(smooth_tcp))
    close_rows = pd.DataFrame([{"metric": key, "value": value} for key, value in close.items()])
    old_report = pd.read_csv(args.out / "quality_report.csv")
    pd.concat([old_report, close_rows], ignore_index=True).to_csv(args.out / "quality_report.csv", index=False)
    pd.DataFrame(
        {
            "t": profile.t,
            "actual_tcp_x": actual_profile.path_points[:, 0],
            "actual_tcp_y": actual_profile.path_points[:, 1],
            "actual_tcp_z": actual_profile.path_points[:, 2],
            "actual_tcp_speed": actual_profile.tcp_speed,
            "normal_angle_error_deg": normal_angle_error_deg,
            "standoff_error_mm": standoff_error * 1000.0,
        }
    ).to_csv(args.out / "tool_pose_error.csv", index=False)
    pd.DataFrame(
        [
            {"joint": i + 1, "max_jerk": report.jerk_abs_max[i], "p95_jerk": report.jerk_p95[i], "p99_jerk": report.jerk_p99[i]}
            for i in range(len(report.jerk_abs_max))
        ]
    ).to_csv(args.out / "joint_jerk_report.csv", index=False)
    pd.DataFrame(report.spike_details).to_csv(args.out / "spike_details.csv", index=False)

    save_path_plot(tunnel, smooth_surface, smooth_tcp, args.out / "path_3d.png")
    if args.path_mode == "closed_horseshoe":
        save_closed_section_plot(smooth_surface, smooth_tcp, args.out / "closed_contour_section.png")
    save_dynamics_plots(actual_profile, args.out / "dynamics.png")
    save_coverage_heatmap(tunnel, smooth_surface, tool.footprint_width, args.out / "coverage_heatmap.png")
    if not args.no_animation:
        save_animation(robot, actual_profile, tunnel, args.out / "animation.gif")

    print("Simulation complete")
    print(f"path mode={args.path_mode}, overlap={selected_overlap:.2f}, pitch/pass spacing={raw_path.pass_spacing:.4f} m")
    print(f"coverage={report.coverage_rate:.3%}, repeat={report.repeat_rate:.3%}")
    print(
        f"TCP speed mean/min/max={report.speed_mean:.5f}/{report.speed_min:.5f}/{report.speed_max:.5f} m/s, "
        f"fluctuation={report.tcp_speed_fluctuation:.3%}"
    )
    print(f"IK success={report.ik_success_rate:.3%}, max IK error={report.max_ik_error:.4f}")
    print(f"joint jumps={len(report.jump_indices)}, spikes={len(report.spike_indices)}")
    print(
        f"normal angle mean/max={normal_angle_mean:.2f}/{normal_angle_max:.2f} deg, "
        f"stand-off max abs={standoff_max_abs_mm:.2f} mm"
    )
    print(f"FK path deviation p95/max={fk_path_deviation_p95_mm:.2f}/{fk_path_deviation_max_mm:.2f} mm")
    print(f"TCP speed p05/p95 fluctuation={speed_robust_fluctuation:.3%}")
    print(f"time profile={profile.method}")
    for note in report.notes:
        print(f"NOTE: {note}")
    print(f"Outputs: {args.out}")


if __name__ == "__main__":
    main()
