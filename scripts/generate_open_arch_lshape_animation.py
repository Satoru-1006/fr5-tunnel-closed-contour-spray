from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.official_fairino import load_official_fr5_robot


def read_xyz(path: Path, phase_start: int, phase_end: int) -> np.ndarray:
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    points = np.asarray([[float(row[key]) for key in ("x", "y", "z")] for row in rows])
    return points[phase_start : phase_end + 1]


def wall_margin(points: np.ndarray, radius: float = 0.60, spring_height: float = 0.50) -> float:
    samples = np.vstack([np.linspace(a, b, 31) for a, b in zip(points[:-1], points[1:])])
    x = samples[:, 0]
    z = samples[:, 2]
    ceiling = spring_height + np.sqrt(np.maximum(radius**2 - x**2, 0.0))
    return float(np.min(np.minimum.reduce([x + radius, radius - x, z, ceiling - z])))


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a planar open-arch animation with an unfolded FR5 posture.")
    parser.add_argument("--surface-path", type=Path, default=ROOT / "outputs" / "surface_path.csv")
    parser.add_argument("--tcp-path", type=Path, default=ROOT / "outputs" / "tcp_path.csv")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs" / "animation_open_arch_lshape.gif")
    parser.add_argument("--phase-start", type=int, default=30)
    parser.add_argument("--phase-end", type=int, default=210)
    parser.add_argument("--base-y", type=float, default=0.45)
    parser.add_argument("--work-plane-y", type=float, default=0.55)
    parser.add_argument("--tunnel-length", type=float, default=0.90)
    parser.add_argument("--fps", type=int, default=24)
    args = parser.parse_args()

    surface_targets = read_xyz(args.surface_path, args.phase_start, args.phase_end)
    tcp_targets = read_xyz(args.tcp_path, args.phase_start, args.phase_end)
    surface_targets[:, 1] = args.work_plane_y
    tcp_targets[:, 1] = args.work_plane_y
    model, _ = load_official_fr5_robot(ROOT)
    robot = model.with_base([0.0, args.base_y, 0.20], yaw=np.pi)
    tool = np.eye(4)
    tool[:3, 3] = [0.0, 0.0, 0.150]
    tool_inv = np.linalg.inv(tool)

    seed = np.deg2rad([101.4, -37.1, 66.7, 52.3, 55.0, 13.9])
    q_rows: list[np.ndarray] = []
    ik_errors: list[float] = []
    wall_directions: list[np.ndarray] = []
    for surface_point, tcp_point in zip(surface_targets, tcp_targets):
        tool_z = surface_point - tcp_point
        tool_z /= np.linalg.norm(tool_z)
        x_axis = np.cross(np.asarray([0.0, 1.0, 0.0]), tool_z)
        x_axis /= np.linalg.norm(x_axis)
        y_axis = np.cross(tool_z, x_axis)
        target_tcp = np.eye(4)
        target_tcp[:3, :3] = np.column_stack([x_axis, y_axis, tool_z])
        target_tcp[:3, 3] = tcp_point
        target_flange = target_tcp @ tool_inv
        q, error, ok = robot.ik(
            target_flange,
            seed,
            orientation_weight=0.40,
            continuity_weight=0.004,
            full_orientation_weight=0.0,
        )
        if not ok:
            raise RuntimeError(f"IK failed at open-arch point {len(q_rows)} with error={error:.6g}")
        q_rows.append(q)
        ik_errors.append(float(error))
        wall_directions.append(tool_z)
        seed = q

    raw_q_path = np.asarray(q_rows)
    raw_target_tcp = np.asarray(tcp_targets)
    raw_wall_directions = np.asarray(wall_directions)
    raw_actual_tcp = np.asarray([(robot.fk(q) @ tool)[:3, 3] for q in raw_q_path])
    raw_s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(raw_actual_tcp, axis=0), axis=1))]
    uniform_s = np.linspace(0.0, raw_s[-1], len(raw_q_path))
    q_path = np.column_stack([np.interp(uniform_s, raw_s, raw_q_path[:, j]) for j in range(6)])
    target_tcp = np.column_stack([np.interp(uniform_s, raw_s, raw_target_tcp[:, j]) for j in range(3)])
    wall_directions_array = np.column_stack(
        [np.interp(uniform_s, raw_s, raw_wall_directions[:, j]) for j in range(3)]
    )
    wall_directions_array /= np.maximum(np.linalg.norm(wall_directions_array, axis=1, keepdims=True), 1e-12)
    joint_frames = [robot.all_joint_frames(q) for q in q_path]
    actual_tcp = np.asarray([(robot.fk(q) @ tool)[:3, 3] for q in q_path])
    actual_tool_z = np.asarray([(robot.fk(q) @ tool)[:3, 2] for q in q_path])
    wall_angle_error_deg = np.rad2deg(
        np.arccos(np.clip(np.sum(actual_tool_z * wall_directions_array, axis=1), -1.0, 1.0))
    )
    wrapped_steps = (np.diff(q_path, axis=0) + np.pi) % (2.0 * np.pi) - np.pi
    max_step_deg = float(np.rad2deg(np.max(np.abs(wrapped_steps))))
    tcp_step = np.linalg.norm(np.diff(actual_tcp, axis=0), axis=1)
    tcp_step_p05 = float(np.percentile(tcp_step, 5))
    tcp_step_p95 = float(np.percentile(tcp_step, 95))
    tcp_step_fluctuation = float((tcp_step_p95 - tcp_step_p05) / max(float(np.mean(tcp_step)), 1e-12))

    tunnel = {
        "width": 1.20,
        "height": 1.10,
        "length": args.tunnel_length,
    }
    radius = tunnel["width"] / 2.0
    spring_height = tunnel["height"] - radius
    link_radius = 0.04
    all_points = np.vstack([frame[:, :3, 3] for frame in joint_frames] + [actual_tcp])
    x_all, y_all, z_all = all_points[:, 0], all_points[:, 1], all_points[:, 2]
    ceiling = spring_height + np.sqrt(np.maximum(radius**2 - np.minimum(np.abs(x_all), radius) ** 2, 0.0))
    inside_samples = (
        (y_all >= link_radius)
        & (y_all <= tunnel["length"] - link_radius)
        & (x_all >= -radius + link_radius)
        & (x_all <= radius - link_radius)
        & (z_all >= link_radius)
        & (z_all <= ceiling - link_radius)
    )
    def conservative_margin(points: np.ndarray) -> float:
        samples = np.vstack(
            [np.linspace(a, b, 41) for a, b in zip(points[:-1], points[1:])]
        )
        x, y, z = samples[:, 0], samples[:, 1], samples[:, 2]
        roof = spring_height + np.sqrt(np.maximum(radius**2 - np.minimum(np.abs(x), radius) ** 2, 0.0))
        return float(np.min([x + radius, radius - x, y, tunnel["length"] - y, z, roof - z]))

    sampled_clearances = [
        conservative_margin(np.vstack([frame[:, :3, 3], tcp_point]))
        for frame, tcp_point in zip(joint_frames, actual_tcp)
    ]
    outside_frame_count = sum(clearance < link_radius for clearance in sampled_clearances)

    bend_angles = []
    margins = []
    for frame, tcp_point in zip(joint_frames, actual_tcp):
        points = np.vstack([frame[:, :3, 3], tcp_point])
        upper = points[3] - points[2]
        fore = points[4] - points[3]
        bend_angles.append(
            float(np.rad2deg(np.arccos(np.clip(np.dot(upper, fore) / (np.linalg.norm(upper) * np.linalg.norm(fore)), -1.0, 1.0))))
        )
        margins.append(wall_margin(points))

    ik_csv = args.out.with_name(args.out.stem + "_ik.csv")
    ik_csv.parent.mkdir(parents=True, exist_ok=True)
    with ik_csv.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["point", *[f"j{i}_q" for i in range(1, 7)], "tcp_x", "tcp_y", "tcp_z"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for index, (q, tcp_point) in enumerate(zip(q_path, actual_tcp)):
            writer.writerow(
                {
                    "point": index,
                    **{f"j{i + 1}_q": float(q[i]) for i in range(6)},
                    "tcp_x": float(tcp_point[0]),
                    "tcp_y": float(tcp_point[1]),
                    "tcp_z": float(tcp_point[2]),
                }
            )

    fig = plt.figure(figsize=(8.8, 7.0), facecolor="white")
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(target_tcp[:, 0], target_tcp[:, 1], target_tcp[:, 2], color="#d62728", lw=2.2, label="planar open arch")
    arm, = ax.plot([], [], [], marker="o", markersize=5, lw=3.2, color="#202124", label="FR5 main arm")
    wrist_line, = ax.plot([], [], [], lw=2.4, color="#5f6368", label="wrist housing")
    tool_line, = ax.plot([], [], [], lw=4.0, color="#7b1fa2", label="flange to TCP 150 mm")
    spray_axis, = ax.plot([], [], [], lw=2.2, linestyle="--", color="#1565c0", label="TCP points to wall")
    tcp_dot, = ax.plot([], [], [], marker="o", markersize=7, color="#d62728")
    trace, = ax.plot([], [], [], lw=2.4, color="#2e7d32", label="completed path")
    info = ax.text2D(
        0.02,
        0.97,
        "",
        transform=ax.transAxes,
        va="top",
        fontsize=10,
        bbox={"facecolor": "white", "edgecolor": "#dadce0", "alpha": 0.94},
    )
    ax.set_xlim(-0.72, 0.72)
    ax.set_ylim(0.0, max(0.90, args.tunnel_length))
    ax.set_zlim(0.0, 1.16)
    ax.set_xlabel("X / m")
    ax.set_ylabel("Y / m")
    ax.set_zlabel("Z / m")
    ax.set_title("FR5 planar open arch | base set back for unfolded reach | TCP 150 mm", pad=14)
    ax.view_init(elev=24, azim=-63)
    ax.legend(loc="upper right", fontsize=8)

    def update(index: int):
        points = joint_frames[index][:, :3, 3]
        # Keep the true flange-to-TCP segment separate: its direction is the
        # constrained spray direction and must not be replaced by a visual shortcut.
        main_points = points[:5]
        wrist_anchor = points[4]
        flange = points[-1]
        tcp_point = actual_tcp[index]
        arm.set_data(main_points[:, 0], main_points[:, 1])
        arm.set_3d_properties(main_points[:, 2])
        wrist_line.set_data([wrist_anchor[0], flange[0]], [wrist_anchor[1], flange[1]])
        wrist_line.set_3d_properties([wrist_anchor[2], flange[2]])
        tool_line.set_data([flange[0], tcp_point[0]], [flange[1], tcp_point[1]])
        tool_line.set_3d_properties([flange[2], tcp_point[2]])
        wall_tip = tcp_point + wall_directions_array[index] * 0.10
        spray_axis.set_data([tcp_point[0], wall_tip[0]], [tcp_point[1], wall_tip[1]])
        spray_axis.set_3d_properties([tcp_point[2], wall_tip[2]])
        tcp_dot.set_data([tcp_point[0]], [tcp_point[1]])
        tcp_dot.set_3d_properties([tcp_point[2]])
        trace.set_data(actual_tcp[: index + 1, 0], actual_tcp[: index + 1, 1])
        trace.set_3d_properties(actual_tcp[: index + 1, 2])
        info.set_text(
            f"point {index:03d} / {len(q_path) - 1}\n"
            f"TCP plane Y = {args.work_plane_y:.2f} m | max joint step = {max_step_deg:.2f} deg"
        )
        return arm, wrist_line, tool_line, spray_axis, tcp_dot, trace, info

    animation = FuncAnimation(fig, update, frames=len(q_path), interval=1000 / args.fps, blit=False)
    try:
        animation.save(args.out, writer=PillowWriter(fps=args.fps))
    finally:
        plt.close(fig)

    max_plane_error = float(np.max(np.abs(actual_tcp[:, 1] - args.work_plane_y)))
    validation = {
        "surface_path_source": str(args.surface_path.resolve()),
        "tcp_path_source": str(args.tcp_path.resolve()),
        "path_shape": "left_vertical_then_arch_then_right_vertical_open_bottom",
        "base_xyz_m": [0.0, args.base_y, 0.20],
        "work_plane_y_m": args.work_plane_y,
        "tunnel_length_m": args.tunnel_length,
        "conservative_link_radius_m": link_radius,
        "minimum_sampled_centerline_clearance_m": min(sampled_clearances),
        "virtual_tcp_xyz_m": [0.0, 0.0, 0.15],
        "waypoint_count": len(q_path),
        "ik_success_count": len(q_path),
        "ik_error_max": max(ik_errors),
        "tcp_plane_error_max_m": max_plane_error,
        "tcp_wall_direction_error_max_deg": float(np.max(wall_angle_error_deg)),
        "tcp_wall_direction_error_mean_deg": float(np.mean(wall_angle_error_deg)),
        "max_adjacent_joint_step_deg": max_step_deg,
        "tcp_step_p05_m": tcp_step_p05,
        "tcp_step_p95_m": tcp_step_p95,
        "tcp_step_fluctuation_ratio": tcp_step_fluctuation,
        "main_arm_bend_angle_min_deg": min(bend_angles),
        "main_arm_bend_angle_max_deg": max(bend_angles),
        "simple_wall_outside_frame_count": sum(margin < 0.0 for margin in margins),
        "simple_wall_min_link_margin_m": min(margins),
        "container_outside_frame_count": int(outside_frame_count),
        "container_inside_sample_count": int(np.sum(inside_samples)),
        "container_total_sample_count": int(len(all_points)),
        "status": "pass" if max_step_deg <= 20.0 and min(margins) >= 0.0 and max_plane_error <= 0.003 and np.max(wall_angle_error_deg) <= 1.0 and outside_frame_count == 0 else "fail",
    }
    validation_path = args.out.with_name(args.out.stem + "_validation.json")
    validation_path.write_text(json.dumps(validation, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        f"Wrote {args.out.resolve()}; max_step_deg={max_step_deg:.6f}; "
        f"plane_error_mm={max_plane_error * 1000.0:.3f}; min_wall_margin_m={min(margins):.6f}"
    )


if __name__ == "__main__":
    main()
