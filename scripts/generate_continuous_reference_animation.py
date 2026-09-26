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
from src.tunnel_geometry import HorseshoeTunnel, TunnelConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the smooth FR5 reference-style GIF.")
    parser.add_argument("--ik", type=Path, default=ROOT / "outputs" / "ik_waypoints.csv")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs" / "animation_optimized.gif")
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--max-frames", type=int, default=180)
    parser.add_argument("--phase-start", type=int, default=30, help="Left-bottom start of the open arch.")
    parser.add_argument("--phase-end", type=int, default=210, help="Right-bottom end of the open arch.")
    args = parser.parse_args()

    with args.ik.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    q_all = np.asarray([[float(row[f"q{i}"]) for i in range(1, 7)] for row in rows])
    if not (0 <= args.phase_start < args.phase_end < min(240, len(q_all))):
        raise ValueError("phase range must satisfy 0 <= phase_start < phase_end < 240")
    q = q_all[args.phase_start : args.phase_end + 1]
    wrapped_steps = (np.diff(q, axis=0) + np.pi) % (2.0 * np.pi) - np.pi
    max_step_deg = float(np.rad2deg(np.max(np.abs(wrapped_steps))))

    robot, _ = load_official_fr5_robot(ROOT)
    robot = robot.with_base([0.0, 0.0, 0.20], yaw=np.pi)
    tcp_offset = np.asarray([0.0, 0.0, 0.150])
    joint_frames = [robot.all_joint_frames(qi) for qi in q]
    tcp = np.asarray(
        [frame[-1, :3, 3] + frame[-1, :3, :3] @ tcp_offset for frame in joint_frames]
    )
    wall_radius = 0.60
    wall_spring_height = 0.50
    min_wall_margin = float("inf")
    outside_frame_count = 0
    for frame, tcp_point in zip(joint_frames, tcp):
        points = np.vstack([frame[:, :3, 3], tcp_point])
        samples = np.vstack([np.linspace(a, b, 31) for a, b in zip(points[:-1], points[1:])])
        x = samples[:, 0]
        z = samples[:, 2]
        ceiling = wall_spring_height + np.sqrt(np.maximum(wall_radius**2 - x**2, 0.0))
        margin = np.minimum.reduce([x + wall_radius, wall_radius - x, z, ceiling - z])
        min_wall_margin = min(min_wall_margin, float(np.min(margin)))
        outside_frame_count += int(np.any(margin < 0.0))

    tunnel = HorseshoeTunnel(TunnelConfig(width=1.20, height=1.10, length=0.60, fillet_radius=0.20))
    grid, _ = tunnel.surface_grid(0.10, 0.20)
    step = max(1, len(q) // args.max_frames)
    frame_ids = np.arange(0, len(q), step)

    fig = plt.figure(figsize=(8, 6), facecolor="white")
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(grid[:, 0], grid[:, 1], grid[:, 2], s=1, alpha=0.14, c="#777777")
    ax.plot(tcp[:, 0], tcp[:, 1], tcp[:, 2], lw=1.25, color="#d62728", label="150 mm TCP path")
    arm, = ax.plot([], [], [], marker="o", markersize=5, lw=2.8, color="#222222", label="FR5")
    tool, = ax.plot([], [], [], lw=3.2, color="#7b1fa2", label="virtual TCP 150 mm")
    tcp_dot, = ax.plot([], [], [], marker="o", markersize=7, color="#d62728")
    trace, = ax.plot([], [], [], lw=1.8, color="#2e7d32", alpha=0.85, label="completed path")
    info = ax.text2D(
        0.02,
        0.96,
        "",
        transform=ax.transAxes,
        va="top",
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "#dadce0", "alpha": 0.92},
    )

    all_points = np.vstack([grid, tcp, np.vstack([frame[:, :3, 3] for frame in joint_frames])])
    mins, maxs = all_points.min(axis=0), all_points.max(axis=0)
    center = (mins + maxs) / 2.0
    radius = max(float(np.max(maxs - mins)) / 2.0, 0.62)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(0.0, center[2] + radius)
    ax.set_xlabel("X / m")
    ax.set_ylabel("Y / m")
    ax.set_zlabel("Z / m")
    ax.set_title("FR5 open arch motion | virtual TCP = 150 mm", pad=14)
    ax.view_init(elev=24, azim=-63)
    ax.legend(loc="upper right", fontsize=8)

    def update(source_index: int):
        frame = joint_frames[source_index]
        points = frame[:, :3, 3]
        flange = points[-1]
        tcp_point = tcp[source_index]
        arm.set_data(points[:, 0], points[:, 1])
        arm.set_3d_properties(points[:, 2])
        tool.set_data([flange[0], tcp_point[0]], [flange[1], tcp_point[1]])
        tool.set_3d_properties([flange[2], tcp_point[2]])
        tcp_dot.set_data([tcp_point[0]], [tcp_point[1]])
        tcp_dot.set_3d_properties([tcp_point[2]])
        trace.set_data(tcp[: source_index + 1, 0], tcp[: source_index + 1, 1])
        trace.set_3d_properties(tcp[: source_index + 1, 2])
        info.set_text(
            f"open-arch point {source_index:03d} / {len(q) - 1}\n"
            f"max adjacent joint step = {max_step_deg:.2f} deg"
        )
        return arm, tool, tcp_dot, trace, info

    animation = FuncAnimation(fig, update, frames=frame_ids, interval=1000 / args.fps, blit=False)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    try:
        animation.save(args.out, writer=PillowWriter(fps=args.fps))
    finally:
        plt.close(fig)
    validation = {
        "source": str(args.ik.resolve()),
        "source_phase_start": args.phase_start,
        "source_phase_end": args.phase_end,
        "path_shape": "left_vertical_then_arch_then_right_vertical_open_bottom",
        "waypoint_count": len(q),
        "animation_frame_count": len(frame_ids),
        "virtual_tcp_xyz_m": [0.0, 0.0, 0.15],
        "max_adjacent_joint_step_deg": max_step_deg,
        "simple_wall_outside_frame_count": outside_frame_count,
        "simple_wall_min_link_margin_m": min_wall_margin,
        "status": "pass" if max_step_deg <= 20.0 and outside_frame_count == 0 else "fail",
    }
    validation_path = args.out.with_name(args.out.stem + "_validation.json")
    validation_path.write_text(json.dumps(validation, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        f"Wrote {args.out.resolve()} with {len(frame_ids)} frames; "
        f"max_step_deg={max_step_deg:.6f}; min_wall_margin_m={min_wall_margin:.6f}"
    )


if __name__ == "__main__":
    main()
