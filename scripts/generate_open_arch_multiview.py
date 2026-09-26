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


VIEWS = {
    "left_front": (24, -63, "Left-front view"),
    "front": (18, -90, "Front view"),
    "right_front": (24, -117, "Right-front view"),
}


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> None:
    parser = argparse.ArgumentParser(description="Render the true six-axis open-arch motion from three viewpoints.")
    parser.add_argument("--ik", type=Path, default=ROOT / "outputs" / "animation_open_arch_lshape_ik.csv")
    parser.add_argument("--surface", type=Path, default=ROOT / "outputs" / "surface_path.csv")
    parser.add_argument("--tcp", type=Path, default=ROOT / "outputs" / "tcp_path.csv")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "outputs")
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--base-y", type=float, default=0.45)
    parser.add_argument("--work-plane-y", type=float, default=0.55)
    parser.add_argument("--tunnel-length", type=float, default=0.90)
    args = parser.parse_args()

    ik = rows(args.ik)
    q = np.asarray([[float(row[f"j{i}_q"]) for i in range(1, 7)] for row in ik])
    actual_tcp = np.asarray([[float(row[key]) for key in ("tcp_x", "tcp_y", "tcp_z")] for row in ik])
    start_phase = 30
    end_phase = start_phase + len(q)
    wall = np.asarray([[float(row[key]) for key in ("x", "y", "z")] for row in rows(args.surface)][start_phase:end_phase])
    target = np.asarray([[float(row[key]) for key in ("x", "y", "z")] for row in rows(args.tcp)][start_phase:end_phase])
    wall[:, 1] = args.work_plane_y
    target[:, 1] = args.work_plane_y
    direction = wall - target
    direction /= np.maximum(np.linalg.norm(direction, axis=1, keepdims=True), 1e-12)

    model, _ = load_official_fr5_robot(ROOT)
    robot = model.with_base([0.0, args.base_y, 0.20], yaw=np.pi)
    frames = [robot.all_joint_frames(qi)[:, :3, 3] for qi in q]
    tunnel = HorseshoeTunnel(TunnelConfig(width=1.20, height=1.10, length=args.tunnel_length, fillet_radius=0.20))
    grid, _ = tunnel.surface_grid(0.10, 0.20)
    max_step = float(np.rad2deg(np.max(np.abs((np.diff(q, axis=0) + np.pi) % (2 * np.pi) - np.pi))))
    q_range_deg = np.rad2deg(np.ptp(q, axis=0))
    tcp_step = np.linalg.norm(np.diff(actual_tcp, axis=0), axis=1)
    tcp_step_p05 = float(np.percentile(tcp_step, 5))
    tcp_step_p95 = float(np.percentile(tcp_step, 95))
    tcp_step_fluctuation = float((tcp_step_p95 - tcp_step_p05) / max(float(np.mean(tcp_step)), 1e-12))
    radius = tunnel.width / 2.0
    spring_z = tunnel.height - radius
    link_radius = 0.04
    all_joint_points = np.vstack(frames)
    x = all_joint_points[:, 0]
    z = all_joint_points[:, 2]
    roof = spring_z + np.sqrt(np.maximum(radius**2 - np.minimum(np.abs(x), radius) ** 2, 0.0))
    y = all_joint_points[:, 1]
    inside_mask = (
        (y >= link_radius) & (y <= args.tunnel_length - link_radius)
        & (x >= -radius + link_radius) & (x <= radius - link_radius)
        & (z >= link_radius) & (z <= roof - link_radius)
    )

    def conservative_margin(points: np.ndarray) -> float:
        samples = np.vstack([np.linspace(a, b, 41) for a, b in zip(points[:-1], points[1:])])
        sx, sy, sz = samples[:, 0], samples[:, 1], samples[:, 2]
        ceiling = spring_z + np.sqrt(np.maximum(radius**2 - np.minimum(np.abs(sx), radius) ** 2, 0.0))
        return float(np.min([sx + radius, radius - sx, sy, args.tunnel_length - sy, sz, ceiling - sz]))

    sampled_clearances = [
        conservative_margin(np.vstack([frame, tcp_point]))
        for frame, tcp_point in zip(frames, actual_tcp)
    ]
    container_outside_frame_count = sum(clearance < link_radius for clearance in sampled_clearances)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    for slug, (elevation, azimuth, label) in VIEWS.items():
        fig = plt.figure(figsize=(8.8, 7.0), facecolor="white")
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(grid[:, 0], grid[:, 1], grid[:, 2], s=1, alpha=0.12, color="#777777")
        ax.plot(target[:, 0], target[:, 1], target[:, 2], color="#d62728", lw=2.0, label="open-arch target")
        arm, = ax.plot([], [], [], "-o", color="#202124", lw=2.8, markersize=4.5, label="FR5 full six-axis chain")
        tool, = ax.plot([], [], [], color="#7b1fa2", lw=4.0, label="flange to TCP (150 mm)")
        nozzle, = ax.plot([], [], [], "--", color="#1565c0", lw=2.0, label="TCP points to wall")
        tcp_dot, = ax.plot([], [], [], "o", color="#d62728", markersize=7)
        trace, = ax.plot([], [], [], color="#2e7d32", lw=2.1, label="completed TCP path")
        text = ax.text2D(
            0.02,
            0.96,
            "",
            transform=ax.transAxes,
            va="top",
            fontsize=9,
            bbox={"facecolor": "white", "edgecolor": "#dadce0", "alpha": 0.94},
        )
        ax.set_xlim(-0.72, 0.72)
        ax.set_ylim(0.0, max(0.90, args.tunnel_length))
        ax.set_zlim(0.0, 1.16)
        ax.set_xlabel("X / m")
        ax.set_ylabel("Y / m")
        ax.set_zlabel("Z / m")
        ax.set_title(f"FR5 true joint motion | open arch | {label}", pad=14)
        ax.view_init(elev=elevation, azim=azimuth)
        ax.legend(loc="upper right", fontsize=7.5)

        def update(index: int):
            point = frames[index]
            flange = point[-1]
            tcp = actual_tcp[index]
            tip = tcp + direction[index] * 0.10
            arm.set_data(point[:, 0], point[:, 1])
            arm.set_3d_properties(point[:, 2])
            tool.set_data([flange[0], tcp[0]], [flange[1], tcp[1]])
            tool.set_3d_properties([flange[2], tcp[2]])
            nozzle.set_data([tcp[0], tip[0]], [tcp[1], tip[1]])
            nozzle.set_3d_properties([tcp[2], tip[2]])
            tcp_dot.set_data([tcp[0]], [tcp[1]])
            tcp_dot.set_3d_properties([tcp[2]])
            trace.set_data(actual_tcp[: index + 1, 0], actual_tcp[: index + 1, 1])
            trace.set_3d_properties(actual_tcp[: index + 1, 2])
            text.set_text(f"{label}\npoint {index:03d}/{len(q) - 1} | max adjacent step {max_step:.2f} deg")
            return arm, tool, nozzle, tcp_dot, trace, text

        animation = FuncAnimation(fig, update, frames=len(q), interval=1000 / args.fps, blit=False)
        out = args.out_dir / f"animation_open_arch_true_{slug}.gif"
        try:
            animation.save(out, writer=PillowWriter(fps=args.fps))
        finally:
            plt.close(fig)
        written.append(str(out))
        print(f"Wrote {out}")

    (args.out_dir / "animation_open_arch_true_multiview.json").write_text(
        json.dumps(
            {
                "source_ik": str(args.ik.resolve()),
                "views": written,
                "waypoint_count": len(q),
                "max_adjacent_joint_step_deg": max_step,
                "tcp_step_p05_m": tcp_step_p05,
                "tcp_step_p95_m": tcp_step_p95,
                "tcp_step_fluctuation_ratio": tcp_step_fluctuation,
                "joint_range_deg": {f"j{i + 1}": float(value) for i, value in enumerate(q_range_deg)},
                "max_joint_range_deg": float(np.max(q_range_deg)),
                "full_rotation_joint_count": int(np.sum(q_range_deg >= 300.0)),
                "container_outside_frame_count": int(container_outside_frame_count),
                "container_inside_sample_count": int(np.sum(inside_mask)),
                "container_total_sample_count": int(all_joint_points.shape[0]),
                "container_length_m": args.tunnel_length,
                "conservative_link_radius_m": link_radius,
                "minimum_sampled_centerline_clearance_m": float(min(sampled_clearances)),
                "tcp_y_m": [float(np.min(actual_tcp[:, 1])), float(np.max(actual_tcp[:, 1]))],
                "display": "full six-axis joint chain with true flange-to-TCP segment",
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
