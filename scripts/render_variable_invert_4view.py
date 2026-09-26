from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.official_fairino import load_official_fr5_robot
from scripts.run_fr5_invert_variable_section import section


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser(description="Render the variable-invert motion as one synchronized 2x2 four-view GIF.")
    parser.add_argument("--csv", type=Path, default=Path(r"C:\Users\86198\Desktop\ww\variable_invert_tcp_poses.csv"))
    parser.add_argument("--out", type=Path, default=Path(r"C:\Users\86198\Desktop\ww\variable_invert_motion_4view.gif"))
    parser.add_argument("--frames", type=int, default=360)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--start-time", type=float, default=None)
    parser.add_argument("--duration", type=float, default=30.0)
    args = parser.parse_args()

    rows = read_csv(args.csv)
    q_source = np.asarray([[float(row[f"j{i}"]) for i in range(1, 7)] for row in rows], dtype=float)
    time_source = np.asarray([float(row.get("time_s", i * 0.04)) for i, row in enumerate(rows)], dtype=float)
    start = float(time_source[0] if args.start_time is None else max(time_source[0], args.start_time))
    end = float(time_source[-1] if args.duration is None else min(time_source[-1], start + args.duration))
    render_time = np.linspace(start, end, min(args.frames, len(q_source)))
    frame_index = np.searchsorted(time_source, render_time, side="left").clip(0, len(q_source) - 1)
    frame_index = np.unique(frame_index)

    robot, _ = load_official_fr5_robot(ROOT)
    robot = robot.with_base([0.0, -0.25, 0.20], yaw=np.pi)
    tool_offset = np.asarray([0.0, 0.0, 0.150])

    joint_points = []
    tcp_actual = []
    for source_index in frame_index:
        frames = robot.all_joint_frames(q_source[source_index])
        flange = frames[-1]
        tcp = flange[:3, 3] + flange[:3, :3] @ tool_offset
        joint_points.append(np.vstack([frames[:, :3, 3], tcp]))
        tcp_actual.append(tcp)
    tcp_actual = np.asarray(tcp_actual)

    # Show the same five longitudinal sections used by the simulation.
    wall_sections = []
    path_sections = []
    for yy in np.linspace(0.0, 0.45, 5):
        u = yy / 0.45
        width = 0.78 + 0.08 * np.sin(2 * np.pi * u)
        height = 0.76 + 0.06 * np.cos(2 * np.pi * u)
        invert = 0.03 + 0.015 * np.sin(np.pi * u)
        wall, _ = section(width, height, invert, 120)
        path, _ = section(width - 0.10, height - 0.10, invert, 120)
        wall = np.vstack([wall, wall[0]])
        path = np.vstack([path, path[0]])
        wall_sections.append(np.c_[wall[:, 0], np.full(len(wall), yy), wall[:, 1] + 0.14])
        path_sections.append(np.c_[path[:, 0], np.full(len(path), yy), path[:, 1] + 0.19])

    views = [
        ("Front", 20, -90, False),
        ("Left-front", 24, -63, False),
        ("Right-front", 24, -117, False),
        ("Top (plan)", 0, 0, True),
    ]
    fig = plt.figure(figsize=(14.0, 9.0), facecolor="white")
    fig.patch.set_facecolor("white")
    fig.patch.set_alpha(1.0)
    artists = []
    first_handles = None
    top_axis = None
    for subplot, (title, elevation, azimuth, is_top) in enumerate(views, 1):
        ax = fig.add_subplot(2, 2, subplot) if is_top else fig.add_subplot(2, 2, subplot, projection="3d")
        ax.set_facecolor("white")
        if not is_top:
            ax.set_proj_type("ortho")
        wall_handle = None
        path_handle = None
        for wall, path in zip(wall_sections, path_sections):
            if is_top:
                wall_handle, = ax.plot(wall[:, 0], wall[:, 1], color="#9e9e9e", lw=0.85, alpha=0.42)
                path_handle, = ax.plot(path[:, 0], path[:, 1], color="#d32f2f", lw=1.45, alpha=0.88)
            else:
                wall_handle, = ax.plot(wall[:, 0], wall[:, 1], wall[:, 2], color="#9e9e9e", lw=0.85, alpha=0.42)
                path_handle, = ax.plot(path[:, 0], path[:, 1], path[:, 2], color="#d32f2f", lw=1.45, alpha=0.88)
        # Longitudinal guides make the variable-section progression visible.
        for x in np.linspace(-0.35, 0.35, 5):
            if is_top:
                ax.plot([x, x], [0.0, 0.45], color="#bdbdbd", lw=0.45, alpha=0.25)
            else:
                ax.plot([x, x], [0.0, 0.45], [0.14, 0.14], color="#bdbdbd", lw=0.45, alpha=0.25)

        if is_top:
            arm, = ax.plot([], [], "-o", color="#202124", lw=2.8, markersize=3.8, label="FR5")
            tool, = ax.plot([], [], color="#7b1fa2", lw=3.8, label="150 mm virtual TCP")
            tcp_dot, = ax.plot([], [], "o", color="#2e7d32", markersize=6.5, label="TCP")
        else:
            arm, = ax.plot([], [], [], "-o", color="#202124", lw=2.8, markersize=3.8, label="FR5")
            tool, = ax.plot([], [], [], color="#7b1fa2", lw=3.8, label="150 mm virtual TCP")
            tcp_dot, = ax.plot([], [], [], "o", color="#2e7d32", markersize=6.5, label="TCP")
        if subplot == 1:
            first_handles = [wall_handle, path_handle, arm, tool, tcp_dot]
        if is_top:
            # The top view is deliberately 2-D: it shows X-Y placement and
            # avoids the overlapping 3-D Z-axis labels from the old version.
            ax.plot([0.0], [-0.25], marker="s", color="#202124", markersize=4)
            ax.set_xlim(-0.70, 0.70); ax.set_ylim(-0.50, 0.70)
            ax.set_aspect("equal", adjustable="box")
            ax.set_xlabel("X / m"); ax.set_ylabel("Y / m")
        else:
            ax.plot([0.0], [-0.25], [0.20], marker="s", color="#202124", markersize=4)
            ax.set_xlim(-0.50, 0.50); ax.set_ylim(-0.08, 0.50); ax.set_zlim(0.05, 1.00)
            ax.set_xlabel("X / m", labelpad=1); ax.set_ylabel("Y / m", labelpad=1); ax.set_zlabel("Z / m", labelpad=1)
        ax.set_title(title, fontsize=11, pad=5)
        if not is_top:
            ax.view_init(elev=elevation, azim=azimuth)
        else:
            top_axis = ax
        artists.append((ax, arm, tool, tcp_dot, is_top))

    status = fig.text(
        0.5,
        0.965,
        "",
        ha="center",
        va="top",
        fontsize=12,
        bbox={"facecolor": "white", "edgecolor": "#d5d9df", "alpha": 0.96, "boxstyle": "round,pad=0.4"},
    )
    fig.legend(
        first_handles,
        ["tunnel wall", "planned TCP path", "FR5", "150 mm virtual TCP", "current TCP"],
        loc="lower center",
        ncol=6,
        bbox_to_anchor=(0.5, 0.035),
        fontsize=8.5,
        frameon=False,
    )
    fig.text(0.5, 0.012, "带仰拱变截面隧道 | 底部中点起始 | 四视角同步播放", ha="center", fontsize=9.5, color="#333333")
    fig.subplots_adjust(left=0.035, right=0.985, bottom=0.085, top=0.91, wspace=0.08, hspace=0.16)
    if top_axis is not None:
        # Apply after subplots_adjust; otherwise the global layout pass
        # restores the oversized quadrant.
        pos = top_axis.get_position()
        top_axis.set_position([pos.x0 + 0.050, pos.y0 + 0.085, pos.width * 0.696, pos.height * 0.672])
        top_axis.tick_params(labelsize=7)
        top_axis.xaxis.label.set_size(8)
        top_axis.yaxis.label.set_size(8)

    def update(frame_number: int):
        points = joint_points[frame_number]
        flange, tcp = points[-2], points[-1]
        returned = []
        for _, arm, tool, tcp_dot, is_top in artists:
            arm.set_data(points[:-1, 0], points[:-1, 1])
            tool.set_data([flange[0], tcp[0]], [flange[1], tcp[1]])
            tcp_dot.set_data([tcp[0]], [tcp[1]])
            if not is_top:
                arm.set_3d_properties(points[:-1, 2])
                tool.set_3d_properties([flange[2], tcp[2]])
                tcp_dot.set_3d_properties([tcp[2]])
            returned.extend([arm, tool, tcp_dot])
        status.set_text(f"FR5 带仰拱变截面喷涂 | source t = {time_source[frame_index[frame_number]]:07.2f} s / {time_source[-1]:07.2f} s")
        return (*returned, status)

    animation = FuncAnimation(fig, update, frames=len(frame_index), interval=1000 / args.fps, blit=False)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    try:
        animation.save(args.out, writer=PillowWriter(fps=args.fps), savefig_kwargs={"facecolor": "white", "transparent": False})
    finally:
        plt.close(fig)
    print(f"Wrote {args.out.resolve()} with {len(frame_index)} synchronized frames")


if __name__ == "__main__":
    main()
