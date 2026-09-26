from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.official_fairino import load_official_fr5_robot
from src.tunnel_geometry import HorseshoeTunnel, TunnelConfig


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def set_equal(ax, points: np.ndarray) -> None:
    lo, hi = points.min(axis=0), points.max(axis=0)
    center = (lo + hi) / 2.0
    radius = max(float(np.max(hi - lo)) / 2.0, 0.55)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(max(-0.05, center[2] - radius), center[2] + radius)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render one closed horseshoe loop with smooth spray-off reorientation.")
    parser.add_argument("--outputs", type=Path, default=ROOT / "outputs")
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "outputs" / "animation_standard_horseshoe_segmented_multiview.gif",
    )
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--hold-frames", type=int, default=8)
    args = parser.parse_args()

    outputs = args.outputs.resolve()
    trajectory = read_rows(outputs / "moveit_smoothed_joint_trajectory.csv")
    transitions = read_rows(outputs / "smooth_reorientation_transitions.csv")
    q_process = np.asarray([[float(row[f"j{i}_q"]) for i in range(1, 7)] for row in trajectory])

    transition_pairs = [(110, 111), (210, 211), (232, 233), (237, 238)]
    transitions_by_pair: dict[tuple[int, int], list[np.ndarray]] = {pair: [] for pair in transition_pairs}
    for row in transitions:
        pair = (int(row["from_index"]), int(row["to_index"]))
        if pair in transitions_by_pair and int(row["loop_index"]) == 0:
            transitions_by_pair[pair].append(np.asarray([float(row[f"j{i}_q"]) for i in range(1, 7)]))

    frames: list[dict[str, object]] = []
    cursor = 0
    for transition_id, (start, end) in enumerate(transition_pairs):
        for process_index in range(cursor, start + 1):
            frames.append({"q": q_process[process_index], "type": "process", "phase": process_index, "transition": -1})
        for _ in range(args.hold_frames):
            frames.append({"q": q_process[start], "type": "hold", "phase": start, "transition": transition_id})
        for q in transitions_by_pair[(start, end)]:
            frames.append({"q": q, "type": "transition", "phase": start, "transition": transition_id})
        for _ in range(args.hold_frames):
            frames.append({"q": q_process[end], "type": "hold", "phase": end, "transition": transition_id})
        cursor = end
    for process_index in range(cursor, 240):
        frames.append({"q": q_process[process_index], "type": "process", "phase": process_index, "transition": -1})

    robot, _ = load_official_fr5_robot(ROOT)
    robot = robot.with_base([0.0, 0.0, 0.20], yaw=np.pi)
    tcp_offset = np.asarray([0.0, 0.0, 0.150])

    joint_points: list[np.ndarray] = []
    tcp_positions: list[np.ndarray] = []
    for frame in frames:
        transforms = robot.all_joint_frames(np.asarray(frame["q"], dtype=float))
        flange = transforms[-1]
        tcp = flange[:3, 3] + flange[:3, :3] @ tcp_offset
        joint_points.append(np.vstack([transforms[:, :3, 3], tcp]))
        tcp_positions.append(tcp)
    tcp_positions_array = np.asarray(tcp_positions)

    target_tcp = []
    for q in q_process[:240]:
        flange = robot.fk(q)
        target_tcp.append(flange[:3, 3] + flange[:3, :3] @ tcp_offset)
    target_tcp_array = np.asarray(target_tcp)

    tunnel = HorseshoeTunnel(TunnelConfig(width=1.20, height=1.10, length=0.60, fillet_radius=0.20))
    grid, _ = tunnel.surface_grid(0.10, 0.20)
    views = [("Front", 20, -90), ("Left-front", 25, -35), ("Right-front", 25, -145), ("Top", 80, -90)]
    fig = plt.figure(figsize=(12, 9), facecolor="#f7f9fc")
    artists = []
    for subplot, (title, elev, azim) in enumerate(views, 1):
        ax = fig.add_subplot(2, 2, subplot, projection="3d")
        ax.set_facecolor("#f7f9fc")
        ax.scatter(grid[:, 0], grid[:, 1], grid[:, 2], s=1, alpha=0.10, color="#777777")
        ax.plot(target_tcp_array[:, 0], target_tcp_array[:, 1], target_tcp_array[:, 2], color="#d32f2f", lw=1.2)
        robot_line, = ax.plot([], [], [], marker="o", lw=2.5, color="#202124", markersize=4)
        tool_line, = ax.plot([], [], [], lw=3.5, color="#7b1fa2")
        tcp_dot, = ax.plot([], [], [], marker="o", markersize=7, color="#2e7d32")
        trail, = ax.plot([], [], [], lw=2.0, color="#2e7d32", alpha=0.8)
        set_equal(ax, np.vstack([grid, target_tcp_array, np.vstack(joint_points)]))
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(title)
        ax.set_xlabel("X / m"); ax.set_ylabel("Y / m"); ax.set_zlabel("Z / m")
        artists.append((robot_line, tool_line, tcp_dot, trail))

    status_text = fig.text(
        0.5,
        0.965,
        "",
        ha="center",
        va="top",
        fontsize=12,
        bbox={"facecolor": "white", "edgecolor": "#dadce0", "alpha": 0.94, "boxstyle": "round,pad=0.4"},
    )
    fig.text(0.5, 0.015, "Green: spray ON   Orange: spray OFF / stopped or smooth reorientation", ha="center", fontsize=10)

    def update(frame_index: int):
        frame = frames[frame_index]
        points = joint_points[frame_index]
        flange, tcp = points[-2], points[-1]
        frame_type = str(frame["type"])
        process_on = frame_type == "process"
        color = "#2e7d32" if process_on else "#ef6c00"
        path = np.asarray(
            [tcp_positions_array[i] if frames[i]["type"] == "process" else np.full(3, np.nan) for i in range(frame_index + 1)]
        )
        returned = []
        for robot_line, tool_line, tcp_dot, trail in artists:
            robot_line.set_data(points[:-1, 0], points[:-1, 1]); robot_line.set_3d_properties(points[:-1, 2])
            tool_line.set_data([flange[0], tcp[0]], [flange[1], tcp[1]]); tool_line.set_3d_properties([flange[2], tcp[2]])
            tcp_dot.set_color(color); tcp_dot.set_data([tcp[0]], [tcp[1]]); tcp_dot.set_3d_properties([tcp[2]])
            trail.set_data(path[:, 0], path[:, 1]); trail.set_3d_properties(path[:, 2])
            returned.extend([robot_line, tool_line, tcp_dot, trail])
        if frame_type == "process":
            state = "SPRAY ON | contour tracking"
        elif frame_type == "hold":
            state = "SPRAY OFF | stopped before/after reorientation"
        else:
            state = "SPRAY OFF | smooth configuration change"
        status_text.set_text(f"Standard horseshoe segmented process | phase {int(frame['phase']):03d} | {state}")
        return (*returned, status_text)

    animation = FuncAnimation(fig, update, frames=len(frames), interval=1000 / args.fps, blit=False)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    try:
        animation.save(args.out, writer=PillowWriter(fps=args.fps))
    finally:
        plt.close(fig)

    q_frames = np.asarray([frame["q"] for frame in frames], dtype=float)
    max_step_deg = float(np.max(np.abs(np.diff(q_frames, axis=0))) * 180.0 / np.pi)
    print(f"Wrote {args.out.resolve()} with {len(frames)} frames")
    print(f"Rendered maximum adjacent joint step: {max_step_deg:.6f} deg")


if __name__ == "__main__":
    main()
