from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

ROOT = Path(__file__).resolve().parents[1]

import sys

sys.path.insert(0, str(ROOT))

from src.official_fairino import load_official_fr5_robot
from src.tunnel_geometry import HorseshoeTunnel, TunnelConfig


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def sampled_indices(start: int, end: int, stride: int) -> np.ndarray:
    count = max(2, int(np.ceil((end - start + 1) / stride)))
    return np.unique(np.linspace(start, end, count).round().astype(int))


def set_equal(ax, points: np.ndarray) -> None:
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = (mins + maxs) / 2.0
    radius = max(float(np.max(maxs - mins)) / 2.0, 0.55)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(max(-0.05, center[2] - radius), center[2] + radius)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the segmented FR5 process/reorientation GIF.")
    parser.add_argument("--outputs", type=Path, default=ROOT / "outputs")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs" / "animation_segmented.gif")
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--process-stride", type=int, default=4)
    parser.add_argument("--transition-frames", type=int, default=10)
    args = parser.parse_args()

    outputs = args.outputs.resolve()
    trajectory = read_rows(outputs / "moveit_smoothed_joint_trajectory.csv")
    plan = read_rows(outputs / "segmented_process_plan.csv")
    transitions = read_rows(outputs / "smooth_reorientation_transitions.csv")
    q_process = np.asarray([[float(row[f"j{i}_q"]) for i in range(1, 7)] for row in trajectory])

    transitions_by_pair: dict[tuple[int, int], list[dict[str, str]]] = {}
    for row in transitions:
        pair = (int(row["from_index"]), int(row["to_index"]))
        transitions_by_pair.setdefault(pair, []).append(row)

    frames: list[dict[str, object]] = []
    for row in plan:
        start = int(row["start_index"])
        end = int(row["end_index"])
        segment_id = int(row["segment_id"])
        if row["type"] == "process":
            for index in sampled_indices(start, end, args.process_stride):
                frames.append(
                    {
                        "q": q_process[index],
                        "tcp_index": index,
                        "type": "process",
                        "segment_id": segment_id,
                        "phase": index % 240,
                    }
                )
        else:
            rows = transitions_by_pair[(start, end)]
            local = sampled_indices(0, len(rows) - 1, max(1, len(rows) // args.transition_frames))
            for index in local:
                sample = rows[index]
                frames.append(
                    {
                        "q": np.asarray([float(sample[f"j{i}_q"]) for i in range(1, 7)]),
                        "tcp_index": start,
                        "type": "transition",
                        "segment_id": segment_id,
                        "phase": start % 240,
                    }
                )

    robot, _ = load_official_fr5_robot(ROOT)
    # Match the MoveIt/offline visualization frame used by the validated reference GIF.
    # Omitting this base transform made the same joint data appear outside the tunnel.
    robot = robot.with_base([0.0, 0.0, 0.20], yaw=np.pi)
    tcp_offset = np.asarray([0.0, 0.0, 0.150])
    target_tcp = np.asarray(
        [
            (lambda flange: flange[:3, 3] + flange[:3, :3] @ tcp_offset)(robot.fk(q))
            for q in q_process
        ]
    )
    tcp_positions = []
    joint_points = []
    for frame in frames:
        q = np.asarray(frame["q"], dtype=float)
        transforms = robot.all_joint_frames(q)
        flange = transforms[-1]
        tcp = flange[:3, 3] + flange[:3, :3] @ tcp_offset
        tcp_positions.append(tcp)
        joint_points.append(np.vstack([transforms[:, :3, 3], tcp]))
    tcp_positions = np.asarray(tcp_positions)

    fig = plt.figure(figsize=(9.6, 7.2), facecolor="#f7f9fc")
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("#f7f9fc")
    tunnel = HorseshoeTunnel(TunnelConfig(width=1.20, height=1.10, length=0.60, fillet_radius=0.20))
    grid, _ = tunnel.surface_grid(0.10, 0.20)
    ax.scatter(grid[:, 0], grid[:, 1], grid[:, 2], s=1, alpha=0.13, color="#777777")
    ax.plot(target_tcp[:, 0], target_tcp[:, 1], target_tcp[:, 2], color="#d32f2f", lw=1.5, alpha=0.8, label="150 mm TCP target")
    robot_line, = ax.plot([], [], [], marker="o", lw=3.0, color="#202124", markersize=4, label="FR5")
    tool_line, = ax.plot([], [], [], lw=4.0, color="#7b1fa2", label="150 mm virtual TCP")
    tcp_dot, = ax.plot([], [], [], marker="o", markersize=7, color="#2e7d32")
    trail, = ax.plot([], [], [], lw=2.2, color="#2e7d32", alpha=0.8, label="completed process")
    stop_dot, = ax.plot([], [], [], marker="s", markersize=8, color="#ef6c00")
    status_text = ax.text2D(
        0.02,
        0.96,
        "",
        transform=ax.transAxes,
        va="top",
        fontsize=11,
        bbox={"facecolor": "white", "edgecolor": "#dadce0", "alpha": 0.94, "boxstyle": "round,pad=0.45"},
    )
    ax.text2D(0.02, 0.04, "Green: process ON   Orange: process OFF / smooth reorientation", transform=ax.transAxes, fontsize=9)
    set_equal(ax, np.vstack([target_tcp, np.vstack(joint_points)]))
    ax.set_xlabel("X / m")
    ax.set_ylabel("Y / m")
    ax.set_zlabel("Z / m")
    ax.set_title("FR5 segmented closed-contour motion | virtual TCP = 150 mm", pad=16, fontsize=13, fontweight="bold")
    ax.view_init(elev=24, azim=-63)
    ax.legend(loc="upper right", fontsize=8)

    def update(frame_index: int):
        frame = frames[frame_index]
        points = joint_points[frame_index]
        flange = points[-2]
        tcp = points[-1]
        is_process = frame["type"] == "process"
        color = "#2e7d32" if is_process else "#ef6c00"
        robot_line.set_data(points[:-1, 0], points[:-1, 1])
        robot_line.set_3d_properties(points[:-1, 2])
        tool_line.set_data([flange[0], tcp[0]], [flange[1], tcp[1]])
        tool_line.set_3d_properties([flange[2], tcp[2]])
        tcp_dot.set_color(color)
        tcp_dot.set_data([tcp[0]], [tcp[1]])
        tcp_dot.set_3d_properties([tcp[2]])
        if is_process:
            stop_dot.set_data([], [])
            stop_dot.set_3d_properties([])
        else:
            stop_dot.set_data([tcp[0]], [tcp[1]])
            stop_dot.set_3d_properties([tcp[2]])
        # NaN separators prevent the green process trail from connecting across
        # spray-off reorientation intervals and falsely suggesting overspray.
        path = np.asarray(
            [
                tcp_positions[i] if frames[i]["type"] == "process" else np.full(3, np.nan)
                for i in range(frame_index + 1)
            ]
        )
        trail.set_data(path[:, 0], path[:, 1])
        trail.set_3d_properties(path[:, 2])
        status_text.set_text(
            f"Segment {frame['segment_id']:02d} | phase {frame['phase']:03d}\n"
            + ("PROCESS ON · contour tracking" if is_process else "PROCESS OFF · smooth reorientation")
        )
        return robot_line, tool_line, tcp_dot, trail, stop_dot, status_text

    animation = FuncAnimation(fig, update, frames=len(frames), interval=1000 / args.fps, blit=False)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    try:
        animation.save(args.out, writer=PillowWriter(fps=args.fps))
    finally:
        plt.close(fig)
    print(f"Wrote {args.out.resolve()} with {len(frames)} frames")


if __name__ == "__main__":
    main()
