#!/usr/bin/env python3
"""Render the MoveIt-validated global-roll/backtracking candidate path."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.official_fairino import load_official_fr5_robot


def read_xyz(path: Path) -> np.ndarray:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return np.asarray([[float(row[key]) for key in ("x", "y", "z")] for row in rows], dtype=float)


def read_joint_bank(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    rows.sort(key=lambda row: int(row["waypoint"]))
    q = np.asarray([[float(row[f"q{i}"]) for i in range(1, 7)] for row in rows], dtype=float)
    roll = np.asarray([float(row.get("roll_curve_deg", row.get("roll_deg", 0.0))) for row in rows], dtype=float)
    tilt = np.asarray(
        [max(abs(float(row.get("tilt_x_deg", 0.0))), abs(float(row.get("tilt_y_deg", 0.0)))) for row in rows],
        dtype=float,
    )
    return q, roll, tilt


def interpolate_joint_path(q: np.ndarray, frames_per_segment: int) -> tuple[np.ndarray, np.ndarray]:
    phase = np.linspace(0.0, len(q) - 1.0, max(2, (len(q) - 1) * frames_per_segment + 1))
    lo = np.floor(phase).astype(int)
    hi = np.minimum(lo + 1, len(q) - 1)
    alpha = (phase - lo)[:, None]
    return q[lo] * (1.0 - alpha) + q[hi] * alpha, phase


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--candidate-bank",
        type=Path,
        default=Path(r"D:\robot_arm_reinforcement_learning\data\ik_candidate_bank.csv"),
    )
    parser.add_argument("--out", type=Path, default=ROOT / "outputs" / "global_roll_backtracking_animation.gif")
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--frames-per-segment", type=int, default=3)
    args = parser.parse_args()

    q_source, roll_source, tilt_source = read_joint_bank(args.candidate_bank)
    q, phase = interpolate_joint_path(q_source, args.frames_per_segment)
    surface = read_xyz(ROOT / "outputs" / "surface_path.csv")
    target = read_xyz(ROOT / "outputs" / "tcp_path.csv")
    wall = surface[:240]
    wall_open = surface[30:211].copy()
    target_open = target[30:211].copy()
    wall_open[:, 1] = 0.30
    target_open[:, 1] = 0.30

    robot_model, _ = load_official_fr5_robot(ROOT)
    robot = robot_model.with_base([0.0, 0.10, 0.20], yaw=np.pi)
    tcp_offset = np.asarray([0.0, 0.0, 0.150])
    joint_points: list[np.ndarray] = []
    tcp_points: list[np.ndarray] = []
    for qi in q:
        transforms = robot.all_joint_frames(qi)
        flange = transforms[-1]
        tcp = flange[:3, 3] + flange[:3, :3] @ tcp_offset
        joint_points.append(np.vstack([transforms[:, :3, 3], tcp]))
        tcp_points.append(tcp)
    tcp_actual = np.asarray(tcp_points)

    views = [("Front", 18, -90), ("Left-front", 24, -63), ("Right-front", 24, -117), ("Top", 90, -90)]
    fig = plt.figure(figsize=(14.5, 8.8), facecolor="white")
    artists = []
    for subplot, (title, elevation, azimuth) in enumerate(views, 1):
        ax = fig.add_subplot(2, 2, subplot, projection="3d")
        ax.set_proj_type("ortho")
        wall_line, = ax.plot(wall[:, 0], np.full(len(wall), 0.30), wall[:, 2], "--", color="#245b8a", lw=1.7)
        target_line, = ax.plot(target_open[:, 0], target_open[:, 1], target_open[:, 2], color="#c62828", lw=2.2)
        arm_line, = ax.plot([], [], [], color="#111111", lw=4.0, marker="o", ms=4)
        tcp_line, = ax.plot([], [], [], color="#7b1fa2", lw=4.0)
        spray_line, = ax.plot([], [], [], color="#ef6c00", lw=2.0)
        trace_line, = ax.plot([], [], [], color="#2e7d32", lw=1.8)
        ax.set_title(f"{title} | global roll + backtracking", fontsize=11)
        ax.set_xlim(-0.72, 0.72)
        ax.set_ylim(0.0, 0.95)
        ax.set_zlim(-0.28, 1.18)
        ax.set_xlabel("X / m")
        ax.set_ylabel("Y / m")
        ax.set_zlabel("Z / m")
        ax.view_init(elev=elevation, azim=azimuth)
        artists.append((arm_line, tcp_line, spray_line, trace_line))

    fig.text(
        0.5,
        0.015,
        "Blue dashed = tunnel wall | Red = TCP target path | Green = validated TCP | Orange = spray direction | purple = TCP tool",
        ha="center",
        fontsize=9,
    )
    status = fig.text(0.5, 0.975, "", ha="center", fontsize=10, color="#333333")

    def update(frame: int):
        points = joint_points[frame]
        flange, tcp = points[-2], points[-1]
        target_index = min(int(round(phase[frame])), len(target_open) - 1)
        wall_point = wall_open[target_index]
        spray_tip = tcp + (wall_point - tcp) / max(float(np.linalg.norm(wall_point - tcp)), 1e-12) * 0.12
        returned = []
        for arm_line, tcp_line, spray_line, trace_line in artists:
            arm_line.set_data(points[:-1, 0], points[:-1, 1])
            arm_line.set_3d_properties(points[:-1, 2])
            tcp_line.set_data([flange[0], tcp[0]], [flange[1], tcp[1]])
            tcp_line.set_3d_properties([flange[2], tcp[2]])
            spray_line.set_data([tcp[0], spray_tip[0]], [tcp[1], spray_tip[1]])
            spray_line.set_3d_properties([tcp[2], spray_tip[2]])
            trace_line.set_data(tcp_actual[: frame + 1, 0], tcp_actual[: frame + 1, 1])
            trace_line.set_3d_properties(tcp_actual[: frame + 1, 2])
            returned.extend([arm_line, tcp_line, spray_line, trace_line])
        source_index = min(int(round(phase[frame])), len(roll_source) - 1)
        status.set_text(
            f"MoveIt validated global roll/backtracking | point {source_index + 1}/{len(q_source)} "
            f"| roll {roll_source[source_index]:.1f} deg | TCP reparam tilt {tilt_source[source_index]:.1f} deg"
        )
        return returned + [status]

    animation = FuncAnimation(fig, update, frames=len(q), interval=1000 / args.fps, blit=False)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    animation.save(args.out, writer=PillowWriter(fps=args.fps))
    plt.close(fig)
    max_step = float(np.rad2deg(np.max(np.abs(np.diff(q, axis=0)))))
    validation = {
        "source_candidate_bank": str(args.candidate_bank.resolve()),
        "source_waypoint_count": int(len(q_source)),
        "rendered_frame_count": int(len(q)),
        "max_rendered_adjacent_joint_step_deg": max_step,
        "status": "pass" if max_step <= 3.0 else "fail",
    }
    args.out.with_name(args.out.stem + "_validation.json").write_text(
        json.dumps(validation, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote {args.out.resolve()}")
    print(json.dumps(validation, ensure_ascii=False))


if __name__ == "__main__":
    main()
