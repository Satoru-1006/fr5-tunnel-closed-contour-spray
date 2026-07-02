from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

from .robot_model import FR5Robot
from .time_parameterization import TimeProfile
from .tunnel_geometry import HorseshoeTunnel


def _set_equal(ax, pts: np.ndarray) -> None:
    mins, maxs = pts.min(axis=0), pts.max(axis=0)
    center = (mins + maxs) / 2
    radius = max(np.max(maxs - mins) / 2, 0.1)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def save_path_plot(tunnel: HorseshoeTunnel, surface: np.ndarray, tcp: np.ndarray, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    grid, _ = tunnel.surface_grid(0.08, 0.16)
    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(grid[:, 0], grid[:, 1], grid[:, 2], s=1, alpha=0.18, c="#888888")
    ax.plot(surface[:, 0], surface[:, 1], surface[:, 2], lw=1.0, c="#1f77b4", label="wall path")
    ax.plot(tcp[:, 0], tcp[:, 1], tcp[:, 2], lw=1.4, c="#d62728", label="TCP path")
    ax.set_xlabel("X / m")
    ax.set_ylabel("Y / m")
    ax.set_zlabel("Z / m")
    ax.legend()
    _set_equal(ax, np.vstack([grid, tcp]))
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def save_dynamics_plots(profile: TimeProfile, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, axs = plt.subplots(4, 1, figsize=(10, 11), sharex=True)
    speed_error = (profile.tcp_speed - np.mean(profile.tcp_speed)) * 1000.0
    axs[0].plot(profile.t, speed_error)
    axs[0].set_ylabel("TCP speed error mm/s")
    axs[0].ticklabel_format(useOffset=False)
    axs[1].plot(profile.t, profile.q)
    axs[1].set_ylabel("q rad")
    axs[2].plot(profile.t, profile.dq)
    axs[2].set_ylabel("dq rad/s")
    axs[3].plot(profile.t, profile.ddq)
    axs[3].set_ylabel("ddq rad/s2")
    axs[3].set_xlabel("time / s")
    for ax in axs:
        ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out.with_name(out.stem + "_main.png"), dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(profile.t, profile.jerk)
    ax.set_xlabel("time / s")
    ax.set_ylabel("jerk rad/s3")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out.with_name(out.stem + "_jerk.png"), dpi=180)
    plt.close(fig)


def save_coverage_heatmap(tunnel: HorseshoeTunnel, surface_path: np.ndarray, footprint_width: float, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    samples, _ = tunnel.surface_grid(section_spacing=footprint_width * 0.30, longitudinal_spacing=footprint_width * 0.30)
    d = np.sqrt(np.sum((samples[:, None, :] - surface_path[None, :, :]) ** 2, axis=2))
    hits = np.sum(d <= footprint_width / 2.0, axis=1)
    fig = plt.figure(figsize=(9, 5))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(samples[:, 0], samples[:, 1], samples[:, 2], c=hits, s=8, cmap="viridis", vmin=0)
    ax.plot(surface_path[:, 0], surface_path[:, 1], surface_path[:, 2], lw=0.8, c="#d62728", alpha=0.75)
    fig.colorbar(sc, ax=ax, shrink=0.72, label="coverage hit count")
    ax.set_xlabel("X / m")
    ax.set_ylabel("Y / m")
    ax.set_zlabel("Z / m")
    ax.set_title("Coverage heatmap")
    _set_equal(ax, np.vstack([samples, surface_path]))
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def save_closed_section_plot(surface_path: np.ndarray, tcp_path: np.ndarray, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(surface_path[:, 0], surface_path[:, 2], lw=3.0, c="#1f77b4", label="wall closed contour")
    ax.plot(tcp_path[:, 0], tcp_path[:, 2], lw=2.0, c="#d62728", label="TCP offset contour")
    ax.scatter([surface_path[0, 0]], [surface_path[0, 2]], s=55, c="#ff0000", label="start")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X / m")
    ax.set_ylabel("Z / m")
    ax.set_title("Closed Horseshoe Contour Tracking Section")
    ax.grid(True, alpha=0.35)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def save_animation(robot: FR5Robot, profile: TimeProfile, tunnel: HorseshoeTunnel, out: Path, max_frames: int = 180) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    grid, _ = tunnel.surface_grid(0.10, 0.20)
    step = max(1, len(profile.t) // max_frames)
    frame_ids = np.arange(0, len(profile.t), step)
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(grid[:, 0], grid[:, 1], grid[:, 2], s=1, alpha=0.14, c="#777777")
    ax.plot(profile.path_points[:, 0], profile.path_points[:, 1], profile.path_points[:, 2], lw=1.0, c="#d62728")
    line, = ax.plot([], [], [], marker="o", lw=2.5, c="#222222")
    tcp_dot, = ax.plot([], [], [], marker="o", c="#d62728")
    _set_equal(ax, np.vstack([grid, profile.path_points]))
    ax.set_xlabel("X / m")
    ax.set_ylabel("Y / m")
    ax.set_zlabel("Z / m")

    def update(frame_idx: int):
        q = profile.q[frame_idx]
        frames = robot.all_joint_frames(q)
        pts = frames[:, :3, 3]
        line.set_data(pts[:, 0], pts[:, 1])
        line.set_3d_properties(pts[:, 2])
        p = profile.path_points[frame_idx]
        tcp_dot.set_data([p[0]], [p[1]])
        tcp_dot.set_3d_properties([p[2]])
        return line, tcp_dot

    ani = FuncAnimation(fig, update, frames=frame_ids, interval=40, blit=False)
    try:
        ani.save(out, writer=PillowWriter(fps=24))
    finally:
        plt.close(fig)
