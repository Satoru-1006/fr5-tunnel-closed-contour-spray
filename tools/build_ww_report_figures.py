from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd
from PIL import Image, ImageOps


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_fr5_invert_variable_section import (  # noqa: E402
    make_targets,
    section,
    tunnel_link_clearance,
)
from src.official_fairino import load_official_fr5_robot  # noqa: E402
from src.time_parameterization import make_segmented_ruckig_time_profile  # noqa: E402


PALETTE = {
    "navy": "#0F4D92",
    "blue": "#3775BA",
    "teal": "#42949E",
    "orange": "#E28E2C",
    "red": "#B64342",
    "purple": "#7C6CCF",
    "gray": "#767676",
    "light": "#D8D8D8",
    "black": "#272727",
    "green": "#2E9E44",
}
JOINT_COLORS = [
    PALETTE["navy"],
    PALETTE["blue"],
    PALETTE["teal"],
    PALETTE["orange"],
    PALETTE["red"],
    PALETTE["purple"],
]


def apply_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Times New Roman", "DejaVu Sans"],
            "font.size": 7,
            "axes.labelsize": 7,
            "axes.titlesize": 8,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "legend.fontsize": 6.2,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.7,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def save_figure(fig: plt.Figure, out_dir: Path, stem: str, dpi: int = 450) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("svg", "pdf", "png"):
        fig.savefig(
            out_dir / f"{stem}.{ext}",
            dpi=dpi if ext == "png" else None,
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close(fig)


def panel_label(ax, label: str) -> None:
    ax.text(
        -0.10,
        1.04,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        fontweight="bold",
    )


def workflow_figure(out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 2.55))
    ax.set_axis_off()
    labels = [
        "Variable tunnel\ngeometry",
        "TCP targets and\ntool orientation",
        "FR5 URDF\nkinematics",
        "Multi-seed IK and\ncollision gate",
        "Segmented Ruckig\nretiming",
        "CSV / JSON /\nanimation outputs",
    ]
    xs = np.linspace(0.03, 0.84, len(labels))
    colors = ["#E8EEF5", "#E0F0F0", "#E8EEF5", "#F0E0D0", "#E4E4F0", "#E4CCD8"]
    for i, (x, label, color) in enumerate(zip(xs, labels, colors)):
        patch = FancyBboxPatch(
            (x, 0.36),
            0.13,
            0.30,
            boxstyle="round,pad=0.012,rounding_size=0.015",
            linewidth=0.8,
            edgecolor="#4D4D4D",
            facecolor=color,
            transform=ax.transAxes,
        )
        ax.add_patch(patch)
        ax.text(x + 0.065, 0.51, label, transform=ax.transAxes, ha="center", va="center", fontsize=6.4)
        if i < len(labels) - 1:
            ax.annotate(
                "",
                xy=(xs[i + 1] - 0.008, 0.51),
                xytext=(x + 0.138, 0.51),
                xycoords=ax.transAxes,
                arrowprops={"arrowstyle": "-|>", "lw": 0.8, "color": "#4D4D4D"},
            )
    ax.text(0.03, 0.18, "Inputs", transform=ax.transAxes, fontsize=6.5, color=PALETTE["gray"], ha="left")
    ax.text(0.97, 0.18, "Auditable outputs", transform=ax.transAxes, fontsize=6.5, color=PALETTE["gray"], ha="right")
    ax.plot([0.03, 0.97], [0.25, 0.25], transform=ax.transAxes, color=PALETTE["light"], lw=0.8)
    save_figure(fig, out_dir, "fig_1_1_workflow")


def geometry_figures(out_dir: Path) -> None:
    stations = np.linspace(0.0, 0.45, 5)
    fig, axes = plt.subplots(1, 5, figsize=(7.2, 2.05), sharex=True, sharey=True, constrained_layout=True)
    for i, (ax, y) in enumerate(zip(axes, stations)):
        u = y / 0.45
        width = 0.78 + 0.08 * np.sin(2 * np.pi * u)
        height = 0.76 + 0.06 * np.cos(2 * np.pi * u)
        invert = 0.03 + 0.015 * np.sin(np.pi * u)
        wall, _ = section(width, height, invert, 180)
        tcp, _ = section(width - 0.10, height - 0.10, invert, 180)
        wall = np.vstack([wall, wall[0]])
        tcp = np.vstack([tcp, tcp[0]])
        ax.plot(wall[:, 0], wall[:, 1] + 0.14, color=PALETTE["gray"], lw=1.0, label="Tunnel wall")
        ax.plot(tcp[:, 0], tcp[:, 1] + 0.19, color=PALETTE["red"], lw=1.4, label="TCP contour")
        ax.set_aspect("equal")
        ax.set_title(f"y = {y:.3f} m", pad=3)
        ax.set_xlabel("X (m)")
        if i == 0:
            ax.set_ylabel("Z (m)")
        ax.tick_params(length=2.5)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.04), ncol=2)
    save_figure(fig, out_dir, "fig_2_1_variable_sections")

    dense_y = np.linspace(0.0, 0.45, 301)
    u = dense_y / 0.45
    width = 0.78 + 0.08 * np.sin(2 * np.pi * u)
    height = 0.76 + 0.06 * np.cos(2 * np.pi * u)
    invert = 0.03 + 0.015 * np.sin(np.pi * u)
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.05), constrained_layout=True)
    for label, values, color, ax, letter in zip(
        ["Width (m)", "Height (m)", "Invert depth (m)"],
        [width, height, invert],
        [PALETTE["navy"], PALETTE["teal"], PALETTE["orange"]],
        axes,
        "abc",
    ):
        ax.plot(dense_y, values, color=color, lw=1.5)
        ax.scatter(stations, np.interp(stations, dense_y, values), color=color, s=15, zorder=3)
        ax.set_xlabel("Longitudinal position, y (m)")
        ax.set_ylabel(label)
        ax.margins(x=0.03)
        panel_label(ax, letter)
    save_figure(fig, out_dir, "fig_2_2_section_parameters")


def ik_figures(out_dir: Path, waypoints: pd.DataFrame, robot) -> None:
    idx = waypoints["index"].to_numpy()
    q_deg = np.rad2deg(waypoints[[f"j{i}" for i in range(1, 7)]].to_numpy())
    qmin = np.rad2deg(robot.limits.q_min)
    qmax = np.rad2deg(robot.limits.q_max)
    fig, axes = plt.subplots(3, 2, figsize=(7.2, 5.2), sharex=True, constrained_layout=True)
    for j, ax in enumerate(axes.flat):
        ax.plot(idx, q_deg[:, j], color=JOINT_COLORS[j], lw=1.0)
        ax.axhline(qmin[j], color=PALETTE["gray"], lw=0.6, ls="--")
        ax.axhline(qmax[j], color=PALETTE["gray"], lw=0.6, ls="--")
        ax.set_ylabel(f"q{j + 1} (deg)")
        if j >= 4:
            ax.set_xlabel("IK waypoint index")
        panel_label(ax, chr(ord("a") + j))
    save_figure(fig, out_dir, "fig_3_1_ik_waypoint_angles")

    wrapped = (np.diff(waypoints[[f"j{i}" for i in range(1, 7)]].to_numpy(), axis=0) + np.pi) % (2 * np.pi) - np.pi
    steps = np.rad2deg(np.abs(wrapped))
    max_step = steps.max(axis=1)
    max_joint = steps.argmax(axis=1) + 1
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 2.35), constrained_layout=True, gridspec_kw={"width_ratios": [1.8, 1]})
    ax1.plot(idx[1:], max_step, color=PALETTE["navy"], lw=1.0)
    ax1.axhline(25.0, color=PALETTE["red"], lw=0.8, ls="--", label="25 deg selection threshold")
    max_i = int(np.argmax(max_step))
    ax1.scatter(idx[max_i + 1], max_step[max_i], color=PALETTE["red"], s=18, zorder=4)
    ax1.annotate(
        f"{max_step[max_i]:.2f} deg (J{max_joint[max_i]})",
        (idx[max_i + 1], max_step[max_i]),
        xytext=(8, 10),
        textcoords="offset points",
        fontsize=6.2,
        arrowprops={"arrowstyle": "-", "lw": 0.6, "color": PALETTE["gray"]},
    )
    ax1.set_xlabel("Waypoint transition index")
    ax1.set_ylabel("Maximum joint step (deg)")
    ax1.legend(loc="upper right")
    panel_label(ax1, "a")
    maxima = steps.max(axis=0)
    bars = ax2.bar(np.arange(1, 7), maxima, color=JOINT_COLORS, width=0.68)
    ax2.set_xlabel("Joint")
    ax2.set_ylabel("Maximum step (deg)")
    ax2.set_xticks(range(1, 7))
    ax2.bar_label(bars, fmt="%.1f", fontsize=5.8, padding=2)
    panel_label(ax2, "b")
    save_figure(fig, out_dir, "fig_3_2_joint_continuity")


def trajectory_figures(
    out_dir: Path,
    tcp_df: pd.DataFrame,
    profile,
    validation: dict,
    robot,
) -> dict:
    time = tcp_df["time_s"].to_numpy()
    actual = tcp_df[["x", "y", "z"]].to_numpy()
    q = tcp_df[[f"j{i}" for i in range(1, 7)]].to_numpy()
    planned = profile.path_points
    error = np.linalg.norm(actual - planned, axis=1)
    speed_actual = np.linalg.norm(np.gradient(actual, axis=0) / np.gradient(time)[:, None], axis=1)

    fig = plt.figure(figsize=(7.2, 4.45), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(planned[:, 0], planned[:, 1], planned[:, 2], color=PALETTE["gray"], lw=0.8, alpha=0.8, label="Planned TCP")
    stride = max(1, len(actual) // 1500)
    sc = ax.scatter(
        actual[::stride, 0],
        actual[::stride, 1],
        actual[::stride, 2],
        c=time[::stride],
        cmap="viridis",
        s=2.4,
        alpha=0.85,
        label="Realized TCP",
    )
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.view_init(elev=24, azim=-63)
    ax.legend(loc="upper left", bbox_to_anchor=(0.00, 0.98))
    cbar = fig.colorbar(sc, ax=ax, shrink=0.63, pad=0.08)
    cbar.set_label("Time (s)")
    save_figure(fig, out_dir, "fig_4_1_tcp_trajectory_3d")

    fig, axes = plt.subplots(2, 1, figsize=(7.2, 4.25), sharex=True, constrained_layout=True)
    for j, (label, color) in enumerate(zip(["X", "Y", "Z"], [PALETTE["navy"], PALETTE["teal"], PALETTE["orange"]])):
        axes[0].plot(time, actual[:, j], lw=0.9, color=color, label=label)
    axes[0].set_ylabel("TCP position (m)")
    axes[0].legend(loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=3)
    panel_label(axes[0], "a")
    axes[1].plot(time, speed_actual, color=PALETTE["navy"], lw=0.8, label="Realized speed")
    axes[1].axhline(validation["target_tcp_speed_m_s"], color=PALETTE["red"], ls="--", lw=0.8, label="Target speed")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("TCP speed (m s$^{-1}$)")
    axes[1].set_ylim(bottom=0)
    axes[1].legend(loc="upper right")
    panel_label(axes[1], "b")
    save_figure(fig, out_dir, "fig_4_2_tcp_position_speed")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 2.55), constrained_layout=True, gridspec_kw={"width_ratios": [1.8, 1]})
    ax1.plot(time, error * 1000.0, color=PALETTE["red"], lw=0.75)
    ax1.axhline(10.0, color=PALETTE["black"], ls="--", lw=0.8, label="10 mm criterion")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("TCP path error (mm)")
    ax1.legend(loc="upper right")
    panel_label(ax1, "a")
    sorted_e = np.sort(error * 1000.0)
    ecdf = np.arange(1, len(sorted_e) + 1) / len(sorted_e)
    ax2.plot(sorted_e, ecdf, color=PALETTE["navy"], lw=1.2)
    ax2.axvline(10.0, color=PALETTE["black"], ls="--", lw=0.8)
    ax2.set_xlabel("TCP path error (mm)")
    ax2.set_ylabel("Empirical cumulative probability")
    ax2.set_xlim(left=0)
    ax2.set_ylim(0, 1.02)
    panel_label(ax2, "b")
    save_figure(fig, out_dir, "fig_4_3_tcp_path_error")

    q_deg = np.rad2deg(q)
    fig, axes = plt.subplots(3, 2, figsize=(7.2, 5.2), sharex=True, constrained_layout=True)
    for j, ax in enumerate(axes.flat):
        ax.plot(time, q_deg[:, j], color=JOINT_COLORS[j], lw=0.75)
        ax.set_ylabel(f"q{j + 1} (deg)")
        if j >= 4:
            ax.set_xlabel("Time (s)")
        panel_label(ax, chr(ord("a") + j))
    save_figure(fig, out_dir, "fig_4_4_time_joint_angles")

    dq_deg = np.rad2deg(profile.dq)
    ddq_deg = np.rad2deg(profile.ddq)
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 4.3), constrained_layout=True)
    for j in range(6):
        axes[0].plot(time, dq_deg[:, j], color=JOINT_COLORS[j], lw=0.65, label=f"J{j + 1}")
    axes[0].set_ylabel("Joint velocity (deg s$^{-1}$)")
    axes[0].set_xlabel("Time (s)")
    axes[0].legend(loc="upper center", bbox_to_anchor=(0.5, 1.04), ncol=6)
    panel_label(axes[0], "a")
    vel_peak = np.max(np.abs(profile.dq), axis=0)
    acc_peak = np.max(np.abs(profile.ddq), axis=0)
    vel_util = 100.0 * vel_peak / robot.limits.dq_max
    acc_util = 100.0 * acc_peak / robot.limits.ddq_max
    xj = np.arange(1, 7)
    width = 0.34
    b1 = axes[1].bar(xj - width / 2, vel_util, width=width, color=PALETTE["navy"], label="Velocity limit use")
    b2 = axes[1].bar(xj + width / 2, acc_util, width=width, color=PALETTE["orange"], label="Acceleration limit use")
    axes[1].axhline(100.0, color=PALETTE["black"], lw=0.8, ls="--")
    axes[1].set_xticks(xj)
    axes[1].set_xlabel("Joint")
    axes[1].set_ylabel("Peak limit utilization (%)")
    axes[1].set_ylim(0, max(110.0, float(np.max(acc_util)) * 1.13))
    axes[1].legend(loc="upper left", ncol=2)
    axes[1].bar_label(b1, fmt="%.0f", fontsize=5.5, padding=1)
    axes[1].bar_label(b2, fmt="%.0f", fontsize=5.5, padding=1)
    panel_label(axes[1], "b")
    save_figure(fig, out_dir, "fig_4_5_joint_velocity_acceleration")

    clearance_stride = max(1, len(q) // 200)
    clearances = np.asarray(
        [tunnel_link_clearance(robot, qi, validation["length_m"]) for qi in q[::clearance_stride]]
    )
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.4), constrained_layout=True)
    metrics = [
        ("Maximum IK error", validation["ik_error_max"] * 1000, 10.0, "mm", False),
        ("Maximum adjacent joint step", validation["max_adjacent_joint_step_deg"], 20.0, "deg", False),
        ("Maximum realized TCP error", validation["realized_tcp_path_error_max_m"] * 1000, 10.0, "mm", False),
        ("Minimum sampled clearance", validation["minimum_sampled_link_clearance_m"] * 1000, 0.0, "mm", True),
    ]
    for ax, (name, value, threshold, unit, greater_is_bad), letter in zip(axes.flat, metrics, "abcd"):
        color = PALETTE["red"] if (value > threshold if not greater_is_bad else value < threshold) else PALETTE["green"]
        ax.barh([0], [value], color=color, height=0.42)
        ax.axvline(threshold, color=PALETTE["black"], ls="--", lw=0.8)
        ax.set_yticks([])
        ax.set_xlabel(unit)
        ax.set_title(name, loc="left", pad=3)
        ax.text(value, 0, f" {value:.2f}", va="center", ha="left" if value >= 0 else "right", fontsize=6.5)
        xmin = min(0.0, value, threshold)
        xmax = max(0.0, value, threshold)
        margin = max((xmax - xmin) * 0.25, 1.0)
        ax.set_xlim(xmin - margin, xmax + margin)
        panel_label(ax, letter)
    save_figure(fig, out_dir, "fig_5_1_validation_metrics")

    stats = {
        "trajectory_duration_s": float(time[-1]),
        "sample_count": int(len(time)),
        "median_dt_s": float(np.median(np.diff(time))),
        "actual_path_length_m": float(np.linalg.norm(np.diff(actual, axis=0), axis=1).sum()),
        "planned_path_length_m": float(np.linalg.norm(np.diff(planned, axis=0), axis=1).sum()),
        "tcp_speed_mean_m_s": float(np.mean(speed_actual)),
        "tcp_speed_median_m_s": float(np.median(speed_actual)),
        "tcp_speed_p95_m_s": float(np.percentile(speed_actual, 95)),
        "tcp_speed_max_m_s": float(np.max(speed_actual)),
        "tcp_error_mean_m": float(np.mean(error)),
        "tcp_error_median_m": float(np.median(error)),
        "tcp_error_p95_m": float(np.percentile(error, 95)),
        "tcp_error_max_m": float(np.max(error)),
        "tcp_error_over_10mm_count": int(np.sum(error > 0.01)),
        "joint_angle_min_deg": np.min(q_deg, axis=0).tolist(),
        "joint_angle_max_deg": np.max(q_deg, axis=0).tolist(),
        "joint_velocity_peak_deg_s": np.max(np.abs(dq_deg), axis=0).tolist(),
        "joint_acceleration_peak_deg_s2": np.max(np.abs(ddq_deg), axis=0).tolist(),
        "sampled_clearance_min_m": float(np.min(clearances)),
        "sampled_collision_count": int(np.sum(clearances < 0)),
    }
    return stats


def animation_plate(out_dir: Path, gif_path: Path) -> None:
    with Image.open(gif_path) as gif:
        n = getattr(gif, "n_frames", 1)
        indices = np.linspace(0, max(0, n - 1), 4).round().astype(int)
        frames = []
        for idx in indices:
            gif.seek(int(idx))
            frame = gif.convert("RGB")
            frame = ImageOps.contain(frame, (1200, 800), Image.Resampling.LANCZOS)
            frames.append(frame)
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.9), constrained_layout=True)
    for ax, frame, idx, letter in zip(axes.flat, frames, indices, "abcd"):
        ax.imshow(frame)
        ax.set_axis_off()
        ax.set_title(f"Animation frame {idx + 1}/{n}", fontsize=7, pad=2)
        panel_label(ax, letter)
    save_figure(fig, out_dir, "fig_5_2_animation_frames", dpi=300)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path(r"C:\Users\86198\Desktop\ww"))
    parser.add_argument("--out-dir", type=Path, default=ROOT / "tmp" / "ww_report" / "figures")
    args = parser.parse_args()
    apply_style()

    waypoints = pd.read_csv(args.data_dir / "variable_invert_ik_waypoints.csv")
    tcp_df = pd.read_csv(args.data_dir / "variable_invert_tcp_poses.csv")
    validation = json.loads((args.data_dir / "variable_invert_validation.json").read_text(encoding="utf-8"))

    robot, _ = load_official_fr5_robot(ROOT)
    robot = robot.with_base([0.0, -0.25, 0.20], yaw=np.pi)
    targets, _, _ = make_targets(
        validation["length_m"],
        validation["stations"],
        24,
        validation["stand_off_m"],
    )
    q_waypoints = waypoints[[f"j{i}" for i in range(1, 7)]].to_numpy()
    profile = make_segmented_ruckig_time_profile(
        targets,
        q_waypoints,
        robot.limits,
        target_tcp_speed=validation["target_tcp_speed_m_s"],
        dt=0.04,
        closed_loop=False,
    )
    if len(profile.t) != len(tcp_df):
        raise RuntimeError(f"Reconstructed profile length {len(profile.t)} != CSV length {len(tcp_df)}")

    workflow_figure(args.out_dir)
    geometry_figures(args.out_dir)
    ik_figures(args.out_dir, waypoints, robot)
    stats = trajectory_figures(args.out_dir, tcp_df, profile, validation, robot)
    animation_plate(args.out_dir, args.data_dir / "variable_invert_motion_4view.gif")

    stats["validation"] = validation
    stats["figure_backend"] = "Python/matplotlib"
    stats["source_data"] = [
        "variable_invert_ik_waypoints.csv",
        "variable_invert_tcp_poses.csv",
        "variable_invert_validation.json",
        "variable_invert_motion_4view.gif",
    ]
    (args.out_dir.parent / "derived_statistics.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
