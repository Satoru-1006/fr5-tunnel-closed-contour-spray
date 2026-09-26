from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.path import Path as PlotPath
import numpy as np
from scipy.interpolate import CubicHermiteSpline

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.official_fairino import load_official_fr5_robot


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def xyz(rows: list[dict[str, str]]) -> np.ndarray:
    return np.asarray([[float(row[key]) for key in ("x", "y", "z")] for row in rows], dtype=float)


def resample_uniform_time(
    t: np.ndarray,
    q: np.ndarray,
    dq: np.ndarray,
    max_step_deg: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Resample the Ruckig result at uniform time using its position and velocity data."""
    unwrapped = np.unwrap(q, axis=0)
    spline = CubicHermiteSpline(t, unwrapped, dq, axis=0)
    max_velocity = max(float(np.max(np.abs(dq))), 1e-9)
    duration = float(t[-1] - t[0])
    frame_count = max(360, int(np.ceil(duration * max_velocity / np.deg2rad(max_step_deg))) + 1)
    while True:
        render_t = np.linspace(float(t[0]), float(t[-1]), frame_count)
        rendered = np.asarray(spline(render_t), dtype=float)
        max_step = float(np.rad2deg(np.max(np.abs(np.diff(rendered, axis=0)))))
        if max_step <= max_step_deg + 1e-9:
            break
        frame_count = int(np.ceil(frame_count * max_step / max_step_deg)) + 1
    source_phase = np.interp(render_t, t, np.arange(len(t), dtype=float))
    return rendered, source_phase, render_t


def interpolate_path(points: np.ndarray, phase: np.ndarray) -> np.ndarray:
    lo = np.floor(phase).astype(int)
    hi = np.minimum(lo + 1, len(points) - 1)
    alpha = (phase - lo)[:, None]
    return points[lo] * (1.0 - alpha) + points[hi] * alpha


def link_samples(points: np.ndarray, samples_per_link: int = 20) -> np.ndarray:
    samples = []
    alpha = np.linspace(0.0, 1.0, samples_per_link)[:, None]
    for start, end in zip(points[:-1], points[1:]):
        samples.append(start * (1.0 - alpha) + end * alpha)
    return np.vstack(samples)


def retime_gif(path: Path, target_duration_s: float) -> tuple[int, float]:
    """Set distributed 10 ms GIF delays without dropping or duplicating frames."""
    data = bytearray(path.read_bytes())
    delay_positions: list[int] = []
    if data[:6] not in (b"GIF87a", b"GIF89a"):
        raise RuntimeError(f"Not a GIF file: {path}")
    cursor = 13
    packed = data[10]
    if packed & 0x80:
        cursor += 3 * (2 ** ((packed & 0x07) + 1))

    def skip_subblocks(position: int) -> int:
        while True:
            size = data[position]
            position += 1
            if size == 0:
                return position
            position += size

    while cursor < len(data):
        marker = data[cursor]
        if marker == 0x3B:
            break
        if marker == 0x21:
            label = data[cursor + 1]
            if label == 0xF9:
                block_size = data[cursor + 2]
                if block_size != 4 or data[cursor + 7] != 0:
                    raise RuntimeError(f"Malformed GIF graphic-control block at byte {cursor}")
                delay_positions.append(cursor + 4)
                cursor += 8
            else:
                cursor = skip_subblocks(cursor + 2)
            continue
        if marker == 0x2C:
            image_packed = data[cursor + 9]
            cursor += 10
            if image_packed & 0x80:
                cursor += 3 * (2 ** ((image_packed & 0x07) + 1))
            cursor += 1  # LZW minimum code size
            cursor = skip_subblocks(cursor)
            continue
        raise RuntimeError(f"Unexpected GIF block marker 0x{marker:02x} at byte {cursor}")
    if not delay_positions:
        raise RuntimeError(f"No GIF graphic-control blocks found in {path}")
    target_centiseconds = max(len(delay_positions), int(round(target_duration_s * 100.0)))
    base_delay = target_centiseconds // len(delay_positions)
    extra = target_centiseconds - base_delay * len(delay_positions)
    accumulator = 0
    written_centiseconds = 0
    for position in delay_positions:
        delay = base_delay
        accumulator += extra
        if accumulator >= len(delay_positions):
            delay += 1
            accumulator -= len(delay_positions)
        data[position : position + 2] = int(delay).to_bytes(2, "little")
        written_centiseconds += delay
    path.write_bytes(data)
    return len(delay_positions), written_centiseconds / 100.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Render the strict MoveIt open-arch path as an inside-tunnel wiper motion.")
    parser.add_argument(
        "--trajectory",
        type=Path,
        default=ROOT / "outputs" / "open_arch_moveit_strict" / "moveit_smoothed_joint_trajectory.csv",
    )
    parser.add_argument("--outputs", type=Path, default=ROOT / "outputs")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--target-display-duration", type=float, default=15.0)
    parser.add_argument("--max-render-step-deg", type=float, default=3.0)
    parser.add_argument("--base-y", type=float, default=0.10)
    parser.add_argument("--work-plane-y", type=float, default=0.30)
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "outputs" / "animation_standard_horseshoe_wiper_multiview.gif",
    )
    args = parser.parse_args()

    trajectory_rows = read_rows(args.trajectory)
    t_source = np.asarray([float(row["t"]) for row in trajectory_rows], dtype=float)
    q_source = np.asarray([[float(row[f"j{i}_q"]) for i in range(1, 7)] for row in trajectory_rows], dtype=float)
    dq_source = np.asarray([[float(row[f"j{i}_dq"]) for i in range(1, 7)] for row in trajectory_rows], dtype=float)
    q, phase, render_t = resample_uniform_time(t_source, q_source, dq_source, args.max_render_step_deg)

    surface_all = xyz(read_rows(args.outputs / "surface_path.csv"))
    target_all = xyz(read_rows(args.outputs / "tcp_path.csv"))
    wall_contour = surface_all[:240]
    wall_open = surface_all[30:211]
    target_open = target_all[30:211]
    wall_open[:, 1] = args.work_plane_y
    target_open[:, 1] = args.work_plane_y
    wall_at_frame = interpolate_path(wall_open, phase)
    target_at_frame = interpolate_path(target_open, phase)

    robot_model, _ = load_official_fr5_robot(ROOT)
    robot = robot_model.with_base([0.0, args.base_y, 0.20], yaw=np.pi)
    tcp_offset = np.asarray([0.0, 0.0, 0.150])
    joint_points: list[np.ndarray] = []
    tcp_actual: list[np.ndarray] = []
    for qi in q:
        transforms = robot.all_joint_frames(qi)
        flange = transforms[-1]
        tcp = flange[:3, 3] + flange[:3, :3] @ tcp_offset
        joint_points.append(np.vstack([transforms[:, :3, 3], tcp]))
        tcp_actual.append(tcp)
    tcp_actual_array = np.asarray(tcp_actual)

    polygon = PlotPath(np.column_stack([wall_contour[:, 0], wall_contour[:, 2]]), closed=True)
    outside_frames = 0
    min_wall_margin_proxy = np.inf
    for points in joint_points:
        samples = link_samples(points[:-1])
        inside = polygon.contains_points(samples[:, [0, 2]], radius=1e-8)
        outside_frames += int(not np.all(inside))
        # Conservative distance to sampled wall contour, useful as a visual-clearance proxy.
        distances = np.linalg.norm(samples[:, None, [0, 2]] - wall_contour[None, :, [0, 2]], axis=2)
        min_wall_margin_proxy = min(min_wall_margin_proxy, float(np.min(distances)))

    q_step_deg = np.rad2deg(np.abs(np.diff(q, axis=0)))
    max_adjacent_step = float(np.max(q_step_deg))
    q_range_deg = np.rad2deg(np.ptp(q, axis=0))
    tcp_frame_step = np.linalg.norm(np.diff(tcp_actual_array, axis=0), axis=1)
    tcp_step_p05 = float(np.percentile(tcp_frame_step, 5))
    tcp_step_p95 = float(np.percentile(tcp_frame_step, 95))
    tcp_step_fluctuation = float((tcp_step_p95 - tcp_step_p05) / max(np.mean(tcp_frame_step), 1e-12))

    views = [("Front", 18, -90), ("Left-front", 24, -63), ("Right-front", 24, -117), ("Top (plan)", 90, -90)]
    fig = plt.figure(figsize=(14.5, 8.8), facecolor="white")
    artists = []
    legend_handles = None
    legend_labels = None
    for subplot, (title, elevation, azimuth) in enumerate(views, 1):
        ax = fig.add_subplot(2, 2, subplot, projection="3d")
        ax.set_facecolor("white")
        ax.set_proj_type("ortho")
        # One physical work section is shown. Sparse longitudinal guides retain
        # the tunnel context without creating three competing blue rings.
        wall_line, = ax.plot(
            wall_contour[:, 0],
            np.full(len(wall_contour), 0.30),
            wall_contour[:, 2],
            color="#245b8a",
            lw=1.8,
            ls="--",
            alpha=0.78,
            label="tunnel wall section",
        )
        for contour_index in np.linspace(0, len(wall_contour) - 1, 14).round().astype(int):
            ax.plot(
                [wall_contour[contour_index, 0]] * 2,
                [0.0, 0.90],
                [wall_contour[contour_index, 2]] * 2,
                color="#245b8a",
                lw=0.7,
                alpha=0.10,
            )
        target_line, = ax.plot(
            target_open[:, 0],
            target_open[:, 1],
            target_open[:, 2],
            color="#c62828",
            lw=2.4,
            label="TCP path (0.26 m stand-off)",
        )
        connector = None
        for path_index in range(0, len(target_open), 30):
            line, = ax.plot(
                [target_open[path_index, 0], wall_open[path_index, 0]],
                [target_open[path_index, 1], wall_open[path_index, 1]],
                [target_open[path_index, 2], wall_open[path_index, 2]],
                color="#ef6c00",
                lw=0.9,
                ls=":",
                alpha=0.45,
                label="stand-off direction" if connector is None else None,
            )
            connector = line if connector is None else connector
        arm, = ax.plot([], [], [], "-o", color="#202124", lw=2.8, markersize=4.2, label="FR5")
        tool, = ax.plot([], [], [], color="#7b1fa2", lw=4.0, label="150 mm virtual TCP")
        spray, = ax.plot([], [], [], "--", color="#ef6c00", lw=2.2, label="active spray direction")
        tcp_dot, = ax.plot([], [], [], "o", color="#d32f2f", markersize=7)
        trace, = ax.plot([], [], [], color="#2e7d32", lw=2.2, label="completed spray path")
        ax.plot([0.0], [args.base_y], [0.20], marker="s", color="#202124", markersize=5, label="robot base")
        ax.set_xlim(-0.68, 0.68); ax.set_ylim(0.0, 0.90); ax.set_zlim(0.0, 1.15)
        ax.set_xlabel("X / m"); ax.set_ylabel("Y / m"); ax.set_zlabel("Z / m")
        ax.set_title(title); ax.view_init(elev=elevation, azim=azimuth)
        if subplot == 4:
            ax.set_zlabel("")
            ax.set_zticks([])
            ax.plot([-0.60, 0.60], [args.work_plane_y, args.work_plane_y], [0.0, 0.0], color="#245b8a", lw=1.0, alpha=0.45)
            ax.text(0.61, args.work_plane_y, 0.0, "work section", color="#245b8a", fontsize=7)
            ax.text(0.02, args.base_y, 0.20, "base", color="#202124", fontsize=7)
        if legend_handles is None:
            legend_handles = [wall_line, target_line, connector, arm, tool, spray, trace]
            legend_labels = [
                "Tunnel wall section",
                "TCP path (0.26 m stand-off)",
                "Stand-off direction",
                "FR5",
                "150 mm virtual TCP",
                "Active spray direction",
                "Completed spray path",
            ]
        artists.append((arm, tool, spray, tcp_dot, trace))

    status = fig.text(
        0.5,
        0.965,
        "",
        ha="center",
        va="top",
        fontsize=12,
        bbox={"facecolor": "white", "edgecolor": "#dadce0", "alpha": 0.95, "boxstyle": "round,pad=0.4"},
    )
    fig.subplots_adjust(left=0.045, right=0.825, bottom=0.07, top=0.90, wspace=0.12, hspace=0.20)
    fig.legend(
        legend_handles,
        legend_labels,
        loc="upper right",
        bbox_to_anchor=(0.985, 0.895),
        fontsize=8.5,
        frameon=True,
        framealpha=0.96,
        edgecolor="#c8cdd2",
        title="Figure legend",
        title_fontsize=9,
    )
    fig.text(
        0.43,
        0.018,
        "MoveIt2 + Ruckig trajectory | orthographic views | blue wall and red TCP path are separated by the physical 0.26 m stand-off",
        ha="center",
        fontsize=9.5,
        color="#333333",
    )

    def update(frame_index: int):
        points = joint_points[frame_index]
        flange, tcp = points[-2], points[-1]
        direction = wall_at_frame[frame_index] - target_at_frame[frame_index]
        direction /= max(float(np.linalg.norm(direction)), 1e-12)
        spray_tip = tcp + direction * 0.12
        returned = []
        for arm, tool, spray, tcp_dot, trace in artists:
            arm.set_data(points[:-1, 0], points[:-1, 1]); arm.set_3d_properties(points[:-1, 2])
            tool.set_data([flange[0], tcp[0]], [flange[1], tcp[1]]); tool.set_3d_properties([flange[2], tcp[2]])
            spray.set_data([tcp[0], spray_tip[0]], [tcp[1], spray_tip[1]]); spray.set_3d_properties([tcp[2], spray_tip[2]])
            tcp_dot.set_data([tcp[0]], [tcp[1]]); tcp_dot.set_3d_properties([tcp[2]])
            trace.set_data(tcp_actual_array[: frame_index + 1, 0], tcp_actual_array[: frame_index + 1, 1])
            trace.set_3d_properties(tcp_actual_array[: frame_index + 1, 2])
            returned.extend([arm, tool, spray, tcp_dot, trace])
        status.set_text(
            f"Inside-tunnel constant-speed wiper | source t {render_t[frame_index]:07.2f} s "
            f"| phase {phase[frame_index] + 30:06.2f} | SPRAY ON"
        )
        return (*returned, status)

    animation = FuncAnimation(fig, update, frames=len(q), interval=1000 / args.fps, blit=False)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    try:
        animation.save(args.out, writer=PillowWriter(fps=args.fps))
    finally:
        plt.close(fig)
    gif_frame_count, gif_duration_s = retime_gif(args.out, args.target_display_duration)

    validation = {
        "source_trajectory": str(args.trajectory.resolve()),
        "source_waypoint_count": len(q_source),
        "rendered_frame_count": len(q),
        "source_duration_s": float(t_source[-1] - t_source[0]),
        "display_duration_s": gif_duration_s,
        "gif_frame_count": gif_frame_count,
        "gif_timing": "distributed 10/20 ms delays; no rendered frames dropped",
        "time_sampling": "uniform source-time sampling with CubicHermiteSpline(q,dq)",
        "process_path": "left vertical -> upper arch -> right vertical; open bottom",
        "motion_semantics": "inside-tunnel wiper-like sweep; no bottom closure or return rotation",
        "max_rendered_adjacent_joint_step_deg": max_adjacent_step,
        "rendered_tcp_step_p05_m": tcp_step_p05,
        "rendered_tcp_step_p95_m": tcp_step_p95,
        "rendered_tcp_step_p05_p95_fluctuation": tcp_step_fluctuation,
        "joint_range_deg": {f"j{i + 1}": float(value) for i, value in enumerate(q_range_deg)},
        "j1_full_rotation": bool(q_range_deg[0] >= 300.0),
        "j6_full_rotation": bool(q_range_deg[5] >= 300.0),
        "simple_wall_outside_frame_count": outside_frames,
        "sampled_min_link_to_wall_distance_m": min_wall_margin_proxy,
        "virtual_tcp_xyz_m": [0.0, 0.0, 0.150],
        "robot_base_xyz_m": [0.0, args.base_y, 0.20],
        "work_section_y_m": args.work_plane_y,
        "camera_projection": "orthographic; top view is true plan view",
        "legend_layout": "figure-level legend in reserved upper-right margin",
        "status": "pass" if max_adjacent_step <= args.max_render_step_deg + 1e-9 and outside_frames == 0 else "fail",
    }
    validation_path = args.out.with_name(args.out.stem + "_validation.json")
    validation_path.write_text(json.dumps(validation, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.out.resolve()} with {len(q)} frames")
    print(json.dumps(validation, ensure_ascii=False))


if __name__ == "__main__":
    main()
