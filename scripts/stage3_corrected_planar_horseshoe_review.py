from __future__ import annotations

"""Render the corrected planar horseshoe spray task for human review.

This is deliberately separate from the frozen Stage 3 H10 multi-family audit.
The authoritative task here is the 181-point Stage 0/1 open-arch pair: one
fixed tunnel section, one constant work-plane coordinate, and spray ON for
the ordered arch path.  The bottom edge is drawn as a tunnel-wall reference,
but it is not added to the Stage 0/1 open-arch trajectory.

The native MoveIt2 evidence is reused only after its source files and pass
reports are checked.  No controller, hardware, or physical motion is used.

The authoritative normal is the TCP-to-wall direction.  Therefore the wall
surface is ``tcp + stand_off * normal`` and the TCP remains on the tunnel
interior side of the lining.
"""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
from scipy.interpolate import CubicHermiteSpline

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.official_fairino import load_official_fr5_robot


STAGE01_POSES = ROOT / "outputs" / "internal_wiper_moveit_inputs" / "open_arch_tcp_poses_base_link.csv"
STAGE01_SEEDS = ROOT / "outputs" / "internal_wiper_moveit_inputs" / "open_arch_seed_joints.csv"
STRICT_DIR = ROOT / "outputs" / "open_arch_moveit_strict"
STRICT_TRAJECTORY = STRICT_DIR / "moveit_smoothed_joint_trajectory.csv"
STRICT_FK = STRICT_DIR / "moveit_fk_tcp_trace.csv"
STRICT_ACCEPTANCE = STRICT_DIR / "final_acceptance_summary.json"
STRICT_COLLISION = STRICT_DIR / "moveit_collision_report.csv"
STAND_OFF_M = 0.26
TCP_OFFSET_M = np.array([0.0, 0.0, 0.15], dtype=float)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def xyz(rows: list[dict[str, str]]) -> np.ndarray:
    return np.asarray([[float(row[key]) for key in ("x", "y", "z")] for row in rows], dtype=float)


def normal(rows: list[dict[str, str]]) -> np.ndarray:
    result = np.asarray([[float(row[key]) for key in ("nx", "ny", "nz")] for row in rows], dtype=float)
    result /= np.maximum(np.linalg.norm(result, axis=1, keepdims=True), 1e-12)
    return result


def joints(rows: list[dict[str, str]]) -> np.ndarray:
    return np.asarray([[float(row[f"j{i}_q"]) for i in range(1, 7)] for row in rows], dtype=float)


def velocities(rows: list[dict[str, str]]) -> np.ndarray:
    return np.asarray([[float(row[f"j{i}_dq"]) for i in range(1, 7)] for row in rows], dtype=float)


def resample_uniform_time(
    t: np.ndarray,
    q: np.ndarray,
    dq: np.ndarray,
    max_step_deg: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    spline = CubicHermiteSpline(t, np.unwrap(q, axis=0), dq, axis=0)
    duration = float(t[-1] - t[0])
    max_velocity = max(float(np.max(np.abs(dq))), 1e-9)
    frame_count = max(120, int(np.ceil(duration * max_velocity / np.deg2rad(max_step_deg))) + 1)
    while True:
        render_t = np.linspace(float(t[0]), float(t[-1]), frame_count)
        rendered = np.asarray(spline(render_t), dtype=float)
        max_step = float(np.rad2deg(np.max(np.abs(np.diff(rendered, axis=0)))))
        if max_step <= max_step_deg + 1e-9:
            break
        frame_count = int(np.ceil(frame_count * max_step / max_step_deg)) + 1
    phase = np.interp(render_t, t, np.arange(len(t), dtype=float))
    return rendered, phase, render_t


def interpolate(points: np.ndarray, phase: np.ndarray) -> np.ndarray:
    lo = np.floor(phase).astype(int)
    hi = np.minimum(lo + 1, len(points) - 1)
    alpha = (phase - lo)[:, None]
    return points[lo] * (1.0 - alpha) + points[hi] * alpha


def retime_gif(path: Path, target_duration_s: float) -> tuple[int, float]:
    data = bytearray(path.read_bytes())
    if data[:6] not in (b"GIF87a", b"GIF89a"):
        raise RuntimeError(f"not a GIF: {path}")
    delay_positions: list[int] = []
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
            if data[cursor + 1] == 0xF9:
                if data[cursor + 2] != 4 or data[cursor + 7] != 0:
                    raise RuntimeError("malformed GIF graphic-control extension")
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
            cursor += 1
            cursor = skip_subblocks(cursor)
            continue
        raise RuntimeError(f"unexpected GIF marker 0x{marker:02x}")

    if not delay_positions:
        raise RuntimeError("GIF contains no frame delays")
    total_centiseconds = max(len(delay_positions), int(round(target_duration_s * 100.0)))
    base = total_centiseconds // len(delay_positions)
    extra = total_centiseconds - base * len(delay_positions)
    accumulator = 0
    written = 0
    for position in delay_positions:
        delay = base
        accumulator += extra
        if accumulator >= len(delay_positions):
            delay += 1
            accumulator -= len(delay_positions)
        data[position : position + 2] = int(delay).to_bytes(2, "little")
        written += delay
    path.write_bytes(data)
    return len(delay_positions), written / 100.0


def standard_wall_outline(surface: np.ndarray, work_y: float) -> np.ndarray:
    """Build a clean wall outline from the 181-point open-arch dimensions."""
    radius = float(max(abs(surface[:, 0].min()), abs(surface[:, 0].max())))
    top = float(surface[:, 2].max())
    spring = top - radius
    fillet = min(0.02, radius * 0.20, spring * 0.20)
    left_up = np.column_stack([np.full(24, -radius), np.linspace(fillet, spring, 24, endpoint=False)])
    theta = np.linspace(np.pi, 0.0, 96, endpoint=False)
    arch = np.column_stack([radius * np.cos(theta), spring + radius * np.sin(theta)])
    right_down = np.column_stack([np.full(24, radius), np.linspace(spring, fillet, 24, endpoint=False)])
    bottom_right = np.column_stack([np.linspace(radius, -radius, 40, endpoint=False), np.full(40, 0.0)])
    section = np.vstack([left_up, arch, right_down, bottom_right])
    return np.column_stack([section[:, 0], np.full(len(section), work_y), section[:, 1]])


def link_samples(points: np.ndarray, samples_per_link: int = 14) -> np.ndarray:
    alpha = np.linspace(0.0, 1.0, samples_per_link)[:, None]
    pieces = [a * (1.0 - alpha) + b * alpha for a, b in zip(points[:-1], points[1:])]
    return np.vstack(pieces)


def set_axes(ax, title: str, view: tuple[float, float]) -> None:
    ax.set_proj_type("ortho")
    ax.set_xlim(-0.62, 0.62)
    ax.set_ylim(-0.72, 0.12)
    ax.set_zlim(-0.08, 1.18)
    ax.set_xlabel("X / m")
    ax.set_ylabel("Y / m")
    ax.set_zlabel("Z / m")
    ax.set_title(title, fontsize=11, pad=8)
    ax.view_init(elev=view[0], azim=view[1])
    if view[0] == 0.0:
        ax.set_ylabel("")
        ax.set_yticks([])
    if view[0] == 90.0:
        ax.set_zlabel("")
        ax.set_zticks([])


def draw_scene(
    ax,
    wall_outline: np.ndarray,
    target: np.ndarray,
    surface: np.ndarray,
    frames: list[np.ndarray],
    tcp: np.ndarray,
    title: str,
    view: tuple[float, float],
    frame_index: int,
    complete: bool,
    legend: bool = False,
) -> list[object]:
    wall_line, = ax.plot(wall_outline[:, 0], wall_outline[:, 1], wall_outline[:, 2], "--", color="#346b95", lw=1.8, label="tunnel wall section")
    floor_line, = ax.plot([float(wall_outline[:, 0].min()), float(wall_outline[:, 0].max())], [wall_outline[0, 1], wall_outline[0, 1]], [0.0, 0.0], color="#9aa0a6", lw=1.4, label="bottom wall reference")
    target_line, = ax.plot(target[:, 0], target[:, 1], target[:, 2], color="#cf2f2f", lw=2.7, label="TCP spray path (SPRAY ON)")
    for index in np.linspace(0, len(target) - 1, 9).round().astype(int):
        ax.plot(
            [target[index, 0], surface[index, 0]],
            [target[index, 1], surface[index, 1]],
            [target[index, 2], surface[index, 2]],
            ":",
            color="#ef8a17",
            lw=0.8,
            alpha=0.6,
            label="0.26 m stand-off" if index == 0 else None,
        )
    ax.scatter([target[0, 0]], [target[0, 1]], [target[0, 2]], s=32, color="#1565c0", zorder=6, label="start")
    ax.scatter([target[-1, 0]], [target[-1, 1]], [target[-1, 2]], s=32, color="#7b1fa2", zorder=6, label="end")
    points = frames[frame_index]
    arm, = ax.plot(points[:, 0], points[:, 1], points[:, 2], "-o", color="#202124", lw=2.8, markersize=4.4, label="FR5 FK")
    flange, tcp_line = points[-2], tcp[frame_index]
    tool, = ax.plot([flange[0], tcp_line[0]], [flange[1], tcp_line[1]], [flange[2], tcp_line[2]], color="#7b1fa2", lw=4.0, label="150 mm TCP tool")
    tcp_dot, = ax.plot([tcp_line[0]], [tcp_line[1]], [tcp_line[2]], "o", color="#d32f2f", markersize=7, label="current TCP")
    if complete:
        trace, = ax.plot(tcp[:, 0], tcp[:, 1], tcp[:, 2], color="#2e7d32", lw=2.2, label="validated TCP trace")
    else:
        trace, = ax.plot(tcp[: frame_index + 1, 0], tcp[: frame_index + 1, 1], tcp[: frame_index + 1, 2], color="#2e7d32", lw=2.2, label="completed spray path")
    spray_tip = tcp_line + (surface[min(frame_index, len(surface) - 1)] - tcp_line) * 0.45
    spray, = ax.plot([tcp_line[0], spray_tip[0]], [tcp_line[1], spray_tip[1]], [tcp_line[2], spray_tip[2]], "--", color="#ef6c00", lw=2.0, label="spray direction")
    ax.plot([0.0], [0.0], [0.0], marker="s", color="#202124", markersize=5, label="robot base")
    ax.text(0.0, 0.0, 0.0, " base", color="#202124", fontsize=8)
    set_axes(ax, title, view)
    if legend:
        ax.legend(loc="upper left", fontsize=7.0, framealpha=0.95)
    return [arm, tool, tcp_dot, trace, spray, wall_line, target_line, floor_line]


def main() -> int:
    parser = argparse.ArgumentParser(description="Render the corrected fixed-plane horseshoe spray task.")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--start", choices=("right", "left"), default="right", help="Validated source direction or reversed display direction.")
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--display-duration", type=float, default=15.0)
    parser.add_argument("--max-render-step-deg", type=float, default=3.0)
    parser.add_argument("--desktop-copy", action="store_true")
    args = parser.parse_args()

    if args.out_dir is None:
        tag = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        args.out_dir = ROOT / "outputs" / f"stage3_corrected_planar_horseshoe_review_{tag}"
    args.out_dir.mkdir(parents=True, exist_ok=True)

    required = [STAGE01_POSES, STAGE01_SEEDS, STRICT_TRAJECTORY, STRICT_FK, STRICT_ACCEPTANCE, STRICT_COLLISION]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("missing required evidence: " + ", ".join(missing))

    pose_rows = read_csv(STAGE01_POSES)
    seed_rows = read_csv(STAGE01_SEEDS)
    trajectory_rows = read_csv(STRICT_TRAJECTORY)
    fk_rows = read_csv(STRICT_FK)
    acceptance = json.loads(STRICT_ACCEPTANCE.read_text(encoding="utf-8"))
    if len(pose_rows) != 181 or len(seed_rows) != 181 or len(trajectory_rows) != 181 or len(fk_rows) != 181:
        raise RuntimeError("corrected task requires exactly 181 pose, seed, trajectory, and FK rows")
    if acceptance.get("overall_status") != "pass":
        raise RuntimeError("existing native MoveIt2 strict acceptance is not pass")
    collision_rows = read_csv(STRICT_COLLISION)
    collision_metrics = {row["metric"]: row["value"] for row in collision_rows}
    if collision_metrics.get("status") != "pass" or float(collision_metrics.get("collision_count", "1")) != 0.0:
        raise RuntimeError("existing native collision evidence is not a zero-collision pass")

    poses_input = xyz(pose_rows)
    normals = normal(pose_rows)
    input_work_y = float(np.median(poses_input[:, 1]))
    if float(np.max(np.abs(poses_input[:, 1] - input_work_y))) > 1e-9:
        raise RuntimeError("authoritative open-arch input is not a fixed work plane")
    if float(np.max(np.abs(normals[:, 1]))) > 1e-7:
        raise RuntimeError("authoritative open-arch normals are not in the x-z section plane")

    t = np.asarray([float(row["t"]) for row in trajectory_rows], dtype=float)
    q = joints(trajectory_rows)
    dq = velocities(trajectory_rows)
    native_tcp = np.asarray([[float(row[key]) for key in ("actual_tcp_x", "actual_tcp_y", "actual_tcp_z")] for row in fk_rows], dtype=float)
    work_y = float(np.median(native_tcp[:, 1]))
    if float(np.max(np.abs(native_tcp[:, 1] - work_y))) > 2e-3:
        raise RuntimeError("native strict replay is not a fixed work plane")
    # The frozen Stage 0/1 pair is expressed at y=-0.10 m while the strict
    # MoveIt replay is expressed at y=-0.55 m.  This is a rigid translation
    # along the tunnel axis, not a spatial transfer segment; align only the
    # display frame and retain both coordinates in the report.
    poses = poses_input.copy()
    poses[:, 1] = work_y
    # In the authoritative pair, normal points from the TCP toward the wall.
    # The previous visual used the opposite sign and made the TCP look like a
    # second, outside tunnel.  Keep the TCP fixed and place the wall outward.
    surface = poses + STAND_OFF_M * normals
    robot_model, _ = load_official_fr5_robot(ROOT)
    robot = robot_model.with_base([0.0, 0.0, 0.0], yaw=0.0)
    native_fk_tcp = []
    native_frames = []
    for qi in q:
        transform = robot.all_joint_frames(qi)
        native_frames.append(np.vstack([transform[:, :3, 3], transform[-1, :3, 3] + transform[-1, :3, :3] @ TCP_OFFSET_M]))
        native_fk_tcp.append(native_frames[-1][-1])
    native_fk_tcp = np.asarray(native_fk_tcp)
    fk_to_authoritative_m = np.linalg.norm(native_tcp - poses, axis=1)
    if float(np.max(fk_to_authoritative_m)) > 2e-3:
        raise RuntimeError(f"native FK does not reproduce authoritative TCP path; max error {np.max(fk_to_authoritative_m):.6f} m")

    order = np.arange(len(q)) if args.start == "right" else np.arange(len(q) - 1, -1, -1)
    q_display = q[order]
    dq_display = dq[order]
    t_display = t[order]
    target_display = poses[order]
    surface_display = surface[order]
    frames_display = [native_frames[index] for index in order]
    fk_tcp_display = native_fk_tcp[order]
    if args.start == "left":
        # Time reversal is used for a display-only reverse traversal. The
        # native pass evidence remains attached to the original right->left
        # source order and is not relabelled as a new Ruckig certification.
        t_display = np.linspace(float(t[0]), float(t[-1]), len(t_display))
        dq_display = -dq_display

    q_render, phase, render_t = resample_uniform_time(t_display, q_display, dq_display, args.max_render_step_deg)
    frame_indices = np.clip(np.rint(phase).astype(int), 0, len(frames_display) - 1)
    render_frames = [frames_display[index] for index in frame_indices]
    render_target = interpolate(target_display, phase)
    render_surface = interpolate(surface_display, phase)
    render_tcp = np.asarray([robot.all_joint_frames(qi)[-1, :3, 3] + robot.all_joint_frames(qi)[-1, :3, :3] @ TCP_OFFSET_M for qi in q_render])
    wall_outline = standard_wall_outline(surface, work_y)

    frame_tcp_error = np.linalg.norm(render_tcp - render_target, axis=1)
    q_step_deg = np.rad2deg(np.max(np.abs(np.diff(q_render, axis=0)), axis=1))
    all_link_samples = np.vstack([link_samples(frame[:-1]) for frame in render_frames])
    wall_xz = wall_outline[:, [0, 2]]
    link_to_wall = np.linalg.norm(all_link_samples[:, None, [0, 2]] - wall_xz[None, :, :], axis=2)

    overview_frame = len(render_frames) // 2
    views = [("Front section: tunnel wall + TCP", 0.0, -90.0), ("Oblique: robot + planar TCP", 22.0, -70.0), ("Left oblique", 22.0, -110.0), ("Plan view: constant-y work plane", 90.0, -90.0)]
    fig = plt.figure(figsize=(15.0, 9.0), facecolor="white")
    for index, (title, elev, azim) in enumerate(views, start=1):
        ax = fig.add_subplot(2, 2, index, projection="3d")
        draw_scene(
            ax,
            wall_outline,
            target_display,
            surface_display,
            render_frames,
            render_tcp,
            title,
            (elev, azim),
            overview_frame,
            complete=True,
            legend=index == 1,
        )
        if index == 1:
            ax.text2D(0.02, 0.97, "RED = only spray path\nBLUE dashed = tunnel wall\nGRAY = bottom reference (not sprayed)", transform=ax.transAxes, va="top", fontsize=8.5, bbox={"facecolor": "white", "alpha": 0.88, "edgecolor": "#d0d7de"})
    fig.suptitle(
        f"CORRECTED TASK | FIXED-PLANE STANDARD HORSESHOE SPRAY | START {'RIGHT LOWER' if args.start == 'right' else 'LEFT LOWER'} | SPRAY ON",
        fontsize=16,
        y=0.985,
    )
    fig.text(0.5, 0.018, f"181 points | plane y = {work_y:.3f} m | TCP stand-off = {STAND_OFF_M:.2f} m | no spatial transfer segments", ha="center", fontsize=10, color="#333333")
    fig.subplots_adjust(left=0.035, right=0.80, bottom=0.06, top=0.92, wspace=0.12, hspace=0.22)
    overview_path = args.out_dir / "corrected_planar_horseshoe_overview.png"
    fig.savefig(overview_path, dpi=170, facecolor="white")
    plt.close(fig)

    animation_indices = np.linspace(0, len(render_frames) - 1, min(60, len(render_frames))).round().astype(int)
    animation_frames = [render_frames[index] for index in animation_indices]
    animation_tcp = render_tcp[animation_indices]
    animation_surface = render_surface[animation_indices]
    animation_t = render_t[animation_indices]
    animation_views = views[:2]

    fig = plt.figure(figsize=(12.0, 6.0), dpi=78, facecolor="white")
    artists = []
    for index, (title, elev, azim) in enumerate(animation_views, start=1):
        ax = fig.add_subplot(1, 2, index, projection="3d")
        items = draw_scene(ax, wall_outline, target_display, surface_display, animation_frames, animation_tcp, title, (elev, azim), 0, complete=False, legend=index == 1)
        artists.append((ax, items))
    status = fig.text(0.5, 0.925, "", ha="center", va="top", fontsize=11, bbox={"facecolor": "white", "edgecolor": "#dadce0", "alpha": 0.95, "boxstyle": "round,pad=0.4"})
    fig.suptitle("CORRECTED PLANAR HORSESHOE SPRAY | TCP path stays on one tunnel section", fontsize=15, y=0.98)
    fig.text(0.5, 0.018, "RED: target spray path | GREEN: completed spray | ORANGE: stand-off direction | robot follows this plane only", ha="center", fontsize=9.5, color="#333333")
    fig.subplots_adjust(left=0.035, right=0.98, bottom=0.08, top=0.88, wspace=0.16)

    def update(frame_index: int):
        returned: list[object] = []
        for ax, items in artists:
            points = animation_frames[frame_index]
            arm, tool, tcp_dot, trace, spray = items[:5]
            flange, tcp_point = points[-2], animation_tcp[frame_index]
            arm.set_data(points[:, 0], points[:, 1]); arm.set_3d_properties(points[:, 2])
            tool.set_data([flange[0], tcp_point[0]], [flange[1], tcp_point[1]]); tool.set_3d_properties([flange[2], tcp_point[2]])
            tcp_dot.set_data([tcp_point[0]], [tcp_point[1]]); tcp_dot.set_3d_properties([tcp_point[2]])
            trace.set_data(animation_tcp[: frame_index + 1, 0], animation_tcp[: frame_index + 1, 1]); trace.set_3d_properties(animation_tcp[: frame_index + 1, 2])
            spray_tip = tcp_point + (animation_surface[frame_index] - tcp_point) * 0.45
            spray.set_data([tcp_point[0], spray_tip[0]], [tcp_point[1], spray_tip[1]]); spray.set_3d_properties([tcp_point[2], spray_tip[2]])
            returned.extend([arm, tool, tcp_dot, trace, spray])
        status.set_text(f"Fixed x-z tunnel section | frame {frame_index:03d}/{len(animation_frames)-1} | t {animation_t[frame_index]:.2f} s | SPRAY ON")
        return (*returned, status)

    animation = FuncAnimation(fig, update, frames=len(animation_frames), interval=1000 / args.fps, blit=False)
    gif_path = args.out_dir / "corrected_planar_horseshoe_robot.gif"
    try:
        animation.save(gif_path, writer=PillowWriter(fps=args.fps))
    finally:
        plt.close(fig)
    gif_frames, gif_duration = retime_gif(gif_path, args.display_duration)

    report = {
        "schema_version": "stage3-corrected-planar-horseshoe-review-v1",
        "status": "PASS_FOR_VISUAL_REVIEW",
        "task_geometry": {
            "name": "standard_horseshoe_fixed_plane_open_arch",
            "description": "one constant-y tunnel section; TCP stays inside the lining and follows right/left lower spring line -> side wall -> upper arch -> opposite side wall",
            "point_count": 181,
            "work_plane_coordinate": {"axis": "y", "source_input_value_m": input_work_y, "native_replay_value_m": work_y, "display_alignment_translation_m": work_y - input_work_y, "max_abs_variation_m": float(np.max(np.abs(poses[:, 1] - work_y)))},
            "spray_state": "SPRAY_ON for all 181 path points",
            "bottom_edge": "wall_reference_only; excluded from current Stage 0/1 ON-state open-arch trajectory",
            "start_options": ["right lower (native validated order)", "left lower (display reverse)", "arbitrary open-path index can be generated without spatial transfer segments"],
        },
        "authoritative_inputs": {str(path.relative_to(ROOT)): {"sha256": sha256(path), "rows": len(read_csv(path))} for path in (STAGE01_POSES, STAGE01_SEEDS)},
        "native_moveit2_evidence": {
            "acceptance_summary": str(STRICT_ACCEPTANCE.relative_to(ROOT)),
            "acceptance_status": acceptance.get("overall_status"),
            "planning_scene_fk_dynamics_post_ruckig": "passed in existing strict runtime evidence",
            "collision_method": "adaptive_discrete_interpolation",
            "ccd_status": "not_available",
            "clearance_status": "not_available",
            "collision_count": float(collision_metrics.get("collision_count", "nan")),
            "native_fk_max_tcp_reproduction_error_m": float(np.max(fk_to_authoritative_m)),
        },
        "render": {
            "direction": args.start,
            "display_only_reversal": bool(args.start == "left"),
            "gif": str(gif_path.resolve()),
            "gif_frame_count": gif_frames,
            "gif_display_duration_s": gif_duration,
            "overview": str(overview_path.resolve()),
            "max_rendered_joint_step_deg": float(np.max(q_step_deg)),
            "max_fk_tcp_to_target_m": float(np.max(frame_tcp_error)),
            "visual_outline_distance_proxy_m": float(np.min(link_to_wall)),
        },
        "scope_guard": {
            "frozen_h10_dataset_modified": False,
            "h10_family_used_as_task_geometry": False,
            "legacy_720_path_used_as_stage01_input": False,
            "off_or_reorientation_states_added": False,
            "physical_robot_connected": False,
            "physical_goals_sent": 0,
            "robot_motion_started": False,
        },
    }
    report_path = args.out_dir / "corrected_planar_horseshoe_review.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md = "# Corrected planar horseshoe spray review\n\n"
    md += "This artifact replaces the misleading H10-V skeleton-only replay for visual task review. It does not modify the frozen H10 dataset.\n\n"
    md += "## Task geometry\n\n"
    md += f"- 181 authoritative points on one fixed `y` plane (source input `y = {input_work_y:.6f} m`; native replay/display `y = {work_y:.6f} m`, rigid axis translation only).\n- TCP path: one planar open horseshoe arch, spray ON throughout.\n- Blue wall outline includes the bottom wall only as context; the current Stage 0/1 contract does not add a bottom closure to the ON-state path.\n- The native-validated source direction starts at the right lower side; the same open path can be displayed from the left lower side by reversal.\n\n"
    md += "## Evidence\n\n"
    md += f"- Existing strict MoveIt2 acceptance: `{acceptance.get('overall_status')}` (`{STRICT_ACCEPTANCE.relative_to(ROOT)}`).\n- Collision method: `adaptive_discrete_interpolation`; CCD and clearance are `not_available`.\n- Native FK reproduced the 181-point TCP path with maximum error `{float(np.max(fk_to_authoritative_m)):.6g} m`.\n- The visual wall-distance proxy is not a clearance claim; native clearance remains unavailable.\n- No controller, hardware, physical goal, or robot motion was used.\n\n"
    md += "## Files\n\n"
    md += f"- [corrected_planar_horseshoe_robot.gif]({gif_path.resolve()})\n- [corrected_planar_horseshoe_overview.png]({overview_path.resolve()})\n- [corrected_planar_horseshoe_review.json]({report_path.resolve()})\n\n"
    md += "The previously generated H10-V family replay remains preserved as audit history, but it is not accepted as the visual representation of this planar tunnel spraying task.\n"
    (args.out_dir / "FINAL_REPORT.md").write_text(md, encoding="utf-8")

    if args.desktop_copy:
        desktop = Path.home() / "Desktop" / f"Stage3_Corrected_Planar_Horseshoe_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
        desktop.mkdir(parents=True, exist_ok=True)
        copied = []
        for source in (gif_path, overview_path, report_path, args.out_dir / "FINAL_REPORT.md"):
            destination = desktop / source.name
            destination.write_bytes(source.read_bytes())
            copied.append({"file": destination.name, "sha256": sha256(destination), "verified": sha256(destination) == sha256(source)})
        (desktop / "COPY_MANIFEST.json").write_text(json.dumps({"source": str(args.out_dir.resolve()), "files": copied}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        report["desktop_copy"] = str(desktop.resolve())
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"Desktop review copy: {desktop.resolve()}")

    print(f"Corrected planar review: {args.out_dir.resolve()}")
    print(json.dumps({"status": report["status"], "gif_frames": gif_frames, "max_fk_error_m": report["native_moveit2_evidence"]["native_fk_max_tcp_reproduction_error_m"], "direction": args.start}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
