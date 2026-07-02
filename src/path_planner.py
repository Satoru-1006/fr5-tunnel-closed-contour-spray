from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .robot_model import make_tool_frame
from .smoothing import periodic_bspline_with_normals
from .tool_models import ContactTool, SprayTool, ToolModel
from .tunnel_geometry import HorseshoeTunnel, TunnelConfig


@dataclass
class CoveragePath:
    surface_points: np.ndarray
    tcp_points: np.ndarray
    normals: np.ndarray
    poses: list[np.ndarray]
    pass_spacing: float


def generate_serpentine_path(
    tunnel: HorseshoeTunnel,
    tool: ToolModel,
    pass_overlap: float = 0.25,
    longitudinal_step: float = 0.045,
) -> CoveragePath:
    width = tool.footprint_width
    pass_spacing = max(width * (1.0 - pass_overlap), width * 0.35)
    section, normals_2d, s = tunnel.section(pass_spacing)
    target_s = np.arange(0.0, s[-1], pass_spacing)
    idx = np.unique(np.clip(np.searchsorted(s, target_s), 0, len(section) - 1))
    section = section[idx]
    normals_2d = normals_2d[idx]

    stand_off = tool.spray_distance if isinstance(tool, SprayTool) else tool.stand_off
    ys_forward = np.arange(0.0, tunnel.length + longitudinal_step * 0.5, longitudinal_step)
    ys_backward = ys_forward[::-1]

    surf, tcp, normals = [], [], []
    for i, (xz, n2) in enumerate(zip(section, normals_2d)):
        ys = ys_forward if i % 2 == 0 else ys_backward
        for y in ys:
            n3 = np.array([n2[0], 0.0, n2[1]])
            p = np.array([xz[0], y, xz[1]])
            surf.append(p)
            normals.append(n3)
            tcp.append(p + stand_off * n3)
    surf_arr = np.asarray(surf)
    tcp_arr = np.asarray(tcp)
    normals_arr = np.asarray(normals)
    poses = [make_tool_frame(p, n) for p, n in zip(tcp_arr, normals_arr)]
    return CoveragePath(surf_arr, tcp_arr, normals_arr, poses, pass_spacing)


def _closed_section_samples(
    tunnel: HorseshoeTunnel,
    spacing: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    section, normals_2d, s_open = tunnel.section(spacing)
    closing_len = float(np.linalg.norm(section[0] - section[-1]))
    s_closed = np.r_[s_open, s_open[-1] + closing_len]
    section_closed = np.vstack([section, section[0]])
    normals_closed = np.vstack([normals_2d, normals_2d[0]])
    return section_closed, normals_closed, s_closed, float(s_closed[-1])


def generate_helical_path(
    tunnel: HorseshoeTunnel,
    tool: ToolModel,
    pass_overlap: float = 0.25,
    spatial_step: float = 0.025,
    section_spacing: float = 0.018,
) -> CoveragePath:
    """Generate a closed horseshoe helical path advancing along tunnel length.

    The TCP completes repeated closed horseshoe loops around the lining section.
    After each loop it advances by ``pitch`` in tunnel Y, creating the spiral
    forward spray pattern shown in the reference animation.
    """

    width = tool.footprint_width
    pitch = max(width * (1.0 - pass_overlap), width * 0.35)
    section, normals_2d, s_closed, perimeter = _closed_section_samples(tunnel, section_spacing)
    total_section_s = perimeter * (tunnel.length / pitch + 1.0)
    section_s = np.arange(0.0, total_section_s + spatial_step * 0.5, spatial_step)
    y = section_s / perimeter * pitch
    keep = y <= tunnel.length
    section_s = section_s[keep]
    y = y[keep]
    s_mod = np.mod(section_s, perimeter)

    x = np.interp(s_mod, s_closed, section[:, 0])
    z = np.interp(s_mod, s_closed, section[:, 1])
    nx = np.interp(s_mod, s_closed, normals_2d[:, 0])
    nz = np.interp(s_mod, s_closed, normals_2d[:, 1])
    normal_len = np.maximum(np.hypot(nx, nz), 1e-12)
    nx, nz = nx / normal_len, nz / normal_len

    stand_off = tool.spray_distance if isinstance(tool, SprayTool) else tool.stand_off
    surf_arr = np.column_stack([x, y, z])
    normals_arr = np.column_stack([nx, np.zeros_like(nx), nz])
    tcp_arr = surf_arr + stand_off * normals_arr
    poses = [make_tool_frame(p, n) for p, n in zip(tcp_arr, normals_arr)]
    return CoveragePath(surf_arr, tcp_arr, normals_arr, poses, pitch)


def generate_closed_horseshoe_path(
    tunnel: HorseshoeTunnel,
    tool: ToolModel,
    station_y: float | None = None,
    section_spacing: float = 0.010,
    loops: int = 1,
) -> CoveragePath:
    """Generate a closed horseshoe-shaped coating path at one tunnel station.

    This matches the reference animation style: the end effector traces the
    horseshoe lining section itself, not a longitudinal helix.
    """

    station = tunnel.length * 0.5 if station_y is None else float(station_y)
    section, normals_2d, _, _ = _closed_section_samples(tunnel, section_spacing)
    section = section[:-1]
    normals_2d = normals_2d[:-1]
    if loops > 1:
        section = np.vstack([section for _ in range(loops)])
        normals_2d = np.vstack([normals_2d for _ in range(loops)])

    stand_off = tool.spray_distance if isinstance(tool, SprayTool) else tool.stand_off
    surf_arr = np.column_stack([section[:, 0], np.full(len(section), station), section[:, 1]])
    normals_arr = np.column_stack([normals_2d[:, 0], np.zeros(len(normals_2d)), normals_2d[:, 1]])
    tcp_arr = surf_arr + stand_off * normals_arr
    poses = [make_tool_frame(p, n) for p, n in zip(tcp_arr, normals_arr)]
    return CoveragePath(surf_arr, tcp_arr, normals_arr, poses, tool.footprint_width)


def generate_closed_horseshoe_contour_path(
    R: float,
    H: float,
    y0: float,
    stand_off: float,
    n_loops: int,
    samples_per_loop: int,
    fillet_radius: float = 0.03,
    c2_smoothness: float = 1e-8,
) -> CoveragePath:
    """Closed Contour Tracking on a fixed standard horseshoe cross-section.

    Geometry:
        left wall: x = -R, z: 0 -> H
        top arch: center (0, H), radius R, theta pi -> 0
        right wall: x = R, z: H -> 0
        bottom edge: z = 0, x: R -> -R

    The path repeats the same closed contour for every loop. Y is always y0.
    """

    if R <= 0.0 or H <= 0.0:
        raise ValueError("R and H must be positive for a standard horseshoe contour.")
    if fillet_radius <= 0.0:
        raise ValueError("fillet_radius must be > 0 for smooth constant-speed closed contour tracking.")
    if stand_off <= 0.0:
        raise ValueError("stand_off must be positive.")
    if stand_off >= R:
        raise ValueError(f"stand_off={stand_off:.3f} must be smaller than arch radius R={R:.3f}.")
    if stand_off >= fillet_radius:
        raise ValueError(
            f"stand_off={stand_off:.3f} must be smaller than fillet_radius={fillet_radius:.3f}; "
            "otherwise the bottom corner offset degenerates or self-intersects."
        )

    samples = max(samples_per_loop, 32)

    def arc_points(center: tuple[float, float], radius: float, a0: float, a1: float, n: int) -> np.ndarray:
        a = np.linspace(a0, a1, max(2, n), endpoint=False)
        return np.column_stack([center[0] + radius * np.cos(a), center[1] + radius * np.sin(a)])

    def line_points(p0: tuple[float, float], p1: tuple[float, float], n: int) -> np.ndarray:
        return np.linspace(np.asarray(p0, dtype=float), np.asarray(p1, dtype=float), max(2, n), endpoint=False)

    r = fillet_radius
    lengths = np.array(
        [
            max(H - r, 0.0),
            np.pi * R,
            max(H - r, 0.0),
            np.pi * r / 2.0,
            max(2.0 * (R - r), 0.0),
            np.pi * r / 2.0,
        ]
    )
    raw_counts = samples * lengths / max(np.sum(lengths), 1e-12)
    counts = np.maximum(4, np.floor(raw_counts).astype(int))
    diff = int(samples - np.sum(counts))
    order = np.argsort(raw_counts - np.floor(raw_counts))[::-1]
    for i in range(abs(diff)):
        idx = order[i % len(order)]
        if diff > 0:
            counts[idx] += 1
        elif counts[idx] > 4:
            counts[idx] -= 1

    wall_segments = [
        line_points((-R, r), (-R, H), counts[0]),
        arc_points((0.0, H), R, np.pi, 0.0, counts[1]),
        line_points((R, H), (R, r), counts[2]),
        arc_points((R - r, r), r, 0.0, -np.pi / 2.0, counts[3]),
        line_points((R - r, 0.0), (-R + r, 0.0), counts[4]),
        arc_points((-R + r, r), r, -np.pi / 2.0, -np.pi, counts[5]),
    ]
    raw_wall_xz = np.vstack(wall_segments)
    wall_xz, normals_2d, _ = periodic_bspline_with_normals(
        raw_wall_xz,
        samples=samples,
        smoothness=c2_smoothness,
    )
    tcp_xz = wall_xz + stand_off * normals_2d
    one_loop_surface = np.column_stack([wall_xz[:, 0], np.full(len(wall_xz), y0), wall_xz[:, 1]])
    one_loop_tcp = np.column_stack([tcp_xz[:, 0], np.full(len(tcp_xz), y0), tcp_xz[:, 1]])
    one_loop_normals = np.column_stack([normals_2d[:, 0], np.zeros(len(normals_2d)), normals_2d[:, 1]])
    start_idx = int(np.argmin(np.abs(one_loop_surface[:, 0]) + 10.0 * np.abs(one_loop_surface[:, 2])))
    one_loop_surface = np.roll(one_loop_surface, -start_idx, axis=0)
    one_loop_tcp = np.roll(one_loop_tcp, -start_idx, axis=0)
    one_loop_normals = np.roll(one_loop_normals, -start_idx, axis=0)
    loops = max(1, int(n_loops))
    surf_arr = np.vstack([one_loop_surface for _ in range(loops)])
    normals_arr = np.vstack([one_loop_normals for _ in range(loops)])
    tcp_arr = np.vstack([one_loop_tcp for _ in range(loops)])
    poses = [make_tool_frame(p, n) for p, n in zip(tcp_arr, normals_arr)]
    return CoveragePath(surf_arr, tcp_arr, normals_arr, poses, 2.0 * R)
