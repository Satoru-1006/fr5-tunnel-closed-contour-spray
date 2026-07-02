from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import splprep, splev


@dataclass(frozen=True)
class TunnelConfig:
    width: float = 1.20
    height: float = 1.10
    length: float = 1.60
    invert_depth: float = 0.10
    fillet_radius: float = 0.03
    scale: float = 1.0


class HorseshoeTunnel:
    """Parameterized horseshoe tunnel lining surface."""

    def __init__(self, config: TunnelConfig) -> None:
        self.config = config
        self.width = config.width * config.scale
        self.height = config.height * config.scale
        self.length = config.length * config.scale
        self.invert_depth = config.invert_depth * config.scale
        self.fillet_radius = max(0.0, config.fillet_radius * config.scale)
        if self.height <= self.width / 2:
            raise ValueError("height must be larger than width / 2 for a horseshoe arch.")

    def section(self, spacing: float = 0.05) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.standard_section(spacing)

    def _horseshoe_control_polygon(self) -> np.ndarray:
        r = self.width / 2
        spring_z = self.height - r
        theta = np.linspace(np.pi, 0.0, 32, endpoint=True)
        arch = np.column_stack([r * np.cos(theta), spring_z + r * np.sin(theta)])
        n_side = max(8, int(spring_z / max(self.width * 0.035, 1e-3)))
        right = np.column_stack([np.full(n_side, r), np.linspace(spring_z, 0.0, n_side)])
        bottom = np.column_stack([np.linspace(r, -r, max(18, int(self.width / max(self.width * 0.035, 1e-3)))), np.zeros(max(18, int(self.width / max(self.width * 0.035, 1e-3))))])
        left = np.column_stack([np.full(n_side, -r), np.linspace(0.0, spring_z, n_side)])
        return np.vstack([arch, right[1:], bottom[1:], left[1:-1]])

    def smooth_section(self, spacing: float = 0.05) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return a C2 closed horseshoe section sampled uniformly by arc length."""

        controls = self._horseshoe_control_polygon()
        chord = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(np.vstack([controls, controls[0]]), axis=0), axis=1))]
        perimeter_guess = float(chord[-1])
        # Periodic cubic B-spline gives C2 continuity around the whole lining
        # while following the standard horseshoe target: vertical walls,
        # semicircular arch, and closed bottom edge.
        smoothing = (0.0012 * self.width) ** 2 * len(controls)
        tck, _ = splprep(controls.T, s=smoothing, k=3, per=True)
        dense_n = max(600, int(perimeter_guess / max(spacing, 1e-4)) * 12)
        u_dense = np.linspace(0.0, 1.0, dense_n, endpoint=False)
        dense = np.column_stack(splev(u_dense, tck))
        dense_closed = np.vstack([dense, dense[0]])
        s_dense = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(dense_closed, axis=0), axis=1))]
        target_s = np.arange(0.0, s_dense[-1], spacing)
        u_by_s = np.interp(target_s, s_dense, np.r_[u_dense, 1.0])
        points = np.column_stack(splev(u_by_s, tck))
        deriv = np.column_stack(splev(u_by_s, tck, der=1))
        tangents = deriv / np.maximum(np.linalg.norm(deriv, axis=1, keepdims=True), 1e-12)

        normals_a = np.column_stack([-tangents[:, 1], tangents[:, 0]])
        center = np.array([0.0, self.height * 0.45])
        to_center = center - points
        signs = np.sign(np.sum(normals_a * to_center, axis=1, keepdims=True))
        normals = normals_a * np.where(signs == 0.0, 1.0, signs)
        normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)

        s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(np.vstack([points, points[0]]), axis=0), axis=1))][:-1]
        return points, normals, s

    def standard_section(self, spacing: float = 0.05) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Standard closed horseshoe: left wall, top semicircle, right wall, bottom edge.

        Bottom corners use a small process fillet when ``fillet_radius > 0`` so
        the tracking path can be run at near-constant speed without hard 90 deg
        curvature impulses.
        """

        r = self.width / 2.0
        h = self.height - r
        f = min(self.fillet_radius, r * 0.35, h * 0.45 if h > 0 else 0.0)
        pts: list[np.ndarray] = []

        def append_line(p0: tuple[float, float], p1: tuple[float, float]) -> None:
            p0a = np.asarray(p0, dtype=float)
            p1a = np.asarray(p1, dtype=float)
            length = float(np.linalg.norm(p1a - p0a))
            n = max(2, int(length / max(spacing, 1e-4)))
            seg = np.linspace(p0a, p1a, n, endpoint=False)
            pts.extend(seg)

        def append_arc(center: tuple[float, float], radius: float, a0: float, a1: float) -> None:
            length = abs(a1 - a0) * radius
            n = max(6, int(length / max(spacing, 1e-4)))
            a = np.linspace(a0, a1, n, endpoint=False)
            pts.extend(np.column_stack([center[0] + radius * np.cos(a), center[1] + radius * np.sin(a)]))

        # Counterclockwise contour: bottom-left -> left wall -> arch -> right wall -> bottom edge.
        append_line((-r, f), (-r, h))
        append_arc((0.0, h), r, np.pi, 0.0)
        append_line((r, h), (r, f))
        if f > 0.0:
            append_arc((r - f, f), f, 0.0, -np.pi / 2.0)
            append_line((r - f, 0.0), (-r + f, 0.0))
            append_arc((-r + f, f), f, -np.pi / 2.0, -np.pi)
        else:
            append_line((r, 0.0), (-r, 0.0))

        points = np.asarray(pts)
        points_closed = np.vstack([points, points[0]])
        tangents = np.gradient(points_closed, axis=0)[:-1]
        tangents /= np.maximum(np.linalg.norm(tangents, axis=1, keepdims=True), 1e-12)
        normals_a = np.column_stack([-tangents[:, 1], tangents[:, 0]])
        center = np.array([0.0, h * 0.45])
        to_center = center - points
        signs = np.sign(np.sum(normals_a * to_center, axis=1, keepdims=True))
        normals = normals_a * np.where(signs == 0.0, 1.0, signs)
        normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
        s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(points_closed, axis=0), axis=1))][:-1]
        return points, normals, s

    def surface_grid(self, section_spacing: float, longitudinal_spacing: float) -> tuple[np.ndarray, np.ndarray]:
        section, normals_2d, _ = self.section(section_spacing)
        ys = np.arange(0.0, self.length + longitudinal_spacing * 0.5, longitudinal_spacing)
        pts, normals = [], []
        for y in ys:
            xyz = np.column_stack([section[:, 0], np.full(len(section), y), section[:, 1]])
            n3 = np.column_stack([normals_2d[:, 0], np.zeros(len(section)), normals_2d[:, 1]])
            pts.append(xyz)
            normals.append(n3)
        return np.vstack(pts), np.vstack(normals)

    def warn_if_real_tunnel_exceeds_fr5(self, reach_m: float = 0.92) -> str | None:
        if max(self.width, self.height, self.length) > reach_m * 2.0:
            return (
                "Tunnel dimensions exceed a fixed FR5 reach envelope. Use a rail, mobile base, "
                "lift platform, or station-by-station workcell strategy for real-size tunnels."
            )
        return None
