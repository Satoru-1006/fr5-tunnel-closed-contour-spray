from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline, splprep, splev
from scipy.signal import savgol_filter


def cumulative_arclength(points: np.ndarray) -> np.ndarray:
    return np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]


def smooth_and_resample(points: np.ndarray, spacing: float = 0.025, smoothness: float = 0.003) -> tuple[np.ndarray, np.ndarray]:
    """B-spline smooth a path and resample it by arc length."""

    if len(points) < 5:
        return points.copy(), cumulative_arclength(points)
    u = cumulative_arclength(points)
    keep = np.r_[True, np.diff(u) > 1e-8]
    pts = points[keep]
    u = cumulative_arclength(pts)
    u_norm = u / max(u[-1], 1e-12)
    tck, _ = splprep(pts.T, u=u_norm, s=smoothness * len(pts), k=min(3, len(pts) - 1))
    dense_n = max(len(pts) * 5, int(u[-1] / max(spacing, 1e-4)) * 3)
    ud = np.linspace(0.0, 1.0, dense_n)
    dense = np.column_stack(splev(ud, tck))
    sd = cumulative_arclength(dense)
    target_s = np.arange(0.0, sd[-1], spacing)
    resampled = np.column_stack([np.interp(target_s, sd, dense[:, i]) for i in range(3)])
    return resampled, cumulative_arclength(resampled)


def cubic_arclength_resample(points: np.ndarray, spacing: float = 0.02) -> tuple[np.ndarray, np.ndarray]:
    """Fit x(s), y(s), z(s) cubic splines and resample by constant 3D arc length."""

    s = cumulative_arclength(points)
    keep = np.r_[True, np.diff(s) > 1e-9]
    pts = points[keep]
    s = cumulative_arclength(pts)
    bc = "natural"
    splines = [CubicSpline(s, pts[:, axis], bc_type=bc) for axis in range(3)]
    dense_s = np.linspace(0.0, s[-1], max(len(pts) * 8, int(s[-1] / max(spacing, 1e-4)) * 4))
    dense = np.column_stack([cs(dense_s) for cs in splines])
    dense_arc = cumulative_arclength(dense)
    target = np.arange(0.0, dense_arc[-1], spacing)
    resampled = np.column_stack([np.interp(target, dense_arc, dense[:, axis]) for axis in range(3)])
    return resampled, cumulative_arclength(resampled)


def periodic_bspline_arclength_resample(
    points: np.ndarray,
    spacing: float = 0.01,
    smoothness: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Periodic B-spline fit for closed C2 paths, then constant arc-length sampling."""

    if len(points) < 6:
        return points.copy(), cumulative_arclength(points)
    pts = points.copy()
    if np.linalg.norm(pts[0] - pts[-1]) < 1e-8:
        pts = pts[:-1]
    tck, _ = splprep(pts.T, s=smoothness * len(pts), k=3, per=True)
    dense_n = max(len(pts) * 16, 600)
    u_dense = np.linspace(0.0, 1.0, dense_n, endpoint=False)
    dense = np.column_stack(splev(u_dense, tck))
    dense_closed = np.vstack([dense, dense[0]])
    dense_arc = cumulative_arclength(dense_closed)
    target = np.arange(0.0, dense_arc[-1], spacing)
    u_closed = np.r_[u_dense, 1.0]
    u_target = np.interp(target, dense_arc, u_closed)
    resampled = np.column_stack(splev(u_target, tck))
    return resampled, cumulative_arclength(np.vstack([resampled, resampled[0]]))[:-1]


def periodic_bspline_with_normals(
    points: np.ndarray,
    samples: int,
    smoothness: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return a C2 periodic curve, inward normals, and uniform arc length.

    ``points`` must describe one closed X-Z contour in clockwise order.  The
    spline derivative is used directly for the normal, avoiding finite-
    difference and nearest-point normal noise at segment boundaries.
    """

    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) < 8:
        raise ValueError("A periodic X-Z contour needs at least eight 2D points.")
    if np.linalg.norm(pts[0] - pts[-1]) < 1e-10:
        pts = pts[:-1]

    # Arc-length-like input parameters keep dense straight and curved source
    # segments from biasing the periodic fit.
    closed = np.vstack([pts, pts[0]])
    source_s = cumulative_arclength(closed)
    u = source_s / max(source_s[-1], 1e-12)
    # For ``per=True`` FITPACK expects the final sample to duplicate the first;
    # it deliberately discards that final observation while closing the fit.
    tck, _ = splprep(closed.T, u=u, s=max(0.0, smoothness) * len(pts), k=3, per=True)

    dense_n = max(4096, int(samples) * 32)
    u_dense = np.linspace(0.0, 1.0, dense_n, endpoint=False)
    dense = np.column_stack(splev(u_dense, tck))
    dense_closed = np.vstack([dense, dense[0]])
    dense_s = cumulative_arclength(dense_closed)
    target_s = np.linspace(0.0, dense_s[-1], max(32, int(samples)), endpoint=False)
    u_closed = np.r_[u_dense, 1.0]
    u_target = np.interp(target_s, dense_s, u_closed)

    curve = np.column_stack(splev(u_target, tck, der=0))
    derivative = np.column_stack(splev(u_target, tck, der=1))
    tangent = derivative / np.maximum(np.linalg.norm(derivative, axis=1, keepdims=True), 1e-12)
    # The source contour is clockwise, so its interior is on the tangent's
    # right side: n = [t_z, -t_x].
    normals = np.column_stack([tangent[:, 1], -tangent[:, 0]])
    return curve, normals, target_s


def smooth_vectors(vectors: np.ndarray, window: int = 21) -> np.ndarray:
    n = len(vectors)
    if n < 7:
        out = vectors.copy()
    else:
        win = min(window, n - (1 - n % 2))
        win = max(5, win if win % 2 == 1 else win - 1)
        out = savgol_filter(vectors, window_length=win, polyorder=3, axis=0, mode="interp")
    out /= np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-12)
    return out
