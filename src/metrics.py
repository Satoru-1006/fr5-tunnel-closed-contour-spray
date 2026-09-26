from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import find_peaks

from .tunnel_geometry import HorseshoeTunnel


@dataclass(frozen=True)
class QualityReport:
    coverage_rate: float
    repeat_rate: float
    tcp_speed_fluctuation: float
    ik_success_rate: float
    max_ik_error: float
    speed_mean: float
    speed_min: float
    speed_max: float
    jerk_abs_max: np.ndarray
    jerk_p95: np.ndarray
    jerk_p99: np.ndarray
    jump_indices: np.ndarray
    spike_indices: np.ndarray
    spike_details: list[dict[str, float | int | str]]
    notes: list[str]


def detect_spikes(signal: np.ndarray, z_threshold: float = 20.0) -> np.ndarray:
    arr = np.asarray(signal)
    mag = np.linalg.norm(arr, axis=1) if arr.ndim == 2 else np.abs(arr)
    trim = max(3, int(0.03 * len(mag)))
    core = mag[trim:-trim] if len(mag) > 2 * trim + 5 else mag
    med = np.median(core)
    mad = np.median(np.abs(core - med)) + 1e-12
    z = 0.6745 * (mag - med) / mad
    idx = np.flatnonzero(z > z_threshold)
    return idx[(idx >= trim) & (idx < len(mag) - trim)]


def detect_joint_jumps(q: np.ndarray, threshold_deg: float = 12.0) -> np.ndarray:
    dq_step = np.abs(np.diff(q, axis=0))
    return np.flatnonzero(np.any(dq_step > np.deg2rad(threshold_deg), axis=1)) + 1


def jerk_statistics(jerk: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mag = np.abs(jerk)
    return np.max(mag, axis=0), np.percentile(mag, 95, axis=0), np.percentile(mag, 99, axis=0)


def detect_jerk_spikes(
    jerk: np.ndarray,
    t: np.ndarray | None = None,
    ratio_threshold: float = 6.0,
    min_abs_jerk: float = 0.25,
    jerk_limits: np.ndarray | None = None,
    limit_warning_fraction: float = 0.30,
    local_ratio_threshold: float = 3.0,
    local_baseline_radius: int = 5,
) -> tuple[np.ndarray, list[dict[str, float | int | str]]]:
    """Find candidate jerk peaks for diagnostics.

    This is a diagnostic heuristic, not a continuous-jerk certificate or a
    substitute for the configured limit audit.  The global p95 outlier rule
    is intentionally supplemented by a local-baseline rule: a short,
    localized excursion can be well below ``limit_warning_fraction`` of a
    generous configured limit while still being a meaningful discontinuity
    relative to the surrounding trajectory.
    """

    jerk_values = np.asarray(jerk, dtype=float)
    if jerk_values.ndim == 1:
        jerk_values = jerk_values[:, None]
    elif jerk_values.ndim != 2:
        raise ValueError("jerk must be a one- or two-dimensional finite array")
    if jerk_values.shape[1] == 0:
        raise ValueError("jerk must contain at least one joint")
    if not np.all(np.isfinite(jerk_values)):
        raise ValueError("jerk must contain only finite values")

    threshold_values = {
        "ratio_threshold": ratio_threshold,
        "min_abs_jerk": min_abs_jerk,
        "limit_warning_fraction": limit_warning_fraction,
        "local_ratio_threshold": local_ratio_threshold,
    }
    for name, value in threshold_values.items():
        numeric_value = float(value)
        if not np.isfinite(numeric_value) or numeric_value < 0.0:
            raise ValueError(f"{name} must be a finite non-negative number")
        threshold_values[name] = numeric_value
    ratio_threshold = threshold_values["ratio_threshold"]
    min_abs_jerk = threshold_values["min_abs_jerk"]
    limit_warning_fraction = threshold_values["limit_warning_fraction"]
    local_ratio_threshold = threshold_values["local_ratio_threshold"]

    sample_count, joint_count = jerk_values.shape
    if t is not None:
        time_values = np.asarray(t, dtype=float)
        if time_values.ndim != 1 or len(time_values) != sample_count:
            raise ValueError("t must be a one-dimensional array matching jerk samples")
        if not np.all(np.isfinite(time_values)):
            raise ValueError("t must contain only finite values")
    else:
        time_values = None

    if sample_count == 0:
        return np.empty(0, dtype=int), []

    mag = np.abs(jerk_values)
    p95 = np.percentile(mag, 95, axis=0) + 1e-12
    global_ratio_mask = (mag > ratio_threshold * p95) & (mag > min_abs_jerk)
    if jerk_limits is not None:
        limits = np.asarray(jerk_limits, dtype=float)
        if limits.ndim == 0:
            limits = np.repeat(limits, joint_count)
        if limits.ndim != 1 or len(limits) != joint_count:
            raise ValueError("jerk_limits must be a one-dimensional array matching jerk joints")
        if not np.all(np.isfinite(limits)) or np.any(limits < 0.0):
            raise ValueError("jerk_limits must contain finite non-negative values")
        warning_band = np.maximum(limits[None, :] * limit_warning_fraction, min_abs_jerk)
        global_ratio_mask = global_ratio_mask & (mag > warning_band)
        limit_mask = mag > limits[None, :] * 1.02
    else:
        limit_mask = np.zeros_like(mag, dtype=bool)

    # Estimate the local baseline from neighboring samples while excluding
    # the candidate itself.  Keeping this independent of the global warning
    # band is what preserves localized low-amplitude diagnostics.
    radius = int(local_baseline_radius)
    if radius < 1:
        raise ValueError("local_baseline_radius must be at least one sample")
    local_baseline = np.empty_like(mag)
    for row in range(mag.shape[0]):
        start = max(0, row - radius)
        stop = min(mag.shape[0], row + radius + 1)
        neighborhood = mag[start:stop]
        if neighborhood.shape[0] > 1:
            without_candidate = np.delete(neighborhood, row - start, axis=0)
            local_baseline[row] = np.median(without_candidate, axis=0)
        else:  # pragma: no cover - guarded for completeness on one-sample input
            local_baseline[row] = neighborhood[0]
    local_ratio = mag / np.maximum(local_baseline, 1e-12)
    local_ratio_mask = (local_ratio > local_ratio_threshold) & (mag > min_abs_jerk)
    mask = global_ratio_mask | local_ratio_mask | limit_mask
    peak_pairs: list[tuple[int, int]] = []
    for joint in range(mag.shape[1]):
        peaks, _ = find_peaks(mag[:, joint], height=0.0, distance=5)
        # scipy's interior-only peak finder omits an excursion at either
        # boundary.  Include a boundary only when it is strictly larger than
        # its sole neighbor, keeping flat signals out of the diagnostic list.
        boundary_peaks: list[int] = []
        if sample_count > 1:
            if mag[0, joint] > 0.0 and mag[0, joint] > mag[1, joint]:
                boundary_peaks.append(0)
            if mag[-1, joint] > 0.0 and mag[-1, joint] > mag[-2, joint]:
                boundary_peaks.append(sample_count - 1)
        for row in np.unique(np.concatenate((peaks, np.asarray(boundary_peaks, dtype=int)))):
            if mask[row, joint]:
                peak_pairs.append((int(row), joint))
    peak_pairs.sort()
    details: list[dict[str, float | int | str]] = []
    for row, joint in peak_pairs[:50]:
        details.append(
            {
                "time": float(time_values[row]) if time_values is not None else float(row),
                "path_index": int(row),
                "joint": int(joint + 1),
                "jerk": float(jerk_values[row, joint]),
                "p95_joint_jerk": float(p95[joint]),
                "local_baseline_jerk": float(local_baseline[row, joint]),
                "local_baseline_ratio": float(local_ratio[row, joint]),
                "measurement_scope": "diagnostic_heuristic_only",
                "reason": (
                    "joint-space jerk exceeds configured limit"
                    if limit_mask[row, joint]
                    else (
                        "joint-space jerk is a max/p95 outlier; inspect local path curvature and IK continuity"
                        if global_ratio_mask[row, joint]
                        else "joint-space jerk is a local baseline diagnostic outlier; inspect local path curvature and IK continuity"
                    )
                ),
            }
        )
    rows = np.asarray([row for row, _ in peak_pairs], dtype=int)
    return np.unique(rows), details


def coverage_metrics(tunnel: HorseshoeTunnel, surface_path: np.ndarray, footprint_width: float) -> tuple[float, float, np.ndarray]:
    s_path = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(surface_path, axis=0), axis=1))]
    keep_s = np.arange(0.0, s_path[-1] + footprint_width * 0.5, footprint_width * 0.65)
    idx = np.unique(np.clip(np.searchsorted(s_path, keep_s), 0, len(surface_path) - 1))
    effective_path = surface_path[idx]
    samples, _ = tunnel.surface_grid(section_spacing=footprint_width * 0.45, longitudinal_spacing=footprint_width * 0.45)
    d2 = np.sum((samples[:, None, :] - effective_path[None, :, :]) ** 2, axis=2)
    min_d = np.sqrt(np.min(d2, axis=1))
    hits = np.sum(np.sqrt(d2) <= footprint_width / 2.0, axis=1)
    coverage = float(np.mean(min_d <= footprint_width / 2.0))
    repeat = float(np.mean(hits >= 2))
    return coverage, repeat, hits


def polyline_distance(points: np.ndarray, polyline: np.ndarray, closed: bool = True) -> np.ndarray:
    """Distance from points to a 3D polyline, using segment projection."""

    pts = np.asarray(points, dtype=float)
    line = np.asarray(polyline, dtype=float)
    if len(line) < 2:
        raise ValueError("polyline must contain at least two points")
    if closed:
        seg_start = line
        seg_end = np.roll(line, -1, axis=0)
    else:
        seg_start = line[:-1]
        seg_end = line[1:]
    best = np.full(len(pts), np.inf)
    for a, b in zip(seg_start, seg_end):
        edge = b - a
        denom = max(float(np.dot(edge, edge)), 1e-15)
        alpha = np.clip(np.sum((pts - a) * edge, axis=1) / denom, 0.0, 1.0)
        projected = a + alpha[:, None] * edge
        best = np.minimum(best, np.linalg.norm(pts - projected, axis=1))
    return best


def project_points_to_polyline(
    points: np.ndarray,
    polyline: np.ndarray,
    normals: np.ndarray | None = None,
    closed: bool = True,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    """Project points to nearest polyline segments and optionally interpolate normals."""

    pts = np.asarray(points, dtype=float)
    line = np.asarray(polyline, dtype=float)
    if len(line) < 2:
        raise ValueError("polyline must contain at least two points")
    if normals is not None:
        normal_arr = np.asarray(normals, dtype=float)
        if len(normal_arr) != len(line):
            raise ValueError("normals and polyline must have the same length")
    else:
        normal_arr = None
    if closed:
        seg_start = line
        seg_end = np.roll(line, -1, axis=0)
        normal_start = normal_arr
        normal_end = np.roll(normal_arr, -1, axis=0) if normal_arr is not None else None
    else:
        seg_start = line[:-1]
        seg_end = line[1:]
        normal_start = normal_arr[:-1] if normal_arr is not None else None
        normal_end = normal_arr[1:] if normal_arr is not None else None
    best_d2 = np.full(len(pts), np.inf)
    best_projection = np.zeros_like(pts)
    best_normal = np.zeros_like(pts) if normal_arr is not None else None
    for i, (a, b) in enumerate(zip(seg_start, seg_end)):
        edge = b - a
        denom = max(float(np.dot(edge, edge)), 1e-15)
        alpha = np.clip(np.sum((pts - a) * edge, axis=1) / denom, 0.0, 1.0)
        projected = a + alpha[:, None] * edge
        d2 = np.sum((pts - projected) ** 2, axis=1)
        choose = d2 < best_d2
        best_d2[choose] = d2[choose]
        best_projection[choose] = projected[choose]
        if best_normal is not None and normal_start is not None and normal_end is not None:
            interpolated = (1.0 - alpha[:, None]) * normal_start[i] + alpha[:, None] * normal_end[i]
            interpolated /= np.maximum(np.linalg.norm(interpolated, axis=1, keepdims=True), 1e-12)
            best_normal[choose] = interpolated[choose]
    return best_projection, best_normal, np.sqrt(best_d2)


def tcp_speed_from_positions(points: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Estimate TCP speed from FK positions without endpoint gradient artifacts."""

    pts = np.asarray(points, dtype=float)
    time = np.asarray(t, dtype=float)
    if len(pts) != len(time):
        raise ValueError("points and t must have the same length")
    if len(pts) < 2:
        return np.zeros(len(pts), dtype=float)
    dt = np.diff(time)
    if np.any(dt <= 0.0):
        raise ValueError("t must be strictly increasing")
    segment_speed = np.linalg.norm(np.diff(pts, axis=0), axis=1) / dt
    speed = np.empty(len(pts), dtype=float)
    speed[0] = segment_speed[0]
    speed[-1] = segment_speed[-1]
    if len(pts) > 2:
        speed[1:-1] = 0.5 * (segment_speed[:-1] + segment_speed[1:])
    return speed


def section_coverage_metrics(tunnel: HorseshoeTunnel, surface_path: np.ndarray, footprint_width: float) -> tuple[float, float, np.ndarray]:
    section, _, _ = tunnel.section(footprint_width * 0.20)
    station = float(np.median(surface_path[:, 1]))
    samples = np.column_stack([section[:, 0], np.full(len(section), station), section[:, 1]])
    s_path = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(surface_path, axis=0), axis=1))]
    keep_s = np.arange(0.0, s_path[-1] + footprint_width * 0.5, footprint_width * 0.75)
    idx = np.unique(np.clip(np.searchsorted(s_path, keep_s), 0, len(surface_path) - 1))
    effective_path = surface_path[idx]
    d2 = np.sum((samples[:, None, :] - effective_path[None, :, :]) ** 2, axis=2)
    min_d = np.sqrt(np.min(d2, axis=1))
    hits = np.sum(np.sqrt(d2) <= footprint_width / 2.0, axis=1)
    coverage = float(np.mean(min_d <= footprint_width / 2.0))
    repeat = float(np.mean(hits >= 2))
    return coverage, repeat, hits


def closure_errors(points: np.ndarray, samples_per_loop: int) -> dict[str, float]:
    if samples_per_loop <= 3 or len(points) < samples_per_loop:
        return {
            "close_position_error": float("nan"),
            "close_tangent_error": float("nan"),
            "close_curvature_error": float("nan"),
        }
    loop = points[:samples_per_loop]
    next_start = points[samples_per_loop] if len(points) > samples_per_loop else loop[0]
    pos_err = float(np.linalg.norm(next_start - loop[0]))

    prev_p = loop[-1]
    p0 = loop[0]
    p1 = loop[1]
    pm2 = loop[-2]
    tangent_start = p1 - prev_p
    tangent_end = next_start - prev_p
    tangent_start /= max(np.linalg.norm(tangent_start), 1e-12)
    tangent_end /= max(np.linalg.norm(tangent_end), 1e-12)
    tangent_err = float(np.linalg.norm(tangent_start - tangent_end))

    def curvature(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
        ab = b - a
        bc = c - b
        ac = c - a
        cross = np.linalg.norm(np.cross(ab, bc))
        denom = max(np.linalg.norm(ab) * np.linalg.norm(bc) * np.linalg.norm(ac), 1e-12)
        return float(2.0 * cross / denom)

    k_start = curvature(prev_p, p0, p1)
    k_end = curvature(pm2, prev_p, next_start)
    curv_err = abs(k_start - k_end)
    return {
        "close_position_error": pos_err,
        "close_tangent_error": tangent_err,
        "close_curvature_error": curv_err,
    }


def build_quality_report(
    tunnel: HorseshoeTunnel,
    surface_path: np.ndarray,
    footprint_width: float,
    tcp_speed: np.ndarray,
    q: np.ndarray,
    ddq: np.ndarray,
    jerk: np.ndarray,
    ik_success: np.ndarray,
    ik_errors: np.ndarray,
    t: np.ndarray | None = None,
    section_only: bool = False,
    jerk_limits: np.ndarray | None = None,
) -> QualityReport:
    metric_fn = section_coverage_metrics if section_only else coverage_metrics
    coverage, repeat, _ = metric_fn(tunnel, surface_path, footprint_width)
    speed_mean = max(float(np.mean(tcp_speed)), 1e-12)
    speed_min = float(np.min(tcp_speed))
    speed_max = float(np.max(tcp_speed))
    speed_p05 = float(np.percentile(tcp_speed, 5))
    speed_p95 = float(np.percentile(tcp_speed, 95))
    fluct = float((speed_p95 - speed_p05) / speed_mean)
    jumps = detect_joint_jumps(q)
    jerk_max, jerk_p95, jerk_p99 = jerk_statistics(jerk)
    jerk_spikes, spike_details = detect_jerk_spikes(jerk, t=t, jerk_limits=jerk_limits)
    spikes = jerk_spikes
    notes: list[str] = []
    if len(jumps):
        notes.append("Detected joint jumps: use previous IK seed, denser sampling, or avoid singular/limit regions.")
    if len(spikes):
        notes.append("Detected dynamic spikes: increase C2 path smoothing, reduce target speed, retime with tighter jerk limits, or inspect local IK continuity.")
    if coverage < 0.98:
        notes.append("Coverage below 98%: reduce pitch or increase spray/contact footprint.")
    return QualityReport(
        coverage,
        repeat,
        fluct,
        float(np.mean(ik_success)),
        float(np.max(ik_errors)),
        speed_mean,
        speed_min,
        speed_max,
        jerk_max,
        jerk_p95,
        jerk_p99,
        jumps,
        spikes,
        spike_details,
        notes,
    )
