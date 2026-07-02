from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import CubicSpline, make_interp_spline, splrep, splev
from scipy.signal import savgol_filter

from .smoothing import cumulative_arclength
from .robot_model import JointLimits


@dataclass(frozen=True)
class TimeProfile:
    t: np.ndarray
    path_points: np.ndarray
    tcp_speed: np.ndarray
    q: np.ndarray
    dq: np.ndarray
    ddq: np.ndarray
    jerk: np.ndarray
    method: str = "fallback"


def _gradient(x: np.ndarray, t: np.ndarray) -> np.ndarray:
    return np.gradient(x, t, axis=0, edge_order=2)


def make_uniform_time_profile(
    path_points: np.ndarray,
    q: np.ndarray,
    target_tcp_speed: float = 0.08,
    dt: float = 0.04,
    smooth_window: int = 51,
) -> TimeProfile:
    s = cumulative_arclength(path_points)
    total_time = max(s[-1] / max(target_tcp_speed, 1e-4), dt * (len(path_points) - 1))
    t = np.arange(0.0, total_time + dt * 0.5, dt)
    s_uniform = np.linspace(0.0, s[-1], len(t))
    pts = np.column_stack([np.interp(s_uniform, s, path_points[:, i]) for i in range(3)])
    q_interp = np.column_stack([np.interp(s_uniform, s, q[:, i]) for i in range(q.shape[1])])

    win = min(smooth_window, len(t) - (1 - len(t) % 2))
    if win >= 7:
        win = win if win % 2 == 1 else win - 1
        q_interp = savgol_filter(q_interp, window_length=win, polyorder=3, axis=0, mode="interp")

    tcp_speed = np.maximum(_gradient(s_uniform, t), 0.0)
    dq = _gradient(q_interp, t)
    ddq = _gradient(dq, t)
    jerk = _gradient(ddq, t)
    return TimeProfile(t, pts, tcp_speed, q_interp, dq, ddq, jerk)


def _sample_cubic_profile(path_points: np.ndarray, q: np.ndarray, total_time: float, dt: float, smooth_window: int) -> TimeProfile:
    s = cumulative_arclength(path_points)
    keep = np.r_[True, np.diff(s) > 1e-9]
    pts = path_points[keep]
    q_interp = q[keep]
    t = np.linspace(0.0, total_time, len(pts))
    tcp_speed = np.full(len(t), cumulative_arclength(pts)[-1] / max(t[-1] - t[0], 1e-12))
    dq = _gradient(q_interp, t)
    ddq = _gradient(dq, t)
    jerk = _gradient(ddq, t)
    return TimeProfile(t, pts, tcp_speed, q_interp, dq, ddq, jerk, method="cubic-retimed")


def make_jerk_limited_time_profile(
    path_points: np.ndarray,
    q: np.ndarray,
    limits: JointLimits,
    target_tcp_speed: float = 0.08,
    dt: float = 0.04,
    smooth_window: int = 101,
    jerk_safety: float = 0.08,
) -> TimeProfile:
    """Fallback joint-space retiming when Ruckig/toppra are not installed.

    It is not a replacement for Ruckig's online trajectory generation. It fits
    C2 cubic profiles and stretches time until velocity, acceleration, and jerk
    stay inside configured joint limits.
    """

    total_length = cumulative_arclength(path_points)[-1]
    total_time = max(total_length / max(target_tcp_speed, 1e-4), dt * (len(path_points) - 1))
    profile = _sample_cubic_profile(path_points, q, total_time, dt, smooth_window)
    for _ in range(8):
        vel_ratio = float(np.max(np.abs(profile.dq) / np.maximum(limits.dq_max, 1e-9)))
        acc_ratio = float(np.max(np.abs(profile.ddq) / np.maximum(limits.ddq_max, 1e-9)))
        jerk_ratio = float(np.max(np.abs(profile.jerk) / np.maximum(limits.jerk_max * jerk_safety, 1e-9)))
        scale = max(vel_ratio, np.sqrt(acc_ratio), np.cbrt(jerk_ratio), 1.0)
        if scale <= 1.01:
            break
        total_time *= scale * 1.05
        profile = _sample_cubic_profile(path_points, q, total_time, dt, smooth_window)
    return TimeProfile(profile.t, profile.path_points, profile.tcp_speed, profile.q, profile.dq, profile.ddq, profile.jerk, method="cubic-retimed")


def make_ruckig_time_profile(*args, **kwargs) -> TimeProfile:
    return make_segmented_ruckig_time_profile(*args, **kwargs)


def _closed_loop_size(path_points: np.ndarray) -> int:
    distances = np.linalg.norm(path_points[1:] - path_points[0], axis=1)
    candidates = np.flatnonzero(distances < 1e-9) + 1
    candidates = candidates[candidates >= 16]
    return int(candidates[0]) if len(candidates) else len(path_points)


def make_closed_loop_ruckig_time_profile(
    path_points: np.ndarray,
    q: np.ndarray,
    limits: JointLimits,
    target_tcp_speed: float = 0.08,
    dt: float = 0.01,
    fk_position=None,
    joint_smoothness: float = 1e-6,
) -> TimeProfile:
    """Time-parameterize one periodic joint path without segment overshoot.

    The last repeated IK loop is used because sequential IK has already settled
    onto a consistent branch there. A periodic quintic B-spline gives C4 joint
    motion; its derivatives map joint velocity, acceleration, and jerk limits
    to one scalar path-speed bound. Ruckig then generates the scalar path law
    for the complete repeated contour, after which q(s) is sampled analytically.
    """

    try:
        import ruckig
    except ImportError as exc:
        raise RuntimeError("Ruckig is not installed. Install `ruckig` to use this profile.") from exc

    loop_size = _closed_loop_size(path_points)
    loops = max(1, len(path_points) // loop_size)
    start = (loops - 1) * loop_size
    one_path = np.asarray(path_points[start : start + loop_size], dtype=float)
    one_q = np.asarray(q[start : start + loop_size], dtype=float)
    if len(one_path) != loop_size:
        raise ValueError("Incomplete final closed loop in path input.")

    path_closed = np.vstack([one_path, one_path[0]])
    q_closed = np.vstack([one_q, one_q[0]])
    s_closed = cumulative_arclength(path_closed)
    loop_length = float(s_closed[-1])
    if loop_length <= 1e-9:
        raise ValueError("Closed path length is zero.")

    q_tcks = [
        splrep(
            s_closed,
            q_closed[:, joint],
            k=5,
            s=max(0.0, joint_smoothness) * len(q_closed),
            per=True,
        )
        for joint in range(q_closed.shape[1])
    ]

    def base_q(values: np.ndarray, derivative: int = 0) -> np.ndarray:
        return np.column_stack([splev(values, tck, der=derivative) for tck in q_tcks])

    # Reparameterize the smoothed joint curve by the arc length of FK(q), not
    # by the ideal Cartesian target. This is what makes the measured TCP speed
    # approximately constant after joint-space smoothing.
    if fk_position is not None:
        source_probe = np.linspace(0.0, loop_length, max(2049, loop_size * 16), endpoint=True)
        source_q = base_q(source_probe)
        source_xyz = np.asarray([fk_position(qi) for qi in source_q], dtype=float)
        actual_arc = cumulative_arclength(source_xyz)
        keep = np.r_[True, np.diff(actual_arc) > 1e-10]
        actual_arc = actual_arc[keep]
        source_probe = source_probe[keep]
        actual_loop_length = float(actual_arc[-1])
        arc_nodes = np.linspace(0.0, actual_loop_length, max(1024, loop_size * 8), endpoint=False)
        source_nodes = np.interp(arc_nodes, actual_arc, source_probe)
        q_nodes = base_q(source_nodes)
        q_arc_closed = np.vstack([q_nodes, q_nodes[0]])
        arc_closed = np.r_[arc_nodes, actual_loop_length]
        q_spline = make_interp_spline(arc_closed, q_arc_closed, k=5, bc_type="periodic", axis=0)
        loop_coordinate_length = actual_loop_length

        def q_by_coordinate(values: np.ndarray, derivative: int = 0) -> np.ndarray:
            return q_spline(values, derivative)
    else:
        path_spline = CubicSpline(s_closed, path_closed, bc_type="periodic", axis=0)
        loop_coordinate_length = loop_length

        def q_by_coordinate(values: np.ndarray, derivative: int = 0) -> np.ndarray:
            return base_q(values, derivative)

    probe_s = np.linspace(0.0, loop_coordinate_length, max(4096, loop_size * 24), endpoint=False)
    q1 = np.abs(q_by_coordinate(probe_s, 1))
    q2 = np.abs(q_by_coordinate(probe_s, 2))
    q3 = np.abs(q_by_coordinate(probe_s, 3))

    def positive_min(values: np.ndarray) -> float:
        finite = values[np.isfinite(values) & (values > 0.0)]
        return float(np.min(finite)) if finite.size else np.inf

    speed_v = positive_min(limits.dq_max[None, :] / np.maximum(q1, 1e-12))
    speed_a = positive_min(np.sqrt(limits.ddq_max[None, :] / np.maximum(q2, 1e-12)))
    speed_j = positive_min(np.cbrt(limits.jerk_max[None, :] / np.maximum(q3, 1e-12)))
    path_speed = min(float(target_tcp_speed), speed_v, speed_a, speed_j) * 0.92
    if not np.isfinite(path_speed) or path_speed <= 1e-6:
        raise RuntimeError("No positive closed-loop path speed satisfies the joint limits.")

    total_length = loop_coordinate_length * loops
    inp = ruckig.InputParameter(1)
    trajectory = ruckig.Trajectory(1)
    inp.current_position = [0.0]
    inp.current_velocity = [path_speed]
    inp.current_acceleration = [0.0]
    inp.target_position = [total_length]
    inp.target_velocity = [path_speed]
    inp.target_acceleration = [0.0]
    inp.max_velocity = [path_speed]
    inp.max_acceleration = [max(path_speed, 0.1)]
    inp.max_jerk = [max(path_speed * 10.0, 1.0)]
    result = ruckig.Ruckig(1).calculate(inp, trajectory)
    if result < 0:
        raise RuntimeError(f"Ruckig failed for the closed path coordinate: result={result}")

    t = np.arange(0.0, trajectory.duration, dt)
    if len(t) < 2:
        t = np.linspace(0.0, trajectory.duration, 2, endpoint=False)
    path_coordinate = np.asarray([trajectory.at_time(float(ti))[0][0] for ti in t])
    path_velocity = np.asarray([trajectory.at_time(float(ti))[1][0] for ti in t])
    path_acceleration = np.asarray([trajectory.at_time(float(ti))[2][0] for ti in t])
    s_mod = np.mod(path_coordinate, loop_coordinate_length)

    q_out = q_by_coordinate(s_mod)
    q1_out = q_by_coordinate(s_mod, 1)
    q2_out = q_by_coordinate(s_mod, 2)
    q3_out = q_by_coordinate(s_mod, 3)
    dq = q1_out * path_velocity[:, None]
    ddq = q2_out * path_velocity[:, None] ** 2 + q1_out * path_acceleration[:, None]
    # The generated cyclic path law is constant-speed; retaining the general
    # acceleration terms here makes the formula explicit and auditable.
    jerk = q3_out * path_velocity[:, None] ** 3 + 3.0 * q2_out * path_velocity[:, None] * path_acceleration[:, None]
    if fk_position is not None:
        points = np.asarray([fk_position(qi) for qi in q_out], dtype=float)
        speed = np.maximum(np.gradient(cumulative_arclength(points), t, edge_order=1), 0.0)
    else:
        points = path_spline(s_mod)
        speed = np.linalg.norm(path_spline(s_mod, 1), axis=1) * path_velocity

    if np.any(q_out < limits.q_min - 1e-8) or np.any(q_out > limits.q_max + 1e-8):
        raise RuntimeError("Periodic joint spline overshoots an FR5 position limit.")
    return TimeProfile(
        t,
        points,
        speed,
        q_out,
        dq,
        ddq,
        jerk,
        method=f"ruckig-periodic-c4:{path_speed:.5f}",
    )


def _estimate_waypoint_velocities(
    q: np.ndarray,
    s: np.ndarray,
    target_tcp_speed: float,
    limits: JointLimits,
    closed_loop: bool = False,
) -> np.ndarray:
    nominal_t = s / max(target_tcp_speed, 1e-6)
    v = np.zeros_like(q)
    if len(q) > 2:
        dt = np.maximum(nominal_t[2:] - nominal_t[:-2], 1e-6)
        v[1:-1] = (q[2:] - q[:-2]) / dt[:, None]
        if closed_loop:
            v[0] = v[1]
            v[-1] = v[-2]
    v = np.clip(v, -limits.dq_max * 0.85, limits.dq_max * 0.85)
    return v


def make_segmented_ruckig_time_profile(
    path_points: np.ndarray,
    q: np.ndarray,
    limits: JointLimits,
    target_tcp_speed: float = 0.08,
    dt: float = 0.01,
    closed_loop: bool = False,
) -> TimeProfile:
    """Generate a joint trajectory with Ruckig for each waypoint segment.

    This is a real Ruckig integration for offline simulation. It uses the
    community-compatible segmented API: each neighboring waypoint pair is
    solved with velocity and acceleration constraints, then sampled at ``dt``.
    """

    try:
        import ruckig
    except ImportError as exc:
        raise RuntimeError("Ruckig is not installed. Install `ruckig` to use this profile.") from exc

    dof = q.shape[1]
    s = cumulative_arclength(path_points)
    waypoint_vel = _estimate_waypoint_velocities(q, s, target_tcp_speed, limits, closed_loop=closed_loop)
    all_t: list[float] = []
    all_q: list[np.ndarray] = []
    all_dq: list[np.ndarray] = []
    all_ddq: list[np.ndarray] = []
    all_jerk: list[np.ndarray] = []
    all_pts: list[np.ndarray] = []
    current_time = 0.0

    for i in range(len(q) - 1):
        inp = ruckig.InputParameter(dof)
        traj = ruckig.Trajectory(dof)
        inp.current_position = q[i].tolist()
        inp.current_velocity = waypoint_vel[i].tolist()
        inp.current_acceleration = np.zeros(dof).tolist()
        inp.target_position = q[i + 1].tolist()
        inp.target_velocity = waypoint_vel[i + 1].tolist()
        inp.target_acceleration = np.zeros(dof).tolist()
        inp.max_velocity = limits.dq_max.tolist()
        inp.max_acceleration = limits.ddq_max.tolist()
        inp.max_jerk = limits.jerk_max.tolist()
        inp.min_position = limits.q_min.tolist()
        inp.max_position = limits.q_max.tolist()
        ds = max(float(s[i + 1] - s[i]), 1e-6)
        inp.minimum_duration = ds / max(target_tcp_speed, 1e-6)

        result = ruckig.Ruckig(dof).calculate(inp, traj)
        if result < 0:
            raise RuntimeError(f"Ruckig failed at segment {i}: result={result}")

        n = max(2, int(np.ceil(traj.duration / dt)) + 1)
        ts = np.linspace(0.0, traj.duration, n, endpoint=True)
        prev_acc = None
        prev_t = None
        for j, local_t in enumerate(ts):
            if i > 0 and j == 0:
                continue
            pos, vel, acc = traj.at_time(float(local_t))
            pos_arr = np.asarray(pos, dtype=float)
            vel_arr = np.asarray(vel, dtype=float)
            acc_arr = np.asarray(acc, dtype=float)
            if prev_acc is None or prev_t is None or local_t == prev_t:
                jerk_arr = np.zeros(dof)
            else:
                jerk_arr = (acc_arr - prev_acc) / max(float(local_t - prev_t), 1e-9)
            alpha = 0.0 if traj.duration <= 1e-9 else float(local_t / traj.duration)
            pt = (1.0 - alpha) * path_points[i] + alpha * path_points[i + 1]
            all_t.append(current_time + float(local_t))
            all_q.append(pos_arr)
            all_dq.append(vel_arr)
            all_ddq.append(acc_arr)
            all_jerk.append(jerk_arr)
            all_pts.append(pt)
            prev_acc = acc_arr
            prev_t = local_t
        current_time += float(traj.duration)

    t = np.asarray(all_t)
    pts = np.asarray(all_pts)
    q_arr = np.asarray(all_q)
    dq_arr = np.asarray(all_dq)
    ddq_arr = np.asarray(all_ddq)
    jerk_arr = np.asarray(all_jerk)
    tcp_speed = np.maximum(np.gradient(cumulative_arclength(pts), t, edge_order=1), 0.0)
    return TimeProfile(t, pts, tcp_speed, q_arr, dq_arr, ddq_arr, jerk_arr, method="ruckig-segmented")


def make_adaptive_ruckig_time_profile(
    path_points: np.ndarray,
    q: np.ndarray,
    limits: JointLimits,
    target_tcp_speed: float = 0.08,
    dt: float = 0.01,
    max_speed_fluctuation: float = 0.05,
    closed_loop: bool = False,
    fk_position=None,
    joint_smoothness: float = 1e-6,
) -> TimeProfile:
    """Try the fastest Ruckig profile whose realized TCP speed is near-uniform."""

    if closed_loop:
        return make_closed_loop_ruckig_time_profile(
            path_points,
            q,
            limits,
            target_tcp_speed=target_tcp_speed,
            dt=dt,
            fk_position=fk_position,
            joint_smoothness=joint_smoothness,
        )

    candidates = [target_tcp_speed * scale for scale in (1.0, 0.75, 0.5, 0.375, 0.25, 0.1875, 0.125)]
    best: TimeProfile | None = None
    for speed in candidates:
        profile = make_segmented_ruckig_time_profile(path_points, q, limits, speed, dt, closed_loop=closed_loop)
        mean_speed = max(float(np.mean(profile.tcp_speed)), 1e-12)
        fluctuation = float((np.max(profile.tcp_speed) - np.min(profile.tcp_speed)) / mean_speed)
        best = profile
        if fluctuation <= max_speed_fluctuation:
            return TimeProfile(
                profile.t,
                profile.path_points,
                profile.tcp_speed,
                profile.q,
                profile.dq,
                profile.ddq,
                profile.jerk,
                method=f"ruckig-segmented-adaptive:{speed:.5f}",
            )
    assert best is not None
    return TimeProfile(best.t, best.path_points, best.tcp_speed, best.q, best.dq, best.ddq, best.jerk, method="ruckig-segmented-adaptive:lowest-tested")
