"""Stage5B shadow-only causal audit for the wp66 -> wp67 IK branch jump.

This file deliberately contains no trajectory repair logic.  It runs the
installed MoveIt KDL solver against the frozen Stage5AC endpoint poses, first
following the authoritative q66 seed and then enumerating deterministic seed
families for a layer/edge diagnostic graph.  The companion launch file builds
the control model and an in-memory j6-only shadow model.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np


JOINTS = [f"j{i}" for i in range(1, 7)]
GROUP = "fairino5_v6_group"
EE = "spray_tcp_link"
START_WAYPOINT = 66
END_WAYPOINT = 67
CONTINUITY_GATE_DEG = 20.0
CLUSTER_TOLERANCE_RAD = 1.0e-5
FK_POSITION_TOLERANCE_M = 1.0e-4
FK_ORIENTATION_TOLERANCE_RAD = 1.0e-3


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def matrix_of(value: Any) -> np.ndarray:
    raw = value.matrix() if hasattr(value, "matrix") else value
    return np.asarray(raw, dtype=float)


def pose_arrays(row: dict[str, str]) -> tuple[np.ndarray, np.ndarray]:
    position = np.asarray([float(row[key]) for key in ("x", "y", "z")], dtype=float)
    quaternion = np.asarray([float(row[key]) for key in ("qx", "qy", "qz", "qw")], dtype=float)
    norm = float(np.linalg.norm(quaternion))
    if norm <= 1.0e-15 or not np.isfinite(norm):
        raise RuntimeError("invalid_target_quaternion")
    return position, quaternion / norm


def pose_message(position: np.ndarray, quaternion_xyzw: np.ndarray):
    from geometry_msgs.msg import Pose

    pose = Pose()
    pose.position.x, pose.position.y, pose.position.z = [float(x) for x in position]
    pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = [float(x) for x in quaternion_xyzw]
    return pose


def shortest_delta(target: np.ndarray, source: np.ndarray) -> np.ndarray:
    return (np.asarray(target, dtype=float) - np.asarray(source, dtype=float) + math.pi) % (2.0 * math.pi) - math.pi


def interpolate_pose(a: dict[str, str], b: dict[str, str], fraction: float) -> tuple[np.ndarray, np.ndarray]:
    from scipy.spatial.transform import Rotation, Slerp

    pa, qa = pose_arrays(a)
    pb, qb = pose_arrays(b)
    if float(np.dot(qa, qb)) < 0.0:
        qb = -qb
    rotations = Rotation.from_quat(np.stack([qa, qb], axis=0))
    quaternion = Slerp([0.0, 1.0], rotations)([float(fraction)]).as_quat()[0]
    return (1.0 - fraction) * pa + fraction * pb, quaternion


def pose_error(actual: np.ndarray, position: np.ndarray, quaternion_xyzw: np.ndarray) -> tuple[float, float]:
    from scipy.spatial.transform import Rotation

    observed = Rotation.from_matrix(actual[:3, :3]).as_quat()
    observed /= max(float(np.linalg.norm(observed)), 1.0e-15)
    expected = np.asarray(quaternion_xyzw, dtype=float)
    expected /= max(float(np.linalg.norm(expected)), 1.0e-15)
    angle = 2.0 * math.acos(float(np.clip(abs(float(np.dot(observed, expected))), -1.0, 1.0)))
    return float(np.linalg.norm(actual[:3, 3] - position)), float(angle)


def q_from_row(row: dict[str, str]) -> np.ndarray:
    return np.asarray([float(row[f"j{i}_q"]) for i in range(1, 7)], dtype=float)


def limits_from_group(group: Any) -> tuple[np.ndarray, np.ndarray]:
    lows: list[float] = []
    highs: list[float] = []
    for bound in group.active_joint_model_bounds:
        item = bound[0] if isinstance(bound, (list, tuple)) else bound
        lows.append(float(item.min_position))
        highs.append(float(item.max_position))
    return np.asarray(lows, dtype=float), np.asarray(highs, dtype=float)


def direct_fk_record(state: Any, q: np.ndarray, target_position: np.ndarray, target_quaternion: np.ndarray, lows: np.ndarray, highs: np.ndarray) -> dict[str, Any]:
    state.set_joint_group_positions(GROUP, [float(x) for x in q])
    state.update()
    transform = matrix_of(state.get_global_link_transform(EE))
    position_error, orientation_error = pose_error(transform, target_position, target_quaternion)
    lower_margin = q - lows
    upper_margin = highs - q
    return {
        "q": q.tolist(),
        "fk_position_error_m": position_error,
        "fk_orientation_error_rad": orientation_error,
        "lower_limit_margin_rad": lower_margin.tolist(),
        "upper_limit_margin_rad": upper_margin.tolist(),
        "joint_limits": bool(np.all(q >= lows) and np.all(q <= highs)),
        "finite": bool(np.isfinite(q).all() and np.isfinite(transform).all()),
        "target_position": target_position.tolist(),
        "target_quaternion_xyzw": target_quaternion.tolist(),
    }


def jacobian_metrics(state: Any) -> dict[str, Any]:
    try:
        jacobian = np.asarray(state.get_jacobian(GROUP, EE, np.zeros(3, dtype=float), False), dtype=float)
    except TypeError:
        jacobian = np.asarray(state.get_jacobian(GROUP, EE, np.zeros(3, dtype=float)), dtype=float)
    singular = np.linalg.svd(jacobian, compute_uv=False)
    if singular.size == 0 or not np.isfinite(singular).all():
        raise RuntimeError("invalid_jacobian_svd")
    sigma_max = float(np.max(singular))
    sigma_min = float(np.min(singular))
    condition = float(sigma_max / sigma_min) if sigma_min > 1.0e-15 else math.inf
    return {
        "jacobian_shape": list(jacobian.shape),
        "singular_values": [float(x) for x in singular],
        "sigma_min": sigma_min,
        "sigma_max": sigma_max,
        "condition_number": condition,
        "manipulability": float(np.prod(singular)),
        "jacobian_status": "native_MoveIt_RobotState_SVD",
    }


def unique_q(values: list[np.ndarray], tolerance: float = CLUSTER_TOLERANCE_RAD) -> list[np.ndarray]:
    result: list[np.ndarray] = []
    for value in values:
        candidate = np.asarray(value, dtype=float)
        if not np.isfinite(candidate).all():
            continue
        if not any(float(np.max(np.abs(candidate - old))) <= tolerance for old in result):
            result.append(candidate.copy())
    return result


def periodic_legal_variants(seed: np.ndarray, lows: np.ndarray, highs: np.ndarray) -> list[np.ndarray]:
    variants = [seed.copy()]
    for joint in range(6):
        for multiple in (-1.0, 1.0):
            candidate = seed.copy()
            candidate[joint] += multiple * 2.0 * math.pi
            if lows[joint] - 1.0e-12 <= candidate[joint] <= highs[joint] + 1.0e-12:
                variants.append(candidate)
    return variants


def seed_families(q66: np.ndarray, q67: np.ndarray, previous: np.ndarray | None, lows: np.ndarray, highs: np.ndarray, sample_index: int) -> list[tuple[str, np.ndarray]]:
    """Deterministic seed families required by the causal audit contract."""

    requested: list[tuple[str, np.ndarray]] = []
    current = previous if previous is not None else q66
    requested.append(("current_branch", current.copy()))
    requested.append(("authoritative_q66", q66.copy()))
    requested.append(("authoritative_q67", q67.copy()))

    # Small deterministic perturbations around the current branch.
    for joint in range(6):
        for sign in (-1.0, 1.0):
            value = current.copy()
            value[joint] += sign * math.radians(2.0)
            requested.append(("current_branch_2deg", value))

    # Representative deterministic grid seeds over the legal model box.
    center = 0.5 * (lows + highs)
    requested.extend([
        ("deterministic_grid_center", center),
        ("deterministic_grid_lower_mid", 0.5 * lows + 0.5 * current),
        ("deterministic_grid_upper_mid", 0.5 * highs + 0.5 * current),
        ("deterministic_grid_q66_q67_mid", 0.5 * (q66 + q67)),
    ])

    rng = np.random.default_rng(20260903 + int(sample_index))
    for random_index in range(4):
        requested.append(("fixed_random", lows + rng.random(6) * (highs - lows)))

    # Known arm/wrist alternatives observed in the existing Stage5B probes.
    for base_name, base in (("q66", q66), ("q67", q67)):
        for signs in ((-1.0, -1.0, -1.0), (-1.0, 1.0, 1.0), (1.0, -1.0, 1.0), (1.0, 1.0, -1.0)):
            value = base.copy()
            for joint, sign in zip((0, 2, 4), signs):
                value[joint] += sign * math.pi
            requested.append((f"known_alternative_{base_name}", value))

    for name, base in (("q66", q66), ("q67", q67), ("current", current)):
        for value in periodic_legal_variants(base, lows, highs):
            requested.append(("legal_periodic_" + name, value))

    # Boundary seeds, especially the j6 upper boundary, are diagnostic only.
    for label, value in (("j6_lower_boundary", lows.copy()), ("j6_upper_boundary", highs.copy())):
        boundary = current.copy()
        boundary[5] = value[5]
        requested.append((label, boundary))

    result: list[tuple[str, np.ndarray]] = []
    seen: list[np.ndarray] = []
    for family, value in requested:
        candidate = np.asarray(value, dtype=float)
        if not np.isfinite(candidate).all():
            continue
        if any(float(np.max(np.abs(candidate - old))) <= CLUSTER_TOLERANCE_RAD for old in seen):
            continue
        seen.append(candidate.copy())
        result.append((family, candidate))
    return result


def solve_one(state: Any, seed: np.ndarray, position: np.ndarray, quaternion: np.ndarray, timeout_s: float) -> tuple[bool, np.ndarray | None, float]:
    state.set_joint_group_positions(GROUP, [float(x) for x in seed])
    state.update()
    started = time.perf_counter()
    solved = bool(state.set_from_ik(GROUP, pose_message(position, quaternion), EE, float(timeout_s)))
    elapsed = time.perf_counter() - started
    if not solved:
        return False, None, elapsed
    state.update()
    return True, np.asarray(state.get_joint_group_positions(GROUP), dtype=float).copy(), elapsed


def run_layer_enumeration(model: Any, poses: list[dict[str, str]], q66: np.ndarray, q67: np.ndarray, lows: np.ndarray, highs: np.ndarray, timeout_s: float, subdivisions: int, output: Path, mode: str) -> list[dict[str, Any]]:
    state = __import__("moveit.core.robot_state", fromlist=["RobotState"]).RobotState(model)
    layers: list[dict[str, Any]] = []
    previous_layer: list[np.ndarray] = []
    with (output / f"ik_layers_{mode}.jsonl").open("w", encoding="utf-8") as handle:
        for sample_index in range(subdivisions + 1):
            fraction = float(sample_index) / float(subdivisions)
            position, quaternion = interpolate_pose(poses[START_WAYPOINT], poses[END_WAYPOINT], fraction)
            previous_seed = previous_layer[0] if previous_layer else None
            seeds = seed_families(q66, q67, previous_seed, lows, highs, sample_index)
            candidates: list[dict[str, Any]] = []
            attempts = 0
            successes = 0
            for family, seed in seeds:
                attempts += 1
                solved, q, elapsed = solve_one(state, seed, position, quaternion, timeout_s)
                if not solved or q is None:
                    continue
                successes += 1
                state.update()
                fk = matrix_of(state.get_global_link_transform(EE))
                pos_error, rot_error = pose_error(fk, position, quaternion)
                valid = bool(
                    np.isfinite(q).all()
                    and np.all(q >= lows)
                    and np.all(q <= highs)
                    and pos_error <= FK_POSITION_TOLERANCE_M
                    and rot_error <= FK_ORIENTATION_TOLERANCE_RAD
                )
                if not valid:
                    continue
                if any(float(np.max(np.abs(q - np.asarray(item["q"], dtype=float)))) <= CLUSTER_TOLERANCE_RAD for item in candidates):
                    continue
                candidates.append({"cluster_index": len(candidates), "seed_family": family, "q": q.tolist(), "ik_elapsed_s": elapsed, "fk_position_error_m": pos_error, "fk_orientation_error_rad": rot_error})
            candidates.sort(key=lambda item: tuple(float(x) for x in item["q"]))
            for index, item in enumerate(candidates):
                item["cluster_index"] = index
            current_layer = [np.asarray(item["q"], dtype=float) for item in candidates]
            min_bridge = None
            min_l2 = None
            left_index = None
            right_index = None
            if previous_layer and current_layer:
                bridge_rows = []
                for left_index_candidate, left in enumerate(previous_layer):
                    for right_index_candidate, right in enumerate(current_layer):
                        delta = shortest_delta(right, left)
                        bridge_rows.append((float(np.rad2deg(np.max(np.abs(delta)))), float(np.linalg.norm(delta)), left_index_candidate, right_index_candidate))
                bridge_rows.sort()
                min_bridge, min_l2, left_index, right_index = bridge_rows[0]
            row = {
                "mode": mode,
                "sample_index": sample_index,
                "s": fraction,
                "target_x": float(position[0]),
                "target_y": float(position[1]),
                "target_z": float(position[2]),
                "candidate_count": len(candidates),
                "seed_attempts": attempts,
                "solver_successes": successes,
                "cluster_tolerance_rad": CLUSTER_TOLERANCE_RAD,
                "minimum_bridge_deg": min_bridge,
                "minimum_bridge_l2_rad": min_l2,
                "bridge_left_cluster": left_index,
                "bridge_right_cluster": right_index,
            }
            layers.append(row)
            handle.write(json.dumps({"statistics": row, "candidates": candidates}, ensure_ascii=False, allow_nan=False) + "\n")
            previous_layer = current_layer

    with (output / f"ik_layer_statistics_{mode}.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = list(layers[0].keys()) if layers else ["mode", "sample_index"]
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(layers)
    return layers


def run_continuation(model: Any, group: Any, poses: list[dict[str, str]], source_q: list[np.ndarray], lows: np.ndarray, highs: np.ndarray, timeout_s: float, subdivisions: int, output: Path, mode: str) -> dict[str, Any]:
    from moveit.core.robot_state import RobotState

    state = RobotState(model)
    q66 = source_q[START_WAYPOINT]
    q67 = source_q[END_WAYPOINT]
    records: list[dict[str, Any]] = []
    previous: np.ndarray | None = None
    first_failure: dict[str, Any] | None = None
    last_success: dict[str, Any] | None = None
    stopped = False
    for sample_index in range(subdivisions + 1):
        fraction = float(sample_index) / float(subdivisions)
        position, quaternion = interpolate_pose(poses[START_WAYPOINT], poses[END_WAYPOINT], fraction)
        base: dict[str, Any] = {
            "sample_index": sample_index,
            "s": fraction,
            "target_x": float(position[0]),
            "target_y": float(position[1]),
            "target_z": float(position[2]),
            "target_qx": float(quaternion[0]),
            "target_qy": float(quaternion[1]),
            "target_qz": float(quaternion[2]),
            "target_qw": float(quaternion[3]),
            "ik_success": False,
            "solver_timeout": False,
            "solver_attempts": 0,
            "solver_status": "NOT_RUN_AFTER_FIRST_FAILURE" if stopped else "PENDING",
            "termination_reason": "not_exposed_by_MoveItPy",
            "solver_elapsed_s": 0.0,
            "iterations": None,
            "seed_q": None if previous is None else previous.tolist(),
            "q": None,
            "raw_delta_rad": None,
            "wrap_delta_rad": None,
            "raw_delta_deg": None,
            "wrap_delta_deg": None,
            "max_wrap_delta_deg": None,
            "lower_limit_margin_rad": None,
            "upper_limit_margin_rad": None,
            "joint_limit_margin_min_rad": None,
            "closest_limit_joint": None,
            "fk_position_error_m": None,
            "fk_orientation_error_rad": None,
            "jacobian_singular_values": None,
            "sigma_min": None,
            "sigma_max": None,
            "condition_number": None,
            "manipulability": None,
            "branch_continuous_from_previous": None,
            "branch_failure_class": None,
        }
        if stopped:
            records.append(base)
            continue

        if sample_index == 0:
            q = q66.copy()
            fk_record = direct_fk_record(state, q, position, quaternion, lows, highs)
            jac = jacobian_metrics(state)
            solved = True
            elapsed = 0.0
            solver_status = "authoritative_q66_seed_direct_fk"
        else:
            seed = previous.copy() if previous is not None else q66.copy()
            solved, q, elapsed = solve_one(state, seed, position, quaternion, timeout_s)
            solver_status = "KDL_IK_SUCCESS" if solved else "KDL_IK_FAILURE"
            fk_record = None
            jac = None
            if solved and q is not None:
                fk_record = direct_fk_record(state, q, position, quaternion, lows, highs)
                jac = jacobian_metrics(state)
        base["ik_success"] = bool(solved)
        base["solver_attempts"] = 0 if sample_index == 0 else 1
        base["solver_elapsed_s"] = elapsed
        base["solver_timeout"] = bool(sample_index != 0 and elapsed > max(timeout_s * 1.5, timeout_s + 1.0e-3))
        base["solver_status"] = solver_status
        if solved and q is not None and fk_record is not None and jac is not None:
            raw = None if previous is None else q - previous
            wrapped = None if previous is None else shortest_delta(q, previous)
            step_deg = None if wrapped is None else float(np.rad2deg(np.max(np.abs(wrapped))))
            q_margin = np.minimum(fk_record["lower_limit_margin_rad"], fk_record["upper_limit_margin_rad"])
            closest = int(np.argmin(np.asarray(q_margin, dtype=float)))
            continuous = None if previous is None else bool(
                step_deg <= CONTINUITY_GATE_DEG
                and fk_record["joint_limits"]
                and fk_record["finite"]
                and fk_record["fk_position_error_m"] <= FK_POSITION_TOLERANCE_M
                and fk_record["fk_orientation_error_rad"] <= FK_ORIENTATION_TOLERANCE_RAD
            )
            base.update({
                "q": q.tolist(),
                "raw_delta_rad": None if raw is None else raw.tolist(),
                "wrap_delta_rad": None if wrapped is None else wrapped.tolist(),
                "raw_delta_deg": None if raw is None else np.rad2deg(raw).tolist(),
                "wrap_delta_deg": None if wrapped is None else np.rad2deg(wrapped).tolist(),
                "max_wrap_delta_deg": step_deg,
                "lower_limit_margin_rad": fk_record["lower_limit_margin_rad"],
                "upper_limit_margin_rad": fk_record["upper_limit_margin_rad"],
                "joint_limit_margin_min_rad": float(q_margin[closest]),
                "closest_limit_joint": JOINTS[closest],
                "fk_position_error_m": fk_record["fk_position_error_m"],
                "fk_orientation_error_rad": fk_record["fk_orientation_error_rad"],
                "jacobian_singular_values": jac["singular_values"],
                "sigma_min": jac["sigma_min"],
                "sigma_max": jac["sigma_max"],
                "condition_number": jac["condition_number"],
                "manipulability": jac["manipulability"],
                "branch_continuous_from_previous": continuous,
            })
            if not fk_record["joint_limits"]:
                base["branch_failure_class"] = "joint_limit_violation"
            elif fk_record["fk_position_error_m"] > FK_POSITION_TOLERANCE_M or fk_record["fk_orientation_error_rad"] > FK_ORIENTATION_TOLERANCE_RAD:
                base["branch_failure_class"] = "fk_invalid"
            elif continuous is False:
                base["branch_failure_class"] = "continuity_break_returned_by_kdl"
            if continuous is False:
                first_failure = dict(base)
                stopped = True
            else:
                previous = q.copy()
                last_success = dict(base)
        else:
            base["branch_failure_class"] = "ik_solver_failure"
            first_failure = dict(base)
            stopped = True
        records.append(base)

    continuation_rows = [row for row in records if row["q"] is not None]
    max_step = max((float(row["max_wrap_delta_deg"]) for row in continuation_rows if row["max_wrap_delta_deg"] is not None), default=None)
    max_j6_step = max((abs(float(row["wrap_delta_deg"][5])) for row in continuation_rows if row["wrap_delta_deg"] is not None), default=None)
    summary = {
        "mode": mode,
        "solver": "MoveIt KDL KinematicsPlugin via RobotState.set_from_ik",
        "model_joint_names": list(group.active_joint_model_names),
        "joint_lower_rad": lows.tolist(),
        "joint_upper_rad": highs.tolist(),
        "ik_timeout_s": timeout_s,
        "sample_intervals": subdivisions,
        "sample_count": len(records),
        "ik_success_count": sum(1 for row in records if row["ik_success"]),
        "ik_success_rate": float(sum(1 for row in records if row["ik_success"]) / len(records)) if records else 0.0,
        "first_branch_failure_sample": first_failure,
        "last_continuous_success_sample": last_success,
        "max_wrap_aware_step_deg": max_step,
        "max_j6_wrap_aware_step_deg": max_j6_step,
        "completed_to_endpoint": bool(records and records[-1]["s"] == 1.0 and records[-1]["q"] is not None),
        "continuation_interpolation": "linear_position_shortest_quaternion_slerp",
        "selection_rule": "same previous successful q is next IK seed; stop at first continuity/solver failure",
        "records": records,
    }
    write_json(output / f"continuation_{mode}_{subdivisions}.json", summary)
    fields = [
        "sample_index", "s", "target_x", "target_y", "target_z", "target_qx", "target_qy", "target_qz", "target_qw",
        "ik_success", "solver_timeout", "solver_attempts", "solver_elapsed_s", "solver_status", "termination_reason", "iterations",
        "seed_q", "q", "raw_delta_rad", "wrap_delta_rad", "raw_delta_deg", "wrap_delta_deg", "max_wrap_delta_deg",
        "lower_limit_margin_rad", "upper_limit_margin_rad", "joint_limit_margin_min_rad", "closest_limit_joint",
        "fk_position_error_m", "fk_orientation_error_rad", "jacobian_singular_values", "sigma_min", "sigma_max",
        "condition_number", "manipulability", "branch_continuous_from_previous", "branch_failure_class",
    ]
    with (output / f"continuation_{mode}_{subdivisions}.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in records:
            writer.writerow({field: json.dumps(row[field], ensure_ascii=False) if isinstance(row[field], (list, dict)) else row[field] for field in fields})
    with (output / f"jacobian_diagnostics_{mode}_{subdivisions}.csv").open("w", encoding="utf-8", newline="") as handle:
        jac_fields = ["sample_index", "s", "q", "sigma_1", "sigma_2", "sigma_3", "sigma_4", "sigma_5", "sigma_6", "sigma_min", "sigma_max", "condition_number", "manipulability", "closest_limit_joint", "joint_limit_margin_min_rad"]
        writer = csv.DictWriter(handle, fieldnames=jac_fields, lineterminator="\n")
        writer.writeheader()
        for row in records:
            singular = row["jacobian_singular_values"] or []
            writer.writerow({
                "sample_index": row["sample_index"], "s": row["s"], "q": json.dumps(row["q"], ensure_ascii=False) if row["q"] is not None else None,
                **{f"sigma_{i + 1}": singular[i] if i < len(singular) else None for i in range(6)},
                "sigma_min": row["sigma_min"], "sigma_max": row["sigma_max"], "condition_number": row["condition_number"], "manipulability": row["manipulability"],
                "closest_limit_joint": row["closest_limit_joint"], "joint_limit_margin_min_rad": row["joint_limit_margin_min_rad"],
            })
    return summary


def run(node: Any) -> int:  # pragma: no cover - requires ROS 2 / MoveIt runtime
    from moveit.core.robot_state import RobotState
    from moveit.planning import MoveItPy

    poses_path = Path(str(node.get_parameter("poses_csv").value)).resolve()
    source_path = Path(str(node.get_parameter("source_q_csv").value)).resolve()
    output = Path(str(node.get_parameter("output_dir").value)).resolve()
    model_path = Path(str(node.get_parameter("model_path").value)).resolve()
    mode = str(node.get_parameter("mode").value)
    subdivisions = int(node.get_parameter("subdivisions").value)
    timeout_s = float(node.get_parameter("ik_timeout_s").value)
    if mode not in {"control_175", "shadow_360"}:
        raise RuntimeError(f"invalid_audit_mode:{mode}")
    if subdivisions < 2 or timeout_s <= 0.0:
        raise RuntimeError("invalid_audit_parameters")
    poses = read_csv(poses_path)
    source_rows = read_csv(source_path)
    if len(poses) != 181 or len(source_rows) != 181:
        raise RuntimeError(f"authoritative_shape_mismatch:{len(poses)}/{len(source_rows)}")
    source_q = [q_from_row(row) for row in source_rows]

    moveit = MoveItPy(node_name=f"stage5b_ik_root_cause_{mode}")
    model = moveit.get_robot_model()
    group = model.get_joint_model_group(GROUP)
    if group is None or list(group.active_joint_model_names) != JOINTS:
        raise RuntimeError(f"joint_mapping_mismatch:{list(group.active_joint_model_names) if group else None}")
    lows, highs = limits_from_group(group)
    requested_model_j6 = None
    if model_path.is_file():
        description_root = ET.parse(model_path).getroot()
        for description_j6 in description_root.iter("joint"):
            if description_j6.attrib.get("name") != "j6":
                continue
            description_limit = description_j6.find("limit")
            if description_limit is not None:
                requested_model_j6 = {"lower_rad": float(description_limit.attrib["lower"]), "upper_rad": float(description_limit.attrib["upper"])}
                break
    output.mkdir(parents=True, exist_ok=True)

    source_fk: list[dict[str, Any]] = []
    fk_state = RobotState(model)
    for waypoint in range(65, 69):
        position, quaternion = pose_arrays(poses[waypoint])
        record = direct_fk_record(fk_state, source_q[waypoint], position, quaternion, lows, highs)
        jac = jacobian_metrics(fk_state)
        source_fk.append({"waypoint": waypoint, "source_q": source_q[waypoint].tolist(), **record, **jac})
    write_json(output / "ground_truth_fk_65_68.json", {"mode": mode, "rows": source_fk})

    continuation_256 = run_continuation(model, group, poses, source_q, lows, highs, timeout_s, 256, output, mode)
    continuation_512 = run_continuation(model, group, poses, source_q, lows, highs, timeout_s, 512, output, mode)
    layers = run_layer_enumeration(model, poses, source_q[START_WAYPOINT], source_q[END_WAYPOINT], lows, highs, timeout_s, 512, output, mode)
    with (output / f"ik_layer_transition_graph_{mode}.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = ["mode", "left_sample_index", "right_sample_index", "left_s", "right_s", "candidate_count_before", "candidate_count_after", "minimum_bridge_deg", "minimum_bridge_l2_rad", "bridge_left_cluster", "bridge_right_cluster"]
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for left, right in zip(layers, layers[1:]):
            writer.writerow({
                "mode": mode, "left_sample_index": left["sample_index"], "right_sample_index": right["sample_index"], "left_s": left["s"], "right_s": right["s"],
                "candidate_count_before": left["candidate_count"], "candidate_count_after": right["candidate_count"], "minimum_bridge_deg": right["minimum_bridge_deg"], "minimum_bridge_l2_rad": right["minimum_bridge_l2_rad"],
                "bridge_left_cluster": right["bridge_left_cluster"], "bridge_right_cluster": right["bridge_right_cluster"],
            })
    run_metadata = {
        "schema_version": "stage5b-ik-root-cause-run-v1",
        "mode": mode,
        "scope": "ROOT_CAUSE_AUDIT_SHADOW_DIAGNOSTIC_ONLY",
        "model_path": str(model_path),
        "model_sha256": sha256(model_path) if model_path.is_file() else None,
        "poses_path": str(poses_path),
        "source_q_path": str(source_path),
        "poses_sha256": sha256(poses_path),
        "source_q_sha256": sha256(source_path),
        "joint_names": JOINTS,
        "joint_lower_rad": lows.tolist(),
        "joint_upper_rad": highs.tolist(),
        "requested_model_j6_limit_rad": requested_model_j6,
        "effective_moveit_j6_limit_rad": {"lower_rad": float(lows[5]), "upper_rad": float(highs[5])},
        "requested_vs_effective_j6_limit_match": bool(requested_model_j6 is not None and math.isclose(requested_model_j6["lower_rad"], float(lows[5]), abs_tol=1.0e-12) and math.isclose(requested_model_j6["upper_rad"], float(highs[5]), abs_tol=1.0e-12)),
        "kinematics_solver": "kdl_kinematics_plugin/KDLKinematicsPlugin",
        "ik_timeout_s": timeout_s,
        "grids": [256, 512],
        "layer_grid": 512,
        "collision_filter": "not_used_for_branch_selection; empty-world causal IK audit",
        "formal_model_untouched": True,
        "formal_trajectory_untouched": True,
        "promotion": "NO_PROMOTION",
        "gazebo": "NOT_RUN",
        "continuation_256_summary": {key: value for key, value in continuation_256.items() if key != "records"},
        "continuation_512_summary": {key: value for key, value in continuation_512.items() if key != "records"},
        "layer_count": len(layers),
        "layer_candidate_count_min": min((int(row["candidate_count"]) for row in layers), default=None),
        "layer_candidate_count_max": max((int(row["candidate_count"]) for row in layers), default=None),
    }
    write_json(output / "run_metadata.json", run_metadata)
    print(json.dumps({"mode": mode, "subdivisions": [256, 512], "first_failure_s": (continuation_512.get("first_branch_failure_sample") or {}).get("s"), "completed_to_endpoint": continuation_512["completed_to_endpoint"], "layer_count": len(layers)}, ensure_ascii=False), flush=True)
    return 0


def main() -> int:  # pragma: no cover - requires ROS 2 / MoveIt runtime
    import rclpy
    from rclpy.node import Node

    rclpy.init()
    node = Node("stage5b_ik_root_cause_audit_wrapper")
    node.declare_parameter("poses_csv", "")
    node.declare_parameter("source_q_csv", "")
    node.declare_parameter("output_dir", "")
    node.declare_parameter("model_path", "")
    node.declare_parameter("mode", "control_175")
    node.declare_parameter("subdivisions", 512)
    node.declare_parameter("ik_timeout_s", 0.005)
    try:
        return run(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
