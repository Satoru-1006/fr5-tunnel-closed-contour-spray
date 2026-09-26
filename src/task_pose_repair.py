"""Stage 1.7 task-space repair and second-order path primitives.

The module is intentionally ROS-free.  MoveIt2 is an execution backend used by
``tools/run_stage17_pose_repair.py``; this file owns the single task-pose
parameterisation, frame convention, candidate provenance, and exact second-
order dynamic-programming state.  No candidate marked diagnostic-only can be
returned by the formal-layer helpers.
"""

from __future__ import annotations

import csv
import hashlib
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
from scipy.spatial.transform import Rotation

from src.ik_candidate_graph import circular_joint_delta, nearest_equivalent_joint_positions


COLLISION_METHOD = "adaptive_discrete_interpolation"
UNAVAILABLE = "not_available"


def _unit(value: Iterable[float], *, name: str) -> np.ndarray:
    array = np.asarray(list(value), dtype=float).reshape(3)
    norm = float(np.linalg.norm(array))
    if norm < 1.0e-12:
        raise ValueError(f"{name} must be non-zero")
    return array / norm


def _unit_any(value: Iterable[float], *, name: str) -> np.ndarray:
    array = np.asarray(list(value), dtype=float).reshape(-1)
    norm = float(np.linalg.norm(array))
    if norm < 1.0e-12:
        raise ValueError(f"{name} must be non-zero")
    return array / norm


def _as_float_tuple(value: Iterable[float], size: int) -> tuple[float, ...]:
    result = tuple(float(item) for item in value)
    if len(result) != size:
        raise ValueError(f"expected {size} values, got {len(result)}")
    return result


def quaternion_to_matrix(quaternion_xyzw: Iterable[float]) -> np.ndarray:
    q = _unit_any(quaternion_xyzw, name="quaternion")
    return Rotation.from_quat(q).as_matrix()


def matrix_to_quaternion(matrix: np.ndarray) -> np.ndarray:
    return Rotation.from_matrix(np.asarray(matrix, dtype=float)).as_quat()


@dataclass(frozen=True, slots=True)
class TaskPoseRepairConfig:
    waypoint_count_expected: int
    nominal_standoff_m: float
    tcp_points_to_wall: bool
    max_joint_step_deg: float
    formal_position_mm: float
    formal_standoff_mm: float
    formal_roll_deg: float
    formal_normal_deg: float
    fk_position_error_m: float
    diagnostic_position_mm: float
    diagnostic_standoff_mm: float
    diagnostic_roll_deg: float
    diagnostic_normal_deg: float
    max_candidates_per_waypoint: int
    tangential_offsets_mm: tuple[float, ...]
    longitudinal_offsets_mm: tuple[float, ...]
    standoff_offsets_mm: tuple[float, ...]
    roll_offsets_deg: tuple[float, ...]
    windows: tuple[dict[str, Any], ...]
    cost_weights: Mapping[str, float]
    collision_interpolation_step_deg: float = 0.5
    output_directory: str = "outputs/ik_graph_stage17"

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "TaskPoseRepairConfig":
        scene = data["scene"]
        process = data["process"]
        formal = data["formal_study_constraints"]
        diagnostic = data["diagnostic_limits"]
        generation = data["candidate_generation"]
        return cls(
            waypoint_count_expected=int(scene["waypoint_count_expected"]),
            nominal_standoff_m=float(process["nominal_standoff_m"]),
            tcp_points_to_wall=bool(process.get("tcp_points_to_wall", True)),
            max_joint_step_deg=float(process["max_joint_step_deg"]),
            formal_position_mm=float(formal["tangential_longitudinal_offset_norm_max_mm"]),
            formal_standoff_mm=float(formal["standoff_offset_max_mm"]),
            formal_roll_deg=float(formal["roll_offset_max_deg"]),
            formal_normal_deg=float(formal["normal_error_max_deg"]),
            fk_position_error_m=float(formal["fk_position_error_max_m"]),
            diagnostic_position_mm=float(diagnostic["tangential_longitudinal_offset_norm_max_mm"]),
            diagnostic_standoff_mm=float(diagnostic["standoff_offset_max_mm"]),
            diagnostic_roll_deg=float(diagnostic["roll_offset_max_deg"]),
            diagnostic_normal_deg=float(diagnostic["normal_error_max_deg"]),
            max_candidates_per_waypoint=int(generation["max_task_pose_candidates_per_waypoint"]),
            tangential_offsets_mm=tuple(float(x) for x in generation["tangential_offsets_mm"]),
            longitudinal_offsets_mm=tuple(float(x) for x in generation["longitudinal_offsets_mm"]),
            standoff_offsets_mm=tuple(float(x) for x in generation["standoff_offsets_mm"]),
            roll_offsets_deg=tuple(float(x) for x in generation["roll_offsets_deg"]),
            windows=tuple(dict(x) for x in data.get("repair_windows", [])),
            cost_weights={str(k): float(v) for k, v in data.get("cost_weights", {}).items()},
            collision_interpolation_step_deg=float(data["collision"]["interpolation_step_deg"]),
            output_directory=str(data.get("outputs", {}).get("directory", "outputs/ik_graph_stage17")),
        )

    def window_for(self, waypoint_id: int) -> dict[str, Any] | None:
        for window in self.windows:
            if int(window["start_waypoint"]) <= waypoint_id <= int(window["end_waypoint"]):
                return window
        return None


@dataclass(frozen=True, slots=True)
class LocalSurfaceFrame:
    waypoint_id: int
    surface_point_m: np.ndarray
    tcp_position_m: np.ndarray
    tangent: np.ndarray
    longitudinal: np.ndarray
    normal: np.ndarray
    determinant: float
    frame_flip_corrected: bool = False
    quaternion_sign_corrected: bool = False
    surface_reconstruction_sign: float = 1.0
    surface_reconstruction_method: str = "tcp_plus_standoff_times_input_normal"

    def __post_init__(self) -> None:
        for name in ("surface_point_m", "tcp_position_m", "tangent", "longitudinal", "normal"):
            object.__setattr__(self, name, np.asarray(getattr(self, name), dtype=float).reshape(3))

    def to_record(self) -> dict[str, Any]:
        return {
            "waypoint_id": self.waypoint_id,
            "surface_x": float(self.surface_point_m[0]), "surface_y": float(self.surface_point_m[1]), "surface_z": float(self.surface_point_m[2]),
            "tcp_x": float(self.tcp_position_m[0]), "tcp_y": float(self.tcp_position_m[1]), "tcp_z": float(self.tcp_position_m[2]),
            "local_tangent_x": float(self.tangent[0]), "local_tangent_y": float(self.tangent[1]), "local_tangent_z": float(self.tangent[2]),
            "local_longitudinal_x": float(self.longitudinal[0]), "local_longitudinal_y": float(self.longitudinal[1]), "local_longitudinal_z": float(self.longitudinal[2]),
            "local_normal_x": float(self.normal[0]), "local_normal_y": float(self.normal[1]), "local_normal_z": float(self.normal[2]),
            "local_frame_determinant": float(self.determinant),
            "frame_flip_corrected": bool(self.frame_flip_corrected),
            "quaternion_sign_corrected": bool(self.quaternion_sign_corrected),
            "surface_reconstruction_sign": float(self.surface_reconstruction_sign),
            "surface_reconstruction_method": self.surface_reconstruction_method,
        }

@dataclass(frozen=True, slots=True)
class TaskPoseCandidate:
    waypoint_id: int
    source_phase_id: str
    source_csv_row: int
    task_pose_candidate_id: str
    nominal_surface_point_m: np.ndarray
    repaired_surface_point_m: np.ndarray
    nominal_tcp_position_m: np.ndarray
    repaired_tcp_position_m: np.ndarray
    nominal_quaternion_xyzw: np.ndarray
    repaired_quaternion_xyzw: np.ndarray
    tangential_offset_mm: float
    longitudinal_offset_mm: float
    standoff_offset_mm: float
    roll_offset_deg: float
    actual_position_offset_mm: float
    actual_standoff_m: float
    target_normal: np.ndarray
    normal_error_deg: float
    local_frame: LocalSurfaceFrame
    generation_strategy: str
    is_nominal: bool
    formal_constraint_pass: bool
    diagnostic_only: bool
    reject_reason: str | None = None
    node_cost: float = 0.0

    def __post_init__(self) -> None:
        for name in ("nominal_surface_point_m", "repaired_surface_point_m", "nominal_tcp_position_m", "repaired_tcp_position_m", "nominal_quaternion_xyzw", "repaired_quaternion_xyzw", "target_normal"):
            value = np.asarray(getattr(self, name), dtype=float)
            object.__setattr__(self, name, value.copy())

    def to_record(self) -> dict[str, Any]:
        record: dict[str, Any] = {
            "waypoint_id": self.waypoint_id, "source_phase_id": self.source_phase_id, "source_csv_row": self.source_csv_row,
            "task_pose_candidate_id": self.task_pose_candidate_id,
            "tangential_offset_mm": self.tangential_offset_mm, "longitudinal_offset_mm": self.longitudinal_offset_mm,
            "standoff_offset_mm": self.standoff_offset_mm, "roll_offset_deg": self.roll_offset_deg,
            "actual_position_offset_mm": self.actual_position_offset_mm, "actual_standoff_m": self.actual_standoff_m,
            "normal_error_deg": self.normal_error_deg, "generation_strategy": self.generation_strategy,
            "is_nominal": self.is_nominal, "formal_constraint_pass": self.formal_constraint_pass,
            "diagnostic_only": self.diagnostic_only, "reject_reason": self.reject_reason, "node_cost": self.node_cost,
            "nominal_surface_point_xyz_m": self.nominal_surface_point_m.tolist(), "repaired_surface_point_xyz_m": self.repaired_surface_point_m.tolist(),
            "nominal_tcp_position_xyz_m": self.nominal_tcp_position_m.tolist(), "repaired_tcp_position_xyz_m": self.repaired_tcp_position_m.tolist(),
            "nominal_quaternion_xyzw": self.nominal_quaternion_xyzw.tolist(), "repaired_quaternion_xyzw": self.repaired_quaternion_xyzw.tolist(),
            "target_normal_xyz": self.target_normal.tolist(),
        }
        record.update(self.local_frame.to_record())
        return record


@dataclass(slots=True)
class TaskPoseRepairReport:
    raw_candidate_count: int = 0
    formal_candidate_count: int = 0
    diagnostic_candidate_count: int = 0
    rejected_candidate_count: int = 0
    generation_truncated_waypoint_count: int = 0
    truncation: list[dict[str, Any]] = field(default_factory=list)


def read_pose_csv(path: Path, expected_count: int = 181) -> tuple[list[dict[str, str]], str]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != expected_count:
        raise ValueError(f"Stage 1.7 requires exactly {expected_count} pose rows, got {len(rows)}")
    required = {"x", "y", "z", "qx", "qy", "qz", "qw", "nx", "ny", "nz"}
    missing = sorted(required - set(rows[0]))
    if missing:
        raise ValueError(f"pose CSV missing columns: {missing}")
    return rows, hashlib.sha256(path.read_bytes()).hexdigest()


def recover_surface_frames(rows: Sequence[Mapping[str, str]], nominal_standoff_m: float, tcp_points_to_wall: bool = True) -> tuple[list[LocalSurfaceFrame], list[np.ndarray], list[bool]]:
    tcp = np.asarray([[float(row[axis]) for axis in ("x", "y", "z")] for row in rows], dtype=float)
    normals = np.asarray([_unit([float(row[f"n{axis}"]) for axis in ("x", "y", "z")], name="input normal") for row in rows])
    sign = 1.0 if tcp_points_to_wall else -1.0
    surface = tcp + sign * nominal_standoff_m * normals
    frames: list[LocalSurfaceFrame] = []
    quaternions: list[np.ndarray] = []
    q_corrected: list[bool] = []
    for i in range(len(rows)):
        delta = surface[min(i + 1, len(rows) - 1)] - surface[max(i - 1, 0)]
        if np.linalg.norm(delta) < 1.0e-10:
            delta = tcp[min(i + 1, len(rows) - 1)] - tcp[max(i - 1, 0)]
        n = normals[i].copy()
        frame_flip = False
        if frames and float(np.dot(n, frames[-1].normal)) < 0.0:
            n = -n
            frame_flip = True
        t = delta - float(np.dot(delta, n)) * n
        t = _unit(t, name=f"tangent[{i}]")
        if frames and float(np.dot(t, frames[-1].tangent)) < 0.0:
            t = -t
            frame_flip = True
        l = _unit(np.cross(n, t), name=f"longitudinal[{i}]")
        t = _unit(np.cross(l, n), name=f"tangent[{i}]")
        determinant = float(np.linalg.det(np.column_stack([t, l, n])))
        if determinant < 0.0:
            t = -t
            l = -l
            determinant = float(np.linalg.det(np.column_stack([t, l, n])))
            frame_flip = True
        frames.append(LocalSurfaceFrame(i, surface[i], tcp[i], t, l, n, determinant, frame_flip, False, sign))
        q = _unit_any([float(rows[i][key]) for key in ("qx", "qy", "qz", "qw")], name="input quaternion")
        corrected = bool(quaternions and float(np.dot(q, quaternions[-1])) < 0.0)
        if corrected:
            q = -q
        quaternions.append(q)
        q_corrected.append(corrected)
    frames = [LocalSurfaceFrame(f.waypoint_id, f.surface_point_m, f.tcp_position_m, f.tangent, f.longitudinal, f.normal, f.determinant, f.frame_flip_corrected, q_corrected[i], f.surface_reconstruction_sign, f.surface_reconstruction_method) for i, f in enumerate(frames)]
    return frames, quaternions, q_corrected


def _pose_quaternion_for_normal(nominal_rotation: np.ndarray, target_normal: np.ndarray, roll_deg: float) -> np.ndarray:
    n = _unit(target_normal, name="target normal")
    x = nominal_rotation[:, 0] - float(np.dot(nominal_rotation[:, 0], n)) * n
    if np.linalg.norm(x) < 1.0e-8:
        x = nominal_rotation[:, 1] - float(np.dot(nominal_rotation[:, 1], n)) * n
    x = _unit(x, name="aligned tool x")
    y = _unit(np.cross(n, x), name="aligned tool y")
    base = np.column_stack([x, y, n])
    # ``base`` maps tool-local axes to the scene.  The roll is therefore a
    # right multiplication about the tool-local z axis, exactly as in the
    # existing MoveIt bridge; using the world normal here would tilt the tool.
    rolled = base @ Rotation.from_rotvec(np.array([0.0, 0.0, math.radians(float(roll_deg))])).as_matrix()
    return matrix_to_quaternion(rolled)


def generate_task_pose_candidates(frames: Sequence[LocalSurfaceFrame], nominal_quaternions: Sequence[np.ndarray], config: TaskPoseRepairConfig, *, source_phase_id: str = "stage_1_7_task_pose_repair") -> tuple[list[list[TaskPoseCandidate]], TaskPoseRepairReport]:
    layers: list[list[TaskPoseCandidate]] = []
    report = TaskPoseRepairReport()
    position_values = [(float(x), 0.0, 0.0, "single_tangent") for x in config.tangential_offsets_mm if x != 0.0]
    position_values += [(0.0, float(x), 0.0, "single_longitudinal") for x in config.longitudinal_offsets_mm if x != 0.0]
    position_values += [(0.0, 0.0, float(x), "single_standoff") for x in config.standoff_offsets_mm if x != 0.0]
    roll_values = [(0.0, 0.0, 0.0, "single_roll", float(x)) for x in config.roll_offsets_deg if x != 0.0]
    for frame, nominal_q in zip(frames, nominal_quaternions):
        window = config.window_for(frame.waypoint_id)
        requests: list[tuple[float, float, float, float, str]] = [(0.0, 0.0, 0.0, 0.0, "nominal")]
        if window is not None:
            requests += [(a, b, c, 0.0, source) for a, b, c, source in position_values]
            requests += [(a, b, c, d, source) for a, b, c, source, d in roll_values]
        candidates: list[TaskPoseCandidate] = []
        for tangent_mm, longitudinal_mm, standoff_mm, roll_deg, strategy in requests:
            radial = math.hypot(tangent_mm, longitudinal_mm)
            formal = radial <= config.formal_position_mm + 1e-9 and abs(standoff_mm) <= config.formal_standoff_mm + 1e-9 and abs(roll_deg) <= config.formal_roll_deg + 1e-9
            diagnostic = not formal
            repaired_surface = frame.surface_point_m + 1e-3 * (tangent_mm * frame.tangent + longitudinal_mm * frame.longitudinal)
            repaired_normal = frame.normal.copy()
            direction = -1.0 if config.tcp_points_to_wall else 1.0
            repaired_tcp = repaired_surface + direction * (config.nominal_standoff_m + 1e-3 * standoff_mm) * repaired_normal
            repaired_q = _pose_quaternion_for_normal(quaternion_to_matrix(nominal_q), repaired_normal, roll_deg)
            position_offset = float(np.linalg.norm(repaired_tcp - frame.tcp_position_m) * 1000.0)
            w = config.cost_weights
            node_cost = float(w.get("task_position_offset", 1.0) * (radial / max(config.formal_position_mm, 1e-12)) ** 2 + w.get("standoff_offset", 1.0) * (standoff_mm / max(config.formal_standoff_mm, 1e-12)) ** 2 + w.get("roll_offset", 0.5) * (roll_deg / max(config.formal_roll_deg, 1e-12)) ** 2 + (w.get("nominal_anchor", 2.0) if not (tangent_mm or longitudinal_mm or standoff_mm or roll_deg) else 0.0))
            candidate = TaskPoseCandidate(frame.waypoint_id, source_phase_id, frame.waypoint_id + 2, f"wp{frame.waypoint_id:03d}:tp{len(candidates):03d}", frame.surface_point_m, repaired_surface, frame.tcp_position_m, repaired_tcp, np.asarray(nominal_q), repaired_q, tangent_mm, longitudinal_mm, standoff_mm, roll_deg, position_offset, config.nominal_standoff_m + 1e-3 * standoff_mm, repaired_normal, 0.0, frame, strategy, not (tangent_mm or longitudinal_mm or standoff_mm or roll_deg), formal, diagnostic, None if formal else "outside_formal_task_pose_tolerance", node_cost)
            candidates.append(candidate)
        raw = len(candidates)
        candidates.sort(key=lambda item: (not item.is_nominal, item.node_cost, item.task_pose_candidate_id))
        if len(candidates) > config.max_candidates_per_waypoint:
            kept = candidates[:config.max_candidates_per_waypoint]
            if not any(item.is_nominal for item in kept):
                kept[-1] = next(item for item in candidates if item.is_nominal)
            report.generation_truncated_waypoint_count += 1
            report.truncation.append({"waypoint_id": frame.waypoint_id, "raw_candidate_count": raw, "retained_candidate_count": len(kept), "ordering": "nominal_then_normalized_node_cost_then_id"})
            candidates = kept
        layers.append(candidates)
        report.raw_candidate_count += raw
        report.formal_candidate_count += sum(item.formal_constraint_pass and not item.diagnostic_only for item in candidates)
        report.diagnostic_candidate_count += sum(item.diagnostic_only for item in candidates)
    return layers, report


def evaluate_transition(source: Mapping[str, Any], target: Mapping[str, Any], *, max_joint_step_deg: float, interpolation_step_deg: float, collision_checker: Callable[[np.ndarray], tuple[bool, str]] | None = None) -> dict[str, Any]:
    q0 = np.asarray(source["q_unwrapped_rad"], dtype=float)
    q1 = nearest_equivalent_joint_positions(np.asarray(target["q_rad"], dtype=float), q0)
    delta = q1 - q0
    max_step = float(np.rad2deg(np.max(np.abs(delta))))
    result: dict[str, Any] = {"from_waypoint": int(source["waypoint_id"]), "to_waypoint": int(target["waypoint_id"]), "from_task_pose_candidate_id": source.get("task_pose_candidate_id"), "to_task_pose_candidate_id": target.get("task_pose_candidate_id"), "from_ik_candidate_id": source.get("ik_candidate_id"), "to_ik_candidate_id": target.get("ik_candidate_id"), "delta_q_wrapped_rad": circular_joint_delta(delta).tolist(), "delta_q_unwrapped_rad": delta.tolist(), "max_joint_step_deg": max_step, "total_joint_motion_rad": float(np.sum(np.abs(delta))), "interpolation_count": None, "interpolation_collision_status": "not_run", "first_collision_alpha": None, "ccd_status": UNAVAILABLE, "clearance_status": UNAVAILABLE, "valid": False, "reject_reason": None}
    if not bool(source.get("formal_constraint_pass", False)) or not bool(target.get("formal_constraint_pass", False)) or bool(source.get("diagnostic_only", False)) or bool(target.get("diagnostic_only", False)):
        result["reject_reason"] = "diagnostic_or_nonformal_node"
        return result
    if max_step > max_joint_step_deg + 1e-9:
        result["reject_reason"] = "joint_step_gate"
        return result
    count = max(2, int(math.ceil(max_step / max(interpolation_step_deg, 1e-9))) + 1)
    result["interpolation_count"] = count
    if collision_checker is None:
        result["interpolation_collision_status"] = "not_run_backend_unavailable"
        result["reject_reason"] = "planning_scene_not_run"
        return result
    for alpha in np.linspace(0.0, 1.0, count):
        colliding, summary = collision_checker(q0 + float(alpha) * delta)
        if colliding:
            result.update({"interpolation_collision_status": "fail", "first_collision_alpha": float(alpha), "collision_summary": str(summary), "reject_reason": "interpolated_collision"})
            return result
    result.update({"interpolation_collision_status": "pass", "valid": True, "collision_summary": "contacts=0"})
    return result


def solve_second_order_dp(layers: Sequence[Sequence[Mapping[str, Any]]], edge_lookup: Callable[[Mapping[str, Any], Mapping[str, Any]], Mapping[str, Any]], *, cost_weights: Mapping[str, float] | None = None) -> dict[str, Any]:
    """Exact second-order DP over ``(previous_node, current_node)`` states."""
    weights = {"repair_second_difference": 1.0, "joint_motion": 1.0, "joint_max_step": 1.0, "repair_first_difference": 1.0, "roll_first_difference": 1.0, "standoff_first_difference": 1.0}
    weights.update({str(k): float(v) for k, v in (cost_weights or {}).items()})
    if not layers:
        return {"found": False, "candidate_ids": [], "total_cost": None, "first_unreachable_waypoint": 0, "second_order_cost": 0.0}
    valid = lambda node: bool(node.get("valid", node.get("formal_constraint_pass", False))) and not bool(node.get("diagnostic_only", False))
    if len(layers) == 1:
        choices = [node for node in layers[0] if valid(node)]
        if not choices:
            return {"found": False, "candidate_ids": [], "total_cost": None, "first_unreachable_waypoint": 0, "second_order_cost": 0.0}
        best = min(choices, key=lambda node: float(node.get("node_cost", 0.0)))
        return {"found": True, "candidate_ids": [str(best["candidate_id"])], "total_cost": float(best.get("node_cost", 0.0)), "first_unreachable_waypoint": None, "second_order_cost": 0.0}
    states: dict[tuple[str, str], tuple[float, list[str], float]] = {}
    for a in layers[0]:
        if not valid(a):
            continue
        for b in layers[1]:
            if not valid(b):
                continue
            edge = edge_lookup(a, b)
            if not bool(edge.get("valid", False)):
                continue
            cost = float(a.get("node_cost", 0.0)) + float(b.get("node_cost", 0.0)) + _edge_cost(edge, weights)
            states[(str(a["candidate_id"]), str(b["candidate_id"]))] = (cost, [str(a["candidate_id"]), str(b["candidate_id"])], 0.0)
    if not states:
        return {"found": False, "candidate_ids": [], "total_cost": None, "first_unreachable_waypoint": 1, "second_order_cost": 0.0}
    for index in range(2, len(layers)):
        by_id = {str(node["candidate_id"]): node for node in layers[index - 1]}
        prev_by_id = {str(node["candidate_id"]): node for node in layers[index - 2]}
        next_states: dict[tuple[str, str], tuple[float, list[str], float]] = {}
        for (prev_id, current_id), (cost, path, second_cost) in states.items():
            prev = prev_by_id.get(prev_id)
            current = by_id.get(current_id)
            if prev is None or current is None:
                continue
            for nxt in layers[index]:
                if not valid(nxt):
                    continue
                edge = edge_lookup(current, nxt)
                if not bool(edge.get("valid", False)):
                    continue
                u0 = np.asarray(prev.get("u", [0, 0, 0, 0]), dtype=float)
                u1 = np.asarray(current.get("u", [0, 0, 0, 0]), dtype=float)
                u2 = np.asarray(nxt.get("u", [0, 0, 0, 0]), dtype=float)
                second = float(np.sum((u2 - 2.0 * u1 + u0) ** 2))
                total = cost + float(nxt.get("node_cost", 0.0)) + _edge_cost(edge, weights) + weights["repair_second_difference"] * second
                key = (current_id, str(nxt["candidate_id"]))
                if key not in next_states or total < next_states[key][0]:
                    next_states[key] = (total, path + [str(nxt["candidate_id"])], second_cost + second)
        states = next_states
        if not states:
            return {"found": False, "candidate_ids": [], "total_cost": None, "first_unreachable_waypoint": index, "second_order_cost": 0.0}
    best = min(states.values(), key=lambda item: item[0])
    return {"found": True, "candidate_ids": best[1], "total_cost": float(best[0]), "first_unreachable_waypoint": None, "second_order_cost": float(best[2])}


def _edge_cost(edge: Mapping[str, Any], weights: Mapping[str, float]) -> float:
    return weights.get("joint_motion", 1.0) * float(edge.get("total_joint_motion_rad", 0.0)) + weights.get("joint_max_step", 1.0) * float(edge.get("max_joint_step_deg", 0.0)) / 20.0 + weights.get("repair_first_difference", 1.0) * float(edge.get("repair_first_difference", 0.0)) + weights.get("roll_first_difference", 1.0) * float(edge.get("roll_first_difference", 0.0)) + weights.get("standoff_first_difference", 1.0) * float(edge.get("standoff_first_difference", 0.0))


def formal_candidate_records(layers: Sequence[Sequence[TaskPoseCandidate]]) -> list[dict[str, Any]]:
    return [candidate.to_record() for layer in layers for candidate in layer if candidate.formal_constraint_pass and not candidate.diagnostic_only]
