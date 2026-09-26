"""Pure-Python data structures for the Stage 0/1 layered IK graph.

This module deliberately has no ROS2, MoveIt2, pandas, or pyarrow import at
module load time. The ROS adapter lives in the exporter so offline graph and
solver tests remain deterministic and runnable on Windows.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np


def circular_joint_delta(delta: Iterable[float]) -> np.ndarray:
    """Return shortest revolute deltas in [-pi, pi)."""

    return (np.asarray(list(delta), dtype=float) + np.pi) % (2.0 * np.pi) - np.pi


def nearest_equivalent_joint_positions(positions: Iterable[float], reference: Iterable[float]) -> np.ndarray:
    """Unwrap each revolute joint to the equivalent value nearest to reference."""

    q = np.asarray(list(positions), dtype=float).copy()
    ref = np.asarray(list(reference), dtype=float)
    if q.shape != ref.shape:
        raise ValueError(f"positions/reference shape mismatch: {q.shape} vs {ref.shape}")
    for i, value in enumerate(q):
        candidates = value + 2.0 * np.pi * np.arange(-3, 4, dtype=float)
        q[i] = candidates[int(np.argmin(np.abs(candidates - ref[i])))]
    return q


@dataclass(slots=True)
class IKCandidate:
    waypoint: int
    candidate_id: str
    q_rad: np.ndarray
    q_unwrapped_rad: np.ndarray
    position_error_m: float | None = None
    normal_error_deg: float | None = None
    tcp_standoff_error_m: float | None = None
    roll_deg: float | None = None
    seed_source: str = "unknown"
    node_collision: bool | None = None
    collision_summary: str = "not_available"
    valid: bool = False
    reject_reason: str | None = None
    # Stage 1.7 provenance and formal/diagnostic separation.  All fields are
    # optional to preserve Stage 0/1 and Stage 1.5/1.6 callers.
    task_pose_candidate_id: str | None = None
    nominal_waypoint_id: int | None = None
    repair_window_id: str | None = None
    tangential_offset_mm: float | None = None
    longitudinal_offset_mm: float | None = None
    standoff_offset_mm: float | None = None
    repair_cost: float | None = None
    formal_constraint_pass: bool | None = None
    diagnostic_only: bool = False

    def __post_init__(self) -> None:
        self.q_rad = np.asarray(self.q_rad, dtype=float).reshape(-1)
        self.q_unwrapped_rad = np.asarray(self.q_unwrapped_rad, dtype=float).reshape(-1)
        if self.q_rad.shape != self.q_unwrapped_rad.shape:
            raise ValueError("q_rad and q_unwrapped_rad must have the same shape")

    def to_record(self) -> dict[str, Any]:
        return {
            "waypoint": int(self.waypoint),
            "candidate_id": self.candidate_id,
            "q_rad": self.q_rad.tolist(),
            "q_unwrapped_rad": self.q_unwrapped_rad.tolist(),
            "position_error_m": self.position_error_m,
            "normal_error_deg": self.normal_error_deg,
            "tcp_standoff_error_m": self.tcp_standoff_error_m,
            "roll_deg": self.roll_deg,
            "seed_source": self.seed_source,
            "node_collision": self.node_collision,
            "collision_summary": self.collision_summary,
            "valid": bool(self.valid),
            "reject_reason": self.reject_reason,
            "task_pose_candidate_id": self.task_pose_candidate_id,
            "nominal_waypoint_id": self.nominal_waypoint_id,
            "repair_window_id": self.repair_window_id,
            "tangential_offset_mm": self.tangential_offset_mm,
            "longitudinal_offset_mm": self.longitudinal_offset_mm,
            "standoff_offset_mm": self.standoff_offset_mm,
            "roll_offset_deg": self.roll_deg,
            "repair_cost": self.repair_cost,
            "formal_constraint_pass": self.formal_constraint_pass,
            "diagnostic_only": bool(self.diagnostic_only),
        }


@dataclass(slots=True)
class TransitionEdge:
    from_waypoint: int
    to_waypoint: int
    from_candidate_id: str
    to_candidate_id: str
    delta_q_wrapped_rad: np.ndarray
    delta_q_unwrapped_rad: np.ndarray
    max_joint_step_deg: float | None
    interpolation_count: int | None
    collision_status: str
    collision_summary: str = "not_checked"
    ccd_status: str = "not_available"
    clearance_status: str = "not_available"
    min_clearance_m: float | None = None
    min_clearance_alpha: float | None = None
    tcp_distance_error_m: float | None = None
    normal_error_deg: float | None = None
    cost: float | None = None
    valid: bool = False
    reject_reason: str | None = None

    def to_record(self) -> dict[str, Any]:
        return {
            "from_waypoint": int(self.from_waypoint),
            "to_waypoint": int(self.to_waypoint),
            "from_candidate_id": self.from_candidate_id,
            "to_candidate_id": self.to_candidate_id,
            "delta_q_wrapped_rad": np.asarray(self.delta_q_wrapped_rad, dtype=float).tolist(),
            "delta_q_unwrapped_rad": np.asarray(self.delta_q_unwrapped_rad, dtype=float).tolist(),
            "max_joint_step_deg": self.max_joint_step_deg,
            "interpolation_count": self.interpolation_count,
            "collision_status": self.collision_status,
            "collision_summary": self.collision_summary,
            "ccd_status": self.ccd_status,
            "clearance_status": self.clearance_status,
            "min_clearance_m": self.min_clearance_m,
            "min_clearance_alpha": self.min_clearance_alpha,
            "tcp_distance_error_m": self.tcp_distance_error_m,
            "normal_error_deg": self.normal_error_deg,
            "cost": self.cost,
            "valid": bool(self.valid),
            "reject_reason": self.reject_reason,
        }


@dataclass(slots=True)
class GraphSolution:
    found: bool
    candidate_ids: list[str] = field(default_factory=list)
    q_path_rad: np.ndarray | None = None
    total_cost: float | None = None
    first_unreachable_waypoint: int | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_record(self) -> dict[str, Any]:
        return {
            "found": bool(self.found),
            "candidate_ids": list(self.candidate_ids),
            "q_path_rad": None if self.q_path_rad is None else np.asarray(self.q_path_rad).tolist(),
            "total_cost": self.total_cost,
            "first_unreachable_waypoint": self.first_unreachable_waypoint,
            "diagnostics": self.diagnostics,
        }


@dataclass
class LayeredIKGraph:
    waypoint_count: int
    candidates: dict[int, list[IKCandidate]] = field(default_factory=dict)
    edges: dict[tuple[int, int], list[TransitionEdge]] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_candidate(self, candidate: IKCandidate) -> None:
        if candidate.waypoint < 0 or candidate.waypoint >= self.waypoint_count:
            raise ValueError(f"candidate waypoint out of range: {candidate.waypoint}")
        layer = self.candidates.setdefault(candidate.waypoint, [])
        if any(item.candidate_id == candidate.candidate_id for item in layer):
            raise ValueError(f"duplicate candidate id: {candidate.candidate_id}")
        layer.append(candidate)

    def add_edge(self, edge: TransitionEdge) -> None:
        if edge.to_waypoint != edge.from_waypoint + 1:
            raise ValueError("Stage 0/1 edges must connect adjacent layers")
        self.edges.setdefault((edge.from_waypoint, edge.to_waypoint), []).append(edge)

    def layer(self, waypoint: int) -> list[IKCandidate]:
        return list(self.candidates.get(waypoint, []))

    def edge_layer(self, waypoint: int) -> list[TransitionEdge]:
        return list(self.edges.get((waypoint, waypoint + 1), []))

    def layer_candidate_counts(self) -> list[int]:
        return [len(self.candidates.get(i, [])) for i in range(self.waypoint_count)]

    def edge_count(self, valid_only: bool = False) -> int:
        values = [edge for layer in self.edges.values() for edge in layer]
        return sum(bool(edge.valid) for edge in values) if valid_only else len(values)

    def node_count(self, valid_only: bool = False) -> int:
        values = [candidate for layer in self.candidates.values() for candidate in layer]
        return sum(bool(candidate.valid) for candidate in values) if valid_only else len(values)

    def node_records(self) -> list[dict[str, Any]]:
        return [candidate.to_record() for i in range(self.waypoint_count) for candidate in self.layer(i)]

    def edge_records(self) -> list[dict[str, Any]]:
        return [edge.to_record() for key in sorted(self.edges) for edge in self.edges[key]]
