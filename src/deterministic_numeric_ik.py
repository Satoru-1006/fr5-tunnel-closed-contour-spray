"""Deterministic numerical IK backend for the Stage 1.9 completion run.

This backend is intentionally small and auditable.  It uses the expanded
runtime URDF already frozen by Stage 1.9.3, a fixed scipy least-squares
configuration, and the caller-provided explicit seed.  It does not call ROS,
MoveIt IK, or any random API.  MoveIt2 still validates every returned joint
vector in the completion runner before it can enter the candidate graph.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import xml.etree.ElementTree as ET

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from src.robot_model import FR5Robot, _rpy_transform


@dataclass(frozen=True, slots=True)
class NumericIKResult:
    success: bool
    q_rad: np.ndarray
    position_error_m: float
    tool_z_error_deg: float
    solver_status: int
    function_evaluations: int
    message: str


class DeterministicNumericIKSolver:
    """Fixed-policy, seed-explicit numerical IK over the frozen runtime URDF."""

    solver_name = "deterministic_numeric_urdf_scipy_least_squares"
    solver_version = "scipy-least_squares-fixed-v1"
    api_entry = "DeterministicNumericIKSolver.solve"
    hidden_random_api_calls = 0

    def __init__(
        self,
        urdf_path: str | Path,
        joint_limits_path: str | Path,
        *,
        position_tolerance_m: float = 0.006,
        tool_z_tolerance_deg: float = 10.0,
        orientation_weight: float = 3.0,
        continuity_weight: float = 0.0,
        max_nfev: int = 160,
    ) -> None:
        self.urdf_path = Path(urdf_path).resolve()
        self.joint_limits_path = Path(joint_limits_path).resolve()
        self.robot = FR5Robot.from_official_urdf(self.urdf_path, self.joint_limits_path)
        self.tcp_transform = self._read_fixed_tcp_transform()
        self.position_tolerance_m = float(position_tolerance_m)
        self.tool_z_tolerance_deg = float(tool_z_tolerance_deg)
        self.orientation_weight = float(orientation_weight)
        self.continuity_weight = float(continuity_weight)
        self.max_nfev = int(max_nfev)

    def _read_fixed_tcp_transform(self) -> np.ndarray:
        """Read the frozen wrist3_link -> spray_tcp_link fixed joint."""

        root = ET.parse(self.urdf_path).getroot()
        for joint in root.findall("joint"):
            if joint.attrib.get("name") != "spray_tcp_fixed_joint":
                continue
            origin = joint.find("origin")
            xyz = np.fromstring(origin.attrib.get("xyz", "0 0 0") if origin is not None else "0 0 0", sep=" ")
            rpy = np.fromstring(origin.attrib.get("rpy", "0 0 0") if origin is not None else "0 0 0", sep=" ")
            return _rpy_transform(xyz, rpy)
        raise ValueError("expanded runtime URDF lacks spray_tcp_fixed_joint")

    def fk_tcp(self, q: np.ndarray) -> np.ndarray:
        return self.robot.fk(q) @ self.tcp_transform

    @staticmethod
    def pose_matrix(position_xyz_m: Iterable[float], quaternion_xyzw: Iterable[float]) -> np.ndarray:
        target = np.eye(4, dtype=float)
        target[:3, :3] = Rotation.from_quat(np.asarray(list(quaternion_xyzw), dtype=float)).as_matrix()
        target[:3, 3] = np.asarray(list(position_xyz_m), dtype=float)
        return target

    def solve(self, target: np.ndarray, seed: Iterable[float]) -> NumericIKResult:
        target = np.asarray(target, dtype=float)
        q_seed = np.asarray(list(seed), dtype=float).reshape(6)
        q_seed = np.clip(q_seed, self.robot.limits.q_min + 1.0e-9, self.robot.limits.q_max - 1.0e-9)
        target_position = target[:3, 3]
        target_tool_z = target[:3, 2]
        target_tool_z /= max(float(np.linalg.norm(target_tool_z)), 1.0e-15)

        def residual(q: np.ndarray) -> np.ndarray:
            actual = self.fk_tcp(q)
            actual_tool_z = actual[:3, 2]
            actual_tool_z /= max(float(np.linalg.norm(actual_tool_z)), 1.0e-15)
            values = [*(actual[:3, 3] - target_position), *(self.orientation_weight * (actual_tool_z - target_tool_z))]
            if self.continuity_weight:
                values.extend(self.continuity_weight * (q - q_seed))
            return np.asarray(values, dtype=float)

        result = least_squares(
            residual,
            q_seed,
            bounds=(self.robot.limits.q_min, self.robot.limits.q_max),
            method="trf",
            xtol=1.0e-10,
            ftol=1.0e-10,
            gtol=1.0e-10,
            max_nfev=self.max_nfev,
        )
        q = self.robot._nearest_equivalent(np.asarray(result.x, dtype=float), q_seed)
        actual = self.fk_tcp(q)
        position_error = float(np.linalg.norm(actual[:3, 3] - target_position))
        actual_tool_z = actual[:3, 2] / max(float(np.linalg.norm(actual[:3, 2])), 1.0e-15)
        tool_z_error_deg = float(np.rad2deg(np.arccos(np.clip(np.dot(actual_tool_z, target_tool_z), -1.0, 1.0))))
        within_limits = bool(np.all(q >= self.robot.limits.q_min - 1.0e-9) and np.all(q <= self.robot.limits.q_max + 1.0e-9))
        success = bool(
            result.success
            and within_limits
            and position_error <= self.position_tolerance_m
            and tool_z_error_deg <= self.tool_z_tolerance_deg
        )
        return NumericIKResult(
            success=success,
            q_rad=q,
            position_error_m=position_error,
            tool_z_error_deg=tool_z_error_deg,
            solver_status=int(result.status),
            function_evaluations=int(result.nfev),
            message=str(result.message),
        )
