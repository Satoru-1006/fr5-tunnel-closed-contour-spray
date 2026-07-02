from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import xml.etree.ElementTree as ET

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class DHLink:
    """Standard DH link: a, alpha, d, theta_offset."""

    a: float
    alpha: float
    d: float
    theta_offset: float = 0.0


@dataclass(frozen=True)
class JointLimits:
    q_min: np.ndarray
    q_max: np.ndarray
    dq_max: np.ndarray
    ddq_max: np.ndarray
    jerk_max: np.ndarray


@dataclass(frozen=True)
class URDFJoint:
    name: str
    xyz: np.ndarray
    rpy: np.ndarray
    axis: np.ndarray


def _dh_transform(a: float, alpha: float, d: float, theta: float) -> np.ndarray:
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array(
        [
            [ct, -st * ca, st * sa, a * ct],
            [st, ct * ca, -ct * sa, a * st],
            [0.0, sa, ca, d],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=float,
    )


def _rpy_transform(xyz: np.ndarray, rpy: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = Rotation.from_euler("xyz", rpy).as_matrix()
    T[:3, 3] = xyz
    return T


def _axis_angle_transform(axis: np.ndarray, theta: float) -> np.ndarray:
    T = np.eye(4)
    axis = axis / max(np.linalg.norm(axis), 1e-12)
    T[:3, :3] = Rotation.from_rotvec(axis * theta).as_matrix()
    return T


def make_tool_frame(position: np.ndarray, inward_normal: np.ndarray, y_hint: np.ndarray | None = None) -> np.ndarray:
    """Build a TCP frame whose tool -Z axis points toward the wall normal.

    The convention is useful for spray/contact tools: the tool's local -Z axis
    points from TCP to the lining surface, while +Z points along the inward wall
    normal from the surface to the robot.
    """

    z_axis = np.asarray(inward_normal, dtype=float)
    z_axis /= max(np.linalg.norm(z_axis), 1e-12)
    y_axis = np.array([0.0, 1.0, 0.0]) if y_hint is None else np.asarray(y_hint, dtype=float)
    if abs(np.dot(y_axis, z_axis)) > 0.94:
        y_axis = np.array([1.0, 0.0, 0.0])
    x_axis = np.cross(y_axis, z_axis)
    x_axis /= max(np.linalg.norm(x_axis), 1e-12)
    y_axis = np.cross(z_axis, x_axis)
    T = np.eye(4)
    T[:3, :3] = np.column_stack([x_axis, y_axis, z_axis])
    T[:3, 3] = position
    return T


class FR5Robot:
    """FR5-compatible robot wrapper.

    This class intentionally does not invent real FR5 calibration parameters.
    Use ``from_dh`` or ``from_roboticstoolbox_urdf`` for production data. The
    ``placeholder`` constructor exists only to run the full simulation pipeline.
    """

    def __init__(
        self,
        links: Iterable[DHLink],
        limits: JointLimits,
        name: str = "FR5",
        base_transform: np.ndarray | None = None,
        urdf_joints: Iterable[URDFJoint] | None = None,
    ) -> None:
        self.links = list(links)
        self.urdf_joints = list(urdf_joints or [])
        if len(self.links) != 6 and len(self.urdf_joints) != 6:
            raise ValueError("FR5 model must contain six DH links or six URDF joints.")
        self.limits = limits
        self.name = name
        self.base_transform = np.eye(4) if base_transform is None else np.asarray(base_transform, dtype=float)
        self.base_inv = np.linalg.inv(self.base_transform)

    @classmethod
    def placeholder(cls) -> "FR5Robot":
        links = [
            DHLink(0.0, np.pi / 2, 0.152),
            DHLink(-0.425, 0.0, 0.0),
            DHLink(-0.395, 0.0, 0.0),
            DHLink(0.0, np.pi / 2, 0.125),
            DHLink(0.0, -np.pi / 2, 0.100),
            DHLink(0.0, 0.0, 0.100),
        ]
        q_min = np.deg2rad(np.array([-170, -120, -170, -190, -120, -360], dtype=float))
        q_max = np.deg2rad(np.array([170, 120, 170, 190, 120, 360], dtype=float))
        dq = np.deg2rad(np.full(6, 90.0))
        ddq = np.deg2rad(np.full(6, 180.0))
        jerk = np.deg2rad(np.full(6, 800.0))
        return cls(links, JointLimits(q_min, q_max, dq, ddq, jerk), name="FR5 placeholder DH")

    @classmethod
    def from_dh(cls, links: Iterable[DHLink], limits: JointLimits) -> "FR5Robot":
        return cls(links, limits, name="FR5 custom DH")

    @classmethod
    def from_official_urdf(
        cls,
        urdf_path: str | Path,
        joint_limits_yaml: str | Path | None = None,
    ) -> "FR5Robot":
        urdf_path = Path(urdf_path)
        root = ET.parse(urdf_path).getroot()
        records: list[tuple[URDFJoint, float, float, float]] = []
        for joint in root.findall("joint"):
            if joint.attrib.get("type") not in {"revolute", "continuous"}:
                continue
            name = joint.attrib["name"]
            if not name.startswith("j"):
                continue
            origin = joint.find("origin")
            axis_el = joint.find("axis")
            limit = joint.find("limit")
            xyz = np.fromstring(origin.attrib.get("xyz", "0 0 0") if origin is not None else "0 0 0", sep=" ")
            rpy = np.fromstring(origin.attrib.get("rpy", "0 0 0") if origin is not None else "0 0 0", sep=" ")
            axis = np.fromstring(axis_el.attrib.get("xyz", "0 0 1") if axis_el is not None else "0 0 1", sep=" ")
            if limit is not None:
                qlo = float(limit.attrib.get("lower", "-6.28318530718"))
                qhi = float(limit.attrib.get("upper", "6.28318530718"))
                qvel = float(limit.attrib.get("velocity", "1.0"))
            else:
                qlo = -2.0 * np.pi
                qhi = 2.0 * np.pi
                qvel = 1.0
            records.append((URDFJoint(name, xyz, rpy, axis), qlo, qhi, qvel))
        records = sorted(records, key=lambda item: item[0].name)
        if len(records) != 6:
            raise ValueError(f"Expected 6 FR5 joints in {urdf_path}, found {len(records)}.")
        joints = [record[0] for record in records]
        q_min_arr = np.asarray([record[1] for record in records], dtype=float)
        q_max_arr = np.asarray([record[2] for record in records], dtype=float)
        dq_arr = np.asarray([record[3] for record in records], dtype=float)
        ddq_arr = np.full(6, 0.7, dtype=float)
        if joint_limits_yaml and Path(joint_limits_yaml).exists():
            text = Path(joint_limits_yaml).read_text(encoding="utf-8")
            for i in range(6):
                marker = f"  j{i + 1}:"
                block_start = text.find(marker)
                if block_start >= 0:
                    block = text[block_start : text.find("\n  j", block_start + len(marker)) if text.find("\n  j", block_start + len(marker)) >= 0 else len(text)]
                    for line in block.splitlines():
                        stripped = line.strip()
                        if stripped.startswith("max_velocity:"):
                            dq_arr[i] = float(stripped.split(":", 1)[1])
                        elif stripped.startswith("max_acceleration:"):
                            ddq_arr[i] = float(stripped.split(":", 1)[1])
        jerk_arr = np.full(6, 8.0, dtype=float)
        return cls(
            links=[],
            limits=JointLimits(q_min_arr, q_max_arr, dq_arr, ddq_arr, jerk_arr),
            name=f"Official URDF {urdf_path.name}",
            urdf_joints=joints,
        )

    @classmethod
    def from_roboticstoolbox_urdf(cls, urdf_path: str | Path):
        """Load a URDF with Robotics Toolbox when installed.

        The returned object is the Robotics Toolbox robot so users can use its
        native solvers directly in a ROS2/MoveIt2 migration.
        """

        from roboticstoolbox import ERobot  # type: ignore

        return ERobot.URDF(str(urdf_path))

    def fk(self, q: np.ndarray) -> np.ndarray:
        T = self.base_transform.copy()
        if self.urdf_joints:
            for qi, joint in zip(q, self.urdf_joints):
                T = T @ _rpy_transform(joint.xyz, joint.rpy) @ _axis_angle_transform(joint.axis, qi)
        else:
            for qi, link in zip(q, self.links):
                T = T @ _dh_transform(link.a, link.alpha, link.d, qi + link.theta_offset)
        return T

    def all_joint_frames(self, q: np.ndarray) -> np.ndarray:
        frames = [self.base_transform.copy()]
        T = self.base_transform.copy()
        if self.urdf_joints:
            for qi, joint in zip(q, self.urdf_joints):
                T = T @ _rpy_transform(joint.xyz, joint.rpy) @ _axis_angle_transform(joint.axis, qi)
                frames.append(T.copy())
        else:
            for qi, link in zip(q, self.links):
                T = T @ _dh_transform(link.a, link.alpha, link.d, qi + link.theta_offset)
                frames.append(T.copy())
        return np.asarray(frames)

    def ik(
        self,
        target: np.ndarray,
        q_seed: np.ndarray,
        orientation_weight: float = 0.1,
        continuity_weight: float = 0.018,
        full_orientation_weight: float = 0.0,
    ) -> tuple[np.ndarray, float, bool]:
        target_pos = target[:3, 3]
        target_rot = Rotation.from_matrix(target[:3, :3])
        target_tool_z = target[:3, 2]
        q_seed = np.clip(q_seed, self.limits.q_min + 1e-6, self.limits.q_max - 1e-6)

        def residual(q: np.ndarray) -> np.ndarray:
            T = self.fk(q)
            pos_err = T[:3, 3] - target_pos
            actual_tool_z = T[:3, 2]
            axis_err = actual_tool_z - target_tool_z
            rot_err = (target_rot.inv() * Rotation.from_matrix(T[:3, :3])).as_rotvec()
            return np.r_[
                pos_err,
                orientation_weight * axis_err,
                full_orientation_weight * rot_err,
                continuity_weight * (q - q_seed),
            ]

        result = least_squares(
            residual,
            q_seed,
            bounds=(self.limits.q_min, self.limits.q_max),
            xtol=1e-8,
            ftol=1e-8,
            gtol=1e-8,
            max_nfev=160,
        )
        q = self._nearest_equivalent(result.x, q_seed)
        pos_err = float(np.linalg.norm(self.fk(q)[:3, 3] - target_pos))
        return q, pos_err, bool(result.success and pos_err < 0.035)

    def _nearest_equivalent(self, q: np.ndarray, q_ref: np.ndarray) -> np.ndarray:
        out = q.copy()
        for i in range(len(out)):
            candidates = out[i] + 2.0 * np.pi * np.arange(-2, 3)
            valid = candidates[(candidates >= self.limits.q_min[i]) & (candidates <= self.limits.q_max[i])]
            if len(valid):
                out[i] = valid[np.argmin(np.abs(valid - q_ref[i]))]
        return out

    def with_base(self, xyz: Iterable[float], yaw: float = 0.0) -> "FR5Robot":
        c, s = np.cos(yaw), np.sin(yaw)
        T = np.array(
            [
                [c, -s, 0.0, 0.0],
                [s, c, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        T[:3, 3] = np.asarray(list(xyz), dtype=float)
        return FR5Robot(self.links, self.limits, name=f"{self.name} with base", base_transform=T, urdf_joints=self.urdf_joints)

    def solve_trajectory_ik(
        self,
        poses: list[np.ndarray],
        q0: np.ndarray | None = None,
        max_step_deg: float = 180.0,
        orientation_weight: float = 0.1,
        continuity_weight: float = 0.018,
        full_orientation_weight: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        q_seed = np.zeros(6) if q0 is None else np.asarray(q0, dtype=float)
        max_step = np.deg2rad(max_step_deg)
        qs, errors, success = [], [], []
        for pose in poses:
            q, err, ok = self.ik(
                pose,
                q_seed,
                orientation_weight=orientation_weight,
                continuity_weight=continuity_weight,
                full_orientation_weight=full_orientation_weight,
            )
            step = q - q_seed
            if np.any(np.abs(step) > max_step):
                q_retry, err_retry, ok_retry = self.ik(
                    pose,
                    q_seed,
                    orientation_weight=orientation_weight,
                    continuity_weight=max(continuity_weight * 8.0, 0.1),
                    full_orientation_weight=full_orientation_weight,
                )
                if np.max(np.abs(q_retry - q_seed)) < np.max(np.abs(step)):
                    q, err, ok = q_retry, err_retry, ok_retry
                    step = q - q_seed
            if np.any(np.abs(step) > max_step):
                q = q_seed + np.clip(step, -max_step, max_step)
                err = float(np.linalg.norm(self.fk(q)[:3, 3] - pose[:3, 3]))
                ok = False
            qs.append(q)
            errors.append(err)
            success.append(ok)
            q_seed = q
        q_arr = np.unwrap(np.asarray(qs), axis=0)
        return q_arr, np.asarray(errors), np.asarray(success, dtype=bool)

    def check_joint_limits(self, q: np.ndarray, dq: np.ndarray, ddq: np.ndarray, jerk: np.ndarray) -> dict[str, bool]:
        return {
            "q": bool(np.all((q >= self.limits.q_min - 1e-9) & (q <= self.limits.q_max + 1e-9))),
            "dq": bool(np.all(np.abs(dq) <= self.limits.dq_max + 1e-9)),
            "ddq": bool(np.all(np.abs(ddq) <= self.limits.ddq_max + 1e-9)),
            "jerk": bool(np.all(np.abs(jerk) <= self.limits.jerk_max + 1e-9)),
        }
