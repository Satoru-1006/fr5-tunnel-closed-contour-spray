"""D65 -> Stage 5A JointTrajectory adapter.

This module is deliberately ROS-message independent.  It validates the frozen
CSV representation and emits a JSON description consumed by the one-goal
FollowJointTrajectory recorder.  It never interpolates, resamples, retimes,
replans, or invokes Ruckig.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np


JOINT_NAMES = [f"j{i}" for i in range(1, 7)]
FIELDS = ["t"] + [f"{joint}_{suffix}" for suffix in ("q", "dq", "ddq") for joint in JOINT_NAMES] + [f"{joint}_jerk" for joint in JOINT_NAMES]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def semantic_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class FrozenTrajectory:
    source: Path
    trajectory_id: str
    times: np.ndarray
    q: np.ndarray
    dq: np.ndarray
    ddq: np.ndarray
    jerk: np.ndarray
    source_sha256: str

    @property
    def state_count(self) -> int:
        return int(self.times.shape[0])

    @property
    def interval_count(self) -> int:
        return max(0, self.state_count - 1)


def _array(rows: list[Mapping[str, str]], suffix: str) -> np.ndarray:
    return np.asarray([[float(row[f"{joint}_{suffix}"]) for joint in JOINT_NAMES] for row in rows], dtype=float)


def load_frozen_trajectory(
    path: Path,
    *,
    trajectory_id: str,
    expected_states: int,
    expected_intervals: int,
) -> FrozenTrajectory:
    path = path.resolve()
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != FIELDS:
            raise ValueError(f"trajectory_header_mismatch:{path}:expected={FIELDS}:actual={reader.fieldnames}")
        rows = list(reader)
    if len(rows) != expected_states or len(rows) - 1 != expected_intervals:
        raise ValueError(f"trajectory_shape_mismatch:{trajectory_id}:{len(rows)}/{len(rows)-1}")
    times = np.asarray([float(row["t"]) for row in rows], dtype=float)
    q, dq, ddq, jerk = (_array(rows, suffix) for suffix in ("q", "dq", "ddq", "jerk"))
    for name, value in (("t", times), ("q", q), ("dq", dq), ("ddq", ddq), ("jerk", jerk)):
        if not np.isfinite(value).all():
            raise ValueError(f"trajectory_nonfinite:{trajectory_id}:{name}")
    if abs(float(times[0])) > 1e-15 or np.any(np.diff(times) <= 0.0):
        raise ValueError(f"trajectory_time_contract_failed:{trajectory_id}")
    return FrozenTrajectory(path, trajectory_id, times, q, dq, ddq, jerk, sha256_file(path))


def duration_dict(seconds: float) -> dict[str, Any]:
    if not math.isfinite(seconds) or seconds < 0.0:
        raise ValueError(f"invalid_duration:{seconds}")
    nanoseconds = int(round(seconds * 1_000_000_000.0))
    sec, nsec = divmod(nanoseconds, 1_000_000_000)
    return {"sec": int(sec), "nanosec": int(nsec), "seconds": float(sec) + float(nsec) * 1e-9}


def build_goal(trajectory: FrozenTrajectory) -> dict[str, Any]:
    points: list[dict[str, Any]] = []
    for i, seconds in enumerate(trajectory.times):
        points.append(
            {
                "positions": [float(value) for value in trajectory.q[i]],
                "velocities": [float(value) for value in trajectory.dq[i]],
                "accelerations": [float(value) for value in trajectory.ddq[i]],
                "time_from_start": duration_dict(float(seconds)),
            }
        )
    return {
        "schema_version": "stage5a-d65-follow-joint-trajectory-goal-v1",
        "action_type": "control_msgs/action/FollowJointTrajectory",
        "controller": "fairino5_controller",
        "joint_names": list(JOINT_NAMES),
        "points": points,
        "goal_identity": {
            "trajectory_id": trajectory.trajectory_id,
            "source": str(trajectory.source),
            "source_sha256": trajectory.source_sha256,
            "state_count": trajectory.state_count,
            "interval_count": trajectory.interval_count,
            "duration_s": float(trajectory.times[-1]),
            "positions_sha256": semantic_sha256(trajectory.q.tolist()),
            "velocities_sha256": semantic_sha256(trajectory.dq.tolist()),
            "accelerations_sha256": semantic_sha256(trajectory.ddq.tolist()),
            "jerk_sidecar_sha256": semantic_sha256(trajectory.jerk.tolist()),
            "timestamps_sha256": semantic_sha256(trajectory.times.tolist()),
        },
        "semantic_contract": {
            "positions_from_d65_q": True,
            "velocities_from_d65_dq": True,
            "accelerations_from_d65_ddq": True,
            "timestamps_from_d65_t": True,
            "jerk_is_sidecar_only": True,
            "replan": False,
            "retime": False,
            "reruckig": False,
            "resample": False,
        },
    }


def preflight(
    trajectory: FrozenTrajectory,
    *,
    position_limits: Mapping[str, tuple[float, float]],
    velocity_limits: Mapping[str, float],
    acceleration_limits: Mapping[str, float],
) -> dict[str, Any]:
    q_low = np.asarray([position_limits[joint][0] for joint in JOINT_NAMES], dtype=float)
    q_high = np.asarray([position_limits[joint][1] for joint in JOINT_NAMES], dtype=float)
    v_limit = np.asarray([velocity_limits[joint] for joint in JOINT_NAMES], dtype=float)
    a_limit = np.asarray([acceleration_limits[joint] for joint in JOINT_NAMES], dtype=float)
    q_ok = bool(np.all(trajectory.q >= q_low) and np.all(trajectory.q <= q_high))
    v_ok = bool(np.all(np.abs(trajectory.dq) <= v_limit + 1e-12))
    a_ok = bool(np.all(np.abs(trajectory.ddq) <= a_limit + 1e-12))
    result = {
        "trajectory_id": trajectory.trajectory_id,
        "source": str(trajectory.source),
        "source_sha256": trajectory.source_sha256,
        "state_count": trajectory.state_count,
        "interval_count": trajectory.interval_count,
        "joint_names": list(JOINT_NAMES),
        "time": {"finite": True, "t0_zero": abs(float(trajectory.times[0])) <= 1e-15, "strictly_increasing": bool(np.all(np.diff(trajectory.times) > 0.0)), "duration_s": float(trajectory.times[-1])},
        "arrays": {"q_shape": list(trajectory.q.shape), "dq_shape": list(trajectory.dq.shape), "ddq_shape": list(trajectory.ddq.shape), "jerk_shape": list(trajectory.jerk.shape)},
        "first_state": {"q": trajectory.q[0].tolist(), "dq": trajectory.dq[0].tolist(), "ddq": trajectory.ddq[0].tolist()},
        "final_state": {"q": trajectory.q[-1].tolist(), "dq": trajectory.dq[-1].tolist(), "ddq": trajectory.ddq[-1].tolist()},
        "limits": {"q": q_ok, "dq": v_ok, "ddq": a_ok, "position_lower_rad": q_low.tolist(), "position_upper_rad": q_high.tolist(), "max_velocity_rad_s": v_limit.tolist(), "max_acceleration_rad_s2": a_limit.tolist()},
        "passed": bool(q_ok and v_ok and a_ok),
    }
    return result
