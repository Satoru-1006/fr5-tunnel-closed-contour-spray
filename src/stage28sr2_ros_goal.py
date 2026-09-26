"""Deterministic conversion and semantic verification of a frozen FJT goal."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Any, Mapping


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


@dataclass
class _Duration:
    sec: int = 0
    nanosec: int = 0


@dataclass
class _Header:
    stamp: _Duration = field(default_factory=_Duration)


@dataclass
class _Point:
    positions: list[float] = field(default_factory=list)
    velocities: list[float] = field(default_factory=list)
    accelerations: list[float] = field(default_factory=list)
    effort: list[float] = field(default_factory=list)
    time_from_start: _Duration = field(default_factory=_Duration)


@dataclass
class _Trajectory:
    joint_names: list[str] = field(default_factory=list)
    points: list[_Point] = field(default_factory=list)
    header: _Header = field(default_factory=_Header)


@dataclass
class FrozenGoalMessage:
    trajectory: _Trajectory = field(default_factory=_Trajectory)
    path_tolerance: list[Any] = field(default_factory=list)
    goal_tolerance: list[Any] = field(default_factory=list)
    goal_time_tolerance: _Duration = field(default_factory=_Duration)


def _trajectory(source: Mapping[str, Any]) -> Mapping[str, Any]:
    value = source.get("trajectory")
    return value if isinstance(value, Mapping) else source


def _stamp(source: Mapping[str, Any], trajectory: Mapping[str, Any]) -> tuple[int, int]:
    for container in (trajectory, source):
        candidate = container.get("header") or container.get("trajectory_header_stamp")
        if isinstance(candidate, Mapping):
            stamp = candidate.get("stamp", candidate)
            if isinstance(stamp, Mapping):
                return int(stamp.get("sec", 0)), int(stamp.get("nanosec", 0))
    return 0, 0


def _duration(value: Any) -> _Duration:
    if not isinstance(value, Mapping):
        raise ValueError("duration must be an object")
    return _Duration(int(value.get("sec", 0)), int(value.get("nanosec", 0)))


def _tolerance(source: Any) -> list[Any]:
    if source in (None, []):
        return []
    if not isinstance(source, list):
        raise ValueError("tolerances must be a list")
    result: list[Any] = []
    for index, item in enumerate(source):
        if not isinstance(item, Mapping):
            raise ValueError(f"tolerance {index} must be an object")
        # ROS JointTolerance messages materialize unspecified fields at their
        # wire defaults.  Canonicalize both offline and generated-message
        # paths to those same defaults; this does not inject a non-zero limit.
        normalized: dict[str, Any] = {
            "name": str(item.get("name", "")),
            "position": _number(item.get("position", 0.0), f"tolerance[{index}].position"),
            "velocity": _number(item.get("velocity", 0.0), f"tolerance[{index}].velocity"),
            "acceleration": _number(item.get("acceleration", 0.0), f"tolerance[{index}].acceleration"),
        }
        result.append(normalized)
    return result


def _actual_tolerance(values: Any) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for value in values or []:
        if isinstance(value, Mapping):
            result.append(dict(value))
        else:
            result.append({
                "name": str(getattr(value, "name", "")),
                "position": float(getattr(value, "position", 0.0)),
                "velocity": float(getattr(value, "velocity", 0.0)),
                "acceleration": float(getattr(value, "acceleration", 0.0)),
            })
    return result


def _build_fallback(source: Mapping[str, Any]) -> FrozenGoalMessage:
    trajectory = _trajectory(source)
    message = FrozenGoalMessage()
    message.trajectory.joint_names = [str(value) for value in trajectory["joint_names"]]
    sec, nanosec = _stamp(source, trajectory)
    message.trajectory.header.stamp = _Duration(sec, nanosec)
    for index, item in enumerate(trajectory["points"]):
        if not isinstance(item, Mapping):
            raise ValueError(f"points[{index}] must be an object")
        message.trajectory.points.append(_Point(
            positions=[_number(value, f"points[{index}].positions") for value in item.get("positions", [])],
            velocities=[_number(value, f"points[{index}].velocities") for value in item.get("velocities", [])],
            accelerations=[_number(value, f"points[{index}].accelerations") for value in item.get("accelerations", [])],
            effort=[_number(value, f"points[{index}].effort") for value in item.get("effort", [])],
            time_from_start=_duration(item.get("time_from_start", {})),
        ))
    message.path_tolerance = _tolerance(source.get("path_tolerance", trajectory.get("path_tolerance", [])))
    message.goal_tolerance = _tolerance(source.get("goal_tolerance", trajectory.get("goal_tolerance", [])))
    message.goal_time_tolerance = _duration(source.get("goal_time_tolerance", trajectory.get("goal_time_tolerance", {})))
    return message


def build_ros_goal(source: Mapping[str, Any], *, message_types: Mapping[str, Any] | None = None) -> Any:
    """Build the actual ``control_msgs.action.FollowJointTrajectory.Goal``.

    Imports are lazy so offline tests can verify all semantics without ROS.
    ``message_types`` is a test seam for generated ROS message stand-ins.
    """

    if message_types is None:
        try:
            from control_msgs.action import FollowJointTrajectory  # type: ignore
            from control_msgs.msg import JointTolerance  # type: ignore
            from trajectory_msgs.msg import JointTrajectoryPoint  # type: ignore
        except ImportError:
            return _build_fallback(source)
    else:
        FollowJointTrajectory = message_types["FollowJointTrajectory"]
        JointTolerance = message_types["JointTolerance"]
        JointTrajectoryPoint = message_types["JointTrajectoryPoint"]
    trajectory = _trajectory(source)
    goal = FollowJointTrajectory.Goal()
    sec, nanosec = _stamp(source, trajectory)
    goal.trajectory.header.stamp.sec = sec
    goal.trajectory.header.stamp.nanosec = nanosec
    goal.trajectory.joint_names = [str(value) for value in trajectory["joint_names"]]
    for index, item in enumerate(trajectory["points"]):
        point = JointTrajectoryPoint()
        point.positions = [_number(value, f"points[{index}].positions") for value in item.get("positions", [])]
        point.velocities = [_number(value, f"points[{index}].velocities") for value in item.get("velocities", [])]
        point.accelerations = [_number(value, f"points[{index}].accelerations") for value in item.get("accelerations", [])]
        point.effort = [_number(value, f"points[{index}].effort") for value in item.get("effort", [])]
        point.time_from_start.sec = int(item["time_from_start"].get("sec", 0))
        point.time_from_start.nanosec = int(item["time_from_start"].get("nanosec", 0))
        goal.trajectory.points.append(point)

    def make_tolerances(values: Any) -> list[Any]:
        messages: list[Any] = []
        for item in _tolerance(values):
            message = JointTolerance()
            for key, value in item.items():
                setattr(message, key, value)
            messages.append(message)
        return messages

    goal.path_tolerance = make_tolerances(source.get("path_tolerance", trajectory.get("path_tolerance", [])))
    goal.goal_tolerance = make_tolerances(source.get("goal_tolerance", trajectory.get("goal_tolerance", [])))
    goal_time = _duration(source.get("goal_time_tolerance", trajectory.get("goal_time_tolerance", {})))
    goal.goal_time_tolerance.sec = goal_time.sec
    goal.goal_time_tolerance.nanosec = goal_time.nanosec
    return goal


def _duration_dict(value: Any) -> dict[str, int]:
    return {"sec": int(getattr(value, "sec", 0)), "nanosec": int(getattr(value, "nanosec", 0))}


def goal_semantics(goal: Any) -> dict[str, Any]:
    trajectory = goal.trajectory
    points: list[dict[str, Any]] = []
    for point in trajectory.points:
        points.append({
            "positions": [float(value) for value in point.positions],
            "velocities": [float(value) for value in point.velocities],
            "accelerations": [float(value) for value in point.accelerations],
            "effort": [float(value) for value in getattr(point, "effort", [])],
            "time_from_start": _duration_dict(point.time_from_start),
        })
    return {
        "joint_names": [str(value) for value in trajectory.joint_names],
        "points": points,
        "trajectory_header_stamp": _duration_dict(trajectory.header.stamp),
        "path_tolerance": _actual_tolerance(getattr(goal, "path_tolerance", [])),
        "goal_tolerance": _actual_tolerance(getattr(goal, "goal_tolerance", [])),
        "goal_time_tolerance": _duration_dict(goal.goal_time_tolerance),
    }


def _canonical_source(source: Mapping[str, Any]) -> dict[str, Any]:
    trajectory = _trajectory(source)
    sec, nanosec = _stamp(source, trajectory)
    points: list[dict[str, Any]] = []
    for index, item in enumerate(trajectory.get("points", [])):
        points.append({
            "positions": [_number(value, f"points[{index}].positions") for value in item.get("positions", [])],
            "velocities": [_number(value, f"points[{index}].velocities") for value in item.get("velocities", [])],
            "accelerations": [_number(value, f"points[{index}].accelerations") for value in item.get("accelerations", [])],
            "effort": [_number(value, f"points[{index}].effort") for value in item.get("effort", [])],
            "time_from_start": {"sec": int(item.get("time_from_start", {}).get("sec", 0)), "nanosec": int(item.get("time_from_start", {}).get("nanosec", 0))},
        })
    return {
        "joint_names": [str(value) for value in trajectory.get("joint_names", [])],
        "points": points,
        "trajectory_header_stamp": {"sec": sec, "nanosec": nanosec},
        "path_tolerance": _tolerance(source.get("path_tolerance", trajectory.get("path_tolerance", []))),
        "goal_tolerance": _tolerance(source.get("goal_tolerance", trajectory.get("goal_tolerance", []))),
        "goal_time_tolerance": {"sec": int(source.get("goal_time_tolerance", trajectory.get("goal_time_tolerance", {})).get("sec", 0)), "nanosec": int(source.get("goal_time_tolerance", trajectory.get("goal_time_tolerance", {})).get("nanosec", 0))},
    }


def verify_goal_semantics(source: Mapping[str, Any], goal: Any) -> dict[str, Any]:
    expected = _canonical_source(source)
    actual = goal_semantics(goal)
    checks = {
        "joint_order_equal": actual["joint_names"] == expected["joint_names"],
        "point_count_equal": len(actual["points"]) == len(expected["points"]),
        "positions_equal": all(a["positions"] == b["positions"] for a, b in zip(actual["points"], expected["points"])) and len(actual["points"]) == len(expected["points"]),
        "velocities_equal": all(a["velocities"] == b["velocities"] for a, b in zip(actual["points"], expected["points"])) and len(actual["points"]) == len(expected["points"]),
        "accelerations_equal": all(a["accelerations"] == b["accelerations"] for a, b in zip(actual["points"], expected["points"])) and len(actual["points"]) == len(expected["points"]),
        "time_from_start_equal": all(a["time_from_start"] == b["time_from_start"] for a, b in zip(actual["points"], expected["points"])) and len(actual["points"]) == len(expected["points"]),
        "trajectory_header_semantics_verified": actual["trajectory_header_stamp"] == expected["trajectory_header_stamp"],
        "path_tolerance_semantics_verified": actual["path_tolerance"] == expected["path_tolerance"],
        "goal_tolerance_semantics_verified": actual["goal_tolerance"] == expected["goal_tolerance"],
        "goal_time_tolerance_semantics_verified": actual["goal_time_tolerance"] == expected["goal_time_tolerance"],
    }
    differences: list[float] = []
    for actual_point, expected_point in zip(actual["points"], expected["points"]):
        for field_name in ("positions", "velocities", "accelerations"):
            differences.extend(abs(float(a) - float(b)) for a, b in zip(actual_point[field_name], expected_point[field_name]))
    max_difference = max(differences, default=0.0)
    canonical_digest_value = hashlib.sha256(json.dumps(actual, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    return {
        **checks,
        "max_numeric_difference": max_difference,
        "canonical_semantic_digest": canonical_digest_value,
        "passed": all(checks.values()) and max_difference == 0.0,
    }
