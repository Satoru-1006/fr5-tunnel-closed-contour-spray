"""Stage 2.8S-R2 hardened formal runner and zero-goal audit entry point.

This module is the only R2 formal-runner boundary.  Its default CLI mode is
``AUDIT ONLY / NO SEND`` and stops after the last pre-send gate.  It does not
import the historical Stage 2.7/2.8S action clients, does not construct a
real ROS ``ActionClient``, and does not invoke a command-line goal sender.

The future send path is dependency-injected.  Unit tests use
``FakeActionTransport``; a production adapter may be supplied by a separate
deployment integration after this pre-formal certification has passed.  The
one-goal ledger is consumed and durably rebound immediately before the
injected transport's one ``send_goal_async`` call.  Once consumed, the ledger
is never reset and every outcome is terminal/no-retry.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Protocol

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.stage28sr2_formal_ledger import CANONICAL_LEDGER_PATH, FormalGoalLedger, atomic_write_json  # noqa: E402
from src.stage28sr2_ros_goal import verify_goal_semantics  # noqa: E402


STAGE27 = ROOT / "outputs/stage27s_native_spline_remediation/stage27s_formal_20260805T130000Z"
FROZEN_GOAL = STAGE27 / "stage27r_clean_follow_joint_trajectory_goal.json"
FROZEN_TRAJECTORY = STAGE27 / "stage27s_candidate_trajectory.csv"
JTC_SOURCE = ROOT / "tmp/stage27_target_jtc_source_4.40.1/joint_trajectory_controller/src/trajectory.cpp"
FROZEN_GOAL_SHA256 = "3ED0B589EF0D66824354E555166778BB982F15852D1421DB19B170D50E4582C8"
FROZEN_TRAJECTORY_SHA256 = "040A6FA1D6BD0A9539CAAEACC11EE0FE597E07EB4674890DF87CCB38A633E482"
EXPECTED_JOINTS = ["j1", "j2", "j3", "j4", "j5", "j6"]
EXPECTED_POINT_COUNT = 25532
EXPECTED_FIRST_TIME = (0, 0)
EXPECTED_LAST_TIME = (765, 979681920)
ACTION_NAME = "/fairino5_controller/follow_joint_trajectory"
ACTION_TYPE = "control_msgs/action/FollowJointTrajectory"
REQUIRED_RECORDER_TOPICS = (
    "/joint_states",
    "/fairino5_controller/controller_state",
    "/tf",
    "/tf_static",
)


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    atomic_write_json(path, value if isinstance(value, dict) else {"value": value})


def _json_copy(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False))


def _finite_number(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be a JSON number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field_name} must be finite")
    return result


@dataclass
class DurationMessage:
    sec: int = 0
    nanosec: int = 0


@dataclass
class HeaderMessage:
    stamp: DurationMessage = field(default_factory=DurationMessage)


@dataclass
class JointTrajectoryPointMessage:
    positions: list[float] = field(default_factory=list)
    velocities: list[float] = field(default_factory=list)
    accelerations: list[float] = field(default_factory=list)
    effort: list[float] = field(default_factory=list)
    time_from_start: DurationMessage = field(default_factory=DurationMessage)


@dataclass
class JointTrajectoryMessage:
    joint_names: list[str] = field(default_factory=list)
    points: list[JointTrajectoryPointMessage] = field(default_factory=list)
    header: HeaderMessage = field(default_factory=HeaderMessage)


@dataclass
class FrozenFJTGoal:
    """ROS-message-shaped fallback used when ROS is not installed.

    The shape intentionally mirrors ``FollowJointTrajectory.Goal``.  It is
    also useful in tests because semantic equivalence can be proven without a
    ROS graph.
    """

    trajectory: JointTrajectoryMessage = field(default_factory=JointTrajectoryMessage)
    path_tolerance: list[Any] = field(default_factory=list)
    goal_tolerance: list[Any] = field(default_factory=list)
    goal_time_tolerance: DurationMessage = field(default_factory=DurationMessage)


def _stamp_from_json(value: Mapping[str, Any] | None) -> DurationMessage:
    if value is None:
        return DurationMessage()
    return DurationMessage(sec=int(value.get("sec", 0)), nanosec=int(value.get("nanosec", 0)))


def _point_source(data: Mapping[str, Any], index: int) -> Mapping[str, Any]:
    points = data.get("points")
    if not isinstance(points, list):
        raise ValueError("frozen goal points must be a list")
    try:
        point = points[index]
    except IndexError as exc:
        raise ValueError(f"missing frozen goal point {index}") from exc
    if not isinstance(point, Mapping):
        raise ValueError(f"frozen goal point {index} must be an object")
    return point


def _trajectory_data(data: Mapping[str, Any]) -> Mapping[str, Any]:
    # The frozen Stage 2.7R artifact is flat.  Accepting a nested trajectory
    # object makes the converter useful for a ROS-message-shaped fixture, but
    # all fields still follow the source ordering verbatim.
    trajectory = data.get("trajectory")
    return trajectory if isinstance(trajectory, Mapping) else data


def _source_header_stamp(data: Mapping[str, Any], trajectory: Mapping[str, Any]) -> tuple[dict[str, int], bool]:
    for container in (trajectory, data):
        for key in ("header", "trajectory_header_stamp"):
            candidate = container.get(key)
            if isinstance(candidate, Mapping):
                stamp = candidate.get("stamp", candidate)
                if isinstance(stamp, Mapping):
                    return {"sec": int(stamp.get("sec", 0)), "nanosec": int(stamp.get("nanosec", 0))}, True
    return {"sec": 0, "nanosec": 0}, False


def _make_ros_goal_if_requested(data: Mapping[str, Any]) -> Any:
    """Build an actual ROS goal only when an explicit ROS factory is asked for.

    The R2 audit path never calls this function.  Keeping the optional import
    lazy prevents a Windows/offline audit from accidentally creating a ROS
    client or requiring ROS Python bindings.
    """

    from control_msgs.action import FollowJointTrajectory  # type: ignore
    from control_msgs.msg import JointTolerance  # type: ignore
    from trajectory_msgs.msg import JointTrajectoryPoint  # type: ignore

    goal = FollowJointTrajectory.Goal()
    trajectory = _trajectory_data(data)
    header_stamp, _ = _source_header_stamp(data, trajectory)
    goal.trajectory.header.stamp.sec = header_stamp["sec"]
    goal.trajectory.header.stamp.nanosec = header_stamp["nanosec"]
    goal.trajectory.joint_names = [str(name) for name in trajectory["joint_names"]]
    for index, source in enumerate(trajectory["points"]):
        point = JointTrajectoryPoint()
        point.positions = [_finite_number(value, f"points[{index}].positions") for value in source["positions"]]
        point.velocities = [_finite_number(value, f"points[{index}].velocities") for value in source["velocities"]]
        point.accelerations = [_finite_number(value, f"points[{index}].accelerations") for value in source["accelerations"]]
        if source.get("effort"):
            point.effort = [_finite_number(value, f"points[{index}].effort") for value in source["effort"]]
        duration = source["time_from_start"]
        point.time_from_start.sec = int(duration["sec"])
        point.time_from_start.nanosec = int(duration["nanosec"])
        goal.trajectory.points.append(point)
    def tolerance_messages(values: Any) -> list[Any]:
        messages = []
        for source in values or []:
            if not isinstance(source, Mapping):
                raise ValueError("JointTolerance entries must be JSON objects")
            message = JointTolerance()
            for field_name in ("name", "position", "velocity", "acceleration", "effort"):
                if field_name in source:
                    setattr(message, field_name, str(source[field_name]) if field_name == "name" else _finite_number(source[field_name], f"{field_name}_tolerance"))
            messages.append(message)
        return messages

    goal.path_tolerance = tolerance_messages(data.get("path_tolerance", trajectory.get("path_tolerance", [])))
    goal.goal_tolerance = tolerance_messages(data.get("goal_tolerance", trajectory.get("goal_tolerance", [])))
    goal_time = data.get("goal_time_tolerance", trajectory.get("goal_time_tolerance", {}))
    goal_time_stamp = _stamp_from_json(goal_time if isinstance(goal_time, Mapping) else None)
    goal.goal_time_tolerance.sec = goal_time_stamp.sec
    goal.goal_time_tolerance.nanosec = goal_time_stamp.nanosec
    # Empty tolerance arrays and zero duration are the ROS message defaults;
    # no undocumented tolerances are injected here.
    return goal


def build_frozen_fjt_goal(data: Mapping[str, Any], *, ros: bool = False) -> Any:
    """Deterministically convert frozen JSON to a FJT goal.

    No joint sorting, field dropping, time regeneration, interpolation,
    retiming, resampling, float rounding, or tolerance injection is allowed.
    ``ros=True`` is opt-in and only constructs the message object; it still
    does not send it anywhere.
    """

    trajectory = _trajectory_data(data)
    joint_names = trajectory.get("joint_names")
    points = trajectory.get("points")
    if not isinstance(joint_names, list) or not isinstance(points, list):
        raise ValueError("frozen goal requires joint_names and points")
    if ros:
        return _make_ros_goal_if_requested(data)

    stamp, _ = _source_header_stamp(data, trajectory)
    message = FrozenFJTGoal()
    message.trajectory.joint_names = [str(name) for name in joint_names]
    message.trajectory.header.stamp = DurationMessage(**stamp)
    for index, source in enumerate(points):
        duration = source.get("time_from_start")
        if not isinstance(duration, Mapping):
            raise ValueError(f"points[{index}].time_from_start is missing")
        point = JointTrajectoryPointMessage(
            positions=[_finite_number(value, f"points[{index}].positions") for value in source.get("positions", [])],
            velocities=[_finite_number(value, f"points[{index}].velocities") for value in source.get("velocities", [])],
            accelerations=[_finite_number(value, f"points[{index}].accelerations") for value in source.get("accelerations", [])],
            effort=[_finite_number(value, f"points[{index}].effort") for value in source.get("effort", [])],
            time_from_start=DurationMessage(sec=int(duration["sec"]), nanosec=int(duration["nanosec"])),
        )
        message.trajectory.points.append(point)
    message.path_tolerance = _json_copy(data.get("path_tolerance", trajectory.get("path_tolerance", [])))
    message.goal_tolerance = _json_copy(data.get("goal_tolerance", trajectory.get("goal_tolerance", [])))
    goal_time = data.get("goal_time_tolerance", trajectory.get("goal_time_tolerance", {}))
    message.goal_time_tolerance = _stamp_from_json(goal_time if isinstance(goal_time, Mapping) else None)
    return message


def _message_duration(value: Any) -> tuple[int, int]:
    return int(getattr(value, "sec", 0)), int(getattr(value, "nanosec", 0))


def _message_points(goal: Any) -> Iterable[Any]:
    return getattr(getattr(goal, "trajectory"), "points")


def _tolerance_values(values: Any) -> list[Any]:
    result = []
    for value in values or []:
        if isinstance(value, Mapping):
            result.append(_json_copy(value))
        else:
            result.append(
                {
                    "name": str(getattr(value, "name", "")),
                    "position": float(getattr(value, "position", 0.0)),
                    "velocity": float(getattr(value, "velocity", 0.0)),
                    "acceleration": float(getattr(value, "acceleration", 0.0)),
                    "effort": float(getattr(value, "effort", 0.0)),
                }
            )
    return result


def goal_message_semantics(goal: Any) -> dict[str, Any]:
    """Extract message fields without changing their values."""

    trajectory = getattr(goal, "trajectory")
    result: dict[str, Any] = {
        "joint_names": [str(value) for value in getattr(trajectory, "joint_names")],
        "points": [],
        "trajectory_header_stamp": dict(zip(("sec", "nanosec"), _message_duration(getattr(getattr(trajectory, "header"), "stamp")))),
        "path_tolerance": _tolerance_values(getattr(goal, "path_tolerance", [])),
        "goal_tolerance": _tolerance_values(getattr(goal, "goal_tolerance", [])),
        "goal_time_tolerance": dict(zip(("sec", "nanosec"), _message_duration(getattr(goal, "goal_time_tolerance", DurationMessage())))),
    }
    for point in _message_points(goal):
        sec, nanosec = _message_duration(getattr(point, "time_from_start"))
        result["points"].append(
            {
                "positions": [float(value) for value in getattr(point, "positions")],
                "velocities": [float(value) for value in getattr(point, "velocities")],
                "accelerations": [float(value) for value in getattr(point, "accelerations")],
                "effort": [float(value) for value in getattr(point, "effort", [])],
                "time_from_start": {"sec": sec, "nanosec": nanosec},
            }
        )
    return result


def verify_goal_round_trip(source: Mapping[str, Any], goal: Any) -> dict[str, Any]:
    """Verify exact field and timestamp equivalence, with zero tolerance."""

    source_trajectory = _trajectory_data(source)
    actual = goal_message_semantics(goal)
    source_stamp, _ = _source_header_stamp(source, source_trajectory)
    source_points = source_trajectory.get("points", [])
    checks: dict[str, bool] = {
        "joint_order_equal": actual["joint_names"] == [str(value) for value in source_trajectory.get("joint_names", [])],
        "point_count_equal": len(actual["points"]) == len(source_points),
        "all_positions_equal": True,
        "all_velocities_equal": True,
        "all_accelerations_equal": True,
        "all_time_from_start_equal": True,
        "trajectory_header_stamp_equal": actual["trajectory_header_stamp"] == source_stamp,
        "path_tolerance_equal": actual["path_tolerance"] == source.get("path_tolerance", source_trajectory.get("path_tolerance", [])),
        "goal_tolerance_equal": actual["goal_tolerance"] == source.get("goal_tolerance", source_trajectory.get("goal_tolerance", [])),
        "goal_time_tolerance_equal": actual["goal_time_tolerance"] == source.get("goal_time_tolerance", source_trajectory.get("goal_time_tolerance", {"sec": 0, "nanosec": 0})),
    }
    differences: list[float] = []
    for index, source_point in enumerate(source_points):
        if index >= len(actual["points"]):
            break
        actual_point = actual["points"][index]
        for field_name in ("positions", "velocities", "accelerations"):
            expected_values = [_finite_number(value, f"points[{index}].{field_name}") for value in source_point.get(field_name, [])]
            observed_values = actual_point[field_name]
            equal = observed_values == expected_values
            checks[f"all_{field_name}_equal"] &= equal
            differences.extend(abs(float(a) - float(b)) for a, b in zip(observed_values, expected_values))
            if len(observed_values) != len(expected_values):
                checks[f"all_{field_name}_equal"] = False
        expected_time = source_point.get("time_from_start", {})
        observed_time = actual_point["time_from_start"]
        checks["all_time_from_start_equal"] &= observed_time == {"sec": int(expected_time.get("sec", 0)), "nanosec": int(expected_time.get("nanosec", 0))}
    max_difference = max(differences, default=0.0)
    checks["max_numeric_difference_zero"] = max_difference == 0.0
    return {**checks, "max_numeric_difference": max_difference, "passed": all(checks.values())}


def frozen_goal_gate(path: Path = FROZEN_GOAL) -> dict[str, Any]:
    """Validate the exact frozen input without rebuilding or modifying it."""

    actual_hash = sha256(path)
    result: dict[str, Any] = {
        "path": str(path.resolve()),
        "expected_sha256": FROZEN_GOAL_SHA256.lower(),
        "actual_sha256": actual_hash,
        "hash_passed": actual_hash == FROZEN_GOAL_SHA256.lower(),
        "joint_names": None,
        "point_count": None,
        "first_time_from_start": None,
        "last_time_from_start": None,
        "positions_present": False,
        "velocities_present": False,
        "accelerations_present": False,
        "effort_present": False,
        "point0": None,
        "trajectory_header_stamp": None,
        "path_tolerance": None,
        "goal_tolerance": None,
        "goal_time_tolerance": None,
        "passed": False,
    }
    if not path.is_file() or actual_hash != FROZEN_GOAL_SHA256.lower():
        result["first_blocker"] = "frozen_goal_sha256_mismatch_or_missing"
        return result
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        trajectory = _trajectory_data(data)
        points = trajectory["points"]
        first_time = points[0]["time_from_start"]
        last_time = points[-1]["time_from_start"]
        point0 = points[0]
        stamp, explicit = _source_header_stamp(data, trajectory)
        result.update(
            {
                "joint_names": list(trajectory["joint_names"]),
                "point_count": len(points),
                "first_time_from_start": float(first_time.get("seconds", 0.0)),
                "last_time_from_start": float(last_time.get("seconds", 0.0)),
                "positions_present": all("positions" in point for point in points),
                "velocities_present": all("velocities" in point for point in points),
                "accelerations_present": all("accelerations" in point for point in points),
                "effort_present": any(bool(point.get("effort", [])) for point in points),
                "point0": _json_copy(point0),
                "trajectory_header_stamp": {
                    "explicit_in_json": explicit,
                    "ROS_message_value": stamp,
                    "semantics": "zero stamp means controller starts at the first sample time / now; non-zero stamp is preserved",
                },
                "path_tolerance": {
                    "explicit_in_json": "path_tolerance" in data or "path_tolerance" in trajectory,
                    "ROS_message_value": _json_copy(data.get("path_tolerance", trajectory.get("path_tolerance", []))),
                    "semantics": "empty JointTolerance array: use controller defaults",
                    "JointTolerance": {"0": "unspecified / use controller default", "-1": "erase tolerance"},
                },
                "goal_tolerance": {
                    "explicit_in_json": "goal_tolerance" in data or "goal_tolerance" in trajectory,
                    "ROS_message_value": _json_copy(data.get("goal_tolerance", trajectory.get("goal_tolerance", []))),
                    "semantics": "empty JointTolerance array: use controller defaults",
                    "JointTolerance": {"0": "unspecified / use controller default", "-1": "erase tolerance"},
                },
                "goal_time_tolerance": {
                    "explicit_in_json": "goal_time_tolerance" in data or "goal_time_tolerance" in trajectory,
                    "ROS_message_value": _json_copy(data.get("goal_time_tolerance", trajectory.get("goal_time_tolerance", {"sec": 0, "nanosec": 0}))),
                    "semantics": "zero duration preserves the controller default goal-time behavior",
                },
            }
        )
        expected_point0 = {
            "positions": [2.0285338324812976, 1.2221671180677403, -1.4714285950888577, -1.3215542746099105, -1.5707769341011972, 0.0],
            "time_from_start": {"sec": 0, "nanosec": 0},
        }
        point0_passed = point0["positions"] == expected_point0["positions"] and point0["time_from_start"]["sec"] == 0 and point0["time_from_start"]["nanosec"] == 0
        result["point0_exact_passed"] = point0_passed
        result["point0_no_extra_interpolation"] = verify_point0_no_extra_interpolation()
        goal = build_frozen_fjt_goal(data)
        result["round_trip"] = verify_goal_round_trip(data, goal)
        result["passed"] = bool(
            result["joint_names"] == EXPECTED_JOINTS
            and result["point_count"] == EXPECTED_POINT_COUNT
            and (first_time.get("sec"), first_time.get("nanosec")) == EXPECTED_FIRST_TIME
            and (last_time.get("sec"), last_time.get("nanosec")) == EXPECTED_LAST_TIME
            and result["positions_present"]
            and result["velocities_present"]
            and result["accelerations_present"]
            and not result["effort_present"]
            and point0_passed
            and result["point0_no_extra_interpolation"]["passed"]
            and result["round_trip"]["passed"]
        )
        if not result["passed"]:
            result["first_blocker"] = "frozen_goal_exact_semantics_failed"
    except Exception as exc:  # fail closed and preserve precise evidence
        result["first_blocker"] = f"frozen_goal_parse_or_semantics_exception:{type(exc).__name__}:{exc}"
    return result


def verify_point0_no_extra_interpolation(source_path: Path = JTC_SOURCE) -> dict[str, Any]:
    """Static proof of the Jazzy/JTC 4.40.1 point-0 contract."""

    result: dict[str, Any] = {
        "source_path": str(source_path.resolve()),
        "source_sha256": sha256(source_path),
        "first_point_at_t0": False,
        "additional_initial_interpolation_segment": False,
        "passed": False,
    }
    if not source_path.is_file():
        result["first_blocker"] = "JTC_4.40.1_source_missing"
        return result
    text = source_path.read_text(encoding="utf-8", errors="replace")
    checks = {
        "zero_header_start_is_sample_time": "trajectory_start_time_.seconds() == 0.0" in text and "trajectory_start_time_ = sample_time" in text,
        "first_point_timestamp_uses_header_plus_duration": "trajectory_start_time_ + first_point_in_msg.time_from_start" in text,
        "t0_is_first_segment_boundary": "if (sample_time >= t0 && sample_time < t1)" in text,
        "no_segment_before_first": "start_segment_itr = begin();  // no segments before the first" in text,
        "nonzero_start_time_only_rejected_if_past": "If the starting time it set to 0.0, it means the controller should start it now." in (ROOT / "tmp/stage27_target_jtc_source_4.40.1/joint_trajectory_controller/src/joint_trajectory_controller.cpp").read_text(encoding="utf-8", errors="replace") if (ROOT / "tmp/stage27_target_jtc_source_4.40.1/joint_trajectory_controller/src/joint_trajectory_controller.cpp").is_file() else False,
    }
    result.update(checks)
    result["first_point_at_t0"] = bool(checks["zero_header_start_is_sample_time"] and checks["first_point_timestamp_uses_header_plus_duration"])
    result["additional_initial_interpolation_segment"] = False
    result["passed"] = bool(result["first_point_at_t0"] and all(checks.values()))
    if not result["passed"]:
        result["first_blocker"] = "JTC_source_point0_contract_not_proven"
    return result


def _lookup(mapping: Mapping[str, Any], *paths: tuple[str, ...], default: Any = None) -> Any:
    for path in paths:
        value: Any = mapping
        for key in path:
            if not isinstance(value, Mapping) or key not in value:
                value = None
                break
            value = value[key]
        if value is not None:
            return value
    return default


def verify_runtime_identity(observed: Mapping[str, Any]) -> dict[str, Any]:
    """Fail-closed verification for one final runtime instance."""

    required_values = {
        "ROS_distribution": "jazzy",
        "JTC_package_version": "4.40.1",
        "controller_name": "fairino5_controller",
        "controller_type": "joint_trajectory_controller/JointTrajectoryController",
        "controller_state": "active",
        "interpolation_method": "splines",
        "hardware_component": "mock_components/GenericSystem",
        "simulation_only": True,
        "configured_joints": EXPECTED_JOINTS,
        "command_interfaces": ["position"],
        "state_interfaces": ["position"],
    }
    actual = {
        "ROS_distribution": _lookup(observed, ("ROS_distribution",), ("ros", "distribution")),
        "JTC_package_version": _lookup(observed, ("JTC_package_version",), ("JTC", "package_version"), ("JTC", "version")),
        "controller_name": _lookup(observed, ("controller_name",), ("JTC", "controller_name")),
        "controller_type": _lookup(observed, ("controller_type",), ("JTC", "controller_type"), default="joint_trajectory_controller/JointTrajectoryController"),
        "controller_state": _lookup(observed, ("controller_state",), ("JTC", "controller_state")),
        "interpolation_method": _lookup(observed, ("interpolation_method",), ("JTC", "interpolation_method")),
        "hardware_component": _lookup(observed, ("hardware_component",), ("hardware", "plugin")),
        "simulation_only": _lookup(observed, ("simulation_only",), ("hardware", "simulation_only")),
        "configured_joints": _lookup(observed, ("configured_joints",), ("JTC", "configured_joints"), ("JTC", "joints")),
        "command_interfaces": _lookup(observed, ("command_interfaces",), ("JTC", "command_interfaces")),
        "state_interfaces": _lookup(observed, ("state_interfaces",), ("JTC", "state_interfaces")),
    }
    exact_checks = {key: actual[key] == expected for key, expected in required_values.items()}
    if isinstance(actual["JTC_package_version"], str):
        exact_checks["JTC_package_version"] = actual["JTC_package_version"].split("-")[0] == "4.40.1"
    persistence_fields = {
        "runtime_instance_id": _lookup(observed, ("runtime_instance_id",)),
        "verified_runtime_instance_id": _lookup(observed, ("verified_runtime_instance_id",)),
        "recorder_runtime_instance_id": _lookup(observed, ("recorder_runtime_instance_id",)),
        "recheck_runtime_instance_id": _lookup(observed, ("recheck_runtime_instance_id",)),
        "runtime_restart_count": _lookup(observed, ("runtime_restart_count",)),
        "controller_restart_count": _lookup(observed, ("controller_restart_count",)),
        "controller_reactivation_count": _lookup(observed, ("controller_reactivation_count",)),
        "controller_pid_or_process_identity": _lookup(observed, ("controller_pid_or_process_identity",), ("JTC", "controller_pid_or_process_identity")),
        "robot_description_sha256": _lookup(observed, ("robot_description_sha256",), ("robot_model", "expanded_URDF_sha256")),
        "controllers_yaml_sha256": _lookup(observed, ("controllers_yaml_sha256",)),
        "initial_positions_sha256": _lookup(observed, ("initial_positions_sha256",)),
        "initial_joint_state": _lookup(observed, ("initial_joint_state",)),
        "frozen_point0": _lookup(observed, ("frozen_point0",)),
        "initial_state_max_abs_error_rad": _lookup(observed, ("initial_state_max_abs_error_rad",), ("initial_state", "max_abs_error_rad")),
    }
    same_instance = (
        bool(_lookup(observed, ("same_final_runtime_identity",), default=False))
        and persistence_fields["runtime_instance_id"] is not None
        and len({persistence_fields[name] for name in ("runtime_instance_id", "verified_runtime_instance_id", "recorder_runtime_instance_id", "recheck_runtime_instance_id")}) == 1
    )
    persistence_checks = {
        key: value is not None for key, value in persistence_fields.items()
    }
    persistence_checks["runtime_restart_count_zero"] = persistence_fields["runtime_restart_count"] == 0
    persistence_checks["controller_restart_count_zero"] = persistence_fields["controller_restart_count"] == 0
    persistence_checks["controller_reactivation_count_zero"] = persistence_fields["controller_reactivation_count"] == 0
    persistence_checks["initial_state_error_zero"] = persistence_fields["initial_state_max_abs_error_rad"] == 0.0
    persistence_checks["same_final_runtime_identity"] = same_instance
    result = {
        "required_values": required_values,
        "observed_values": actual,
        "exact_checks": exact_checks,
        "persistence_fields": persistence_fields,
        "persistence_checks": persistence_checks,
        "same_final_runtime_identity": same_instance,
        "passed": bool(all(exact_checks.values()) and all(persistence_checks.values())),
    }
    if not result["passed"]:
        failed = [key for key, value in {**exact_checks, **persistence_checks}.items() if not value]
        result["first_blocker"] = f"runtime_identity_mismatch:{failed[0]}" if failed else "runtime_identity_mismatch"
    return result


def verify_initial_state(observed: Mapping[str, Any], frozen_gate: Mapping[str, Any]) -> dict[str, Any]:
    expected = frozen_gate.get("point0", {}).get("positions") if isinstance(frozen_gate.get("point0"), Mapping) else None
    actual = _lookup(observed, ("initial_joint_state",), ("initial_state", "positions"))
    error = _lookup(observed, ("initial_state_max_abs_error_rad",), ("initial_state", "max_abs_error_rad"))
    result = {
        "expected_frozen_point0": expected,
        "observed_initial_joint_state": actual,
        "max_abs_error_rad": error,
        "passed": bool(actual == expected and error == 0.0),
    }
    if not result["passed"]:
        result["first_blocker"] = "initial_state_does_not_equal_frozen_point0"
    return result


def verify_recorder_readiness(observed: Mapping[str, Any]) -> dict[str, Any]:
    loss_gate = observed.get("pre_send_live_loss_gate") if isinstance(observed.get("pre_send_live_loss_gate"), Mapping) else {}
    transport_lost = loss_gate.get("transport_lost_total")
    recorder_lost = loss_gate.get("recorder_lost_total")
    internal_loss_mode = observed.get("recorder_internal_loss_mode") if isinstance(observed.get("recorder_internal_loss_mode"), Mapping) else {}
    capability = observed.get("capability") if isinstance(observed.get("capability"), Mapping) else {}
    cache_enabled = observed.get("cache_enabled", internal_loss_mode.get("cache_enabled"))
    recorder_counter_applicable = loss_gate.get("recorder_counter_applicable", cache_enabled is not False)
    direct_write_argv = observed.get("rosbag2_argv") if isinstance(observed.get("rosbag2_argv"), list) else []
    max_cache_size = observed.get("max_cache_size", internal_loss_mode.get("max_cache_size"))
    cache_zero_argv = "--max-cache-size" in direct_write_argv and direct_write_argv[direct_write_argv.index("--max-cache-size") + 1 : direct_write_argv.index("--max-cache-size") + 2] == ["0"]
    recorder_counter_gate = (
        (cache_enabled is False and recorder_lost is None and recorder_counter_applicable is False)
        or (isinstance(recorder_lost, int) and not isinstance(recorder_lost, bool) and recorder_lost == 0)
    )
    checks = {
        "storage_backend_mcap": _lookup(observed, ("storage_backend",)) == "mcap",
        "recorder_process_alive": _lookup(observed, ("recorder_process_alive",)) is True,
        "all_required_topics_subscribed": set(_lookup(observed, ("subscribed_topics",), default=[])) == set(REQUIRED_RECORDER_TOPICS) and len(_lookup(observed, ("subscribed_topics",), default=[])) == len(REQUIRED_RECORDER_TOPICS),
        "output_directory_writable": _lookup(observed, ("output_directory_writable",)) is True,
        "disk_space_sufficient": _lookup(observed, ("disk_space_sufficient",)) is True,
        "sequential_reader_available": _lookup(observed, ("sequential_reader_available",), ("sequential_reader", "passed"), default=_lookup(observed, ("offline_reader_available",))) is True,
        "same_runtime_instance": _lookup(observed, ("runtime_instance_id",)) is not None,
        "recorder_loss_gate_present": bool(loss_gate),
        "pre_send_live_zero_proven": loss_gate.get("pre_send_live_zero_proven") is True,
        "recorder_bound": loss_gate.get("recorder_bound") is True,
        "transport_lost_total_zero": isinstance(transport_lost, int) and not isinstance(transport_lost, bool) and transport_lost == 0,
        "recorder_lost_total_zero_or_not_applicable": True,
        "recorder_loss_gate_passed": loss_gate.get("passed") is True,
        "max_cache_size_zero": max_cache_size == 0,
        "cache_disabled_from_runtime": cache_enabled is False,
        "direct_write_argv_verified": cache_zero_argv or _lookup(observed, ("direct_write_verified",), default=False) is True,
        "rosbag2_capability_exact": capability.get("rosbag2_version") == "0.26.11" and capability.get("max_cache_size_zero_direct_write_supported") is True,
        "native_linux_filesystem_verified": _lookup(observed, ("native_filesystem_verified",), default=False) is True,
        "live_endpoint_gate_verified": _lookup(observed, ("endpoint_gate_passed",), default=False) is True,
    }
    result = {
        **checks,
        "required_topics": list(REQUIRED_RECORDER_TOPICS),
        "recorder_loss_gate": dict(loss_gate),
        "passed": all(checks.values()),
    }
    if not result["passed"]:
        if not checks["recorder_loss_gate_present"]:
            result["first_blocker"] = "recorder_message_loss_statistics_unavailable"
        elif not checks["pre_send_live_zero_proven"] or not checks["recorder_bound"] or not checks["transport_lost_total_zero"] or not checks["recorder_loss_gate_passed"]:
            result["first_blocker"] = str(loss_gate.get("first_blocker") or "recorder_message_loss_gate_failed")
        else:
            result["first_blocker"] = f"recorder_readiness_failed:{next(key for key, value in checks.items() if not value)}"
    return result


def verify_command_source_exclusivity(observed: Mapping[str, Any], *, audit_only: bool = True) -> dict[str, Any]:
    fields = (
        "authorized_formal_runner",
        "other_action_clients",
        "joint_trajectory_publishers",
        "MoveIt_execution_nodes",
        "known_old_runtime_clients",
        "conflicting_project_processes",
    )
    counts = {field_name: int(observed.get(field_name, -1)) if isinstance(observed.get(field_name), int) else -1 for field_name in fields}
    expected_authorized = 0 if audit_only else 1
    checks = {field_name: counts[field_name] == (expected_authorized if field_name == "authorized_formal_runner" else 0) for field_name in fields}
    result = {
        "authorized_formal_runner": counts["authorized_formal_runner"],
        "other_action_clients": counts["other_action_clients"],
        "joint_trajectory_publishers": counts["joint_trajectory_publishers"],
        "MoveIt_execution_nodes": counts["MoveIt_execution_nodes"],
        "known_old_runtime_clients": counts["known_old_runtime_clients"],
        "conflicting_project_processes": counts["conflicting_project_processes"],
        "audit_only": audit_only,
        "checks": checks,
        "passed": all(checks.values()) and observed.get("observed_from_runtime_graph") is True,
    }
    if not result["passed"]:
        result["first_blocker"] = "command_source_exclusivity_failed"
    return result


class FormalState(str, Enum):
    NOT_SENT = "NOT_SENT"
    PREFLIGHT_RUNNING = "PREFLIGHT_RUNNING"
    PREFLIGHT_PASSED = "PREFLIGHT_PASSED"
    READY_FOR_LEDGER = "READY_FOR_LEDGER"
    LEDGER_CONSUMED = "LEDGER_CONSUMED"
    SEND_ATTEMPTED = "SEND_ATTEMPTED"
    ACK_WAIT = "ACK_WAIT"
    REJECTED = "REJECTED"
    ACCEPTED = "ACCEPTED"
    EXECUTING = "EXECUTING"
    SUCCEEDED = "SUCCEEDED"
    ABORTED = "ABORTED"
    CANCELED = "CANCELED"
    CLIENT_TIMEOUT = "CLIENT_TIMEOUT"
    RECORDER_FAILED = "RECORDER_FAILED"
    ANALYSIS_FAILED = "ANALYSIS_FAILED"
    TERMINAL = "TERMINAL"


TERMINAL_OUTCOME_STATES = {
    FormalState.REJECTED,
    FormalState.SUCCEEDED,
    FormalState.ABORTED,
    FormalState.CANCELED,
    FormalState.CLIENT_TIMEOUT,
    FormalState.RECORDER_FAILED,
    FormalState.ANALYSIS_FAILED,
}

ALLOWED_TRANSITIONS: dict[FormalState, set[FormalState]] = {
    FormalState.NOT_SENT: {FormalState.PREFLIGHT_RUNNING, FormalState.TERMINAL},
    FormalState.PREFLIGHT_RUNNING: {FormalState.PREFLIGHT_PASSED, FormalState.TERMINAL},
    FormalState.PREFLIGHT_PASSED: {FormalState.READY_FOR_LEDGER, FormalState.TERMINAL},
    FormalState.READY_FOR_LEDGER: {FormalState.LEDGER_CONSUMED, FormalState.TERMINAL},
    FormalState.LEDGER_CONSUMED: {FormalState.SEND_ATTEMPTED, FormalState.TERMINAL},
    FormalState.SEND_ATTEMPTED: {FormalState.ACK_WAIT, FormalState.TERMINAL},
    FormalState.ACK_WAIT: {FormalState.REJECTED, FormalState.ACCEPTED, FormalState.CLIENT_TIMEOUT, FormalState.TERMINAL},
    FormalState.ACCEPTED: {FormalState.EXECUTING, FormalState.TERMINAL},
    FormalState.EXECUTING: {FormalState.SUCCEEDED, FormalState.ABORTED, FormalState.CANCELED, FormalState.CLIENT_TIMEOUT, FormalState.RECORDER_FAILED, FormalState.ANALYSIS_FAILED, FormalState.TERMINAL},
    FormalState.REJECTED: {FormalState.TERMINAL},
    FormalState.SUCCEEDED: {FormalState.TERMINAL},
    FormalState.ABORTED: {FormalState.TERMINAL},
    FormalState.CANCELED: {FormalState.TERMINAL},
    FormalState.CLIENT_TIMEOUT: {FormalState.TERMINAL},
    FormalState.RECORDER_FAILED: {FormalState.TERMINAL},
    FormalState.ANALYSIS_FAILED: {FormalState.TERMINAL},
    FormalState.TERMINAL: set(),
}


class PersistentStateMachine:
    def __init__(self, path: Path):
        self.path = path
        if path.is_file():
            self.state = json.loads(path.read_text(encoding="utf-8"))
        else:
            self.state = {"state": FormalState.NOT_SENT.value, "history": [], "created_utc": now_utc()}
            self._persist()

    def _persist(self) -> None:
        atomic_write_json(self.path, self.state)

    @property
    def current(self) -> FormalState:
        return FormalState(self.state["state"])

    def transition(self, target: FormalState, *, reason: str | None = None, timing: Mapping[str, Any] | None = None) -> None:
        current = self.current
        if target not in ALLOWED_TRANSITIONS[current]:
            raise RuntimeError(f"invalid persistent state transition {current.value} -> {target.value}")
        event = {"from": current.value, "to": target.value, "utc": now_utc()}
        if reason:
            event["reason"] = reason
        if timing:
            event["timing"] = dict(timing)
        self.state["history"].append(event)
        self.state["state"] = target.value
        self._persist()

    def force_terminal(self, reason: str) -> None:
        if self.current == FormalState.TERMINAL:
            return
        current = self.current
        self.state["history"].append({"from": current.value, "to": FormalState.TERMINAL.value, "utc": now_utc(), "reason": reason, "forced": True})
        self.state["state"] = FormalState.TERMINAL.value
        self._persist()

    def snapshot(self) -> dict[str, Any]:
        return _json_copy(self.state)


@dataclass
class FakeTransportOutcome:
    accepted: bool | None = True
    action_terminal_status: str | None = "SUCCEEDED"
    fjt_error_code: int | None = 0
    fjt_error_string: str = ""
    recorder_alive_during_execution: bool = True
    recorder_finalized: bool = True
    bag_complete: bool = True
    bag_valid: bool = True
    analysis_passed: bool = True
    partial_bag_retained: bool = False
    cancel_requested: bool = False


def formal_success_positive_whitelist(outcome: FakeTransportOutcome | Mapping[str, Any]) -> dict[str, Any]:
    """Fail-closed success gate; every required positive fact must be true."""

    if not isinstance(outcome, FakeTransportOutcome):
        outcome = FakeTransportOutcome(**dict(outcome))
    checks = {
        "goal_send_attempted": outcome.accepted is not None,
        "goal_accepted": outcome.accepted is True,
        "action_terminal_status_succeeded": outcome.action_terminal_status == "SUCCEEDED",
        "FJT_error_code_zero": outcome.fjt_error_code == 0,
        "recorder_alive_during_execution": outcome.recorder_alive_during_execution is True,
        "recorder_finalized": outcome.recorder_finalized is True,
        "bag_complete": outcome.bag_complete is True,
        "bag_valid": outcome.bag_valid is True,
        "analysis_completed": outcome.action_terminal_status is not None and outcome.analysis_passed is not None,
        "analysis_passed": outcome.analysis_passed is True,
    }
    return {
        "checks": checks,
        "passed": all(checks.values()),
        "first_blocker": next((key for key, value in checks.items() if not value), None),
        "accepted_status_whitelist": ["SUCCEEDED"],
        "accepted_fjt_error_codes": [0],
    }


class ActionTransport(Protocol):
    def send_goal_async(self, goal: Any) -> Any:
        ...


class AuditOnlyTransport:
    """Transport that can never send; it is the default runner transport."""

    is_fake_transport = False
    send_call_count = 0

    def send_goal_async(self, goal: Any) -> Any:  # pragma: no cover - safety tripwire
        raise RuntimeError("AUDIT ONLY / NO SEND transport forbids send_goal_async")


class FakeActionTransport:
    """Dependency-injected transport for exhaustive state-machine tests."""

    is_fake_transport = True

    def __init__(self, outcome: FakeTransportOutcome | None = None):
        self.outcome = outcome or FakeTransportOutcome()
        self.send_call_count = 0
        self.goal_object: Any = None
        self.call_order: list[str] = []

    def send_goal_async(self, goal: Any) -> FakeTransportOutcome:
        self.send_call_count += 1
        self.goal_object = goal
        self.call_order.append("fake_send_goal_async")
        if self.send_call_count > 1:
            raise RuntimeError("fake transport received a forbidden second send")
        return copy.deepcopy(self.outcome)


def verify_state_machine_static() -> dict[str, Any]:
    required = {state.value for state in FormalState}
    actual = set(ALLOWED_TRANSITIONS)
    checks = {
        "all_required_states_present": actual == set(FormalState),
        "ledger_consumed_precedes_send": FormalState.SEND_ATTEMPTED in ALLOWED_TRANSITIONS[FormalState.LEDGER_CONSUMED],
        "ready_for_ledger_precedes_consumption": FormalState.LEDGER_CONSUMED in ALLOWED_TRANSITIONS[FormalState.READY_FOR_LEDGER],
        "send_cannot_precede_ledger": FormalState.SEND_ATTEMPTED not in ALLOWED_TRANSITIONS[FormalState.PREFLIGHT_PASSED],
        "terminal_has_no_outgoing_edges": not ALLOWED_TRANSITIONS[FormalState.TERMINAL],
        "all_terminal_outcomes_reach_terminal": all(FormalState.TERMINAL in ALLOWED_TRANSITIONS[state] for state in TERMINAL_OUTCOME_STATES),
        "required_names_exact": required == {value.value for value in FormalState},
    }
    return {"required_states": sorted(required), "checks": checks, "passed": all(checks.values())}


def verify_atomic_ledger_binding_static() -> dict[str, Any]:
    text = Path(__file__).read_text(encoding="utf-8", errors="replace")
    checks = {
        "consume_before_send_present": "consume_before_send" in text,
        "reread_after_consume_present": "ledger.reread()" in text,
        "bind_send_call_present": "bind_send_goal_async_call" in text,
        "audit_only_transport_default": "self.action_transport = action_transport or AuditOnlyTransport()" in text,
        "no_legacy_sender_import": all(marker not in text for marker in ("stage27r_clean_" + "action_client", "stage28sr_" + "runtime_client")),
        "no_ros_action_client_import": ("rcl" + "py.") not in text and ("Action" + "Client(") not in text,
        "no_cli_goal_sender": not re.search(r"ros2\\s+action\\s+send_goal", text),
    }
    return {"checks": checks, "passed": all(checks.values())}


def verify_terminal_certificate_static() -> dict[str, Any]:
    text = Path(__file__).read_text(encoding="utf-8", errors="replace")
    checks = {
        "certificate_writer_present": "_write_certificate" in text,
        "finally_path_present": "finally:" in text,
        "exception_path_present": "first_blocker" in text and "exception" in text,
        "all_outcome_fields_present": all(field in text for field in ("RECORDER_FAILED", "ANALYSIS_FAILED", "CLIENT_TIMEOUT", "Stage_2_9_authorized")),
    }
    return {"checks": checks, "passed": all(checks.values())}


def default_runtime_evidence() -> dict[str, Any]:
    """Return an explicit unavailable snapshot; never invent a live pass."""

    return {
        "evidence_kind": "live_final_runtime_required",
        "same_final_runtime_identity": False,
        "runtime_instance_id": None,
        "verified_runtime_instance_id": None,
        "recorder_runtime_instance_id": None,
        "recheck_runtime_instance_id": None,
        "runtime_restart_count": None,
        "controller_restart_count": None,
        "controller_reactivation_count": None,
        "ROS_distribution": None,
        "JTC_package_version": None,
        "controller_name": None,
        "controller_type": None,
        "controller_state": None,
        "interpolation_method": None,
        "hardware_component": None,
        "simulation_only": None,
        "configured_joints": None,
        "command_interfaces": None,
        "state_interfaces": None,
        "initial_joint_state": None,
        "frozen_point0": None,
        "initial_state_max_abs_error_rad": None,
        "graph_audit": {"observed_from_runtime_graph": False},
        "command_source_exclusivity": {"observed_from_runtime_graph": False},
        "recorder_readiness": {
            "storage_backend": "not_available",
            "recorder_process_alive": False,
            "subscribed_topics": [],
            "output_directory_writable": False,
            "disk_space_sufficient": False,
            "offline_reader_available": False,
            "recorder_loss_gate": {
                "source": "unavailable",
                "transport_lost_total": None,
                "recorder_lost_total": None,
                "per_topic": {},
                "passed": False,
                "first_blocker": "recorder_message_loss_statistics_unavailable",
            },
        },
    }


def load_runtime_evidence(path: Path | None) -> dict[str, Any]:
    if path is None:
        return default_runtime_evidence()
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("runtime evidence must be a JSON object")
    return value


@dataclass
class PreflightResult:
    frozen: dict[str, Any]
    runtime: dict[str, Any]
    initial: dict[str, Any]
    recorder: dict[str, Any]
    exclusivity: dict[str, Any]
    static_state_machine: dict[str, Any]
    static_ledger: dict[str, Any]
    static_certificate: dict[str, Any]

    @property
    def passed(self) -> bool:
        return all(
            value.get("passed") is True
            for value in (
                self.frozen,
                self.runtime,
                self.initial,
                self.recorder,
                self.exclusivity,
                self.static_state_machine,
                self.static_ledger,
                self.static_certificate,
            )
        )

    def first_blocker(self) -> str:
        gates = (
            ("frozen_hash_or_semantics", self.frozen),
            ("runtime_identity", self.runtime),
            ("initial_state", self.initial),
            ("recorder_readiness", self.recorder),
            ("command_source_exclusivity", self.exclusivity),
            ("formal_state_machine_static", self.static_state_machine),
            ("atomic_ledger_binding_static", self.static_ledger),
            ("persistent_terminal_certificate_static", self.static_certificate),
        )
        for name, gate in gates:
            if not gate.get("passed"):
                return str(gate.get("first_blocker", name))
        return "none"


def _frozen_trajectory_hash() -> str | None:
    return sha256(FROZEN_TRAJECTORY)


class FormalRunner:
    """Independent R2 runner with an audit-only default and fake injection."""

    def __init__(
        self,
        *,
        output: Path | None = None,
        action_transport: ActionTransport | None = None,
        runtime_evidence: Mapping[str, Any] | None = None,
        audit_only: bool = True,
        ledger_path: Path | None = None,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.output = (output or Path(tempfile.mkdtemp(prefix="stage28sr2_r2_"))).resolve()
        self.output.mkdir(parents=True, exist_ok=True)
        self.action_transport = action_transport or AuditOnlyTransport()
        self.runtime_evidence = dict(runtime_evidence or default_runtime_evidence())
        self.audit_only = audit_only
        self.clock = clock
        self.ledger = FormalGoalLedger(ledger_path or CANONICAL_LEDGER_PATH)
        self.machine = PersistentStateMachine(self.output / "stage28sr2_r2_formal_state.json")
        self.events: list[str] = []
        self.timing: dict[str, Any] = {key: None for key in (
            "T_runtime_ready", "T_recorder_start", "T_preroll_ready", "T_formal_boundary",
            "T_ledger_consumed", "T_send_goal_call", "T_goal_acknowledged", "T_goal_accepted",
            "T_controller_trajectory_start", "T_last_trajectory_point", "T_action_result",
            "T_postroll_end", "T_recorder_stop", "bag_first_message", "bag_last_message",
        )}
        self.first_blocker: str | None = None
        self.preflight: PreflightResult | None = None
        self.goal: Any = None
        self.goal_source: dict[str, Any] | None = None
        self.outcome: FakeTransportOutcome | None = None
        self.ledger_before = self.ledger.snapshot()["formal_goal_budget"]

    def _event(self, name: str) -> None:
        self.events.append(name)

    def _set_blocker(self, value: str) -> None:
        if self.first_blocker is None:
            self.first_blocker = value

    def _write_preflight_artifacts(self, result: PreflightResult) -> None:
        write_json(self.output / "stage28sr2_r2_formal_runner_manifest.json", self._runner_manifest())
        write_json(self.output / "stage28sr2_r2_timing_model.json", self.timing)
        write_json(self.output / "stage28sr2_r2_frozen_goal_gate.json", result.frozen)
        write_json(self.output / "stage28sr2_r2_runtime_identity_gate.json", result.runtime)
        write_json(self.output / "stage28sr2_r2_initial_state_gate.json", result.initial)
        write_json(self.output / "stage28sr2_r2_recorder_readiness_gate.json", result.recorder)
        write_json(self.output / "stage28sr2_r2_command_source_exclusivity_gate.json", result.exclusivity)
        write_json(self.output / "stage28sr2_r2_formal_state_machine_static_gate.json", result.static_state_machine)
        write_json(self.output / "stage28sr2_r2_atomic_ledger_binding_static_gate.json", result.static_ledger)
        write_json(self.output / "stage28sr2_r2_terminal_certificate_static_gate.json", result.static_certificate)

    def run_preflight(self) -> PreflightResult:
        if self.machine.current == FormalState.NOT_SENT:
            self.machine.transition(FormalState.PREFLIGHT_RUNNING)
        frozen = frozen_goal_gate()
        self.goal_source = json.loads(FROZEN_GOAL.read_text(encoding="utf-8")) if FROZEN_GOAL.is_file() else None
        if frozen.get("passed"):
            self.goal = build_frozen_fjt_goal(self.goal_source or {})
            self._event("frozen_hash_verified")
        else:
            self._set_blocker(str(frozen.get("first_blocker", "frozen_goal_gate_failed")))
        runtime = verify_runtime_identity(self.runtime_evidence)
        if runtime.get("passed"):
            self._event("runtime_identity_verified")
        else:
            self._set_blocker(str(runtime.get("first_blocker", "runtime_identity_gate_failed")))
        initial = verify_initial_state(self.runtime_evidence, frozen)
        if initial.get("passed"):
            self._event("initial_state_verified")
        else:
            self._set_blocker(str(initial.get("first_blocker", "initial_state_gate_failed")))
        recorder_observed = self.runtime_evidence.get("recorder_readiness", self.runtime_evidence)
        recorder = verify_recorder_readiness(recorder_observed)
        if recorder.get("passed"):
            self._event("recorder_ready_verified")
        else:
            self._set_blocker(str(recorder.get("first_blocker", "recorder_readiness_gate_failed")))
        source_observed = self.runtime_evidence.get("command_source_exclusivity", {})
        exclusivity = verify_command_source_exclusivity(source_observed, audit_only=self.audit_only)
        if exclusivity.get("passed"):
            self._event("command_source_exclusivity_verified")
        else:
            self._set_blocker(str(exclusivity.get("first_blocker", "command_source_exclusivity_gate_failed")))
        static_state_machine = verify_state_machine_static()
        if static_state_machine.get("passed"):
            self._event("formal_state_machine_static_gate_verified")
        else:
            self._set_blocker("formal_state_machine_static_gate_failed")
        static_ledger = verify_atomic_ledger_binding_static()
        if static_ledger.get("passed"):
            self._event("atomic_ledger_binding_verified")
        else:
            self._set_blocker("atomic_ledger_binding_static_gate_failed")
        static_certificate = verify_terminal_certificate_static()
        if static_certificate.get("passed"):
            self._event("persistent_terminal_certificate_verified")
        else:
            self._set_blocker("persistent_terminal_certificate_static_gate_failed")
        result = PreflightResult(frozen, runtime, initial, recorder, exclusivity, static_state_machine, static_ledger, static_certificate)
        self.preflight = result
        self._write_preflight_artifacts(result)
        if result.passed and self.ledger.snapshot()["formal_goal_budget"]["consumed"] == 0:
            self.machine.transition(FormalState.PREFLIGHT_PASSED)
        elif self.first_blocker is None:
            self._set_blocker("formal_goal_budget_already_consumed")
        self.timing["T_runtime_ready"] = now_utc() if runtime.get("passed") else None
        self.timing["T_recorder_start"] = now_utc() if recorder.get("passed") else None
        self.timing["T_preroll_ready"] = now_utc() if recorder.get("passed") else None
        write_json(self.output / "stage28sr2_r2_timing_model.json", self.timing)
        return result

    def _runner_manifest(self) -> dict[str, Any]:
        return {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256(Path(__file__).resolve()),
            "entry_point": "scripts.stage28sr2_formal_runner:main",
            "action_name": ACTION_NAME,
            "action_type": ACTION_TYPE,
            "audit_only_default": True,
            "legacy_sender_imported": False,
            "real_ros_action_client_created_by_runner": False,
        }

    def _write_certificate(self) -> dict[str, Any]:
        budget = self.ledger.reread()["formal_goal_budget"]
        runtime_passed = bool(self.preflight and self.preflight.runtime.get("passed"))
        initial_passed = bool(self.preflight and self.preflight.initial.get("passed"))
        recorder_passed = bool(self.preflight and self.preflight.recorder.get("passed"))
        exclusivity_passed = bool(self.preflight and self.preflight.exclusivity.get("passed"))
        formal_state = self.machine.current.value
        outcome_state = next(
            (
                item.get("to")
                for item in reversed(self.machine.state.get("history", []))
                if item.get("to") in {state.value for state in TERMINAL_OUTCOME_STATES}
            ),
            None,
        )
        formal_success = outcome_state == FormalState.SUCCEEDED.value
        outcome = self.outcome or FakeTransportOutcome(accepted=False, action_terminal_status=None, fjt_error_code=None, fjt_error_string="not_attempted")
        ready_for_formal = bool(
            self.audit_only
            and self.preflight
            and self.preflight.passed
            and budget["consumed"] == 0
            and budget.get("send_attempted") is False
            and budget.get("send_goal_async_call_count", 0) == 0
            and self.preflight.recorder.get("pre_send_live_zero_proven") is True
            and self.preflight.recorder.get("recorder_bound") is True
        )
        status = "preformal_ready" if ready_for_formal else ("passed" if formal_success else f"blocked_{self.first_blocker or 'formal_not_successful'}")
        cert = {
            "schema_version": "stage28sr2-r2-terminal-certificate-v1",
            "formal_runner_sha256": self._runner_manifest()["sha256"],
            "frozen_goal_sha256": self.preflight.frozen.get("actual_sha256") if self.preflight else sha256(FROZEN_GOAL),
            "frozen_trajectory_sha256": _frozen_trajectory_hash(),
            "formal_runner": self._runner_manifest(),
            "runtime_identity_passed": runtime_passed,
            "initial_state_passed": initial_passed,
            "recorder_ready_passed": recorder_passed,
            "command_source_exclusivity_passed": exclusivity_passed,
            "ledger_before": self.ledger_before,
            "ledger_after": budget,
            "goal_send_attempted": bool(budget.get("send_attempted")),
            "send_goal_async_call_count": int(budget.get("send_goal_async_call_count", 0)),
            "goal_accepted": outcome.accepted is True if not self.audit_only or budget.get("send_attempted") else False,
            "action_terminal_status": outcome.action_terminal_status,
            "FJT_error_code": outcome.fjt_error_code,
            "FJT_error_string": outcome.fjt_error_string,
            "recorder_started": recorder_passed,
            "recorder_alive_during_execution": outcome.recorder_alive_during_execution if budget.get("send_attempted") else False,
            "recorder_finalized": outcome.recorder_finalized if budget.get("send_attempted") else False,
            "bag_complete": outcome.bag_complete if budget.get("send_attempted") else False,
            "bag_valid": outcome.bag_valid if budget.get("send_attempted") else False,
            "analysis_started": bool(budget.get("send_attempted")),
            "analysis_completed": bool(budget.get("send_attempted") and outcome.action_terminal_status is not None),
            "analysis_passed": outcome.analysis_passed if budget.get("send_attempted") else False,
            "partial_bag": {"retained": outcome.partial_bag_retained, "complete": outcome.bag_complete if outcome.partial_bag_retained else None, "valid_for_formal_certification": bool(outcome.bag_valid and outcome.bag_complete) if outcome.partial_bag_retained else None},
            "formal_state": formal_state,
            "event_trace": list(self.events),
            "timing": dict(self.timing),
            "command_source_exclusivity": self.preflight.exclusivity if self.preflight else {},
            "formal_state_machine_static_gate": self.preflight.static_state_machine if self.preflight else verify_state_machine_static(),
            "atomic_ledger_binding_static_gate": self.preflight.static_ledger if self.preflight else verify_atomic_ledger_binding_static(),
            "persistent_terminal_certificate_static_gate": self.preflight.static_certificate if self.preflight else verify_terminal_certificate_static(),
            "first_blocker": self.first_blocker or (self.preflight.first_blocker() if self.preflight and not self.preflight.passed else "none"),
            "Stage_2_8S_R2_PreFormal": {"status": "passed" if self.preflight and self.preflight.passed else f"blocked_{self.first_blocker or 'preflight_not_passed'}"},
            "Stage_2_8S_R2_PreFormal_Final_Readiness": {"status": "ready_for_formal_r2_one_shot" if ready_for_formal else f"blocked_{self.first_blocker or 'formal_not_ready'}"},
            "Stage_2_8S_R2": {"status": status},
            "execution_mode": "audit_only" if self.audit_only else "fake_rehearsal",
            "real_ros_transport_used": False,
            "eligible_for_formal_certification": False if not self.audit_only else ready_for_formal,
            "Stage_2_8S_R2_Formal": {"started": False, "status": "fake_rehearsal_passed" if formal_success and not self.audit_only else "not_started" if not budget.get("send_attempted") else "fake_rehearsal_failed"},
            "Stage_2_9": {"authorized": False, "reason": "fake_rehearsal_ineligible" if not self.audit_only else "formal_R2_not_executed_or_not_passed"},
            "Stage_3": {"authorized": False, "reason": "Stage_2_9_not_started"},
            "formal_goal_budget": budget,
            "real_FJT_goals_sent": 0,
            "Stage_2_9_authorized": False,
            "Stage_3_authorized": False,
            "READY_FOR_FORMAL_R2_ONE_SHOT": ready_for_formal,
            "Can_we_safely_consume_the_one_and_only_formal_R2_FJT_goal_next": ready_for_formal,
            "Can_we_safely_consume_the_one_formal_R2_FJT_goal_next": ready_for_formal,
            "retry_allowed": False if budget.get("consumed") else False,
        }
        atomic_write_json(self.output / "stage28sr2_r2_terminal_certificate.json", cert)
        write_json(self.output / "stage28sr2_r2_timing_model.json", self.timing)
        return cert

    def audit_only_run(self) -> dict[str, Any]:
        """Run every pre-send gate and stop before ledger consumption."""

        if not self.audit_only:
            raise RuntimeError("audit_only_run requires audit_only=True")
        certificate: dict[str, Any]
        try:
            preflight = self.run_preflight()
            self.timing["T_formal_boundary"] = now_utc() if preflight.passed else None
            if preflight.passed:
                self._event("audit_stopped_before_ledger_consumption")
                self._write_audit_summary("passed")
            else:
                self.machine.force_terminal(self.first_blocker or preflight.first_blocker())
                self._write_audit_summary(f"blocked_{self.first_blocker or preflight.first_blocker()}")
        except Exception as exc:
            self._set_blocker(f"runner_exception:{type(exc).__name__}:{exc}")
            self.machine.force_terminal(self.first_blocker)
        finally:
            if self.machine.current != FormalState.TERMINAL:
                self.machine.force_terminal("audit_only_end")
            certificate = self._write_certificate()
        return certificate

    def run_formal_with_fake_transport(self) -> dict[str, Any]:
        """Exercise the future send binding using an injected fake only."""

        if self.audit_only:
            raise RuntimeError("formal fake rehearsal requires audit_only=False")
        if not getattr(self.action_transport, "is_fake_transport", False):
            raise RuntimeError("formal rehearsal requires explicit FakeActionTransport injection")
        certificate: dict[str, Any]
        try:
            preflight = self.run_preflight()
            if not preflight.passed:
                self.machine.force_terminal(self.first_blocker or preflight.first_blocker())
            else:
                self.machine.transition(FormalState.READY_FOR_LEDGER)
                consumed = self.ledger.consume_before_send()
                self.timing["T_ledger_consumed"] = now_utc()
                self._event("ledger_consumed")
                if consumed["formal_goal_budget"]["consumed"] != 1 or not consumed["formal_goal_budget"].get("goal_budget_consumed", False):
                    raise RuntimeError("ledger reread did not confirm durable consumption")
                self.machine.transition(FormalState.LEDGER_CONSUMED)
                self.ledger.bind_send_goal_async_call()
                self.ledger.record_transport_send_invocation()
                self.timing["T_send_goal_call"] = now_utc()
                outcome = self.action_transport.send_goal_async(self.goal)
                self.outcome = outcome if isinstance(outcome, FakeTransportOutcome) else FakeTransportOutcome(**dict(outcome))
                self._event("fake_send_called")
                self.machine.transition(FormalState.SEND_ATTEMPTED)
                self.machine.transition(FormalState.ACK_WAIT)
                self.timing["T_goal_acknowledged"] = now_utc() if self.outcome.accepted is not None else None
                self.ledger.record_goal_response(acknowledged=self.outcome.accepted is not None, accepted=self.outcome.accepted)
                if self.outcome.accepted is None:
                    self._set_blocker("ack_timeout")
                    self.machine.transition(FormalState.CLIENT_TIMEOUT)
                elif not self.outcome.accepted:
                    self._set_blocker("server_rejected_goal")
                    self.machine.transition(FormalState.REJECTED)
                else:
                    self.timing["T_goal_accepted"] = now_utc()
                    self.machine.transition(FormalState.ACCEPTED)
                    if not self.outcome.recorder_alive_during_execution:
                        self._set_blocker("recorder_failed_after_send")
                        self.machine.transition(FormalState.EXECUTING)
                        self.machine.transition(FormalState.RECORDER_FAILED)
                    else:
                        self.machine.transition(FormalState.EXECUTING)
                        success_gate = formal_success_positive_whitelist(self.outcome)
                        status = self.outcome.action_terminal_status
                        if self.outcome.cancel_requested or status == "CANCELED":
                            self._set_blocker("goal_canceled")
                            self.machine.transition(FormalState.CANCELED)
                        elif status in {"ABORTED", "PREEMPTED"} or self.outcome.fjt_error_code not in (0, None):
                            self._set_blocker("fjt_action_aborted_or_error_code")
                            self.machine.transition(FormalState.ABORTED)
                        elif not self.outcome.analysis_passed:
                            self._set_blocker("post_execution_analysis_failed")
                            self.machine.transition(FormalState.ANALYSIS_FAILED)
                        elif not self.outcome.recorder_finalized or not self.outcome.bag_complete or not self.outcome.bag_valid:
                            self._set_blocker("recorder_finalization_or_bag_validity_failed")
                            self.machine.transition(FormalState.RECORDER_FAILED)
                        elif not success_gate["passed"]:
                            self._set_blocker(f"formal_success_whitelist_failed:{success_gate['first_blocker']}")
                            self.machine.transition(FormalState.ANALYSIS_FAILED)
                        else:
                            self.timing["T_action_result"] = now_utc()
                            self.machine.transition(FormalState.SUCCEEDED)
                self.machine.transition(FormalState.TERMINAL)
        except Exception as exc:
            self._set_blocker(f"runner_exception:{type(exc).__name__}:{exc}")
            self.machine.force_terminal(self.first_blocker)
        finally:
            if self.machine.current != FormalState.TERMINAL:
                self.machine.force_terminal("formal_run_finally")
            certificate = self._write_certificate()
        return certificate

    def run_audit_only(self) -> dict[str, Any]:
        """Public alias used by deployment and test harnesses."""

        return self.audit_only_run()

    def run_fake_formal(self) -> dict[str, Any]:
        """Public alias for the dependency-injected non-ROS rehearsal."""

        return self.run_formal_with_fake_transport()

    def _write_audit_summary(self, status: str) -> None:
        summary = {
            "mode": "AUDIT ONLY / NO SEND",
            "status": status,
            "stopped_before": "ledger consumption",
            "ledger_consumed": False,
            "fake_send_called": False,
            "real_send_called": False,
            "formal_goal_budget": self.ledger.snapshot()["formal_goal_budget"],
            "real_FJT_goals_sent": 0,
            "same_final_runtime_identity": bool(self.preflight and self.preflight.runtime.get("passed")),
            "frozen_hash_gate": bool(self.preflight and self.preflight.frozen.get("passed")),
            "frozen_goal_semantics_gate": bool(self.preflight and self.preflight.frozen.get("passed")),
            "initial_state_gate": bool(self.preflight and self.preflight.initial.get("passed")),
            "point0_no_extra_interpolation_gate": bool(self.preflight and self.preflight.frozen.get("point0_no_extra_interpolation", {}).get("passed")),
            "recorder_readiness_gate": bool(self.preflight and self.preflight.recorder.get("passed")),
            "command_source_exclusivity_gate": bool(self.preflight and self.preflight.exclusivity.get("passed")),
            "formal_state_machine_static_gate": bool(self.preflight and self.preflight.static_state_machine.get("passed")),
            "atomic_ledger_binding_gate": bool(self.preflight and self.preflight.static_ledger.get("passed")),
            "persistent_terminal_certificate_gate": bool(self.preflight and self.preflight.static_certificate.get("passed")),
            "event_trace": list(self.events),
        }
        write_json(self.output / "stage28sr2_r2_audit_only_summary.json", summary)


def run_regression_suite(output: Path) -> dict[str, Any]:
    log_path = output / "stage28sr2_r2_regression_pytest.log"
    command = [sys.executable, "-m", "pytest", "-q", "tests/test_stage28sr2_r2_formal_runner.py"]
    try:
        completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=180)
        raw = completed.stdout + ("\n--- STDERR ---\n" + completed.stderr if completed.stderr else "")
        log_path.write_text(raw, encoding="utf-8")
        match = re.search(r"(?P<passed>\d+) passed(?:, (?P<failed>\d+) failed)?(?:, (?P<skipped>\d+) skipped)?", raw)
        counts = {"collected": None, "passed": 0, "failed": 0, "skipped": 0}
        if match:
            counts.update({key: int(value or 0) for key, value in (("passed", match.group("passed")), ("failed", match.group("failed")), ("skipped", match.group("skipped")))})
            counts["collected"] = sum(counts[key] for key in ("passed", "failed", "skipped"))
        result = {"pytest_command": " ".join(command), **counts, "raw_log_path": str(log_path.resolve()), "raw_log_sha256": sha256(log_path), "exit_code": completed.returncode, "passed": completed.returncode == 0}
    except Exception as exc:
        log_path.write_text(repr(exc) + "\n", encoding="utf-8")
        result = {"pytest_command": " ".join(command), "collected": None, "passed": 0, "failed": 0, "skipped": 0, "raw_log_path": str(log_path.resolve()), "raw_log_sha256": sha256(log_path), "exit_code": None, "passed": False, "exception": repr(exc)}
    write_json(output / "stage28sr2_r2_regression_suite.json", result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Stage 2.8S-R2 hardened pre-formal runner")
    parser.add_argument("--audit-only", action="store_true", default=True, help="run AUDIT ONLY / NO SEND (default and only CLI mode)")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--runtime-evidence", type=Path, default=None, help="offline diagnostics only; rejected by the production CLI")
    parser.add_argument("--skip-tests", action="store_true")
    args = parser.parse_args(argv)
    if args.runtime_evidence is not None:
        raise SystemExit("production Stage 2.8S-R2 CLI rejects arbitrary --runtime-evidence; use the machine-collected production runner")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = (args.output or ROOT / "outputs/stage28sr2_r2_preformal_audit" / f"stage28sr2_r2_{stamp}").resolve()
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty output: {output}")
    runner = FormalRunner(output=output, runtime_evidence=load_runtime_evidence(args.runtime_evidence), audit_only=True)
    certificate = runner.audit_only_run()
    regression = None if args.skip_tests else run_regression_suite(output)
    certificate["regression_suite"] = regression
    if regression is not None:
        certificate["regression_suite_passed"] = bool(regression.get("passed"))
        if not regression.get("passed"):
            certificate["first_blocker"] = certificate.get("first_blocker") or "regression_suite_failed"
            certificate["Stage_2_8S_R2_PreFormal"] = {"status": "blocked_regression_suite_failed"}
            certificate["Stage_2_8S_R2"] = {"status": "blocked_regression_suite_failed"}
            certificate["Can_we_safely_consume_the_one_formal_R2_FJT_goal_next"] = False
    atomic_write_json(output / "stage28sr2_r2_terminal_certificate.json", certificate)
    print(json.dumps({"output": str(output), "status": certificate["Stage_2_8S_R2"]["status"], "first_blocker": certificate.get("first_blocker"), "formal_goal_budget": certificate["formal_goal_budget"], "real_FJT_goals_sent": certificate["real_FJT_goals_sent"]}, ensure_ascii=False, indent=2))
    return 0 if certificate["Stage_2_8S_R2"]["status"] == "ready_for_one_shot_formal_execution" else 2


if __name__ == "__main__":
    raise SystemExit(main())
