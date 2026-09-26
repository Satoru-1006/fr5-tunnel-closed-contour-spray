"""Deterministic process-aware stress operators and fail-closed validity semantics.

TCP poses are target poses expressed in ``base_link`` and use ``[x,y,z,qx,qy,qz,qw]``.
Joint states are radians in the source trajectory's declared joint order. Missing
process fields remain missing; this module does not infer FK, collision, clearance,
dynamics, or hardware results.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import json
import math
from typing import Any, Mapping, Sequence

import numpy as np


class CapabilityStatus(str, Enum):
    AVAILABLE = "AVAILABLE"
    NOT_AVAILABLE = "NOT_AVAILABLE"
    UNKNOWN = "UNKNOWN"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class StressFamily(str, Enum):
    BASE_TRANSLATION = "BASE_TRANSLATION"
    BASE_ROTATION = "BASE_ROTATION"
    TCP_TRANSLATION = "TCP_TRANSLATION"
    TCP_ROTATION = "TCP_ROTATION"
    SURFACE_NORMAL = "SURFACE_NORMAL"
    SURFACE_OFFSET = "SURFACE_OFFSET"
    STANDOFF = "STANDOFF"
    JOINT_STATE = "JOINT_STATE"
    TIMING_SCALE = "TIMING_SCALE"
    REGRESSION_CONTROL = "REGRESSION_CONTROL"


class ResultStatus(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    UNKNOWN = "UNKNOWN"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class ApplicationStatus(str, Enum):
    APPLIED = "APPLIED"
    UNKNOWN = "UNKNOWN"
    NOT_AVAILABLE = "NOT_AVAILABLE"


class FailureCode(str, Enum):
    IK_INVALID = "IK_INVALID"
    JOINT_LIMIT = "JOINT_LIMIT"
    NEAR_SINGULAR = "NEAR_SINGULAR"
    ENV_COLLISION = "ENV_COLLISION"
    SELF_COLLISION = "SELF_COLLISION"
    LOW_CLEARANCE = "LOW_CLEARANCE"
    TCP_POSITION_ERROR = "TCP_POSITION_ERROR"
    TCP_ORIENTATION_ERROR = "TCP_ORIENTATION_ERROR"
    STANDOFF_ERROR = "STANDOFF_ERROR"
    NORMAL_ANGLE_ERROR = "NORMAL_ANGLE_ERROR"
    VELOCITY_LIMIT = "VELOCITY_LIMIT"
    ACCELERATION_LIMIT = "ACCELERATION_LIMIT"
    JERK_LIMIT = "JERK_LIMIT"
    COVERAGE_FAILURE = "COVERAGE_FAILURE"
    JOINT_DISCONTINUITY = "JOINT_DISCONTINUITY"
    INTERPOLATED_COLLISION = "INTERPOLATED_COLLISION"
    DYNAMIC_FEASIBILITY = "DYNAMIC_FEASIBILITY"
    CAPABILITY_UNAVAILABLE = "CAPABILITY_UNAVAILABLE"
    UNKNOWN = "UNKNOWN"


class FailureScope(str, Enum):
    POINT = "POINT"
    TRANSITION = "TRANSITION"
    TRAJECTORY = "TRAJECTORY"


class LegacyMappingStatus(str, Enum):
    MAPPED = "MAPPED"
    PARTIALLY_MAPPED = "PARTIALLY_MAPPED"
    LEGACY_UNMAPPED = "LEGACY_UNMAPPED"


@dataclass(frozen=True)
class StressOperator:
    operator_id: str
    family: StressFamily
    amplitude: float | None
    units: str
    target_scope: str
    seed: int | None = None
    parameters: Mapping[str, Any] = field(default_factory=dict)
    availability: CapabilityStatus = CapabilityStatus.AVAILABLE
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.operator_id or not self.units or not self.target_scope:
            raise ValueError("operator_id, units, and target_scope are required")
        if self.amplitude is not None and not math.isfinite(float(self.amplitude)):
            raise ValueError("operator amplitude must be finite or None")
        object.__setattr__(self, "family", StressFamily(self.family))
        object.__setattr__(self, "availability", CapabilityStatus(self.availability))

    def to_record(self) -> dict[str, Any]:
        return {
            "operator_id": self.operator_id,
            "family": self.family.value,
            "amplitude": self.amplitude,
            "units": self.units,
            "target_scope": self.target_scope,
            "seed": self.seed,
            "parameters": _jsonable(self.parameters),
            "availability": self.availability.value,
            "metadata": _jsonable(self.metadata),
        }


@dataclass(frozen=True)
class ProcessTrajectory:
    """Optional-field view of one trajectory; absent fields are not synthesized."""

    tcp_poses: np.ndarray | None = None
    joint_states: np.ndarray | None = None
    timestamps_s: np.ndarray | None = None
    surface_normals: np.ndarray | None = None
    surface_offsets_m: np.ndarray | None = None
    stand_off_m: np.ndarray | None = None
    joint_lower_rad: np.ndarray | None = None
    joint_upper_rad: np.ndarray | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        fields = (
            ("tcp_poses", self.tcp_poses, 2),
            ("joint_states", self.joint_states, 2),
            ("timestamps_s", self.timestamps_s, 1),
            ("surface_normals", self.surface_normals, 2),
            ("surface_offsets_m", self.surface_offsets_m, 1),
            ("stand_off_m", self.stand_off_m, 1),
            ("joint_lower_rad", self.joint_lower_rad, 1),
            ("joint_upper_rad", self.joint_upper_rad, 1),
        )
        lengths: set[int] = set()
        for name, value, ndim in fields:
            if value is None:
                continue
            array = np.asarray(value, dtype=np.float64)
            if array.ndim != ndim:
                raise ValueError(f"{name} must have {ndim} dimensions")
            if name == "tcp_poses" and array.shape[1] != 7:
                raise ValueError("tcp_poses columns must be xyz + quaternion xyzw")
            if name == "surface_normals" and array.shape[1] != 3:
                raise ValueError("surface_normals must have three columns")
            if name in {"joint_lower_rad", "joint_upper_rad"}:
                continue
            lengths.add(int(array.shape[0]))
            frozen = np.array(array, copy=True)
            frozen.setflags(write=False)
            object.__setattr__(self, name, frozen)
        if len(lengths) > 1:
            raise ValueError("trajectory fields must have the same waypoint count")
        if self.joint_lower_rad is not None or self.joint_upper_rad is not None:
            if self.joint_lower_rad is None or self.joint_upper_rad is None:
                raise ValueError("both joint limit vectors are required together")
            lower = np.asarray(self.joint_lower_rad, dtype=np.float64)
            upper = np.asarray(self.joint_upper_rad, dtype=np.float64)
            if lower.ndim != 1 or upper.ndim != 1 or lower.shape != upper.shape:
                raise ValueError("joint limit vectors must have matching one-dimensional shapes")
            if self.joint_states is not None and self.joint_states.shape[1] != lower.size:
                raise ValueError("joint limit width must match joint_states")
            for name, array in (("joint_lower_rad", lower), ("joint_upper_rad", upper)):
                frozen = np.array(array, copy=True)
                frozen.setflags(write=False)
                object.__setattr__(self, name, frozen)

    @property
    def waypoint_count(self) -> int | None:
        for value in (self.joint_states, self.tcp_poses, self.timestamps_s, self.surface_normals):
            if value is not None:
                return int(value.shape[0])
        return None

    def to_record(self) -> dict[str, Any]:
        return {
            "tcp_poses": _jsonable(self.tcp_poses),
            "joint_states": _jsonable(self.joint_states),
            "timestamps_s": _jsonable(self.timestamps_s),
            "surface_normals": _jsonable(self.surface_normals),
            "surface_offsets_m": _jsonable(self.surface_offsets_m),
            "stand_off_m": _jsonable(self.stand_off_m),
            "joint_lower_rad": _jsonable(self.joint_lower_rad),
            "joint_upper_rad": _jsonable(self.joint_upper_rad),
            "metadata": _jsonable(self.metadata),
        }


@dataclass(frozen=True)
class StressApplication:
    operator: StressOperator
    trajectory: ProcessTrajectory
    status: ApplicationStatus
    failure_codes: tuple[FailureCode, ...] = ()
    reason: str | None = None

    def serialize(self) -> str:
        return json.dumps(
            {
                "operator": self.operator.to_record(),
                "status": self.status.value,
                "failure_codes": [code.value for code in self.failure_codes],
                "reason": self.reason,
                "trajectory": self.trajectory.to_record(),
            },
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )


@dataclass(frozen=True)
class ValidityCheck:
    status: ResultStatus
    failure_codes: tuple[FailureCode, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", ResultStatus(self.status))
        object.__setattr__(self, "failure_codes", tuple(FailureCode(code) for code in self.failure_codes))


@dataclass(frozen=True)
class FailureEvent:
    scope: FailureScope
    code: FailureCode
    index: int | None = None


@dataclass(frozen=True)
class TrajectoryValidity:
    point_validity: ResultStatus
    transition_validity: ResultStatus
    trajectory_validity: ResultStatus
    failures: tuple[FailureEvent, ...]
    capability_status: Mapping[str, CapabilityStatus]


@dataclass(frozen=True)
class LegacyCaseMapping:
    case_id: str
    family: str
    status: LegacyMappingStatus
    operators: tuple[StressOperator, ...]
    notes: tuple[str, ...] = ()

    def to_record(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "legacy_family": self.family,
            "mapping_status": self.status.value,
            "operators": [operator.to_record() for operator in self.operators],
            "notes": list(self.notes),
        }


@dataclass(frozen=True)
class CapabilityRecord:
    status: CapabilityStatus
    semantics: str


# This is backend capability accounting, not evidence that a measurement ran for
# a given case or that the frozen FR5 trajectory passed.
D46_CAPABILITY_PROFILE = {
    "continuous_robot_world_collision_api": CapabilityRecord(
        CapabilityStatus.AVAILABLE,
        "MoveIt2 CollisionEnvBullet robot-world endpoint-pair API is callable; this is not a proof over arbitrary continuous q(t).",
    ),
    "adaptive_discrete_interpolation": CapabilityRecord(
        CapabilityStatus.AVAILABLE,
        "D41 reports intermediate samples with adaptive_discrete_interpolation semantics.",
    ),
    "continuous_self_collision": CapabilityRecord(
        CapabilityStatus.NOT_AVAILABLE,
        "D41 native provenance declares the MoveIt Bullet wrapper self-motion API unavailable; discrete self checks remain separate.",
    ),
    "fcl_robot_world_distance": CapabilityRecord(
        CapabilityStatus.AVAILABLE,
        "Native FCL distance API can report a per-state value; missing/non-finite per-case values remain unknown.",
    ),
    "fcl_self_distance": CapabilityRecord(
        CapabilityStatus.AVAILABLE,
        "Native FCL self-distance API can report a per-state value; it is not continuous self-collision CCD.",
    ),
    "moveit_fk_and_jacobian": CapabilityRecord(
        CapabilityStatus.AVAILABLE,
        "D46 builds native MoveIt2 FK and RobotState Jacobian measurements; this does not establish physical calibration or a risk threshold.",
    ),
    "generated_case_finite_difference_dynamics": CapabilityRecord(
        CapabilityStatus.AVAILABLE,
        "D46 uses NumPy finite differences on declared scaled timestamps for audit-only generated cases; this is not a post-Ruckig certificate.",
    ),
    "frozen_post_ruckig_regression_replay": CapabilityRecord(
        CapabilityStatus.AVAILABLE,
        "D46 retains captured D41 post-Ruckig q/v/a/jerk for REGRESSION controls; this does not apply to generated perturbation cases.",
    ),
    "model_based_required_torque": CapabilityRecord(
        CapabilityStatus.NOT_AVAILABLE,
        "D46 records torque as unavailable because no validated inverse-dynamics backend or physical torque telemetry is present.",
    ),
    "calibrated_tcp_uncertainty": CapabilityRecord(
        CapabilityStatus.NOT_AVAILABLE,
        "No calibrated physical TCP uncertainty is supplied to the D46 benchmark.",
    ),
    "clearance_acceptance_threshold": CapabilityRecord(
        CapabilityStatus.UNKNOWN,
        "FCL distances can be measured, but the project records clearance acceptance thresholds as unresolved.",
    ),
    "physical_singularity_threshold": CapabilityRecord(
        CapabilityStatus.UNKNOWN,
        "Jacobian SVD is measured, but no physical singularity-risk acceptance threshold is established.",
    ),
    "vendor_certified_dynamic_limits": CapabilityRecord(
        CapabilityStatus.UNKNOWN,
        "Configured velocity/acceleration/jerk limits are not manufacturer-certified in the D46 provenance.",
    ),
}


def apply_stress(trajectory: ProcessTrajectory, operator: StressOperator) -> StressApplication:
    """Apply one deterministic operator, returning UNKNOWN instead of fabricating data."""
    if operator.availability == CapabilityStatus.NOT_AVAILABLE:
        return _unresolved(trajectory, operator, ApplicationStatus.NOT_AVAILABLE, FailureCode.CAPABILITY_UNAVAILABLE, "operator capability is not available")
    if operator.availability in {CapabilityStatus.UNKNOWN, CapabilityStatus.NOT_APPLICABLE}:
        return _unresolved(trajectory, operator, ApplicationStatus.UNKNOWN, FailureCode.UNKNOWN, "operator capability is not established")
    if operator.amplitude is None:
        return _unresolved(trajectory, operator, ApplicationStatus.UNKNOWN, FailureCode.UNKNOWN, "operator amplitude is not known")
    if operator.amplitude == 0.0:
        return StressApplication(operator, trajectory, ApplicationStatus.APPLIED)

    family = operator.family
    params = operator.parameters
    if family in {StressFamily.BASE_TRANSLATION, StressFamily.BASE_ROTATION, StressFamily.TCP_TRANSLATION, StressFamily.TCP_ROTATION}:
        if trajectory.tcp_poses is None:
            return _missing(trajectory, operator, "tcp_poses")
        poses = np.array(trajectory.tcp_poses, copy=True)
        if family in {StressFamily.BASE_TRANSLATION, StressFamily.TCP_TRANSLATION}:
            key = "direction_base" if family == StressFamily.BASE_TRANSLATION else "direction_tcp"
            direction = _unit_vector(params.get(key), 3, key)
            if family == StressFamily.BASE_TRANSLATION:
                poses[:, :3] += operator.amplitude * direction
            else:
                for index, quat in enumerate(poses[:, 3:7]):
                    poses[index, :3] += operator.amplitude * _rotate(quat, direction)
        else:
            axis_key = "axis_base" if family == StressFamily.BASE_ROTATION else "axis_tcp"
            axis = _unit_vector(params.get(axis_key), 3, axis_key)
            delta = _axis_angle_quaternion(axis, operator.amplitude)
            if family == StressFamily.BASE_ROTATION:
                origin = _as_vector(params.get("origin_base_m", (0.0, 0.0, 0.0)), 3, "origin_base_m")
                for index, quat in enumerate(poses[:, 3:7]):
                    poses[index, :3] = origin + _rotate(delta, poses[index, :3] - origin)
                    poses[index, 3:7] = _quat_multiply(delta, _unit_quaternion(quat))
            else:
                for index, quat in enumerate(poses[:, 3:7]):
                    poses[index, 3:7] = _quat_multiply(_unit_quaternion(quat), delta)
        return _applied(trajectory, operator, tcp_poses=poses)

    if family == StressFamily.SURFACE_NORMAL:
        if trajectory.surface_normals is None:
            return _missing(trajectory, operator, "surface_normals")
        axis = _unit_vector(params.get("axis_base"), 3, "axis_base")
        delta = _axis_angle_quaternion(axis, operator.amplitude)
        normals = np.array(trajectory.surface_normals, copy=True)
        norms = np.linalg.norm(normals, axis=1)
        if not np.isfinite(normals).all() or np.any(norms <= 1e-12):
            return _unresolved(trajectory, operator, ApplicationStatus.UNKNOWN, FailureCode.UNKNOWN, "surface normal is non-finite or zero length")
        for index, normal in enumerate(normals):
            normals[index] = _rotate(delta, normal / norms[index])
            normals[index] /= np.linalg.norm(normals[index])
        return _applied(trajectory, operator, surface_normals=normals)

    if family in {StressFamily.SURFACE_OFFSET, StressFamily.STANDOFF}:
        name = "surface_offsets_m" if family == StressFamily.SURFACE_OFFSET else "stand_off_m"
        current = getattr(trajectory, name)
        if current is None:
            return _missing(trajectory, operator, name)
        return _applied(trajectory, operator, **{name: current + operator.amplitude})

    if family == StressFamily.TIMING_SCALE:
        if trajectory.timestamps_s is None:
            return _missing(trajectory, operator, "timestamps_s")
        valid, reason = validate_timestamps(trajectory.timestamps_s)
        scale = float(params.get("scale_factor", 1.0 + operator.amplitude))
        if not valid:
            return _unresolved(trajectory, operator, ApplicationStatus.UNKNOWN, FailureCode.UNKNOWN, reason)
        if not math.isfinite(scale) or scale <= 0.0:
            raise ValueError("TIMING_SCALE requires a finite positive scale_factor")
        if not math.isclose(scale, 1.0 + operator.amplitude, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("scale_factor must equal 1 + signed operator amplitude")
        times = trajectory.timestamps_s[0] + (trajectory.timestamps_s - trajectory.timestamps_s[0]) * scale
        return _applied(trajectory, operator, timestamps_s=times)

    if family == StressFamily.JOINT_STATE:
        if trajectory.joint_states is None:
            return _missing(trajectory, operator, "joint_states")
        states = np.array(trajectory.joint_states, copy=True)
        mode = str(params.get("mode", "directional"))
        if mode == "directional":
            states += operator.amplitude * _unit_vector(params.get("direction"), states.shape[1], "direction")
        elif mode == "seeded_gaussian":
            if operator.seed is None:
                return _unresolved(trajectory, operator, ApplicationStatus.UNKNOWN, FailureCode.UNKNOWN, "seeded joint noise requires a seed")
            states += operator.amplitude * np.random.default_rng(operator.seed).standard_normal(states.shape)
        elif mode == "wave":
            frequencies = np.asarray(params.get("frequencies"), dtype=np.float64)
            phases = np.asarray(params.get("phases"), dtype=np.float64)
            if frequencies.shape != (states.shape[1],) or phases.shape != (states.shape[1],):
                raise ValueError("wave mode requires one frequency and phase per joint")
            x = np.linspace(0.0, 1.0, len(states))
            waves = np.sin(np.pi * x)[:, None] * np.sin(2.0 * np.pi * frequencies[None, :] * x[:, None] + phases[None, :])
            states += operator.amplitude * waves
        elif mode == "wave_to_target":
            target = np.asarray(params.get("target_joint_state_rad"), dtype=np.float64)
            if target.shape != (states.shape[1],):
                raise ValueError("wave_to_target requires one target value per joint")
            x = np.linspace(0.0, 1.0, len(states))
            frequency = float(params["frequency"])
            phase = float(params.get("phase", 0.0))
            wave = np.sin(np.pi * x) * np.sin(2.0 * np.pi * frequency * x + phase)
            states += operator.amplitude * wave[:, None] * (target[None, :] - states)
        elif mode == "joint_offset":
            joint = int(params["joint_index"])
            if not 0 <= joint < states.shape[1]:
                raise ValueError("joint_index is outside joint_states")
            states[:, joint] += operator.amplitude
        elif mode == "boundary_push":
            if trajectory.joint_lower_rad is None or trajectory.joint_upper_rad is None:
                return _missing(trajectory, operator, "joint limits")
            joint = int(params["joint_index"])
            direction = int(params["direction"])
            margin = float(params["margin_rad"])
            if not 0 <= joint < states.shape[1] or direction not in {-1, 1} or margin < 0.0:
                raise ValueError("invalid boundary_push parameters")
            if operator.amplitude * direction < 0.0 or not math.isclose(abs(operator.amplitude), margin, rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError("signed amplitude and boundary margin/direction must agree")
            bound = trajectory.joint_upper_rad[joint] if direction > 0 else trajectory.joint_lower_rad[joint]
            endpoint = bound - direction * margin
            reference = np.max(states[:, joint]) if direction > 0 else np.min(states[:, joint])
            states[:, joint] += endpoint - reference
        else:
            raise ValueError(f"unsupported JOINT_STATE mode: {mode}")
        return _applied(trajectory, operator, joint_states=states)

    if family == StressFamily.REGRESSION_CONTROL:
        return _unresolved(trajectory, operator, ApplicationStatus.UNKNOWN, FailureCode.UNKNOWN, "regression control is an identity descriptor, not a perturbation")
    raise ValueError(f"unsupported stress family: {family.value}")


def validate_timestamps(timestamps_s: Sequence[float] | np.ndarray | None) -> tuple[bool, str | None]:
    if timestamps_s is None:
        return False, "timestamps are unavailable"
    values = np.asarray(timestamps_s, dtype=np.float64)
    if values.ndim != 1 or values.size < 2 or not np.isfinite(values).all():
        return False, "timestamps must contain at least two finite values"
    if np.any(np.diff(values) <= 0.0):
        return False, "timestamps must be strictly increasing with positive intervals"
    if values[-1] - values[0] <= 0.0:
        return False, "trajectory duration must be positive"
    return True, None


def check_joint_limits(trajectory: ProcessTrajectory) -> tuple[ValidityCheck, ...]:
    """Classify each waypoint; never clip an out-of-range state into validity."""
    q = trajectory.joint_states
    if q is None:
        return (ValidityCheck(ResultStatus.UNKNOWN, (FailureCode.UNKNOWN,)),)
    if trajectory.joint_lower_rad is None or trajectory.joint_upper_rad is None:
        return tuple(ValidityCheck(ResultStatus.UNKNOWN, (FailureCode.UNKNOWN,)) for _ in q)
    checks: list[ValidityCheck] = []
    for row in q:
        if not np.isfinite(row).all():
            checks.append(ValidityCheck(ResultStatus.UNKNOWN, (FailureCode.UNKNOWN,)))
        elif np.any(row < trajectory.joint_lower_rad) or np.any(row > trajectory.joint_upper_rad):
            checks.append(ValidityCheck(ResultStatus.FAIL, (FailureCode.JOINT_LIMIT,)))
        else:
            checks.append(ValidityCheck(ResultStatus.PASS))
    return tuple(checks)


def aggregate_trajectory_validity(
    point_checks: Sequence[ValidityCheck],
    transition_checks: Sequence[ValidityCheck],
    required_capabilities: Mapping[str, CapabilityStatus] | None = None,
) -> TrajectoryValidity:
    """Keep point, transition, and whole-trajectory outcomes separate."""
    points = tuple(point_checks)
    transitions = tuple(transition_checks)
    failures: list[FailureEvent] = []
    for index, check in enumerate(points):
        failures.extend(FailureEvent(FailureScope.POINT, code, index) for code in check.failure_codes)
    for index, check in enumerate(transitions):
        failures.extend(FailureEvent(FailureScope.TRANSITION, code, index) for code in check.failure_codes)

    point_status = _combine_status([check.status for check in points], empty_unknown=True)
    expected_edges = max(0, len(points) - 1)
    transition_status = _combine_status([check.status for check in transitions], empty_unknown=expected_edges > 0)
    if len(transitions) != expected_edges:
        if transition_status != ResultStatus.FAIL:
            transition_status = ResultStatus.UNKNOWN
        failures.append(FailureEvent(FailureScope.TRAJECTORY, FailureCode.UNKNOWN))

    capability_status = {name: CapabilityStatus(value) for name, value in (required_capabilities or {}).items()}
    capability_unresolved = False
    for status in capability_status.values():
        if status == CapabilityStatus.NOT_AVAILABLE:
            capability_unresolved = True
            failures.append(FailureEvent(FailureScope.TRAJECTORY, FailureCode.CAPABILITY_UNAVAILABLE))
        elif status == CapabilityStatus.UNKNOWN:
            capability_unresolved = True
            failures.append(FailureEvent(FailureScope.TRAJECTORY, FailureCode.UNKNOWN))

    if ResultStatus.FAIL in {point_status, transition_status}:
        trajectory_status = ResultStatus.FAIL
    elif point_status != ResultStatus.PASS or transition_status != ResultStatus.PASS or capability_unresolved:
        trajectory_status = ResultStatus.UNKNOWN
    else:
        trajectory_status = ResultStatus.PASS
    return TrajectoryValidity(point_status, transition_status, trajectory_status, tuple(failures), capability_status)


def map_legacy_d46_case(case: Mapping[str, Any]) -> LegacyCaseMapping:
    """Describe an existing D46 case without changing its files or generation."""
    case_id = str(case.get("case_id", "unknown"))
    family = str(case.get("family", "UNKNOWN")).upper()
    index = int(case.get("index", -1))
    generation = str(case.get("generation", ""))
    seed = int(case["seed"]) if case.get("seed") is not None else None
    operators: list[StressOperator] = []
    notes: list[str] = []
    partial = False

    def add_joint(amplitude: float | None, units: str, parameters: Mapping[str, Any], *, availability: CapabilityStatus = CapabilityStatus.AVAILABLE, operator_seed: int | None = seed) -> None:
        operators.append(StressOperator(
            operator_id=f"legacy-d46:{case_id}:joint-state",
            family=StressFamily.JOINT_STATE,
            amplitude=amplitude,
            units=units,
            target_scope="all trajectory joint states",
            seed=operator_seed,
            parameters=parameters,
            availability=availability,
            metadata={"legacy_generation": generation, "legacy_case_seed_is_identity": True},
        ))

    if family == "NORMAL" and index >= 0:
        amplitude = 0.00015 + 0.00035 * (index % 5) / 4.0
        add_joint(amplitude, "rad", {
            "mode": "wave",
            "frequencies": [1 + (joint % 3) for joint in range(6)],
            "phases": [0.13 * index + joint for joint in range(6)],
            "legacy_clip_policy": "np.clip(base_q + delta, lower + 1e-5, upper - 1e-5)",
        })
    elif family == "BOUNDARY" and index >= 0:
        margins = (0.0001, 0.001, 0.005, 0.01)
        margin = margins[(index // 6) % len(margins)]
        direction = 1 if index % 2 == 0 else -1
        add_joint(direction * margin, "rad", {
            "mode": "boundary_push", "joint_index": index % 6,
            "direction": direction, "margin_rad": margin,
            "legacy_clip_policy": "np.clip(..., lower + 1e-7, upper - 1e-7)",
        })
    elif family == "COLLISION_SENSITIVE" and index >= 0:
        fraction = 0.15 + 0.85 * ((index % 17) / 16.0)
        target = [-1.5388388093126519, 0.089560286735724581, -2.6759293795387253,
                  -3.0521654331547774, -2.0899656685260903, -2.8783824178455135]
        add_joint(fraction, "ratio", {
            "mode": "wave_to_target", "frequency": 0.5 + (index % 3) * 0.25,
            "phase": 0.0, "target_joint_state_rad": target,
            "legacy_clip_policy": "np.clip(base_q + excursion, lower + 1e-6, upper - 1e-6)",
        })
    elif family == "ADVERSARIAL" and index >= 0:
        if index < 20 and generation == "deliberate_j3_position_limit_exceedance":
            amplitude = 0.01 + 0.001 * (index % 5)
            add_joint(amplitude, "rad", {"mode": "joint_offset", "joint_index": 2, "direction": 1,
                                          "hard_invalid_input_by_design": True})
        elif generation == "high_curvature_short_horizon_stress":
            amplitude = 0.01 + 0.04 * ((index % 10) / 9.0)
            add_joint(amplitude, "rad", {
                "mode": "wave", "frequencies": [3 + (joint % 5) for joint in range(6)],
                "phases": [0.7 * index + joint for joint in range(6)],
                "legacy_clip_policy": "np.clip(base_q + delta, lower + 1e-6, upper - 1e-6)",
            })
        else:
            return LegacyCaseMapping(case_id, family, LegacyMappingStatus.LEGACY_UNMAPPED, (), ("unrecognized ADVERSARIAL generation descriptor",))
    elif family == "PERTURBATION" and index >= 0:
        sigma = (0.0005, 0.001, 0.002, 0.005)[index % 4]
        add_joint(sigma, "rad", {
            "mode": "wave", "frequencies": [1 + (joint % 4) for joint in range(6)],
            "phase_source": "frozen global RNG stream seed 460046; realized phase vector not stored in case record",
            "legacy_clip_policy": "np.clip(base_q + delta, lower + 1e-6, upper - 1e-6)",
        }, availability=CapabilityStatus.UNKNOWN, operator_seed=460046)
        shift = np.asarray(case.get("target_shift_m", ()), dtype=np.float64)
        if shift.shape == (3,) and np.isfinite(shift).all():
            magnitude = float(np.linalg.norm(shift))
            direction = (shift / magnitude).tolist() if magnitude > 0.0 else [0.0, 0.0, 0.0]
            operators.append(StressOperator(
                operator_id=f"legacy-d46:{case_id}:target-shift", family=StressFamily.BASE_TRANSLATION,
                amplitude=magnitude, units="m", target_scope="target TCP poses in base_link",
                seed=seed, parameters={"direction_base": direction},
                metadata={"legacy_semantics": "target comparison shift in base_link; not a calibrated robot-workpiece transform"},
            ))
        else:
            partial = True
            notes.append("target_shift_m is absent or malformed")
        partial = True
        notes.append("the D46 case record does not preserve the realized joint phase vector; joint-state replay is UNKNOWN")
    elif family == "REGRESSION" and generation in {
        "actual_frozen_stage3_post_ruckig_replay", "frozen_pre_ruckig_waypoint_replay", "repeated_frozen_post_ruckig_replay"
    }:
        operators.append(StressOperator(
            operator_id=f"legacy-d46:{case_id}:identity-replay", family=StressFamily.REGRESSION_CONTROL,
            amplitude=0.0, units="1", target_scope="full trajectory", seed=seed,
            parameters={"mode": "identity_replay", "generation": generation},
        ))
    else:
        return LegacyCaseMapping(case_id, family, LegacyMappingStatus.LEGACY_UNMAPPED, (), ("family or generation descriptor is not recognized",))

    if "time_scale" in case:
        scale = float(case["time_scale"])
        if not math.isfinite(scale) or scale <= 0.0:
            return LegacyCaseMapping(case_id, family, LegacyMappingStatus.LEGACY_UNMAPPED, tuple(operators), ("legacy time_scale is not positive and finite",))
        if scale != 1.0:
            operators.append(StressOperator(
                operator_id=f"legacy-d46:{case_id}:timing-scale", family=StressFamily.TIMING_SCALE,
                amplitude=scale - 1.0, units="ratio", target_scope="all trajectory timestamps",
                seed=seed, parameters={"scale_factor": scale},
                metadata={"legacy_generation": generation},
            ))
    status = LegacyMappingStatus.PARTIALLY_MAPPED if partial else LegacyMappingStatus.MAPPED
    if any(operator.availability == CapabilityStatus.UNKNOWN for operator in operators):
        status = LegacyMappingStatus.PARTIALLY_MAPPED
    if "np.clip" in " ".join(json.dumps(operator.parameters) for operator in operators):
        notes.append("legacy generation recorded a clip policy; apply_stress itself never clips joint states")
    return LegacyCaseMapping(case_id, family, status, tuple(operators), tuple(notes))


def _combine_status(statuses: Sequence[ResultStatus], *, empty_unknown: bool) -> ResultStatus:
    if any(status == ResultStatus.FAIL for status in statuses):
        return ResultStatus.FAIL
    if not statuses:
        return ResultStatus.UNKNOWN if empty_unknown else ResultStatus.PASS
    if any(status in {ResultStatus.UNKNOWN, ResultStatus.NOT_APPLICABLE} for status in statuses):
        return ResultStatus.UNKNOWN
    return ResultStatus.PASS


def _applied(trajectory: ProcessTrajectory, operator: StressOperator, **changes: Any) -> StressApplication:
    from dataclasses import replace

    return StressApplication(operator, replace(trajectory, **changes), ApplicationStatus.APPLIED)


def _unresolved(trajectory: ProcessTrajectory, operator: StressOperator, status: ApplicationStatus, code: FailureCode, reason: str) -> StressApplication:
    return StressApplication(operator, trajectory, status, (code,), reason)


def _missing(trajectory: ProcessTrajectory, operator: StressOperator, field_name: str) -> StressApplication:
    return _unresolved(trajectory, operator, ApplicationStatus.NOT_AVAILABLE, FailureCode.CAPABILITY_UNAVAILABLE, f"required trajectory field unavailable: {field_name}")


def _as_vector(value: Any, size: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (size,) or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite vector of length {size}")
    return array


def _unit_vector(value: Any, size: int, name: str) -> np.ndarray:
    array = _as_vector(value, size, name)
    norm = float(np.linalg.norm(array))
    if norm <= 1e-12:
        raise ValueError(f"{name} must have nonzero direction")
    return array / norm


def _unit_quaternion(value: Any) -> np.ndarray:
    quat = _as_vector(value, 4, "quaternion_xyzw")
    norm = float(np.linalg.norm(quat))
    if norm <= 1e-12:
        raise ValueError("quaternion must have nonzero norm")
    return quat / norm


def _quat_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.asarray((aw * bx + ax * bw + ay * bz - az * by,
                       aw * by - ax * bz + ay * bw + az * bx,
                       aw * bz + ax * by - ay * bx + az * bw,
                       aw * bw - ax * bx - ay * by - az * bz), dtype=np.float64)


def _axis_angle_quaternion(axis: np.ndarray, angle: float) -> np.ndarray:
    half = 0.5 * angle
    return np.asarray((*list(axis * math.sin(half)), math.cos(half)), dtype=np.float64)


def _rotate(quaternion_xyzw: Any, vector: np.ndarray) -> np.ndarray:
    q = _unit_quaternion(quaternion_xyzw)
    pure = np.asarray((vector[0], vector[1], vector[2], 0.0), dtype=np.float64)
    conjugate = np.asarray((-q[0], -q[1], -q[2], q[3]), dtype=np.float64)
    return _quat_multiply(_quat_multiply(q, pure), conjugate)[:3]


def _jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value
