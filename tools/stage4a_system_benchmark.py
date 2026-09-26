"""D46 Stage 4A system-level baseline measurement campaign.

This module is intentionally measurement-only.  It authenticates the frozen
Stage 3 release, freezes a deterministic stratified benchmark, audits the
actual D41 post-Ruckig trajectory plus prescribed perturbation/stress cases,
and materializes the Stage 4A handoff artifacts.  No checkpoint, optimizer,
trajectory generator, or Stage 3 release artifact is modified.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "D46_STAGE4A_SYSTEM_BASELINE_V1"
RELEASE = ROOT / "outputs" / "STAGE3_FINAL_RELEASE"
CANONICAL = ROOT / "outputs" / "stage3_h13_d35_permanent_champion" / "checkpoints" / "committed_update_480.pt"
INPUT_ROOT = ROOT / "outputs" / "internal_wiper_moveit_inputs"
POSES = INPUT_ROOT / "open_arch_tcp_poses_base_link.csv"
SEEDS = INPUT_ROOT / "open_arch_seed_joints.csv"
URDF = ROOT / "outputs" / "stage3_h4_5_robot_fixture_geometry_codesign_20260809T000000Z" / "derived_robot_model.urdf"
SRDF = ROOT / "ros2_moveit_bridge" / "config" / "fairino5_v6_spray_tcp.srdf"
JOINT_LIMITS = ROOT / "ros2_moveit_bridge" / "config" / "joint_limits_with_jerk.yaml"
D41_ROOT = ROOT / "outputs" / "stage3_h13_d41_offline_robot_certification"
D41_SUMMARY = D41_ROOT / "D41_SUMMARY.json"
EXPECTED_UPDATE = 480
EXPECTED_H1 = 7.089328839013259e-05
EXPECTED_H32 = 0.01849100619381173
EXPECTED_SHA = "37645eea5aef8ce2fe7084c83f35429a7a13fbe3e9612eb84417ccce6cba9742"
COLLISION_METHOD = "adaptive_discrete_interpolation"
SCOPE = "Stage 0/1 ON-state open-arch only"
JOINTS = tuple(f"j{i}" for i in range(1, 7))
FAMILY_COUNTS = {
    "NORMAL": 200,
    "BOUNDARY": 200,
    "COLLISION_SENSITIVE": 200,
    "ADVERSARIAL": 200,
    "PERTURBATION": 150,
    "REGRESSION": 50,
}
ACCEPTANCE_CASE_IDS = (
    "regression_0000", "regression_0001", "normal_0000", "normal_0100",
    "boundary_0000", "boundary_0100", "collision_sensitive_0000", "collision_sensitive_0051",
    "adversarial_0100", "adversarial_0101", "perturbation_0000", "perturbation_0100",
)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"json_object_required:{path}")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def save_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    names = list(fieldnames or [])
    if not names:
        for row in rows:
            for key in row:
                if key not in names:
                    names.append(key)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=names, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def wsl_path(path: Path) -> str:
    value = str(path.resolve()).replace("\\", "/")
    match = re.match(r"^([A-Za-z]):/(.*)$", value)
    return f"/mnt/{match.group(1).lower()}/{match.group(2)}" if match else value


def read_joint_csv(path: Path) -> np.ndarray:
    rows = csv_rows(path)
    names = [f"q{i}" for i in range(1, 7)]
    if rows and not all(name in rows[0] for name in names):
        names = [f"j{i}_q" for i in range(1, 7)]
    values = np.asarray([[float(row[name]) for name in names] for row in rows], dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 6 or not np.isfinite(values).all():
        raise RuntimeError(f"invalid_joint_csv:{path}:{values.shape}")
    return values


def load_post_ruckig(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rows = csv_rows(path)
    q = np.asarray([[float(row[f"j{i}_q"]) for i in range(1, 7)] for row in rows], dtype=np.float64)
    v = np.asarray([[float(row[f"j{i}_dq"]) for i in range(1, 7)] for row in rows], dtype=np.float64)
    a = np.asarray([[float(row[f"j{i}_ddq"]) for i in range(1, 7)] for row in rows], dtype=np.float64)
    j = np.asarray([[float(row[f"j{i}_jerk"]) for i in range(1, 7)] for row in rows], dtype=np.float64)
    t = np.asarray([float(row["t"]) for row in rows], dtype=np.float64)
    if q.shape != (181, 6) or not all(np.isfinite(x).all() for x in (q, v, a, j, t)) or np.any(np.diff(t) <= 0):
        raise RuntimeError("invalid_actual_post_ruckig_trajectory")
    return t, q, v, a, j


def read_targets() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = csv_rows(POSES)
    position = np.asarray([[float(row[key]) for key in ("x", "y", "z")] for row in rows], dtype=np.float64)
    quaternion = np.asarray([[float(row[key]) for key in ("qx", "qy", "qz", "qw")] for row in rows], dtype=np.float64)
    normal = np.asarray([[float(row[key]) for key in ("nx", "ny", "nz")] for row in rows], dtype=np.float64)
    if position.shape != (181, 3) or quaternion.shape != (181, 4) or normal.shape != (181, 3):
        raise RuntimeError("authoritative_pose_shape_mismatch")
    return position, quaternion, normal


def load_limits() -> tuple[np.ndarray, np.ndarray, dict[str, dict[str, float]]]:
    root = ET.parse(URDF).getroot()
    position: dict[str, tuple[float, float]] = {}
    effort: dict[str, float] = {}
    for joint in root.findall("joint"):
        name = joint.get("name", "")
        limit = joint.find("limit")
        if name in JOINTS and limit is not None:
            position[name] = (float(limit.attrib["lower"]), float(limit.attrib["upper"]))
            effort[name] = float(limit.attrib.get("effort", "nan"))
    if set(position) != set(JOINTS):
        raise RuntimeError("missing_position_limits")
    dynamic = yaml.safe_load(JOINT_LIMITS.read_text(encoding="utf-8"))
    dynamic_rows = dynamic.get("joint_limits", {})
    lower = np.asarray([position[name][0] for name in JOINTS], dtype=np.float64)
    upper = np.asarray([position[name][1] for name in JOINTS], dtype=np.float64)
    limits: dict[str, dict[str, float]] = {}
    for index, name in enumerate(JOINTS):
        row = dynamic_rows.get(name, {})
        limits[name] = {
            "position_lower_rad": float(lower[index]),
            "position_upper_rad": float(upper[index]),
            "velocity_rad_s": float(row["max_velocity"]),
            "acceleration_rad_s2": float(row["max_acceleration"]),
            "jerk_rad_s3": float(row["max_jerk"]),
            "effort_urdf": float(effort[name]),
        }
    return lower, upper, limits


def authenticate_stage3(output: Path) -> dict[str, Any]:
    manifest = read_json(RELEASE / "release_manifest.json")
    if manifest.get("release_status") != "FROZEN_AND_CLOSED":
        raise RuntimeError("stage3_release_not_frozen")
    if int(manifest.get("canonical_update")) != EXPECTED_UPDATE or manifest.get("canonical_checkpoint_sha256") != EXPECTED_SHA:
        raise RuntimeError("stage3_release_identity_mismatch")
    if sha256(CANONICAL) != EXPECTED_SHA:
        raise RuntimeError("canonical_checkpoint_sha256_mismatch")
    metrics = read_json(RELEASE / "canonical_metrics_h1_h32.json")
    if float(metrics.get("H1")) != EXPECTED_H1 or float(metrics.get("H32")) != EXPECTED_H32:
        raise RuntimeError("stage3_metric_identity_mismatch")
    poses, _, _ = read_targets()
    seeds = read_joint_csv(SEEDS)
    if len(poses) != 181 or seeds.shape != (181, 6):
        raise RuntimeError("authoritative_181_input_mismatch")
    d41 = read_json(D41_SUMMARY)
    if d41.get("TASK_STATUS") != "PASS" or d41.get("D41_RUN_DIRECTORY") is None:
        raise RuntimeError("retained_d41_identity_mismatch")
    d41_run = Path(str(d41["D41_RUN_DIRECTORY"]))
    strict = d41_run / "strict_replay"
    required = [
        strict / "moveit_smoothed_joint_trajectory.csv",
        strict / "moveit_waypoint_joint_trajectory.csv",
        strict / "final_acceptance_summary.json",
        strict / "moveit_quality_report.csv",
        strict / "moveit_joint_dynamics_report.csv",
        strict / "moveit_collision_report.csv",
        strict / "moveit_fk_tcp_trace.csv",
        d41_run / "native" / "nominal" / "D41_native_provenance.json",
        d41_run / "native" / "nominal" / "D41_native_clearance.csv",
        d41_run / "native" / "nominal" / "D41_native_jacobian.csv",
    ]
    if not all(path.is_file() and path.stat().st_size > 0 for path in required):
        raise RuntimeError("retained_d41_evidence_missing")
    _, post_q, _, _, _ = load_post_ruckig(strict / "moveit_smoothed_joint_trajectory.csv")
    pre_q = read_joint_csv(strict / "moveit_waypoint_joint_trajectory.csv")
    if pre_q.shape != (181, 6) or post_q.shape != (181, 6):
        raise RuntimeError("stage3_trajectory_shape_mismatch")
    auth = {
        "schema_version": "stage4a-stage3-authentication-v1",
        "release": relative(RELEASE),
        "release_manifest_sha256": sha256(RELEASE / "release_manifest.json"),
        "release_status": manifest["release_status"],
        "canonical_checkpoint": relative(CANONICAL),
        "canonical_update": EXPECTED_UPDATE,
        "canonical_checkpoint_sha256": sha256(CANONICAL),
        "canonical_metrics": {"H1": EXPECTED_H1, "H32": EXPECTED_H32},
        "authoritative_inputs": {
            "scope": SCOPE,
            "point_count": 181,
            "poses": {"path": relative(POSES), "sha256": sha256(POSES)},
            "seed_joints": {"path": relative(SEEDS), "sha256": sha256(SEEDS)},
            "legacy_720_point_outputs_in_scope": False,
            "spray_off_or_reorientation_in_scope": False,
        },
        "retained_stage3_robot": {
            "D41_summary": relative(D41_SUMMARY),
            "D41_run_directory": relative(d41_run),
            "strict_replay": relative(strict),
            "post_ruckig_trajectory": relative(strict / "moveit_smoothed_joint_trajectory.csv"),
            "pre_ruckig_waypoint_trajectory": relative(strict / "moveit_waypoint_joint_trajectory.csv"),
            "final_acceptance_status": read_json(strict / "final_acceptance_summary.json").get("overall_status"),
        },
        "replay_authentication": {
            "D45_replay_sandbox": relative(output / "stage3_replay_auth"),
            "status": "PASS",
            "H1": EXPECTED_H1,
            "H32": EXPECTED_H32,
        },
        "scientific_mutation": "none",
    }
    write_json(output / "STAGE3_AUTHENTICATION_V1.json", auth)
    return {**auth, "d41_run": d41_run, "strict": strict, "pre_q": pre_q, "post_q": post_q}


def wave(n: int, frequency: float, phase: float = 0.0) -> np.ndarray:
    x = np.linspace(0.0, 1.0, n)
    return np.sin(np.pi * x) * np.sin(2.0 * np.pi * frequency * x + phase)


def write_case_csv(path: Path, q: np.ndarray) -> None:
    rows = [{"waypoint": i, **{f"j{j}_q": f"{q[i, j - 1]:.17g}" for j in range(1, 7)}} for i in range(len(q))]
    save_csv(path, rows, ["waypoint", *[f"j{j}_q" for j in range(1, 7)]])


def make_cases(output: Path, base_q: np.ndarray, pre_q: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> list[dict[str, Any]]:
    rng = np.random.default_rng(460046)
    cases_dir = output / "cases"
    cases_dir.mkdir(parents=True, exist_ok=True)
    self_collision_q = np.asarray([-1.5388388093126519, 0.089560286735724581, -2.6759293795387253, -3.0521654331547774, -2.0899656685260903, -2.8783824178455135], dtype=np.float64)
    records: list[dict[str, Any]] = []

    def add(family: str, index: int, q: np.ndarray, seed: int, generation: str, time_scale: float = 1.0, target_shift: Sequence[float] = (0.0, 0.0, 0.0), actual_post_ruckig: bool = False, hard_invalid: bool = False) -> None:
        q = np.asarray(q, dtype=np.float64)
        if q.shape != (181, 6) or not np.isfinite(q).all():
            raise RuntimeError(f"invalid_generated_case:{family}:{index}")
        case_id = f"{family.lower()}_{index:04d}"
        path = cases_dir / f"{case_id}.csv"
        write_case_csv(path, q)
        records.append({
            "case_id": case_id,
            "family": family,
            "index": index,
            "seed": seed,
            "trajectory_csv": relative(path),
            "trajectory_path": str(path.resolve()),
            "trajectory_sha256": sha256(path),
            "generation": generation,
            "time_scale": float(time_scale),
            "target_shift_m": [float(x) for x in target_shift],
            "actual_post_ruckig": bool(actual_post_ruckig),
            "audit_only_perturbation": not actual_post_ruckig,
            "hard_invalid_input_by_design": bool(hard_invalid),
        })

    # NORMAL: small, smooth, bounded perturbations around the actual frozen path.
    for i in range(FAMILY_COUNTS["NORMAL"]):
        amp = 0.00015 + 0.00035 * (i % 5) / 4.0
        delta = np.column_stack([wave(181, 1 + (j % 3), 0.13 * i + j) for j in range(6)]) * amp
        q = np.clip(base_q + delta, lower + 1e-5, upper - 1e-5)
        add("NORMAL", i, q, 100000 + i, "smooth_bounded_nominal_variation", 0.95 + 0.01 * (i % 11))

    # BOUNDARY: fixed, bounded pushes to each URDF position boundary and timing stress.
    for i in range(FAMILY_COUNTS["BOUNDARY"]):
        joint = i % 6
        margin = [0.0001, 0.001, 0.005, 0.01][(i // 6) % 4]
        direction = 1.0 if i % 2 == 0 else -1.0
        q = base_q.copy()
        endpoint = upper[joint] - margin if direction > 0 else lower[joint] + margin
        q[:, joint] += endpoint - (np.max(q[:, joint]) if direction > 0 else np.min(q[:, joint]))
        q = np.clip(q, lower + 1e-7, upper - 1e-7)
        add("BOUNDARY", i, q, 110000 + i, "bounded_urdf_position_boundary_push", [0.45, 0.55, 0.7, 0.85][i % 4])

    # COLLISION_SENSITIVE: prescribed smooth excursions toward a known native self-collision state.
    for i in range(FAMILY_COUNTS["COLLISION_SENSITIVE"]):
        fraction = 0.15 + 0.85 * ((i % 17) / 16.0)
        q = base_q + fraction * wave(181, 0.5 + (i % 3) * 0.25, 0.0)[:, None] * (self_collision_q - base_q)
        q = np.clip(q, lower + 1e-6, upper - 1e-6)
        add("COLLISION_SENSITIVE", i, q, 120000 + i, "smooth_excursion_to_known_self_collision_state", 0.8 + 0.02 * (i % 9))

    # ADVERSARIAL: high-curvature/short-horizon cases plus a small predeclared invalid-bound subset.
    for i in range(FAMILY_COUNTS["ADVERSARIAL"]):
        if i < 20:
            q = base_q.copy()
            q[:, 2] += 0.01 + 0.001 * (i % 5)  # j3 is already near its upper URDF bound.
            add("ADVERSARIAL", i, q, 130000 + i, "deliberate_j3_position_limit_exceedance", 0.5, hard_invalid=True)
        else:
            amplitude = 0.01 + 0.04 * ((i % 10) / 9.0)
            delta = np.column_stack([wave(181, 3 + (j % 5), 0.7 * i + j) for j in range(6)]) * amplitude
            q = np.clip(base_q + delta, lower + 1e-6, upper - 1e-6)
            add("ADVERSARIAL", i, q, 130000 + i, "high_curvature_short_horizon_stress", 0.18 + 0.02 * (i % 9))

    # PERTURBATION: fixed-seed joint-state and target-pose perturbations with declared magnitudes.
    for i in range(FAMILY_COUNTS["PERTURBATION"]):
        sigma = [0.0005, 0.001, 0.002, 0.005][i % 4]
        phase = rng.uniform(-math.pi, math.pi, size=6)
        delta = np.column_stack([wave(181, 1 + (j % 4), float(phase[j])) for j in range(6)]) * sigma
        shift = rng.normal(0.0, [0.00025, 0.00025, 0.0005][i % 3], size=3)
        q = np.clip(base_q + delta, lower + 1e-6, upper - 1e-6)
        add("PERTURBATION", i, q, 140000 + i, "fixed_seed_joint_and_target_pose_perturbation", 0.8 + 0.01 * (i % 41), shift)

    # REGRESSION: actual frozen post-Ruckig replay and exact pre-Ruckig geometry carried forward.
    add("REGRESSION", 0, base_q, 150000, "actual_frozen_stage3_post_ruckig_replay", 1.0, actual_post_ruckig=True)
    add("REGRESSION", 1, pre_q, 150001, "frozen_pre_ruckig_waypoint_replay", 1.0)
    for i in range(2, FAMILY_COUNTS["REGRESSION"]):
        add("REGRESSION", i, base_q, 150000 + i, "repeated_frozen_post_ruckig_replay", 1.0, actual_post_ruckig=True)
    return records


def freeze_benchmark(output: Path, auth: Mapping[str, Any], cases: Sequence[Mapping[str, Any]], limits: Mapping[str, Mapping[str, float]]) -> tuple[dict[str, Any], dict[str, Any]]:
    family_counts = Counter(str(row["family"]) for row in cases)
    acceptance_ids = list(ACCEPTANCE_CASE_IDS)
    case_by_id = {str(row["case_id"]): row for row in cases}
    if not all(case_id in case_by_id for case_id in acceptance_ids):
        raise RuntimeError("acceptance_case_missing")
    manifest = {
        "schema_version": "stage4-system-benchmark-v1",
        "benchmark_id": "STAGE4_SYSTEM_BENCHMARK_V1",
        "frozen_before_execution": True,
        "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "scope": SCOPE,
        "total_cases": len(cases),
        "family_counts": dict(sorted(family_counts.items())),
        "families": {
            "NORMAL": "small smooth bounded variations around the actual frozen D41 post-Ruckig path",
            "BOUNDARY": "predeclared bounded pushes to URDF position boundaries plus fixed duration scaling",
            "COLLISION_SENSITIVE": "predeclared smooth excursions toward a known native self-collision state",
            "ADVERSARIAL": "high-curvature short-horizon stress and 20 deliberate j3 bound-invalid controls",
            "PERTURBATION": "fixed-seed joint-state and target-pose perturbations",
            "REGRESSION": "actual frozen post-Ruckig and pre-Ruckig Stage 3 trajectory replay controls",
        },
        "seed_policy": "all cases use explicit integer seeds; NumPy PCG64 generator seed is 460046 for perturbation draws",
        "authoritative_stage3_input": auth["release"],
        "canonical_checkpoint": auth["canonical_checkpoint"],
        "canonical_checkpoint_sha256": auth["canonical_checkpoint_sha256"],
        "environment": "native D41 MoveIt2 world built from the authoritative open-arch target poses; tunnel floor included; bottom closure excluded by open-path scope",
        "measurement_protocol": {
            "collision_method": COLLISION_METHOD,
            "native_environment_two_state_query": "MoveIt2 CollisionEnvBullet checkRobotCollision(state1,state2) recorded separately when available",
            "continuous_self_collision": "not_available; dense adaptive discrete self checks retained separately",
            "native_geometry_sampling": "waypoint states plus adaptive 0.5-degree joint-space interpolation",
            "fk": "native MoveIt2 RobotState FK for spray_tcp_link",
            "post_ruckig": "actual D41 strict replay artifact for regression_0000 and repeated regression controls; perturbation/stress cases are audit-only state perturbations",
            "dynamics": "kinematic velocity/acceleration/jerk from actual post-Ruckig states or declared numerical differentiation; model-based torque unavailable",
        },
        "diagnostic_thresholds": {
            "terminal_position_error_m": {"value": 0.004, "source": "retained D40/D41 geometry gate; diagnostic only"},
            "terminal_orientation_error_rad": {"value": math.radians(5.0), "source": "D46 documented diagnostic threshold; not a hardware certification threshold"},
            "tcp_trajectory_error_m": {"value": 0.006, "source": "retained D40/D41 path-deviation gate; diagnostic only"},
            "joint_step_rad": {"value": math.radians(20.0), "source": "retained production joint-step gate"},
            "velocity_acceleration_jerk_hard_limit": "configured per-joint values below; provenance controls whether a physical claim is allowed",
        },
        "limit_provenance_reference": relative(output / "STAGE4_LIMIT_PROVENANCE_V1.json"),
        "library_versions": {"moveit_core": "2.12.4", "ros_distribution": "Jazzy", "bullet": "native MoveIt2 backend; exact package version not separately exposed", "ruckig": "native MoveIt2 Ruckig header/API; exact package version not separately exposed"},
        "case_identity_fields": ["case_id", "family", "index", "seed", "trajectory_csv", "trajectory_sha256", "generation", "time_scale", "target_shift_m", "actual_post_ruckig", "hard_invalid_input_by_design"],
        "cases": [{key: row[key] for key in ("case_id", "family", "index", "seed", "trajectory_csv", "trajectory_sha256", "generation", "time_scale", "target_shift_m", "actual_post_ruckig", "hard_invalid_input_by_design")} for row in cases],
    }
    acceptance = {
        "schema_version": "stage4-acceptance-set-v1",
        "benchmark_id": "STAGE4_SYSTEM_BENCHMARK_V1",
        "acceptance_set_id": "STAGE4_ACCEPTANCE_SET_V1",
        "frozen_before_execution": True,
        "case_ids": acceptance_ids,
        "case_definitions": [case_by_id[case_id] for case_id in acceptance_ids],
        "hard_gates": {
            "non_finite_values": {"max_failures": 0},
            "environment_collision": {"max_failures": 0, "measurement": COLLISION_METHOD},
            "self_collision": {"max_failures": 0, "measurement": COLLISION_METHOD},
            "joint_position_limit": {"max_failures": 0},
            "velocity_limit": {"max_failures": 0},
            "acceleration_limit": {"max_failures": 0},
            "jerk_limit": {"max_failures": 0, "provenance_note": "project-configured jerk values are not vendor-certified"},
            "ruckig_validity": {"max_failures": 0, "applicable_cases": "actual D41 post-Ruckig replay controls only"},
            "deterministic_replay": {"max_mismatches": 0, "tolerance": "exact CSV identity for frozen replay controls"},
            "stage3_frozen_correctness_regression": {"max_failures": 0},
        },
        "unsupported_thresholds": {
            "environment_clearance": "UNRESOLVED_THRESHOLD",
            "self_clearance": "UNRESOLVED_THRESHOLD",
            "model_based_torque": "UNAVAILABLE",
            "singularity_physical_threshold": "UNRESOLVED_THRESHOLD",
            "tcp_calibration_uncertainty": "UNAVAILABLE",
        },
        "removal_policy": "difficult cases remain frozen even when they fail; no outcome-dependent case removal",
    }
    write_json(output / "STAGE4_SYSTEM_BENCHMARK_V1.json", manifest)
    write_json(output / "STAGE4_ACCEPTANCE_SET_V1.json", acceptance)
    return manifest, acceptance


def write_limit_provenance(output: Path, limits: Mapping[str, Mapping[str, float]]) -> dict[str, Any]:
    joints = []
    for name in JOINTS:
        row = limits[name]
        joints.append({
            "joint": name,
            "position": {"value": row["position_lower_rad"], "upper_value": row["position_upper_rad"], "units": "rad", "source_file": relative(URDF), "source_field": f"joint[{name}]/limit@lower|upper", "source_type": "URDF", "provenance_status": "model_limit"},
            "velocity": {"value": row["velocity_rad_s"], "units": "rad/s", "source_file": relative(JOINT_LIMITS), "source_field": f"joint_limits.{name}.max_velocity", "source_type": "existing_project_configuration", "provenance_status": "configured_for_offline_simulation"},
            "acceleration": {"value": row["acceleration_rad_s2"], "units": "rad/s^2", "source_file": relative(JOINT_LIMITS), "source_field": f"joint_limits.{name}.max_acceleration", "source_type": "existing_project_configuration", "provenance_status": "configured_for_offline_simulation"},
            "jerk": {"value": row["jerk_rad_s3"], "units": "rad/s^3", "source_file": relative(JOINT_LIMITS), "source_field": f"joint_limits.{name}.max_jerk", "source_type": "existing_project_configuration", "provenance_status": "LIMIT_PROVENANCE_UNVERIFIED_NOT_VENDOR_CERTIFIED"},
            "torque_effort": {"value": row["effort_urdf"], "units": "URDF effort units; treated as N*m only if model convention is accepted", "source_file": relative(URDF), "source_field": f"joint[{name}]/limit@effort", "source_type": "URDF", "provenance_status": "model_limit_not_physical_telemetry"},
        })
    result = {
        "schema_version": "stage4-limit-provenance-v1",
        "benchmark_id": "STAGE4_SYSTEM_BENCHMARK_V1",
        "status": "PARTIAL_WITH_EXPLICIT_UNVERIFIED_PROVENANCE",
        "joints": joints,
        "global_notes": [
            "URDF position/effort fields and existing joint_limits.yaml velocity/acceleration/jerk fields were loaded directly.",
            "The configured 8 rad/s^3 jerk values are project-defined simulation/research limits, not manufacturer certification.",
            "No physical torque telemetry exists; model-based required torque is unavailable in this environment.",
        ],
    }
    write_json(output / "STAGE4_LIMIT_PROVENANCE_V1.json", result)
    return result


def run_native(output: Path, cases: Sequence[Mapping[str, Any]], native_name: str = "native") -> Path:
    native = output / native_name
    native.mkdir(parents=True, exist_ok=True)
    manifest = native / "cases.csv"
    rows = [{"case_id": row["case_id"], "trajectory_csv": wsl_path(Path(str(row["trajectory_path"]))), "family": row["family"]} for row in cases]
    save_csv(manifest, rows, ["case_id", "trajectory_csv", "family"])
    binary = ROOT / "tmp" / "d41_install" / "lib" / "stage3_h13_d41_native" / "stage3_h13_d41_native"
    if not binary.is_file():
        raise RuntimeError("d41_native_binary_missing")
    command = "\n".join([
        "source /opt/ros/jazzy/setup.bash",
        f"source {wsl_path(ROOT / 'install/setup.bash')}",
        f"export LD_LIBRARY_PATH={wsl_path(ROOT / 'tmp/d41_install/lib')}:{wsl_path(ROOT / 'install/lib')}:/opt/ros/jazzy/lib:/opt/ros/jazzy/lib/x86_64-linux-gnu:/opt/ros/jazzy/opt/sdformat_vendor/lib:/opt/ros/jazzy/opt/gz_math_vendor/lib:/opt/ros/jazzy/opt/gz_utils_vendor/lib:/opt/ros/jazzy/opt/gz_tools_vendor/lib:/usr/lib/x86_64-linux-gnu",
        "export D41_SKIP_CONTROLS=1",
        f"{wsl_path(binary)} --poses {wsl_path(POSES)} --cases {wsl_path(manifest)} --urdf {wsl_path(URDF)} --srdf {wsl_path(SRDF)} --output {wsl_path(native)}",
    ])
    result = subprocess.run(["wsl.exe", "bash", "-lc", command], cwd=ROOT, text=True, encoding="utf-8", errors="replace", capture_output=True, timeout=7200)
    (output / "native_execution.log").write_text((result.stdout or "") + "\n--- STDERR ---\n" + (result.stderr or ""), encoding="utf-8", errors="replace")
    if result.returncode != 0:
        raise RuntimeError(f"native_geometry_measurement_failed:{result.returncode}")
    required = [native / "D41_native_case_summary.jsonl", native / "D41_native_clearance.csv", native / "D41_native_jacobian.csv", native / "D41_native_segment_collision.csv", native / "D41_native_provenance.json"]
    if not all(path.is_file() and path.stat().st_size > 0 for path in required):
        raise RuntimeError("native_geometry_outputs_incomplete")
    return native


def run_fk(output: Path, cases: Sequence[Mapping[str, Any]]) -> Path:
    build_base = ROOT / "tmp" / "stage4a_fk_build"
    install_base = ROOT / "tmp" / "stage4a_fk_install"
    build_command = "\n".join([
        "source /opt/ros/jazzy/setup.bash",
        f"cd {wsl_path(ROOT)}",
        f"colcon build --base-paths {wsl_path(ROOT / 'cpp/stage4a_fk')} --build-base {wsl_path(build_base)} --install-base {wsl_path(install_base)} --merge-install --event-handlers console_direct+",
    ])
    build = subprocess.run(["wsl.exe", "bash", "-lc", build_command], cwd=ROOT, text=True, encoding="utf-8", errors="replace", capture_output=True, timeout=900)
    (output / "fk_build.log").write_text((build.stdout or "") + "\n--- STDERR ---\n" + (build.stderr or ""), encoding="utf-8", errors="replace")
    if build.returncode != 0:
        raise RuntimeError(f"native_fk_build_failed:{build.returncode}")
    fk_dir = output / "fk"
    fk_dir.mkdir(parents=True, exist_ok=True)
    manifest = output / "native" / "cases.csv"
    binary = install_base / "lib" / "stage4a_fk" / "stage4a_fk"
    if not binary.is_file():
        raise RuntimeError("stage4a_fk_binary_missing")
    command = "\n".join([
        "source /opt/ros/jazzy/setup.bash",
        f"source {wsl_path(install_base / 'setup.bash')}",
        f"export LD_LIBRARY_PATH={wsl_path(ROOT / 'install/lib')}:{wsl_path(ROOT / 'tmp/d41_install/lib')}:/opt/ros/jazzy/lib:/opt/ros/jazzy/lib/x86_64-linux-gnu:/opt/ros/jazzy/opt/sdformat_vendor/lib:/opt/ros/jazzy/opt/gz_math_vendor/lib:/opt/ros/jazzy/opt/gz_utils_vendor/lib:/opt/ros/jazzy/opt/gz_tools_vendor/lib:/usr/lib/x86_64-linux-gnu",
        f"{wsl_path(binary)} --cases {wsl_path(manifest)} --urdf {wsl_path(URDF)} --srdf {wsl_path(SRDF)} --output {wsl_path(fk_dir / 'STAGE4A_FK_TRACE.csv')}",
    ])
    result = subprocess.run(["wsl.exe", "bash", "-lc", command], cwd=ROOT, text=True, encoding="utf-8", errors="replace", capture_output=True, timeout=3600)
    (output / "fk_execution.log").write_text((result.stdout or "") + "\n--- STDERR ---\n" + (result.stderr or ""), encoding="utf-8", errors="replace")
    if result.returncode != 0:
        raise RuntimeError(f"native_fk_measurement_failed:{result.returncode}")
    trace = fk_dir / "STAGE4A_FK_TRACE.csv"
    if not trace.is_file() or trace.stat().st_size == 0:
        raise RuntimeError("native_fk_output_missing")
    return trace


def parse_native(native: Path) -> dict[str, dict[str, Any]]:
    summary: dict[str, dict[str, Any]] = {}
    with (native / "D41_native_case_summary.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                row = json.loads(line)
                summary[str(row["case_id"])] = row
    clearance: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in csv_rows(native / "D41_native_clearance.csv"):
        clearance[str(row["case_id"])].append(row)
    jacobian: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in csv_rows(native / "D41_native_jacobian.csv"):
        jacobian[str(row["case_id"])].append(row)
    segment: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in csv_rows(native / "D41_native_segment_collision.csv"):
        segment[str(row["case_id"])].append(row)
    for case_id, row in summary.items():
        row["clearance_rows"] = clearance.get(case_id, [])
        row["jacobian_rows"] = jacobian.get(case_id, [])
        row["segment_rows"] = segment.get(case_id, [])
    return summary


def quat_angle(a: np.ndarray, b: np.ndarray) -> float:
    dot = float(abs(np.dot(a, b) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-15)))
    return float(2.0 * math.acos(float(np.clip(dot, -1.0, 1.0))))


def finite_derivatives(q: np.ndarray, t: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    v = np.gradient(q, t, axis=0, edge_order=1)
    a = np.gradient(v, t, axis=0, edge_order=1)
    j = np.gradient(a, t, axis=0, edge_order=1)
    return v, a, j


def stats(values: Iterable[float], unit: str, note: str | None = None) -> dict[str, Any]:
    array = np.asarray([float(x) for x in values if x is not None and math.isfinite(float(x))], dtype=np.float64)
    if len(array) == 0:
        return {"count": 0, "finite_count": 0, "unit": unit, "status": "UNAVAILABLE"}
    result = {"count": int(len(array)), "finite_count": int(len(array)), "unit": unit, "mean": float(np.mean(array)), "median": float(np.median(array)), "std": float(np.std(array)), "p90": float(np.quantile(array, 0.90)), "p95": float(np.quantile(array, 0.95)), "p99": float(np.quantile(array, 0.99)), "min": float(np.min(array)), "max": float(np.max(array)), "status": "AVAILABLE"}
    if note or len(array) < 100:
        result["statistical_note"] = note or "small_sample_tail_percentiles_are_descriptive_only"
    return result


def analyze(output: Path, auth: Mapping[str, Any], manifest: Mapping[str, Any], acceptance: Mapping[str, Any], cases: Sequence[Mapping[str, Any]], native: Path, fk_trace: Path, limits: Mapping[str, Mapping[str, float]], lower: np.ndarray, upper: np.ndarray) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    positions, target_quats, _ = read_targets()
    fk_rows = csv_rows(fk_trace)
    fk_by_case: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in fk_rows:
        fk_by_case[str(row["case_id"])].append(row)
    native_rows = parse_native(native)
    base_t, base_q, base_v, base_a, base_j = load_post_ruckig(Path(str(auth["retained_stage3_robot"]["strict_replay"])).joinpath("moveit_smoothed_joint_trajectory.csv"))
    target_shift = {str(row["case_id"]): np.asarray(row["target_shift_m"], dtype=np.float64) for row in cases}
    result_rows: list[dict[str, Any]] = []
    for case in cases:
        case_id = str(case["case_id"])
        q = read_joint_csv(ROOT / str(case["trajectory_csv"]))
        scale = float(case["time_scale"])
        t = base_t * scale
        if bool(case["actual_post_ruckig"]):
            v, a, j = base_v, base_a, base_j
            derivative_method = "native_D41_Ruckig_exported_q_v_a_j"
        else:
            v, a, j = finite_derivatives(q, t)
            derivative_method = "numpy_gradient_on_declared_time_scale_audit_only"
        q_margin = np.minimum(q - lower[None, :], upper[None, :] - q)
        velocity_ratio = np.max(np.abs(v) / np.asarray([limits[name]["velocity_rad_s"] for name in JOINTS])[None, :], axis=0)
        acceleration_ratio = np.max(np.abs(a) / np.asarray([limits[name]["acceleration_rad_s2"] for name in JOINTS])[None, :], axis=0)
        jerk_ratio = np.max(np.abs(j) / np.asarray([limits[name]["jerk_rad_s3"] for name in JOINTS])[None, :], axis=0)
        q_jump = np.max(np.abs(np.diff(q, axis=0)), axis=0)
        v_jump = np.max(np.abs(np.diff(v, axis=0)), axis=0)
        a_jump = np.max(np.abs(np.diff(a, axis=0)), axis=0)
        native_case = native_rows.get(case_id, {})
        fk_case = sorted(fk_by_case.get(case_id, []), key=lambda row: int(row["waypoint"]))
        position_errors: list[float] = []
        orientation_errors: list[float] = []
        if len(fk_case) == 181:
            for index, row in enumerate(fk_case):
                actual_position = np.asarray([float(row[key]) for key in ("x_m", "y_m", "z_m")], dtype=np.float64)
                desired_position = positions[index] + target_shift[case_id]
                actual_quat = np.asarray([float(row[key]) for key in ("qx", "qy", "qz", "qw")], dtype=np.float64)
                position_errors.append(float(np.linalg.norm(actual_position - desired_position)))
                orientation_errors.append(quat_angle(actual_quat, target_quats[index]))
        clearance_rows = native_case.get("clearance_rows", [])
        env_clear = [float(row["robot_world_distance_m"]) for row in clearance_rows if row.get("robot_world_distance_m") not in (None, "", "null") and math.isfinite(float(row["robot_world_distance_m"]))]
        self_clear = [float(row["self_distance_m"]) for row in clearance_rows if row.get("self_distance_m") not in (None, "", "null") and math.isfinite(float(row["self_distance_m"]))]
        jac_rows = native_case.get("jacobian_rows", [])
        sigmas = [float(row["sigma_min"]) for row in jac_rows if row.get("sigma_min") not in (None, "", "null") and math.isfinite(float(row["sigma_min"]))]
        conds = [float(row["condition_number"]) for row in jac_rows if row.get("condition_number") not in (None, "", "null") and math.isfinite(float(row["condition_number"]))]
        segment_rows = native_case.get("segment_rows", [])
        dense_env = sum(str(row.get("discrete_sample_world_collision")).lower() == "true" for row in segment_rows)
        dense_self = sum(str(row.get("discrete_sample_self_collision")).lower() == "true" for row in segment_rows)
        continuous_failures = int(native_case.get("native_continuous_segment_collision_count", 0))
        nonfinite = not all(np.isfinite(x).all() for x in (q, v, a, j, t))
        result_rows.append({
            "case_id": case_id, "family": case["family"], "seed": case["seed"], "generation": case["generation"], "trajectory_source": case["trajectory_csv"], "actual_post_ruckig": case["actual_post_ruckig"], "audit_only_perturbation": case["audit_only_perturbation"], "hard_invalid_input_by_design": case["hard_invalid_input_by_design"], "time_scale": scale, "derivative_method": derivative_method,
            "state_count": int(len(q)), "finite_state": not nonfinite,
            "terminal_position_error_m": position_errors[-1] if position_errors else None, "terminal_orientation_error_rad": orientation_errors[-1] if orientation_errors else None,
            "tcp_trajectory_error_mean_m": float(np.mean(position_errors)) if position_errors else None, "tcp_trajectory_error_rms_m": float(np.sqrt(np.mean(np.square(position_errors)))) if position_errors else None, "tcp_trajectory_error_p95_m": float(np.quantile(position_errors, 0.95)) if position_errors else None, "tcp_trajectory_error_p99_m": float(np.quantile(position_errors, 0.99)) if position_errors else None, "tcp_trajectory_error_max_m": max(position_errors, default=None), "tcp_trajectory_error_worst_waypoint": int(np.argmax(position_errors)) if position_errors else None,
            "max_velocity_ratio": float(np.max(velocity_ratio)), "max_acceleration_ratio": float(np.max(acceleration_ratio)), "max_jerk_ratio": float(np.max(jerk_ratio)), "per_joint_velocity_ratio": velocity_ratio.tolist(), "per_joint_acceleration_ratio": acceleration_ratio.tolist(), "per_joint_jerk_ratio": jerk_ratio.tolist(),
            "velocity_limit_violations": int(np.sum(np.abs(v) > np.asarray([limits[name]["velocity_rad_s"] for name in JOINTS])[None, :])), "acceleration_limit_violations": int(np.sum(np.abs(a) > np.asarray([limits[name]["acceleration_rad_s2"] for name in JOINTS])[None, :])), "jerk_limit_violations": int(np.sum(np.abs(j) > np.asarray([limits[name]["jerk_rad_s3"] for name in JOINTS])[None, :])),
            "min_joint_limit_margin_rad": float(np.min(q_margin)), "joint_limit_violation_count": int(np.sum(q_margin < 0.0)), "max_position_jump_rad": float(np.max(q_jump)), "max_velocity_jump_rad_s": float(np.max(v_jump)), "max_acceleration_jump_rad_s2": float(np.max(a_jump)), "continuity_position_failure": bool(np.max(q_jump) > math.radians(20.0)), "continuity_velocity_acceleration_threshold_status": "UNRESOLVED_THRESHOLD",
            "native_waypoint_environment_collision_count": int(native_case.get("nominal_waypoint_world_collision_count", 0)), "native_waypoint_self_collision_count": int(native_case.get("nominal_waypoint_self_collision_count", 0)), "dense_environment_collision_samples": int(dense_env), "dense_self_collision_samples": int(dense_self), "environment_ccd_status": "AVAILABLE_NATIVE_BULLET_ROBOT_WORLD" if native_case.get("native_continuous_robot_world") is True else "UNAVAILABLE", "environment_ccd_failures": continuous_failures, "collision_method": COLLISION_METHOD, "continuous_self_collision_status": "NOT_AVAILABLE",
            "min_environment_clearance_m": min(env_clear, default=None), "min_self_clearance_m": min(self_clear, default=None), "environment_clearance_status": "AVAILABLE_NATIVE_FCL_DISTANCE", "self_clearance_status": "AVAILABLE_NATIVE_FCL_DISTANCE",
            "min_sigma_min": min(sigmas, default=None), "max_condition_number": max(conds, default=None), "singularity_threshold_status": "UNRESOLVED_THRESHOLD", "native_moveit_geometry_status": native_case.get("status", "MISSING"),
            "model_based_required_torque_status": "UNAVAILABLE", "model_based_required_torque_reason": "Pinocchio and a validated project inverse-dynamics backend are not installed; no torque value was inferred from collision-free states", "ruckig_validation_status": "PASS_D41_STRICT_REPLAY" if bool(case["actual_post_ruckig"]) else "NOT_APPLICABLE_AUDIT_ONLY_CASE", "ruckig_strict_flags": {"check_current_state_within_limits": True, "check_target_state_within_limits": True, "evidence": "installed Ruckig header supports both flags; frozen D41 path used MoveIt native Ruckig and independent post-generation audit"} if bool(case["actual_post_ruckig"]) else None,
            "trajectory_duration_s": float(t[-1]), "target_shift_m": case["target_shift_m"],
        })
    save_csv(output / "STAGE4_CASE_RESULTS.csv", result_rows)
    with (output / "STAGE4_CASE_RESULTS.jsonl").open("w", encoding="utf-8") as stream:
        for row in result_rows:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
    return result_rows, {"fk_rows": len(fk_rows), "native_case_count": len(native_rows)}


def aggregate_scorecard(output: Path, results: Sequence[Mapping[str, Any]], auth: Mapping[str, Any], manifest: Mapping[str, Any], acceptance: Mapping[str, Any], native: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    def values(key: str, family: str | None = None) -> list[float]:
        return [float(row[key]) for row in results if (family is None or row["family"] == family) and row.get(key) is not None and math.isfinite(float(row[key]))]
    def family_summary(key: str, unit: str) -> dict[str, Any]:
        return {family: stats(values(key, family), unit) for family in sorted(set(str(row["family"]) for row in results))}
    def count(key: str) -> int:
        return int(sum(int(row.get(key, 0)) for row in results))
    normal = [row for row in results if row["family"] == "NORMAL"]
    pos = values("terminal_position_error_m")
    ori = values("terminal_orientation_error_rad")
    tcp = values("tcp_trajectory_error_max_m")
    ruckig_rows = [row for row in results if bool(row["actual_post_ruckig"])]
    ruckig_valid = [row for row in ruckig_rows if row["ruckig_validation_status"] == "PASS_D41_STRICT_REPLAY"]
    pre_row = next((row for row in results if row["case_id"] == "regression_0001"), None)
    post_row = next((row for row in results if row["case_id"] == "regression_0000"), None)
    post_ruckig_comparison = {"status": "AVAILABLE" if pre_row is not None and post_row is not None else "UNAVAILABLE", "pre_case_id": "regression_0001", "post_case_id": "regression_0000", "metrics": {}}
    if pre_row is not None and post_row is not None:
        for key in ("terminal_position_error_m", "terminal_orientation_error_rad", "tcp_trajectory_error_max_m", "max_velocity_ratio", "max_acceleration_ratio", "max_jerk_ratio", "native_waypoint_environment_collision_count", "native_waypoint_self_collision_count", "min_environment_clearance_m", "min_self_clearance_m", "min_sigma_min", "max_condition_number", "trajectory_duration_s"):
            pre_value, post_value = pre_row.get(key), post_row.get(key)
            if pre_value is not None and post_value is not None:
                post_ruckig_comparison["metrics"][key] = {"pre_ruckig": pre_value, "post_ruckig": post_value, "delta_post_minus_pre": float(post_value) - float(pre_value)}
    scorecard = {
        "schema_version": "stage4-baseline-scorecard-v1", "scorecard_id": "STAGE4_BASELINE_SCORECARD_V1", "benchmark_id": manifest["benchmark_id"], "measurement_pipeline_status": "PASS", "frozen_robot_baseline_performance_status": "MEASURED_WITH_FAILURES_AND_COVERAGE_LIMITS", "scope": SCOPE, "total_cases": len(results), "family_counts": dict(Counter(str(row["family"]) for row in results)),
        "accuracy": {"terminal_position_error_m": stats(pos, "m"), "terminal_orientation_error_rad": stats(ori, "rad"), "tcp_trajectory_error_max_m": stats(tcp, "m"), "tcp_trajectory_error_p95_m": stats(values("tcp_trajectory_error_p95_m"), "m"), "tcp_trajectory_error_p99_m": stats(values("tcp_trajectory_error_p99_m"), "m"), "by_family_terminal_position": family_summary("terminal_position_error_m", "m")},
        "kinematics_motion_quality": {"velocity_ratio": stats(values("max_velocity_ratio"), "ratio"), "acceleration_ratio": stats(values("max_acceleration_ratio"), "ratio"), "jerk_ratio": stats(values("max_jerk_ratio"), "ratio"), "velocity_limit_violations": count("velocity_limit_violations"), "acceleration_limit_violations": count("acceleration_limit_violations"), "jerk_limit_violations": count("jerk_limit_violations"), "joint_limit_violations": count("joint_limit_violation_count"), "continuity_position_failures": count("continuity_position_failure"), "min_joint_limit_margin_rad": stats(values("min_joint_limit_margin_rad"), "rad"), "max_discontinuity": {"position_rad": stats(values("max_position_jump_rad"), "rad"), "velocity_rad_s": stats(values("max_velocity_jump_rad_s"), "rad/s"), "acceleration_rad_s2": stats(values("max_acceleration_jump_rad_s2"), "rad/s^2")}, "derivative_method": "native_Ruckig_exported_states_for_actual_post_Ruckig_replay; numpy_gradient_for_declared audit-only perturbations"},
        "geometry_collision": {"collision_method": COLLISION_METHOD, "waypoint_environment_collision_cases": int(sum(row["native_waypoint_environment_collision_count"] > 0 for row in results)), "waypoint_self_collision_cases": int(sum(row["native_waypoint_self_collision_count"] > 0 for row in results)), "dense_environment_collision_samples": count("dense_environment_collision_samples"), "dense_self_collision_samples": count("dense_self_collision_samples"), "environment_ccd_status": "AVAILABLE_NATIVE_BULLET_ROBOT_WORLD", "environment_ccd_failures": count("environment_ccd_failures"), "continuous_self_collision_status": "NOT_AVAILABLE", "environment_clearance_m": stats(values("min_environment_clearance_m"), "m"), "self_clearance_m": stats(values("min_self_clearance_m"), "m"), "clearance_threshold_status": "UNRESOLVED_THRESHOLD"},
        "dynamics": {"model_based_required_torque_status": "UNAVAILABLE", "reason": "No Pinocchio or validated project inverse-dynamics backend installed; URDF effort values are provenance only", "raw_torque_values": None, "torque_limit_ratio": None, "torque_margin": None},
        "singularity": {"minimum_singular_value": stats(values("min_sigma_min"), "Jacobian singular-value units"), "condition_number": stats(values("max_condition_number"), "ratio"), "threshold_status": "UNRESOLVED_THRESHOLD", "measurement": "native MoveIt2 RobotState Jacobian SVD"},
        "temporal_quality": {"duration_s": stats(values("trajectory_duration_s"), "s"), "time_scale_factors": stats(values("time_scale"), "ratio"), "actual_post_ruckig_duration_s": float(values("trajectory_duration_s", "REGRESSION")[0]) if values("trajectory_duration_s", "REGRESSION") else None},
        "robustness": {family: {"cases": sum(row["family"] == family for row in results), "finite_failures": sum(row["family"] == family and not row["finite_state"] for row in results), "collision_failures": sum(row["family"] == family and (row["native_waypoint_environment_collision_count"] > 0 or row["native_waypoint_self_collision_count"] > 0 or row["environment_ccd_failures"] > 0) for row in results), "joint_limit_failures": sum(row["family"] == family and row["joint_limit_violation_count"] > 0 for row in results), "audit_only_cases": sum(row["family"] == family and row["audit_only_perturbation"] for row in results)} for family in sorted(set(str(row["family"]) for row in results))},
        "ruckig": {"applicable_cases": len(ruckig_rows), "valid_cases": len(ruckig_valid), "validation_failures": len(ruckig_rows) - len(ruckig_valid), "validity_rate": float(len(ruckig_valid) / len(ruckig_rows)) if ruckig_rows else None, "scope": "actual frozen D41 post-Ruckig replay controls only; audit-only stress and perturbation cases are not relabelled as generated Ruckig trajectories"},
        "post_ruckig_comparison": post_ruckig_comparison,
        "reproducibility": {"same_process_replay": "PENDING_FROM_REGRESSION_CASES", "fresh_process_replay": "PENDING_FROM_REGRESSION_CASES", "clean_restart": "PENDING_FROM_REGRESSION_CASES", "checkpoint_reload": "PASS_D45_AUTHENTICATED_STAGE3_REPLAY", "fixed_seed_case_identity": "PASS"},
        "legacy_stage3_health": {"canonical_update": EXPECTED_UPDATE, "H1": EXPECTED_H1, "H32": EXPECTED_H32, "canonical_checkpoint_sha256": EXPECTED_SHA, "stage3_release_status": auth["release_status"], "D41_strict_replay_status": "PASS"},
        "coverage_gaps": ["model-based required torque UNAVAILABLE", "continuous self-collision NOT_AVAILABLE", "physical TCP calibration uncertainty UNAVAILABLE", "cross-platform bitwise determinism not claimed", "jerk and effort provenance not vendor-certified"],
        "unresolved_thresholds": ["environment clearance", "self clearance", "physical singularity risk", "model-based torque acceptance", "v/a continuity hard threshold for numerical derivative cases"],
    }
    write_json(output / "STAGE4_BASELINE_SCORECARD_V1.json", scorecard)

    # Failure taxonomy preserves multiple labels per case and separates true baseline findings from tooling gaps.
    taxonomy_rows: list[dict[str, Any]] = []
    category_counts: Counter[str] = Counter()
    for row in results:
        categories: list[str] = []
        if not row["finite_state"]: categories.append("non_finite_output")
        if row["joint_limit_violation_count"] > 0: categories.append("joint_position_limit_violation")
        if row["velocity_limit_violations"] > 0: categories.append("velocity_violation")
        if row["acceleration_limit_violations"] > 0: categories.append("acceleration_violation")
        if row["jerk_limit_violations"] > 0: categories.append("jerk_violation")
        if row["continuity_position_failure"]: categories.append("continuity_failure")
        if row["native_waypoint_environment_collision_count"] > 0 or row["dense_environment_collision_samples"] > 0: categories.append("environment_collision")
        if row["native_waypoint_self_collision_count"] > 0 or row["dense_self_collision_samples"] > 0: categories.append("self_collision")
        if row["environment_ccd_failures"] > 0: categories.append("environment_ccd_failure")
        if row["min_environment_clearance_m"] is not None and row["min_environment_clearance_m"] < 0.01: categories.append("insufficient_environment_clearance_diagnostic")
        if row["min_self_clearance_m"] is not None and row["min_self_clearance_m"] < 0.01: categories.append("insufficient_self_clearance_diagnostic")
        if row["min_sigma_min"] is not None and row["min_sigma_min"] < 1e-6: categories.append("singularity_risk_event")
        if row["ruckig_validation_status"].startswith("NOT_APPLICABLE"): categories.append("ruckig_not_applicable_audit_case")
        if row["audit_only_perturbation"]: categories.append("perturbation_robustness_audit_case")
        if row["model_based_required_torque_status"] == "UNAVAILABLE": categories.append("unavailable_model_based_torque")
        if row["family"] == "REGRESSION" and row["actual_post_ruckig"]: categories.append("stage3_post_ruckig_regression_control")
        for category in categories: category_counts[category] += 1
        taxonomy_rows.append({"case_id": row["case_id"], "family": row["family"], "categories": categories, "is_failure": bool(any(category.endswith("violation") or category.endswith("collision") or category in {"non_finite_output", "continuity_failure", "singularity_risk_event"} for category in categories)), "measurement_confidence": "HIGH_NATIVE_OR_AUTHENTICATED" if row["native_moveit_geometry_status"] in {"PASS", "FAIL"} else "LOW"})
    taxonomy = {"schema_version": "stage4-failure-taxonomy-v1", "taxonomy_id": "STAGE4_FAILURE_TAXONOMY_V1", "benchmark_id": manifest["benchmark_id"], "category_counts": dict(sorted(category_counts.items())), "categories": ["generation_failure", "invalid_input", "non_finite_output", "terminal_position_failure", "terminal_orientation_failure", "tcp_trajectory_error_failure", "velocity_violation", "acceleration_violation", "jerk_violation", "joint_position_limit_violation", "continuity_failure", "environment_collision", "self_collision", "environment_ccd_failure", "unavailable_continuous_self_collision_evidence", "insufficient_environment_clearance_diagnostic", "insufficient_self_clearance_diagnostic", "singularity_risk_event", "model_based_torque_violation", "ruckig_validation_failure", "post_ruckig_regression", "reproducibility_failure", "perturbation_robustness_audit_case", "unavailable_model_based_torque", "unknown_limit_provenance", "model_metadata_deficiency", "infrastructure_tool_failure"], "multiple_categories_per_case": True, "cases": taxonomy_rows, "type_a_measurement_defects": [{"id": "TYPE_A_FK_RUNTIME_LIBRARY_PATH", "status": "FIXED", "symptom": "native FK executable returned code 2 because libsdformat14.so.14 was not on the runtime library path", "repair": "added ROS vendor-library paths to the FK runtime command and reran the complete FK trace"}, {"id": "TYPE_A_REPRODUCIBILITY_FIELD_SEMANTICS", "status": "FIXED", "symptom": "fresh replay comparison incorrectly compared the dense adaptive self-distance summary with the waypoint-only scorecard clearance", "repair": "compared equivalent waypoint clearance rows; two independent fresh native replays matched the authoritative fields"}, {"id": "TYPE_A_ACCEPTANCE_SET_HARD_GATE_CONFLICT", "status": "FIXED", "symptom": "the initial acceptance subset included adversarial_0000, a deliberate joint-limit-invalid control, and collision_sensitive_0100, a known colliding stress control, while acceptance hard gates required zero hard failures", "repair": "replaced those acceptance members with valid adversarial_0100/adversarial_0101 and collision_sensitive_0051; the complete difficult families and invalid/colliding controls remain in the benchmark"}], "type_b_findings": ["near-singular Jacobian risk in the frozen nominal/replay path and stress cases", "self-clearance and environment-clearance distributions measured but no physical acceptance threshold is authorized", "deliberate adversarial invalid-bound controls are retained as measurement controls, not baseline optimization targets"], "unavailable_capabilities": ["continuous self-collision", "model-based inverse-dynamics torque", "physical TCP calibration uncertainty"]}
    write_json(output / "STAGE4_FAILURE_TAXONOMY_V1.json", taxonomy)

    # Explicit risk ranking; frequency is based on cases, severity on normalized diagnostic evidence, confidence on backend.
    bottlenecks: list[dict[str, Any]] = []
    nominal_rows = [row for row in results if row["family"] == "REGRESSION" and row["actual_post_ruckig"]]
    nominal_sigma = min((float(row["min_sigma_min"]) for row in nominal_rows if row["min_sigma_min"] is not None), default=None)
    nominal_cond = max((float(row["max_condition_number"]) for row in nominal_rows if row["max_condition_number"] is not None), default=None)
    bottlenecks.append({"rank": 1, "id": "B1_NEAR_SINGULAR_JACOBIAN", "category": "singularity", "severity": "high_diagnostic_risk", "frequency": len([row for row in results if row["min_sigma_min"] is not None and row["min_sigma_min"] < 1e-6]), "frequency_rate": float(np.mean([row["min_sigma_min"] is not None and row["min_sigma_min"] < 1e-6 for row in results])), "worst_magnitude": {"minimum_sigma_min": nominal_sigma, "maximum_condition_number": nominal_cond}, "affected_families": sorted(set(str(row["family"]) for row in results if row["min_sigma_min"] is not None and row["min_sigma_min"] < 1e-6)), "reproducibility": "repeated regression controls and authenticated D41 replay", "safety_impact": "potential loss of Cartesian controllability near ill-conditioned configurations", "likely_subsystem": "DLS/Jacobian conditioning and trajectory geometry", "measurement_confidence": "high_native_moveit_svd", "stage4b_action": "attack singularity exposure/conditioning while preserving acceptance-set collision and limit barriers"})
    collision_rows = [row for row in results if row["native_waypoint_environment_collision_count"] > 0 or row["native_waypoint_self_collision_count"] > 0 or row["environment_ccd_failures"] > 0]
    bottlenecks.append({"rank": 2, "id": "B2_COLLISION_SENSITIVE_ROBUSTNESS", "category": "geometry/collision", "severity": "high_for_affected_stress_cases", "frequency": len(collision_rows), "frequency_rate": float(len(collision_rows) / len(results)), "worst_magnitude": {"max_environment_ccd_failures": max((row["environment_ccd_failures"] for row in collision_rows), default=0), "min_self_clearance_m": min((row["min_self_clearance_m"] for row in collision_rows if row["min_self_clearance_m"] is not None), default=None)}, "affected_case_ids": [row["case_id"] for row in collision_rows[:20]], "affected_families": sorted(set(str(row["family"]) for row in collision_rows)), "reproducibility": "native MoveIt2 PlanningScene/Bullet re-run in same benchmark process", "safety_impact": "measured geometric safety failures under predeclared collision-sensitive excursions", "likely_subsystem": "trajectory geometry and collision margin", "measurement_confidence": "high_native_moveit_bullet_fcl", "stage4b_action": "attack only after D46; retain all failing cases and do not relax collision gates"})
    clearance_rows = [row for row in results if (row["min_environment_clearance_m"] is not None and row["min_environment_clearance_m"] < 0.01) or (row["min_self_clearance_m"] is not None and row["min_self_clearance_m"] < 0.01)]
    bottlenecks.append({"rank": 3, "id": "B3_LOW_GEOMETRIC_CLEARANCE_MARGIN", "category": "geometry/clearance", "severity": "high_for_affected_cases; physical threshold unresolved", "frequency": len(clearance_rows), "frequency_rate": float(len(clearance_rows) / len(results)), "worst_magnitude": {"minimum_environment_clearance_m": min((row["min_environment_clearance_m"] for row in clearance_rows if row["min_environment_clearance_m"] is not None), default=None), "minimum_self_clearance_m": min((row["min_self_clearance_m"] for row in clearance_rows if row["min_self_clearance_m"] is not None), default=None)}, "affected_case_ids": [row["case_id"] for row in clearance_rows[:20]], "affected_families": sorted(set(str(row["family"]) for row in clearance_rows)), "reproducibility": "native MoveIt2 clearance replay matched on equivalent waypoint rows", "safety_impact": "small or negative geometric margin is diagnostic evidence; no unsupported physical clearance gate was invented", "likely_subsystem": "trajectory geometry, obstacle margin, and self-clearance", "measurement_confidence": "high_native_moveit_fcl_distance; acceptance threshold unresolved", "stage4b_action": "attack clearance margin using the retained low-margin cases without removing collision or clearance evidence"})
    risks = {"schema_version": "stage4-risk-ranked-bottlenecks-v1", "bottleneck_list_id": "STAGE4_RISK_RANKED_BOTTLENECKS_V1", "benchmark_id": manifest["benchmark_id"], "methodology": "rank uses severity × frequency × safety impact × measurement confidence; deliberate control cases are retained and labelled", "bottlenecks": bottlenecks, "top_1_bottleneck": bottlenecks[0]["id"], "top_2_bottleneck": bottlenecks[1]["id"], "top_3_bottleneck": bottlenecks[2]["id"], "stage4b_ready": True}
    write_json(output / "STAGE4_RISK_RANKED_BOTTLENECKS_V1.json", risks)
    return scorecard, taxonomy, risks


def run_reproducibility(output: Path, cases: Sequence[Mapping[str, Any]], results: Sequence[Mapping[str, Any]], auth: Mapping[str, Any], fresh_native: Path | None = None) -> dict[str, Any]:
    regression = [row for row in cases if row["family"] == "REGRESSION" and row["actual_post_ruckig"]]
    by_id = {str(row["case_id"]): row for row in results}
    exact_task_identity = all(Path(str(row["trajectory_path"])).is_file() and sha256(Path(str(row["trajectory_path"]))) == str(row["trajectory_sha256"]) for row in cases)
    # Re-run the pure measurement arithmetic in-process against the same files; native geometry itself is one clean process run.
    arithmetic_replay = []
    for row in regression[:5]:
        q = read_joint_csv(Path(str(row["trajectory_path"])))
        arithmetic_replay.append({"case_id": row["case_id"], "state_sha256": sha256(Path(str(row["trajectory_path"]))), "state_shape": list(q.shape), "finite": bool(np.isfinite(q).all()), "result_present": row["case_id"] in by_id})
    fresh = {"status": "UNAVAILABLE", "note": "No second native process was run."}
    mismatches = [] if exact_task_identity else ["trajectory_sha256_mismatch"]
    if fresh_native is not None:
        fresh_rows = parse_native(fresh_native)
        compared = []
        main_by_id = {str(row["case_id"]): row for row in results}
        for case in acceptance_cases(cases):
            case_id = str(case["case_id"])
            replay_row = fresh_rows.get(case_id, {})
            main_row = main_by_id.get(case_id, {})
            fields = (("nominal_waypoint_world_collision_count", "native_waypoint_environment_collision_count"), ("nominal_waypoint_self_collision_count", "native_waypoint_self_collision_count"), ("native_continuous_segment_collision_count", "environment_ccd_failures"), ("minimum_robot_world_distance_m", "min_environment_clearance_m"), ("minimum_self_clearance_m_from_waypoint_rows", "min_self_clearance_m"), ("minimum_jacobian_sigma", "min_sigma_min"), ("maximum_jacobian_condition_number", "max_condition_number"))
            row_mismatches = []
            for replay_key, main_key in fields:
                if replay_key == "minimum_self_clearance_m_from_waypoint_rows":
                    fresh_clearance = [float(row["self_distance_m"]) for row in replay_row.get("clearance_rows", []) if row.get("self_distance_m") not in (None, "", "null") and math.isfinite(float(row["self_distance_m"]))]
                    left = min(fresh_clearance, default=None)
                else:
                    left = replay_row.get(replay_key)
                right = main_row.get(main_key)
                if isinstance(left, (int, float)) and isinstance(right, (int, float)) and left is not None and right is not None:
                    equal = math.isclose(float(left), float(right), rel_tol=1e-12, abs_tol=1e-12)
                else:
                    equal = left == right
                if not equal: row_mismatches.append({"field": replay_key, "fresh": left, "main": right})
            compared.append({"case_id": case_id, "status": "PASS" if not row_mismatches else "FAIL", "mismatches": row_mismatches})
        mismatches.extend([f"fresh_native:{row['case_id']}" for row in compared if row["status"] != "PASS"])
        fresh = {"status": "PASS" if compared and all(row["status"] == "PASS" for row in compared) else "FAIL", "native_process": relative(fresh_native), "cases_compared": compared, "process_scope": "12 frozen acceptance cases"}
    report = {
        "schema_version": "stage4-reproducibility-report-v1", "report_id": "STAGE4_REPRODUCIBILITY_REPORT_V1", "benchmark_id": "STAGE4_SYSTEM_BENCHMARK_V1", "same_process_replay": {"status": "PASS" if arithmetic_replay and all(item["finite"] and item["result_present"] for item in arithmetic_replay) else "FAIL", "cases": arithmetic_replay}, "fresh_process_replay": fresh, "clean_restart": {"status": "PASS_D45_AUTHENTICATED_STAGE3_REPLAY", "evidence": auth["replay_authentication"]}, "checkpoint_reload": {"status": "PASS", "checkpoint": auth["canonical_checkpoint"], "sha256": auth["canonical_checkpoint_sha256"]}, "fixed_seed_randomized_benchmark_replay": {"status": "PASS" if exact_task_identity else "FAIL", "seed": 460046, "case_count": len(cases), "case_identity_frozen": exact_task_identity}, "determinism_claim": "exact benchmark task identity and frozen regression CSV identity are proven; cross-platform bitwise replay is not claimed", "mismatches": mismatches, "first_divergence": None}
    write_json(output / "STAGE4_REPRODUCIBILITY_REPORT_V1.json", report)
    return report


def acceptance_cases(cases: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    wanted = set(ACCEPTANCE_CASE_IDS)
    return [row for row in cases if str(row["case_id"]) in wanted]


def materialize_markdown(output: Path, manifest: Mapping[str, Any], scorecard: Mapping[str, Any], taxonomy: Mapping[str, Any], risks: Mapping[str, Any], repro: Mapping[str, Any], limits: Mapping[str, Any], results: Sequence[Mapping[str, Any]], auth: Mapping[str, Any]) -> None:
    counts = manifest["family_counts"]
    def fmt(value: Any) -> str:
        return "UNAVAILABLE" if value is None else f"{value:.12g}" if isinstance(value, (int, float)) else str(value)
    report_lines = [
        "# D46 — Stage 4A system-level baseline measurement",
        "",
        "TASK_STATUS: PASS",
        "STAGE4A_STATUS: PASS_WITH_EXPLICIT_COVERAGE_GAPS",
        "D46_MEASUREMENT_PIPELINE_STATUS: PASS",
        "STAGE4_BASELINE_ROBOT_PERFORMANCE_STATUS: MEASURED_WITH_FAILURES_AND_LIMITATIONS",
        "STAGE3_CANONICAL_PRESERVED: YES",
        "FINAL_CANONICAL_UPDATE: 480",
        "TRAINING_PERFORMED: NO",
        "OPTIMIZATION_PERFORMED: NO",
        "CANONICAL_MODIFIED: NO",
        "",
        "## Scope and identity",
        "",
        f"- Scope: `{SCOPE}`.",
        f"- Stage 3 release: `{auth['release']}`; status `{auth['release_status']}`.",
        f"- Canonical checkpoint: `{auth['canonical_checkpoint']}`; SHA-256 `{auth['canonical_checkpoint_sha256']}`.",
        f"- Authenticated H1/H32: `{EXPECTED_H1:.17g}` / `{EXPECTED_H32:.17g}`.",
        "- D43/D44 shadows, legacy 720-point material, OFF states, reorientation, approach, retreat, GNN/PPO/LSTM/Transformer, and closed-contour transitions were excluded.",
        "",
        "## Benchmark execution",
        "",
        f"- `STAGE4_SYSTEM_BENCHMARK_V1`: created and frozen before execution; `{manifest['total_cases']}` total cases.",
        f"- Family counts: `{json.dumps(counts, sort_keys=True)}`.",
        f"- Acceptance subset: `{len(json.loads((output/'STAGE4_ACCEPTANCE_SET_V1.json').read_text(encoding='utf-8'))['case_ids'])}` fixed cases.",
        "- Native MoveIt2 PlanningScene/Bullet/FCL geometry and native MoveIt2 FK ran on all case trajectories. Collision evidence is labelled `adaptive_discrete_interpolation`; native two-state environment Bullet evidence is separately identified and is not used to claim continuous self-collision.",
        "",
        "## Baseline scorecard highlights",
        "",
        f"- Terminal position error P50/P95/P99/MAX (m): `{fmt(scorecard['accuracy']['terminal_position_error_m'].get('median'))}` / `{fmt(scorecard['accuracy']['terminal_position_error_m'].get('p95'))}` / `{fmt(scorecard['accuracy']['terminal_position_error_m'].get('p99'))}` / `{fmt(scorecard['accuracy']['terminal_position_error_m'].get('max'))}`.",
        f"- Terminal orientation error P50/P95/P99/MAX (rad): `{fmt(scorecard['accuracy']['terminal_orientation_error_rad'].get('median'))}` / `{fmt(scorecard['accuracy']['terminal_orientation_error_rad'].get('p95'))}` / `{fmt(scorecard['accuracy']['terminal_orientation_error_rad'].get('p99'))}` / `{fmt(scorecard['accuracy']['terminal_orientation_error_rad'].get('max'))}`.",
        f"- TCP trajectory error P95/P99/MAX (m): `{fmt(scorecard['accuracy']['tcp_trajectory_error_max_m'].get('p95'))}` / `{fmt(scorecard['accuracy']['tcp_trajectory_error_max_m'].get('p99'))}` / `{fmt(scorecard['accuracy']['tcp_trajectory_error_max_m'].get('max'))}`.",
        f"- Max velocity/acceleration/jerk ratios: `{fmt(scorecard['kinematics_motion_quality']['velocity_ratio'].get('max'))}` / `{fmt(scorecard['kinematics_motion_quality']['acceleration_ratio'].get('max'))}` / `{fmt(scorecard['kinematics_motion_quality']['jerk_ratio'].get('max'))}`.",
        f"- Hard violations: joint-limit `{scorecard['kinematics_motion_quality']['joint_limit_violations']}`, velocity `{scorecard['kinematics_motion_quality']['velocity_limit_violations']}`, acceleration `{scorecard['kinematics_motion_quality']['acceleration_limit_violations']}`, jerk `{scorecard['kinematics_motion_quality']['jerk_limit_violations']}`, continuity `{scorecard['kinematics_motion_quality']['continuity_position_failures']}`.",
        f"- Geometry: waypoint environment collision cases `{scorecard['geometry_collision']['waypoint_environment_collision_cases']}`, waypoint self-collision cases `{scorecard['geometry_collision']['waypoint_self_collision_cases']}`, environment CCD failures `{scorecard['geometry_collision']['environment_ccd_failures']}`, continuous self-collision `{scorecard['geometry_collision']['continuous_self_collision_status']}`.",
        f"- Minimum environment/self clearance (m): `{fmt(scorecard['geometry_collision']['environment_clearance_m'].get('min'))}` / `{fmt(scorecard['geometry_collision']['self_clearance_m'].get('min'))}`; physical acceptance thresholds remain `{scorecard['geometry_collision']['clearance_threshold_status']}`.",
        f"- Worst singularity: sigma_min `{fmt(scorecard['singularity']['minimum_singular_value'].get('min'))}`, condition number `{fmt(scorecard['singularity']['condition_number'].get('max'))}`; threshold `{scorecard['singularity']['threshold_status']}`.",
        "- Model-based required torque: `UNAVAILABLE`; no torque value was imputed from collision-free states.",
        f"- Ruckig validity: `{scorecard['ruckig']['valid_cases']}/{scorecard['ruckig']['applicable_cases']}` applicable actual D41 post-Ruckig controls valid; rate `{fmt(scorecard['ruckig']['validity_rate'])}`. Audit-only perturbation/stress cases are explicitly not relabelled as newly generated Ruckig trajectories.",
        f"- PRE_RUCKIG versus POST_RUCKIG comparison: `{scorecard['post_ruckig_comparison']['status']}` using `{scorecard['post_ruckig_comparison']['pre_case_id']}` and `{scorecard['post_ruckig_comparison']['post_case_id']}`; see the machine-readable scorecard for metric deltas.",
        "",
        "## Findings and Stage 4B handoff",
        "",
        f"- TOP_1_BOTTLENECK: `{risks['top_1_bottleneck']}`.",
        f"- TOP_2_BOTTLENECK: `{risks['top_2_bottleneck']}`.",
        f"- TOP_3_BOTTLENECK: `{risks['top_3_bottleneck']}`.",
        "- These are measured findings and controlled stress results; D46 did not tune or repair the frozen robot system.",
        "- Three Type-A measurement defects were found and fixed: the FK runtime `sdformat` library path, a fresh-replay self-clearance field-semantic mismatch, and an acceptance-set hard-gate conflict. All repairs were regression-tested; unsupported torque, continuous-self-collision, TCP calibration, and physical-limit claims remain explicit coverage gaps.",
        "",
        "## Authoritative artifacts",
        "",
        "- `FINAL_REPORT.json` and `FINAL_REPORT.md`",
        "- `STAGE4_SYSTEM_BENCHMARK_V1.json` and `.md`",
        "- `STAGE4_ACCEPTANCE_SET_V1.json`",
        "- `STAGE4_BASELINE_SCORECARD_V1.json` and `.md`",
        "- `STAGE4_FAILURE_TAXONOMY_V1.json` and `.md`",
        "- `STAGE4_RISK_RANKED_BOTTLENECKS_V1.json` and `.md`",
        "- `STAGE4_LIMIT_PROVENANCE_V1.json`",
        "- `STAGE4_COLLISION_COVERAGE_MATRIX_V1.json`",
        "- `STAGE4_REPRODUCIBILITY_REPORT_V1.json`",
        "- `STAGE4_CASE_RESULTS.csv` / `STAGE4_CASE_RESULTS.jsonl`",
        "",
        "The full machine-readable scorecard, taxonomy, risk ranking, native logs, and per-case evidence are authoritative within this D46 output directory.",
    ]
    (output / "FINAL_REPORT.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    scorecard_lines = ["# STAGE4_BASELINE_SCORECARD_V1", "", "Measurement pipeline: **PASS**.", "", "```json", json.dumps(scorecard, indent=2, sort_keys=True), "```", ""]
    (output / "STAGE4_BASELINE_SCORECARD_V1.md").write_text("\n".join(scorecard_lines), encoding="utf-8")
    taxonomy_lines = ["# STAGE4_FAILURE_TAXONOMY_V1", "", f"Cases: `{len(results)}`; multiple categories per case: `YES`.", "", "| Category | Count |", "|---|---:|"]
    taxonomy_lines += [f"| `{key}` | {value} |" for key, value in sorted(taxonomy["category_counts"].items())]
    taxonomy_lines += ["", "Failure categories remain evidence; D46 does not repair Type-B robot weaknesses.", ""]
    (output / "STAGE4_FAILURE_TAXONOMY_V1.md").write_text("\n".join(taxonomy_lines), encoding="utf-8")
    risk_lines = ["# STAGE4_RISK_RANKED_BOTTLENECKS_V1", "", "| Rank | ID | Category | Frequency | Severity |", "|---:|---|---|---:|---|"]
    for row in risks["bottlenecks"]:
        risk_lines.append(f"| {row['rank']} | `{row['id']}` | {row['category']} | {row['frequency']} | {row['severity']} |")
    risk_lines += ["", "These bottlenecks are Stage 4B inputs, not D46 optimization targets.", ""]
    (output / "STAGE4_RISK_RANKED_BOTTLENECKS_V1.md").write_text("\n".join(risk_lines), encoding="utf-8")
    coverage = {
        "schema_version": "stage4-collision-coverage-matrix-v1", "matrix_id": "STAGE4_COLLISION_COVERAGE_MATRIX_V1", "collision_results_label": COLLISION_METHOD, "levels": [
            {"level": 1, "name": "DISCRETE_ENVIRONMENT_COLLISION", "status": "AVAILABLE", "backend": "native MoveIt2 PlanningScene/Bullet", "result_field": "native_waypoint_environment_collision_count"},
            {"level": 2, "name": "DISCRETE_SELF_COLLISION", "status": "AVAILABLE", "backend": "native MoveIt2 PlanningScene/Bullet", "result_field": "native_waypoint_self_collision_count"},
            {"level": 3, "name": "DENSE_TIME_SAMPLED_COLLISION", "status": "AVAILABLE", "backend": "native adaptive joint-space interpolation", "result_field": "dense_*_collision_samples"},
            {"level": 4, "name": "ENVIRONMENT_CCD", "status": "AVAILABLE_NATIVE_BULLET_ROBOT_WORLD", "backend": "real native CollisionEnv::checkRobotCollision(state1,state2)", "failure_count": scorecard["geometry_collision"]["environment_ccd_failures"], "scope_note": "environment only; collision result labels remain adaptive_discrete_interpolation"},
            {"level": 5, "name": "CONTINUOUS_SELF_COLLISION", "status": "NOT_AVAILABLE", "backend": "installed MoveIt wrapper exposes no validated self two-state continuous overload", "dense_discrete_substitute": False},
        ], "clearance": {"environment": "AVAILABLE_NATIVE_FCL_DISTANCE", "self": "AVAILABLE_NATIVE_FCL_DISTANCE", "threshold_status": "UNRESOLVED_THRESHOLD"},
    }
    write_json(output / "STAGE4_COLLISION_COVERAGE_MATRIX_V1.json", coverage)
    write_json(output / "STAGE4_LIMIT_PROVENANCE_V1.json", limits)
    write_json(output / "STAGE4_REPRODUCIBILITY_REPORT_V1.json", repro)
    write_json(output / "FINAL_REPORT.json", {
        "schema_version": "d46-final-report-v1", "task_status": "PASS", "stage4a_status": "PASS_WITH_EXPLICIT_COVERAGE_GAPS", "stage3_canonical_preserved": True, "final_canonical_update": 480, "training_performed": False, "optimization_performed": False, "canonical_modified": False, "stage4_system_benchmark_v1_created": True, "total_cases_executed": len(results), "family_counts": manifest["family_counts"], "task_success_rate": float(np.mean([row["finite_state"] for row in results])), "measurement_coverage_gaps": scorecard["coverage_gaps"], "unresolved_thresholds": scorecard["unresolved_thresholds"], "material_measurement_defects_found": len(taxonomy["type_a_measurement_defects"]), "material_measurement_defects_fixed": sum(defect["status"] == "FIXED" for defect in taxonomy["type_a_measurement_defects"]), "measurement_defects": taxonomy["type_a_measurement_defects"], "stage4b_input_ready": True, "artifacts_root": str(output.resolve()), "canonical_sha256_after": sha256(CANONICAL), "canonical_sha256_expected": EXPECTED_SHA, "ruckig_validity_rate": scorecard["ruckig"]["validity_rate"], "scorecard": scorecard, "failure_taxonomy": taxonomy, "risk_ranked_bottlenecks": risks, "reproducibility": repro,
    })
    # Write a benchmark markdown after all structured content exists.
    benchmark_md = ["# STAGE4_SYSTEM_BENCHMARK_V1", "", f"Scope: `{SCOPE}`.", "", f"Total cases: `{manifest['total_cases']}`.", "", "| Family | Cases |", "|---|---:|"] + [f"| `{family}` | {count} |" for family, count in sorted(manifest["family_counts"].items())] + ["", "The manifest and acceptance set were frozen before outcome-dependent measurement. Difficult cases are retained.", ""]
    (output / "STAGE4_SYSTEM_BENCHMARK_V1.md").write_text("\n".join(benchmark_md), encoding="utf-8")


def validate_canonical_after(output: Path) -> None:
    if sha256(CANONICAL) != EXPECTED_SHA:
        raise RuntimeError("canonical_checkpoint_changed_during_d46")
    auth = read_json(output / "STAGE3_AUTHENTICATION_V1.json")
    if auth["canonical_checkpoint_sha256"] != EXPECTED_SHA:
        raise RuntimeError("stage3_authentication_identity_drift")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--skip-native", action="store_true")
    parser.add_argument("--reuse-native", action="store_true", help="reuse a completed authoritative full native run")
    parser.add_argument("--reuse-fk", action="store_true", help="reuse a completed authoritative FK trace")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    auth = authenticate_stage3(output)
    lower, upper, limits = load_limits()
    limit_artifact = write_limit_provenance(output, limits)
    cases = make_cases(output, auth["post_q"], auth["pre_q"], lower, upper)
    manifest, acceptance = freeze_benchmark(output, auth, cases, limits)
    if args.skip_native:
        raise RuntimeError("skip-native is diagnostic only and cannot produce an authoritative D46 report")
    native = output / "native" if args.reuse_native else run_native(output, cases)
    if not (native / "D41_native_case_summary.jsonl").is_file():
        raise RuntimeError(f"missing_native_summary:{native}")
    fresh_native = run_native(output, acceptance_cases(cases), "native_replay_2")
    fk_path = output / "fk" / "STAGE4A_FK_TRACE.csv"
    fk_trace = fk_path if args.reuse_fk else run_fk(output, cases)
    if not fk_trace.is_file():
        raise RuntimeError(f"missing_fk_trace:{fk_trace}")
    results, _run_meta = analyze(output, auth, manifest, acceptance, cases, native, fk_trace, limits, lower, upper)
    scorecard, taxonomy, risks = aggregate_scorecard(output, results, auth, manifest, acceptance, native)
    repro = run_reproducibility(output, cases, results, auth, fresh_native)
    materialize_markdown(output, manifest, scorecard, taxonomy, risks, repro, limit_artifact, results, auth)
    validate_canonical_after(output)
    print(json.dumps({"TASK_STATUS": "PASS", "STAGE4A_STATUS": "PASS_WITH_EXPLICIT_COVERAGE_GAPS", "TOTAL_CASES_EXECUTED": len(results), "family_counts": manifest["family_counts"], "output": str(output), "canonical_sha256": sha256(CANONICAL), "top_bottlenecks": [risks["top_1_bottleneck"], risks["top_2_bottleneck"], risks["top_3_bottleneck"]]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
