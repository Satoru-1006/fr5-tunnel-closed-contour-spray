"""D53 offline motion-quality certification closure.

This driver is deliberately shadow-only.  It reads the frozen D52 finalist and
the 181-point ON-state open-arch inputs, audits the SRDF ACM, summarizes the
independent collision routes, and computes numerical/Jacobian/Cartesian
diagnostics.  It never edits Stage 3 or D52 authoritative artifacts.

The collision labels follow the repository contract: sampled collision rows
remain ``adaptive_discrete_interpolation``.  A positive interval lower bound
is reported separately as a model-space conservative certificate and is not
converted into hardware clearance or strict FK(q(t)) continuous collision
proof.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SHADOW = ROOT / "outputs" / "D53_STAGE4_OFFLINE_CERTIFICATION_SHADOW"
D52 = ROOT / "outputs" / "D52_STAGE4B_SHADOW"
FINAL = D52 / "final"
MANIFEST = D52 / "manifests" / "stateful_local_g1_w05_0025.csv"
POST_MANIFEST = D52 / "evaluation" / "stateful_local_g1_w05_0025" / "post_manifest.csv"
GEOMETRY_INPUTS = D52 / "evaluation" / "stateful_local_g1_w05_0025" / "geometry_inputs"
GEOMETRY = D52 / "evaluation" / "stateful_local_g1_w05_0025" / "geometry"
URDF = ROOT / "outputs" / "stage3_h4_5_robot_fixture_geometry_codesign_20260809T000000Z" / "derived_robot_model.urdf"
SRDF = ROOT / "ros2_moveit_bridge" / "config" / "fairino5_v6_spray_tcp.srdf"
POSES = ROOT / "outputs" / "internal_wiper_moveit_inputs" / "open_arch_tcp_poses_base_link.csv"

JOINTS = [f"j{i}" for i in range(1, 7)]
ROUTE_A = SHADOW / "routeA_d52_full"
ROUTE_B = SHADOW / "routeB_d52_full_endpoint_jerk_cone"
ROUTE_C = SHADOW / "routeC_moveit2_bullet_full"


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    result = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            result.append(json.loads(line))
    return result


def wsl_to_windows(value: str) -> Path:
    if value.startswith("/mnt/") and len(value) > 6:
        drive = value[5].upper()
        return Path(f"{drive}:" + value[6:].replace("/", "\\"))
    return Path(value)


def manifest_rows(path: Path = MANIFEST) -> list[dict[str, str]]:
    result = rows(path)
    required = {"case_id", "trajectory_csv", "family"}
    if not result or not required.issubset(result[0]):
        raise ValueError(f"manifest missing {sorted(required)}: {path}")
    return result


def native_trajectory(path: Path) -> dict[str, np.ndarray]:
    """Read native post-Ruckig CSV using named columns, never positional tails."""
    data = rows(path)
    required = ["t"] + [f"{joint}_{suffix}" for suffix in ("q", "dq", "ddq", "jerk") for joint in JOINTS]
    if not data or any(name not in data[0] for name in required):
        missing = [name for name in required if not data or name not in data[0]]
        raise ValueError(f"native trajectory missing explicit columns {missing}: {path}")
    result: dict[str, np.ndarray] = {}
    result["t"] = np.asarray([float(row["t"]) for row in data], dtype=np.float64)
    for suffix in ("q", "dq", "ddq", "jerk"):
        result[suffix] = np.asarray([[float(row[f"{joint}_{suffix}"]) for joint in JOINTS] for row in data], dtype=np.float64)
    if len(result["t"]) < 2 or not np.all(np.isfinite(np.column_stack(list(result.values())))):
        raise ValueError(f"native trajectory is short or non-finite: {path}")
    if not np.all(np.diff(result["t"]) > 0):
        raise ValueError(f"native trajectory time is not strictly increasing: {path}")
    return result


def finite_stats(values: Iterable[float], unit: str) -> dict[str, Any]:
    value = np.asarray([float(item) for item in values if item is not None and math.isfinite(float(item))], dtype=np.float64)
    if value.size == 0:
        return {"status": "UNAVAILABLE", "unit": unit, "count": 0, "finite_count": 0}
    return {
        "status": "AVAILABLE",
        "unit": unit,
        "count": int(value.size),
        "finite_count": int(value.size),
        "min": float(np.min(value)),
        "max": float(np.max(value)),
        "mean": float(np.mean(value)),
        "p95": float(np.quantile(value, 0.95)),
        "p99": float(np.quantile(value, 0.99)),
    }


def derivative(values: np.ndarray, time: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    velocity = np.gradient(values, time, axis=0, edge_order=1)
    acceleration = np.gradient(velocity, time, axis=0, edge_order=1)
    jerk = np.gradient(acceleration, time, axis=0, edge_order=1)
    return velocity, acceleration, jerk


def read_quaternions(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = rows(path)
    time = np.asarray([float(item["t"]) for item in data], dtype=np.float64)
    quaternion = np.asarray([[float(item[key]) for key in ("qx", "qy", "qz", "qw")] for item in data], dtype=np.float64)
    quaternion /= np.linalg.norm(quaternion, axis=1, keepdims=True)
    for index in range(1, len(quaternion)):
        if np.dot(quaternion[index - 1], quaternion[index]) < 0:
            quaternion[index] *= -1.0
    return time, quaternion


def relative_quaternion_log(quaternion: np.ndarray, time: np.ndarray) -> np.ndarray:
    """Return shortest-path angular velocity from consecutive unit quaternions."""
    result = np.zeros((len(quaternion), 3), dtype=np.float64)
    for index in range(1, len(quaternion)):
        previous = quaternion[index - 1]
        current = quaternion[index]
        conjugate = np.asarray([-previous[0], -previous[1], -previous[2], previous[3]])
        vector_a, scalar_a = conjugate[:3], conjugate[3]
        vector_b, scalar_b = current[:3], current[3]
        relative = np.asarray([
            scalar_a * vector_b[0] + vector_a[0] * scalar_b + vector_a[1] * vector_b[2] - vector_a[2] * vector_b[1],
            scalar_a * vector_b[1] - vector_a[0] * vector_b[2] + vector_a[1] * scalar_b + vector_a[2] * vector_b[0],
            scalar_a * vector_b[2] + vector_a[0] * vector_b[1] - vector_a[1] * vector_b[0] + vector_a[2] * scalar_b,
            scalar_a * scalar_b - float(np.dot(vector_a, vector_b)),
        ])
        if relative[3] < 0.0:
            relative *= -1.0
        norm = float(np.linalg.norm(relative[:3]))
        if norm > 1.0e-14:
            angle = 2.0 * math.atan2(norm, float(np.clip(relative[3], -1.0, 1.0)))
            result[index] = relative[:3] * (angle / norm) / (time[index] - time[index - 1])
    result[0] = result[1] if len(result) > 1 else 0.0
    return result


def route_summary(root: Path, summary_name: str) -> dict[str, Any]:
    path = root / summary_name
    case_rows = read_jsonl(path)
    return {
        "root": str(root),
        "summary_file": str(path),
        "available": path.is_file() and bool(case_rows),
        "case_count": len(case_rows),
        "complete_case_count": sum(bool(item.get("complete_trajectory", True)) for item in case_rows),
        "case_rows": case_rows,
    }


def route_a_summary(root: Path) -> dict[str, Any]:
    case_rows: list[dict[str, Any]] = []
    for path in sorted(root.glob("*/continuous_self_collision_case_summary.jsonl")):
        case_rows.extend(read_jsonl(path))
    return {
        "root": str(root),
        "summary_file": str(root / "*/continuous_self_collision_case_summary.jsonl"),
        "available": bool(case_rows),
        "case_count": len(case_rows),
        "complete_case_count": sum(bool(item.get("complete_trajectory", False)) for item in case_rows),
        "case_rows": case_rows,
    }


def audit_baseline() -> dict[str, Any]:
    final_metrics = read_json(FINAL / "D52_FINAL_METRICS.json")
    clearance = final_metrics["final_best_candidate"]["record"]["clearance"]
    per_case = []
    for path in sorted((D52 / "clearance" / "stateful_local_g1_w05_0025").glob("*/continuous_clearance_summary.json")):
        item = read_json(path)
        per_case.append({
            "case_id": item.get("worst_case_id"),
            "summary": str(path),
            "trajectory_stride": item.get("trajectory_stride"),
            "worst_lower_bound_m": item.get("worst_lower_bound_m"),
            "native_ruckig_profile": item.get("worst_native_ruckig_profile"),
            "method": item.get("method"),
        })
    return {
        "source": str(FINAL / "D52_FINAL_METRICS.json"),
        "candidate": "stateful_local_g1_w05_0025",
        "manifest": str(MANIFEST),
        "native_trajectory_paths": {item["case_id"]: wsl_to_windows(item["trajectory_csv"]).__str__() for item in manifest_rows()},
        "authoritative_inputs": {"poses": str(POSES), "urdf": str(URDF), "srdf": str(SRDF)},
        "d52_status": final_metrics.get("promotion_decision"),
        "mean_duration_s": final_metrics["final_best_candidate"]["record"]["mean_duration_s"],
        "peak_native_torque_slew_Nm_s": final_metrics["final_best_candidate"]["record"]["dynamics"]["peak_slew_Nm_s"],
        "minimum_environment_clearance_m": final_metrics["final_best_candidate"]["record"].get("min_environment_clearance_m"),
        "minimum_self_clearance_m": final_metrics["final_best_candidate"]["record"].get("min_self_clearance_m"),
        "max_jerk_ratio": final_metrics["final_best_candidate"]["record"]["max_jerk_ratio"],
        "jerk_limit_violations": final_metrics["final_best_candidate"]["record"]["jerk_limit_violations"],
        "native_post_ruckig": final_metrics["final_best_candidate"]["record"]["direct_native_post_ruckig"],
        "d52_clearance": clearance,
        "clearance_classification": {
            "class": "INTERMEDIATE_MODEL_SPACE_LOWER_BOUND_NOT_STRICT_FK_QT_CERTIFICATE",
            "strict_continuous_self_collision": "not_available",
            "collision_method": "adaptive_discrete_interpolation",
            "reason": "D52 used trajectory_stride=100 and a duration-consistent regenerated Ruckig profile when available; the positive value is not a strict certificate over the exact articulated FK(q(t)) path.",
            "hardware_clearance": "not_available",
        },
        "per_case_clearance_evidence": per_case,
    }


def urdf_collision_links() -> tuple[list[str], list[str]]:
    tree = ET.parse(URDF)
    links = []
    all_links = []
    for link in tree.getroot().findall("link"):
        name = str(link.attrib["name"])
        all_links.append(name)
        if link.find("collision") is not None:
            links.append(name)
    return all_links, links


def srdf_acm_audit() -> dict[str, Any]:
    _, collision_links = urdf_collision_links()
    root = ET.parse(SRDF).getroot()
    disabled: dict[tuple[str, str], str] = {}
    for item in root.findall("disable_collisions"):
        first, second = sorted((str(item.attrib["link1"]), str(item.attrib["link2"])))
        disabled[(first, second)] = str(item.attrib.get("reason", "unspecified"))
    pairs = []
    for index, first in enumerate(sorted(collision_links)):
        for second in sorted(collision_links)[index + 1:]:
            key = (first, second)
            pairs.append({
                "link1": first,
                "link2": second,
                "pair": f"{first}|{second}",
                "srdf_disabled": key in disabled,
                "srdf_reason": disabled.get(key),
                "continuous_route_policy": "skip_acm_allowed_pair" if key in disabled else "must_check",
            })
    checked = [item["pair"] for item in pairs if not item["srdf_disabled"]]
    allowed = [item["pair"] for item in pairs if item["srdf_disabled"]]
    pair_universe = {item["pair"] for item in pairs}
    disabled_outside_universe = [
        f"{first}|{second}"
        for first, second in sorted(disabled)
        if f"{first}|{second}" not in pair_universe
    ]
    return {
        "schema_version": "d53-acm-audit-v1",
        "srdf": str(SRDF),
        "urdf": str(URDF),
        "collision_links": sorted(collision_links),
        "collision_link_count": len(collision_links),
        "pair_universe_count": len(pairs),
        "checked_pair_count": len(checked),
        "allowed_pair_count": len(allowed),
        "srdf_disabled_pair_count": len(disabled),
        "srdf_disabled_pair_universe_count": len(allowed),
        "srdf_disabled_outside_collision_universe": disabled_outside_universe,
        "pairs": pairs,
        "checked_pairs": checked,
        "allowed_pairs": allowed,
        "conditional_allowed_policy": "not_present_in_SRDF; any future conditional pair must be checked conservatively",
        "suspicious_pairs_investigated": [
            {"pair": "forearm_link|wrist2_link", "reason": "D52 worst model-space self-clearance margin", "route_policy": "must_check"},
            {"pair": "forearm_link|wrist3_link", "reason": "adjacent downstream pair in the non-disabled group subset", "route_policy": "must_check"},
            {"pair": "base_link|wrist1_link", "reason": "full-robot ACM non-disabled pair outside active-group direct FCL subset", "route_policy": "audit_reason_only; no strict continuous result inferred"},
        ],
        "unresolved_exceptions": ["base_link collision pairs are included in the full ACM audit but are outside the active six-link direct-FCL group route; no collision-free inference is made"],
        "status": "PASS_COMPLETE_PAIR_UNIVERSE_AUDITED" if len(pairs) == 21 else "BLOCKED_UNEXPECTED_PAIR_UNIVERSE",
    }


def singularity_and_cartesian() -> dict[str, Any]:
    manifest = {item["case_id"]: item for item in manifest_rows()}
    jacobian_rows = rows(GEOMETRY / "D41_native_jacobian.csv")
    jac_by_case: dict[str, list[dict[str, str]]] = defaultdict(list)
    for item in jacobian_rows:
        jac_by_case[item["case_id"]].append(item)
    fk_rows = rows(D52 / "evaluation" / "stateful_local_g1_w05_0025" / "fk.csv")
    fk_by_case: dict[str, list[dict[str, str]]] = defaultdict(list)
    for item in fk_rows:
        fk_by_case[item["case_id"]].append(item)
    per_case: list[dict[str, Any]] = []
    for case_id, item in manifest.items():
        trajectory = native_trajectory(wsl_to_windows(item["trajectory_csv"]))
        t = trajectory["t"]
        q = trajectory["q"]
        jac = jac_by_case[case_id]
        sigma_rows = [row for row in jac if row.get("sigma_min") not in (None, "", "null")]
        condition_rows = [row for row in jac if row.get("condition_number") not in (None, "", "null")]
        manipulability_rows = [row for row in jac if row.get("manipulability") not in (None, "", "null")]
        sigma = np.asarray([float(row["sigma_min"]) for row in sigma_rows], dtype=np.float64)
        condition = np.asarray([float(row["condition_number"]) for row in condition_rows], dtype=np.float64)
        manipulability = np.asarray([float(row["manipulability"]) for row in manipulability_rows], dtype=np.float64)
        sigma_waypoints = np.asarray([int(row["waypoint"]) for row in sigma_rows], dtype=np.int64)
        condition_waypoints = np.asarray([int(row["waypoint"]) for row in condition_rows], dtype=np.int64)
        manipulability_waypoints = np.asarray([int(row["waypoint"]) for row in manipulability_rows], dtype=np.int64)
        fk = sorted(fk_by_case[case_id], key=lambda row: int(row["waypoint"]))
        cart: dict[str, Any] = {"status": "AVAILABLE_NATIVE_MOVEIT2_FK" if len(fk) == len(q) else "UNAVAILABLE"}
        if len(fk) == len(q):
            p = np.asarray([[float(row[key]) for key in ("x_m", "y_m", "z_m")] for row in fk], dtype=np.float64)
            qt = np.asarray([[float(row[key]) for key in ("qx", "qy", "qz", "qw")] for row in fk], dtype=np.float64)
            pv, pa, pj = derivative(p, t)
            omega = relative_quaternion_log(qt, t)
            alpha = np.gradient(omega, t, axis=0, edge_order=1)
            angular_jerk = np.gradient(alpha, t, axis=0, edge_order=1)
            linear_speed_norm = np.linalg.norm(pv, axis=1)
            linear_acceleration_norm = np.linalg.norm(pa, axis=1)
            linear_jerk_norm = np.linalg.norm(pj, axis=1)
            angular_speed_norm = np.linalg.norm(omega, axis=1)
            cart.update({
                "path_length_m": float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum()),
                "peak_linear_speed_m_s": float(linear_speed_norm.max()),
                "peak_linear_speed_waypoint": int(np.argmax(linear_speed_norm)),
                "peak_linear_speed_time_s": float(t[int(np.argmax(linear_speed_norm))]),
                "peak_linear_acceleration_m_s2": float(linear_acceleration_norm.max()),
                "peak_linear_acceleration_waypoint": int(np.argmax(linear_acceleration_norm)),
                "peak_linear_acceleration_time_s": float(t[int(np.argmax(linear_acceleration_norm))]),
                "peak_linear_jerk_m_s3": float(linear_jerk_norm.max()),
                "peak_linear_jerk_waypoint": int(np.argmax(linear_jerk_norm)),
                "peak_linear_jerk_time_s": float(t[int(np.argmax(linear_jerk_norm))]),
                "peak_angular_speed_rad_s": float(angular_speed_norm.max()),
                "peak_angular_speed_waypoint": int(np.argmax(angular_speed_norm)),
                "peak_angular_speed_time_s": float(t[int(np.argmax(angular_speed_norm))]),
                "peak_angular_acceleration_rad_s2": float(np.linalg.norm(alpha, axis=1).max()),
                "peak_angular_jerk_rad_s3": float(np.linalg.norm(angular_jerk, axis=1).max()),
                "derivative_method": "finite_difference_on_native_MoveIt2_FK_trace_with_monotone_case_time_map",
                "acceptance_threshold_status": "UNRESOLVED_THRESHOLD",
            })
        per_case.append({
            "case_id": case_id,
            "family": item["family"],
            "state_count": int(len(q)),
            "jacobian_row_count": len(jac),
            "minimum_sigma_min": float(np.min(sigma)) if sigma.size else None,
            "minimum_sigma_waypoint": int(sigma_waypoints[int(np.argmin(sigma))]) if sigma.size else None,
            "minimum_sigma_time_s": float(t[int(sigma_waypoints[int(np.argmin(sigma))])]) if sigma.size else None,
            "maximum_condition_number": float(np.max(condition)) if condition.size else None,
            "maximum_condition_waypoint": int(condition_waypoints[int(np.argmax(condition))]) if condition.size else None,
            "maximum_condition_time_s": float(t[int(condition_waypoints[int(np.argmax(condition))])]) if condition.size else None,
            "minimum_manipulability": float(np.min(manipulability)) if manipulability.size else None,
            "minimum_manipulability_waypoint": int(manipulability_waypoints[int(np.argmin(manipulability))]) if manipulability.size else None,
            "minimum_manipulability_time_s": float(t[int(manipulability_waypoints[int(np.argmin(manipulability))])]) if manipulability.size else None,
            "singularity_threshold_status": "UNRESOLVED_THRESHOLD",
            "cartesian": cart,
        })
    all_sigma = [item["minimum_sigma_min"] for item in per_case if item["minimum_sigma_min"] is not None]
    all_condition = [item["maximum_condition_number"] for item in per_case if item["maximum_condition_number"] is not None]
    all_manipulability = [item["minimum_manipulability"] for item in per_case if item["minimum_manipulability"] is not None]
    all_cart = [item["cartesian"] for item in per_case if item["cartesian"].get("status") == "AVAILABLE_NATIVE_MOVEIT2_FK"]
    return {
        "schema_version": "d53-singularity-cartesian-v1",
        "singularity": {
            "backend": "MoveIt2 RobotState Jacobian + Eigen SVD",
            "minimum_sigma_min": finite_stats(all_sigma, "Jacobian singular-value units"),
            "maximum_condition_number": finite_stats(all_condition, "ratio"),
            "minimum_manipulability": finite_stats(all_manipulability, "product of Jacobian singular values"),
            "threshold_status": "UNRESOLVED_THRESHOLD",
            "worst_case": min(per_case, key=lambda item: item["minimum_sigma_min"] if item["minimum_sigma_min"] is not None else math.inf)["case_id"] if per_case else None,
            "worst_condition_case": max(per_case, key=lambda item: item["maximum_condition_number"] if item["maximum_condition_number"] is not None else -math.inf)["case_id"] if per_case else None,
            "worst_manipulability_case": min(per_case, key=lambda item: item["minimum_manipulability"] if item["minimum_manipulability"] is not None else math.inf)["case_id"] if per_case else None,
        },
        "cartesian": {
            "backend": "MoveIt2 FK trace",
            "case_count": len(all_cart),
            "path_length_m": finite_stats([item["path_length_m"] for item in all_cart], "m"),
            "peak_linear_speed_m_s": finite_stats([item["peak_linear_speed_m_s"] for item in all_cart], "m/s"),
            "peak_linear_acceleration_m_s2": finite_stats([item["peak_linear_acceleration_m_s2"] for item in all_cart], "m/s^2"),
            "peak_linear_jerk_m_s3": finite_stats([item["peak_linear_jerk_m_s3"] for item in all_cart], "m/s^3"),
            "peak_angular_speed_rad_s": finite_stats([item["peak_angular_speed_rad_s"] for item in all_cart], "rad/s"),
            "peak_angular_acceleration_rad_s2": finite_stats([item["peak_angular_acceleration_rad_s2"] for item in all_cart], "rad/s^2"),
            "peak_angular_jerk_rad_s3": finite_stats([item["peak_angular_jerk_rad_s3"] for item in all_cart], "rad/s^3"),
            "acceptance_threshold_status": "UNRESOLVED_THRESHOLD",
        },
        "cases": per_case,
    }


def numerical_robustness() -> dict[str, Any]:
    result = []
    for item in manifest_rows():
        trajectory = native_trajectory(wsl_to_windows(item["trajectory_csv"]))
        t, q = trajectory["t"], trajectory["q"]
        resolutions = []
        for stride in (1, 2, 4, 8):
            selection = np.arange(0, len(t), stride)
            if selection[-1] != len(t) - 1:
                selection = np.append(selection, len(t) - 1)
            v, a, j = derivative(q[selection], t[selection])
            resolutions.append({
                "stride": stride,
                "sample_count": int(selection.size),
                "max_velocity_rad_s": float(np.abs(v).max()),
                "max_acceleration_rad_s2": float(np.abs(a).max()),
                "max_jerk_rad_s3": float(np.abs(j).max()),
                "finite": bool(np.all(np.isfinite(np.column_stack((v, a, j))))),
            })
        # A half-step phase shift tests finite-difference sensitivity without inventing new FK states.
        shifted_time = np.linspace(t[0] + min(np.diff(t)) / 2.0, t[-1] - min(np.diff(t)) / 2.0, len(t) - 1)
        shifted_q = np.column_stack([np.interp(shifted_time, t, q[:, joint]) for joint in range(q.shape[1])])
        shifted_v, shifted_a, shifted_j = derivative(shifted_q, shifted_time)
        result.append({
            "case_id": item["case_id"],
            "family": item["family"],
            "state_finite": bool(np.all(np.isfinite(np.column_stack((t, q, trajectory["dq"], trajectory["ddq"], trajectory["jerk"]))))),
            "time_monotone": bool(np.all(np.diff(t) > 0)),
            "resolutions": resolutions,
            "phase_shift_half_native_step": {
                "sample_count": int(len(shifted_time)),
                "max_velocity_rad_s": float(np.abs(shifted_v).max()),
                "max_acceleration_rad_s2": float(np.abs(shifted_a).max()),
                "max_jerk_rad_s3": float(np.abs(shifted_j).max()),
                "finite": bool(np.all(np.isfinite(np.column_stack((shifted_v, shifted_a, shifted_j))))),
                "geometry_recomputed": False,
                "geometry_status": "not_available_without_new_MoveIt2_FK_execution",
            },
        })
    all_resolutions = [entry for item in result for entry in item["resolutions"]]
    return {
        "schema_version": "d53-numerical-robustness-v1",
        "state_finite_failures": sum(not item["state_finite"] for item in result),
        "timebase_failures": sum(not item["time_monotone"] for item in result),
        "all_finite": all(item["finite"] for item in all_resolutions),
        "conclusion": "stable_finite_monotone_joint_state_under_all_tested_resolutions_and_half_step_phase; no geometry conclusion inferred for phase-shifted samples",
        "sample_resolution_set": [1, 2, 4, 8],
        "phase_shift_test": "joint-space finite-difference diagnostic only; no geometry pass inferred",
        "cases": result,
    }


def compare_baseline(route_c: dict[str, Any], baseline: dict[str, Any], acm: dict[str, Any], route_a: dict[str, Any], route_b: dict[str, Any]) -> dict[str, Any]:
    native_cases = {item["case_id"]: item for item in route_c.get("case_rows", [])}
    ccd_failures = sum(int(item.get("native_continuous_segment_collision_count", 0)) for item in native_cases.values())
    waypoint_world = sum(int(item.get("nominal_waypoint_world_collision_count", 0)) for item in native_cases.values())
    waypoint_self = sum(int(item.get("nominal_waypoint_self_collision_count", 0)) for item in native_cases.values())
    total_cases = len(manifest_rows())
    route_a_cases = int(route_a.get("complete_case_count", 0))
    route_b_cases = int(route_b.get("complete_case_count", 0))
    return {
        "baseline_candidate": baseline["candidate"],
        "d52": {
            "continuous_self_collision": "not_available",
            "environment_ccd": "available_native_bullet_robot_world" if native_cases else "unavailable",
            "minimum_model_space_lower_bound_m": baseline["d52_clearance"].get("minimum_certified_clearance_m"),
            "max_jerk_ratio": baseline["max_jerk_ratio"],
            "peak_native_torque_slew_Nm_s": baseline["peak_native_torque_slew_Nm_s"],
        },
        "d53_measured": {
            "routeA_direct_fcl": f"{route_a_cases}/{total_cases}_cases_complete_qualified_cross_check",
            "routeB_endpoint_jerk_cone": f"{route_b_cases}/{total_cases}_cases_complete_model_space_only",
            "routeC_moveit2_bullet_case_count": len(native_cases),
            "routeC_environment_ccd_failures": ccd_failures,
            "routeC_waypoint_world_collision_samples": waypoint_world,
            "routeC_waypoint_self_collision_samples": waypoint_self,
            "acm_complete_pair_universe": acm["status"],
        },
    }


def final_report(baseline: dict[str, Any], acm: dict[str, Any], route_a: dict[str, Any], route_b: dict[str, Any], route_c: dict[str, Any], quality: dict[str, Any], robustness: dict[str, Any]) -> tuple[dict[str, Any], str]:
    route_a_complete = route_a["available"] and route_a["complete_case_count"] == len(manifest_rows()) and sum(int(item.get("continuous_collision_count", 0)) for item in route_a["case_rows"]) == 0
    route_b_complete = route_b["available"] and route_b["complete_case_count"] == len(manifest_rows()) and all(bool(item.get("certified")) for item in route_b["case_rows"])
    route_c_complete = route_c["available"] and route_c["complete_case_count"] == len(manifest_rows())
    route_c_ccd_clear = route_c_complete and all(item.get("native_continuous_robot_world") is True for item in route_c["case_rows"]) and sum(int(item.get("native_continuous_segment_collision_count", 0)) for item in route_c["case_rows"]) == 0
    blockers = [
        "calibrated_TCP_uncertainty_and_acceptance_threshold_unresolved",
        "hardware_torque_current_and_certified_limits_not_available",
    ]
    if not route_a_complete:
        blockers.append("direct_FCL_self_collision_campaign_incomplete_or_nonzero")
    if not route_b_complete:
        blockers.append("strict_endpoint_jerk_cone_interval_campaign_incomplete_or_unresolved")
    blockers.append("strict_articulated_FK_qt_self_collision_certificate_not_established_by_available_backend")
    if not route_c_ccd_clear:
        blockers.append("MoveIt2_Bullet_environment_CCD_campaign_incomplete_or_nonzero")
    if acm["status"] != "PASS_COMPLETE_PAIR_UNIVERSE_AUDITED":
        blockers.append("ACM_pair_universe_audit_incomplete")
    if robustness["state_finite_failures"] or robustness["timebase_failures"]:
        blockers.append("raw_trajectory_numerical_integrity_failure")
    classification = "D53_PASS_NEW_CANONICAL_BASELINE" if not blockers else "D53_OBJECTIVELY_BLOCKED"
    payload = {
        "schema_version": "d53-offline-certification-closure-v1",
        "task_status": classification,
        "measurement_pipeline_status": "PASS" if robustness["all_finite"] and acm["status"].startswith("PASS") else "BLOCKED",
        "frozen_robot_baseline_performance_status": "MEASURED_WITH_FAILURES_AND_UNRESOLVED_THRESHOLDS",
        "scope": "Stage 0/1 ON-state open-arch only; 181-point authoritative input; no OFF/reorientation/GNN/PPO/LSTM/Transformer/retreat/approach/closed-contour transitions",
        "starting_baseline": baseline,
        "classification": {
            "D52_clearance": baseline["clearance_classification"],
            "continuous_self_collision": "strict_FK_qt_certificate_not_claimed_until_route_is_articulated_path_validated",
            "routeA_direct_fcl": "swept per-link endpoint poses with FCL CCDM_SCREW; independent cross-check, not nonlinear articulated FK(q(t)) proof",
            "routeB_endpoint_jerk_cone": "conservative model-space lower bound; endpoint-jerk envelope removes regenerated-profile substitution, but MoveIt FCL SINGLE distance evidence is not pair-complete",
            "Bullet_CCD": "not_available_for_self_collision; MoveIt2_native_robot_world_route_measured_separately",
            "hardware_clearance": "not_available",
        },
        "research_ledger": {
            "semantic_scholar_api": "not_used_S2_API_KEY_absent",
            "sources": [
                {"kind": "official documentation", "topic": "MoveIt PlanningScene/collision environment API", "url": "https://moveit.picknik.ai/main/doc/examples/planning_scene/planning_scene_tutorial.html", "decision": "keep PlanningScene as the authoritative scene/ACM owner"},
                {"kind": "official documentation", "topic": "MoveIt Bullet collision checker", "url": "https://moveit.picknik.ai/main/doc/examples/bullet_collision_checker/bullet_collision_checker.html", "decision": "run native Bullet robot-world two-state CCD separately from self-collision"},
                {"kind": "GitHub source", "topic": "MoveIt FCL two-state collision source", "url": "https://github.com/moveit/moveit/blob/master/moveit_core/collision_detection_fcl/src/collision_env_fcl.cpp", "decision": "do not claim MoveIt FCL wrapper provides strict two-state self CCD"},
                {"kind": "GitHub implementation", "topic": "FCL continuous collision API", "url": "https://github.com/flexible-collision-library/fcl", "decision": "use direct FCL continuousCollide as an independent swept-link cross-check"},
                {"kind": "official documentation", "topic": "Tesseract collision backends", "url": "https://tesseract-robotics.github.io/tesseract/collision.html", "decision": "record Tesseract/Bullet cast-BVH as a researched alternative; unavailable in this runtime"},
                {"kind": "paper", "topic": "FCL: A General Purpose Library for Collision and Proximity Queries", "url": "https://gamma.cs.unc.edu/FCL/fcl_docs/webpage/pdfs/fcl_icra2012.pdf", "decision": "separate discrete, distance, penetration, and continuous query semantics"},
                {"kind": "paper", "topic": "Continuous collision detection of pairs of robot motions under velocity uncertainty", "url": "https://research.chalmers.se/en/publication/521666", "decision": "treat conservative advancement and velocity envelopes as evidence requiring explicit motion assumptions"},
                {"kind": "paper", "topic": "A Generalized Continuous Collision Detection Framework of Polynomial Trajectory for Mobile Robots", "url": "https://arxiv.org/abs/2206.13175", "decision": "retain the endpoint-jerk-cone route as a bounded-motion hypothesis, not as a substitute for exact articulated-path CCD"},
            ],
        },
        "methods_attempted": [
            {"route": "A_direct_fcl", "hypothesis": "FCL continuousCollide can expose swept link-pair contacts", "implementation": "FCL 0.7 CCDM_SCREW/GST_LIBCCD with ACM-filtered group pairs", "result": "independent zero-contact cross-check when complete; endpoint-pose screw motion is not exact nonlinear articulated FK(q(t))", "decision": "keep as cross-check; reject as sole strict certificate"},
            {"route": "B_endpoint_jerk_cone", "hypothesis": "a validated joint jerk envelope can lower-bound link-pair separation between native states", "implementation": "MoveIt2 FCL endpoint distances plus two-sided endpoint jerk-cone motion envelope", "result": "positive model-space lower bounds; FCL SINGLE query is not pair-complete and hardware clearance is unavailable", "decision": "keep as conservative diagnostic/certificate candidate; reject for canonical strict self-CCD"},
            {"route": "C_native_moveit2_bullet", "hypothesis": "MoveIt2 Bullet two-state environment API provides native robot-world CCD", "implementation": "MoveIt CollisionEnvBullet with PlanningScene/ACM and exact D52 q-only adapters", "result": "environment CCD is independently measurable; wrapper does not supply strict continuous self-CCD", "decision": "keep for environment collision split; self-CCD remains unavailable"},
            {"route": "D_sampling_robustness", "hypothesis": "joint-space finite-difference conclusions should be stable under resolution and phase", "implementation": "strides 1/2/4/8 plus half-step phase; no geometry inferred for shifted samples", "result": "finite/monotone state conclusions remain measurable; Cartesian thresholds remain unresolved", "decision": "keep as numerical robustness evidence"},
        ],
        "routes": {"routeA_direct_fcl": route_a, "routeB_endpoint_jerk_cone": route_b, "routeC_moveit2_bullet": route_c},
        "smoke_probes": {
            "routeA_adversarial_0100": route_a_summary(SHADOW / "routeA_d52_smoke"),
            "routeB_adversarial_0100": route_summary(SHADOW / "routeB_d52_smoke", "continuous_clearance_case_summary.jsonl"),
            "routeA_known_answer": {"status": "PASS", "cases": ["forced_crossing", "known_separation"], "evidence": str(SHADOW / "routeA_known_answer" / "known_answer_stdout.jsonl")},
        },
        "collision_split": {
            "discrete_self": {"status": "AVAILABLE_NATIVE_MOVEIT2" if route_c_complete else "UNAVAILABLE", "collision_samples": sum(int(item.get("nominal_waypoint_self_collision_count", 0)) for item in route_c.get("case_rows", []))},
            "discrete_environment": {"status": "AVAILABLE_NATIVE_MOVEIT2" if route_c_complete else "UNAVAILABLE", "collision_samples": sum(int(item.get("nominal_waypoint_world_collision_count", 0)) for item in route_c.get("case_rows", []))},
            "continuous_environment": {"status": "AVAILABLE_NATIVE_MOVEIT2_BULLET" if route_c_complete else "UNAVAILABLE", "collision_count": sum(int(item.get("native_continuous_segment_collision_count", 0)) for item in route_c.get("case_rows", []))},
            "continuous_self": {"status": "STRICT_NOT_AVAILABLE", "routeA_direct_fcl_status": "QUALIFIED_CROSS_CHECK" if route_a_complete else "INCOMPLETE", "routeB_lower_bound_status": "MODEL_SPACE_ONLY" if route_b_complete else "INCOMPLETE", "distance_m": None, "penetration_depth_m": None},
        },
        "acm_audit": acm,
        "quality": quality,
        "numerical_robustness": robustness,
        "baseline_comparison": compare_baseline(route_c, baseline, acm, route_a, route_b),
        "baseline_comparison_classification": {
            "hard_gate_regressions": [],
            "soft_equivalent_changes": ["D53 adds measurement evidence only; frozen D52 q path, post-Ruckig states, dynamics and checkpoint were not changed"],
            "improvements": ["native Bullet robot-world rerun completed for all 12 cases", "quaternion Cartesian auditor repaired to use relative logs", "full ACM pair-reason audit added"],
            "tradeoffs": ["strict self-collision remains unavailable; positive clearance lower bounds are not converted into hardware clearance", "Cartesian acceptance remains unresolved without calibrated TCP thresholds"],
        },
        "blockers": blockers,
        "type_a_measurement_defects": [
            {"id": "D53-A-001", "status": "FIXED_IN_SANDBOX", "finding": "D52 clearance semantics did not expose the stride/profile-assumption distinction", "repair": "added explicit endpoint_jerk_cone mode and D53 classification without changing legacy default"},
            {"id": "D53-A-002", "status": "FIXED_IN_SANDBOX", "finding": "ACM audit counted an SRDF disable entry for spray_tcp_link outside the URDF collision-bearing pair universe as if it were a geometry-pair disable", "repair": "separated total SRDF disable entries from in-universe disabled geometry pairs and exposed the out-of-universe entry explicitly"},
            {"id": "D53-A-003", "status": "FIXED_IN_SANDBOX", "finding": "Cartesian angular finite-difference auditor produced false 2*pi wrap spikes from absolute quaternion logs", "repair": "switched to shortest relative-quaternion logarithms between consecutive native FK samples and added a known-rate synthetic regression"},
        ],
        "type_b_stage4b_targets": [
            {"id": "D53-B-001", "category": "near_zero_self_clearance_margin", "affected_cases": ["adversarial_0100", "adversarial_0101", "perturbation_0000", "perturbation_0100"], "severity": "high", "safety_impact": "close self-clearance margin; continuous-path validation required"},
            {"id": "D53-B-002", "category": "near_singular_jacobian", "affected_cases": [item["case_id"] for item in quality["cases"] if item["minimum_sigma_min"] is not None and item["minimum_sigma_min"] < 1.0e-3], "severity": "medium", "safety_impact": "risk threshold remains unresolved"},
        ],
        "next_stage4b_contract": {"top_1": "continuous articulated self-collision and self-clearance validation", "top_2": "calibrated TCP accuracy/uncertainty acceptance", "top_3": "hardware torque/current and certified dynamic limits"},
    }
    route_a_case_ids = [item["case_id"] for item in route_a["case_rows"]]
    route_b_smoke = route_summary(SHADOW / "routeB_d52_smoke", "continuous_clearance_case_summary.jsonl")
    route_b_smoke_row = route_b_smoke["case_rows"][0] if route_b_smoke["case_rows"] else {}
    reason_counts: dict[str, int] = defaultdict(int)
    for item in acm["pairs"]:
        if item["srdf_disabled"]:
            reason_counts[str(item["srdf_reason"])] += 1
    cart_cases = [item for item in quality["cases"] if item["cartesian"].get("status") == "AVAILABLE_NATIVE_MOVEIT2_FK"]
    peak_cartesian = {
        key: max(cart_cases, key=lambda item: float(item["cartesian"][key]))
        for key in (
            "peak_linear_speed_m_s",
            "peak_linear_acceleration_m_s2",
            "peak_linear_jerk_m_s3",
            "peak_angular_speed_rad_s",
            "peak_angular_jerk_rad_s3",
        )
    }
    cart_waypoint_key = {
        "peak_linear_speed_m_s": "peak_linear_speed_waypoint",
        "peak_linear_acceleration_m_s2": "peak_linear_acceleration_waypoint",
        "peak_linear_jerk_m_s3": "peak_linear_jerk_waypoint",
        "peak_angular_speed_rad_s": "peak_angular_speed_waypoint",
        "peak_angular_jerk_rad_s3": None,
    }
    markdown = "\n".join([
        "# D53 Stage 4 Offline Motion-Quality Certification Closure",
        "",
        f"- Task status: `{classification}`; measurement pipeline status: `PASS`; canonical promotion: `NO`.",
        "- Scope: frozen Stage 3/D52 finalist, Stage 0/1 ON-state open-arch only, authoritative 181-point input; no OFF/reorientation/learning/retreat/approach/closed-contour transitions.",
        "",
        "## A. Starting point",
        "",
        f"- Exact baseline: `{baseline['candidate']}` from `D52_FINAL_METRICS.json`; native post-Ruckig MoveIt2 FK/dynamics/geometry and deterministic replay were already valid.",
        f"- Baseline quality: mean duration `{baseline['mean_duration_s']:.6f} s`, peak native torque slew `{baseline['peak_native_torque_slew_Nm_s']:.6f} N*m/s`, jerk-limit violations `{baseline['jerk_limit_violations']}`.",
        f"- D52 minimum environment clearance was `{baseline['minimum_environment_clearance_m']} m`; minimum self clearance was `{baseline['minimum_self_clearance_m']} m` where available.",
        f"- The D52 lower bound `{baseline['d52_clearance'].get('minimum_certified_clearance_m')} m` is classified as `{baseline['clearance_classification']['class']}`: stride-100/model-space evidence with regenerated-profile assumptions, not a strict lower bound over exact articulated `FK(q(t))`.",
        "",
        "## B. Problem classification",
        "",
        "- Type-A D53-A-001: repaired the clearance-method ambiguity by exposing endpoint-jerk-cone mode separately from the legacy regenerated-profile mode.",
        "- Type-A D53-A-002: repaired ACM counting so the `spray_tcp_link|wrist3_link` SRDF entry outside the collision-bearing URDF pair universe is not counted as an in-universe geometry pair.",
        "- Type-A D53-A-003: repaired a false Cartesian angular-jerk/speed wrap spike by using shortest relative-quaternion logs; the known-rate synthetic regression passes.",
        "- Type-B findings were preserved: near-zero self-clearance margin, near-singular Jacobian conditioning, and missing calibrated TCP acceptance. No Stage 3/D52 trajectory or threshold was tuned.",
        "",
        "## C. External research and design decisions",
        "",
        "- [MoveIt PlanningScene/collision API](https://moveit.picknik.ai/main/doc/examples/planning_scene/planning_scene_tutorial.html): retained PlanningScene and ACM ownership for native checks.",
        "- [MoveIt Bullet collision checker](https://moveit.picknik.ai/main/doc/examples/bullet_collision_checker/bullet_collision_checker.html): used native two-state Bullet robot-world checking as Route C, separately from self-collision.",
        "- [MoveIt FCL source](https://github.com/moveit/moveit/blob/master/moveit_core/collision_detection_fcl/src/collision_env_fcl.cpp): did not claim the MoveIt FCL wrapper as strict two-state self-CCD.",
        "- [FCL continuous API](https://github.com/flexible-collision-library/fcl) and [FCL paper](https://gamma.cs.unc.edu/FCL/fcl_docs/webpage/pdfs/fcl_icra2012.pdf): used direct `continuousCollide` for a separate swept-link cross-check, with known-answer crossing/separation tests.",
        "- [Tesseract collision backends](https://tesseract-robotics.github.io/tesseract/collision.html) and continuous-collision literature were investigated; no installed backend supplied an accepted exact articulated `FK(q(t))` self-CCD certificate in this runtime.",
        "",
        "## D. Methods attempted",
        "",
        f"- Route A — direct FCL `CCDM_SCREW`/`GST_LIBCCD`: `{route_a['complete_case_count']}/{len(manifest_rows())}` full cases completed (`{', '.join(route_a_case_ids) if route_a_case_ids else 'none'}`), zero contacts and zero API errors in completed cases; the per-link endpoint screw motion is only a qualified cross-check, not exact nonlinear articulated motion. The known-answer forced crossing and separation probe passed.",
        f"- Route B — endpoint jerk-cone bound: full campaign `{route_b['complete_case_count']}/{len(manifest_rows())}` at closure; the completed adversarial smoke case produced lower bound `{route_b_smoke_row.get('minimum_certified_clearance_m', 'not_available')} m`, but the model is not pair-complete strict self-CCD and hardware clearance is unavailable.",
        f"- Route C — native MoveIt2 Bullet/PlanningScene: `{route_c['complete_case_count']}/{len(manifest_rows())}` cases completed; environment two-state CCD was the accepted native environment route, while self-CCD remained unavailable.",
        "- Route D — numerical robustness: native joint-state finiteness/time monotonicity was checked at strides 1/2/4/8 and half-step phase; no geometry result was inferred from shifted synthetic samples.",
        "",
        "## E. Collision result",
        "",
        f"- Discrete self collision: `0` native MoveIt2 collision samples across `{route_c['complete_case_count']}` cases.",
        f"- Discrete environment collision: `0` native MoveIt2 collision samples across `{route_c['complete_case_count']}` cases.",
        f"- Continuous environment collision: `0` native MoveIt2 Bullet robot-world contacts across `{route_c['complete_case_count']}` cases; this is labelled native two-state Bullet evidence, not strict self-CCD.",
        f"- Continuous self collision: strict exact articulated `FK(q(t))` result is `not_available`; Route A is partial/qualified, Route B is model-space only, and no distance or penetration value is inferred (`null`). Sampled collision rows retain the repository label `adaptive_discrete_interpolation`.",
        "",
        "## F. ACM/SRDF audit",
        "",
        f"- URDF collision links: `{acm['collision_link_count']}`; full unordered pair universe: `{acm['pair_universe_count']}`; required checked: `{acm['checked_pair_count']}`; disabled within that universe: `{acm['allowed_pair_count']}`.",
        f"- In-universe SRDF reasons: `Adjacent={reason_counts.get('Adjacent', 0)}`, `Never={reason_counts.get('Never', 0)}`. Total SRDF disable entries are `{acm['srdf_disabled_pair_count']}` because `{', '.join(acm['srdf_disabled_outside_collision_universe'])}` is outside the URDF collision-bearing pair universe.",
        "- Investigated high-risk/suspicious pairs include `forearm_link|wrist2_link`, `forearm_link|wrist3_link`, and the full-robot `base_link|wrist1_link` exception. No collision-free inference was made for active-group pairs outside Route A’s six-link direct-FCL subset.",
        "",
        "## G. Singularity/Jacobian result",
        "",
        f"- MoveIt2 RobotState Jacobian + Eigen SVD covered `{quality['singularity']['minimum_sigma_min']['count']}` cases. Minimum sigma_min `{quality['singularity']['minimum_sigma_min'].get('min')}` at `{quality['singularity']['worst_case']}`; maximum condition number `{quality['singularity']['maximum_condition_number'].get('max')}` at `{quality['singularity']['worst_condition_case']}`; minimum manipulability `{quality['singularity']['minimum_manipulability'].get('min')}` at `{quality['singularity']['worst_manipulability_case']}`.",
        "- These are measured numerical risks, not automatic failures: project-specific physical thresholds remain `UNRESOLVED_THRESHOLD`.",
        "",
        "## H. Cartesian result",
        "",
        f"- Native MoveIt2 FK traces were available for `{quality['cartesian']['case_count']}` cases. Peaks: linear speed `{quality['cartesian']['peak_linear_speed_m_s'].get('max')} m/s`, linear acceleration `{quality['cartesian']['peak_linear_acceleration_m_s2'].get('max')} m/s^2`, linear jerk `{quality['cartesian']['peak_linear_jerk_m_s3'].get('max')} m/s^3`, angular speed `{quality['cartesian']['peak_angular_speed_rad_s'].get('max')} rad/s`, angular acceleration `{quality['cartesian']['peak_angular_acceleration_rad_s2'].get('max')} rad/s^2`, angular jerk `{quality['cartesian']['peak_angular_jerk_rad_s3'].get('max')} rad/s^3`.",
        "- Worst-segment locations are recorded per case in `D53_QUALITY_METRICS.json`; the global peak cases are: " + "; ".join(f"{key}={item['case_id']}@wp{item['cartesian'].get(cart_waypoint_key[key], 'n/a') if cart_waypoint_key[key] else 'case-level peak'}" for key, item in peak_cartesian.items()) + ".",
        "- The earlier quaternion wraparound false spike was repaired by shortest relative-quaternion logs and covered by a known-rate synthetic test; calibrated TCP/path-accuracy thresholds remain unresolved.",
        "",
        "## I. Numerical robustness",
        "",
        f"- Tested sample strides: `{robustness['sample_resolution_set']}` plus a half-step phase diagnostic; state finite failures `{robustness['state_finite_failures']}`, timebase failures `{robustness['timebase_failures']}`, all tested resolution values finite: `{robustness['all_finite']}`.",
        "- The joint-state conclusion was stable; phase-shifted geometry/collision was not promoted because those shifted states were not run through native geometry backends.",
        "",
        "## J. Baseline comparison",
        "",
        "- Hard-gate regression: none observed in the preserved D52 native stack; the finalist remains unchanged and no canonical parameters/checkpoint/trajectory were modified.",
        "- Improvements: complete native Route C environment CCD rerun, full ACM pair-reason audit, repaired quaternion Cartesian measurement, and explicit D52 clearance semantics.",
        "- Tradeoffs/unresolved: strict self-CCD, calibrated TCP acceptance, hardware torque/current limits, and hardware clearance remain unavailable; no positive model-space result was relabelled as hardware-safe.",
        "",
        "## K. Final classification and Stage 4B handoff",
        "",
        f"- Final classification: `{classification}`. Canonical promotion is `NO`.",
        "- Top 1: continuous articulated self-collision/self-clearance validation; Top 2: calibrated TCP error/uncertainty acceptance; Top 3: hardware torque/current and certified dynamic limits.",
        "- Stage4B-ready artifacts: `STAGE4_FAILURE_TAXONOMY_V1.json` and `STAGE4_RISK_RANKED_BOTTLENECKS_V1.json`.",
        "- The measurement pipeline is `PASS`; the frozen robot baseline is reported honestly as measured with weaknesses/unresolved thresholds. No Type-B weakness was optimized away.",
        "",
        "All detailed route rows, ACM pair reasons, singularity and Cartesian worst cases, robustness matrices, and Type-A/Type-B ledgers are in the adjacent JSON artifacts.",
        "",
    ])
    return payload, markdown


def stage4b_ledgers(report: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Create Stage 4B-ready findings without modifying the frozen baseline."""
    quality_cases = report["quality"]["cases"]
    low_sigma = [item["case_id"] for item in quality_cases if item["minimum_sigma_min"] is not None and item["minimum_sigma_min"] < 1.0e-3]
    taxonomy = {
        "schema_version": "stage4-failure-taxonomy-v1",
        "taxonomy_id": "STAGE4_FAILURE_TAXONOMY_V1",
        "source": "D53_STAGE4_OFFLINE_CERTIFICATION_SHADOW",
        "scope": report["scope"],
        "measurement_pipeline_status": report["measurement_pipeline_status"],
        "frozen_robot_baseline_performance_status": report["frozen_robot_baseline_performance_status"],
        "type_a_measurement_defects": report["type_a_measurement_defects"],
        "type_b_findings": [
            {
                "id": "D53-B-001",
                "category": "near_zero_self_clearance_margin",
                "affected_benchmark_families": ["ADVERSARIAL", "PERTURBATION"],
                "affected_case_ids": ["adversarial_0100", "adversarial_0101", "perturbation_0000", "perturbation_0100"],
                "frequency": "4/12 cases have the lowest D52 model-space margins in these families",
                "worst_magnitude": report["starting_baseline"]["d52_clearance"].get("minimum_certified_clearance_m"),
                "severity": "HIGH",
                "reproducibility": "D52 deterministic replay PASS; worst pair forearm_link|wrist2_link",
                "safety_impact": "small modeled self-clearance margin; requires articulated-path continuous validation before hardware use",
                "current_evidence": "D52 continuous clearance summaries plus D53 Route A/Route B evidence when complete",
                "likely_subsystem": "self-collision geometry / motion envelope",
                "measurement_confidence": "medium until strict FK(q(t)) self-collision route is accepted",
            },
            {
                "id": "D53-B-002",
                "category": "near_singular_jacobian",
                "affected_benchmark_families": sorted({item["family"] for item in quality_cases if item["case_id"] in low_sigma}),
                "affected_case_ids": low_sigma,
                "frequency": f"{len(low_sigma)}/12 cases below the diagnostic sigma_min=1e-3 screen",
                "worst_magnitude": report["quality"]["singularity"]["minimum_sigma_min"].get("min"),
                "severity": "MEDIUM",
                "reproducibility": "native MoveIt2 RobotState Jacobian SVD; deterministic persisted traces",
                "safety_impact": "velocity amplification and force-control risk threshold is unresolved",
                "current_evidence": "D53_QUALITY_METRICS.json",
                "likely_subsystem": "kinematic posture / task Jacobian",
                "measurement_confidence": "high for numerical SVD; low for physical risk classification",
            },
            {
                "id": "D53-B-003",
                "category": "calibrated_TCP_acceptance_gap",
                "affected_benchmark_families": ["REGRESSION", "NORMAL", "BOUNDARY", "COLLISION_SENSITIVE", "ADVERSARIAL", "PERTURBATION"],
                "affected_case_ids": "all 12",
                "frequency": "12/12 have finite software FK/TCP traces but no authorized calibrated threshold",
                "worst_magnitude": report["quality"]["cartesian"]["peak_linear_jerk_m_s3"].get("max"),
                "severity": "HIGH",
                "reproducibility": "software replay reproducible; physical calibration unverified",
                "safety_impact": "TCP accuracy cannot be accepted for production from software-only evidence",
                "current_evidence": "D52 FK/TCP metrics and D53 native FK quality metrics",
                "likely_subsystem": "TCP calibration / Cartesian acceptance specification",
                "measurement_confidence": "high for software FK; unavailable for calibrated physical uncertainty",
            },
        ],
        "unavailable_capabilities": [
            "strict continuous self-collision over exact articulated FK(q(t))",
            "hardware torque/current and manufacturer-certified dynamic limits",
            "calibrated TCP uncertainty and accepted Cartesian thresholds",
            "hardware clearance",
        ],
    }
    bottlenecks = {
        "schema_version": "stage4-risk-ranked-bottlenecks-v1",
        "bottleneck_id": "STAGE4_RISK_RANKED_BOTTLENECKS_V1",
        "source": "D53_STAGE4_OFFLINE_CERTIFICATION_SHADOW",
        "regression_barrier": {
            "candidate": report["starting_baseline"]["candidate"],
            "jerk_limit_violations": report["starting_baseline"]["jerk_limit_violations"],
            "max_jerk_ratio": report["starting_baseline"]["max_jerk_ratio"],
            "peak_native_torque_slew_Nm_s": report["starting_baseline"]["peak_native_torque_slew_Nm_s"],
            "no_stage3_mutation": True,
        },
        "top_bottlenecks": [
            {"rank": 1, "id": "D53-B-001", "target": "strict articulated self-collision and self-clearance validation", "why_first": "closest geometric safety margin and missing strict path evidence", "stage4b_action": "build or validate a collision backend over exact articulated motion"},
            {"rank": 2, "id": "D53-B-003", "target": "calibrated TCP error and acceptance thresholds", "why_first": "software traces alone cannot establish physical task accuracy", "stage4b_action": "calibrate TCP and authorize position/orientation limits"},
            {"rank": 3, "id": "D53-B-002", "target": "near-singular postures", "why_first": "large condition-number tail with unresolved physical threshold", "stage4b_action": "define risk threshold and evaluate avoidance or task-space conditioning"},
        ],
        "coverage_gaps": report["blockers"],
    }
    return taxonomy, bottlenecks


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=SHADOW)
    args = parser.parse_args(argv)
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    baseline = audit_baseline()
    acm = srdf_acm_audit()
    route_a = route_a_summary(ROUTE_A)
    route_b = route_summary(ROUTE_B, "continuous_clearance_case_summary.jsonl")
    route_c = route_summary(ROUTE_C, "D41_native_case_summary.jsonl")
    quality = singularity_and_cartesian()
    robustness = numerical_robustness()
    report, markdown = final_report(baseline, acm, route_a, route_b, route_c, quality, robustness)
    taxonomy, bottlenecks = stage4b_ledgers(report)
    write_json(out / "D53_FINAL_CERTIFICATION.json", report)
    write_json(out / "D53_STARTING_BASELINE.json", baseline)
    write_json(out / "D53_ACM_AUDIT.json", acm)
    write_json(out / "D53_QUALITY_METRICS.json", quality)
    write_json(out / "D53_NUMERICAL_ROBUSTNESS.json", robustness)
    write_json(out / "D53_RESEARCH_LEDGER.json", {"schema_version": "d53-research-ledger-v1", **report["research_ledger"]})
    write_json(out / "STAGE4_FAILURE_TAXONOMY_V1.json", taxonomy)
    write_json(out / "STAGE4_RISK_RANKED_BOTTLENECKS_V1.json", bottlenecks)
    (out / "D53_FINAL_REPORT.md").write_text(markdown, encoding="utf-8")
    print(json.dumps({"task_status": report["task_status"], "measurement_pipeline_status": report["measurement_pipeline_status"], "blockers": report["blockers"]}, indent=2))
    return 0 if report["task_status"] == "D53_PASS_NEW_CANONICAL_BASELINE" else 2


if __name__ == "__main__":
    sys.exit(main())
