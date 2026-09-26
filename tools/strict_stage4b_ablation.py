"""Strict leave-one-module-out analysis for the protected D47 B3 chain.

This is an additive, read-only analysis of already authenticated Stage 4
artifacts.  It never writes to the protected B3 directory.  The three D47
interventions are dependent: B2 and B3 consume the B1 alternate-branch
provider.  Therefore ``without_b1`` is explicitly recorded as the
dependency-constrained D46 fallback, while ``without_b2`` is reconstructed
case-by-case as B1 plus the independently eligible B3 intervention.

No IK query is silently inferred from a joint trajectory.  The D47 benchmark
starts after IK and evaluates supplied joint trajectories through native
MoveIt2/FK/PlanningScene/FCL.  IK success is consequently reported as
UNAVAILABLE, with the reason preserved in the report.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
D46 = ROOT / "outputs" / "D46_STAGE4A_SYSTEM_BASELINE_V1"
D47 = ROOT / "outputs" / "D47_STAGE4B_ROLLING_CHAMPION_V1"
OUT = ROOT / "outputs" / "STAGE4B_STRICT_ABLATION_20260903"
POSES = ROOT / "outputs" / "internal_wiper_moveit_inputs" / "open_arch_tcp_poses_base_link.csv"
SEED = ROOT / "outputs" / "internal_wiper_moveit_inputs" / "open_arch_seed_joints.csv"
STRICT_TRAJECTORY = ROOT / "outputs" / "stage3_h13_d41_offline_robot_certification" / "strict_replay" / "moveit_smoothed_joint_trajectory.csv"
CASE_RESULTS = D46 / "STAGE4_CASE_RESULTS.csv"
COLLISION_METHOD = "adaptive_discrete_interpolation"
CCD_STATUS = "separate_native_robot_world_two_state_backend; not_exact_articulated_self_ccd"
N = 1000
WAYPOINTS = 181
JOINTS = 6
JUMP_LIMIT_RAD = math.radians(20.0)
LOWER = np.asarray([-3.0543, -4.6251, -2.8274, -4.6251, -3.0543, -3.0543], dtype=float)
UPPER = np.asarray([3.0543, 1.4835, 2.8274, 1.4835, 3.0543, 3.0543], dtype=float)
VELOCITY_LIMITS = np.asarray([3.15, 3.15, 3.15, 3.2, 3.2, 3.2], dtype=float)
ACCELERATION_LIMITS = np.asarray([10.0] * JOINTS, dtype=float)
JERK_LIMITS = np.asarray([8.0] * JOINTS, dtype=float)

SOURCE_DIRS = {
    "d46": D46,
    "b1": D47 / "PROMOTED" / "B1",
    "b2": D47 / "PROMOTED" / "B2",
    "b3": D47 / "PROMOTED" / "B3",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def read_q(path: Path) -> np.ndarray:
    rows = read_csv(path)
    names = [f"j{i}_q" for i in range(1, 7)] if rows and "j1_q" in rows[0] else [f"q{i}" for i in range(1, 7)]
    array = np.asarray([[float(row[name]) for name in names] for row in rows], dtype=float)
    if array.shape != (WAYPOINTS, JOINTS) or not np.isfinite(array).all():
        raise RuntimeError(f"invalid_q:{path}:{array.shape}")
    return array


def read_time() -> np.ndarray:
    trajectory = STRICT_TRAJECTORY
    auth_path = D46 / "STAGE3_AUTHENTICATION_V1.json"
    if auth_path.is_file():
        auth = json.loads(auth_path.read_text(encoding="utf-8"))
        relative = auth.get("retained_stage3_robot", {}).get("post_ruckig_trajectory")
        if relative:
            trajectory = ROOT / str(relative)
    if not trajectory.is_file():
        raise RuntimeError(f"missing_authoritative_timebase:{trajectory}")
    rows = read_csv(trajectory)
    values = np.asarray([float(row["t"]) for row in rows], dtype=float)
    if values.shape != (WAYPOINTS,) or not np.isfinite(values).all() or np.any(np.diff(values) <= 0.0):
        raise RuntimeError("invalid_authoritative_timebase")
    return values


def read_targets() -> tuple[np.ndarray, np.ndarray]:
    rows = read_csv(POSES)
    position = np.asarray([[float(row[key]) for key in ("x", "y", "z")] for row in rows], dtype=float)
    quaternion = np.asarray([[float(row[key]) for key in ("qx", "qy", "qz", "qw")] for row in rows], dtype=float)
    if position.shape != (WAYPOINTS, 3) or quaternion.shape != (WAYPOINTS, 4):
        raise RuntimeError("invalid_authoritative_pose_inputs")
    return position, quaternion


def read_case_meta() -> list[dict[str, Any]]:
    rows = read_csv(CASE_RESULTS)
    if len(rows) != N:
        raise RuntimeError(f"case_count:{len(rows)}")
    result: list[dict[str, Any]] = []
    for row in rows:
        result.append(
            {
                "case_id": row["case_id"],
                "family": row["family"],
                "seed": int(row["seed"]),
                "time_scale": float(row["time_scale"]),
                "target_shift_m": json.loads(row.get("target_shift_m", "[0.0, 0.0, 0.0]")),
            }
        )
    if len({row["case_id"] for row in result}) != N or len({row["seed"] for row in result}) != N:
        raise RuntimeError("case_identity_or_seed_not_unique")
    return result


def source_paths(name: str) -> tuple[Path, Path, Path, Path]:
    directory = SOURCE_DIRS[name]
    case_metrics = directory / ("STAGE4_CASE_RESULTS.csv" if name == "d46" else "case_metrics.csv")
    return directory / "cases", directory / "native", directory / "fk" / "STAGE4A_FK_TRACE.csv", case_metrics


def load_native(source: str) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, int]], dict[str, dict[str, np.ndarray]], dict[str, dict[str, str]]]:
    _, native_dir, fk_path, metric_path = source_paths(source)
    summaries: dict[str, dict[str, Any]] = {}
    with (native_dir / "D41_native_case_summary.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                row = json.loads(line)
                summaries[str(row["case_id"])] = row
    if len(summaries) != N:
        raise RuntimeError(f"native_summary_count:{source}:{len(summaries)}")

    collision_counts: dict[str, dict[str, int]] = defaultdict(lambda: {"dense_environment": 0, "dense_self": 0})
    with (native_dir / "D41_native_segment_collision.csv").open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            case_id = str(row["case_id"])
            collision_counts[case_id]["dense_environment"] += str(row.get("discrete_sample_world_collision", "")).lower() == "true"
            collision_counts[case_id]["dense_self"] += str(row.get("discrete_sample_self_collision", "")).lower() == "true"

    fk_rows: dict[str, dict[str, list[Any]]] = defaultdict(lambda: {"waypoint": [], "position": [], "quaternion": []})
    with fk_path.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            case_id = str(row["case_id"])
            fk_rows[case_id]["waypoint"].append(int(row["waypoint"]))
            fk_rows[case_id]["position"].append([float(row[key]) for key in ("x_m", "y_m", "z_m")])
            fk_rows[case_id]["quaternion"].append([float(row[key]) for key in ("qx", "qy", "qz", "qw")])
    fk: dict[str, dict[str, np.ndarray]] = {}
    for case_id, row in fk_rows.items():
        if len(row["waypoint"]) != WAYPOINTS or row["waypoint"] != list(range(WAYPOINTS)):
            raise RuntimeError(f"fk_waypoint_count:{source}:{case_id}:{len(row['waypoint'])}")
        fk[case_id] = {
            "position": np.asarray(row["position"], dtype=float),
            "quaternion": np.asarray(row["quaternion"], dtype=float),
        }
    if len(fk) != N:
        raise RuntimeError(f"fk_case_count:{source}:{len(fk)}")
    metric_rows = {str(row["case_id"]): row for row in read_csv(metric_path)}
    if len(metric_rows) != N:
        raise RuntimeError(f"metric_case_count:{source}:{len(metric_rows)}")
    return summaries, collision_counts, fk, metric_rows


def quaternion_angle(a: np.ndarray, b: np.ndarray) -> float:
    dot = float(abs(np.dot(a, b) / max(np.linalg.norm(a) * np.linalg.norm(b), 1.0e-15)))
    return float(2.0 * math.acos(float(np.clip(dot, -1.0, 1.0))))


def derivatives(q: np.ndarray, t: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    velocity = np.gradient(q, t, axis=0, edge_order=1)
    acceleration = np.gradient(velocity, t, axis=0, edge_order=1)
    jerk = np.gradient(acceleration, t, axis=0, edge_order=1)
    return velocity, acceleration, jerk


def q_digest(q: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(q, dtype="<f8").tobytes(order="C")).hexdigest()


def stats(values: Iterable[float], unit: str, lower_is_better: bool | None = None) -> dict[str, Any]:
    array = np.asarray([float(value) for value in values if value is not None and math.isfinite(float(value))], dtype=float)
    if len(array) == 0:
        return {"status": "UNAVAILABLE", "count": 0, "unit": unit}
    payload: dict[str, Any] = {
        "status": "AVAILABLE",
        "count": int(len(array)),
        "unit": unit,
        "min": float(np.min(array)),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p95": float(np.quantile(array, 0.95)),
        "max": float(np.max(array)),
        "std": float(np.std(array)),
    }
    if lower_is_better is not None:
        payload["direction"] = "lower_is_better" if lower_is_better else "higher_is_better"
    return payload


def q_source(case_id: str, variant: str, sources: Mapping[str, str]) -> tuple[str, Path]:
    if variant == "full_b3":
        source = sources["b3"]
    elif variant == "without_b3":
        source = sources["b2"]
    elif variant == "without_b1":
        source = sources["d46"]
    elif variant == "without_b2":
        # B3's four eligible low-clearance cases are unchanged by B2, while
        # every other case is exactly B1 after B2 is removed.  This composes
        # an exact measured case-level intervention, not a synthetic metric.
        source = sources["b3"] if sources["b3_changed"][case_id] else sources["b1"]
    else:
        raise ValueError(variant)
    case_dir, _, _, _ = source_paths(source)
    return source, case_dir / f"{case_id}.csv"


def load_determinism(source: str) -> str:
    directory = SOURCE_DIRS[source]
    if source == "d46":
        payload = json.loads((directory / "STAGE4_REPRODUCIBILITY_REPORT_V1.json").read_text(encoding="utf-8"))
        return "PASS" if not payload.get("mismatches") else "FAIL"
    payload = json.loads((directory / "deterministic_replay.json").read_text(encoding="utf-8"))
    return str(payload.get("status", "UNAVAILABLE"))


def source_runtime(source: str) -> dict[str, Any]:
    """Return recorded artifact timing; it is not a new fair rerun timing."""
    # PROMOTED directories were copied after the shadow run, so their
    # filesystem creation times collapse the original interval.  Use the
    # immutable shadow run for the recorded timing proxy.
    directory = D46 if source == "d46" else D47 / f"{source.upper()}_SHADOW"
    native_summary = directory / "native" / "D41_native_case_summary.jsonl"
    final = directory / ("STAGE4_BASELINE_SCORECARD_V1.json" if source == "d46" else "metrics.json")
    if not native_summary.exists() or not final.exists():
        return {"status": "UNAVAILABLE", "value_s": None, "note": "missing recorded artifact timestamps"}
    elapsed = final.stat().st_mtime - native_summary.stat().st_ctime
    return {
        "status": "RECORDED_ARTIFACT_WALL_TIME_NOT_NEW_RERUN",
        "value_s": float(max(elapsed, 0.0)),
        "note": "filesystem timestamp proxy from native summary creation to final scorecard; not a controlled timing comparison",
    }


def make_variant_sources(meta: list[dict[str, Any]]) -> tuple[dict[str, str], dict[str, Any]]:
    source_keys = {key: key for key in ("d46", "b1", "b2", "b3")}
    b1_dir, _, _, _ = source_paths("b1")
    b2_dir, _, _, _ = source_paths("b2")
    b3_dir, _, _, _ = source_paths("b3")
    changed_b3: dict[str, bool] = {}
    changed_b2: dict[str, bool] = {}
    changed_b1: dict[str, bool] = {}
    for row in meta:
        case_id = row["case_id"]
        q0 = read_q(source_paths("d46")[0] / f"{case_id}.csv")
        q1 = read_q(b1_dir / f"{case_id}.csv")
        q2 = read_q(b2_dir / f"{case_id}.csv")
        q3 = read_q(b3_dir / f"{case_id}.csv")
        changed_b1[case_id] = not np.array_equal(q0, q1)
        changed_b2[case_id] = not np.array_equal(q1, q2)
        changed_b3[case_id] = not np.array_equal(q2, q3)
        if changed_b3[case_id] and not np.array_equal(q1, q2):
            raise RuntimeError(f"B3_not_independent_of_B2:{case_id}")
    source_keys["b3_changed"] = changed_b3  # type: ignore[assignment]
    proof = {
        "b1_changed_case_count": int(sum(changed_b1.values())),
        "b2_changed_case_count": int(sum(changed_b2.values())),
        "b3_changed_case_count": int(sum(changed_b3.values())),
        "b1_changed_case_ids": [row["case_id"] for row in meta if changed_b1[row["case_id"]]],
        "b2_changed_case_ids": [row["case_id"] for row in meta if changed_b2[row["case_id"]]],
        "b3_changed_case_ids": [row["case_id"] for row in meta if changed_b3[row["case_id"]]],
        "without_b2_composition": "B1 for all cases except exact B3-vs-B2 changed cases; B3 for those unchanged-by-B2 cases",
    }
    if proof["b3_changed_case_count"] != 4:
        raise RuntimeError(f"unexpected_b3_eligible_case_count:{proof['b3_changed_case_count']}")
    return source_keys, proof


def evaluate_variant(
    variant: str,
    meta: list[dict[str, Any]],
    sources: dict[str, str],
    native_cache: Mapping[str, tuple[dict[str, dict[str, Any]], dict[str, dict[str, int]], dict[str, dict[str, np.ndarray]], dict[str, dict[str, str]]]],
    base_time: np.ndarray,
    target_position: np.ndarray,
    target_quaternion: np.ndarray,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    case_rows: list[dict[str, Any]] = []
    for case in meta:
        case_id = case["case_id"]
        source, q_path = q_source(case_id, variant, sources)
        q = read_q(q_path)
        summary, dense_counts, fk, metric_rows = native_cache[source]
        native = summary[case_id]
        measured = metric_rows[case_id]
        fk_case = fk[case_id]
        t = base_time * float(case["time_scale"])
        velocity, acceleration, jerk = derivatives(q, t)
        target = target_position + np.asarray(case["target_shift_m"], dtype=float)[None, :]
        position_error = np.linalg.norm(fk_case["position"] - target, axis=1)
        orientation_error = np.asarray([quaternion_angle(actual, wanted) for actual, wanted in zip(fk_case["quaternion"], target_quaternion)], dtype=float)
        q_margin = np.minimum(q - LOWER[None, :], UPPER[None, :] - q)
        dense = dense_counts.get(case_id, {"dense_environment": 0, "dense_self": 0})
        # Use the source evaluator's already materialized case metrics for
        # collision and clearance semantics.  This keeps D46 and D47 on the
        # same field definition and avoids mixing the native summary's
        # adaptive-refinement minimum with the published clearance column.
        def int_field(key: str, fallback: int = 0) -> int:
            value = measured.get(key)
            return int(float(value)) if value not in (None, "") else fallback

        def float_field(key: str, fallback: float) -> float:
            value = measured.get(key)
            return float(value) if value not in (None, "") else fallback

        nominal_env = int_field("native_waypoint_environment_collision_count", int(native.get("nominal_waypoint_world_collision_count", 0)))
        nominal_self = int_field("native_waypoint_self_collision_count", int(native.get("nominal_waypoint_self_collision_count", 0)))
        ccd = int_field("environment_ccd_failures", int(native.get("native_continuous_segment_collision_count", 0)))
        dense = {
            "dense_environment": int_field("dense_environment_collision_samples", dense["dense_environment"]),
            "dense_self": int_field("dense_self_collision_samples", dense["dense_self"]),
        }
        adaptive_env = int(nominal_env > 0 or dense["dense_environment"] > 0)
        adaptive_self = int(nominal_self > 0 or dense["dense_self"] > 0)
        finite = bool(np.isfinite(q).all() and np.isfinite(velocity).all() and np.isfinite(acceleration).all() and np.isfinite(jerk).all())
        max_jump = float(np.max(np.abs(np.diff(q, axis=0))))
        max_speed_ratio = float(np.max(np.abs(velocity) / VELOCITY_LIMITS[None, :]))
        max_acc_ratio = float(np.max(np.abs(acceleration) / ACCELERATION_LIMITS[None, :]))
        max_jerk_ratio = float(np.max(np.abs(jerk) / JERK_LIMITS[None, :]))
        dt = np.diff(t)
        jerk_sq = np.sum(jerk * jerk, axis=1)
        jerk_ise = float(np.trapezoid(jerk_sq, t) if hasattr(np, "trapezoid") else np.trapz(jerk_sq, t))
        row = {
            "variant": variant,
            "case_id": case_id,
            "family": case["family"],
            "seed": case["seed"],
            "source": source,
            "trajectory_sha256": q_digest(q),
            "finite_state": finite,
            "adaptive_environment_collision": bool(adaptive_env),
            "adaptive_self_collision": bool(adaptive_self),
            "adaptive_collision": bool(adaptive_env or adaptive_self),
            "nominal_environment_collision_count": nominal_env,
            "nominal_self_collision_count": nominal_self,
            "dense_environment_collision_samples": int(dense["dense_environment"]),
            "dense_self_collision_samples": int(dense["dense_self"]),
            "native_robot_world_two_state_ccd_failures": ccd,
            "min_environment_clearance_m": float_field("min_environment_clearance_m", float(native["minimum_robot_world_distance_m"])),
            "min_self_clearance_m": float_field("min_self_clearance_m", float(native["minimum_self_distance_m"])),
            "joint_path_length_rad": float(np.sum(np.linalg.norm(np.diff(q, axis=0), axis=1))),
            "tcp_path_length_m": float(np.sum(np.linalg.norm(np.diff(fk_case["position"], axis=0), axis=1))),
            "jerk_ise_rad2_s5": jerk_ise,
            "jerk_rms_rad_s3": float(np.sqrt(np.mean(jerk * jerk))),
            "max_jerk_ratio": max_jerk_ratio,
            "max_abs_joint_jump_rad": max_jump,
            "joint_jump_events_over_20deg": int(np.sum(np.any(np.abs(np.diff(q, axis=0)) > JUMP_LIMIT_RAD, axis=1))),
            "max_velocity_ratio": max_speed_ratio,
            "max_acceleration_ratio": max_acc_ratio,
            "joint_limit_violation_count": int(np.sum(q_margin < 0.0)),
            "trajectory_duration_s": float(t[-1]),
            "terminal_position_error_m": float(position_error[-1]),
            "terminal_orientation_error_rad": float(orientation_error[-1]),
            "tcp_trajectory_error_p95_m": float(np.quantile(position_error, 0.95)),
            "tcp_trajectory_error_max_m": float(np.max(position_error)),
            "hard_constraint_clean": bool(finite and np.max(q_margin) >= 0.0 and max_speed_ratio <= 1.0 and max_acc_ratio <= 1.0 and max_jerk_ratio <= 1.0 and max_jump <= JUMP_LIMIT_RAD),
        }
        case_rows.append(row)

    collision_cases = [row for row in case_rows if row["adaptive_collision"]]
    hard_clean = [row for row in case_rows if row["hard_constraint_clean"]]
    deterministic_sources = {row["source"] for row in case_rows}
    replay = "PASS" if all(load_determinism(source) == "PASS" for source in deterministic_sources if source != "d46") and ("d46" not in deterministic_sources or load_determinism("d46") == "PASS") else "FAIL"
    aggregate = {
        "variant": variant,
        "case_count": len(case_rows),
        "family_counts": dict(Counter(row["family"] for row in case_rows)),
        "source_case_counts": dict(Counter(row["source"] for row in case_rows)),
        "ik_success_rate": {
            "status": "UNAVAILABLE",
            "value": None,
            "reason": "D47 consumes authenticated joint trajectories; no MoveIt set_from_ik query was executed by this benchmark, so FK validity is not relabelled as IK success.",
        },
        "trajectory_input_completeness_rate": float(np.mean([row["finite_state"] for row in case_rows])),
        "collision": {
            "method": COLLISION_METHOD,
            "adaptive_collision_case_count": len(collision_cases),
            "adaptive_collision_rate": float(len(collision_cases) / len(case_rows)),
            "adaptive_environment_collision_rate": float(np.mean([row["adaptive_environment_collision"] for row in case_rows])),
            "adaptive_self_collision_rate": float(np.mean([row["adaptive_self_collision"] for row in case_rows])),
            "native_robot_world_two_state_ccd_failures": int(sum(row["native_robot_world_two_state_ccd_failures"] for row in case_rows)),
            "ccd_scope": CCD_STATUS,
        },
        "minimum_safety_distance_model_scope": {
            "environment_m": stats((row["min_environment_clearance_m"] for row in case_rows), "m", lower_is_better=False),
            "self_m": stats((row["min_self_clearance_m"] for row in case_rows), "m", lower_is_better=False),
            "physical_clearance_threshold": "UNRESOLVED_THRESHOLD",
        },
        "trajectory_length": {
            "joint_space_rad": stats((row["joint_path_length_rad"] for row in case_rows), "rad", lower_is_better=True),
            "tcp_space_m": stats((row["tcp_path_length_m"] for row in case_rows), "m", lower_is_better=True),
        },
        "smoothness": {
            "jerk_ise": stats((row["jerk_ise_rad2_s5"] for row in case_rows), "rad^2/s^5", lower_is_better=True),
            "jerk_rms": stats((row["jerk_rms_rad_s3"] for row in case_rows), "rad/s^3", lower_is_better=True),
            "max_jerk_ratio": stats((row["max_jerk_ratio"] for row in case_rows), "ratio", lower_is_better=True),
        },
        "joint_jump": {
            "max_abs_jump": stats((row["max_abs_joint_jump_rad"] for row in case_rows), "rad", lower_is_better=True),
            "jump_events_over_20deg": int(sum(row["joint_jump_events_over_20deg"] for row in case_rows)),
            "jump_case_rate_over_20deg": float(np.mean([row["joint_jump_events_over_20deg"] > 0 for row in case_rows])),
        },
        "computation_time": {
            "status": "RECORDED_ARTIFACT_PROXY",
            "sources": {source: source_runtime(source) for source in sorted(deterministic_sources)},
            "note": "This report reuses verified native artifacts for D46/B1/B2/B3. without_B2 is case-level exact composition and has no single-run wall clock; no runtime is invented.",
        },
        "stability": {
            "finite_case_rate": float(np.mean([row["finite_state"] for row in case_rows])),
            "hard_constraint_clean_rate": float(len(hard_clean) / len(case_rows)),
            "deterministic_replay": replay,
            "status": "PASS" if len(hard_clean) == len(case_rows) and replay == "PASS" else "FAIL",
            "note": "Execution stability means finite state, hard motion limits, 20-degree adjacent-joint continuity and verified replay; collision is reported separately.",
        },
        "accuracy_context": {
            "terminal_position_error_m": stats((row["terminal_position_error_m"] for row in case_rows), "m", lower_is_better=True),
            "terminal_orientation_error_rad": stats((row["terminal_orientation_error_rad"] for row in case_rows), "rad", lower_is_better=True),
            "tcp_trajectory_error_p95_m": stats((row["tcp_trajectory_error_p95_m"] for row in case_rows), "m", lower_is_better=True),
            "tcp_trajectory_error_max_m": stats((row["tcp_trajectory_error_max_m"] for row in case_rows), "m", lower_is_better=True),
        },
    }
    return case_rows, aggregate


def delta(value: Any, baseline: Any) -> float | None:
    if value is None or baseline is None:
        return None
    return float(value) - float(baseline)


def compare(aggregates: Mapping[str, dict[str, Any]]) -> dict[str, Any]:
    base = aggregates["full_b3"]
    rows: list[dict[str, Any]] = []
    for variant, payload in aggregates.items():
        env = payload["minimum_safety_distance_model_scope"]["environment_m"]
        self_clear = payload["minimum_safety_distance_model_scope"]["self_m"]
        joint_len = payload["trajectory_length"]["joint_space_rad"]
        tcp_len = payload["trajectory_length"]["tcp_space_m"]
        smooth = payload["smoothness"]["jerk_ise"]
        jump = payload["joint_jump"]["max_abs_jump"]
        rows.append(
            {
                "variant": variant,
                "delta_vs_full": {
                    "adaptive_collision_rate": delta(payload["collision"]["adaptive_collision_rate"], base["collision"]["adaptive_collision_rate"]),
                    "min_environment_clearance_m": delta(env.get("min"), base["minimum_safety_distance_model_scope"]["environment_m"].get("min")),
                    "min_self_clearance_m": delta(self_clear.get("min"), base["minimum_safety_distance_model_scope"]["self_m"].get("min")),
                    "median_joint_path_length_rad": delta(joint_len.get("median"), base["trajectory_length"]["joint_space_rad"].get("median")),
                    "median_tcp_path_length_m": delta(tcp_len.get("median"), base["trajectory_length"]["tcp_space_m"].get("median")),
                    "median_jerk_ise_rad2_s5": delta(smooth.get("median"), base["smoothness"]["jerk_ise"].get("median")),
                    "median_max_abs_jump_rad": delta(jump.get("median"), base["joint_jump"]["max_abs_jump"].get("median")),
                    "hard_constraint_clean_rate": delta(payload["stability"]["hard_constraint_clean_rate"], base["stability"]["hard_constraint_clean_rate"]),
                },
            }
        )
    return {"baseline": "full_b3", "rows": rows}


def markdown(report: dict[str, Any]) -> str:
    aggregates = report["aggregates"]
    comparison = report["comparison"]["rows"]
    labels = {
        "full_b3": "完整 B3 (baseline)",
        "without_b1": "移除 B1",
        "without_b2": "移除 B2",
        "without_b3": "移除 B3",
    }
    lines = [
        "# D47 严格模块消融（software/model shadow）",
        "",
        "## 结论",
        "",
        "当前完整方案按磁盘权威定义为 B1 → B2 → B3。所有变体使用同一 1000-case、六类 family、181 状态、固定 seed、固定初始状态、同一 MoveIt2/FK/PlanningScene/FCL 后端。B3 canonical 未被修改。",
        "",
        "IK 成功率没有被伪造：D47 的 benchmark 输入已经是关节轨迹，本次没有执行 MoveIt `set_from_ik`，因此该列为 `UNAVAILABLE`。轨迹完整性和 FK/碰撞测量仍单独报告。",
        "",
        "## 核心结果",
        "",
        "| 变体 | IK 成功率 | adaptive 碰撞率 | 环境最小间隙 m | 自碰最小间隙 m | 关节路径中位数 rad | TCP 路径中位数 m | jerk ISE 中位数 | 最大关节跳变中位数 rad | 稳定性 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for key in ("full_b3", "without_b1", "without_b2", "without_b3"):
        p = aggregates[key]
        env = p["minimum_safety_distance_model_scope"]["environment_m"]
        self_clear = p["minimum_safety_distance_model_scope"]["self_m"]
        row = "| {label} | UNAVAILABLE | {collision:.3f} | {env:.6f} | {selfc:.6f} | {jlen:.6f} | {tlen:.6f} | {jerk:.6g} | {jump:.6f} | {stable} |".format(
            label=labels[key],
            collision=p["collision"]["adaptive_collision_rate"],
            env=env["min"],
            selfc=self_clear["min"],
            jlen=p["trajectory_length"]["joint_space_rad"]["median"],
            tlen=p["trajectory_length"]["tcp_space_m"]["median"],
            jerk=p["smoothness"]["jerk_ise"]["median"],
            jump=p["joint_jump"]["max_abs_jump"]["median"],
            stable=p["stability"]["status"],
        )
        lines.append(row)
    lines += [
        "",
        "## 系统级判读",
        "",
        "- B1 是依赖性关键模块：移除它只能回退到 D46，因为 B2/B3 的安全分支提供者不再存在；此变体的结果不能被解释为 B1 的纯独立效应，但它准确表示去掉 B1 后完整系统的可运行 fallback。",
        "- B2 是安全关键模块：其移除只保留 B1，并保留 B3 对四个未被 B2 改写的低间隙 case 的处理；碰撞和负间隙回归，说明 B2 的系统贡献不是单一平滑指标能替代的。",
        "- B3 是间隙裕度模块：其移除不必然改变 collision rate，但环境/自间隙下界退回 B2，说明它贡献的是安全裕度而不是基本可行性；不能因碰撞率不变判定 B3 无效。",
        "- `continuous self-collision`、物理安全阈值、硬件/扭矩仍不在本消融的可声明范围内。碰撞主指标统一标为 `adaptive_discrete_interpolation`；native robot-world two-state CCD 另列，不等同于精确 articulated self-CCD。",
        "",
        "## 复现与证据",
        "",
        f"- case/seed/初始状态协议：`{report['paths']['protocol']}`",
        f"- 机器可读总表：`{report['paths']['summary']}`",
        f"- 逐 case 指标：`{report['paths']['case_metrics']}`",
        f"- 变体来源证明：`{report['paths']['composition']}`",
        "",
        "注意：B1/B2/B3/D46 使用已验证的历史 native artifacts；without_B2 是由已测 case-level 轨迹精确组合而成，因而没有被冒充成一次独立运行的 wall-clock。计算时间列只保留记录到的 artifact 时间代理；若需要可发表级 runtime 比较，应在同一干净进程中顺序重跑全部变体。",
        "",
        "## 与完整 baseline 的差值",
        "",
        "| 变体 | Δ adaptive 碰撞率 | Δ 环境最小间隙 m | Δ 自间隙 m | Δ jerk ISE 中位数 | Δ 最大跳变中位数 rad | Δ 稳定性清洁率 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for item in comparison:
        v = item["variant"]
        d = item["delta_vs_full"]
        lines.append("| {label} | {a:.3f} | {b:.6f} | {c:.6f} | {d1:.6g} | {e:.6f} | {f:.3f} |".format(label=labels[v], a=d["adaptive_collision_rate"] or 0.0, b=d["min_environment_clearance_m"] or 0.0, c=d["min_self_clearance_m"] or 0.0, d1=d["median_jerk_ise_rad2_s5"] or 0.0, e=d["median_max_abs_jump_rad"] or 0.0, f=d["hard_constraint_clean_rate"] or 0.0))
    return "\n".join(lines) + "\n"


def main() -> int:
    started = time.perf_counter()
    meta = read_case_meta()
    base_time = read_time()
    target_position, target_quaternion = read_targets()
    sources, composition = make_variant_sources(meta)

    # Hash only fixed authority inputs and the compact case/seed/initial-state
    # identity.  Temporary/generated files are not recursively hashed.
    initial_rows: list[dict[str, Any]] = []
    for row in meta:
        q = read_q(source_paths("d46")[0] / f"{row['case_id']}.csv")
        initial_rows.append({"case_id": row["case_id"], "q0": [float(value) for value in q[0]]})
    protocol = {
        "schema_version": "stage4b-strict-ablation-protocol-v1",
        "baseline": "full_b3",
        "protected_canonical": "outputs/D47_STAGE4B_ROLLING_CHAMPION_V1/PROMOTED/B3",
        "scope": "Stage 0/1 ON-state open-arch only",
        "case_count": N,
        "waypoint_count": WAYPOINTS,
        "joint_count": JOINTS,
        "family_counts": dict(Counter(row["family"] for row in meta)),
        "case_identity_sha256": sha256(CASE_RESULTS),
        "pose_input_sha256": sha256(POSES),
        "seed_input_sha256": sha256(SEED),
        "initial_state_identity_sha256": hashlib.sha256(json.dumps(initial_rows, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest(),
        "random_seed_policy": "frozen per-case seeds from STAGE4_CASE_RESULTS.csv; no new random draws",
        "evaluation_backend": "native MoveIt2 PlanningScene/FK plus FCL distances and recorded native collision outputs",
        "collision_method": COLLISION_METHOD,
        "ccd": CCD_STATUS,
        "clearance": "model-space FCL distance; physical acceptance threshold unresolved",
        "ik": "not_run; no success rate inferred",
        "module_variants": {
            "full_b3": "B1+B2+B3",
            "without_b1": "D46 dependency-constrained fallback; B2/B3 cannot operate without B1 branch provider",
            "without_b2": "B1+B3, exact case-level composition",
            "without_b3": "B1+B2",
        },
    }
    for variant in ("b1", "b2", "b3"):
        variant_initial = []
        for row in meta:
            q = read_q(source_paths(variant)[0] / f"{row['case_id']}.csv")
            variant_initial.append({"case_id": row["case_id"], "q0": [float(value) for value in q[0]]})
        variant_hash = hashlib.sha256(json.dumps(variant_initial, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
        protocol.setdefault("initial_state_checks", {})[variant] = {
            "identity_sha256": variant_hash,
            "matches_d46": variant_hash == protocol["initial_state_identity_sha256"],
        }
    if not all(item["matches_d46"] for item in protocol["initial_state_checks"].values()):
        raise RuntimeError("initial_state_changed_across_existing_variants")
    OUT.mkdir(parents=True, exist_ok=True)
    write_json(OUT / "protocol.json", protocol)
    native_cache = {source: load_native(source) for source in ("d46", "b1", "b2", "b3")}
    all_case_rows: list[dict[str, Any]] = []
    aggregates: dict[str, dict[str, Any]] = {}
    for variant in ("full_b3", "without_b1", "without_b2", "without_b3"):
        rows, aggregate = evaluate_variant(variant, meta, sources, native_cache, base_time, target_position, target_quaternion)
        all_case_rows.extend(rows)
        aggregates[variant] = aggregate
    fieldnames = list(all_case_rows[0].keys())
    write_csv(OUT / "case_metrics.csv", all_case_rows, fieldnames)
    comparison = compare(aggregates)
    report = {
        "schema_version": "stage4b-strict-ablation-report-v1",
        "status": "COMPLETE_WITH_IK_UNAVAILABLE_AND_RUNTIME_REUSE_BOUNDARY",
        "created_at_local": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "elapsed_analysis_s": float(time.perf_counter() - started),
        "protocol": protocol,
        "composition_proof": composition,
        "aggregates": aggregates,
        "comparison": comparison,
        "claim_fence": {
            "collision_method": COLLISION_METHOD,
            "continuous_self_collision": "NOT_AVAILABLE",
            "physical_clearance": "UNRESOLVED_THRESHOLD",
            "hardware": "UNVERIFIED",
            "promotion": "NO_PROMOTION",
        },
        "paths": {
            "protocol": str((OUT / "protocol.json").relative_to(ROOT)),
            "summary": str((OUT / "summary.json").relative_to(ROOT)),
            "case_metrics": str((OUT / "case_metrics.csv").relative_to(ROOT)),
            "composition": str((OUT / "composition_proof.json").relative_to(ROOT)),
        },
    }
    write_json(OUT / "composition_proof.json", composition)
    write_json(OUT / "summary.json", report)
    (OUT / "REPORT.md").write_text(markdown(report), encoding="utf-8", newline="\n")
    print(json.dumps({"status": report["status"], "output": str(OUT), "variants": list(aggregates), "elapsed_analysis_s": report["elapsed_analysis_s"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
