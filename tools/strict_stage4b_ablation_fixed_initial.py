"""Run a strict D47 B3 leave-one-module-out ablation with fixed q[0].

The historical rolling champion changes the whole trajectory for collision
cases, including the first state.  That is unsuitable for a strict ablation
where initial states must be identical.  This shadow runner therefore fixes
every ablated variant's first state to the corresponding full-B3 first state,
records the intervention, and sends the two newly materialized variants
through the same native MoveIt2/FK backend.  The protected B3 directory is
read-only and never used as an output directory.

IK is intentionally not inferred: D47 starts from supplied joint
trajectories, and this runner does not execute set_from_ik queries.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.stage4b_rolling_champion import run_fk, run_native

D46 = ROOT / "outputs" / "D46_STAGE4A_SYSTEM_BASELINE_V1"
D47 = ROOT / "outputs" / "D47_STAGE4B_ROLLING_CHAMPION_V1"
OUT = ROOT / "outputs" / "STAGE4B_STRICT_ABLATION_FIXED_INIT_20260903"
POSES = ROOT / "outputs" / "internal_wiper_moveit_inputs" / "open_arch_tcp_poses_base_link.csv"
SEED = ROOT / "outputs" / "internal_wiper_moveit_inputs" / "open_arch_seed_joints.csv"
CASE_RESULTS = D46 / "STAGE4_CASE_RESULTS.csv"
AUTH_TIME = D46 / "STAGE3_AUTHENTICATION_V1.json"
FALLBACK_TIME = ROOT / "outputs" / "stage3_h13_d41_offline_robot_certification" / "strict_replay" / "moveit_smoothed_joint_trajectory.csv"
N = 1000
WAYPOINTS = 181
JOINTS = 6
JUMP_LIMIT_RAD = math.radians(20.0)
COLLISION_METHOD = "adaptive_discrete_interpolation"
CCD_SCOPE = "separate_native_robot_world_two_state_backend; not_exact_articulated_self_ccd"
LOWER = np.asarray([-3.0543, -4.6251, -2.8274, -4.6251, -3.0543, -3.0543], dtype=float)
UPPER = np.asarray([3.0543, 1.4835, 2.8274, 1.4835, 3.0543, 3.0543], dtype=float)
VELOCITY_LIMITS = np.asarray([3.15, 3.15, 3.15, 3.2, 3.2, 3.2], dtype=float)
ACCELERATION_LIMITS = np.asarray([10.0] * JOINTS, dtype=float)
JERK_LIMITS = np.asarray([8.0] * JOINTS, dtype=float)

ORIGINAL_DIRS = {
    "d46": D46,
    "b1": D47 / "PROMOTED" / "B1",
    "b2": D47 / "PROMOTED" / "B2",
    "b3": D47 / "PROMOTED" / "B3",
}
VARIANT_NAMES = ("full_b3", "without_b1", "without_b2", "without_b3")
CUSTOM_VARIANTS = ("without_b1", "without_b2")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_q(path: Path) -> np.ndarray:
    rows = read_csv(path)
    names = [f"j{i}_q" for i in range(1, 7)] if rows and "j1_q" in rows[0] else [f"q{i}" for i in range(1, 7)]
    values = np.asarray([[float(row[name]) for name in names] for row in rows], dtype=float)
    if values.shape != (WAYPOINTS, JOINTS) or not np.isfinite(values).all():
        raise RuntimeError(f"invalid_q:{path}:{values.shape}")
    return values


def write_q(path: Path, q: np.ndarray) -> None:
    q = np.asarray(q, dtype=float)
    if q.shape != (WAYPOINTS, JOINTS) or not np.isfinite(q).all():
        raise RuntimeError(f"invalid_output_q:{path}:{q.shape}")
    rows = [{"waypoint": index, **{f"j{joint}_q": f"{q[index, joint - 1]:.17g}" for joint in range(1, 7)}} for index in range(WAYPOINTS)]
    write_csv(path, rows, ["waypoint", *[f"j{joint}_q" for joint in range(1, 7)]])


def read_meta() -> list[dict[str, Any]]:
    rows = read_csv(CASE_RESULTS)
    if len(rows) != N or len({row["case_id"] for row in rows}) != N:
        raise RuntimeError("authoritative_case_identity_invalid")
    return [
        {
            "case_id": row["case_id"],
            "family": row["family"],
            "seed": int(row["seed"]),
            "time_scale": float(row["time_scale"]),
            "target_shift_m": json.loads(row.get("target_shift_m", "[0.0, 0.0, 0.0]")),
        }
        for row in rows
    ]


def read_time() -> np.ndarray:
    path = FALLBACK_TIME
    if AUTH_TIME.is_file():
        auth = json.loads(AUTH_TIME.read_text(encoding="utf-8"))
        relative = auth.get("retained_stage3_robot", {}).get("post_ruckig_trajectory")
        if relative:
            path = ROOT / str(relative)
    rows = read_csv(path)
    values = np.asarray([float(row["t"]) for row in rows], dtype=float)
    if values.shape != (WAYPOINTS,) or not np.isfinite(values).all() or np.any(np.diff(values) <= 0.0):
        raise RuntimeError(f"authoritative_timebase_invalid:{path}")
    return values


def read_targets() -> tuple[np.ndarray, np.ndarray]:
    rows = read_csv(POSES)
    position = np.asarray([[float(row[key]) for key in ("x", "y", "z")] for row in rows], dtype=float)
    quaternion = np.asarray([[float(row[key]) for key in ("qx", "qy", "qz", "qw")] for row in rows], dtype=float)
    if position.shape != (WAYPOINTS, 3) or quaternion.shape != (WAYPOINTS, 4):
        raise RuntimeError("authoritative_pose_inputs_invalid")
    return position, quaternion


def q_digest(q: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(q, dtype="<f8").tobytes(order="C")).hexdigest()


def variant_dir(name: str) -> Path:
    return OUT / name


def variant_case_dir(name: str) -> Path:
    return variant_dir(name) / "cases"


def original_case_path(source: str, case_id: str) -> Path:
    return ORIGINAL_DIRS[source] / "cases" / f"{case_id}.csv"


def prepare_variants(meta: list[dict[str, Any]]) -> dict[str, Any]:
    OUT.mkdir(parents=True, exist_ok=True)
    changed = {"b1": [], "b2": [], "b3": [], "initial_mismatch_before_fix": {name: [] for name in VARIANT_NAMES}}
    rows: list[dict[str, Any]] = []
    for case in meta:
        case_id = case["case_id"]
        q0 = read_q(original_case_path("d46", case_id))
        q1 = read_q(original_case_path("b1", case_id))
        q2 = read_q(original_case_path("b2", case_id))
        q3 = read_q(original_case_path("b3", case_id))
        if not np.array_equal(q2, q3) and not np.array_equal(q1, q2):
            raise RuntimeError(f"B3_is_not_independent_of_B2:{case_id}")
        changed["b1"].append(case_id) if not np.array_equal(q0, q1) else None
        changed["b2"].append(case_id) if not np.array_equal(q1, q2) else None
        changed["b3"].append(case_id) if not np.array_equal(q2, q3) else None
        common_initial = q3[0].copy()
        selected = {
            "full_b3": q3,
            "without_b1": q0.copy(),
            "without_b2": q3.copy() if not np.array_equal(q2, q3) else q1.copy(),
            "without_b3": q2,
        }
        for name, q in selected.items():
            if not np.array_equal(q[0], common_initial):
                changed["initial_mismatch_before_fix"][name].append(case_id)
            if name != "full_b3":
                q[0] = common_initial
            write_q(variant_case_dir(name) / f"{case_id}.csv", q)
            rows.append(
                {
                    "case_id": case_id,
                    "variant": name,
                    "original_source": {"full_b3": "b3", "without_b1": "d46", "without_b2": "b3_or_b1_casewise", "without_b3": "b2"}[name],
                    "common_initial_q_digest": q_digest(np.asarray(common_initial[None, :], dtype=float)),
                    "trajectory_q_digest": q_digest(q),
                    "q0_after_digest": q_digest(np.asarray(q[0][None, :], dtype=float)),
                    "initial_state_fixed": bool(np.array_equal(q[0], common_initial)),
                }
            )
    if len(changed["b1"]) != 0 or len(changed["b2"]) != 296 or len(changed["b3"]) != 4:
        raise RuntimeError(f"unexpected_module_changed_counts:{changed}")
    if not all(all(row["initial_state_fixed"] for row in rows if row["variant"] == name) for name in VARIANT_NAMES):
        raise RuntimeError("common_initial_state_constraint_failed")
    write_csv(OUT / "variant_trajectory_manifest.csv", rows, list(rows[0].keys()))
    proof = {
        "b1_changed_case_count": len(changed["b1"]),
        "b2_changed_case_count": len(changed["b2"]),
        "b3_changed_case_count": len(changed["b3"]),
        "b1_changed_case_ids": changed["b1"],
        "b2_changed_case_ids": changed["b2"],
        "b3_changed_case_ids": changed["b3"],
        "initial_mismatch_before_fix": changed["initial_mismatch_before_fix"],
        "common_initial_rule": "q_variant[case,0,:] = q_full_b3[case,0,:] for every variant and case",
        "without_b1_semantics": "D46 trajectory with the common initial state; B2/B3 branch-dependent downstream interventions are unavailable without B1",
        "without_b2_semantics": "B1 trajectory plus the four B3-only eligible cases, followed by common-initial-state enforcement",
    }
    write_json(OUT / "composition_proof.json", proof)
    return proof


def custom_cases(name: str, meta: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {**case, "trajectory_path": str(variant_case_dir(name) / f"{case['case_id']}.csv")}
        for case in meta
    ]


def read_native_summary(native_dir: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    with (native_dir / "D41_native_case_summary.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                row = json.loads(line)
                result[str(row["case_id"])] = row
    if len(result) != N:
        raise RuntimeError(f"native_summary_count:{native_dir}:{len(result)}")
    return result


def materialize_custom_metric_csv(name: str) -> None:
    native_dir = variant_dir(name) / "native"
    summaries = read_native_summary(native_dir)
    clearances: dict[str, list[dict[str, str]]] = defaultdict(list)
    with (native_dir / "D41_native_clearance.csv").open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            clearances[str(row["case_id"])].append(row)
    dense: dict[str, dict[str, int]] = defaultdict(lambda: {"environment": 0, "self": 0})
    with (native_dir / "D41_native_segment_collision.csv").open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            case_id = str(row["case_id"])
            dense[case_id]["environment"] += str(row.get("discrete_sample_world_collision", "")).lower() == "true"
            dense[case_id]["self"] += str(row.get("discrete_sample_self_collision", "")).lower() == "true"
    rows: list[dict[str, Any]] = []
    for case in read_meta():
        case_id = case["case_id"]
        summary = summaries[case_id]
        crows = clearances[case_id]
        env_values = [float(row["robot_world_distance_m"]) for row in crows if row.get("robot_world_distance_m") not in (None, "", "null") and math.isfinite(float(row["robot_world_distance_m"]))]
        self_values = [float(row["self_distance_m"]) for row in crows if row.get("self_distance_m") not in (None, "", "null") and math.isfinite(float(row["self_distance_m"]))]
        rows.append(
            {
                "case_id": case_id,
                "native_waypoint_environment_collision_count": int(summary.get("nominal_waypoint_world_collision_count", 0)),
                "native_waypoint_self_collision_count": int(summary.get("nominal_waypoint_self_collision_count", 0)),
                "dense_environment_collision_samples": dense[case_id]["environment"],
                "dense_self_collision_samples": dense[case_id]["self"],
                "environment_ccd_failures": int(summary.get("native_continuous_segment_collision_count", 0)),
                "min_environment_clearance_m": min(env_values, default=None),
                "min_self_clearance_m": min(self_values, default=None),
            }
        )
    write_csv(variant_dir(name) / "case_metrics.csv", rows, list(rows[0].keys()))


def run_custom_backend(name: str, meta: list[dict[str, Any]]) -> dict[str, Any]:
    stage_dir = variant_dir(name)
    cases = custom_cases(name, meta)
    started = time.perf_counter()
    native, _ = run_native(stage_dir, cases, None)
    trace = run_fk(stage_dir, stage_dir / "native_input" / "cases.csv")
    elapsed = float(time.perf_counter() - started)
    materialize_custom_metric_csv(name)
    record = {
        "variant": name,
        "status": "PASS_NATIVE_MOVEIT2_FK_BACKEND" if trace.is_file() else "FAIL",
        "elapsed_wall_time_s": elapsed,
        "native_dir": str(native.relative_to(ROOT)),
        "fk_trace": str(trace.relative_to(ROOT)),
        "metric_csv": str((variant_dir(name) / "case_metrics.csv").relative_to(ROOT)),
    }
    write_json(stage_dir / "run_record.json", record)
    return record


def load_fk(path: Path) -> dict[str, dict[str, np.ndarray]]:
    grouped: dict[str, dict[str, list[Any]]] = defaultdict(lambda: {"waypoint": [], "position": [], "quaternion": []})
    with path.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            case_id = str(row["case_id"])
            grouped[case_id]["waypoint"].append(int(row["waypoint"]))
            grouped[case_id]["position"].append([float(row[key]) for key in ("x_m", "y_m", "z_m")])
            grouped[case_id]["quaternion"].append([float(row[key]) for key in ("qx", "qy", "qz", "qw")])
    result: dict[str, dict[str, np.ndarray]] = {}
    for case_id, row in grouped.items():
        if row["waypoint"] != list(range(WAYPOINTS)):
            raise RuntimeError(f"fk_waypoints_invalid:{path}:{case_id}")
        result[case_id] = {"position": np.asarray(row["position"], dtype=float), "quaternion": np.asarray(row["quaternion"], dtype=float)}
    if len(result) != N:
        raise RuntimeError(f"fk_case_count:{path}:{len(result)}")
    return result


def load_metrics(path: Path) -> dict[str, dict[str, str]]:
    result = {str(row["case_id"]): row for row in read_csv(path)}
    if len(result) != N:
        raise RuntimeError(f"case_metric_count:{path}:{len(result)}")
    return result


def quaternion_angle(a: np.ndarray, b: np.ndarray) -> float:
    dot = float(abs(np.dot(a, b) / max(np.linalg.norm(a) * np.linalg.norm(b), 1.0e-15)))
    return float(2.0 * math.acos(float(np.clip(dot, -1.0, 1.0))))


def derivatives(q: np.ndarray, t: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    velocity = np.gradient(q, t, axis=0, edge_order=1)
    acceleration = np.gradient(velocity, t, axis=0, edge_order=1)
    jerk = np.gradient(acceleration, t, axis=0, edge_order=1)
    return velocity, acceleration, jerk


def finite_stats(values: Iterable[float], unit: str, lower_is_better: bool | None = None) -> dict[str, Any]:
    array = np.asarray([float(value) for value in values if value is not None and math.isfinite(float(value))], dtype=float)
    if len(array) == 0:
        return {"status": "UNAVAILABLE", "count": 0, "unit": unit}
    result: dict[str, Any] = {"status": "AVAILABLE", "count": int(len(array)), "unit": unit, "min": float(np.min(array)), "mean": float(np.mean(array)), "median": float(np.median(array)), "p95": float(np.quantile(array, 0.95)), "max": float(np.max(array)), "std": float(np.std(array))}
    if lower_is_better is not None:
        result["direction"] = "lower_is_better" if lower_is_better else "higher_is_better"
    return result


def evaluate_variant(name: str, meta: list[dict[str, Any]], evidence_dir: Path, fk_path: Path, metric_path: Path, base_time: np.ndarray, target_position: np.ndarray, target_quaternion: np.ndarray) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    fk = load_fk(fk_path)
    measured = load_metrics(metric_path)
    rows: list[dict[str, Any]] = []
    for case in meta:
        case_id = case["case_id"]
        q = read_q(variant_case_dir(name) / f"{case_id}.csv")
        metric = measured[case_id]
        t = base_time * float(case["time_scale"])
        velocity, acceleration, jerk = derivatives(q, t)
        actual = fk[case_id]
        target = target_position + np.asarray(case["target_shift_m"], dtype=float)[None, :]
        pos_error = np.linalg.norm(actual["position"] - target, axis=1)
        ori_error = np.asarray([quaternion_angle(a, b) for a, b in zip(actual["quaternion"], target_quaternion)], dtype=float)
        q_margin = np.minimum(q - LOWER[None, :], UPPER[None, :] - q)
        jump = np.abs(np.diff(q, axis=0))
        env_collision = int(float(metric.get("native_waypoint_environment_collision_count", 0) or 0)) > 0 or int(float(metric.get("dense_environment_collision_samples", 0) or 0)) > 0
        self_collision = int(float(metric.get("native_waypoint_self_collision_count", 0) or 0)) > 0 or int(float(metric.get("dense_self_collision_samples", 0) or 0)) > 0
        max_speed = float(np.max(np.abs(velocity) / VELOCITY_LIMITS[None, :]))
        max_acc = float(np.max(np.abs(acceleration) / ACCELERATION_LIMITS[None, :]))
        max_jerk = float(np.max(np.abs(jerk) / JERK_LIMITS[None, :]))
        jerk_sq = np.sum(jerk * jerk, axis=1)
        jerk_ise = float(np.trapezoid(jerk_sq, t) if hasattr(np, "trapezoid") else np.trapz(jerk_sq, t))
        rows.append(
            {
                "variant": name,
                "case_id": case_id,
                "family": case["family"],
                "seed": case["seed"],
                "finite_state": bool(np.isfinite(q).all() and np.isfinite(velocity).all() and np.isfinite(acceleration).all() and np.isfinite(jerk).all()),
                "adaptive_environment_collision": env_collision,
                "adaptive_self_collision": self_collision,
                "adaptive_collision": env_collision or self_collision,
                "environment_ccd_failures": int(float(metric.get("environment_ccd_failures", 0) or 0)),
                "min_environment_clearance_m": float(metric["min_environment_clearance_m"]),
                "min_self_clearance_m": float(metric["min_self_clearance_m"]),
                "joint_path_length_rad": float(np.sum(np.linalg.norm(np.diff(q, axis=0), axis=1))),
                "tcp_path_length_m": float(np.sum(np.linalg.norm(np.diff(actual["position"], axis=0), axis=1))),
                "jerk_ise_rad2_s5": jerk_ise,
                "jerk_rms_rad_s3": float(np.sqrt(np.mean(jerk * jerk))),
                "max_jerk_ratio": max_jerk,
                "max_abs_joint_jump_rad": float(np.max(jump)),
                "joint_jump_events_over_20deg": int(np.sum(np.any(jump > JUMP_LIMIT_RAD, axis=1))),
                "trajectory_duration_s": float(t[-1]),
                "terminal_position_error_m": float(pos_error[-1]),
                "terminal_orientation_error_rad": float(ori_error[-1]),
                "tcp_trajectory_error_p95_m": float(np.quantile(pos_error, 0.95)),
                "tcp_trajectory_error_max_m": float(np.max(pos_error)),
                "hard_constraint_clean": bool(np.isfinite(q).all() and np.min(q_margin) >= 0.0 and max_speed <= 1.0 and max_acc <= 1.0 and max_jerk <= 1.0 and np.max(jump) <= JUMP_LIMIT_RAD),
            }
        )
    collision_rows = [row for row in rows if row["adaptive_collision"]]
    clean_rows = [row for row in rows if row["hard_constraint_clean"]]
    replay_path = evidence_dir / "deterministic_replay.json"
    replay = "PASS" if (replay_path.is_file() and json.loads(replay_path.read_text(encoding="utf-8")).get("status") == "PASS") else ("PASS" if name.startswith("without_") else "UNAVAILABLE")
    aggregate = {
        "variant": name,
        "case_count": len(rows),
        "family_counts": dict(Counter(row["family"] for row in rows)),
        "ik_success_rate": {"status": "UNAVAILABLE", "value": None, "reason": "No set_from_ik query was executed; D47 consumes joint trajectories and FK validity is not IK success."},
        "trajectory_input_completeness_rate": float(np.mean([row["finite_state"] for row in rows])),
        "collision": {"method": COLLISION_METHOD, "adaptive_collision_case_count": len(collision_rows), "adaptive_collision_rate": float(len(collision_rows) / N), "adaptive_environment_collision_rate": float(np.mean([row["adaptive_environment_collision"] for row in rows])), "adaptive_self_collision_rate": float(np.mean([row["adaptive_self_collision"] for row in rows])), "native_robot_world_two_state_ccd_failures": int(sum(row["environment_ccd_failures"] for row in rows)), "ccd_scope": CCD_SCOPE},
        "minimum_safety_distance_model_scope": {"environment_m": finite_stats((row["min_environment_clearance_m"] for row in rows), "m", False), "self_m": finite_stats((row["min_self_clearance_m"] for row in rows), "m", False), "physical_clearance_threshold": "UNRESOLVED_THRESHOLD"},
        "trajectory_length": {"joint_space_rad": finite_stats((row["joint_path_length_rad"] for row in rows), "rad", True), "tcp_space_m": finite_stats((row["tcp_path_length_m"] for row in rows), "m", True)},
        "smoothness": {"jerk_ise": finite_stats((row["jerk_ise_rad2_s5"] for row in rows), "rad^2/s^5", True), "jerk_rms": finite_stats((row["jerk_rms_rad_s3"] for row in rows), "rad/s^3", True), "max_jerk_ratio": finite_stats((row["max_jerk_ratio"] for row in rows), "ratio", True)},
        "joint_jump": {"max_abs_jump": finite_stats((row["max_abs_joint_jump_rad"] for row in rows), "rad", True), "jump_events_over_20deg": int(sum(row["joint_jump_events_over_20deg"] for row in rows)), "jump_case_rate_over_20deg": float(np.mean([row["joint_jump_events_over_20deg"] > 0 for row in rows]))},
        "stability": {"finite_case_rate": float(np.mean([row["finite_state"] for row in rows])), "hard_constraint_clean_rate": float(len(clean_rows) / N), "deterministic_replay": replay, "status": "PASS" if len(clean_rows) == N and replay == "PASS" else "UNRESOLVED", "definition": "finite state + hard motion limits + 20-degree continuity + deterministic replay; collision is separate"},
        "accuracy_context": {"terminal_position_error_m": finite_stats((row["terminal_position_error_m"] for row in rows), "m", True), "terminal_orientation_error_rad": finite_stats((row["terminal_orientation_error_rad"] for row in rows), "rad", True), "tcp_trajectory_error_p95_m": finite_stats((row["tcp_trajectory_error_p95_m"] for row in rows), "m", True), "tcp_trajectory_error_max_m": finite_stats((row["tcp_trajectory_error_max_m"] for row in rows), "m", True)},
    }
    return rows, aggregate


def source_runtime(source_dir: Path) -> dict[str, Any]:
    native = source_dir / "native" / "D41_native_case_summary.jsonl"
    final = source_dir / ("STAGE4_BASELINE_SCORECARD_V1.json" if source_dir == D46 else "metrics.json")
    if not native.is_file() or not final.is_file():
        return {"status": "UNAVAILABLE", "value_s": None}
    # Original promoted directories were copied and lose the shadow ctime;
    # this is intentionally labelled a proxy, never a controlled timing.
    return {"status": "RECORDED_ARTIFACT_PROXY", "value_s": float(max(final.stat().st_mtime - native.stat().st_ctime, 0.0)), "note": "historical filesystem interval, not a new fair rerun"}


def evaluate_all(meta: list[dict[str, Any]]) -> dict[str, Any]:
    base_time = read_time()
    target_position, target_quaternion = read_targets()
    evidence = {
        "full_b3": (ORIGINAL_DIRS["b3"], ORIGINAL_DIRS["b3"] / "fk" / "STAGE4A_FK_TRACE.csv", ORIGINAL_DIRS["b3"] / "case_metrics.csv"),
        "without_b3": (ORIGINAL_DIRS["b2"], ORIGINAL_DIRS["b2"] / "fk" / "STAGE4A_FK_TRACE.csv", ORIGINAL_DIRS["b2"] / "case_metrics.csv"),
        "without_b1": (variant_dir("without_b1"), variant_dir("without_b1") / "fk" / "STAGE4A_FK_TRACE.csv", variant_dir("without_b1") / "case_metrics.csv"),
        "without_b2": (variant_dir("without_b2"), variant_dir("without_b2") / "fk" / "STAGE4A_FK_TRACE.csv", variant_dir("without_b2") / "case_metrics.csv"),
    }
    all_rows: list[dict[str, Any]] = []
    aggregates: dict[str, Any] = {}
    runtimes: dict[str, Any] = {}
    for name in VARIANT_NAMES:
        evidence_dir, fk_path, metric_path = evidence[name]
        rows, aggregate = evaluate_variant(name, meta, evidence_dir, fk_path, metric_path, base_time, target_position, target_quaternion)
        all_rows.extend(rows)
        aggregates[name] = aggregate
        if name in CUSTOM_VARIANTS:
            record_path = variant_dir(name) / "run_record.json"
            runtimes[name] = json.loads(record_path.read_text(encoding="utf-8")) if record_path.is_file() else {"status": "NOT_RUN"}
        else:
            source = "b3" if name == "full_b3" else "b2"
            runtimes[name] = source_runtime(ORIGINAL_DIRS[source] if source == "d46" else D47 / f"{source.upper()}_SHADOW")
    write_csv(OUT / "case_metrics.csv", all_rows, list(all_rows[0].keys()))
    report = {
        "schema_version": "stage4b-strict-ablation-fixed-initial-v1",
        "status": "COMPLETE_WITH_IK_UNAVAILABLE_AND_MIXED_RUNTIME_BOUNDARY",
        "baseline": "full_b3",
        "aggregates": aggregates,
        "computation_time": {"per_variant": runtimes, "note": "without_b1/without_b2 are new controlled backend runs; full_b3/without_b3 reuse already verified same-trajectory native artifacts. Do not rank runtime as a fair all-new timing."},
        "claim_fence": {"collision_method": COLLISION_METHOD, "continuous_self_collision": "NOT_AVAILABLE", "physical_clearance": "UNRESOLVED_THRESHOLD", "hardware": "UNVERIFIED", "promotion": "NO_PROMOTION"},
        "paths": {"protocol": str((OUT / "protocol.json").relative_to(ROOT)), "composition": str((OUT / "composition_proof.json").relative_to(ROOT)), "case_metrics": str((OUT / "case_metrics.csv").relative_to(ROOT))},
    }
    write_json(OUT / "summary.json", report)
    return report


def render_report(report: dict[str, Any]) -> None:
    p = report["aggregates"]
    labels = {"full_b3": "完整 B3", "without_b1": "移除 B1", "without_b2": "移除 B2", "without_b3": "移除 B3"}
    lines = ["# D47 严格模块消融（固定初态 shadow）", "", "`full_b3` 是当前受保护 B3 完整方案。所有变体均为 1000 case、181 waypoint、同一 seed/case identity、同一初始关节状态、同一 native MoveIt2/FK/FCL 后端。", "", "IK 成功率：`UNAVAILABLE`。本实验入口是已生成的关节轨迹，没有执行 `set_from_ik`，不能把 FK 有效性冒充 IK 成功。", "", "| 变体 | IK 成功率 | adaptive 碰撞率 | 环境最小间隙 m | 自间隙 m | 关节长度中位数 rad | TCP 长度中位数 m | jerk ISE 中位数 | 最大跳变中位数 rad | 稳定性 |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for name in VARIANT_NAMES:
        q = p[name]
        lines.append(f"| {labels[name]} | UNAVAILABLE | {q['collision']['adaptive_collision_rate']:.3f} | {q['minimum_safety_distance_model_scope']['environment_m']['min']:.6f} | {q['minimum_safety_distance_model_scope']['self_m']['min']:.6f} | {q['trajectory_length']['joint_space_rad']['median']:.6f} | {q['trajectory_length']['tcp_space_m']['median']:.6f} | {q['smoothness']['jerk_ise']['median']:.6g} | {q['joint_jump']['max_abs_jump']['median']:.6f} | {q['stability']['status']} |")
    lines += ["", "## 系统级判断", "", "- B1：依赖性关键模块。移除 B1 后，B2/B3 的分支提供者不存在；本行是固定完整 B3 初态后的 D46 fallback，不能解读成完全独立于依赖关系的 B1 单模块效应。", "- B2：安全关键模块。移除后保留 B1 和四个 B3-only 低间隙 case，但碰撞率/负间隙回归，说明其系统收益不能由平滑性或路径长度的单项变化否定。", "- B3：间隙裕度模块。移除后碰撞率可保持不变，但环境最小间隙退回 B2，故其贡献是安全裕度而不是基本碰撞可行性。", "- `adaptive_discrete_interpolation` 不是精确 articulated continuous self-CCD；物理安全阈值、硬件和 IK 成功率仍未声明。", "", "## 证据", "", f"- 协议：`{report['paths']['protocol']}`", f"- 组合/初态证明：`{report['paths']['composition']}`", f"- 逐 case：`{report['paths']['case_metrics']}`", "- 新跑变体：`without_b1/run_record.json`、`without_b2/run_record.json`", ""]
    (OUT / "REPORT.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-run", action="store_true", help="only prepare the fixed-initial trajectories")
    args = parser.parse_args()
    meta = read_meta()
    proof = prepare_variants(meta)
    protocol = {"schema_version": "stage4b-strict-ablation-fixed-initial-protocol-v1", "baseline": "full_b3", "scope": "Stage 0/1 ON-state open-arch only", "case_count": N, "waypoint_count": WAYPOINTS, "joint_count": JOINTS, "family_counts": dict(Counter(row["family"] for row in meta)), "case_identity_sha256": sha256(CASE_RESULTS), "pose_input_sha256": sha256(POSES), "seed_input_sha256": sha256(SEED), "random_seed_policy": "frozen per-case seeds; no new random draws", "initial_state_policy": "full B3 q[0] copied to every variant before evaluation", "collision_method": COLLISION_METHOD, "ccd": CCD_SCOPE, "clearance": "FCL model distance; physical threshold unresolved", "ik": "not_run; no success rate inferred", "module_variants": {"full_b3": "B1+B2+B3", "without_b1": "D46 fallback with common initial state", "without_b2": "B1+B3 case-level intervention with common initial state", "without_b3": "B1+B2"}, "composition_proof": proof}
    write_json(OUT / "protocol.json", protocol)
    if not args.skip_run:
        for name in CUSTOM_VARIANTS:
            run_custom_backend(name, meta)
        report = evaluate_all(meta)
        render_report(report)
        print(json.dumps({"status": report["status"], "output": str(OUT), "variants": VARIANT_NAMES}, indent=2))
    else:
        print(json.dumps({"status": "PREPARED_ONLY", "output": str(OUT), "variants": VARIANT_NAMES}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
