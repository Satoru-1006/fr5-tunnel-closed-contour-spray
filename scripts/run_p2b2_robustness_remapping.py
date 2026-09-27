#!/usr/bin/env python3
"""Run P2-B2 on the frozen D41/P2-A inputs with native MoveIt2/FK evidence."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ros2_moveit_bridge.p2b2_redundancy_objectives import (  # noqa: E402
    JOINT_NAMES,
    normalized_joint_margin_profile,
    rank_axis_results_by_family,
    rank_joint_axis_results,
    validate_provenance_schema,
    validate_result_schema,
)
import scripts.run_p2b1_solver_policy_ablation as p2b1  # noqa: E402


SOURCE_BASE = "e7bb161cea7396e452584f5c4ee7f158be6a9ca5"
BRANCH = "codex/fr5-p2b2-robustness-remap-redundancy-20260927"
POSES = ROOT / "outputs/internal_wiper_moveit_inputs/open_arch_tcp_poses_base_link.csv"
SEEDS = ROOT / "outputs/internal_wiper_moveit_inputs/open_arch_seed_joints.csv"
D39_SEED = ROOT / "outputs/stage3_h13_d39_causal_task_space_recovery/shadow_candidates/stable_velocity_residual_update.csv"
D39_SEED_RELATIVE = "outputs/stage3_h13_d39_causal_task_space_recovery/shadow_candidates/stable_velocity_residual_update.csv"
D41_RUN = ROOT / "outputs/stage3_h13_d41_offline_robot_certification/run_20260827T152326Z"
D41_PRE = D41_RUN / "strict_replay/moveit_waypoint_joint_trajectory.csv"
D41_POST = D41_RUN / "strict_replay/moveit_smoothed_joint_trajectory.csv"
D41_FK = D41_RUN / "strict_replay/moveit_fk_tcp_trace.csv"
D41_IDENTITY = D41_RUN / "D41_AUTHORITATIVE_IDENTITY.json"
P2A_RESULT = ROOT / "outputs/p2a_axiswise_robustness_margin.json"
P2A_AXIS_MODULE = ROOT / "src/p2a_axiswise_robustness.py"
POSE_COUNT = 181
COLLISION_METHOD = "adaptive_discrete_interpolation"
STRICT_SELF_CCD = "NOT_AVAILABLE"
PROJECT_LIMIT_PROVENANCE = "PROJECT_CONFIGURED_LIMITS"

VARIANT_DESIGN: tuple[dict[str, Any], ...] = (
    {"variant_id": "R0", "solver": "B0", "gain": None, "warm_start_rows": 16, "objective": "none", "objective_family": "none"},
    *(
        {"variant_id": f"R1_center_old_{gain:g}", "solver": "B1", "gain": gain, "warm_start_rows": 16, "objective": "joint_centering", "objective_family": "R1"}
        for gain in (2.5e-4, 5.0e-4, 1.0e-3)
    ),
    *(
        {"variant_id": f"R2_center_wp0_{gain:g}", "solver": "B1", "gain": gain, "warm_start_rows": 1, "objective": "joint_centering", "objective_family": "R2"}
        for gain in (2.5e-4, 5.0e-4, 1.0e-3)
    ),
    *(
        {"variant_id": f"R3_barrier_wp0_{gain:g}", "solver": "B1", "gain": gain, "warm_start_rows": 1, "objective": "joint_limit_barrier", "objective_family": "R3"}
        for gain in (2.5e-4, 5.0e-4, 1.0e-3)
    ),
)


def _git(*args: str) -> str:
    proc = subprocess.run(["git", *args], cwd=ROOT, text=True, encoding="utf-8", capture_output=True)
    if proc.returncode:
        raise RuntimeError(f"git_{args[0]}_failed:{proc.stderr.strip()}")
    return proc.stdout.strip()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _verify_execution_tree(*, clean_replay: bool = False, expected_head: str | None = None) -> tuple[str, str]:
    branch, head = _git("branch", "--show-current"), _git("rev-parse", "HEAD")
    dirty = _git("status", "--porcelain")
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", SOURCE_BASE, "HEAD"],
        cwd=ROOT, text=True, encoding="utf-8", capture_output=True,
    )
    is_based_on_source = ancestor.returncode == 0
    identity_ok = (
        head == expected_head and is_based_on_source
        if clean_replay
        else branch == BRANCH and is_based_on_source
    )
    if not identity_ok or dirty:
        raise RuntimeError(f"execution_tree_identity_mismatch:branch={branch}:head={head}:source_base_ancestor={is_based_on_source}:dirty={bool(dirty)}")
    return branch or "DETACHED_CLEAN_REPLAY", head


def _find_input_hash(inputs: Mapping[str, Any], relative_path: str) -> str | None:
    expected_key = relative_path.replace("\\", "/").casefold()
    for key, value in inputs.items():
        if str(key).replace("\\", "/").casefold() == expected_key:
            return str(value)
    return None


def _verify_frozen_inputs(
    d39_seed: Path, d39_seed_hash: str, frozen_input_source_root: Path,
) -> tuple[dict[str, str], dict[str, Path]]:
    authority = _read_json(D41_IDENTITY)
    if authority.get("point_count") != POSE_COUNT or authority.get("scope") != "Stage 0/1 ON-state open-arch only":
        raise RuntimeError("D41_authoritative_identity_scope_mismatch")
    expected = authority.get("inputs")
    if not isinstance(expected, dict) or len(expected) < 5:
        raise RuntimeError("D41_authoritative_identity_inputs_missing")
    actual: dict[str, str] = {}
    external: dict[str, Path] = {}
    for relative, expected_hash in expected.items():
        normalized_relative = str(relative).replace("\\", "/")
        if normalized_relative.casefold() == D39_SEED_RELATIVE.casefold():
            path = d39_seed
        else:
            path = ROOT / normalized_relative
            if not path.is_file():
                path = frozen_input_source_root / normalized_relative
                if path.is_file():
                    external[str(relative)] = path
        if not path.is_file():
            raise FileNotFoundError(path)
        digest = d39_seed_hash if path == d39_seed else _sha256(path)
        if digest != expected_hash:
            raise RuntimeError(f"frozen_D41_input_identity_mismatch:{relative}")
        actual[relative] = digest
    for path in (D41_PRE, D41_POST, D41_FK, P2A_RESULT, P2A_AXIS_MODULE):
        if not path.is_file():
            raise FileNotFoundError(path)
    if len(_read_csv(POSES)) != POSE_COUNT or len(_read_csv(D41_POST)) != POSE_COUNT:
        raise RuntimeError("frozen_open_arch_or_D41_waypoint_count_mismatch")
    p2a = _read_json(P2A_RESULT)
    if len(p2a.get("axis_results", [])) != 42:
        raise RuntimeError("frozen_P2A_axis_count_changed")
    return actual, external


def _runtime_provenance(underlay: Path, overlay: Path, distro: str, root_out: Path) -> str:
    text = p2b1.runtime_provenance(underlay, overlay, distro, root_out)
    required_tokens = ("Python 3.12", "2.12.4", "0.9.2")
    if any(token not in text for token in required_tokens):
        raise RuntimeError("runtime_version_provenance_incomplete")
    return text


def _profile_pair(pre_path: Path, post_path: Path, lower: np.ndarray, upper: np.ndarray) -> dict[str, Any]:
    pre = p2b1.read_q(pre_path)
    post = p2b1.read_q(post_path)
    return {
        "pre_ruckig": normalized_joint_margin_profile(pre, lower, upper),
        "post_ruckig": normalized_joint_margin_profile(post, lower, upper),
    }


def _strict_validation(row: Mapping[str, Any], lower: np.ndarray, upper: np.ndarray, buffer_rad: float = 1e-4) -> dict[str, Any]:
    output = Path(str(row["output_directory"]))
    strict = output / "strict_replay"
    summary_path = strict / "final_acceptance_summary.json"
    audit_path = strict / "audit_goal_requirements_strict.json"
    pre_path, post_path = Path(str(row.get("pre_ruckig_path") or "")), Path(str(row.get("post_ruckig_path") or ""))
    fd_path = Path(str(row.get("fd_consistency_path") or ""))
    summary = _read_json(summary_path) if summary_path.is_file() else {}
    audit = _read_json(audit_path) if audit_path.is_file() else {}
    metrics = summary.get("metrics", {})
    fd = _read_json(fd_path) if fd_path.is_file() else {}
    profile: dict[str, Any] | None = None
    errors: list[str] = []
    if not pre_path.is_file() or not post_path.is_file():
        errors.append("pre_or_post_ruckig_trajectory_missing")
    else:
        try:
            profile = _profile_pair(pre_path, post_path, lower, upper)
        except (ValueError, OSError) as exc:
            errors.append(f"joint_margin_measurement_failed:{exc}")
    pass_overall = str(row.get("strict_audit_status", "")).upper() == "PASS" and str(row.get("post_ruckig_validation_status", "")).upper() == "PASS"
    finite_fk = all(
        isinstance(metrics.get(key), (int, float, str))
        and math.isfinite(float(metrics[key]))
        for key in ("fk_path_deviation_max_mm", "fk_normal_error_max_deg")
    )
    process_residual_ok = finite_fk and float(metrics["fk_path_deviation_max_mm"]) <= 4.0 and float(metrics["fk_normal_error_max_deg"]) <= 5.0
    fk_ok = metrics.get("result_source") == "moveit2_strict_runtime" and finite_fk
    planning_scene_runtime_ok = audit.get("strict_runtime") is True
    ruckig_ok = str(metrics.get("moveit_ruckig_smoothing_used", "")).lower() == "true"
    bounds_ok = profile is not None and all(
        float(profile[key]["minimum_joint_margin_rad"]) >= buffer_rad - 1.0e-9
        for key in ("pre_ruckig", "post_ruckig")
    )
    continuity_ok = (
        str(metrics.get("process_joint_continuity_status", "")).lower() == "pass"
        and str(metrics.get("joint_continuity_status", "")).lower() == "pass"
        and str(metrics.get("waypoint_post_joint_continuity_match_status", "")).lower() == "pass"
    )
    collision_count = p2b1._number(metrics.get("collision_count"))
    collision_ok = str(metrics.get("collision_status", "")).lower() == "pass" and collision_count == 0.0
    dynamic_ratios = [p2b1._number(metrics.get(key)) for key in ("max_velocity_ratio", "max_acceleration_ratio", "max_jerk_ratio")]
    dynamics_ok = (
        str(metrics.get("process_dynamics_status", "")).lower() == "pass"
        and all(value is not None and value <= 1.0 for value in dynamic_ratios)
    )
    fd_ok = fd.get("status") == "PASS"
    gates = {
        "generation": "PASS" if int(row.get("strict_runner_exit_code", 99)) == 0 and pre_path.is_file() else "FAIL",
        "process_residual": "PASS" if process_residual_ok else "FAIL",
        "process_jacobian_finite_difference": "PASS" if fd_ok else ("NOT_RUN" if not fd else "FAIL"),
        "moveit2_planning_scene_runtime": "PASS" if planning_scene_runtime_ok else "FAIL",
        "moveit2_fk": "PASS" if fk_ok else "FAIL",
        "ruckig": "PASS" if ruckig_ok else "FAIL",
        "post_ruckig_validation": "PASS" if pass_overall else "FAIL",
        "joint_limits": "PASS" if bounds_ok else "FAIL",
        "continuity": "PASS" if continuity_ok else "FAIL",
        "collision_adaptive_discrete_interpolation": "PASS" if collision_ok else "FAIL",
        "configured_dynamics": "PASS" if dynamics_ok else "FAIL",
    }
    complete = all(value == "PASS" for value in gates.values())
    if not complete:
        errors.extend(name for name, status in gates.items() if status != "PASS")
    return {
        "status": "PASS" if complete else "FAIL",
        "gates": gates,
        "errors": errors,
        "strict_runtime": bool(audit.get("strict_runtime")),
        "collision_method": COLLISION_METHOD,
        "strict_self_ccd": STRICT_SELF_CCD,
        "dynamic_limit_provenance": PROJECT_LIMIT_PROVENANCE,
        "collision_count": collision_count,
        "max_velocity_ratio": dynamic_ratios[0],
        "max_acceleration_ratio": dynamic_ratios[1],
        "max_jerk_ratio": dynamic_ratios[2],
        "max_fk_path_error_mm": p2b1._number(metrics.get("fk_path_deviation_max_mm")),
        "max_normal_error_deg": p2b1._number(metrics.get("fk_normal_error_max_deg")),
        "trajectory_metrics": {
            "pre_ruckig": None if profile is None else {key: value for key, value in profile["pre_ruckig"].items() if key not in {"per_waypoint", "raw_joint_margin_matrix_rad", "normalized_joint_margin_matrix"}},
            "post_ruckig": None if profile is None else {key: value for key, value in profile["post_ruckig"].items() if key not in {"per_waypoint", "raw_joint_margin_matrix_rad", "normalized_joint_margin_matrix"}},
        },
        "full_joint_margin_profile": profile,
        "summary_path": str(summary_path) if summary_path.is_file() else None,
    }


def _compare_reference(row: Mapping[str, Any], reference_dir: Path, d46: Any) -> dict[str, Any]:
    reference_pre = reference_dir / "strict_replay/moveit_waypoint_joint_trajectory.csv"
    reference_post = reference_dir / "strict_replay/moveit_smoothed_joint_trajectory.csv"
    if not reference_pre.is_file() or not reference_post.is_file():
        return {"status": "BLOCKED_REFERENCE_TRAJECTORY_MISSING"}
    generated_pre = p2b1.read_q(Path(str(row["pre_ruckig_path"])))
    frozen_pre = p2b1.read_q(reference_pre)
    generated_times, generated_post, *_ = d46.load_post_ruckig(Path(str(row["post_ruckig_path"])))
    frozen_times, frozen_post, *_ = d46.load_post_ruckig(reference_post)
    if generated_pre.shape != frozen_pre.shape or generated_post.shape != frozen_post.shape or generated_pre.shape != (181, 6):
        return {"status": "FAIL", "shape_match": False}
    pre_delta = float(np.max(np.abs(generated_pre - frozen_pre)))
    post_delta = float(np.max(np.abs(generated_post - frozen_post)))
    if generated_times.shape != frozen_times.shape or generated_times.shape != (181,):
        return {"status": "FAIL", "shape_match": False, "time_shape_match": False}
    time_delta = float(np.max(np.abs(generated_times - frozen_times)))
    tolerance = 1e-6
    return {
        "status": "PASS" if pre_delta <= tolerance and post_delta <= tolerance and time_delta <= tolerance else "FAIL",
        "shape_match": True,
        "max_pre_ruckig_joint_delta_rad": pre_delta,
        "max_post_ruckig_joint_delta_rad": post_delta,
        "max_post_ruckig_timestamp_delta_s": time_delta,
        "joint_tolerance_rad": tolerance,
        "timestamp_tolerance_s": tolerance,
        "reference_source": "P2-B1 R0/B0 process-5DOF candidate, P2-B1 artifact publish commit",
    }


def _axis_compact(axis: Mapping[str, Any], variant_id: str) -> dict[str, Any]:
    spec = axis.get("specification", {})
    margin_record = axis.get("estimated_raw_axis_margin") or {}
    first_pass = axis.get("last_passing_perturbation") or {}
    first_fail = axis.get("first_failing_perturbation") or {}
    evidence = first_fail.get("evidence") or first_pass.get("evidence") or {}
    return {
        "variant_id": variant_id,
        "axis_id": spec.get("axis_id"),
        "perturbation_family": spec.get("perturbation_family"),
        "direction": spec.get("direction"),
        "units": spec.get("units"),
        "capability_status": spec.get("capability_status"),
        "axis_status": axis.get("axis_status"),
        "estimated_raw_axis_margin": margin_record.get("value"),
        "last_pass": first_pass.get("magnitude"),
        "first_fail": first_fail.get("magnitude"),
        "refined_boundary": (axis.get("refined_boundary_estimate") or {}).get("value"),
        "search_completeness": axis.get("search_completeness"),
        "monotonicity_observation": axis.get("monotonicity_observation"),
        "failure_intervals": axis.get("failure_intervals", []),
        "dominant_failure_mode": first_fail.get("dominant_failure_mode"),
        "failure_modes": first_fail.get("failure_modes", []),
        "critical_waypoint": first_fail.get("critical_waypoint"),
        "critical_segment": first_fail.get("critical_segment"),
        "realized_perturbation": first_fail.get("realized_perturbation"),
        "joint_limit_margin_rad": evidence.get("joint_limit_margin_min_rad"),
        "self_clearance_diagnostic_m": evidence.get("minimum_self_clearance_m"),
        "environment_clearance_diagnostic_m": evidence.get("minimum_environment_clearance_m"),
        "max_tcp_error_m": evidence.get("max_tcp_path_error_m"),
        "max_normal_error_rad": evidence.get("max_spray_axis_normal_error_rad"),
        "sigma_min": evidence.get("minimum_jacobian_sigma"),
        "condition_number": evidence.get("maximum_jacobian_condition_number"),
        "collision_method": COLLISION_METHOD,
        "strict_self_ccd": STRICT_SELF_CCD,
    }


def _validate_current_target_surface_normals(surface_normals: np.ndarray) -> np.ndarray:
    """Validate the ordered normals belonging to the current 181-point target."""
    normals = np.asarray(surface_normals, dtype=np.float64)
    if normals.shape != (POSE_COUNT, 3) or not np.isfinite(normals).all():
        raise RuntimeError(f"authoritative_target_normals_shape_or_finiteness:{normals.shape}")
    lengths = np.linalg.norm(normals, axis=1)
    if np.any(lengths <= 1e-12) or not np.allclose(lengths, 1.0, rtol=0.0, atol=1e-6):
        raise RuntimeError("authoritative_target_normals_must_be_unit_vectors")
    return normals


def _scan_variant_axes(
    row: Mapping[str, Any], variant_id: str, *, all_formal_axes: bool,
    scratch: Path, native_binary: Path, fk_binary: Path, overlay: Path,
    underlay: Path, urdf: Path, distro: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import tools.stage4a_system_benchmark as d46
    import tools.audit_stage17_reproducibility as d17
    from src.process_aware_stress import ProcessTrajectory
    import src.p2a_axiswise_robustness as p2a

    d46.URDF = urdf
    lower, upper, dynamic_limits = d46.load_limits()
    timestamps, q, _v, _a, _j = d46.load_post_ruckig(Path(str(row["post_ruckig_path"])))
    variant_root = scratch / "robustness_mapping" / variant_id
    fk_root = variant_root / "nominal_fk"
    fk_root.mkdir(parents=True, exist_ok=False)
    p2a._run_existing_fk = lambda batch_root, native_root, module: p2b1._run_fresh_fk(
        batch_root, native_root, module, fk_binary, overlay, underlay, urdf, distro,
    )
    d46.run_native = lambda run_dir, cases, native_name="native": p2b1._run_native_with_fresh_build(
        run_dir, cases, native_name, d46, native_binary, overlay, underlay, urdf, distro,
    )
    case_id = f"{variant_id}_nominal"
    _trace, q_by_case = p2a._run_fk_only_cases(fk_root, [(case_id, q)], d46)
    fk_by_case = p2a._read_fk_trace(fk_root / "fk" / "STAGE4A_FK_TRACE.csv")
    if case_id not in q_by_case or case_id not in fk_by_case:
        raise RuntimeError(f"candidate_nominal_MoveIt_FK_missing:{variant_id}")
    fk = fk_by_case[case_id]
    poses = np.column_stack((fk["position"], fk["quaternion"]))
    _target_positions, _target_quaternions, target_normals = d46.read_targets()
    target_normals = _validate_current_target_surface_normals(target_normals)
    nominal = ProcessTrajectory(
        tcp_poses=poses, joint_states=q, timestamps_s=timestamps,
        surface_normals=target_normals, joint_lower_rad=lower, joint_upper_rad=upper,
        metadata={"robot": "FAIRINO_FR5", "scope": "181-point ON-state open-arch only", "candidate": variant_id,
                  "nominal_source": "fresh MoveIt2 FK of candidate post-Ruckig q",
                  "surface_normal_source": "current authoritative 181-point open-arch target CSV, row-aligned by waypoint index"},
    )
    thresholds = d46.read_json(d46.OUT / "STAGE4_SYSTEM_BENCHMARK_V1.json")["diagnostic_thresholds"]
    specs = p2a._build_axis_specs(q, lower, upper)
    if not all_formal_axes:
        specs = [spec for spec in specs if spec.axis_id.startswith("joint:")]
    evaluator = p2a._D41BatchEvaluator(
        nominal, timestamps, lower, upper, dynamic_limits,
        float(thresholds["tcp_trajectory_error_m"]["value"]),
        float(thresholds["terminal_position_error_m"]["value"]),
        math.radians(float(d17.FORMAL_NORMAL_DEG)),
        float(thresholds["joint_step_rad"]["value"]), variant_root / "scan", d46,
    )
    (variant_root / "scan").mkdir()
    axes = p2a.scan_axes_batched(nominal, specs, evaluator.evaluate_batch)
    full_path = variant_root / "axis_results_full.json"
    _write_json(full_path, {"variant_id": variant_id, "axis_results": axes})
    compact = [_axis_compact(axis, variant_id) for axis in axes]
    return compact, {
        "axis_results_full_path": str(full_path),
        "axis_count": len(axes),
        "joint_axis_count": sum(str(item.get("axis_id", "")).startswith("joint:") for item in compact),
        "available_axis_count": sum(item.get("capability_status") == "AVAILABLE" for item in compact),
        "unavailable_axis_count": sum(item.get("capability_status") != "AVAILABLE" for item in compact),
        "native_batches": evaluator._batch_number,
        "scan_case_count": sum(len(axis.get("coarse_observations", [])) + len(axis.get("refinement_observations", [])) for axis in axes),
        "nominal_fk_trace": str(fk_root / "fk" / "STAGE4A_FK_TRACE.csv"),
    }


def _write_landscape_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise RuntimeError("robustness_landscape_has_no_axis_rows")
    keys = list(rows[0].keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            normalized = {key: json.dumps(row[key], sort_keys=True, allow_nan=False) if isinstance(row[key], (dict, list)) else row[key] for key in keys}
            writer.writerow(normalized)


def _variant_summary(row: Mapping[str, Any], validation: Mapping[str, Any], mapping: Mapping[str, Any],
                     axes: Sequence[Mapping[str, Any]], profile: Mapping[str, Any]) -> dict[str, Any]:
    ranked = rank_joint_axis_results(axes)
    finite = [item for item in ranked if item["margin_rad"] is not None]
    global_joint = finite[0] if finite else None
    pressure_path = Path(str(row.get("solver_pressure_path") or ""))
    pressure = _read_csv(pressure_path) if pressure_path.is_file() else []
    positive_commands = [p2b1._number(r.get("secondary_objective_contribution_norm")) for r in pressure]
    positive_commands = [value for value in positive_commands if value is not None]
    accepted_secondary = sum(
        str(r.get("line_search_accepted", "")).lower() == "true"
        and (p2b1._number(r.get("secondary_objective_contribution_norm")) or 0.0) > 0.0
        for r in pressure
    )
    return {
        "variant_id": row.get("variant_id"), "solver": row.get("solver"),
        "secondary_objective": row.get("secondary_objective"), "gain_rad2": row.get("secondary_gain"),
        "warm_start_rows_frozen": row.get("warm_start_rows"),
        "frozen_waypoint_indices": list(range(int(row.get("warm_start_rows", 0)))),
        "full_validation": dict(validation), "robustness_mapping": dict(mapping),
        "global_joint_axis_margin_rad": None if global_joint is None else global_joint["margin_rad"],
        "global_joint_axis": None if global_joint is None else global_joint["axis_id"],
        "global_joint_axis_failure_mode": None if global_joint is None else (global_joint["result"].get("first_failing_perturbation") or {}).get("dominant_failure_mode"),
        "global_joint_axis_critical_waypoint": None if global_joint is None else (global_joint["result"].get("first_failing_perturbation") or {}).get("critical_waypoint"),
        "minimum_normalized_joint_margin_pre_ruckig": profile["pre_ruckig"]["minimum_normalized_joint_margin"],
        "minimum_normalized_joint_margin_pre_joint": profile["pre_ruckig"]["controlling_joint"],
        "minimum_normalized_joint_margin_pre_waypoint": profile["pre_ruckig"]["controlling_waypoint"],
        "minimum_normalized_joint_margin_pre_side": profile["pre_ruckig"]["controlling_limit_side"],
        "minimum_joint_margin_pre_ruckig_rad": profile["pre_ruckig"]["minimum_joint_margin_rad"],
        "minimum_normalized_joint_margin_post_ruckig": profile["post_ruckig"]["minimum_normalized_joint_margin"],
        "minimum_normalized_joint_margin_post_joint": profile["post_ruckig"]["controlling_joint"],
        "minimum_normalized_joint_margin_post_waypoint": profile["post_ruckig"]["controlling_waypoint"],
        "minimum_normalized_joint_margin_post_side": profile["post_ruckig"]["controlling_limit_side"],
        "minimum_joint_margin_post_ruckig_rad": profile["post_ruckig"]["minimum_joint_margin_rad"],
        "maximum_secondary_command_norm_rad": max(positive_commands, default=0.0),
        "accepted_secondary_step_count": accepted_secondary,
    }


def _release_mapping(mapping: Mapping[str, Any], evidence_root: Path) -> dict[str, Any]:
    output = {key: value for key, value in mapping.items()
              if key not in {"axis_results_full_path", "nominal_fk_trace"}}
    for source_key, target_key in (
        ("axis_results_full_path", "axis_results_archive_member"),
        ("nominal_fk_trace", "nominal_fk_trace_archive_member"),
    ):
        source = mapping.get(source_key)
        if source:
            resolved = Path(str(source)).resolve()
            output[target_key] = resolved.relative_to(evidence_root.resolve()).as_posix()
    return output


def _campaign_status(reference_reproduction: Mapping[str, Any], variants: Sequence[Mapping[str, Any]], landscape: Sequence[Mapping[str, Any]]) -> str:
    all_valid = all((row.get("full_validation") or {}).get("status") == "PASS" for row in variants)
    joint_rows = [row for row in landscape if str(row.get("axis_id", "")).startswith("joint:")]
    expected_variants = {str(row.get("variant_id")) for row in variants}
    rows_per_variant = {
        variant_id: [row for row in joint_rows if row.get("variant_id") == variant_id]
        for variant_id in expected_variants
    }
    all_joint_scanned = (
        len(joint_rows) == 12 * len(expected_variants)
        and all(len(rows) == 12 and all(row.get("capability_status") == "AVAILABLE" for row in rows)
                for rows in rows_per_variant.values())
        and all(
            {str(row.get("axis_id")) for row in rows} == {
                f"joint:j{joint}:{direction}"
                for joint in range(1, 7) for direction in ("positive", "negative")
            }
            for rows in rows_per_variant.values()
        )
    )
    complete_joint = all(
        row.get("estimated_raw_axis_margin") is not None
        or (row.get("axis_status") == "NO_FAILURE_WITHIN_SEARCH_DOMAIN"
            and row.get("search_completeness") == "FULL_COARSE_DOMAIN_SAMPLED")
        for row in joint_rows
    )
    r0_rows = [row for row in landscape if row.get("variant_id") == "R0"]
    b0_supported = [row for row in r0_rows if row.get("capability_status") == "AVAILABLE"]
    mapping_complete = all(
        row.get("estimated_raw_axis_margin") is not None
        or (row.get("axis_status") == "NO_FAILURE_WITHIN_SEARCH_DOMAIN"
            and row.get("search_completeness") == "FULL_COARSE_DOMAIN_SAMPLED")
        for row in b0_supported
    )
    formal_axis_ids = {str(row.get("axis_id")) for row in r0_rows}
    formal_capability_shape = (
        len(r0_rows) == 42 and len(formal_axis_ids) == 42
        and sum(row.get("capability_status") == "AVAILABLE" for row in r0_rows) == 24
        and sum(row.get("capability_status") != "AVAILABLE" for row in r0_rows) == 18
    )
    if (reference_reproduction.get("status") != "PASS" or not all_valid or not all_joint_scanned
            or not complete_joint or not mapping_complete or not formal_capability_shape):
        return "INCOMPLETE"
    return "COMPLETE_WITH_FORMAL_CAPABILITY_GAPS"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--underlay-install", type=Path, required=True)
    parser.add_argument("--fairino-source-repository", type=Path, required=True)
    parser.add_argument("--reference-model", type=Path, required=True, help="P2-B1 byte-matched D41-derived URDF used for source-preserving comparison")
    parser.add_argument("--native-build-install", type=Path, required=True, help="Fresh P2-B2 source-built ROS overlay")
    parser.add_argument("--distro", default="Ubuntu-24.04-D")
    parser.add_argument("--p2b1-b0-reference", type=Path, required=True, help="P2-B1 B0 pre/post-Ruckig trajectory evidence directory")
    parser.add_argument("--d39-seed", type=Path, required=True, help="P2-B1 seed CSV matching the frozen D41 and P2-B1 execution manifests")
    parser.add_argument("--p2b1-execution-manifest", type=Path, required=True, help="P2-B1 execution manifest authenticating the external seed input")
    parser.add_argument("--frozen-input-source-root", type=Path, required=True, help="Read-only fallback root for D41-authenticated ignored inputs absent from the P2-B2 source checkout")
    parser.add_argument("--reference-only", action="store_true", help="Run clean-checkout R0 reference replay and full formal P2-A axes only")
    parser.add_argument("--expected-execution-commit", help="Required with --reference-only to authenticate clean replay source")
    args = parser.parse_args()

    scratch, underlay, fairino_source, reference_model, overlay, b0_reference, d39_seed, p2b1_manifest_path, frozen_input_source_root = (
        path.resolve() for path in (args.scratch, args.underlay_install, args.fairino_source_repository,
                                   args.reference_model, args.native_build_install, args.p2b1_b0_reference,
                                   args.d39_seed, args.p2b1_execution_manifest, args.frozen_input_source_root)
    )
    if not all(path.exists() for path in (scratch, underlay / "setup.bash", fairino_source / ".git", reference_model,
                                           overlay / "setup.bash", d39_seed, p2b1_manifest_path, frozen_input_source_root)):
        raise RuntimeError("P2B2_scratch_or_runtime_dependency_missing")
    if args.reference_only and not args.expected_execution_commit:
        raise RuntimeError("clean_reference_replay_requires_expected_execution_commit")
    if not args.reference_only and args.expected_execution_commit:
        raise RuntimeError("expected_execution_commit_only_valid_with_reference_only")
    branch, head = _verify_execution_tree(
        clean_replay=args.reference_only,
        expected_head=args.expected_execution_commit,
    )
    d41_identity = _read_json(D41_IDENTITY)
    d41_inputs = d41_identity.get("inputs") or {}
    p2b1_manifest = _read_json(p2b1_manifest_path)
    p2b1_inputs = p2b1_manifest.get("identity_sha256") or {}
    expected_d41_seed = _find_input_hash(d41_inputs, D39_SEED_RELATIVE)
    expected_p2b1_seed = _find_input_hash(p2b1_inputs, D39_SEED_RELATIVE)
    d39_seed_hash = _sha256(d39_seed)
    if expected_d41_seed is None or expected_d41_seed != expected_p2b1_seed or d39_seed_hash != expected_d41_seed:
        raise RuntimeError("D39_seed_does_not_match_frozen_D41_and_P2B1_identity")
    inputs, external_frozen_inputs = _verify_frozen_inputs(d39_seed, d39_seed_hash, frozen_input_source_root)
    native_binary = overlay / "lib/stage3_h13_d41_native/stage3_h13_d41_native"
    fk_binary = overlay / "lib/stage4a_fk/stage4a_fk"
    if not native_binary.is_file() or not fk_binary.is_file():
        raise RuntimeError("fresh_source_build_D41_native_or_stage4a_fk_missing")
    out_root = scratch / "p2b2_formal_execution"
    out_root.mkdir(parents=True, exist_ok=False)
    p2b1.D39_SEED = d39_seed
    input_root = out_root / "inputs"
    input_root.mkdir(parents=True, exist_ok=False)
    archived_p2b1_manifest = input_root / "P2B1_execution_manifest.json"
    shutil.copy2(p2b1_manifest_path, archived_p2b1_manifest)
    p2b1_manifest_hash = _sha256(p2b1_manifest_path)
    external_input_members: dict[str, str] = {}
    for relative, source_path in external_frozen_inputs.items():
        destination = input_root / "frozen_authoritative" / Path(str(relative).replace("\\", "/"))
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, destination)
        if _sha256(destination) != inputs[relative]:
            raise RuntimeError(f"archived_frozen_input_byte_mismatch:{relative}")
        external_input_members[relative] = destination.relative_to(scratch).as_posix()
    d39_archive_member = external_input_members.get(
        D39_SEED_RELATIVE,
        (input_root / "frozen_authoritative" / Path(D39_SEED_RELATIVE)).relative_to(scratch).as_posix(),
    )
    if not (scratch / d39_archive_member).is_file():
        d39_archive_path = input_root / "frozen_authoritative" / Path(D39_SEED_RELATIVE)
        d39_archive_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(d39_seed, d39_archive_path)
        if _sha256(d39_archive_path) != d39_seed_hash:
            raise RuntimeError("archived_D39_seed_byte_mismatch")
        d39_archive_member = d39_archive_path.relative_to(scratch).as_posix()
    import tools.stage4a_system_benchmark as d46
    import src.p2a_axiswise_robustness as p2a

    d46.URDF = reference_model
    lower, upper, dynamic_limits = d46.load_limits()
    runtime_text = _runtime_provenance(underlay, overlay, args.distro, out_root)
    model_output = out_root / "model"
    model_output.mkdir(parents=True, exist_ok=False)
    model_reproduction = p2b1.regenerate_d41_urdf(
        underlay, overlay, fairino_source, reference_model, model_output, args.distro,
    )
    fresh_urdf = Path(str(model_reproduction["regenerated_model"])).resolve()
    if model_reproduction.get("status") != "BYTE_MATCH" or not fresh_urdf.is_file():
        raise RuntimeError("D41_derived_model_regeneration_did_not_byte_match")
    d46.URDF = fresh_urdf
    lower, upper, dynamic_limits = d46.load_limits()
    source_remote = subprocess.run(["git", "remote", "get-url", "origin"], cwd=fairino_source, text=True, encoding="utf-8", capture_output=True)
    model_commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=fairino_source, text=True, encoding="utf-8", capture_output=True)
    model_dirty = subprocess.run(["git", "status", "--porcelain"], cwd=fairino_source, text=True, encoding="utf-8", capture_output=True)
    if source_remote.returncode or model_commit.returncode or model_dirty.returncode or "FAIR-INNOVATION/frcobot_ros2" not in source_remote.stdout or model_dirty.stdout.strip():
        raise RuntimeError("FAIRINO_description_source_identity_invalid")
    provenance = {
        "project": "FAIRINO_FR5", "stage": "P2-B2", "repository": "https://github.com/Satoru-1006/fr5-tunnel-closed-contour-spray.git",
        "branch": branch, "source_base_commit": SOURCE_BASE, "execution_code_commit": head,
        "model_source_commit": model_commit.stdout.strip(), "model_source_remote": source_remote.stdout.strip(),
        "collision_method": COLLISION_METHOD,
        "model_source_worktree_clean": True, "d41_derived_model_regeneration": model_reproduction,
        "runtime": {"runner": "Windows Python -> WSL2 Ubuntu-24.04-D", "ros_moveit_ruckig": runtime_text},
        "inputs": inputs,
        "additional_external_inputs": {
            "d39_seed": {"canonical_path": D39_SEED_RELATIVE, "sha256": d39_seed_hash,
                         "p2b1_execution_manifest_sha256": p2b1_manifest_hash,
                         "archive_member": d39_archive_member.removeprefix("p2b2_formal_execution/")},
            "d41_authenticated_external_input_members": external_input_members,
        },
        "configuration": {
            "trajectory_points": POSE_COUNT, "joint_order": list(JOINT_NAMES), "scope": "ON-state open-arch only",
            "tcp_source": "assumed_150mm_placeholder", "stand_off_m": 0.260,
            "time_parameterization": "tcp_arclength", "ruckig": True,
            "dynamic_limit_provenance": PROJECT_LIMIT_PROVENANCE,
            "collision_method": COLLISION_METHOD, "strict_self_ccd": STRICT_SELF_CCD,
            "clearance_acceptance_threshold": "UNRESOLVED_THRESHOLD",
        },
    }
    if not validate_provenance_schema(provenance):
        raise RuntimeError("P2B2_provenance_schema_invalid")
    _write_json(out_root / "source_identity.json", provenance)
    (out_root / "runtime_provenance.txt").write_text(runtime_text + "\n", encoding="utf-8")
    manifest = {
        "project": "FAIRINO_FR5", "stage": "P2-B2", "branch": branch,
        "source_base_commit": SOURCE_BASE, "execution_code_commit": head,
        "execution_tree_clean_at_start": True, "repository": provenance["repository"],
        "runtime": provenance["runtime"], "inputs": inputs,
        "model_source_commit": provenance["model_source_commit"], "collision_method": COLLISION_METHOD,
        "strict_self_ccd": STRICT_SELF_CCD, "hardware_validation": "NOT_RUN", "hardware_safety_certified": "NO",
        "variant_design": list(VARIANT_DESIGN),
        "additional_external_inputs": {
            "d39_seed": {"canonical_path": D39_SEED_RELATIVE, "sha256": d39_seed_hash,
                         "p2b1_execution_manifest_sha256": p2b1_manifest_hash,
                         "archive_member": d39_archive_member},
            "d41_authenticated_external_input_members": external_input_members,
            "p2b1_execution_manifest_archive_member": "p2b2_formal_execution/inputs/P2B1_execution_manifest.json",
        },
        "p2b1_b1_frozen_prefix_rows": 16, "old_metric": "joint:j6:positive only",
    }
    _write_json(out_root / "execution_manifest.json", manifest)

    variants: list[dict[str, Any]] = []
    validations: dict[str, dict[str, Any]] = {}
    scans: list[dict[str, Any]] = []
    maps_by_variant: dict[str, dict[str, Any]] = {}
    selected_designs = VARIANT_DESIGN[:1] if args.reference_only else VARIANT_DESIGN
    for design in selected_designs:
        row = p2b1.run_strict_variant(
            design["variant_id"], design["solver"], 1e-4, design["gain"],
            out_root, overlay, underlay, fresh_urdf, lower, upper, args.distro,
            p2b2_variant="R0" if design["variant_id"] == "R0" else design["objective_family"],
            p2b2_warm_start_rows=design["warm_start_rows"],
            p2b2_secondary_objective=design["objective"],
        )
        row["secondary_objective"] = design["objective"]
        row["warm_start_rows"] = design["warm_start_rows"]
        row["secondary_gain_units"] = "rad^2"
        validation = _strict_validation(row, lower, upper)
        row["full_validation"] = validation
        if design["variant_id"] == "R0":
            row["reference_reproduction"] = _compare_reference(row, b0_reference, d46)
        else:
            row["reference_reproduction"] = {"status": "NOT_APPLICABLE_ABLATION_CANDIDATE"}
        variants.append(row)
        validations[design["variant_id"]] = validation
        _write_json(out_root / "partial_campaign_state.json", {"completed_variants": variants, "status": "RUNNING"})
        if validation["status"] != "PASS":
            continue
        all_formal = design["variant_id"] == "R0"
        compact, mapping = _scan_variant_axes(
            row, design["variant_id"], all_formal_axes=all_formal,
            scratch=out_root, native_binary=native_binary, fk_binary=fk_binary,
            overlay=overlay, underlay=underlay, urdf=fresh_urdf, distro=args.distro,
        )
        row["robustness_mapping"] = mapping
        scans.extend(compact)
        maps_by_variant[design["variant_id"]] = mapping
        full = _read_json(Path(mapping["axis_results_full_path"]))["axis_results"]
        # Capture full per-waypoint pre/post joint margin evidence outside Git.
        profile = validation.get("full_joint_margin_profile")
        if profile is not None:
            margin_root = out_root / "redundancy_ablation" / design["variant_id"]
            margin_root.mkdir(parents=True, exist_ok=True)
            for phase in ("pre_ruckig", "post_ruckig"):
                target = margin_root / f"joint_margins_{phase}.csv"
                rows = profile[phase]["per_waypoint"]
                with target.open("w", newline="", encoding="utf-8") as stream:
                    writer = csv.DictWriter(stream, fieldnames=["waypoint", "joint", "joint_margin_rad", "normalized_joint_margin"], lineterminator="\n")
                    writer.writeheader()
                    for item in rows:
                        for joint_index, joint_name in enumerate(JOINT_NAMES):
                            writer.writerow({"waypoint": item["waypoint"], "joint": joint_name,
                                             "joint_margin_rad": item["joint_margins_rad"][joint_index],
                                             "normalized_joint_margin": item["normalized_joint_margins"][joint_index]})
        row["summary"] = _variant_summary(row, validation, mapping, full if not all_formal else full, validation["full_joint_margin_profile"])
        _write_json(out_root / "partial_campaign_state.json", {"completed_variants": variants, "axis_measurements": scans, "status": "RUNNING"})

    reference_row = next((row for row in variants if row.get("variant_id") == "R0"), {})
    reference_reproduction = reference_row.get("reference_reproduction", {"status": "NOT_RUN"})
    b0_axes_full_path = maps_by_variant.get("R0", {}).get("axis_results_full_path")
    b0_axes_full = _read_json(Path(b0_axes_full_path))["axis_results"] if b0_axes_full_path else []
    family_ranking = rank_axis_results_by_family(b0_axes_full)
    summaries = []
    for row in variants:
        if not row.get("summary"):
            continue
        summary = row["summary"]
        # Store summaries in canonical JSON; full trajectories and pressure rows stay in the evidence archive.
        summaries.append({key: value for key, value in summary.items() if key not in {"full_validation", "robustness_mapping"}} | {
            "full_validation_status": summary["full_validation"]["status"],
            "validation_gates": summary["full_validation"]["gates"],
            "robustness_mapping": _release_mapping(summary["robustness_mapping"], out_root),
        })
    r0_joint = rank_joint_axis_results(b0_axes_full)
    r0_finite = [item for item in r0_joint if item["margin_rad"] is not None]
    controller = r0_finite[0] if r0_finite else None
    first_fail = (controller["result"].get("first_failing_perturbation") or {}) if controller else {}
    static = next((row for row in variants if row.get("variant_id") == "R0"), {})
    post_profile = ((validations.get("R0") or {}).get("full_joint_margin_profile") or {}).get("post_ruckig")
    old_controller = {"joint": "j6", "waypoint": 86, "side": "upper", "failure_mode": "JOINT_LIMIT_FAILURE"}
    new_controller = {
        "axis_id": None if controller is None else controller["axis_id"],
        "margin_rad": None if controller is None else controller["margin_rad"],
        "joint": (None if controller is None else str(controller["axis_id"]).split(":")[1]),
        "waypoint": first_fail.get("critical_waypoint"),
        "failure_mode": first_fail.get("dominant_failure_mode"),
    }
    if new_controller["joint"] is None or new_controller["waypoint"] is None:
        migration = "UNRESOLVED"
    elif new_controller["joint"] != "j6":
        migration = "MIGRATED_JOINT"
    elif new_controller["waypoint"] != 86:
        migration = "MIGRATED_WAYPOINT_ONLY"
    else:
        migration = "PERSISTED_J6_WP86"
    b1_confounded = bool(
        (reference_row.get("robustness_mapping") or {}).get("axis_results_full_path")
        and any(row.get("axis_id") == "joint:j6:positive" and row.get("critical_waypoint") is not None
                and 0 <= int(row["critical_waypoint"]) < 16 and row.get("dominant_failure_mode") == "JOINT_LIMIT_FAILURE"
                for row in scans if row.get("variant_id") == "R0")
    )
    result = {
        "PROJECT": "FAIRINO_FR5", "STAGE": "P2-B2", "SOURCE_BASE_COMMIT": SOURCE_BASE,
        "EXECUTION_CODE_COMMIT": head, "ARTIFACT_PUBLISH_COMMIT": None,
        "P2B2_STATUS": "INCOMPLETE", "P2B2_CANDIDATE_REFERENCE": "R0_B0_process_5DOF",
        "P2B2_REFERENCE_REPRODUCTION": reference_reproduction,
        "AXIS_COUNT": len(p2a._build_axis_specs(p2b1.read_q(Path(str(reference_row.get("post_ruckig_path", D41_POST)))), lower, upper)),
        "JOINT_AXIS_COUNT": 12, "TOTAL_AXIS_MEASUREMENT_ROWS": len(scans),
        "AVAILABLE_FORMAL_AXIS_COUNT": sum(row.get("capability_status") == "AVAILABLE" for row in b0_axes_full),
        "UNAVAILABLE_FORMAL_AXIS_COUNT": sum(row.get("capability_status") != "AVAILABLE" for row in b0_axes_full),
        "ROBUSTNESS_MAPPING_STATUS": "INCOMPLETE",
        "P2B2_GLOBAL_CONTROLLING_AXIS": new_controller["axis_id"],
        "P2B2_GLOBAL_MIN_MARGIN": new_controller["margin_rad"],
        "P2B2_GLOBAL_FIRST_FAILURE_MODE": new_controller["failure_mode"],
        "P2B2_GLOBAL_CRITICAL_WAYPOINT": new_controller["waypoint"],
        "CONTROLLING_JOINT_AXIS": new_controller["axis_id"],
        "CONTROLLING_JOINT_DIRECTION": None if new_controller["axis_id"] is None else new_controller["axis_id"].split(":")[-1],
        "CONTROLLING_JOINT_MARGIN_RAD": new_controller["margin_rad"],
        "CONTROLLING_JOINT_WAYPOINT": new_controller["waypoint"],
        "CONTROLLING_FAILURE_MODE": new_controller["failure_mode"],
        "MIN_NORMALIZED_JOINT_MARGIN": None if post_profile is None else post_profile["minimum_normalized_joint_margin"],
        "MIN_NORMALIZED_JOINT_MARGIN_JOINT": None if post_profile is None else post_profile["controlling_joint"],
        "MIN_NORMALIZED_JOINT_MARGIN_WAYPOINT": None if post_profile is None else post_profile["controlling_waypoint"],
        "MIN_NORMALIZED_JOINT_MARGIN_SIDE": None if post_profile is None else post_profile["controlling_limit_side"],
        "MINIMUM_JOINT_MARGIN_RAD": None if post_profile is None else post_profile["minimum_joint_margin_rad"],
        "P2B1_B1_EFFICACY_CONFOUND": "CONFIRMED" if b1_confounded else "NOT_CONFIRMED",
        "P2B1_B1_FROZEN_PREFIX_WAYPOINTS": list(range(16)),
        "P2B1_B1_PRIOR_METRIC": "joint:j6:positive only",
        "REDUNDANCY_VARIANTS": summaries,
        "BEST_VALIDATED_VARIANT": None,
        "BEST_VARIANT_REASON": "No fully mapped candidate yet",
        "BOTTLENECK_MIGRATION_STATUS": migration,
        "OLD_BOTTLENECK": old_controller, "NEW_BOTTLENECK": new_controller,
        "B0_REFERENCE_AXES": [_axis_compact(axis, "R0") for axis in b0_axes_full],
        "B0_FAMILY_RANKING": {key: [{"axis_id": item["axis_id"], "margin": item["margin"], "units": item["units"]} for item in rows] for key, rows in family_ranking.items()},
        "ALL_MEASURED_AXIS_ROWS": scans,
        "COLLISION_METHOD": COLLISION_METHOD, "STRICT_SELF_CCD": STRICT_SELF_CCD,
        "CLEARANCE_ACCEPTANCE_THRESHOLD": "UNRESOLVED_THRESHOLD",
        "DYNAMIC_LIMIT_PROVENANCE": PROJECT_LIMIT_PROVENANCE,
        "HARDWARE_VALIDATION": "NOT_RUN", "HARDWARE_SAFETY_CERTIFIED": "NO",
        "SOURCE_REPLAY": "NOT_RUN", "BUILD_REPLAY": "NOT_RUN", "TEST_REPLAY": "NOT_RUN",
        "SCIENTIFIC_CAMPAIGN_REPLAY": "NOT_RUN", "ARTIFACT_BYTE_MATCH": "NOT_RUN",
        "LIMITATIONS": [
            "P2B2_GLOBAL_MIN_MARGIN is the minimum among the 12 joint-state axes in radians; TCP translation/rotation are reported by family because no common uncertainty normalization exists.",
            "P2-A formally defines 42 axes: 12 joint, 12 TCP translation/rotation available axes, and 18 base/process axes explicitly unavailable in the frozen evaluator.",
            "Collision observations use adaptive_discrete_interpolation; strict self-collision CCD is not available.",
            "Clearance values are diagnostics and acceptance threshold is unresolved.",
            "Dynamics limits are project-configured, not vendor-certified; hardware validation was not run.",
            "TCP is the assumed 150 mm placeholder; no calibration or physical coating-quality model is included.",
        ],
        "MEASUREMENT_EVIDENCE_MEMBER": "p2b2_formal_execution/",
        "EXECUTION_MANIFEST_MEMBER": "p2b2_formal_execution/execution_manifest.json",
    }
    result["P2B2_STATUS"] = _campaign_status(reference_reproduction, variants, scans)
    if args.reference_only:
        result["P2B2_STATUS"] = (
            "R0_REFERENCE_REPLAY_COMPLETE_NOT_FULL_CAMPAIGN"
            if result["P2B2_STATUS"].startswith("COMPLETE")
            else "R0_REFERENCE_REPLAY_INCOMPLETE"
        )
    result["ROBUSTNESS_MAPPING_STATUS"] = result["P2B2_STATUS"]
    complete_joint_variant_ids = set()
    for variant in variants:
        variant_id = str(variant.get("variant_id"))
        joint = [row for row in scans if row.get("variant_id") == variant_id and str(row.get("axis_id", "")).startswith("joint:")]
        expected_axes = {
            f"joint:j{joint_index}:{direction}"
            for joint_index in range(1, 7) for direction in ("positive", "negative")
        }
        if (len(joint) == 12 and {str(row.get("axis_id")) for row in joint} == expected_axes
                and all(row.get("estimated_raw_axis_margin") is not None
                        or (row.get("axis_status") == "NO_FAILURE_WITHIN_SEARCH_DOMAIN"
                            and row.get("search_completeness") == "FULL_COARSE_DOMAIN_SAMPLED")
                        for row in joint)):
            complete_joint_variant_ids.add(variant_id)
    valid_summaries = [
        row for row in summaries
        if row.get("full_validation_status") == "PASS"
        and row.get("variant_id") in complete_joint_variant_ids
        and row.get("global_joint_axis_margin_rad") is not None
    ]
    best = max(valid_summaries, key=lambda row: float(row["global_joint_axis_margin_rad"]), default=None)
    result["BEST_VALIDATED_VARIANT"] = None if best is None else best["variant_id"]
    result["BEST_VARIANT_REASON"] = "Highest complete 12-direction joint-state deterministic margin among strict-validated candidates" if best else "No candidate had complete strict validation and joint-axis mapping"
    result["BEST_VARIANT_GLOBAL_MARGIN_CHANGE_RAD"] = None if best is None or controller is None else float(best["global_joint_axis_margin_rad"]) - float(controller["margin_rad"])
    if not validate_result_schema(result):
        raise RuntimeError("P2B2_result_schema_invalid")
    _write_json(out_root / "p2b2_robustness_landscape.json", result)
    _write_json(out_root / "p2b2_redundancy_ablation.json", {
        "PROJECT": "FAIRINO_FR5", "STAGE": "P2-B2", "STATUS": result["P2B2_STATUS"],
        "VARIANTS": summaries, "BEST_VALIDATED_VARIANT": result["BEST_VALIDATED_VARIANT"],
        "BEST_VARIANT_GLOBAL_MARGIN_CHANGE_RAD": result["BEST_VARIANT_GLOBAL_MARGIN_CHANGE_RAD"],
        "P2B1_B1_EFFICACY_CONFOUND": result["P2B1_B1_EFFICACY_CONFOUND"],
        "FROZEN_PREFIX_WAYPOINTS": list(range(16)),
        "OBJECTIVE_FORMULAS": {
            "joint_centering": "0.5 * sum_j(((q_j-(l_j+u_j)/2)/((u_j-l_j)/2))^2)",
            "joint_limit_barrier": "sum_j((s_l+0.02)^-2 + (s_u+0.02)^-2), s_l=(q-l)/(u-l), s_u=(u-q)/(u-l)",
            "gradient": "analytic; finite-difference checked; command projected through rank-5 process-task nullspace",
        },
        "GAIN_SCALES": {"values_rad2": [2.5e-4, 5e-4, 1e-3], "scale_basis": "P2-B1 accepted secondary-command scale; small/medium/large are 1x/2x/4x the small reference gain"},
    })
    _write_landscape_csv(out_root / "p2b2_robustness_landscape.csv", scans)
    _write_json(out_root / "partial_campaign_state.json", {"completed_variants": variants, "axis_measurements": scans, "status": result["P2B2_STATUS"]})
    print(json.dumps({
        "P2B2_STATUS": result["P2B2_STATUS"], "P2B2_RESULT": str(out_root / "p2b2_robustness_landscape.json"),
        "variant_count": len(variants), "strict_validated": sum((row.get("full_validation") or {}).get("status") == "PASS" for row in variants),
        "axis_measurement_rows": len(scans), "controlling_joint_axis": result["CONTROLLING_JOINT_AXIS"],
        "controlling_joint_margin_rad": result["CONTROLLING_JOINT_MARGIN_RAD"],
        "controlling_waypoint": result["CONTROLLING_JOINT_WAYPOINT"],
    }, sort_keys=True))
    return 0 if result["P2B2_STATUS"].startswith("COMPLETE") or result["P2B2_STATUS"] == "R0_REFERENCE_REPLAY_COMPLETE_NOT_FULL_CAMPAIGN" else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(json.dumps({"P2B2_STATUS": "BLOCKED_OR_FAILED", "error": f"{type(exc).__name__}: {exc}"}, sort_keys=True))
        raise
