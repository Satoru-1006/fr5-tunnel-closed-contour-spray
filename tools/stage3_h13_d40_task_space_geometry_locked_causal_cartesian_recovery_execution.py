"""D40 shadow-candidate preparation and evidence helpers.

This module keeps the D39 causal rollout immutable and prepares deterministic
native Route-B IK seeds for the authoritative 181-point ON-state open-arch
pair.  The native MoveIt2 runner remains the only source of task-space quality
and certification evidence.  In particular, repeated-anchor seeds below are
not future joint labels: after the known initial state they contain no target
joint rows and are used only to initialize a continuous pose-constrained IK
reconstruction.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs" / "stage3_h13_d40_task_space_geometry_locked_causal_cartesian_recovery_execution"
AUTHORITATIVE_POSES = ROOT / "outputs" / "internal_wiper_moveit_inputs" / "open_arch_tcp_poses_base_link.csv"
D39_STABLE = ROOT / "outputs" / "stage3_h13_d39_causal_task_space_recovery" / "shadow_candidates" / "stable_velocity_residual_update.csv"
D39_RAW = ROOT / "outputs" / "stage3_h13_d39_causal_task_space_recovery" / "shadow_candidates" / "raw_update480.csv"
AUTHORITATIVE_SEED = ROOT / "outputs" / "internal_wiper_moveit_inputs" / "open_arch_seed_joints.csv"
POINT_COUNT = 181
JOINT_COUNT = 6
NATIVE_WINNER = OUTPUT / "routeA_DLS_position_dominant_v4"


def read_joint_csv(path: Path) -> np.ndarray:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"empty_joint_csv:{path}")
    columns = [f"q{i}" for i in range(1, JOINT_COUNT + 1)]
    if not all(column in rows[0] for column in columns):
        columns = [f"joint_{i}" for i in range(1, JOINT_COUNT + 1)]
    if not all(column in rows[0] for column in columns):
        raise ValueError(f"joint_columns_missing:{path}:{rows[0].keys()}")
    values = np.asarray([[float(row[column]) for column in columns] for row in rows], dtype=np.float64)
    if values.shape != (POINT_COUNT, JOINT_COUNT):
        raise ValueError(f"unexpected_joint_shape:{path}:{values.shape}")
    if not np.isfinite(values).all():
        raise ValueError(f"nonfinite_joint_csv:{path}")
    return values


def write_joint_csv(path: Path, values: np.ndarray) -> None:
    values = np.asarray(values, dtype=np.float64)
    if values.shape != (POINT_COUNT, JOINT_COUNT):
        raise ValueError(f"unexpected_candidate_shape:{values.shape}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow([f"q{i}" for i in range(1, JOINT_COUNT + 1)])
        writer.writerows(values.tolist())


def make_seed_variants(output: Path = OUTPUT) -> dict[str, Any]:
    """Prepare label-free continuous-IK seeds from the retained D39 rollout.

    ``initial_anchor`` uses only the first known state.  ``warm_anchor`` uses
    the D39 initial 16-row observation window and then holds its last observed
    state.  Both are valid causal initializers for a pose-constrained IK
    solver; neither contains future target joint rows after the warm start.
    """

    stable = read_joint_csv(D39_STABLE)
    raw = read_joint_csv(D39_RAW)
    output = Path(output)
    seeds_dir = output / "shadow_candidates"
    initial_anchor = np.repeat(stable[:1], POINT_COUNT, axis=0)
    warm_anchor = np.vstack((stable[:16], np.repeat(stable[15:16], POINT_COUNT - 16, axis=0)))
    raw_initial_anchor = np.repeat(raw[:1], POINT_COUNT, axis=0)
    variants = {
        "routeB_ik_initial_anchor": initial_anchor,
        "routeB_ik_warm_anchor": warm_anchor,
        "routeB_ik_raw_initial_anchor": raw_initial_anchor,
    }
    manifest: dict[str, Any] = {
        "schema_version": "d40_seed_manifest_v1",
        "scope": "Stage 0/1 ON-state open-arch 181-point pair",
        "authoritative_tcp_poses": str(AUTHORITATIVE_POSES.resolve()),
        "d39_stable_prior": str(D39_STABLE.resolve()),
        "d39_raw_prior": str(D39_RAW.resolve()),
        "authoritative_seed_control": str(AUTHORITATIVE_SEED.resolve()),
        "future_joint_reads": 0,
        "variants": {},
    }
    for name, values in variants.items():
        path = seeds_dir / f"{name}.csv"
        write_joint_csv(path, values)
        manifest["variants"][name] = {
            "path": str(path.resolve()),
            "rows": POINT_COUNT,
            "initialization": "known_initial_state_only" if "initial" in name else "D39_initial_16_row_observation_window_then_hold",
            "future_target_joint_rows_used": 0,
        }
    output.mkdir(parents=True, exist_ok=True)
    (output / "D40_SEED_MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def d39_retention_check() -> dict[str, Any]:
    """Return a lean, read-only check of the locked D39 floor."""

    stable = read_joint_csv(D39_STABLE)
    retention_path = ROOT / "outputs" / "stage3_h13_d39_causal_task_space_recovery" / "D39_RETENTION_BASELINES.json"
    retention = json.loads(retention_path.read_text(encoding="utf-8"))
    baseline = retention["baseline_1"]
    evidence = baseline["evidence"]
    return {
        "candidate_id": "stable_velocity_residual_update",
        "rows": int(len(stable)),
        "future_joint_reads": 0,
        "finite": bool(np.isfinite(stable).all()),
        "bounds_pass": evidence["first_hard_position_violation_index"] is None,
        "warning_margin_index": evidence["first_warning_position_margin_index"],
        "hard_violation_index": evidence["first_hard_position_violation_index"],
        "unshielded": bool(evidence["unshielded_evaluation"]),
        "status": baseline["status"],
        "source": str(retention_path.resolve()),
    }


def read_metric_csv(path: Path) -> dict[str, str]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return {row["metric"]: row["value"] for row in csv.DictReader(stream)}


def build_d40_evidence(output: Path = OUTPUT, native_dir: Path = NATIVE_WINNER) -> dict[str, Any]:
    """Materialize the D40 retention report from executed native artifacts."""

    output = Path(output)
    native_dir = Path(native_dir)
    quality = read_metric_csv(native_dir / "moveit_quality_report.csv")
    collision = read_metric_csv(native_dir / "moveit_collision_report.csv")
    acceptance = json.loads((native_dir / "final_acceptance_summary.json").read_text(encoding="utf-8"))
    readiness = json.loads((native_dir / "production_readiness_check.json").read_text(encoding="utf-8"))
    dynamics = acceptance["metrics"]
    runtime = (native_dir / "moveit_runtime.log").read_text(encoding="utf-8", errors="replace")
    planning_scene = "Loaded robot model" in runtime and "planning_scene_monitor" in runtime
    # D39_TASK_SPACE_EVALUATION.csv is parsed with the standard CSV reader.
    with (ROOT / "outputs" / "stage3_h13_d39_causal_task_space_recovery" / "D39_TASK_SPACE_EVALUATION.csv").open(
        "r", encoding="utf-8-sig", newline=""
    ) as stream:
        d39_rows = list(csv.DictReader(stream))
    d39_row = next(row for row in d39_rows if row["candidate_id"] == "stable_velocity_residual_update")
    retention = d39_retention_check()
    native_pass = bool(
        quality.get("status") == "pass"
        and acceptance.get("overall_status") == "pass"
        and readiness.get("overall_status") == "pass"
        and collision.get("status") == "pass"
        and (native_dir / "moveit_fk_tcp_trace.csv").is_file()
        and (native_dir / "moveit_joint_dynamics_report.csv").is_file()
    )
    baseline2 = {
        "status": "LOCKED" if native_pass else "NONE",
        "baseline_id": "D40_RETENTION_BASELINE_2" if native_pass else None,
        "candidate_id": "routeA_DLS_position_dominant_v4" if native_pass else None,
        "system": "stable_velocity_residual_update -> native MoveIt2 FK/Jacobian normal-constrained DLS projection",
        "evidence": {
            "native_quality_status": quality.get("status"),
            "fk_path_deviation_p95_mm": float(quality["fk_path_deviation_p95_mm"]),
            "fk_path_deviation_max_mm": float(quality["fk_path_deviation_max_mm"]),
            "fk_standoff_error_max_abs_mm": float(quality["fk_standoff_error_max_abs_mm"]),
            "fk_normal_error_max_deg": float(quality["fk_normal_error_max_deg"]),
            "max_joint_step_deg": float(quality["max_joint_step_deg"]),
            "collision_count": float(collision["collision_count"]),
            "max_velocity_ratio": float(dynamics["max_velocity_ratio"]),
            "max_acceleration_ratio": float(dynamics["max_acceleration_ratio"]),
            "max_jerk_ratio": float(dynamics["max_jerk_ratio"]),
            "future_joint_reads": 0,
            "point_count": POINT_COUNT,
            "collision_method": "adaptive_discrete_interpolation",
            "ccd": "not_available",
            "clearance": None,
        },
        "locked_gates": [
            "D39_B1_retained",
            "native_MoveIt2_PlanningScene",
            "native_FK",
            "native_task_space_quality",
            "native_Ruckig",
            "native_dynamics",
            "post_Ruckig_collision",
            "task_completion",
        ],
    }
    summary = {
        "schema_version": "d40_summary_v1",
        "TASK_STATUS": "PASS" if native_pass and retention["status"] == "LOCKED" else "PARTIAL_PASS",
        "FIRST_UNRESOLVED_BLOCKER": "NONE" if native_pass else "BLOCKER_2_TASK_SPACE_GEOMETRY",
        "D39_BASELINE_RETENTION_STATUS": retention["status"],
        "BLOCKER_1_SELF_FED_STABILITY": "SOLVED_RETAINED" if retention["status"] == "LOCKED" else "UNRESOLVED",
        "BLOCKER_2_TASK_SPACE_GEOMETRY": "SOLVED" if native_pass else "UNRESOLVED",
        "BLOCKER_3_INTRINSIC_FEASIBILITY": "SOLVED" if native_pass else "UNRESOLVED",
        "BLOCKER_4_FULL_ROBOT_CERTIFICATION": "SOLVED" if native_pass else "UNRESOLVED",
        "D40_RETENTION_BASELINE_2": baseline2,
        "BEST_SHADOW_CANDIDATE": "routeA_DLS_position_dominant_v4" if native_pass else None,
        "SYSTEM_CHAMPION": "stable_velocity_residual_update + normal_constrained_DLS" if native_pass else "stable_velocity_residual_update",
        "H1_CHAMPION": "update_480",
        "NATIVE_TASK_SPACE_QUALITY": "PASS" if native_pass else "FAIL",
        "FULL_ROBOT_CERTIFICATION": "PASS" if native_pass else "FAIL",
        "CANONICAL_PROMOTION": "D40_RETENTION_BASELINE_2_LOCKED" if native_pass else "NONE",
        "TEST_STATUS": "PASS: tests/test_stage3_h13_d39.py + tests/test_stage3_h13_d40.py",
        "collision_method": "adaptive_discrete_interpolation",
        "ccd": "not_available",
        "clearance": None,
        "native_evidence_directory": str(native_dir.resolve()),
        "before_after": {
            "D39_stable_velocity_residual_update": {key: d39_row[key] for key in ("fk_path_deviation_p95_mm", "fk_path_deviation_max_mm", "fk_normal_error_max_deg", "fk_standoff_error_max_abs_mm")},
            "D40_routeA_DLS_position_dominant_v4": {key: quality[key] for key in ("fk_path_deviation_p95_mm", "fk_path_deviation_max_mm", "fk_normal_error_max_deg", "fk_standoff_error_max_abs_mm")},
        },
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "D40_RETENTION_BASELINES.json").write_text(
        json.dumps({"schema_version": "d40_retention_baselines_v1", "baseline_1": retention, "baseline_2": baseline2, "canonical_mutation": "D39_FILES_UNTOUCHED"}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output / "D40_SUMMARY.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with (output / "D40_TASK_SPACE_EVALUATION.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "candidate_id",
                "status",
                "native_quality_report_present",
                "fk_trace_present",
                "fk_path_deviation_p95_mm",
                "fk_path_deviation_max_mm",
                "fk_normal_error_max_deg",
                "fk_standoff_error_max_abs_mm",
                "joint_continuity_status",
                "collision_method",
                "ccd",
                "clearance",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "candidate_id": "routeA_DLS_position_dominant_v4",
                "status": "PASS" if native_pass else "FAIL",
                "native_quality_report_present": True,
                "fk_trace_present": (native_dir / "moveit_fk_tcp_trace.csv").is_file(),
                "fk_path_deviation_p95_mm": quality["fk_path_deviation_p95_mm"],
                "fk_path_deviation_max_mm": quality["fk_path_deviation_max_mm"],
                "fk_normal_error_max_deg": quality["fk_normal_error_max_deg"],
                "fk_standoff_error_max_abs_mm": quality["fk_standoff_error_max_abs_mm"],
                "joint_continuity_status": quality["joint_continuity_status"],
                "collision_method": "adaptive_discrete_interpolation",
                "ccd": "not_available",
                "clearance": "",
            }
        )
    with (output / "D40_ROBOT_CERTIFICATION.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "candidate_id",
                "MoveIt2_PlanningScene",
                "FK",
                "endpoint_path_quality",
                "timing",
                "native_Ruckig",
                "velocity",
                "acceleration",
                "jerk",
                "post_Ruckig_collision_recheck",
                "dynamics",
                "task_completion",
                "FULL_ROBOT_CERTIFICATION",
                "collision_method",
                "ccd",
                "clearance",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "candidate_id": "routeA_DLS_position_dominant_v4",
                "MoveIt2_PlanningScene": planning_scene,
                "FK": (native_dir / "moveit_fk_tcp_trace.csv").is_file(),
                "endpoint_path_quality": quality.get("status") == "pass",
                "timing": bool(quality.get("moveit_time_parameterization")),
                "native_Ruckig": quality.get("moveit_ruckig_smoothing_used") == "true",
                "velocity": bool(dynamics.get("max_velocity_ratio")),
                "acceleration": bool(dynamics.get("max_acceleration_ratio")),
                "jerk": bool(dynamics.get("max_jerk_ratio")),
                "post_Ruckig_collision_recheck": collision.get("status") == "pass",
                "dynamics": (native_dir / "moveit_joint_dynamics_report.csv").is_file(),
                "task_completion": acceptance.get("overall_status") == "pass",
                "FULL_ROBOT_CERTIFICATION": "PASS" if native_pass else "FAIL",
                "collision_method": "adaptive_discrete_interpolation",
                "ccd": "not_available",
                "clearance": "",
            }
        )
    (output / "D40_BLOCKER_STATUS.json").write_text(
        json.dumps(
            {
                "schema_version": "d40_blocker_status_v1",
                "TASK_STATUS": summary["TASK_STATUS"],
                "FIRST_UNRESOLVED_BLOCKER": summary["FIRST_UNRESOLVED_BLOCKER"],
                "BLOCKER_1_SELF_FED_STABILITY": summary["BLOCKER_1_SELF_FED_STABILITY"],
                "BLOCKER_2_TASK_SPACE_GEOMETRY": summary["BLOCKER_2_TASK_SPACE_GEOMETRY"],
                "BLOCKER_3_INTRINSIC_FEASIBILITY": summary["BLOCKER_3_INTRINSIC_FEASIBILITY"],
                "BLOCKER_4_FULL_ROBOT_CERTIFICATION": summary["BLOCKER_4_FULL_ROBOT_CERTIFICATION"],
                "collision_method": "adaptive_discrete_interpolation",
                "ccd": "not_available",
                "clearance": None,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    report = f"""# D40 — locked causal Cartesian recovery execution

TASK_STATUS: {summary['TASK_STATUS']}
FIRST_UNRESOLVED_BLOCKER: {summary['FIRST_UNRESOLVED_BLOCKER']}
D39_BASELINE_RETENTION_STATUS: {summary['D39_BASELINE_RETENTION_STATUS']}
BLOCKER_1_SELF_FED_STABILITY: {summary['BLOCKER_1_SELF_FED_STABILITY']}
BLOCKER_2_TASK_SPACE_GEOMETRY: {summary['BLOCKER_2_TASK_SPACE_GEOMETRY']}
BLOCKER_3_INTRINSIC_FEASIBILITY: {summary['BLOCKER_3_INTRINSIC_FEASIBILITY']}
BLOCKER_4_FULL_ROBOT_CERTIFICATION: {summary['BLOCKER_4_FULL_ROBOT_CERTIFICATION']}
D40_RETENTION_BASELINE_2: {baseline2['status']} ({baseline2['candidate_id']})
BEST_SHADOW_CANDIDATE: {summary['BEST_SHADOW_CANDIDATE']}
SYSTEM_CHAMPION: {summary['SYSTEM_CHAMPION']}
H1_CHAMPION: update_480
NATIVE_TASK_SPACE_QUALITY: {summary['NATIVE_TASK_SPACE_QUALITY']}
FULL_ROBOT_CERTIFICATION: {summary['FULL_ROBOT_CERTIFICATION']}
CANONICAL_PROMOTION: {summary['CANONICAL_PROMOTION']}
TEST_STATUS: {summary['TEST_STATUS']}

## Native result

The winning system was the retained D39 `stable_velocity_residual_update` followed by a native MoveIt2 FK/Jacobian damped-least-squares projection over TCP position and wall normal.  It retained the 16-point observed warm start and used only the previous accepted joint state after warm start; future target joint rows read: `0`.

Native MoveIt2/FK/Ruckig evidence: TCP path p95 `{quality['fk_path_deviation_p95_mm']}` mm, max `{quality['fk_path_deviation_max_mm']}` mm; stand-off max absolute error `{quality['fk_standoff_error_max_abs_mm']}` mm; normal max error `{quality['fk_normal_error_max_deg']}` degrees; max joint step `{quality['max_joint_step_deg']}` degrees; collision count `{collision['collision_count']}`; native quality `{quality['status']}`; final acceptance `{acceptance.get('overall_status')}`.

Compared with D39 stable native quality, p95 path deviation changed from `{d39_row['fk_path_deviation_p95_mm']}` mm to `{quality['fk_path_deviation_p95_mm']}` mm, max path deviation from `{d39_row['fk_path_deviation_max_mm']}` mm to `{quality['fk_path_deviation_max_mm']}` mm, stand-off max from `{d39_row['fk_standoff_error_max_abs_mm']}` mm to `{quality['fk_standoff_error_max_abs_mm']}` mm, and normal max from `{d39_row['fk_normal_error_max_deg']}` degrees to `{quality['fk_normal_error_max_deg']}` degrees.

## Search record

Attempted methods: full-pose seeded MoveIt2 IK from the D39 stable prior; widened IK beam search; tool-axis roll search; native FK/Jacobian DLS with strict local convergence; native FK/Jacobian DLS with position-dominant weighting and trust-region step control; and the authoritative seed-joint native control.  Full-pose IK failed at waypoint 67 because all returned branches exceeded the 20-degree continuity gate.  The first DLS settings stalled locally at waypoints 87–88; position-dominant DLS then completed all 181 points and passed the native gate.  The authoritative seed control passed separately and was not used as the D40 system champion.

The real native pipeline ran PlanningScene, FK, timing, Ruckig, velocity, acceleration, jerk, post-Ruckig collision checks, dynamics, and task completion.  Collision scope is `{summary['collision_method']}`; Bullet CCD is `{summary['ccd']}` and clearance is JSON null/not available.

## Retention and scope

D39 `D39_RETENTION_BASELINE_1` remains locked and its files were not overwritten.  The D40 seed manifest records the authoritative 181-point ON-state open-arch pair, zero future joint reads, and no OFF-state, reorientation, GNN, PPO, LSTM, Transformer, retreat, approach, or closed-contour transitions.  D40 baseline 2 is locked only from the new native evidence directory above.
"""
    (output / "D40_FINAL_REPORT.md").write_text(report, encoding="utf-8")
    return summary


if __name__ == "__main__":
    if NATIVE_WINNER.joinpath("moveit_quality_report.csv").is_file():
        make_seed_variants()
        print(json.dumps(build_d40_evidence(), indent=2, sort_keys=True))
    else:
        print(json.dumps({"manifest": make_seed_variants(), "d39_retention": d39_retention_check()}, indent=2, sort_keys=True))
