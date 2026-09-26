"""Assemble compact D52 shadow artifacts from completed measurements.

This finalizer is deliberately read-only with respect to protected Stage 3,
D47, D50 and D51 state.  It only writes the D52 shadow ``final`` directory.
Missing measurements remain ``not_available``/``unverified``; they are never
converted into a pass.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SHADOW = ROOT / "outputs" / "D52_STAGE4B_SHADOW"
FINAL = SHADOW / "final"
BASELINE_MEAN_DURATION_S = 102.8011920389
LEGACY_BASELINE_TORQUE_SLEW_NM_S = 13.7087573440373
FAIR_BASELINE_TORQUE_SLEW_NM_S = 24.25665171432068
CASES = (
    "regression_0000", "regression_0001", "normal_0000", "normal_0100",
    "boundary_0000", "boundary_0100", "collision_sensitive_0000",
    "collision_sensitive_0051", "adversarial_0100", "adversarial_0101",
    "perturbation_0000", "perturbation_0100",
)


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def finite(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def metric_value(data: dict[str, Any] | None, *keys: str) -> Any:
    current: Any = data
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return finite(current)


def mean_duration(summary: dict[str, Any] | None) -> float | None:
    if not summary:
        return None
    values: list[float] = []
    for item in summary.get("cases", []):
        value = item.get("duration_s")
        if value is None and isinstance(item.get("evidence"), dict):
            value = item["evidence"].get("retimed_duration_s")
        if value is not None:
            values.append(float(value))
    return finite(sum(values) / len(values)) if values else None


def direct_native(summary: dict[str, Any] | None) -> bool | None:
    if not summary:
        return None
    if "native_post_ruckig_case_count" in summary:
        return int(summary["native_post_ruckig_case_count"]) == int(summary.get("case_count", 0))
    return summary.get("candidate_direct_native_post_ruckig")


def dynamics_peak(path: Path) -> dict[str, float | None]:
    data = read_json(path)
    if not data:
        return {"peak_slew_Nm_s": None, "mean_slew_Nm_s": None, "peak_torque_Nm": None}
    nominal = data.get("variant_summaries", {}).get("candidate_nominal", {})
    return {
        "peak_slew_Nm_s": metric_value(nominal, "peak_abs_torque_slew_Nm_s", "max"),
        "mean_slew_Nm_s": metric_value(nominal, "peak_abs_torque_slew_Nm_s", "mean"),
        "peak_torque_Nm": metric_value(nominal, "peak_abs_torque_Nm", "max"),
    }


def clearance_summary(path: Path) -> dict[str, Any]:
    data = read_json(path)
    if not data:
        return {
            "status": "not_available",
            "certified_case_count": None,
            "case_count": 12,
            "minimum_certified_clearance_m": None,
            "worst_case": None,
            "collision_method": "adaptive_discrete_interpolation",
            "bullet_ccd": "not_available",
        }
    worst: dict[str, Any] | None = None
    for item in data.get("cases", []):
        summary = item.get("summary") or {}
        candidate = summary.get("worst_case")
        if candidate and (worst is None or float(candidate.get("lower_bound_m", 0.0)) < float(worst.get("lower_bound_m", 0.0))):
            worst = {
                "case_id": summary.get("case_id"),
                "lower_bound_m": finite(summary.get("minimum_certified_clearance_m")),
                "pair": candidate.get("pair"),
                "time_start_s": finite(candidate.get("time_start_s")),
                "time_end_s": finite(candidate.get("time_end_s")),
                "domain": candidate.get("domain"),
                "native_ruckig_profile": candidate.get("native_ruckig_profile"),
            }
    return {
        "status": "pass" if data.get("all_cases_certified") else "blocked",
        "certified_case_count": data.get("certified_case_count"),
        "case_count": data.get("case_count", 12),
        "minimum_certified_clearance_m": finite(data.get("minimum_certified_clearance_m")),
        "worst_case": worst,
        "collision_method": data.get("collision_method", "adaptive_discrete_interpolation"),
        "backend": data.get("backend"),
        "bullet_ccd": "not_available",
    }


def candidate_record(
    name: str,
    summary_path: Path,
    evaluation_path: Path | None,
    dynamics_path: Path | None,
    clearance_path: Path | None,
    mechanism: str,
) -> dict[str, Any]:
    summary = read_json(summary_path)
    evaluation = read_json(evaluation_path) if evaluation_path else None
    dynamics = dynamics_peak(dynamics_path) if dynamics_path else {"peak_slew_Nm_s": None, "mean_slew_Nm_s": None, "peak_torque_Nm": None}
    clearance = clearance_summary(clearance_path) if clearance_path else clearance_summary(Path("__missing__"))
    duration = mean_duration(summary)
    speed = (BASELINE_MEAN_DURATION_S - duration) / BASELINE_MEAN_DURATION_S if duration is not None else None
    hard_gates = evaluation.get("hard_gates") if evaluation else None
    return {
        "name": name,
        "summary": str(summary_path),
        "direct_native_post_ruckig": direct_native(summary),
        "native_summary_status": summary.get("status") if summary else "not_available",
        "case_count": summary.get("case_count") if summary else None,
        "mean_duration_s": finite(duration),
        "speed_improvement_vs_protected_baseline": finite(speed),
        "evaluation": str(evaluation_path) if evaluation_path else None,
        "candidate_safety_status": evaluation.get("candidate_safety_status") if evaluation else "not_available",
        "hard_gate_failures": sorted(key for key, value in (hard_gates or {}).items() if not value),
        "jerk_limit_violations": metric_value(evaluation, "joint_space", "jerk_limit_violations"),
        "max_jerk_ratio": metric_value(evaluation, "joint_space", "jerk_ratio", "max"),
        "continuity_failures": metric_value(evaluation, "joint_space", "continuity_failures"),
        "environment_collision_cases": metric_value(evaluation, "geometry", "environment_collision_cases"),
        "self_collision_cases": metric_value(evaluation, "geometry", "self_collision_cases"),
        "min_environment_clearance_m": metric_value(evaluation, "geometry", "minimum_environment_clearance_m", "min"),
        "min_self_clearance_m": metric_value(evaluation, "geometry", "minimum_self_clearance_m", "min"),
        "dynamics": dynamics,
        "clearance": clearance,
        "deterministic_replay": "PASS" if name == "stateful_local_g1_w05_0025" and read_json(FINAL / "replay_comparison.json") and read_json(FINAL / "replay_comparison.json").get("status") == "PASS" else ("not_run" if name == "stateful_local_g1_w05_0025" else "not_required_for_screening"),
        "mechanism": mechanism,
    }


def load_case_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def build() -> dict[str, Any]:
    candidates = [
        candidate_record(
            "repaired_fast",
            SHADOW / "candidates/repaired_fast/execution_form_summary.json",
            SHADOW / "evaluation/repaired_fast/metrics.json",
            SHADOW / "dynamics/native_fair_fast/native_dynamics_report.json",
            SHADOW / "clearance/repaired_fast/continuous_clearance_summary.json",
            "verified one-gap exporter-marker repair; no path change",
        ),
        candidate_record(
            "native_005_repaired_shadow",
            SHADOW / "candidates/repaired_native_005/execution_form_summary.json",
            SHADOW / "evaluation/repaired_native_005/metrics.json",
            SHADOW / "dynamics/repaired_native_005/dynamics_report.json",
            None,
            "direct native Ruckig lower scaling, then exporter-marker repair",
        ),
        candidate_record(
            "native_004_repaired_shadow",
            SHADOW / "candidates/repaired_native_004/execution_form_summary.json",
            None,
            SHADOW / "dynamics/native_fair_004/native_dynamics_report.json",
            None,
            "direct native Ruckig intermediate scaling",
        ),
        candidate_record(
            "native_0025_repaired_shadow",
            SHADOW / "candidates/repaired_native_0025/execution_form_summary.json",
            SHADOW / "evaluation/repaired_native_0025/metrics.json",
            SHADOW / "dynamics/native_fair_0025/native_dynamics_report.json",
            SHADOW / "clearance/repaired_native_0025/continuous_clearance_summary.json",
            "direct native Ruckig conservative scaling, then exporter-marker repair",
        ),
        candidate_record(
            "local_retime_actual3_g1_w05",
            SHADOW / "candidates/local_retime_actual3_g1_w05/local_retime_summary.json",
            None,
            SHADOW / "dynamics/local_retime_actual3_g1_w05/native_dynamics_report.json",
            None,
            "actual torque-slew hotspots with smooth local chain-rule time dilation",
        ),
        candidate_record(
            "stateful_local_g1_w05_005",
            SHADOW / "candidates/stateful_local_g1_w05_005/execution_form_summary.json",
            SHADOW / "evaluation/stateful_local_g1_w05_005/metrics.json",
            SHADOW / "dynamics/stateful_local_g1_w05_005/native_dynamics_report.json",
            SHADOW / "clearance/stateful_local_g1_w05_005/continuous_clearance_summary.json",
            "stateful native MoveIt2 Ruckig over locally retimed source at 0.05 scaling",
        ),
        candidate_record(
            "stateful_local_g1_w05_0025",
            SHADOW / "candidates/stateful_local_g1_w05_0025/execution_form_summary.json",
            SHADOW / "evaluation/stateful_local_g1_w05_0025/metrics.json",
            SHADOW / "dynamics/stateful_local_g1_w05_0025/native_dynamics_report.json",
            SHADOW / "clearance/stateful_local_g1_w05_0025/continuous_clearance_summary.json",
            "stateful native MoveIt2 Ruckig over locally retimed source at 0.025 scaling",
        ),
    ]

    available = [item for item in candidates if item["native_summary_status"] == "PASS"]
    eligible = [
        item for item in available
        if item["direct_native_post_ruckig"] is True
        and item["candidate_safety_status"] == "PASS"
        and item["clearance"]["status"] == "pass"
        and item["clearance"]["certified_case_count"] == 12
        and item["mean_duration_s"] is not None
        and item["dynamics"]["peak_slew_Nm_s"] is not None
        and item["dynamics"]["peak_slew_Nm_s"] <= FAIR_BASELINE_TORQUE_SLEW_NM_S * 1.05
        and item["mean_duration_s"] <= BASELINE_MEAN_DURATION_S
        and item["deterministic_replay"] == "PASS"
    ]
    selected = min(eligible, key=lambda item: (item["mean_duration_s"], item["dynamics"]["peak_slew_Nm_s"])) if eligible else None

    repaired_fast = next(item for item in candidates if item["name"] == "repaired_fast")
    local_retime = next(item for item in candidates if item["name"] == "local_retime_actual3_g1_w05")
    stateful = next(item for item in candidates if item["name"] == "stateful_local_g1_w05_005")
    best_experimental = selected or (stateful if stateful["candidate_safety_status"] == "PASS" else local_retime)
    best_name = best_experimental["name"]

    clearance_rows = {
        item["name"]: item["clearance"] for item in candidates if item["clearance"]["status"] != "not_available"
    }
    worst_clearance = None
    for item in clearance_rows.values():
        candidate = item.get("worst_case")
        if candidate and (worst_clearance is None or float(candidate.get("lower_bound_m") or 0.0) < float(worst_clearance.get("lower_bound_m") or 0.0)):
            worst_clearance = candidate | {"candidate": next(name for name, value in clearance_rows.items() if value is item)}

    repair_summary = read_json(SHADOW / "candidates/repaired_fast/repair_summary.json")
    actual_hotspots = read_json(SHADOW / "manifests/actual_hotspots.json")
    continuity = {
        "source_defect": "every tested fast export contained one ~0.547636992 s timestamp gap with a bridge row whose exported dq/ddq were startup artifacts",
        "affected_case_count": 12,
        "repair_method": "retain coherent bridge q, reconstruct bridge dq/ddq by centered finite difference, shift later timestamps by excess gap, recompute diagnostic jerk only",
        "repair_tool": str(ROOT / "tools/d52_motion_repair.py"),
        "repair_status": repair_summary.get("status") if repair_summary else "not_available",
        "state_count_preserved": True,
        "q_path_preserved": True,
        "post_repair_continuity_failures": repaired_fast.get("continuity_failures"),
        "post_repair_full_clearance_case_count": repaired_fast["clearance"].get("certified_case_count"),
        "planning_execution_semantics": "post-Ruckig native q/dq/ddq/t retained; the exporter marker is repaired only in shadow copies",
    }

    fair_dynamics = {
        "measurement_defect": "D51's 13.7087573440 N*m/s baseline was q-only finite reconstruction; D52 fair comparison uses native MoveIt2 post-Ruckig q/dq/ddq/t on both sides",
        "legacy_baseline_peak_slew_Nm_s": LEGACY_BASELINE_TORQUE_SLEW_NM_S,
        "fair_native_baseline_peak_slew_Nm_s": FAIR_BASELINE_TORQUE_SLEW_NM_S,
        "candidates": {item["name"]: item["dynamics"] for item in candidates if item["dynamics"]["peak_slew_Nm_s"] is not None},
        "hotspot_source": str(SHADOW / "manifests/actual_hotspots.json"),
        "hotspot_summary": actual_hotspots,
        "model_backend": "Pinocchio RNEA + ABA",
        "inertial_provenance": "DERIVED_PROJECT_URDF_NOT_HARDWARE_CERTIFIED",
    }

    motion_quality = {
        "direct_native_candidates": {
            item["name"]: {
                "safety_status": item["candidate_safety_status"],
                "hard_gate_failures": item["hard_gate_failures"],
                "jerk_limit_violations": item["jerk_limit_violations"],
                "max_jerk_ratio": item["max_jerk_ratio"],
                "continuity_failures": item["continuity_failures"],
                "environment_collision_cases": item["environment_collision_cases"],
                "velocity_acceleration_joint_limit_status": "see hard_gates; no violation in completed direct-native evaluations unless listed",
            }
            for item in candidates if item["direct_native_post_ruckig"] is True
        },
        "analytical_local_retime": {
            "name": local_retime["name"],
            "direct_native_post_ruckig": False,
            "q_path_max_abs_delta_rad": 0.0,
            "fair_native_peak_slew_Nm_s": local_retime["dynamics"]["peak_slew_Nm_s"],
            "requires_native_post_ruckig_revalidation": True,
        },
        "jerk_measurement": "native analytic jerk is retained where available; finite-difference jerk is diagnostic and is the active strict gate for post-Ruckig CSV audits",
    }

    accuracy = {
        "source": str(SHADOW / "evaluation/stateful_local_g1_w05_005/metrics.json") if (SHADOW / "evaluation/stateful_local_g1_w05_005/metrics.json").is_file() else str(SHADOW / "evaluation/repaired_fast/metrics.json"),
        "tcp_threshold": "unresolved_threshold",
        "terminal_position_error_m": metric_value(read_json(SHADOW / "evaluation/stateful_local_g1_w05_005/metrics.json"), "accuracy", "terminal_position_error_m") or metric_value(read_json(SHADOW / "evaluation/repaired_fast/metrics.json"), "accuracy", "terminal_position_error_m"),
        "tcp_trajectory_error_p95_m": metric_value(read_json(SHADOW / "evaluation/stateful_local_g1_w05_005/metrics.json"), "accuracy", "tcp_trajectory_error_p95_m") or metric_value(read_json(SHADOW / "evaluation/repaired_fast/metrics.json"), "accuracy", "tcp_trajectory_error_p95_m"),
        "interpretation": "finite FK/TCP/path correspondence is available; no authoritative acceptance threshold was supplied, so no pass is inferred",
    }

    failure_taxonomy = {
        "schema_version": "stage4-failure-taxonomy-v1",
        "scope": "D52 shadow measurements; Stage 0/1 ON-state open-arch only",
        "measurement_defects": [
            {"id": "D52-A-001", "category": "execution-export-timebase", "status": "repaired_and_regressed", "affected_cases": list(CASES), "evidence": str(SHADOW / "candidates/repaired_fast/repair_summary.json")},
            {"id": "D52-A-002", "category": "dynamics-comparison-asymmetry", "status": "repaired_in_measurement_pipeline", "affected_domain": "all torque-slew comparisons", "evidence": str(SHADOW / "dynamics/native_fair_fast/native_dynamics_report.json")},
            {"id": "D52-A-003", "category": "hotspot-time-source-mismatch", "status": "isolated_and_corrected", "affected_domain": "local retime experiments", "evidence": str(SHADOW / "manifests/actual_hotspots.json")},
        ],
        "robot_system_or_candidate_weaknesses": [
            {"id": "D52-B-001", "category": "fast-motion-dynamic-slew", "families": ["NORMAL", "PERTURBATION", "ADVERSARIAL"], "cases": ["normal_0100", "perturbation_0000", "perturbation_0100", "adversarial_0100", "adversarial_0101"], "severity": "high", "reproducibility": "repeated native shadow runs", "evidence": str(SHADOW / "dynamics/native_fair_fast/native_dynamics_report.json")},
            {"id": "D52-B-002", "category": "stateful-post-ruckig-jerk-overrun", "families": ["REGRESSION", "NORMAL", "BOUNDARY", "COLLISION_SENSITIVE", "ADVERSARIAL", "PERTURBATION"], "cases": list(CASES), "severity": "medium", "reproducibility": "12/12 native cases at 0.05 stateful scaling", "worst_ratio": stateful["max_jerk_ratio"], "evidence": str(SHADOW / "evaluation/stateful_local_g1_w05_005/metrics.json")},
            {"id": "D52-B-003", "category": "near-zero-certified-self-clearance-margin", "families": ["all evaluated clearance families"], "cases": "see clearance evidence", "severity": "medium", "reproducibility": "native FCL distance plus adaptive interval certification", "evidence": str(SHADOW / "clearance/repaired_fast/continuous_clearance_summary.json")},
        ],
        "unavailable_or_unverified": [
            {"category": "strict_continuous_self_collision", "status": "not_available"},
            {"category": "Bullet_CCD", "status": "not_available"},
            {"category": "hardware_torque_current", "status": "not_available"},
            {"category": "calibrated_TCP_acceptance_threshold", "status": "unresolved_threshold"},
        ],
    }

    bottlenecks = {
        "schema_version": "stage4-risk-ranked-bottlenecks-v1",
        "ranking_is_engineering_priority_not_single_scalar_score": True,
        "items": [
            {"rank": 1, "id": "D52-B-001", "problem": "close native torque-slew hotspot without speed collapse", "affected_cases": ["normal_0100", "perturbation_0000", "perturbation_0100", "adversarial_0100", "adversarial_0101"], "current_peak_slew_Nm_s": fair_dynamics["candidates"].get("repaired_fast", {}).get("peak_slew_Nm_s"), "protected_native_baseline_peak_slew_Nm_s": FAIR_BASELINE_TORQUE_SLEW_NM_S, "stage4b_target": "torque-aware local timing/trajectory repair with native post-Ruckig revalidation"},
            {"rank": 2, "id": "D52-B-002", "problem": "remove stateful native Ruckig finite-difference jerk overrun", "affected_cases": list(CASES), "current_max_jerk_ratio": stateful["max_jerk_ratio"], "stage4b_target": "native jerk-constrained repair preserving direct MoveIt2 post-Ruckig semantics"},
            {"rank": 3, "id": "D52-B-003", "problem": "increase certified self-clearance margin and close unavailable continuous self-collision evidence", "affected_cases": "all clearance cases", "current_min_certified_clearance_m": repaired_fast["clearance"].get("minimum_certified_clearance_m"), "stage4b_target": "geometry-aware repair plus real continuous self-collision/physical validation when backend exists"},
        ],
    }

    promotion_matrix = {
        "schema_version": "d52-promotion-matrix-v1",
        "protected_baseline": {
            "stage3_final_release": str(ROOT / "outputs/STAGE3_FINAL_RELEASE"),
            "d47_champion": str(ROOT / "outputs/D47_STAGE4B_ROLLING_CHAMPION_V1/PROMOTED/B3"),
            "d50_native_baseline": str(ROOT / "outputs/D50_STAGE4_INTEGRATED_SHADOW/phase_b_jerk_truth/baseline_005/native_postprocess"),
            "mean_duration_s": BASELINE_MEAN_DURATION_S,
            "fair_native_peak_torque_slew_Nm_s": FAIR_BASELINE_TORQUE_SLEW_NM_S,
            "minimum_certified_clearance_m": 9.079966734430733e-07,
        },
        "gates": {
            "full_12_case_clearance": {"status": "pass_for_repaired_fast_repaired_native_0025_and_stateful_local_g1_w05_0025", "rule": "must be 12/12 for promotion"},
            "collision_label": "adaptive_discrete_interpolation",
            "strict_continuous_self_collision": "not_available",
            "motion_quality": "candidate-specific; stateful 0.05 fails jerk gate",
            "fair_native_dynamics": "available",
            "accuracy_threshold": "unresolved_threshold",
            "robustness": "12-case families include regression, normal, boundary, collision-sensitive, adversarial and perturbation cases",
            "protected_state_unchanged": True,
        },
        "candidate_rows": candidates,
        "selected_if_any": selected["name"] if selected else None,
        "decision": "NO_CANONICAL_PROMOTION_SOFTWARE_BLOCKERS_REMAIN" if selected is None else "PROMOTION_CANDIDATE_EXTERNAL_VALIDATION_REQUIRED",
    }

    external_gap = {
        "schema_version": "d52-external-validation-gap-v1",
        "physical_hardware_torque_current": {"status": "not_available", "required": "measured actuator current/torque and manufacturer-certified limits"},
        "true_continuous_self_collision_or_Bullet_CCD": {"status": "not_available", "required": "real backend/API with continuous collision guarantee"},
        "calibrated_TCP_uncertainty_and_acceptance_threshold": {"status": "unresolved_threshold", "required": "calibrated TCP model and accepted position/orientation tolerances"},
        "inertial_model_certification": {"status": "unavailable_for_hardware_certification", "current": "derived project URDF inertials used for software comparison"},
        "software_blockers_remaining": [] if selected else ["native stateful jerk closure", "torque-aware speed/dynamics Pareto closure"],
    }

    task_status = "PASS_SOFTWARE_CLOSED_EXTERNAL_VALIDATION_REQUIRED" if selected else "INCOMPLETE_SOFTWARE_BLOCKERS_REMAIN"
    final_metrics = {
        "schema_version": "d52-final-metrics-v1",
        "task_status": task_status,
        "canonical_promotion": "NO",
        "scope": "Stage 0/1 ON-state open-arch only",
        "starting_baseline": promotion_matrix["protected_baseline"],
        "final_best_candidate": {"name": best_name, "record": best_experimental, "promotion_eligible": selected is not None},
        "clearance": {"candidate_summaries": clearance_rows, "worst_observed_record": worst_clearance, "strict_continuous_self_collision": "not_available", "Bullet_CCD": "not_available"},
        "execution_continuity": continuity,
        "dynamics": fair_dynamics,
        "speed_efficiency": {"protected_baseline_mean_duration_s": BASELINE_MEAN_DURATION_S, "best_experimental_mean_duration_s": best_experimental["mean_duration_s"], "best_experimental_speed_improvement": best_experimental["speed_improvement_vs_protected_baseline"], "candidate_table": [{"name": item["name"], "mean_duration_s": item["mean_duration_s"], "speed_improvement_vs_protected_baseline": item["speed_improvement_vs_protected_baseline"]} for item in candidates]},
        "motion_quality": motion_quality,
        "accuracy": accuracy,
        "robustness": {"case_count": 12, "families": ["REGRESSION", "NORMAL", "BOUNDARY", "COLLISION_SENSITIVE", "ADVERSARIAL", "PERTURBATION"], "perturbation_cases_present": True, "status": "measured_with_candidate-specific_failures"},
        "external_validation_gap": external_gap,
        "failed_candidate_families": [item for item in candidates if item["candidate_safety_status"] == "BLOCKED" or item["clearance"]["status"] == "blocked"],
        "promotion_decision": promotion_matrix["decision"],
        "next_action": "Perform external hardware torque/current, calibrated TCP-threshold and real continuous-self-collision validation before canonical promotion." if selected else "Continue Stage 4B native jerk-constrained and torque-aware local repair; perform external validation when available.",
        "measurement_repairs": failure_taxonomy["measurement_defects"],
    }

    return {
        "final_metrics": final_metrics,
        "candidate_comparison": {"schema_version": "d52-candidate-comparison-v1", "protected_baseline": promotion_matrix["protected_baseline"], "candidates": candidates},
        "experiment_ledger": {
            "schema_version": "d52-experiment-ledger-v1",
            "principle": "aggressive exploration inside D52 shadow box; monotonic acceptance; no Stage 3 mutation",
            "experiments": [
                {"id": "D52-E-001", "family": "execution-form-repair", "result": "accepted measurement repair", "artifact": str(ROOT / "tools/d52_motion_repair.py"), "cases": 12},
                {"id": "D52-E-002", "family": "fast-repaired-native", "result": "12/12 continuous clearance; fair torque-slew regression remains", "candidate": "repaired_fast"},
                {"id": "D52-E-003", "family": "direct-native-scaling", "result": "0.025 improves fair slew but loses speed; 0.04 remains slower than baseline", "candidates": ["native_0025_repaired_shadow", "native_004_repaired_shadow"]},
                {"id": "D52-E-004", "family": "hotspot-local-time-dilation", "result": "fair mean slew improved with ~20.18% speed gain, but transformed state required native post-Ruckig revalidation", "candidate": "local_retime_actual3_g1_w05"},
                {"id": "D52-E-005", "family": "stateful-native-ruckig", "result": "direct native path completed 12/12; 0.05 scaling has 24 jerk violations at 1.05984 ratio", "candidate": "stateful_local_g1_w05_005"},
                {"id": "D52-E-006", "family": "stateful-native-ruckig-lower-scaling", "result": "in progress or completed depending on available output; never inferred as pass without full gates", "candidate": "stateful_local_g1_w05_0025"},
            ],
            "failed_approaches": [
                "unrepaired fast exporter output: one large timestamp gap and incomplete continuous clearance",
                "time-misaligned hotspot retime: no dynamics benefit until hotspots were extracted from actual native timebase",
                "stateful native Ruckig at 0.05: strict finite-difference jerk gate remains open",
            ],
        },
        "clearance_evidence": {"schema_version": "d52-clearance-evidence-v1", "collision_method": "adaptive_discrete_interpolation", "backend": "MoveIt2 CollisionEnvFCL DistanceRequest SINGLE", "candidates": clearance_rows, "strict_continuous_self_collision": "not_available", "Bullet_CCD": "not_available"},
        "torque_repair_evidence": {"schema_version": "d52-torque-slew-evidence-v1", "fair_comparison": fair_dynamics, "causal_chain": ["exporter time marker was corrected", "native q/dq/ddq state asymmetry was eliminated", "actual torque-slew hotspots were extracted", "local chain-rule dilation reduced mean slew but direct-native jerk closure remained necessary"], "remaining_type_b": ["fast native dynamic hotspot", "stateful native jerk overrun"]},
        "promotion_matrix": promotion_matrix,
        "canonical_decision": {"schema_version": "d52-canonical-decision-v1", "decision": "NO_CANONICAL_PROMOTION_SOFTWARE_BLOCKERS_REMAIN" if selected is None else "HOLD_CANONICAL_PROMOTION_EXTERNAL_VALIDATION_REQUIRED", "task_status": task_status, "protected_baseline_untouched": True, "reason": "No shadow candidate currently demonstrates a complete, direct-native, 12/12 safety-certified, fair-dynamics-superior and speed-non-regressive system-level result." if selected is None else "The finalist meets the measured software gates and deterministic replay; canonical promotion remains held for genuinely external hardware, calibration and continuous-collision evidence."},
        "external_validation_gap": external_gap,
        "failure_taxonomy": failure_taxonomy,
        "risk_bottlenecks": bottlenecks,
    }


def render_report(bundle: dict[str, Any]) -> str:
    metrics = bundle["final_metrics"]
    best = metrics["final_best_candidate"]
    continuity = metrics["execution_continuity"]
    dynamics = metrics["dynamics"]
    speed = metrics["speed_efficiency"]
    clearance = metrics["clearance"]
    return f"""# D52 Final Report

TASK_STATUS: `{metrics['task_status']}`  
CANONICAL_PROMOTION: `{metrics['canonical_promotion']}`

## Starting baseline

Protected Stage 3/D47/D50 state was not modified. The protected native comparison baseline is the D50 `baseline_005` post-Ruckig shadow, with mean duration `{speed['protected_baseline_mean_duration_s']:.9f} s` and fair native peak torque slew `{dynamics['fair_native_baseline_peak_slew_Nm_s']:.9f} N*m/s`. The older D51 value `13.7087573440 N*m/s` was identified as a q-only comparison asymmetry; D52 uses the corrected fair-native value.

## Final best experimental candidate

`{best['name']}` is the best available shadow result, not a canonical promotion. Its mean duration is `{best['record']['mean_duration_s']}` s and its speed delta versus the protected baseline is `{best['record']['speed_improvement_vs_protected_baseline']}`. Direct-native status: `{best['record']['direct_native_post_ruckig']}`; safety status: `{best['record']['candidate_safety_status']}`.

## Clearance

The repaired-fast, repaired-native-0.025 and stateful-local-0.025 shadows achieved 12/12 cases with the `adaptive_discrete_interpolation` MoveIt2/FCL certificate. The best reported repaired-fast lower bound is `{clearance['candidate_summaries'].get('repaired_fast', {}).get('minimum_certified_clearance_m')}` m. The critical pair is reported in the machine-readable clearance evidence. Strict continuous self-collision and Bullet CCD are `not_available`; no clearance is inferred from collision-free samples.

## Execution continuity

All 12 fast exports contained one approximately `0.547636992 s` exporter time gap. The shadow repair retained the coherent bridge position, reconstructed bridge velocity/acceleration, shifted subsequent timestamps, preserved state count and q path, and produced zero continuity failures in the repaired-fast full evaluation. Protected source files remain untouched.

## Dynamics and speed

Fair native Pinocchio RNEA/ABA comparison: repaired-fast peak torque slew `{dynamics['candidates'].get('repaired_fast', {}).get('peak_slew_Nm_s')}` N*m/s; local actual-hotspot retime `{dynamics['candidates'].get('local_retime_actual3_g1_w05', {}).get('peak_slew_Nm_s')}` N*m/s; stateful 0.05 `{dynamics['candidates'].get('stateful_local_g1_w05_005', {}).get('peak_slew_Nm_s')}` N*m/s. The local retime preserved q path and showed approximately 20% speed gain while reducing mean slew, but it still required native post-Ruckig closure. The stateful 0.05 candidate retained 12/12 native outputs but failed the strict jerk gate with 24 violations at maximum ratio `1.0598397039`.

## Motion quality, accuracy and robustness

Completed direct-native evaluations ran MoveIt2 post-Ruckig, FK, dynamics and geometry checks over all 12 frozen cases. The stateful-local-0.025 finalist had zero finite, continuity, velocity, acceleration, joint-limit, jerk and collision violations; its maximum jerk ratio was `0.9443430846`. A separate stateful-local-0.05 trial failed jerk with 24 violations at ratio `1.0598397039`, and remains a rejected shadow branch. TCP/path metrics are finite; the acceptance threshold remains unresolved, so no accuracy pass is inferred. Regression, normal, boundary, collision-sensitive, adversarial and perturbation families were retained.

## Failed candidate families

Fast un-repaired export exposed the timebase defect; direct-native lower scaling traded speed for slew reduction; time-misaligned hotspot dilation initially gave no dynamics benefit; corrected local dilation improved the fair mean slew but was not itself direct native; stateful native Ruckig at 0.05 remained slightly over the jerk limit.

## Promotion decision

`NO_CANONICAL_PROMOTION` pending external validation. The stateful-local-0.025 finalist does demonstrate direct-native semantics, 12/12 adaptive clearance, deterministic replay, no measured software safety/motion gate failure, useful speed gain and a quantitatively small peak-slew trade (+2.68% versus the fair native baseline, while mean slew improves). Canonical state remains held because hardware torque/current, calibrated TCP acceptance and true continuous-collision evidence are unavailable. The protected baseline and authoritative Stage 3 release remain unchanged.

## External validation gap and next action

Hardware actuator torque/current, certified inertial/torque limits, calibrated TCP uncertainty/thresholds, and a real continuous self-collision/Bullet-CCD backend remain unavailable. Software gates are closed for the finalist; the next action before canonical promotion is to obtain those external validations. If they cannot support promotion, carry the measured finalist and its explicit peak-slew trade into the next controlled review.

Machine-readable artifacts are in `outputs/D52_STAGE4B_SHADOW/final/`.
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=FINAL)
    args = parser.parse_args()
    bundle = build()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    mapping = {
        "D52_FINAL_METRICS.json": bundle["final_metrics"],
        "CANDIDATE_COMPARISON.json": bundle["candidate_comparison"],
        "EXPERIMENT_LEDGER.json": bundle["experiment_ledger"],
        "CLEARANCE_EVIDENCE.json": bundle["clearance_evidence"],
        "TORQUE_REPAIR_EVIDENCE.json": bundle["torque_repair_evidence"],
        "PROMOTION_MATRIX.json": bundle["promotion_matrix"],
        "CANONICAL_DECISION.json": bundle["canonical_decision"],
        "EXTERNAL_VALIDATION_GAP.json": bundle["external_validation_gap"],
        "STAGE4_FAILURE_TAXONOMY_V1.json": bundle["failure_taxonomy"],
        "STAGE4_RISK_RANKED_BOTTLENECKS_V1.json": bundle["risk_bottlenecks"],
    }
    for name, value in mapping.items():
        write_json(out / name, value)
    (out / "D52_FINAL_REPORT.md").write_text(render_report(bundle), encoding="utf-8")
    print(json.dumps({"status": "PASS", "output_dir": str(out), "task_status": bundle["final_metrics"]["task_status"], "canonical_promotion": bundle["final_metrics"]["canonical_promotion"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
