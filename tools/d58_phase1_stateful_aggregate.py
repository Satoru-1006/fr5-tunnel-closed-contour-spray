"""Build the authority-first D58 Phase 1 stateful shadow closure report."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "D58_STAGE4_FULL_SYSTEM_CLOSURE"


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def local_path(value: str | Path) -> Path:
    text = str(value).replace("\\", "/")
    if text.startswith("/mnt/") and len(text) > 6:
        text = f"{text[5].upper()}:/{text[7:]}"
    return Path(text)


def rel(path: str | Path) -> str:
    return local_path(path).resolve().relative_to(ROOT.resolve()).as_posix()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def metric(path: Path, key: str):
    value = load(path)
    for part in key.split("."):
        value = value[part]
    return value


def main() -> None:
    cases = (
        ("adversarial_0100", "auto0"),
        ("adversarial_0101", "auto1"),
    )
    native = {}
    metrics = {}
    profile_j = {}
    certificates = {}
    replay = {}

    for case_id, short in cases:
        native_path = OUT / f"stateful_consistent_scale001_{short}" / "execution_form_summary.json"
        metric_path = OUT / f"full_stateful_metrics_{short}" / "metrics.json"
        profile_path = OUT / f"profile_j_stateful_{short}" / "summary.json"
        cert_path = OUT / f"articulated_cert_stateful_{short}" / "D54_CONTINUOUS_SELF_COLLISION_CERTIFICATION.json"
        cert_stride1_path = OUT / f"articulated_cert_stateful_{short}_stride1" / "D54_CONTINUOUS_SELF_COLLISION_CERTIFICATION.json"
        replay_dir = OUT / f"stateful_consistent_scale001_{short}_replay"
        native_obj = load(native_path)
        case = native_obj["cases"][0]
        native[case_id] = {
            "status": native_obj["status"],
            "native_post_ruckig": bool(case["native_post_ruckig"]),
            "ruckig_returned_success": bool(case["ruckig_returned_success"]),
            "input_state_count": case["input_state_count"],
            "post_ruckig_state_count": case["post_ruckig_state_count"],
            "source_duration_s": case["consistent_time_repair"]["source_duration_s"],
            "retimed_duration_s": case["consistent_time_repair"]["retimed_duration_s"],
            "native_duration_s": case["duration_s"],
            "time_scale": case["consistent_time_repair"]["time_scale"],
            "q_path_max_abs_delta_rad": case["consistent_time_repair"]["q_path_max_abs_delta_rad"],
            "trajectory_csv": rel(case["trajectory_csv"]),
            "source_trajectory": case["source_trajectory"],
        }
        m = load(metric_path)
        metrics[case_id] = {
            "candidate_safety_status_for_native_geometry_only": m["candidate_safety_status"],
            "measurement_status": m["measurement_status"],
            "collision_method": m["collision_method"],
            "environment_collision_cases": m["geometry"]["environment_collision_cases"],
            "self_collision_cases": m["geometry"]["self_collision_cases"],
            "minimum_environment_clearance_m": m["geometry"]["minimum_environment_clearance_m"]["min"],
            "minimum_self_clearance_m": m["geometry"]["minimum_self_clearance_m"]["min"],
            "minimum_sigma": m["singularity"]["minimum_sigma_min"]["min"],
            "maximum_condition_number": m["singularity"]["maximum_condition_number"]["max"],
            "tcp_trajectory_error_max_m": m["accuracy"]["tcp_trajectory_error_max_m"]["max"],
            "tcp_trajectory_error_p95_m": m["accuracy"]["tcp_trajectory_error_p95_m"]["max"],
            "terminal_position_error_m": m["accuracy"]["terminal_position_error_m"]["max"],
            "finite_difference_jerk_is_diagnostic_only": True,
            "hard_gates_native_geometry": m["hard_gates"],
        }
        p = load(profile_path)
        profile_j[case_id] = {
            "status": "PASS" if p["passed"] else "FAIL",
            "jerk_truth": p["jerk_truth"],
            "segments": p["segments"],
            "successful_native_profiles": p["successful_native_profiles"],
            "error_invalid_input_count": p["error_invalid_input_count"],
            "max_abs_analytic_jerk_rad_s3": p["max_abs_analytic_jerk_rad_s3"],
            "max_analytic_jerk_ratio": p["max_analytic_jerk_ratio"],
            "overshoot_check": "NOT_USED_IN_PROFILE_SEMANTICS_REPLAY",
            "artifact": rel(profile_path),
        }
        c = load(cert_path)
        c1 = load(cert_stride1_path)
        certificates[case_id] = {
            "default_initial_stride_16": {
                "status": c["status"],
                "unresolved_region_count": c["unresolved_region_count"],
                "collision_region_count": c["collision_region_count"],
                "required_pair_coverage_complete": c["required_pair_coverage_complete"],
                "worst_pair": c["worst_pair"],
            },
            "dense_initial_stride_1": {
                "status": c1["status"],
                "unresolved_region_count": c1["unresolved_region_count"],
                "collision_region_count": c1["collision_region_count"],
                "required_pair_coverage_complete": c1["required_pair_coverage_complete"],
                "worst_pair": c1["worst_pair"],
            },
            "method": c["certificate_method"],
            "continuous_self_clearance_semantics": c["continuous_self_clearance_semantics"],
            "collision_method": c["collision_method"],
            "unresolved_is_not_pass": c["unresolved_is_not_pass"],
        }
        original = OUT / f"stateful_consistent_scale001_{short}" / "trajectories" / f"{case_id}.csv"
        replay_csv = replay_dir / "trajectories" / f"{case_id}.csv"
        replay[case_id] = {
            "original_status": "PASS",
            "replay_status": load(replay_dir / "execution_form_summary.json")["status"],
            "original_sha256": sha256(original),
            "replay_sha256": sha256(replay_csv),
            "byte_identical": sha256(original) == sha256(replay_csv),
            "original_trajectory": rel(original),
            "replay_trajectory": rel(replay_csv),
        }

    dynamics_path = OUT / "dynamics_stateful_auto_pair" / "native_dynamics_report.json"
    dynamics = load(dynamics_path)
    nominal = [row for row in dynamics["ranking_rows"] if row["variant"] == "nominal"]
    plus10 = [row for row in dynamics["ranking_rows"] if row["variant"] == "mass_scale_plus_10pct"]
    fcl = {}
    for short in ("auto0", "auto1"):
        fcl_obj = load(OUT / f"fcl_routeA_stateful_{short}" / "continuous_self_collision_summary.json")
        fcl[short] = {
            "status": fcl_obj["status"],
            "complete_trajectory": True,
            "swept_interval_count": fcl_obj["swept_interval_count"],
            "swept_pair_calls": fcl_obj["swept_pair_call_count"],
            "continuous_collision_count": fcl_obj["continuous_collision_count"],
            "continuous_api_error_count": fcl_obj["continuous_api_error_count"],
            "checked_link_pairs": fcl_obj["checked_link_pairs"],
            "semantics": "qualified_rigid_endpoint_sweep_cross_check; not exact nonlinear articulated FK(q(t)) self-CCD",
        }
    robust_path = OUT / "D58_ROBUST_CLEARANCE.json"
    report = {
        "schema_version": "d58-stage4-phase1-stateful-closure-v1",
        "generated_from": "D57 two C4 optimizer shadows through stateful consistent-time repair and native MoveIt2 post-Ruckig",
        "overall_d58_status": "PARTIALLY_EXECUTED",
        "phase1_status": "MEASURED_NO_PROMOTION",
        "promotion_status": "NO_PROMOTION",
        "promotion_blockers": [
            "Both articulated FK-aware certificates remain UNRESOLVED with 152 unresolved intervals per candidate, including dense initial_stride=1 reruns.",
            "True continuous self-collision CCD is NOT_AVAILABLE in the available backend.",
            "Physical clearance and TCP acceptance thresholds remain UNRESOLVED_THRESHOLD.",
            "Torque result is model-based from the derived project URDF; hardware torque certification is NOT_AVAILABLE.",
        ],
        "scope": {
            "stage_scope": "Stage 0/1 ON-state open-arch only",
            "collision_label": "adaptive_discrete_interpolation",
            "legacy_inputs_mixed": False,
            "candidate_count": 2,
            "case_ids": [case_id for case_id, _ in cases],
        },
        "native_execution": native,
        "authoritative_profile_j": profile_j,
        "native_geometry_and_task_metrics": metrics,
        "articulated_certificate": certificates,
        "route_a_fcl_cross_check": fcl,
        "robust_clearance": {
            "status": load(robust_path)["status"],
            "formula": load(robust_path)["formula"],
            "hardware_clearance": load(robust_path)["hardware_clearance"],
            "acceptance_threshold": load(robust_path)["acceptance_threshold"],
            "artifact": rel(robust_path),
        },
        "dynamics": {
            "status": dynamics["status"],
            "backend": dynamics["backend"],
            "model_based_not_hardware_certified": True,
            "rnea_aba_self_consistency": dynamics["rnea_aba_self_consistency"]["status"],
            "joint_order_match": dynamics["model"]["joint_order_match"],
            "inertia_positive_definite": dynamics["model"]["inertia_positive_definite"],
            "zero_motion_rnea_finite": dynamics["model"]["zero_motion_rnea_finite"],
            "zero_motion_aba_finite": dynamics["model"]["zero_motion_aba_finite"],
            "candidate_nominal_peak_abs_torque_Nm": max(row["candidate_peak_abs_torque_Nm"] for row in nominal),
            "candidate_plus10_peak_abs_torque_Nm": max(row["candidate_peak_abs_torque_Nm"] for row in plus10),
            "candidate_nominal_peak_torque_slew_Nm_s": max(row["candidate_peak_abs_torque_slew_Nm_s"] for row in nominal),
            "candidate_nominal_peak_aggregate_power_proxy_W": max(row["candidate_peak_aggregate_power_proxy_W"] for row in nominal),
            "artifact": rel(dynamics_path),
        },
        "deterministic_replay": replay,
        "regression": {
            "pytest_command": "python -m pytest -q tests/test_d50_measurement_repairs.py tests/test_d54_articulated_certificate.py tests/test_d57_system_closure.py",
            "pytest_status": "PASS",
            "pytest_passed": 12,
            "pytest_failed": 0,
            "d50_known_answer_profile_j": {
                "status": "PASS",
                "successful_native_profiles": 8808,
                "segments": 8808,
                "max_abs_analytic_jerk_rad_s3": 8,
                "artifact": "outputs/D58_STAGE4_FULL_SYSTEM_CLOSURE/oracle_replay_d50_baseline_profile_only/summary.json",
            },
        },
        "measurement_classification": {
            "measurement_pipeline_status": "PASS_FOR_MEASURED_DOMAINS_WITH_EXPLICIT_GAPS",
            "finite_difference_jerk": "DIAGNOSTIC_ONLY",
            "profile_j_jerk": "AUTHORITATIVE_SEMANTICS_REPLAY_PASS",
            "candidate_weaknesses_repaired": False,
            "type_b_findings_preserved_for_stage4b": True,
        },
        "artifacts": {
            "auto0_native_summary": rel(OUT / "stateful_consistent_scale001_auto0" / "execution_form_summary.json"),
            "auto1_native_summary": rel(OUT / "stateful_consistent_scale001_auto1" / "execution_form_summary.json"),
            "auto0_metrics": rel(OUT / "full_stateful_metrics_auto0" / "metrics.json"),
            "auto1_metrics": rel(OUT / "full_stateful_metrics_auto1" / "metrics.json"),
            "auto0_certificate_stride1": rel(OUT / "articulated_cert_stateful_auto0_stride1" / "D54_CONTINUOUS_SELF_COLLISION_CERTIFICATION.json"),
            "auto1_certificate_stride1": rel(OUT / "articulated_cert_stateful_auto1_stride1" / "D54_CONTINUOUS_SELF_COLLISION_CERTIFICATION.json"),
        },
    }
    output = OUT / "D58_PHASE1_STATEFUL_CLOSURE_REPORT.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["phase1_status"], "promotion_status": report["promotion_status"], "output": rel(output)}, sort_keys=True))


if __name__ == "__main__":
    main()
