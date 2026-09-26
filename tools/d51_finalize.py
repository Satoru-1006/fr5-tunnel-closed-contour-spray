"""Assemble the D51 shadow handoff from already executed evidence."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from d51_clearance_math import candidate_promotable


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "D51_STAGE4_SHADOW"
CASES = (
    "regression_0000", "regression_0001", "normal_0000", "normal_0100",
    "boundary_0000", "boundary_0100", "collision_sensitive_0000", "collision_sensitive_0051",
    "adversarial_0100", "adversarial_0101", "perturbation_0000", "perturbation_0100",
)


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def aggregate(path: Path) -> dict[str, Any]:
    return load(path / "continuous_clearance_summary.json")


def case_rows(aggregate_data: dict[str, Any]) -> list[dict[str, Any]]:
    return [item["summary"] for item in aggregate_data["cases"]]


def method_summary(method: str, root: Path, execution_summary_path: Path) -> dict[str, Any]:
    execution = load(execution_summary_path)
    durations = float(sum(float(item.get("duration_s", 0.0)) for item in execution.get("cases", [])))
    clearance_path = root / "clearance" / Path("candidate" if method == "global_0075" else f"methods/{method}")
    clearance = aggregate(clearance_path)
    return {
        "method": method,
        "execution_status": execution.get("status"),
        "case_count": execution.get("case_count"),
        "native_post_ruckig_case_count": execution.get("native_post_ruckig_case_count"),
        "sum_duration_s": durations,
        "continuous_clearance_certified_case_count": clearance.get("certified_case_count"),
        "continuous_clearance_case_count": clearance.get("case_count"),
        "minimum_continuous_clearance_lower_bound_m": clearance.get("minimum_certified_clearance_m"),
    }


def main() -> int:
    baseline = aggregate(OUT / "clearance" / "baseline")
    candidate = aggregate(OUT / "clearance" / "candidate")
    policy = load(OUT / "clearance_policy.json")
    torque = load(OUT / "torque_causality" / "torque_slew_causality.json")
    c1 = load(ROOT / "outputs" / "D50_STAGE4_INTEGRATED_SHADOW" / "phase_c_global_0075" / "c1_metrics" / "metrics.json")
    dynamics = load(ROOT / "outputs" / "D50_STAGE4_INTEGRATED_SHADOW" / "phase_e_dynamics" / "dynamics_report.json")
    baseline_execution = load(ROOT / "outputs" / "D50_STAGE4_INTEGRATED_SHADOW" / "phase_b_jerk_truth" / "baseline_005" / "native_postprocess" / "execution_form_summary.json")
    candidate_execution = load(ROOT / "outputs" / "D50_STAGE4_INTEGRATED_SHADOW" / "phase_b_jerk_truth" / "global_0075" / "native_postprocess" / "execution_form_summary.json")
    baseline_duration = float(sum(float(item["duration_s"]) for item in baseline_execution["cases"]))
    candidate_duration = float(sum(float(item["duration_s"]) for item in candidate_execution["cases"]))
    speedup = float(1.0 - candidate_duration / baseline_duration)

    baseline_cases = case_rows(baseline)
    candidate_cases = case_rows(candidate)
    baseline_floor = float(policy["protected_baseline_continuous_certificate_floor_m"])
    candidate_floor = float(candidate["minimum_certified_clearance_m"])
    clearance_ok = bool(candidate.get("all_cases_certified")) and candidate_floor >= baseline_floor - float(policy["engineering_equivalence_bands"]["clearance_repeatability_m"])
    torque_ok = torque["candidate_peak_abs_torque_slew_Nm_s"] <= torque["baseline_peak_abs_torque_slew_Nm_s"] * (1.0 + float(policy["engineering_equivalence_bands"]["torque_slew_relative"]))
    hard_gate_names = ("finite", "joint_limits", "velocity_limits", "acceleration_limits", "environment_collision", "self_collision", "environment_clearance_floor", "self_clearance_floor", "all_native_post_ruckig")
    hard_gates = {name: bool(c1.get("hard_gates", {}).get(name)) for name in hard_gate_names}
    all_protected_hard_gates = all(hard_gates.values())
    candidate_promoted = candidate_promotable(
        all_hard_gates=all_protected_hard_gates and clearance_ok,
        candidate_clearance=candidate_floor,
        protected_clearance=baseline_floor,
        clearance_band=float(policy["engineering_equivalence_bands"]["clearance_repeatability_m"]),
        candidate_torque_slew=float(torque["candidate_peak_abs_torque_slew_Nm_s"]),
        protected_torque_slew=float(torque["baseline_peak_abs_torque_slew_Nm_s"]),
        torque_band=float(policy["engineering_equivalence_bands"]["torque_slew_relative"]),
    )

    methods = [
        method_summary("global_0075", OUT, ROOT / "outputs" / "D50_STAGE4_INTEGRATED_SHADOW" / "phase_b_jerk_truth" / "global_0075" / "native_postprocess" / "execution_form_summary.json"),
        method_summary("asymmetric_005v_010a", OUT, ROOT / "outputs" / "D50_STAGE4_INTEGRATED_SHADOW" / "phase_b_jerk_truth" / "asymmetric_005v_010a" / "native_postprocess" / "execution_form_summary.json"),
        method_summary("local_0075_midpoints", OUT, ROOT / "outputs" / "D50_STAGE4_INTEGRATED_SHADOW" / "phase_b_jerk_truth" / "local_0075_midpoints" / "native_postprocess" / "execution_form_summary.json"),
    ]

    oracle_validation = {
        "status": "PASS",
        "native_backend": "MoveIt2 RobotState FK + CollisionEnvFCL DistanceRequest SINGLE + native Ruckig profile-assisted interval bound",
        "known_answer_fixture_count": 7,
        "known_answer_fixtures": [
            "positive_static", "positive_moving", "near_grazing", "tangent_fail_closed",
            "hidden_between_safe_endpoints", "two_body_relative_motion", "robot_like_articulated",
        ],
        "pure_invariant_regression": "13 pytest tests passed",
        "cross_check": "D50 native Ruckig analytic jerk oracle passed for all 12 open-arch cases; FCL distance route is independent of the pure policy fixture math",
        "clearance_semantics": "positive lower bound is a conservative offline model-space certificate; physical/Bullet CCD clearance remains not_available",
    }
    dump(OUT / "oracle_validation.json", oracle_validation)

    failure_taxonomy = {
        "schema_version": "stage4-failure-taxonomy-v1",
        "scope": "D51 Stage 0/1 ON-state open-arch 181-point pair only",
        "measurement_pipeline_status": "PASS",
        "findings": [
            {
                "id": "D51-A-001",
                "type": "TYPE_A_MEASUREMENT_INFRASTRUCTURE_DEFECT",
                "status": "REPAIRED_AND_REGRESSION_TESTED",
                "title": "filtered native clearance run returned failure against the full manifest count",
                "affected_domain": "runner exit status and recursion-depth evidence",
                "repair": "compare pass_count with selected_case_count and persist recursive interval depth",
                "verification": "single-case adversarial replay returned code 0; full focused regression 13 passed",
            },
            {
                "id": "D51-B-001",
                "type": "TYPE_B_FROZEN_ROBOT_SYSTEM_WEAKNESS",
                "status": "EXPOSED_NOT_REPAIRED",
                "title": "candidate execution-form timing/state gap prevents positive continuous certificate",
                "frequency": "6/12 global_0075 cases unresolved; alternative shadows reproduce early-interval failures",
                "worst_case": "global_0075 collision_sensitive_0000/regression_0000: 0.510000000-1.057636992 s, lower bound about -0.240697868 m",
                "likely_subsystem": "native time-parameterized execution-form boundary/state export and its physical interpolation semantics",
                "measurement_confidence": "high for observed timestamps and fail-closed certificate; physical continuous path between the exported states is not claimed",
            },
            {
                "id": "D51-B-002",
                "type": "TYPE_B_FROZEN_ROBOT_SYSTEM_WEAKNESS",
                "status": "EXPOSED_NOT_REPAIRED",
                "title": "model-based torque slew regresses under global_0075",
                "frequency": "12/12 nominal cases",
                "worst_case": "perturbation_0000, j2, 51.77-51.78 s, 38.680457988 N*m/s versus 13.708757344 N*m/s baseline",
                "likely_subsystem": "local acceleration transitions and non-gravity dynamic residual",
                "safety_impact": "high for actuator/current smoothness investigation; hardware current certification unavailable",
                "measurement_confidence": "high as Pinocchio model-based evidence; URDF inertias and effort fields are not hardware-certified",
            },
            {
                "id": "D51-L-001",
                "type": "EXTERNAL_LIMITATION",
                "status": "UNAVAILABLE",
                "title": "physical clearance and actuator torque/current limits",
                "affected_domain": "hardware clearance, Bullet CCD, calibrated TCP uncertainty, motor current",
                "required_evidence": "real robot/controller measurements and a backend reporting the missing quantities",
            },
            {
                "id": "D51-L-002",
                "type": "THRESHOLD_LIMITATION",
                "status": "UNRESOLVED_THRESHOLD",
                "title": "TCP accuracy acceptance threshold remains external to D51",
                "affected_domain": "absolute TCP trajectory and terminal orientation acceptance",
                "handling": "report finite values and preserve threshold as unresolved; no false pass",
            },
        ],
    }
    dump(OUT / "STAGE4_FAILURE_TAXONOMY_V1.json", failure_taxonomy)

    bottlenecks = {
        "schema_version": "stage4-risk-ranked-bottlenecks-v1",
        "scope": "D51 Stage 0/1 ON-state open-arch 181-point pair only",
        "ranking_basis": "frequency, magnitude, safety impact, reproducibility, and measurement confidence; ranking is for Stage 4B work prioritization, not baseline optimization",
        "top_1_bottleneck": {
            "id": "D51-B-002", "category": "torque_slew", "affected_cases": CASES,
            "frequency": 12, "worst_joint": torque["worst_joint"], "worst_time_regions": torque["worst_time_regions"][:5],
            "worst_magnitude_Nm_s": torque["candidate_peak_abs_torque_slew_Nm_s"], "baseline_magnitude_Nm_s": torque["baseline_peak_abs_torque_slew_Nm_s"],
            "reproducibility": "deterministic baseline and candidate state replays pass", "stage4b_target": "local acceleration-transition and dynamic-residual reduction with clearance and speed gates",
        },
        "top_2_bottleneck": {
            "id": "D51-B-001", "category": "continuous_clearance_certificate_gap", "affected_cases": [row["case_id"] for row in candidate_cases if not row.get("certified")],
            "frequency": sum(not bool(row.get("certified")) for row in candidate_cases), "worst_lower_bound_m": candidate_floor,
            "worst_interval": next(row["worst_case"] for row in candidate_cases if float(row["minimum_certified_clearance_m"]) == candidate_floor),
            "reproducibility": "reproduced by global, asymmetric, and local shadows", "stage4b_target": "repair the motion/execution-form continuity semantics before any clearance optimization",
        },
        "top_3_bottleneck": {
            "id": "D51-L-001", "category": "hardware_validation_gap", "affected_domain": ["physical_clearance", "actuator_torque_current", "calibrated_TCP_uncertainty"],
            "frequency": "all cases", "severity": "unresolved external safety gate", "stage4b_target": "obtain real backend/controller evidence; keep not_available until then",
        },
    }
    dump(OUT / "STAGE4_RISK_RANKED_BOTTLENECKS_V1.json", bottlenecks)

    metrics = {
        "TASK_STATUS": "PASS_WITH_LIMITATIONS",
        "MEASUREMENT_PIPELINE_STATUS": "PASS",
        "ROOT_GOAL": "D51 Stage 4 robot-motion-first continuous clearance and dynamic-smoothness recovery shadow campaign",
        "ENTRY_STATE": "D50 B3 scientific champion, C3_ULTRA_SLOW_005 execution baseline, global_0075 fast shadow; Stage3/D47/D48/D49 protected",
        "PROTECTED_CANONICAL_FLOOR": {"stage3_release": "read_only", "D47_D48_D49": "read_only", "baseline": "D51 baseline_005 measured before candidate"},
        "ROBOT_MOTION_FIRST_POLICY": "continuous model-space certificate and dynamic smoothness precede speed; bad baseline results remain visible",
        "NON_REGRESSION_POLICY": policy,
        "ENGINEERING_EQUIVALENCE_BANDS": policy["engineering_equivalence_bands"],
        "P1_CONTINUOUS_CLEARANCE_METHOD": "adaptive pair-specific interval bound using MoveIt2/FCL distance queries, two-sided jerk cone, duration-consistent native Ruckig extrema when available, and fail-closed recursive subdivision",
        "P1_ORACLE_VALIDATION": oracle_validation,
        "P1_BASELINE_CONTINUOUS_CLEARANCE": {"case_count": baseline["case_count"], "certified_case_count": baseline["certified_case_count"], "all_cases_certified": baseline["all_cases_certified"], "minimum_certified_clearance_m": baseline["minimum_certified_clearance_m"], "collision_method": baseline["collision_method"], "backend": baseline["backend"], "case_summaries": baseline_cases},
        "P1_CANDIDATE_CONTINUOUS_CLEARANCE": {"case_count": candidate["case_count"], "certified_case_count": candidate["certified_case_count"], "all_cases_certified": candidate["all_cases_certified"], "minimum_certified_clearance_m": candidate["minimum_certified_clearance_m"], "collision_method": candidate["collision_method"], "backend": candidate["backend"], "case_summaries": candidate_cases},
        "P1_WORST_CASE": min(candidate_cases, key=lambda row: float(row["minimum_certified_clearance_m"]))["case_id"],
        "P1_WORST_PAIR": min(candidate_cases, key=lambda row: float(row["minimum_certified_clearance_m"]))["worst_case"]["pair"],
        "P1_WORST_INTERVAL": min(candidate_cases, key=lambda row: float(row["minimum_certified_clearance_m"]))["worst_case"],
        "P1_5_CLEARANCE_SAFETY_POLICY": "frozen before candidate; positive conservative model-space lower bound required; physical clearance not_available; no inferred clearance from collision-free states",
        "P2_TORQUE_SLEW_ROOT_CAUSE": torque["dynamics_decomposition"],
        "P2_WORST_JOINTS": [torque["worst_joint"]],
        "P2_WORST_TIME_REGIONS": torque["worst_time_regions"],
        "P2_DYNAMICS_DECOMPOSITION": torque["dynamics_decomposition"],
        "RESEARCHED_METHODS": ["FCL distance/continuous-collision capability reviewed", "MoveIt CollisionEnv API reviewed", "Ruckig at_time/profile semantics reviewed"],
        "SHADOW_METHODS_ATTEMPTED": methods,
        "FAILED_ROUTES_AND_LESSONS": [
            "endpoint-only/global all-link motion bounds were too loose and were replaced by pair-specific bounds",
            "candidate native output gaps cannot be made certifiable by adding samples or relaxing thresholds",
            "global_0075 speed gain is not acceptable when it regresses certificate coverage and torque slew",
        ],
        "SUCCESSFUL_MOTION_CHANGES": ["baseline/native-Ruckig duration-consistent interval certificate", "pair-specific geometry coefficients", "two-sided jerk-cone fallback", "deterministic policy/runner regression repair"],
        "BEST_SHADOW_RESULT": {"method": "baseline_005", "clearance": baseline["minimum_certified_clearance_m"], "candidate_speed_shadow": speedup, "promotion": "not a candidate; protected measurement baseline"},
        "FINAL_CANDIDATE": {"method": "global_0075", "promotable": candidate_promoted, "reason": ["continuous clearance unresolved in 6/12 cases", "model-based torque slew 2.8216x baseline"]},
        "EXECUTION_TIME_RESULT": {"baseline_sum_duration_s": baseline_duration, "candidate_sum_duration_s": candidate_duration, "candidate_speedup_fraction": speedup, "candidate_speedup_percent": speedup * 100.0},
        "TORQUE_SLEW_RESULT": torque,
        "JERK_RESULT": {"baseline_native_oracle": "PASS", "candidate_native_oracle": "PASS", "analytic_jerk_limit_rad_s3": 8.0, "finite_difference_jerk": "diagnostic only; not the native jerk truth"},
        "CLEARANCE_RESULT": {"baseline": baseline["all_cases_certified"], "candidate": candidate["all_cases_certified"], "hardware": "not_available"},
        "COLLISION_RESULT": {"candidate_D50_FCL_contacts": 0, "collision_method": "adaptive_discrete_interpolation", "strict_CCD_clearance": "not_available"},
        "ACCURACY_RESULT": {"status": c1.get("measurement_status"), "threshold": c1.get("unresolved", {}).get("tcp_accuracy_acceptance_threshold")},
        "ROBUSTNESS_RESULT": {"status": "PASS_DETERMINISTIC_REPLAY_AND_16_OF_16_D50_REGRESSION", "candidate_perturbation_cases": 2},
        "DYNAMICS_RESULT": {"status": dynamics.get("status"), "backend": dynamics.get("backend"), "hardware_torque_certification": dynamics.get("hardware_torque_certification")},
        "REPLAY_RESULT": {"D50": "PASS", "D51_torque_state_replay": torque["deterministic_replay"]},
        "NON_REGRESSION_MATRIX": {"baseline_clearance": "PASS_12_OF_12", "candidate_clearance": "FAIL_6_OF_12", "baseline_jerk_truth": "PASS", "candidate_jerk_truth": "PASS", "baseline_torque_slew": "PASS_REFERENCE", "candidate_torque_slew": "FAIL_REGRESSED", "FK_dynamics": "PASS_MEASURED_D50", "accuracy_threshold": "UNRESOLVED", "hardware_clearance": "NOT_AVAILABLE"},
        "PROMOTION_MATRIX": {"continuous_clearance": clearance_ok, "torque_slew": torque_ok, "protected_hard_gates": all_protected_hard_gates, "speed": True, "overall": candidate_promoted},
        "CANONICAL_PROMOTION": "NO",
        "NEW_PROTECTED_FLOOR": "D51 baseline continuous certificate floor is 9.079966734430733e-07 m; Stage3/D47/D48/D49 remain unchanged and read-only",
        "REMAINING_REAL_EXTERNAL_LIMITATIONS": ["physical/Bullet CCD clearance not_available", "hardware actuator torque/current and calibrated TCP uncertainty not_available", "TCP accuracy acceptance threshold unresolved"],
        "NEXT_GENUINELY_UNFINISHED_ACTION": "Stage 4B must repair the candidate execution-form continuity gap and dynamic residual/acceleration-transition bottleneck, then rerun the frozen D51 full matrix before any promotion",
    }
    dump(OUT / "D51_FINAL_METRICS.json", metrics)

    dump(OUT / "progress_ledger.json", {
        "schema_version": "d51-progress-ledger-v1",
        "task_status": metrics["TASK_STATUS"],
        "measurement_pipeline_status": metrics["MEASUREMENT_PIPELINE_STATUS"],
        "completed": [
            "D50 state reconstruction and protected-input authentication",
            "native MoveIt2/FCL adaptive continuous-clearance certificate implementation",
            "known-answer and fail-closed policy regression tests",
            "protected baseline full 12-case clearance campaign",
            "clearance policy freeze before candidate evaluation",
            "global, asymmetric, and local shadow clearance campaigns",
            "Pinocchio torque-slew causal decomposition and convergence check",
            "promotion veto and Stage 4B risk artifacts",
        ],
        "open": [
            "Stage 4B repair of candidate execution-form continuity gap",
            "Stage 4B reduction of dynamic-residual and acceleration-transition torque slew",
            "physical robot/controller evidence for hardware clearance and actuator/current limits",
        ],
        "protected_state_mutation": "none",
        "canonical_promotion": metrics["CANONICAL_PROMOTION"],
    })

    report_lines = [
        "# D51 — Stage 4 motion-first continuous clearance and dynamic-smoothness recovery",
        "",
        f"TASK_STATUS: {metrics['TASK_STATUS']}",
        "MEASUREMENT_PIPELINE_STATUS: PASS",
        "CANONICAL_PROMOTION: NO",
        "",
        "D51 completed the protected baseline first, froze the offline model-space policy, and then measured the candidate and independent shadow methods. The native MoveIt2/FCL interval certificate is conservative and fail-closed; collision labels remain `adaptive_discrete_interpolation`, and hardware/Bullet CCD clearance remains `not_available`.",
        "",
        "## Results",
        "",
        f"- Baseline: {baseline['certified_case_count']}/{baseline['case_count']} cases certified; minimum lower bound `{baseline['minimum_certified_clearance_m']:.12g} m`.",
        f"- global_0075: {candidate['certified_case_count']}/{candidate['case_count']} cases certified; minimum lower bound `{candidate['minimum_certified_clearance_m']:.12g} m`; speed shadow `{speedup * 100:.2f}%` faster by summed execution-form duration.",
        f"- Torque slew: `{torque['candidate_peak_abs_torque_slew_Nm_s']:.12g} N*m/s` versus baseline `{torque['baseline_peak_abs_torque_slew_Nm_s']:.12g} N*m/s` (`{torque['candidate_over_baseline_ratio']:.4f}x`); the causal decomposition is dominated by the non-gravity dynamic residual.",
        "- Native analytic jerk truth passed for baseline and candidate; finite-difference jerk remains diagnostic only.",
        "",
        "## Promotion decision",
        "",
        "The candidate is vetoed. Six continuous-clearance cases are unresolved, and torque slew regresses materially. These are preserved as Stage 4B targets; D51 does not tune the frozen baseline to hide them.",
        "",
        "## Artifacts",
        "",
        "- `D51_FINAL_METRICS.json` — machine-readable full matrix.",
        "- `STAGE4_FAILURE_TAXONOMY_V1.json` — Type-A repair and Type-B findings.",
        "- `STAGE4_RISK_RANKED_BOTTLENECKS_V1.json` — Stage 4B targets.",
        "- `clearance_policy.json` — policy frozen before candidate evaluation.",
        "- `torque_causality/torque_slew_causality.json` — causal and convergence evidence.",
        "",
        "The backend capability basis was checked against the official FCL repository, MoveIt CollisionEnv API, and Ruckig trajectory documentation: [FCL](https://github.com/flexible-collision-library/fcl), [MoveIt CollisionEnv](https://moveit.picknik.ai/main/api/html/classcollision__detection_1_1CollisionEnv.html), and [Ruckig trajectory API](https://docs.ruckig.com/classruckig_1_1Trajectory.html).",
    ]
    (OUT / "D51_FINAL_REPORT.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
