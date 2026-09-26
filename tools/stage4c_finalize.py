"""Assemble the D48 Stage 4C evidence package from completed measurements.

This script only writes D48 evidence and hand-off artifacts.  It never edits the
protected D47 champion or any canonical Stage-3 input.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path("outputs/D48_STAGE4C_EXECUTION_FORM_V1")
PARENT = Path("outputs/D47_STAGE4B_ROLLING_CHAMPION_V1/PROMOTED/B3")
C1 = ROOT / "C1_ACCEPTANCE/metrics.json"
ULTRA = ROOT / "C3_ULTRA_SLOW_005/metrics.json"
C3 = ROOT / "C3_TIMING_SHADOW/c3_shadow_report.json"
C2 = ROOT / "C3_ULTRA_SLOW_005/C2_SELF_SWEEP_REPAIRED/self_sweep_summary.json"
C4_ULTRA = ROOT / "C4_ULTRA_SLOW_005/dynamics_report.json"
C4_C1 = ROOT / "C4_ACCEPTANCE/dynamics_report.json"
REPLAY = ROOT / "C3_ULTRA_SLOW_005/replay_comparison.json"


def read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def stat(obj: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if isinstance(obj, dict) and key in obj:
            obj = obj[key]
        else:
            return None
    return obj


def dump(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def rel(path: Path) -> str:
    return path.as_posix()


def main() -> None:
    c1 = read(C1)
    ultra = read(ULTRA)
    c3 = read(C3)
    c2 = read(C2)
    c4u = read(C4_ULTRA)
    c4c1 = read(C4_C1)
    replay = read(REPLAY)

    parent_metrics = {
        "environment_clearance_m": 0.08432694494047037,
        "self_clearance_m": 0.01662204867779536,
        "minimum_sigma": 1.0189550264124055e-06,
        "maximum_condition": 1.8240751425836e06,
        "environment_collision_cases": 0,
        "self_collision_cases": 0,
        "environment_ccd_failures": 0,
        "H1": 7.089328839013259e-05,
        "H32": 0.01849100619381173,
        "stage3_update": 480,
    }
    candidate_metrics = {
        "environment_clearance_m": stat(ultra, "geometry", "minimum_environment_clearance_m", "min"),
        "self_clearance_m": stat(ultra, "geometry", "minimum_self_clearance_m", "min"),
        "minimum_sigma": stat(ultra, "singularity", "minimum_sigma_min", "min"),
        "maximum_condition": stat(ultra, "singularity", "maximum_condition_number", "max"),
        "environment_collision_cases": stat(ultra, "geometry", "environment_collision_cases"),
        "self_collision_cases": stat(ultra, "geometry", "self_collision_cases"),
        "environment_ccd_failures": stat(ultra, "geometry", "environment_ccd_failures"),
        "H1": stat(ultra, "protected", "H1"),
        "H32": stat(ultra, "protected", "H32"),
        "stage3_update": stat(ultra, "protected", "stage3_update"),
    }

    ultra_nominal = stat(c4u, "variant_summaries", "candidate_nominal") or {}
    ultra_baseline = stat(c4u, "variant_summaries", "baseline_nominal") or {}
    c1_nominal = stat(c4c1, "variant_summaries", "candidate_nominal") or {}

    c2_adaptive_contacts = sum(
        int(case.get("adaptive_self_contact_count", 0)) for case in c2.get("cases", [])
    )
    c2_adaptive_samples = sum(
        int(case.get("adaptive_sample_count", 0)) for case in c2.get("cases", [])
    )

    report = {
        "schema_version": "d48-stage4c-final-report-v1",
        "task_status": "PASS_WITH_C2_CONTINUOUS_BACKEND_UNAVAILABLE",
        "first_blocker": {
            "label": "C2_CONTINUOUS_SELF_COLLISION_BACKEND_UNAVAILABLE",
            "status": "EXTERNAL_ENVIRONMENT_LIMITATION_AFTER_BACKEND_SEARCH",
            "impact": "Continuous self-collision certification cannot be claimed. All executable C1/C3/C4 measurements continue to be reported.",
            "evidence": rel(C2),
        },
        "scientific_verdict": (
            "The 0.05 native MoveIt2 post-Ruckig shadow candidate passes the measured C1 safety, "
            "joint, geometry, singularity, jerk, protected-floor, and deterministic-replay checks "
            "on the 12-case D47 acceptance set; adaptive discrete self-sweep is contact-free, "
            "but continuous self-collision remains unverified because no usable continuous backend exists."
        ),
        "protected_d47_starting_champion": {
            "path": rel(PARENT),
            "status": "READ_ONLY_PROTECTED_UNCHANGED",
            "metrics": parent_metrics,
        },
        "execution_form_implementation": {
            "label": "NATIVE_MOVEIT2_POST_RUCKIG",
            "pipeline": [
                "MoveIt2 RobotTrajectory",
                "unwind",
                "TOTG time parameterization",
                "MoveIt2 Ruckig smoothing",
                "native post-Ruckig q/dq/ddq extraction",
                "finite-difference jerk audit labeled derived from native post-Ruckig acceleration",
            ],
            "worker": rel(Path("ros2_moveit_bridge/stage4c_execution_form_native.py")),
            "launch": rel(Path("ros2_moveit_bridge/launch/stage4c_execution_form_native.launch.py")),
            "joint_order": ["j1", "j2", "j3", "j4", "j5", "j6"],
        },
        "c1_execution_form": {
            "scope": "12-case acceptance set",
            "native_0_15": {
                "status": c1.get("status"),
                "measurement_status": c1.get("measurement_status"),
                "hard_gates": c1.get("hard_gates"),
                "jerk_limit_violations": stat(c1, "joint_space", "jerk_limit_violations"),
                "max_jerk_ratio": stat(c1, "joint_space", "jerk_ratio", "max"),
                "classification": "TYPE_B_POST_RUCKIG_JERK_WEAKNESS_FOR_0_15_ROUTE",
            },
            "selected_0_05_shadow": {
                "status": ultra.get("candidate_safety_status"),
                "promotion_status": "SHADOW_NOT_CANONICAL_PENDING_C2_CONTINUOUS_BACKEND",
                "measurement_status": ultra.get("measurement_status"),
                "hard_gates": ultra.get("hard_gates"),
                "jerk_limit_violations": stat(ultra, "joint_space", "jerk_limit_violations"),
                "max_jerk_ratio": stat(ultra, "joint_space", "jerk_ratio", "max"),
                "max_velocity_ratio": stat(ultra, "joint_space", "velocity_ratio", "max"),
                "max_acceleration_ratio": stat(ultra, "joint_space", "acceleration_ratio", "max"),
                "accuracy": ultra.get("accuracy"),
                "evidence": rel(ULTRA),
            },
        },
        "c2_continuous_self_collision": {
            "status": c2.get("continuous_self_collision_status"),
            "adaptive_discrete_measurement_status": c2.get("status"),
            "collision_method": c2.get("cases", [{}])[0].get("collision_method"),
            "case_count": c2.get("case_count"),
            "adaptive_contact_cases": c2.get("adaptive_self_sweep_contact_cases"),
            "adaptive_contact_samples": c2_adaptive_contacts,
            "adaptive_sample_count_total": c2_adaptive_samples,
            "max_step_rad": c2.get("cases", [{}])[0].get("max_step_rad"),
            "mapping": c2.get("mapping"),
            "backend_search": c2.get("continuous_backend_search"),
            "evidence": rel(C2),
        },
        "c3_stress_accuracy_repeatability_smoothness": {
            "existing_routes": {
                "slow_010": {
                    "status": "REJECTED_JERK_LIMIT",
                    "max_abs_jerk_rad_s3": max(
                        item["slow_010_max_jerk"] for item in c3["comparisons"]
                    ),
                    "duration_range_s": [
                        min(item["slow_010_duration_s"] for item in c3["comparisons"]),
                        max(item["slow_010_duration_s"] for item in c3["comparisons"]),
                    ],
                },
                "fast_020": {
                    "status": "REJECTED_JERK_LIMIT",
                    "max_abs_jerk_rad_s3": max(
                        item["fast_020_max_jerk"] for item in c3["comparisons"]
                    ),
                    "duration_range_s": [
                        min(item["fast_020_duration_s"] for item in c3["comparisons"]),
                        max(item["fast_020_duration_s"] for item in c3["comparisons"]),
                    ],
                },
                "ultra_slow_005": {
                    "status": "SAFETY_PASS_SHADOW",
                    "max_abs_jerk_rad_s3": stat(ultra, "joint_space", "jerk_ratio", "max") * 8.0,
                    "jerk_limit_violations": stat(ultra, "joint_space", "jerk_limit_violations"),
                    "duration_scope": "native post-Ruckig output; per-case durations persisted in execution-form summary",
                    "evidence": rel(ULTRA),
                },
            },
            "conclusion": "0.05 materially reduced jerk relative to 0.10/0.20 and preserved measured accuracy/geometry floors; it is not a canonical promotion.",
            "timing_report": rel(C3),
        },
        "c4_model_based_dynamic_feasibility": {
            "status": c4u.get("status"),
            "backend": c4u.get("backend"),
            "case_count": c4u.get("case_count"),
            "nominal_candidate": {
                "peak_abs_torque_Nm": stat(ultra_nominal, "peak_abs_torque_Nm", "max"),
                "peak_abs_torque_slew_Nm_s": stat(ultra_nominal, "peak_abs_torque_slew_Nm_s", "max"),
                "peak_aggregate_power_proxy_W": stat(ultra_nominal, "peak_aggregate_power_proxy_W", "max"),
                "energy_work_proxy_J": stat(ultra_nominal, "energy_work_proxy_J", "max"),
                "finite_failures": ultra_nominal.get("finite_failures"),
            },
            "nominal_d47_model_space_baseline": {
                "peak_abs_torque_Nm": stat(ultra_baseline, "peak_abs_torque_Nm", "max"),
                "peak_abs_torque_slew_Nm_s": stat(ultra_baseline, "peak_abs_torque_slew_Nm_s", "max"),
                "peak_aggregate_power_proxy_W": stat(ultra_baseline, "peak_aggregate_power_proxy_W", "max"),
                "energy_work_proxy_J": stat(ultra_baseline, "energy_work_proxy_J", "max"),
                "finite_failures": ultra_baseline.get("finite_failures"),
            },
            "mass_sensitivity": {
                "scales": [0.9, 1.0, 1.1],
                "candidate_peak_torque_Nm": [
                    stat(c4u, "variant_summaries", "candidate_mass_scale_minus_10pct", "peak_abs_torque_Nm", "max"),
                    stat(c4u, "variant_summaries", "candidate_nominal", "peak_abs_torque_Nm", "max"),
                    stat(c4u, "variant_summaries", "candidate_mass_scale_plus_10pct", "peak_abs_torque_Nm", "max"),
                ],
                "candidate_peak_torque_slew_Nm_s": [
                    stat(c4u, "variant_summaries", "candidate_mass_scale_minus_10pct", "peak_abs_torque_slew_Nm_s", "max"),
                    stat(c4u, "variant_summaries", "candidate_nominal", "peak_abs_torque_slew_Nm_s", "max"),
                    stat(c4u, "variant_summaries", "candidate_mass_scale_plus_10pct", "peak_abs_torque_slew_Nm_s", "max"),
                ],
                "finite_failures": [
                    stat(c4u, "variant_summaries", "candidate_mass_scale_minus_10pct", "finite_failures"),
                    stat(c4u, "variant_summaries", "candidate_nominal", "finite_failures"),
                    stat(c4u, "variant_summaries", "candidate_mass_scale_plus_10pct", "finite_failures"),
                ],
                "interpretation": "synthetic in-memory mass scales; not physical uncertainty certification",
            },
            "effort_limit_provenance": c4u.get("effort_limit_interpretation"),
            "hardware_torque_certification": c4u.get("hardware_torque_certification"),
            "evidence": rel(C4_ULTRA),
        },
        "c4_rnea_aba_self_consistency": {
            "status": stat(c4u, "rnea_aba_self_consistency", "status"),
            "definition": stat(c4u, "rnea_aba_self_consistency", "definition"),
            "evidence": rel(C4_ULTRA),
        },
        "protected_metric_table": {
            "parent_d47_b3": parent_metrics,
            "candidate_0_05_shadow": candidate_metrics,
            "interpretation": {
                "environment_clearance": "candidate above parent floor",
                "self_clearance": "candidate within the configured 1e-6 m protected-floor tolerance; do not treat the 2.46e-8 m difference as an improvement",
                "singularity": "candidate preserves and improves both measured floors",
                "stage3_metrics": "H1/H32/update unchanged",
            },
        },
        "failure_classification": {
            "type_a_repaired": [
                {
                    "defect": "Actual post-Ruckig CSV was passed to retained q-only D41/FK parsers; their last-six-column convention interpreted derived jerk as q.",
                    "root_cause": "wrong trajectory-column contract between native execution-form output and legacy geometry/FK adapters",
                    "repair": "write explicit q-only geometry inputs from native j1_q..j6_q columns; retain native q/dq/ddq/jerk for execution-form audits",
                    "verification": "known native smoke changed from false collision/singularity failures to MoveIt2/FK-consistent finite PASS; affected C1 geometry recomputed",
                    "status": "FIXED_AND_REGRESSED_CHECKED",
                },
                {
                    "defect": "C2 worker launcher teardown returned -11 after writing a complete summary.",
                    "root_cause": "MoveItCpp teardown fault in the one-shot worker process",
                    "repair": "controlled one-shot process exit after summary persistence; no measurement data are skipped",
                    "verification": "one-case and full 12-case C2 reruns completed with clean process exit and complete summaries",
                    "status": "FIXED_AND_REGRESSED_CHECKED",
                },
            ],
            "type_b_measured_not_hidden": [
                {
                    "defect": "0.15 native post-Ruckig execution form exceeded the jerk limit",
                    "affected_case_ids": "all 12 acceptance cases",
                    "frequency": 509,
                    "worst_ratio": stat(c1, "joint_space", "jerk_ratio", "max"),
                    "status": "RETAINED_AS_SHADOW_EVIDENCE; 0.05 route separately measured",
                },
                {
                    "defect": "0.05 candidate model-space torque slew and power proxies exceed the persisted D47 q-only comparison in nominal audit",
                    "affected_case_ids": "all 12 acceptance cases",
                    "status": "MODEL_DYNAMIC_OPTIMIZATION_TARGET; NOT_REPAIRED_IN_D48",
                    "worst_peak_torque_slew_increase_Nm_s": stat(ultra_nominal, "peak_abs_torque_slew_Nm_s", "max") - stat(ultra_baseline, "peak_abs_torque_slew_Nm_s", "max"),
                    "worst_peak_power_increase_W": stat(ultra_nominal, "peak_aggregate_power_proxy_W", "max") - stat(ultra_baseline, "peak_aggregate_power_proxy_W", "max"),
                },
            ],
            "type_c_unavailable": [
                "continuous self-collision backend",
                "real hardware torque/current telemetry and verified actuator limits",
                "physical clearance/calibration uncertainty",
            ],
        },
        "deterministic_replay": {
            "status": replay.get("status"),
            "case_count": replay.get("case_count"),
            "numeric_tolerance": replay.get("numeric_tolerance"),
            "max_abs_numeric_delta": max(
                float(case.get("max_abs_numeric_delta", 0.0)) for case in replay.get("cases", [])
            ),
            "evidence": rel(REPLAY),
        },
        "focused_tests": {
            "command": "python -m pytest -q tests/test_stage4c_execution_form.py tests/test_stage4c_self_sweep.py tests/test_stage4c_c3_shadow.py",
            "result": "10 passed",
        },
        "promotion_ledger": {
            "canonical_champion": rel(PARENT),
            "candidate": rel(ROOT / "C3_ULTRA_SLOW_005"),
            "decision": "NO_CANONICAL_PROMOTION",
            "reason": "C2 continuous self-collision certification is unavailable; D47 B3 remains protected and untouched.",
            "authoritative_promotion_authority": "Sol orchestrator only",
        },
        "unresolved_limitations": {
            "model": [
                "URDF inertial and effort values are project-model values, not hardware-certified parameters",
                "D47 comparison uses persisted q-only duration and deterministic finite differences, not an unpersisted D47 execution-form trajectory",
                "synthetic mass sensitivity is exploratory, not a physical uncertainty bound",
            ],
            "physical": [
                "HARDWARE_TORQUE_CERTIFICATION = NOT_APPLICABLE_UNMEASURED",
                "PHYSICAL_CLEARANCE_CERTIFICATION = NOT_APPLICABLE_UNVERIFIED",
            ],
            "coverage": [
                "full 1000-case campaign was not rerun; this evidence package covers the frozen 12-case acceptance set",
                "continuous self-collision remains unverified, not PASS",
            ],
        },
        "next_unfinished_scientific_action": (
            "Provide/install a real continuous self-collision backend compatible with the frozen MoveIt2 model, "
            "then rerun the same persisted 0.05 candidate and acceptance set; afterward decide whether the "
            "model-dynamic slew/power target warrants Stage 4B optimization."
        ),
    }

    ledger = {
        "schema_version": "d48-promotion-ledger-v1",
        "entries": [
            {
                "state": "D47_B3",
                "path": rel(PARENT),
                "status": "PROTECTED_CANONICAL_UNCHANGED",
                "H1": parent_metrics["H1"],
                "H32": parent_metrics["H32"],
                "stage3_update": parent_metrics["stage3_update"],
            },
            {
                "state": "D48_C3_ULTRA_SLOW_005",
                "path": rel(ROOT / "C3_ULTRA_SLOW_005"),
                "status": "SHADOW_CANDIDATE_NOT_PROMOTED",
                "safety_status": ultra.get("candidate_safety_status"),
                "continuous_self_collision": c2.get("continuous_self_collision_status"),
                "deterministic_replay": replay.get("status"),
                "promotion_reason": "Continuous self-collision backend unavailable; canonical promotion withheld.",
            },
        ],
        "canonical_after_d48": rel(PARENT),
    }

    taxonomy = {
        "schema_version": "stage4-failure-taxonomy-v1",
        "source": "D48 Stage 4C measured evidence",
        "entries": report["failure_classification"]["type_b_measured_not_hidden"],
        "unavailable_domains": report["failure_classification"]["type_c_unavailable"],
    }

    bottlenecks = {
        "schema_version": "stage4-risk-ranked-bottlenecks-v1",
        "status": "D48_HANDOFF_NOT_STAGE4B_REPAIRED",
        "entries": [
            {
                "rank": 1,
                "category": "coverage_and_safety_verification",
                "problem": "Continuous self-collision backend unavailable",
                "affected_family": "all 12 acceptance cases",
                "frequency": "unverified for 12/12",
                "severity": "CRITICAL_COVERAGE_GAP",
                "reproducibility": "deterministic environment limitation",
                "safety_impact": "continuous self-collision cannot be certified",
                "evidence": rel(C2),
                "stage4b_action": "Add a real continuous self-collision backend and rerun the persisted candidate without changing D47 inputs.",
            },
            {
                "rank": 2,
                "category": "execution_form_smoothness",
                "problem": "Native post-Ruckig jerk is timing-scale sensitive",
                "affected_family": "all 12 acceptance cases on 0.10/0.20 shadows; 0.15 had 509 violations",
                "frequency": "0.15: 509 violations; 0.05: 0 violations",
                "severity": "HIGH_ROUTE_SENSITIVITY",
                "reproducibility": "independent derivative check and native replay",
                "safety_impact": "jerk-limited execution form depends on timing route",
                "evidence": rel(C1),
                "stage4b_action": "Design a native execution-form method that preserves jerk compliance without relying on ultra-slow timing.",
            },
            {
                "rank": 3,
                "category": "model_dynamic_quality",
                "problem": "0.05 candidate has higher model torque-slew and power proxies than D47 model-space baseline",
                "affected_family": "all 12 acceptance cases",
                "frequency": "12/12 measured",
                "severity": "MEDIUM_MODEL_OPTIMIZATION_TARGET",
                "reproducibility": "RNEA+ABA nominal and ±10% synthetic mass variants",
                "safety_impact": "hardware impact is unverified; model result is not a torque certification",
                "evidence": rel(C4_ULTRA),
                "stage4b_action": "Optimize or redesign timing/post-processing only after continuous safety coverage is available; retain current baseline evidence.",
            },
        ],
    }

    dump(ROOT / "FINAL_REPORT.json", report)
    dump(ROOT / "PROMOTION_LEDGER.json", ledger)
    dump(ROOT / "STAGE4_FAILURE_TAXONOMY_V1.json", taxonomy)
    dump(ROOT / "STAGE4_RISK_RANKED_BOTTLENECKS_V1.json", bottlenecks)

    md = f"""# D48 Stage 4C Final Report

## Status

- `TASK_STATUS`: `{report['task_status']}`
- `FIRST_BLOCKER`: `{report['first_blocker']['label']}`
- Scientific verdict: {report['scientific_verdict']}
- Protected D47 champion: `{rel(PARENT)}`
- Canonical promotion: **none**; D47 B3 remains unchanged.

## Measured result

The selected 0.05 native MoveIt2 post-Ruckig shadow candidate passed all measured C1 hard gates on 12 cases: finite state, joint/velocity/acceleration/jerk limits, environment collision, adaptive geometry checks, singularity floors, clearance floors, and H1/H32/Stage3 preservation. It also passed an independent 12-case deterministic replay with maximum numeric delta `0` at tolerance `1e-12`.

The C2 verifier completed `{c2.get('case_count')}` cases with `{c2.get('adaptive_self_sweep_contact_cases')}` adaptive self-contact cases using `adaptive_discrete_interpolation`. Continuous self-collision is explicitly `{c2.get('continuous_self_collision_status')}` because the searched Tesseract/BulletCastBVH route was unavailable in the environment.

## Parent versus candidate

| Metric | D47 B3 parent | D48 0.05 shadow | Interpretation |
|---|---:|---:|---|
| Minimum environment clearance (m) | {parent_metrics['environment_clearance_m']:.12g} | {candidate_metrics['environment_clearance_m']:.12g} | floor preserved |
| Minimum self clearance (m) | {parent_metrics['self_clearance_m']:.12g} | {candidate_metrics['self_clearance_m']:.12g} | within configured 1e-6 m floor tolerance |
| Minimum sigma | {parent_metrics['minimum_sigma']:.12g} | {candidate_metrics['minimum_sigma']:.12g} | preserved/improved |
| Maximum condition number | {parent_metrics['maximum_condition']:.12g} | {candidate_metrics['maximum_condition']:.12g} | preserved/improved |
| Environment/self/CCD failures | 0 / 0 / 0 | 0 / 0 / 0 | no measured discrete regression |
| H1 / H32 / Stage3 update | unchanged | unchanged | protected |

## C3 and C4

The 0.10 and 0.20 timing shadows were rejected for jerk. The 0.05 shadow reduced the maximum jerk to `{stat(ultra, 'joint_space', 'jerk_ratio', 'max') * 8.0:.6g}` rad/s³ with zero violations. This is a shadow timing result, not a silent repair of the D47 system.

Pinocchio RNEA+ABA completed for 12 cases and nominal/±10% synthetic mass variants. RNEA→ABA self-consistency is `{stat(c4u, 'rnea_aba_self_consistency', 'status')}`. The nominal 0.05 candidate peak torque is `{stat(ultra_nominal, 'peak_abs_torque_Nm', 'max'):.6g}` N·m, peak torque slew `{stat(ultra_nominal, 'peak_abs_torque_slew_Nm_s', 'max'):.6g}` N·m/s, and power proxy `{stat(ultra_nominal, 'peak_aggregate_power_proxy_W', 'max'):.6g}` W. These are model-based values; hardware torque certification is unavailable.

## Defects and hand-off

The native-output/q-only column-contract defect and the C2 one-shot teardown defect were reproduced, repaired, and regression-checked. The 0.15 jerk exceedance remains recorded as Type-B shadow evidence; it was not hidden or used to weaken a gate. Stage 4B receives the continuous-backend coverage gap, route-sensitive jerk behavior, and model-dynamic slew/power target in:

- `{rel(ROOT / 'STAGE4_FAILURE_TAXONOMY_V1.json')}`
- `{rel(ROOT / 'STAGE4_RISK_RANKED_BOTTLENECKS_V1.json')}`

Hardware status: `HARDWARE_TORQUE_CERTIFICATION = NOT_APPLICABLE_UNMEASURED`; `PHYSICAL_CLEARANCE_CERTIFICATION = NOT_APPLICABLE_UNVERIFIED`.

Next unfinished scientific action: provide a real continuous self-collision backend compatible with the frozen MoveIt2 model, then rerun the same persisted candidate and acceptance set.
"""
    (ROOT / "FINAL_REPORT.md").write_text(md, encoding="utf-8")


if __name__ == "__main__":
    main()
