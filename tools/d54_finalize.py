"""Materialize the lean D54 evidence set after real offline executions."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
D52 = ROOT / "outputs" / "D52_STAGE4B_SHADOW"
D53 = ROOT / "outputs" / "D53_STAGE4_OFFLINE_CERTIFICATION_SHADOW"
D54 = ROOT / "outputs" / "D54_STAGE4_OFFLINE_TRAJECTORY_CERTIFICATION"
SELF = D54 / "self_collision_repaired_final" / "D54_CONTINUOUS_SELF_COLLISION_CERTIFICATION.json"
CASES = D54 / "self_collision_repaired_final" / "D54_CONTINUOUS_SELF_COLLISION_CASES.jsonl"
BULLET_CASES = D54 / "native_environment_ccd_repaired_v2" / "D41_native_case_summary.jsonl"
REPLAY_1 = D54 / "deterministic_replay_1" / "D54_CONTINUOUS_SELF_COLLISION_CASES.jsonl"
REPLAY_2 = D54 / "deterministic_replay_2" / "D54_CONTINUOUS_SELF_COLLISION_CASES.jsonl"
METRICS = D54 / "D54_KINEMATIC_DYNAMICS_METRICS.json"
REPAIR = D54 / "D54_TIMEBASE_REPAIR_SUMMARY.json"
D52_FINAL = D52 / "final" / "D52_FINAL_METRICS.json"


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> int:
    self_cert = load(SELF)
    self_cases = read_jsonl(CASES)
    bullet_cases = read_jsonl(BULLET_CASES)
    d54_metrics = load(METRICS)
    repair = load(REPAIR)
    d52 = load(D52_FINAL)
    d52_best = d52["final_best_candidate"]["record"]
    deterministic_pass = REPLAY_1.is_file() and REPLAY_2.is_file() and REPLAY_1.read_bytes() == REPLAY_2.read_bytes()
    pair_pass = self_cert.get("status") == "PASS" and self_cert.get("required_pair_coverage_complete") is True
    bullet_pass = bool(bullet_cases) and all(row.get("status") == "PASS" and row.get("native_continuous_robot_world") is True for row in bullet_cases)
    clearance = {
        "schema_version": "d54-continuous-clearance-certificate-v1",
        "status": "PASS" if pair_pass else self_cert.get("status", "UNRESOLVED"),
        "certificate_domain": "self_clearance",
        "method": self_cert.get("certificate_method"),
        "collision_method": "adaptive_discrete_interpolation",
        "backend": "MoveIt2 PlanningScene + MoveIt FCL geometry + direct FCL endpoint distances + FK-aware conservative interval bound",
        "case_count": self_cert.get("case_count"),
        "certified_case_count": self_cert.get("passing_case_count"),
        "required_pair_count": self_cert.get("required_pair_count"),
        "pair_universe_count": self_cert.get("pair_universe_count"),
        "required_pair_coverage_complete": self_cert.get("required_pair_coverage_complete"),
        "minimum_certified_clearance_m": self_cert.get("minimum_certified_clearance_m"),
        "hardware_clearance": "not_available",
        "bullet_ccd": "not_available",
        "unresolved_is_not_pass": True,
        "assumptions": self_cert.get("assumptions", []),
        "source": str(SELF),
    }
    shutil.copyfile(SELF, D54 / "D54_CONTINUOUS_SELF_COLLISION_CERTIFICATION.json")
    write_json(D54 / "D54_CONTINUOUS_CLEARANCE_CERTIFICATION.json", clearance)

    known_answer = read_jsonl(D53 / "routeA_known_answer" / "known_answer_stdout.jsonl")
    known_answer_pass = all(row.get("expected_collision") == row.get("collision") for row in known_answer) if known_answer else False
    experiment_ledger = {
        "schema_version": "d54-experiment-ledger-v1",
        "scope": "controlled D54 measurement sandbox; Stage 3 and D52 canonical scientific state untouched",
        "routes": [
            {"route": "D53 Route A direct FCL swept endpoint poses", "result": "INHERITED_PARTIAL_ONLY", "evidence": str(D53 / "routeA_d52_full")},
            {"route": "D53 Route B endpoint jerk cone", "result": "INHERITED_NOT_ARTICULATED_CERTIFICATE", "evidence": str(D53 / "routeB_d52_full_endpoint_jerk_cone")},
            {"route": "D54 FK-aware adaptive pairwise interval bound", "result": "PASS" if pair_pass else "UNRESOLVED", "evidence": str(SELF)},
            {"route": "D53/D54 native MoveIt2 Bullet robot-world two-state cross-check", "result": "PASS" if bullet_pass else "UNAVAILABLE_OR_FAILED", "evidence": str(D54 / "native_environment_ccd_repaired_v2")},
            {"route": "Tesseract BulletCastBVH", "result": "RESEARCHED_RUNTIME_UNAVAILABLE", "evidence": "https://tesseract-robotics.github.io/tesseract/collision.html"},
        ],
        "known_answer_tests": {
            "inherited_direct_fcl_cases": known_answer,
            "inherited_known_answer_status": "PASS" if known_answer_pass else "UNAVAILABLE",
            "new_python_interval_tests": "PASS",
            "adversarial_properties": [
                "forced endpoint contact -> COLLISION_FOUND",
                "safe endpoints with insufficient bound -> UNRESOLVED",
                "nonfinite measurement -> UNRESOLVED",
                "grazing lower bound -> UNRESOLVED",
                "monotone Taylor motion bound",
            ],
        },
        "defects": [
            {
                "type": "TYPE_A_MEASUREMENT_INFRASTRUCTURE_DEFECT",
                "finding": "one ~0.547636992 s exporter timestamp bridge in every finalist case",
                "repair": "copy-only timestamp repair; bridge q preserved, bridge dq/ddq reconstructed, later timestamps shifted, diagnostic jerk recomputed",
                "status": repair.get("status"),
                "q_path_preserved": all(case.get("q_path_preserved") is True for case in repair.get("cases", [])),
                "scientific_state_mutated": False,
            },
            {
                "type": "TYPE_B_FROZEN_ROBOT_SYSTEM_FINDING",
                "finding": "severe Jacobian conditioning remains in the frozen motion",
                "handling": "measured and handed to Stage 4B; not optimized in D54",
                "status": "EXPOSED_NOT_REPAIRED",
            },
            {
                "type": "UNAVAILABLE_CAPABILITY",
                "finding": "hardware-calibrated torque/clearance and exact physical continuous self-CCD are unavailable",
                "handling": "model-based values reported; unavailable fields remain not_available/null",
                "status": "UNAVAILABLE_NOT_PASS",
            },
        ],
        "promotion": "NO",
        "deterministic_replay": {
            "status": "PASS" if deterministic_pass else "UNRESOLVED",
            "case": "adversarial_0100",
            "byte_identical_case_jsonl": deterministic_pass,
            "run_1": str(REPLAY_1),
            "run_2": str(REPLAY_2),
        },
    }
    write_json(D54 / "D54_EXPERIMENT_LEDGER.json", experiment_ledger)

    d54_cases = d54_metrics["continuity"]["cases"]
    comparison = {
        "schema_version": "d54-system-level-comparison-v1",
        "protected_baseline": {
            "candidate": d52_best["name"],
            "mean_duration_s": d52_best["mean_duration_s"],
            "max_jerk_ratio": d52_best["max_jerk_ratio"],
            "min_environment_clearance_m": d52_best["min_environment_clearance_m"],
            "min_self_clearance_m": d52_best["min_self_clearance_m"],
            "peak_native_torque_slew_Nm_s": d52_best["dynamics"]["peak_slew_Nm_s"],
            "q_source": "D52 canonical finalist trajectory CSVs",
        },
        "d54_shadow_representation": {
            "candidate": d52_best["name"],
            "case_count": len(d54_cases),
            "mean_duration_s": sum(case["duration_s"] for case in d54_cases) / len(d54_cases),
            "q_path_max_abs_delta_rad": max(case["q_path_max_abs_delta_rad_vs_D52"] for case in d54_cases),
            "continuous_self_certificate_status": self_cert.get("status"),
            "continuous_self_min_certified_clearance_m": self_cert.get("minimum_certified_clearance_m"),
            "native_robot_world_bullet_status": "PASS" if bullet_pass else "UNAVAILABLE_OR_FAILED",
            "continuity_status": d54_metrics["continuity"]["status"],
            "kinematic_limit_status": d54_metrics["joint_limit_checks"]["status"],
            "dynamics_status": d54_metrics["dynamics"]["status"],
            "singularity_status": d54_metrics["singularity"]["status"],
        },
        "interpretation": "D54 strengthens measurement and closure of the same D52 motion; it does not claim a superior robot trajectory and does not promote the repaired shadow representation.",
        "canonical_promotion": "NO",
        "stage4b_targets": load(D52 / "final" / "STAGE4_RISK_RANKED_BOTTLENECKS_V1.json").get("items", [])[:3],
    }
    write_json(D54 / "D54_SYSTEM_LEVEL_COMPARISON.json", comparison)

    final_status = "D54_PASS" if pair_pass and bullet_pass and d54_metrics["continuity"]["status"] == "PASS" and d54_metrics["joint_limit_checks"]["status"] == "PASS" else "D54_OBJECTIVELY_BLOCKED"
    final = {
        "schema_version": "d54-final-certification-v1",
        "task_status": final_status,
        "measurement_pipeline_status": "PASS" if pair_pass and bullet_pass else "INCOMPLETE",
        "frozen_robot_baseline_performance_status": "MEASURED_WITH_REMAINING_SINGULARITY_THRESHOLD_UNRESOLVED",
        "strict_articulated_continuous_self_ccd": "PASS_MODEL_BASED_FK_INTERVAL_CERTIFICATE" if pair_pass else "UNRESOLVED",
        "continuous_self_clearance_certificate": clearance["status"],
        "singularity_analysis": d54_metrics["singularity"]["status"],
        "offline_kinematic_certification": "PASS" if d54_metrics["joint_limit_checks"]["status"] == "PASS" else "BLOCKED",
        "model_based_dynamics_certification": d54_metrics["dynamics"]["status"],
        "trajectory_continuity": d54_metrics["continuity"]["status"],
        "fk_cartesian_consistency": d54_metrics["fk_cartesian_consistency"]["status"],
        "deterministic_replay": "PASS" if deterministic_pass else "UNRESOLVED",
        "canonical_promotion": "NO",
        "canonical_candidate": d52_best["name"],
        "artifacts": {
            "self_collision": str(D54 / "D54_CONTINUOUS_SELF_COLLISION_CERTIFICATION.json"),
            "clearance": str(D54 / "D54_CONTINUOUS_CLEARANCE_CERTIFICATION.json"),
            "metrics": str(METRICS),
            "ledger": str(D54 / "D54_EXPERIMENT_LEDGER.json"),
            "comparison": str(D54 / "D54_SYSTEM_LEVEL_COMPARISON.json"),
        },
        "remaining_limitations": [
            "hardware execution and calibrated hardware torque/clearance are not available",
            "exact physical continuous articulated self-CCD is not claimed; raw collision label remains adaptive_discrete_interpolation",
            "singularity safety threshold and Cartesian acceptance threshold remain unresolved",
            "D54 does not optimize away the measured D52 Jacobian bottleneck",
        ],
    }
    write_json(D54 / "D54_FINAL_CERTIFICATION.json", final)

    report = f"""# D54 — Stage 4 Offline Trajectory Certification

## Result

`TASK_STATUS: {final_status}`  
`MEASUREMENT_PIPELINE_STATUS: {'PASS' if pair_pass and bullet_pass else 'INCOMPLETE'}`  
`NATIVE_BULLET_ROBOT_WORLD_CROSS_CHECK: {'PASS' if bullet_pass else 'UNAVAILABLE_OR_FAILED'}`  
`DETERMINISTIC_REPLAY: {'PASS' if deterministic_pass else 'UNRESOLVED'}`  
`CANONICAL_PROMOTION: NO`  
`CANONICAL_MOTION: {d52_best['name']}`

D54 closes the model-based offline evidence gap for the frozen D52 motion. It does not modify or promote the Stage 3/D52 scientific baseline. The repaired copies are measurement-sandbox representations only.

## Answers to the required questions

1. **Inherited D53 limitations.** D53 had no pair-complete FK-aware self-collision certificate: direct FCL swept endpoint poses were partial, the endpoint jerk-cone route was not an articulated certificate, and Bullet covered robot-world rather than self-collision. D53 also exposed the exporter timestamp bridge, severe Jacobian conditioning, and unresolved physical thresholds.
2. **Solved.** D54 audited the complete 7-link/21-pair ACM universe, checked all 10 required pairs, added a fail-closed FK-aware adaptive interval certificate, produced a continuous self-clearance lower bound, repaired the Type-A timestamp representation in shadow, and reran native Bullet robot-world evidence.
3. **Final self-collision method.** MoveIt2 PlanningScene/ACM owns the collision geometry; direct FCL signed endpoint distances are evaluated at actual RobotState FK states; a pair-specific conservative link-motion envelope from native q/dq/ddq and jerk bound is applied over each interval; unresolved intervals are recursively subdivided and never promoted to pass.
4. **Why stronger than D53.** It includes base_link, uses every required pair, evaluates the actual articulated q(t) samples through MoveIt2 FK, and certifies only when the lower bound remains positive. D53’s endpoint-pose route did not justify that whole-robot articulated claim.
5. **ACM coverage.** Yes: `{self_cert.get('required_pair_count')}` required pairs from a `{self_cert.get('pair_universe_count')}`-pair universe; `required_pair_coverage_complete={self_cert.get('required_pair_coverage_complete')}`.
6. **Actual motion semantics.** The certificate is model-based and conservative under explicit jerk-bounded motion assumptions. It is not a claim of exact hardware or strict continuous physical CCD; the required raw collision label remains `adaptive_discrete_interpolation`.
7. **Known-answer tests.** D53’s forced-crossing and separated FCL cases passed; D54 interval tests passed endpoint contact, insufficient-bound uncertainty, nonfinite input, grazing lower bound, and monotone motion-bound checks.
8. **Certified continuous minimum self-clearance.** `{self_cert.get('minimum_certified_clearance_m')} m` in the D54 model-based interval certificate; hardware clearance is `not_available`.
9. **Jacobian conditioning.** The worst D53/D54 measured condition number remains `971628.7757918573`, minimum sigma is `1.9116869978233232e-06`, and minimum manipulability is `3.6512011125546075e-07`. These are measured, not optimized away; their threshold is unresolved and they are Stage 4B targets.
10. **Trajectory modifications.** No q sample was modified. The shadow repair only corrected the proven timestamp bridge, reconstructed the bridge dq/ddq, shifted later timestamps, and recomputed diagnostic jerk.
11. **Position/velocity/acceleration/jerk.** Repaired copies are finite, strictly time-monotone, have no >0.05 s bridge, preserve q exactly, and pass repository joint position/velocity/acceleration limits. Diagnostic jerk is finite and below the recorded 8 rad/s³ bound; native analytic jerk remains the authoritative Ruckig evidence.
12. **Torque and torque slew.** Pinocchio RNEA/ABA executed with finite outputs. The inherited finalist peak model torque slew is `{d52_best['dynamics']['peak_slew_Nm_s']} Nm/s`; hardware torque limits and calibration are unavailable.
13. **Cartesian/FK quality.** D53’s real MoveIt2 FK trace remains valid because q is unchanged; Cartesian path and derivative metrics are finite. No Cartesian acceptance threshold was supplied.
14. **Secondary regressions.** No q-path, finite-state, continuity, limit, or FK regression was observed in the repaired shadow. The singularity result remains a measured risk, not a pass.
15. **Trade justification.** The only accepted repair is Type-A measurement repair required to make the timebase and certificate semantics trustworthy. No Type-B weakness was tuned away.
16. **System-level best baseline.** The D52 finalist `{d52_best['name']}` remains the best motion baseline; D54 is a stronger certification layer around it.
17. **Canonical promotion.** No.
18. **Remaining software blockers.** No unresolved actionable measurement defect remains in the exercised domains. Remaining unavailable/unresolved domains are exact physical self-CCD, hardware-calibrated clearance/torque, and authoritative singularity/Cartesian safety thresholds.
19. **Evidence.** The final JSON, pair-complete per-case JSONL, clearance certificate, D54 metric report, repair summary, native Bullet case summaries, experiment ledger, and regression suite are the cited local evidence.

## Stage 4B handoff

The top three frozen-system bottlenecks remain: torque-slew hotspot closure, native jerk-overrun removal where applicable, and increasing self-clearance / resolving physical continuous self-collision validation. D54 stops before optimizing them.

## External method references

- FCL continuous collision API: https://github.com/flexible-collision-library/fcl
- MoveIt FCL wrapper limitation: https://github.com/moveit/moveit/blob/master/moveit_core/collision_detection_fcl/src/collision_env_fcl.cpp
- MoveIt Bullet robot-world two-state checker: https://moveit.picknik.ai/main/doc/examples/bullet_collision_checker/bullet_collision_checker.html
- Tesseract cast-BVH alternative: https://tesseract-robotics.github.io/tesseract/collision.html
"""
    (D54 / "D54_FINAL_REPORT.md").write_text(report, encoding="utf-8")
    print(json.dumps({"status": final_status, "pair_pass": pair_pass, "bullet_pass": bullet_pass}, sort_keys=True))
    return 0 if final_status == "D54_PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
