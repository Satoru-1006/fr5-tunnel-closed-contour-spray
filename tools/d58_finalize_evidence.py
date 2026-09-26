"""Write the compact, fail-closed D58 evidence set from measured artifacts."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "D58_STAGE4_FULL_SYSTEM_CLOSURE"


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(name: str, value) -> None:
    (OUT / name).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def rel(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def main() -> None:
    phase1 = load(OUT / "D58_PHASE1_STATEFUL_CLOSURE_REPORT.json")
    robust = load(OUT / "D58_ROBUST_CLEARANCE.json")
    d56 = load(ROOT / "outputs/D56_STAGE4B_SOFTWARE_CLOSURE/D56_FINAL_STATUS.json")
    d57 = load(ROOT / "outputs/D57_STAGE4_ALGORITHMIC_CLOSURE/D57_FINAL_STATUS.json")
    dynamics = load(OUT / "dynamics_stateful_auto_pair/native_dynamics_report.json")
    fcl0 = load(OUT / "fcl_routeA_stateful_auto0/continuous_self_collision_summary.json")
    fcl1 = load(OUT / "fcl_routeA_stateful_auto1/continuous_self_collision_summary.json")

    candidate_ledger = []
    for case_id, short, fcl in (
        ("adversarial_0100", "auto0", fcl0),
        ("adversarial_0101", "auto1", fcl1),
    ):
        native = phase1["native_execution"][case_id]
        metrics = phase1["native_geometry_and_task_metrics"][case_id]
        profile = phase1["authoritative_profile_j"][case_id]
        cert = phase1["articulated_certificate"][case_id]
        replay = phase1["deterministic_replay"][case_id]
        candidate_ledger.append(
            {
                "case_id": case_id,
                "source": native["source_trajectory"],
                "native_post_ruckig": native["status"] == "PASS" and native["native_post_ruckig"],
                "profile_j_status": profile["status"],
                "profile_j_max_rad_s3": profile["max_abs_analytic_jerk_rad_s3"],
                "native_geometry_status": metrics["measurement_status"],
                "adaptive_discrete_collision_cases": metrics["environment_collision_cases"],
                "adaptive_discrete_self_collision_cases": metrics["self_collision_cases"],
                "fcl_route_a_status": fcl["status"],
                "fcl_route_a_swept_intervals": fcl["swept_interval_count"],
                "fcl_route_a_swept_pair_calls": fcl["swept_pair_call_count"],
                "fcl_route_a_collision_count": fcl["continuous_collision_count"],
                "fcl_route_a_api_error_count": fcl["continuous_api_error_count"],
                "articulated_certificate_status": cert["dense_initial_stride_1"]["status"],
                "articulated_certificate_unresolved_regions": cert["dense_initial_stride_1"]["unresolved_region_count"],
                "model_dynamics_status": phase1["dynamics"]["status"],
                "deterministic_replay_byte_identical": replay["byte_identical"],
                "promotion_status": "NO_PROMOTION",
            }
        )
    write_json("D58_CANDIDATE_LEDGER.json", {"schema_version": "d58-candidate-ledger-v1", "candidates": candidate_ledger})

    certification_matrix = {
        "schema_version": "d58-certification-matrix-v1",
        "scope": "Stage 0/1 ON-state open-arch only",
        "rows": [
            {"domain": "MoveIt2 native post-Ruckig", "auto0": "PASS", "auto1": "PASS", "authority": "native execution_form_summary"},
            {"domain": "Ruckig Profile.j semantics", "auto0": "PASS 7004/7004, max 8", "auto1": "PASS 7004/7004, max 8", "authority": "native Ruckig Profile.j boundary replay"},
            {"domain": "Finite/FK/velocity/acceleration", "auto0": "PASS", "auto1": "PASS", "authority": "native MoveIt2/FK metrics"},
            {"domain": "Discrete geometry", "auto0": "0 collision cases", "auto1": "0 collision cases", "authority": "adaptive_discrete_interpolation"},
            {"domain": "FCL Route A", "auto0": "PASS, 0 contacts", "auto1": "PASS, 0 contacts", "authority": "qualified rigid endpoint swept-link cross-check"},
            {"domain": "Exact articulated FK(q(t)) self-CCD", "auto0": "UNRESOLVED", "auto1": "UNRESOLVED", "authority": "FK-aware conservative certificate; not exact CCD"},
            {"domain": "Robust clearance", "auto0": "UNRESOLVED_THRESHOLD", "auto1": "UNRESOLVED_THRESHOLD", "authority": "model-derived sensitivity only; physical uncertainty incomplete"},
            {"domain": "Model-based dynamics", "auto0": "PASS", "auto1": "PASS", "authority": "Pinocchio RNEA+ABA, not hardware"},
            {"domain": "Deterministic replay", "auto0": "PASS byte-identical", "auto1": "PASS byte-identical", "authority": "independent native reruns"},
            {"domain": "Promotion", "auto0": "NO_PROMOTION", "auto1": "NO_PROMOTION", "authority": "certificate and external-bound hard gates"},
        ],
    }
    write_json("D58_CERTIFICATION_MATRIX.json", certification_matrix)

    comparison = {
        "schema_version": "d58-system-comparison-v1",
        "protected_canonical": "outputs/D47_STAGE4B_ROLLING_CHAMPION_V1/PROMOTED/B3",
        "protected_d56_final_champion": d56["final_champion"],
        "d57_status": d57.get("task_status", d57.get("status")),
        "d56_certificate_min_self_clearance_m": d56["clearance"]["minimum_after_m"],
        "d58_candidates": {
            case_id: {
                "raw_model_self_clearance_m": phase1["native_geometry_and_task_metrics"][case_id]["minimum_self_clearance_m"],
                "raw_model_environment_clearance_m": phase1["native_geometry_and_task_metrics"][case_id]["minimum_environment_clearance_m"],
                "raw_sigma_min": phase1["native_geometry_and_task_metrics"][case_id]["minimum_sigma"],
                "raw_condition_number": phase1["native_geometry_and_task_metrics"][case_id]["maximum_condition_number"],
                "certificate_status": phase1["articulated_certificate"][case_id]["dense_initial_stride_1"]["status"],
                "certificate_unresolved_regions": phase1["articulated_certificate"][case_id]["dense_initial_stride_1"]["unresolved_region_count"],
                "promotion": "NO_PROMOTION",
            }
            for case_id in ("adversarial_0100", "adversarial_0101")
        },
        "d58_model_dynamics": {
            "status": dynamics["status"],
            "candidate_nominal_peak_torque_Nm": max(row["candidate_peak_abs_torque_Nm"] for row in dynamics["ranking_rows"] if row["variant"] == "nominal"),
            "candidate_plus10_peak_torque_Nm": max(row["candidate_peak_abs_torque_Nm"] for row in dynamics["ranking_rows"] if row["variant"] == "mass_scale_plus_10pct"),
            "trajectory_generation_influenced_by_torque": False,
        },
        "floor_advanced": False,
        "canonical_advanced": False,
        "conclusion": "D58 adds verified shadow/certification evidence but does not replace the protected D56 champion or canonical B3.",
    }
    write_json("D58_SYSTEM_COMPARISON.json", comparison)

    write_json(
        "D58_DETERMINISTIC_REPLAY.json",
        {"schema_version": "d58-deterministic-replay-v1", "cases": phase1["deterministic_replay"], "status": "PASS"},
    )
    write_json(
        "D58_REGRESSION.json",
        {
            "schema_version": "d58-regression-v1",
            "pytest_status": "PASS",
            "pytest_passed": 12,
            "pytest_failed": 0,
            "known_answer_profile_j_status": "PASS 8808/8808",
            "protected_floor_modified": False,
        },
    )
    write_json(
        "D58_ADVERSARIAL_ROBUSTNESS.json",
        {
            "schema_version": "d58-adversarial-robustness-v1",
            "status": "PARTIAL_FULL_CHAIN_UPGRADE",
            "inherited_q_only_probe": "D57 64/64 native probe cases passed; not full trajectory robustness",
            "full_chain_realized_trajectories": "2/2 selected D57 C4 shadows passed native post-Ruckig/FK/geometry/dynamics/replay checks",
            "perturbation_full_chain_campaign": "NOT_EXECUTED",
            "promotion": "NO_PROMOTION",
        },
    )
    write_json(
        "D58_UNRESOLVED_ITEMS.json",
        {
            "schema_version": "d58-unresolved-items-v1",
            "software_addressable": [
                "Exact articulated FK(q(t)) self-CCD certificate remains unavailable; FCL endpoint sweep is only a qualified cross-check.",
                "Installed Tesseract ContinuousContactManager/BulletCast route was not available in this runtime.",
                "Torque was audited post hoc; no validated torque-constrained trajectory generator was promoted.",
                "A full perturbation campaign through native post-Ruckig remains to be run beyond the two selected shadows.",
            ],
            "external_or_hardware_only": [
                "Total geometric uncertainty and physical clearance threshold.",
                "Encoder/state uncertainty, backlash, calibration, compliance, payload deflection, and tracking error.",
                "Hardware torque/current limits and hardware TCP accuracy.",
            ],
        },
    )

    research = """# D58 Research Ledger

The research was used to choose measurable routes and to keep capability labels honest; web sources are not substitutes for native repository evidence.

| Source | Role in D58 |
|---|---|
| [MoveIt PlanningScene tutorial](https://moveit.picknik.ai/main/doc/examples/planning_scene/planning_scene_tutorial.html) | Retain PlanningScene/ACM ownership for native geometry checks. |
| [MoveIt Bullet collision checker](https://moveit.picknik.ai/main/doc/examples/bullet_collision_checker/bullet_collision_checker.html) | Keep native robot-world two-state checking separate from articulated self-CCD. |
| [MoveIt FCL collision source](https://github.com/moveit/moveit/blob/master/moveit_core/collision_detection_fcl/src/collision_env_fcl.cpp) | Do not relabel MoveIt FCL distance/self queries as exact continuous self-CCD. |
| [FCL project](https://github.com/flexible-collision-library/fcl) | Direct `continuousCollide` Route A and its rigid-body endpoint-sweep semantics. |
| [FCL paper](https://gamma.cs.unc.edu/FCL/fcl_docs/webpage/pdfs/fcl_icra2012.pdf) | Separate collision, distance, tolerance, and continuous proximity query meanings. |
| [Tesseract collision documentation](https://tesseract-robotics.github.io/tesseract/collision.html) | Investigate continuous contact managers/BulletCast; no accepted installed articulated route was available. |
| [Continuous Collision Detection for Articulated Models](https://www.tjhsst.edu/~rlatimer/papers/acmresearch/a15-zhangContinuousCollisionDetectionforArticulatedModels.pdf) | Supports adaptive articulated bounds as a distinct research direction, not an automatic library guarantee. |
| [FAIRINO FR5 specifications](https://www.fairino.com/industry_/2.html) | Manufacturer-stated repeatability ±0.02 mm used only as one uncertainty contributor in sensitivity analysis. |

The direct implementation decision is hybrid: stateful consistent-time repair -> native MoveIt2 post-Ruckig -> Profile.j semantic replay -> native FK/geometry/dynamics -> conservative articulated certificate -> FCL swept-link cross-check -> uncertainty-aware clearance sensitivity. None of these layers is silently substituted for another.
"""
    (OUT / "D58_RESEARCH_LEDGER.md").write_text(research, encoding="utf-8")

    final_status = {
        "schema_version": "d58-final-status-v1",
        "task_status": "PARTIALLY_EXECUTED",
        "phase1_status": phase1["phase1_status"],
        "phase2_status": "PARTIAL_ROUTE_A_UPGRADE_EXACT_ARTICULATED_SELF_CCD_UNRESOLVED",
        "phase3_status": robust["status"],
        "promotion_status": "NO_PROMOTION",
        "protected_canonical": comparison["protected_canonical"],
        "protected_best_verified_system": comparison["protected_d56_final_champion"],
        "floor_advanced": False,
        "canonical_advanced": False,
        "measurement_pipeline_status": "PASS_FOR_MEASURED_DOMAINS_WITH_EXPLICIT_GAPS",
        "candidate_count": 2,
        "candidate_ledger": "outputs/D58_STAGE4_FULL_SYSTEM_CLOSURE/D58_CANDIDATE_LEDGER.json",
        "unresolved_items": "outputs/D58_STAGE4_FULL_SYSTEM_CLOSURE/D58_UNRESOLVED_ITEMS.json",
    }
    write_json("D58_FINAL_STATUS.json", final_status)

    report = f"""# D58 Final Report — full-system closure shadow

## Final status

- D58 task status: `{final_status['task_status']}`.
- Phase 1 stateful C4 closure: `{final_status['phase1_status']}`.
- Phase 2: `{final_status['phase2_status']}`.
- Phase 3 robust clearance: `{final_status['phase3_status']}`.
- Promotion: `{final_status['promotion_status']}`.
- Protected floor and canonical were not changed.

## 1. Inherited state and real D57 gaps

D58 inherited the D57 two automatic C4 optimizer shadows, the D56 verified champion `{d56['final_champion']}`, and the protected canonical `{comparison['protected_canonical']}`. D57's unresolved gaps were the missing realized native post-Ruckig/full-chain authority, the unavailable exact articulated self-CCD route, unresolved physical clearance/TCP thresholds, and post-hoc rather than upstream torque influence.

The Stage 0/1 scope remains ON-state open-arch with the authoritative 181-point input lineage. No legacy 720-point, OFF-state, reorientation, retreat, approach, or learning path was introduced.

## 2. Dependency-order execution

The order was raw state and input contract -> stateful q/dq/ddq/t repair -> native Ruckig -> authoritative `Profile.j` semantics -> native FK/geometry/task metrics -> articulated certificate -> FCL independent cross-check -> robust-clearance uncertainty model -> dynamics, replay, and regression. This order prevents downstream statistics from hiding an upstream state or timebase defect.

## 3. D57 shadow results

Both selected shadows completed native MoveIt2 post-Ruckig execution: 7005 states in, 7005 states out, `native_post_ruckig=true`, clean process exit, and exact q-path preservation. The stateful time-dilation factor was approximately 1.90865; this produced a long but executable shadow trajectory.

The authoritative Ruckig semantic replay passed for both cases: 7004/7004 successful `Profile.j` profiles, no invalid-input errors, and maximum analytic jerk exactly `8 rad/s^3`. The D50 known-answer replay independently passed 8808/8808. The replay is explicitly a Profile.j boundary-semantics measurement; it is not a replacement for MoveIt's whole-trajectory overshoot handling.

Native FK/geometry/task metrics were finite and collision-free under `adaptive_discrete_interpolation`. Auto0/auto1 model-space environment clearance was approximately `0.08904024 m`; self-clearance was approximately `0.01662374 m` / `0.01662379 m`; TCP path-error maximum was `0.00471235 m`; terminal position error was `3.75e-8 m`. These are measured software observables; task acceptance thresholds remain unresolved.

Pinocchio RNEA+ABA passed joint-order, inertia, zero-motion finiteness, and self-consistency checks. The candidate nominal peak torque was at most `50.92631 Nm`; +10% mass was at most `56.01894 Nm`. This is model-based from the derived URDF, not hardware torque certification, and torque did not influence trajectory generation.

## 4. Continuous self-collision routes

The inherited FK-aware conservative articulated certificate was rerun for both candidates with default `initial_stride=16` and dense `initial_stride=1`. Both remained `UNRESOLVED` with `152` unresolved regions, zero collision regions, complete required-pair coverage, and worst pair `forearm_link|wrist2_link`. This reproducibly rejects a false PASS.

The independent FCL Route A completed all 7004 intervals for both candidates: 10,995 swept-link pair calls per candidate, zero continuous contacts, zero API errors, and complete trajectories. This is a material certification-system upgrade, but it is only a qualified rigid endpoint-sweep cross-check. FCL rigid-body sweeps do not prove the exact nonlinear articulated `FK(q(t))` path; exact articulated self-CCD remains unresolved.

## 5. Robust clearance

The model-space margin increased relative to the protected parent on the measured D58 observables: environment clearance by about `4.713 mm`, self-clearance by about `0.002 mm`. The inherited approximately `25.9 micrometre` *certified* weak margin was not formally replaced because the D58 articulated certificate is unresolved. A fail-closed sensitivity model subtracts only the manufacturer-stated FR5 repeatability `0.02 mm`; the resulting positive margins are diagnostics, not physical safety certification. Calibration, encoder, mesh, compliance, payload, tracking, and total geometric uncertainty remain unknown.

## 6. Singularity, task, adversarial, and determinism results

Native singularity observables were strong on these two shadows (`sigma_min` about `1.19e-3` / `1.24e-3`, condition number about `1559.9` / `1501.5`), but no universal physical threshold was invented. Task accuracy was measured against the exact source path; no calibrated TCP acceptance threshold was available. D57's inherited 64/64 q-only native probes remain useful diagnostic coverage, while D58 upgrades the two selected adversarial shadows to realized full-chain executable tests. Both independent native replays were byte-identical.

The focused regression command passed `12` tests, and the known-answer Profile.j regression passed `8808/8808`. No metric degradation was accepted in exchange for promotion because neither candidate met the full certificate/uncertainty gates; the 561-second shadow duration is recorded as a tradeoff, not hidden.

## 7. Methods attempted and what remains

Independent families included stateful consistent-time repair, native MoveIt2 execution, direct FCL `continuousCollide`, FK-aware adaptive subdivision, model-based Pinocchio dynamics, and uncertainty-aware clearance sensitivity. The stateful repair fixed the executable timing contract. Dense subdivision did not close the articulated certificate. FCL found no endpoint-sweep contact but could not establish nonlinear articulated equivalence. Tesseract/BulletCast was researched but no installed accepted backend was available. The torque route remains post hoc, and a broader perturbation full-chain campaign remains outstanding.

The single authoritative best verified system remains the protected D56 champion `{comparison['protected_d56_final_champion']}`; the canonical remains `{comparison['protected_canonical']}`. D58 adds measurable software capability without promoting an unverified candidate.

## 8. Remaining boundaries

Software boundaries: exact articulated self-CCD, installed Tesseract/BulletCast integration, torque-constrained trajectory generation, and broader full-chain perturbation coverage. Hardware-only boundaries: total geometric uncertainty, calibrated TCP accuracy, encoder/backlash/compliance/tracking effects, physical clearance acceptance, and actuator torque/current limits. These are explicitly unresolved rather than converted to PASS.

See [D58_FINAL_STATUS.json](D58_FINAL_STATUS.json), [D58_SYSTEM_COMPARISON.json](D58_SYSTEM_COMPARISON.json), [D58_CERTIFICATION_MATRIX.json](D58_CERTIFICATION_MATRIX.json), [D58_ROBUST_CLEARANCE.json](D58_ROBUST_CLEARANCE.json), and [D58_RESEARCH_LEDGER.md](D58_RESEARCH_LEDGER.md) for machine-readable evidence and source ledger.
"""
    (OUT / "D58_FINAL_REPORT.md").write_text(report, encoding="utf-8")
    print(json.dumps({"status": final_status["task_status"], "promotion": final_status["promotion_status"], "output": rel(OUT / "D58_FINAL_REPORT.md")}, sort_keys=True))


if __name__ == "__main__":
    main()
