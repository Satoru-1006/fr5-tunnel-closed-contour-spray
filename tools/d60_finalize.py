"""Assemble the D60 authoritative status, report, and minimal ledger.

This finalizer reads the D59 authority plus D60 shadow evidence and writes
only the new D60 package.  It never mutates canonical or protected artifacts.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "D60_SYSTEM_LEVEL_CLOSURE"
D59 = ROOT / "outputs" / "D59_OFFLINE_ALGORITHM_SYSTEM_CLOSURE"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def compact_hybrid(path: Path) -> list[dict]:
    payload = load(path)
    return [
        {
            "candidate_id": row.get("candidate_id"),
            "architecture": row.get("architecture"),
            "geometry_stage": row.get("geometry_stage"),
            "native_summary_status": row.get("native_summary_status"),
            "screen_status": row.get("screen_status"),
            "case_count": row.get("case_count"),
            "duration_mean_s": row.get("duration_mean_s"),
            "duration_max_s": row.get("duration_max_s"),
            "max_jerk_ratio": row.get("max_jerk_ratio"),
            "jerk_limit_violations": row.get("jerk_limit_violations"),
            "promotion_status": row.get("promotion_status"),
        }
        for row in payload["candidates"]
    ]


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    d59 = load(D59 / "D59_FINAL_STATUS.json")
    semantic = load(OUT / "semantic_probes" / "D60_SEMANTIC_PROBES.json")
    certificate = load(OUT / "fk_qt_certificate" / "D60_FK_QT_CERTIFICATE_AUDIT.json")
    hybrid_v1 = load(OUT / "hybrid_shadows" / "performance_shadow_report.json")
    hybrid_v2_path = OUT / "hybrid_shadows_v2" / "D60_HYBRID_SHADOWS.json"
    hybrid_v2 = load(hybrid_v2_path)

    d60_status = {
        "schema_version": "d60-final-status-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "D60_TASK_STATUS": "PASS_FOR_MEASURED_OFFLINE_DOMAINS_NO_PROMOTION_REMAINING_ALGORITHMIC_GAPS",
        "D60_SCOPE": "OFFLINE_ALGORITHM_ONLY",
        "CLAIM_SCOPE": "historical scoped software-model evidence; no exact external articulated FK(q(t)) self-CCD or physical safety claim",
        "ARTICULATED_CCD_CLAIM": {
            "state": "HISTORICAL_SCOPED_MODEL_RESULT_SUPERSEDED_BY_D61_D62_D63_D64_COVERAGE_ACCOUNTING",
            "exact_external_articulated_fk_qt_ccd": "UNAVAILABLE_UNVERIFIED",
            "reader_warning": "The two zero-collision/zero-unresolved observations are model-scoped results, not a complete exact articulated continuous self-CCD proof.",
        },
        "TAKEOVER_REORIENTATION_RESULT": "PASS_LIVE_D59_STATE_RECONCILED_NO_PROTECTED_ARTIFACT_MUTATION",
        "STARTING_CANONICAL": d59["CURRENT_CANONICAL"],
        "STARTING_PROTECTED_FLOOR": d59["CURRENT_PROTECTED_FLOOR"],
        "FINAL_PROTECTED_FLOOR": d59["CURRENT_PROTECTED_FLOOR"],
        "FINAL_VERIFIED_CHAMPION": d59["FINAL_VERIFIED_CHAMPION"],
        "BEST_UNPROMOTED_SHADOW": "D59_TIMING025_PAIR_AUTO0_AND_PROVENANCE_CORRECTED_AUTO1_RETAINED; D60_NATIVE_HYBRIDS_REJECTED_BY_JERK",
        "CONTINUOUS_ARTICULATED_COLLISION_STATUS": "SCOPED_PASS_CONSERVATIVE_FK_QT_MODEL_FOR_2_CORRECTLY_ROUTED_FINALISTS; EXACT_EXTERNAL_ARTICULATED_BACKEND_NOT_AVAILABLE",
        "FK_QT_CERTIFICATE_STATUS": "HISTORICAL_SCOPED_PASS_D60_STRIDE1_DEEP_REFINEMENT_2_CASES_0_COLLISION_0_UNRESOLVED_CONSERVATIVE_MODEL_ONLY",
        "HARD_GATES": {
            "protected_d59_full_chain": d59["FULL_CHAIN_ROBUSTNESS_STATUS"],
            "d60_fk_qt_certificate": certificate["hard_gate_interpretation"],
            "d60_native_hybrid_v1": "REJECTED_JERK_ALL_3_CANDIDATES",
            "d60_native_hybrid_v2": "REJECTED_JERK_ALL_3_CANDIDATES",
        },
        "ADVERSARIAL_VALIDATION": {
            "synthetic_semantic_probes": "PASS_7_OF_7",
            "fcl_rigid_known_answer": semantic["fcl_known_answer"].get("status"),
            "d59_full_chain_robustness": d59["FULL_CHAIN_ROBUSTNESS_STATUS"]["pass_rate"],
            "focused_regression": "PASS_12_TESTS",
        },
        "SYSTEM_LEVEL_NET_GAIN": "NO_NEW_PROTECTED_SYSTEM_NET_GAIN; D60 ADDED VALIDATED SEMANTIC AND FK_QT EVIDENCE",
        "PROMOTION": "NO_PROMOTION",
        "MATERIAL_REGRESSION": "NO_PROTECTED_REGRESSION; D60 SHADOW HYBRIDS HAVE MATERIAL JERK REGRESSION AND WERE REJECTED",
        "FAILED_OR_REJECTED_MAJOR_ROUTES": [
            "Tesseract BulletCast unavailable in the active environment; no silent substitution",
            "FCL continuousCollide is a rigid begin/end motion API and cannot by itself certify nonlinear articulated FK(q(t))",
            "global_0075, asymmetric_005v_010a, local_0075_midpoints: native pass but jerk-gate rejected",
            "global_015, asymmetric_010v_015a, local_015_midpoints: native pass but jerk-gate rejected",
            "TOPPRA/COPP and other external research routes were investigated but not integrated into the accepted chain",
        ],
        "SUCCESSFUL_HYBRID_ROUTES": [
            "native MoveIt2/Ruckig plus Profile.j plus FK-aware conservative certificate plus FCL rigid cross-check plus Pinocchio model dynamics: successful validation composition, not a promoted optimizer",
            "D59 fixed-q dynamics-aware time search: bounded shadow route passed measured software gates but was equal to the protected floor",
        ],
        "REMAINING_ALGORITHMIC_GAPS": [
            "backend-independent exact or stronger theorem-level continuous articulated self-collision semantics for the realized q(t)",
            "a hybrid geometry-plus-retiming optimizer that remains within the native jerk contract and demonstrates a net system-level gain",
            "authorized software thresholds for torque, torque slew, singularity, task-space fidelity, and clearance remain unresolved or relative",
        ],
        "FIRST_GENUINELY_UNFINISHED_ACTION": "Obtain or implement a trajectory-aware articulated CCD/certificate with explicit FK(q(t)) semantics, then attack jerk-feasible hybrid path optimization under the same rolling hard gates",
        "evidence": {
            "d59_final_status": "outputs/D59_OFFLINE_ALGORITHM_SYSTEM_CLOSURE/D59_FINAL_STATUS.json",
            "semantic_probes": "outputs/D60_SYSTEM_LEVEL_CLOSURE/semantic_probes/D60_SEMANTIC_PROBES.json",
            "fk_qt_certificate": "outputs/D60_SYSTEM_LEVEL_CLOSURE/fk_qt_certificate/D60_FK_QT_CERTIFICATE_AUDIT.json",
            "hybrid_v1": "outputs/D60_SYSTEM_LEVEL_CLOSURE/hybrid_shadows/performance_shadow_report.json",
            "hybrid_v2": "outputs/D60_SYSTEM_LEVEL_CLOSURE/hybrid_shadows_v2/D60_HYBRID_SHADOWS.json",
            "focused_regression": "python -m pytest -q tests/test_d50_measurement_repairs.py tests/test_d54_articulated_certificate.py tests/test_d57_system_closure.py -> 12 passed",
        },
    }
    (OUT / "D60_FINAL_STATUS.json").write_text(json.dumps(d60_status, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    report = f"""# D60 — System-level algorithmic closure, hybrid exploration, and non-regressive rolling improvement

## Authoritative result

- `D60_TASK_STATUS`: `{d60_status['D60_TASK_STATUS']}`
- `TAKEOVER_REORIENTATION_RESULT`: `{d60_status['TAKEOVER_REORIENTATION_RESULT']}`
- Starting and final protected floor: `{d59['CURRENT_PROTECTED_FLOOR']}`
- Promotion: `NO_PROMOTION`

D60 preserved the D47 canonical and D59/D56 protected floor. All new code and
results are shadow-only under this D60 directory.

Claim fence: the D59/D60 zero-collision/zero-unresolved observations are
historical, scoped results under conservative software models. They are not a
complete exact external articulated `FK(q(t))` self-CCD proof. D61-D64 provide
the later coverage accounting and current freeze boundary.

## 1. Starting state and inherited uncertainty

D59 was reconciled from the live repository. Its corrected auto0/auto1
finalists were native-verified, model-dynamics checked, robust over 12 cases,
deterministically replayed, and certified by the accepted conservative FK-aware
model. The first unfinished dependency was the semantic boundary between a
rigid endpoint-sweep CCD query and exact nonlinear articulated `FK(q(t))` CCD.

## 2. Research and route ordering

The route order followed dependency order: backend semantics and known-answer
probes, then the FK-aware certificate, then bounded hybrid generation, then
hard-gate/adversarial comparison. The external routes investigated were:

- [MoveIt Bullet collision checker](https://moveit.picknik.ai/main/doc/examples/bullet_collision_checker/bullet_collision_checker.html): current documentation demonstrates CCD between two discrete robot states and access through `checkRobotCollision`.
- [Tesseract collision documentation](https://tesseract-robotics.github.io/tesseract/collision.html): documents plugin-loaded discrete/continuous managers and BulletCast managers that cast convex hulls between link poses. The required Tesseract collision plugin was not installed locally.
- [FCL source/API](https://github.com/flexible-collision-library/fcl) and [continuous request reference](https://flexible-collision-library.github.io/d2/d2a/structfcl_1_1ContinuousCollisionRequest.html): the endpoint API supplies goal transforms for moving geometric objects; this is not a joint-space `q(t)` interface.
- [Redon et al. articulated CCD](https://diglib.eg.org/items/13aac165-e652-4c78-a0a6-0b42305f0753): interval/BVH/swept-volume/exact-contact ideas support the direction of a stronger certificate, but no reusable exact implementation was available in this workspace.
- [Ruckig tutorial](https://docs.ruckig.com/tutorial.html): the native chain retains explicit position, velocity, acceleration, and jerk limits and validates inputs/outputs.
- [COPP/TOPP repository](https://github.com/TOPP-THU/copp): third-order path-parameterization routes were identified as future hybrid candidates; they were not silently substituted into the accepted chain.

## 3. Adversarial semantic probes

`D60_SEMANTIC_PROBES.json` records seven deterministic known-answer cases:

1. endpoint-safe middle collision;
2. nonlinear two-link FK path;
3. rapid long-link rotation with identical endpoint pose;
4. narrow grazing collision window missed by a coarse grid;
5. nonuniform `q(t)` with a wrong rigid-proxy contact time;
6. simultaneous nonlinear relative motion of two links;
7. endpoint-transform interpolation versus direct FK.

All `7/7` probes passed. The installed FCL known-answer executable also passed
forced crossing and known separation. The probes establish that endpoint
sampling and rigid endpoint transforms are insufficient semantics for exact
articulated `FK(q(t))` certification.

## 4. Strongest available FK(q(t)) certificate

The D54/Stage4F MoveIt2/FCL conservative certificate was rerun in an isolated
D60 directory with `initial_stride=1` and `max_depth=20` on both correctly
routed D59 finalists. It produced:

- 2/2 passing cases;
- 10/10 required-pair coverage, 21 pair universe, 7 collision links;
- 0 collision regions and 0 unresolved regions in this D60 conservative model
  run;
- minimum certified clearance lower bound `0.004214082104332431 m`;
- independent q/dq/ddq/jerk contract: finite, strictly increasing time,
  jerk bound respected, and bounded-jerk state-transition audit passed for
  both trajectories.

This remains a conservative model-based lower-bound certificate. It is not
renamed as theorem-level exact external articulated CCD, and its zero/zero
observation must not be detached from that scope.

## 5. Hybrid architectures and rejected routes

Two bounded native campaign families were tried. Each composed a geometry path
stage with native MoveIt2/Ruckig execution-form conversion:

### Existing 0.075 family

`global_0075`, `asymmetric_005v_010a`, and `local_0075_midpoints` all reached
native `PASS` for 12 cases but were rejected by the jerk gate, with 24 total
violations per candidate and worst ratios from `1.152` to `1.798`.

### 0.15 family

`global_015`, `asymmetric_010v_015a`, and `local_015_midpoints` all reached
native `PASS` for 12 cases but were rejected by the jerk gate, with 481–604
violations and worst ratio `2.212`. No candidate was eligible for geometry,
FK, articulated certification, replay, or promotion after that hard failure.

These are rejected shadow findings; no jerk threshold was relaxed and no case
was removed. The inherited D59 fixed-q dynamics-aware time search remains a
valid measured shadow route, but its selected result is equal to the protected
floor and therefore did not promote.

## 6. Adversarial/full-gate result

The protected D59 full-chain result remains `12/12 PASS`, including native
execution, FK/geometry/task checks, conservative articulated certification,
model dynamics, deterministic replay, and focused regression. D60 adds the
7-case semantic probe suite, FCL known-answer run, and the deeper FK(q(t))
certificate audit. No protected measurement or scientific artifact was
overwritten.

## 7. Promotion decision and remaining boundary

`NO_PROMOTION` is the correct rolling-floor decision. D60 did not produce a
new protected system-level net gain. It did produce stronger reproducible
semantic evidence and a validated conservative certificate run. The remaining
algorithmic frontier is an exact or stronger trajectory-aware articulated CCD
semantics, followed by a hybrid geometry-plus-retiming optimizer that remains
within the jerk contract and improves the whole-system Pareto position.

## Required authoritative fields

```text
D60_TASK_STATUS={d60_status['D60_TASK_STATUS']}
TAKEOVER_REORIENTATION_RESULT={d60_status['TAKEOVER_REORIENTATION_RESULT']}
STARTING_CANONICAL={d59['CURRENT_CANONICAL']}
STARTING_PROTECTED_FLOOR={d59['CURRENT_PROTECTED_FLOOR']}
FINAL_PROTECTED_FLOOR={d59['CURRENT_PROTECTED_FLOOR']}
FINAL_VERIFIED_CHAMPION={d59['FINAL_VERIFIED_CHAMPION']}
BEST_UNPROMOTED_SHADOW={d60_status['BEST_UNPROMOTED_SHADOW']}
CONTINUOUS_ARTICULATED_COLLISION_STATUS={d60_status['CONTINUOUS_ARTICULATED_COLLISION_STATUS']}
FK_QT_CERTIFICATE_STATUS={d60_status['FK_QT_CERTIFICATE_STATUS']}
HARD_GATES=PROTECTED_D59_PASS; D60_HYBRIDS_REJECTED_JERK
ADVERSARIAL_VALIDATION=PASS_7_OF_7_SEMANTIC; FCL_2_OF_2; D59_12_OF_12
SYSTEM_LEVEL_NET_GAIN={d60_status['SYSTEM_LEVEL_NET_GAIN']}
PROMOTION=NO_PROMOTION
MATERIAL_REGRESSION={d60_status['MATERIAL_REGRESSION']}
FAILED_OR_REJECTED_MAJOR_ROUTES=TESSERACT_UNAVAILABLE; FCL_NOT_EXACT_FK_QT; SIX_NATIVE_HYBRID_SHADOWS_JERK_REJECTED
SUCCESSFUL_HYBRID_ROUTES=VALIDATION_COMPOSITION; D59_FIXED_Q_DYNAMICS_AWARE_SHADOW
REMAINING_ALGORITHMIC_GAPS=EXACT_OR_STRONGER_FK_QT_CCD; JERK_FEASIBLE_HYBRID_NET_GAIN; AUTHORIZED_SOFTWARE_THRESHOLDS
FIRST_GENUINELY_UNFINISHED_ACTION={d60_status['FIRST_GENUINELY_UNFINISHED_ACTION']}
```
"""
    (OUT / "D60_FINAL_REPORT.md").write_text(report, encoding="utf-8")

    ledger = {
        "schema_version": "d60-defect-and-decision-ledger-v1",
        "scope": "D60 shadow-only algorithmic continuation",
        "entries": [
            {
                "classification": "TYPE_A_SHADOW_WRAPPER_DEFECT",
                "status": "FIXED_AND_RERUN",
                "finding": "WSL FCL output was decoded using the Windows GBK default; UTF-8 replacement decoding was added to the probe wrapper.",
                "evidence": "outputs/D60_SYSTEM_LEVEL_CLOSURE/semantic_probes/D60_SEMANTIC_PROBES.json",
            },
            {
                "classification": "TYPE_A_SHADOW_AUDIT_PARSER_DEFECT",
                "status": "FIXED_AND_RERUN",
                "finding": "Independent finite-state audit initially treated nested matrices as scalar vectors; matrix flattening was corrected and the audit passed.",
                "evidence": "outputs/D60_SYSTEM_LEVEL_CLOSURE/fk_qt_certificate/D60_FK_QT_CERTIFICATE_AUDIT.json",
            },
            {
                "classification": "SEMANTIC_DECISION",
                "status": "CONFIRMED",
                "finding": "Endpoint discrete checks, coarse sampling, and rigid endpoint-sweep APIs do not encode exact nonlinear articulated FK(q(t)) semantics.",
                "evidence": "outputs/D60_SYSTEM_LEVEL_CLOSURE/semantic_probes/D60_SEMANTIC_PROBES.json",
            },
            {
                "classification": "TYPE_B_SHADOW_ROUTE_WEAKNESS",
                "status": "MEASURED_AND_REJECTED_NOT_REPAIRED_IN_D60",
                "finding": "Six native geometry-plus-Ruckig hybrid shadows violated the configured 8 rad/s^3 jerk gate.",
                "affected_cases": "D47 12-case acceptance set",
                "evidence": "outputs/D60_SYSTEM_LEVEL_CLOSURE/hybrid_shadows/performance_shadow_report.json and outputs/D60_SYSTEM_LEVEL_CLOSURE/hybrid_shadows_v2/D60_HYBRID_SHADOWS.json",
                "stage4b_target": "design a jerk-feasible hybrid retiming/waypoint method before attempting any promotion",
            },
            {
                "classification": "PROMOTION_DECISION",
                "status": "NO_PROMOTION",
                "finding": "Protected D59 floor retained; D60 adds validated evidence but no non-regressive system-level net gain.",
                "evidence": "outputs/D60_SYSTEM_LEVEL_CLOSURE/D60_FINAL_STATUS.json",
            },
        ],
    }
    (OUT / "D60_DEFECT_AND_DECISION_LEDGER.json").write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": d60_status["D60_TASK_STATUS"], "output": str(OUT)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
