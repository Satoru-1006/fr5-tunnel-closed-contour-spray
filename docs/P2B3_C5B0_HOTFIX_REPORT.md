# FR5 P2-B3-C5B0 HOTFIX report

Date: 2026-10-08
Scope: FAIRINO FR5 only; frozen Stage 3/C4/C1 inputs were not modified. No C5B1 work was started.

## Decision

The measurement pipeline is PASS. The frozen baseline is reported, not optimized:

- historical C5B0 convergence rows are reclassified from the old unconditional PASS to INSUFFICIENT_EVIDENCE because the archived rows do not declare a distance tolerance;
- the historical raw distances, sample counts, pair, and segment are preserved;
- the bounded real FR5 85 -> 86 Bullet world-CCD probe is AVAILABLE and reports no collision;
- strict continuous self-collision, clearance, and hardware validation remain unavailable/null.

## A. Repaired defects

### A1. Discrete convergence

discrete_convergence_summary() now:

- requires at least three coarse-to-fine observations;
- requires an explicit positive distance tolerance;
- validates finite distances, monotone refinement, increasing sample counts, collision-status stability, pair/segment diagnostics, and raw numeric fields;
- returns CONVERGED, NOT_CONVERGED, or INSUFFICIENT_EVIDENCE;
- keeps pair and segment switches visible as independent diagnostics;
- never upgrades missing tolerance or incomplete evidence to PASS.

Known-answer and adversarial tests cover material distance change, coarse-free/fine-collision, pair switch, segment switch, missing tolerance, and incomplete records. The five archived representative C5A/C5B0 rows retain their raw values and are now INSUFFICIENT_EVIDENCE rather than silently passing.

### A2. Registration and SE(3)

Identity registration no longer bypasses validation. Every external 4x4 transform is checked before the identity shortcut for:

- finite values;
- exact homogeneous last row within tolerance;
- orthogonal rotation within tolerance;
- positive unit determinant;
- declared entity/convention and required workpiece transform.

Scale, mirror, shear, NaN, invalid last row, unknown convention, and missing-frame cases fail closed. No projection or “repair” of an invalid transform is performed.

### A3. FCL padding semantics

The C5A FCL known-answer fixture now checks clear, near, and overlap cases using both the padded collision environment and the unpadded distance environment. The expected result is:

padded_unpadded_collision_predicates_equal = true and padded_unpadded_signed_distances_equal = true

for this fixture, while preserving the actual implementation distinction: boolean environment collision uses the active padded environment; distance uses getCollisionEnvUnpadded() with the configured ACM. A zero distance or collision-free sample is not reported as positive clearance.

### A4. Measurement infrastructure repair

An early real-probe attempt exposed an incomplete runtime geometry installation. The probe initially had an unsafe “available” path unless the loaded FR5 collision geometry was checked. The probe now fails closed unless the exact seven-link FR5 collision-geometry set is present. After sourcing the existing FR5 MoveIt installation, the probe was rebuilt and rerun successfully. This is a tooling/environment gate repair, not a change to the robot baseline.

## B. Official algorithm/API reuse

The hotfix reuses the existing MoveIt2 interfaces rather than introducing a new collision algorithm:

- MoveIt2 PlanningScene::getCollisionEnv() and getCollisionEnvUnpadded() for the active padded boolean path and unpadded distance path: [PlanningScene API](https://github.com/moveit/moveit2/blob/main/moveit_core/planning_scene/include/moveit/planning_scene/planning_scene.hpp)
- MoveIt2 Bullet CollisionEnvBullet::checkRobotCollision(req, res, state1, state2, acm) for the one bounded real world-CCD check: [Bullet collision checker documentation](https://moveit.picknik.ai/main/doc/examples/bullet_collision_checker/bullet_collision_checker.html) and [CollisionEnvBullet API](https://moveit.picknik.ai/main/api/html/classcollision__detection_1_1CollisionEnvBullet.html)
- The finite interpolation is labelled adaptive_discrete_interpolation; it is not strict continuous collision detection. The distinction is consistent with [OMPL DiscreteMotionValidator](https://ompl.kavrakilab.org/core/classompl_1_1base_1_1DiscreteMotionValidator.html) and [Tesseract collision checking](https://tesseract-robotics.github.io/tesseract/collision.html).

The real Bullet overload used here is robot-world only. MoveIt’s documented Bullet path does not provide strict self-CCD for this probe, so strict_self_ccd is JSON null with status NOT_AVAILABLE.

## C. Focused acceptance evidence

| Check | Evidence | Result |
|---|---|---|
| Python focused regression | python -m pytest -q tests/test_p2b3_c5b0_registration_spec.py tests/test_p2b3_c5a_metrics.py tests/test_p2b3_c3_physical_uncertainty.py tests/test_p2b3_c4_scene_contract.py | 80 passed |
| Bullet synthetic known-answer | endpoint free, hidden sweep crossing detected, clear transition remains clear | PASS |
| FCL padding known-answer | clear/near/overlap; padded/unpadded predicates and signed distances agree for the fixture | PASS |
| C4 scene identity | canonical manifest_sha256() recompute; 181 unique finite positive boxes | PASS, canonical SHA f2a73cfe736d8379d9b5149bbd69fc98cfba09004f519c1975e2084e80eb0669 |
| C1 identity | 181 waypoints; strictly increasing timestamps; q/timestamp values retained | PASS, q SHA be697bcb976cb69d349ee6c549364ba39041d9f0bda1b57d7f215129f8d5cbb3 |
| Real FR5 Bullet probe | C4 scene, FR5 model, C1 post-Ruckig waypoints 85 -> 86, 18 adaptive samples plus the real two-state API | AVAILABLE; world CCD false, all sampled collision counts 0 |
| Hardware / calibrated clearance | no hardware or calibrated clearance backend in scope | NOT_AVAILABLE / null |

The real probe used:

- scene: outputs/p2b3_c4_authoritative_scene_manifest.json, canonical C4 identity above;
- object count: 181;
- segment: waypoint 85 -> 86;
- times: 240.69927324299999 s -> 243.68350353700001 s;
- adaptive interpolation: 17 steps, 18 samples, 0 collision samples;
- Bullet result: robot_world_ccd_collision=false, robot_world_ccd_pairs=[];
- request semantics: environment padding true, self padding false, ACM applied.

The raw machine-readable record is [p2b3_c5b0_hotfix_result.json](../outputs/p2b3_c5b0_hotfix_result.json).

## D. Frozen-result handling

The prior Drive artifact p2b3_c5b0_result.json (ID 1LB7S0NRNH-MHUNnlGLJFlFLMgOkxYrEi) was read for comparison and was not overwritten. Its old unconditional convergence status is not reused as a new acceptance claim. The hotfix preserves the underlying historical numbers and changes only the interpretation produced by the repaired evaluator.

The C4 documentation hash is a canonical semantic hash, not a raw line-ending-sensitive file hash. Recomputing manifest_sha256() from the clean hotfix worktree returns the documented canonical value; the raw checkout/Drive byte hashes differ only because of serialization/newline representation and are not substituted for the canonical identity. The old five-case record reported a maximum absolute 0.5-degree to 0.1-degree change of 0.00000238303284169 m; without a declared tolerance, that numeric trend is retained but not promoted to CONVERGED.

## E. Delivery

- execution commit: 9100577f8bf2a6c8ba32be268c9bf477a4ab56a3
- artifact publish commit: recorded after this report/result commit
- branch: codex/p2b3-c5b0-hotfix
- Drive destination: existing C5B0 CORE folder; new files are uploaded without replacing the old result.

No Stage 4B optimization and no C5B1 transition is included in this hotfix.
