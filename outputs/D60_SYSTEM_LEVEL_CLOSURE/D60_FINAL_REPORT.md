# D60 — System-level algorithmic closure, hybrid exploration, and non-regressive rolling improvement

## Authoritative result

- `D60_TASK_STATUS`: `PASS_FOR_MEASURED_OFFLINE_DOMAINS_NO_PROMOTION_REMAINING_ALGORITHMIC_GAPS`
- `TAKEOVER_REORIENTATION_RESULT`: `PASS_LIVE_D59_STATE_RECONCILED_NO_PROTECTED_ARTIFACT_MUTATION`
- Starting and final protected floor: `c4_clearance_adversarial0101_amp005_consistent_0025`
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
D60_TASK_STATUS=PASS_FOR_MEASURED_OFFLINE_DOMAINS_NO_PROMOTION_REMAINING_ALGORITHMIC_GAPS
TAKEOVER_REORIENTATION_RESULT=PASS_LIVE_D59_STATE_RECONCILED_NO_PROTECTED_ARTIFACT_MUTATION
STARTING_CANONICAL=outputs/D47_STAGE4B_ROLLING_CHAMPION_V1/PROMOTED/B3
STARTING_PROTECTED_FLOOR=c4_clearance_adversarial0101_amp005_consistent_0025
FINAL_PROTECTED_FLOOR=c4_clearance_adversarial0101_amp005_consistent_0025
FINAL_VERIFIED_CHAMPION=c4_clearance_adversarial0101_amp005_consistent_0025
BEST_UNPROMOTED_SHADOW=D59_TIMING025_PAIR_AUTO0_AND_PROVENANCE_CORRECTED_AUTO1_RETAINED; D60_NATIVE_HYBRIDS_REJECTED_BY_JERK
CONTINUOUS_ARTICULATED_COLLISION_STATUS=SCOPED_PASS_CONSERVATIVE_FK_QT_MODEL_FOR_2_CORRECTLY_ROUTED_FINALISTS; EXACT_EXTERNAL_ARTICULATED_BACKEND_NOT_AVAILABLE
FK_QT_CERTIFICATE_STATUS=HISTORICAL_SCOPED_PASS_D60_STRIDE1_DEEP_REFINEMENT_2_CASES_0_COLLISION_0_UNRESOLVED_CONSERVATIVE_MODEL_ONLY
HARD_GATES=PROTECTED_D59_PASS; D60_HYBRIDS_REJECTED_JERK
ADVERSARIAL_VALIDATION=PASS_7_OF_7_SEMANTIC; FCL_2_OF_2; D59_12_OF_12
SYSTEM_LEVEL_NET_GAIN=NO_NEW_PROTECTED_SYSTEM_NET_GAIN; D60 ADDED VALIDATED SEMANTIC AND FK_QT EVIDENCE
PROMOTION=NO_PROMOTION
MATERIAL_REGRESSION=NO_PROTECTED_REGRESSION; D60 SHADOW HYBRIDS HAVE MATERIAL JERK REGRESSION AND WERE REJECTED
FAILED_OR_REJECTED_MAJOR_ROUTES=TESSERACT_UNAVAILABLE; FCL_NOT_EXACT_FK_QT; SIX_NATIVE_HYBRID_SHADOWS_JERK_REJECTED
SUCCESSFUL_HYBRID_ROUTES=VALIDATION_COMPOSITION; D59_FIXED_Q_DYNAMICS_AWARE_SHADOW
REMAINING_ALGORITHMIC_GAPS=EXACT_OR_STRONGER_FK_QT_CCD; JERK_FEASIBLE_HYBRID_NET_GAIN; AUTHORIZED_SOFTWARE_THRESHOLDS
FIRST_GENUINELY_UNFINISHED_ACTION=Obtain or implement a trajectory-aware articulated CCD/certificate with explicit FK(q(t)) semantics, then attack jerk-feasible hybrid path optimization under the same rolling hard gates
```
