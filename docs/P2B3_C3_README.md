# P2-B3-C3 — Physical Uncertainty Semantics and SE(3) Registration Stress

- **Project:** FAIRINO FR5 tunnel complex-surface spraying
- **Stage status:** `INCOMPLETE_DELIVERY`
- **Parent branch:** `codex/fr5-p2b3-c2-robustness-transfer-20260928`
- **Parent commit:** `fff39145a1e9dcd59fb1d180fe2cf182444820fe`
- **Execution code commit:** `61cf00c661bc03c2edf2e251f97ebdd13349ad4f`
- **C1 nominal:** 181-point ON-state open-arch post-Ruckig path, SHA-256 `be697bcb976cb69d349ee6c549364ba39041d9f0bda1b57d7f215129f8d5cbb3`
- **Measurement scope:** synthetic known-answer tests plus a bounded native smoke of 13 cases; no C2 campaign replay

## Why C3 exists

C2 left the 18 base/workpiece transform and process perturbation directions `NOT_AVAILABLE` because its evaluator did not propagate a registration transform into the world scene. C3 adds explicit semantics and a real MoveIt2 PlanningScene world-object pose update while keeping the C1 joint trajectory and timestamps fixed.

## What the old C2 perturbations mean

C2 joint axes are deterministic joint-state offsets. They do not establish a measured FR5 tracking distribution or hardware failure probability. C2 `tcp_tcp_translation:*` and `tcp_tcp_rotation:*` axes alter the target pose specification used for geometric comparison. They are not physical TCP execution errors, calibration errors, or workpiece-registration errors. C3 preserves each historical axis id and raw CSV value, then adds a derived semantic classification.

## Registration definition and SE(3) convention

For each workpiece-attached entity, nominal `T_base_entity` maps the entity into `base_link`. The perturbation is

`DeltaT_base = [[Exp([phi_base]x), t_base], [0, 1]]`

and the actual pose is

`T_base_entity_actual = DeltaT_base @ T_base_entity_nominal`.

The 3-vector `t_base` is a direct translation in meters expressed in `base_link`; `phi_base` is a right-handed axis-angle rotation vector in radians expressed in `base_link`. Rotation is about the `base_link` origin. ROS quaternion order is `x,y,z,w`. The parameterization uses direct translation plus `Exp(phi)` and does not use the coupled translational coordinates of an SE(3) twist exponential.

The robot base, robot model, nominal `q`, and timestamps remain fixed. Workpiece collision-object poses, surface points/normals, process reference poses, and workpiece-attached frames move together. Registration does not trigger IK or retiming.

## Native scene propagation

Native status: `PASS`. Scene propagation tested: `YES`. The smoke uses the project `build_wall_collision_objects` geometry builder with the frozen 181-point target pose/normal inputs. It applies the requested delta to each real MoveIt collision-object pose in `base_link`, then re-runs the MoveIt2/FCL collision and distance query with the unchanged q path. Nonzero cases are `ENGINEERING_PROPAGATION_SMOKE_ONLY`.

Collision observations use `adaptive_discrete_interpolation` with a 0.5-degree maximum joint-space interpolation step. They are finite-sample software evidence, not strict continuous collision detection. Any MoveIt `CollisionResult.distance` value is reported as sampled model distance only; it is not a physical or continuous clearance guarantee.

The sampled path reported 11154 full-scene collisions over 11154 queries, with 0 self-collision queries. Collision response status is `SATURATED_ALL_CASES`. When the binary result is saturated, these smoke amplitudes cannot rank registration directions; keep the observation as model-only evidence and review the approximate wall geometry independently. This is not a hardware collision verdict.

## Physical uncertainty evidence and normalization

No FR5 measured base/workpiece registration distribution, calibrated TCP uncertainty, joint-tracking distribution, or surface-reconstruction uncertainty bound was found in the FR5 project records searched for C3. Literature and assumed stress magnitudes are not substituted for FR5 measurement. Current status is `NORMALIZED_MARGIN_STATUS = NOT_AVAILABLE_PHYSICAL_BOUNDS_MISSING`.

The uncertainty ontology is in `outputs/p2b3_c3_uncertainty_semantics.csv`; the complete machine-readable result and evidence inventory are in `outputs/p2b3_c3_result.json`. No normalized-margin CSV is emitted without a supported physical bound.

## Validation and limitations

- Synthetic known-answer tests: `PASS` (70 passed, 0 failed).
- Protected C2/P2-A regression tests: `PASS`.
- Native smoke: `PASS`; zero transform was compared with the saved C1 MoveIt FK trace.
- Physical calibrated uncertainty, hardware tracking, hardware validation, strict continuous CCD, and a calibrated surface reconstruction remain unavailable.
- No normalized physical robustness margin, failure probability, global robustness proof, coating-quality certification, or hardware safety certification is claimed.

## Next gate

Before a full registration-margin campaign, obtain a traceable FR5 base/workpiece registration bound or measured distribution, define its frame and measurement conditions, independently review the saturated collision observation and approximate wall geometry, review the C3 scene evaluator and frozen-input evidence, then authorize a separate bounded margin protocol. Keep the C1 trajectory and all C2 canonical results unchanged.
