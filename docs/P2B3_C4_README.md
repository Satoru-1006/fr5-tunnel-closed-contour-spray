# FAIRINO FR5 — P2-B3-C4
## Authoritative Scene Equivalence & Registration-Margin Readiness

- **Status:** `PASS`
- **Project:** `FAIRINO_FR5`
- **Parent:** P2-B3-C3 final delivery `8a60e699951da62e4ca69fe28c5a201096f677b1`
- **Branch:** `codex/fr5-p2b3-c4-authoritative-scene-equivalence-20261002`
- **C1 nominal:** 181 waypoints; SHA-256 `be697bcb976cb69d349ee6c549364ba39041d9f0bda1b57d7f215129f8d5cbb3`
- **Artifacts:** `outputs/p2b3_c4_authoritative_scene_manifest.json`, `outputs/p2b3_c4_result.json`

## Finding: the C3 zero-collision conflict was a scene-source mismatch

The historical C3 nominal run recorded collisions at all 858 sampled states. Re-reading its frozen native runner and comparing its effective scene parameters with the C1 runner found five differences:

| Parameter | C3 evaluator | C1 authority used in C4 |
|---|---:|---:|
| `stand_off_m` | 0.18 | 0.26 |
| `tcp_points_to_wall` | `false` | `true` |
| `wall_thickness_m` | 0.025 | 0.04 |
| `y_thickness_m` | 0.08 | 1.10 |
| `segment_stride` | 4 | 1 |

The C3 result recorded 46 world objects; the C1-derived C4 scene contains 181 (180 wall boxes and the tunnel floor). Open-path, floor, and bottom-closure semantics were also checked against the C1 implementation. C4 now extracts the effective values from the frozen C1 runner and calls its shared wall builder instead of maintaining a second parameter set.

With C1's scene restored, Phase C reproduced the authoritative zero-registration result: 181/181 objects applied, 858 sampled states, zero full-scene collision samples, zero self-collision samples, and FK regression `PASS` with maximum position delta 0 m. The q trajectory SHA and values, timestamps, target poses, and surface normals were unchanged at identity. Scene comparison reported zero differences, including the active ACM fingerprint and runtime API surface.

This strongly attributes the C3 saturation conflict to scene mismatch. The five parameters were not isolated in a one-factor experiment, so C4 does not claim which individual difference caused each historical collision.

## Authoritative scene and collision diagnostics

The manifest uses `base_link`, the C1 `build_wall_collision_objects` builder, an open path, no bottom closure, an included floor at z = -0.20 m, and the C1 effective geometry values above. It records all 181 ordered object descriptions with canonical finite floats and normalized `xyzw` quaternions (`q` and `-q` canonicalize identically).

- Canonical manifest SHA-256: `f2a73cfe736d8379d9b5149bbd69fc98cfba09004f519c1975e2084e80eb0669`
- Active ACM: `AVAILABLE`; 35,721 ordered pairs across 189 robot/world names; 24 allowed pairs; fingerprint `02975b55c25b52d8bb7d77bdbaf05db1ef5f1d58278575fa4fb298e1a3f35661`.
- Padding/scale: `NOT_AVAILABLE`; the active PlanningScene and RobotModel Python bindings expose no verified padding or scale accessor.
- Sampling: `adaptive_discrete_interpolation`, C1 stride 1 and maximum joint interpolation step 0.5 degrees. This is not strict continuous collision detection.
- Phase C's minimum reported full-scene sampled model distance was 0.0166220798 m. It is not robot-world-pair-specific, calibrated, or a continuous/physical clearance guarantee. Robot-world distance, nearest pair/points, and penetration depth remain `NOT_AVAILABLE`.
- `CollisionResult.distance = -1.0` from the old colliding C3 record is rejected as a sentinel, not interpreted as a distance or penetration depth.

## Phase D: bounded propagation smoke

After the Phase C identity gate passed, C4 ran exactly 12 deterministic engineering cases: ±1 mm on each base translation axis and ±0.1 degrees on each base rotation axis. Every case retained the frozen 181-point q/timestamp sequence, applied all 181 scene objects, and checked 858 sampled states. Across all 10,296 case samples, full-scene collisions and self-collisions were both zero. Target poses and surface normals followed the requested transforms.

The aggregate correctly reads `NO_COLLISIONS_OBSERVED`. This bounded collision result does not rank directions, establish clearance, or constitute a registration margin. The smoke amplitudes are engineering propagation inputs, not physical uncertainty bounds.

- `ZERO_TRANSFORM_EQUIVALENCE=PASS`
- `AUTHORITATIVE_SCENE_EQUIVALENCE=PASS`
- `REGISTRATION_MARGIN_READINESS=READY` for a subsequent model-based six-axis signed registration boundary search.
- `PHYSICAL_REGISTRATION_MARGIN=NOT_RUN`
- `PHYSICAL_NORMALIZED_MARGIN=NOT_AVAILABLE_PHYSICAL_BOUNDS_MISSING`
- Physical registration distribution, calibrated TCP uncertainty, execution tracking uncertainty, and calibrated surface reconstruction uncertainty remain unavailable.

## Validation and provenance

Focused C4 scene-contract tests and C3 uncertainty/SE(3) regression tests: **52 passed, 0 failed**. They cover authoritative extraction, canonicalization, manifest drift (including ACM and runtime API surface), missing/extra objects, dimensions/poses/frames, identity gates, sentinel handling, fail-closed unavailable diagnostics, project guard, and synthetic collision-summary known answers.

The final execution-code commit is recorded in the result. Artifact-publish and final-delivery commits are recorded after publication in the existing FAIRINO FR5 `PROJECT_INDEX.md`, avoiding a self-referential result commit. GitHub CI is reported separately from local tests; any unavailable workflow/check state is not a local-test pass.

## Conclusion and next gate

C4 passes its scene-equivalence and propagation gates. The C3 zero-registration saturation is strongly explained by the five verified scene-parameter differences and the 46-versus-181 object counts; restoring C1 reproduced 858/858 collision-free samples. The frozen trajectory was not tuned, no margin search was run, and no Stage 3/C1/C2/C3 scientific baseline was edited. The next scoped step is the authorized six-axis signed registration boundary search, retaining model-only collision evidence and the physical-bound/clearance limitations above.
