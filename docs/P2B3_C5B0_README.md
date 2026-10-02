# P2-B3-C5B0 — Registration Robustness Specification Closure

**Project:** FAIRINO FR5 tunnel continuous-spray research only  
**Status:** `P2B3_C5B0_STATUS=PASS`  
**Readiness:** `P2B3_C5B_READINESS=READY` for a bounded model-based C5B study  
**Boundary campaign:** `NOT_RUN`

This closure freezes the mathematical meaning of the next registration study. It does not search a six-axis boundary, create a physical tolerance, alter the Stage 3 trajectory, or validate hardware.

## Authority and identity

C5A was independently read from the remote branch `codex/fr5-p2b3-c5a-registration-metric-closure-20261002`; the local and remote tip matched:

`a00083264c2f44ef58cd25eafc7550c77b4ff8f6`

The C5A result JSON was read back from Drive file `1Ublv0xnXZVbqAWloIIhIDxn1JOe37nNv`. Its status and measurement pipeline were both `PASS`. The frozen inputs remain:

- C4 parent: `f1b3840db3573142050695cb5e29702f4fd7a7de`
- C1 nominal post-Ruckig SHA-256: `be697bcb976cb69d349ee6c549364ba39041d9f0bda1b57d7f215129f8d5cbb3`
- C4 canonical scene-manifest SHA-256: `f2a73cfe736d8379d9b5149bbd69fc98cfba09004f519c1975e2084e80eb0669`
- C4 world: 181 ordered `BOX` objects in `base_link`
- FAIRINO vendor commit: `60755d44d521a5ad6bee8494cc19522f8801aa20`

C5A remains valid for its declared contract: 25 deterministic cases, 858 adaptive-discrete samples per case, native MoveIt2/FCL signed distance, and no physical or strict-CCD claim.

## Frozen registration semantics

### `BASE_LEFT` — primary

The C5A convention remains primary:

```text
T_base_entity_actual = DeltaT_base_left @ T_base_entity_nominal
```

Translation is expressed in `base_link` metres; the right-handed rotation vector is expressed in `base_link` radians; the rotation origin is the `base_link` origin; composition is on the left. One rigid transform is propagated to the C4 collision objects, target TCP poses, and surface normals. The frozen `q(t)` and timestamps are copied unchanged.

This is a registration parameterization, not a local fixture-spin parameterization. For `rotation_0p5deg_x_minus`, the critical target at waypoint 90 has radius `0.6477447891692404 m` from the base origin. With `theta=0.008726646259971648 rad`, the first-order displacement is `0.005652639641819875 m`, the exact rigid displacement is `0.0056525806950902665 m`, and the C5A measured TCP position degradation is `0.005652672612207367 m`. The same critical waypoint and order of magnitude agree. This is the expected lever-arm consequence of rotating about `base_link`.

### `WORKPIECE_LOCAL` — comparison / sensitivity only

The comparison convention is:

```text
T_base_workpiece_actual = T_base_workpiece_nominal @ Exp(xi_workpiece^)
```

All workpiece-attached collision objects, target poses, and normals use the same rigid transform. The physical fixture frame and origin are not authenticated: `WORKPIECE_ORIGIN_STATUS=NOT_AVAILABLE`.

For reproducibility only, C5B0 uses the centroid of the 181 authoritative C4 object poses as an `ENGINEERING_REFERENCE_ONLY` origin:

`[0.0015304433462192122, -0.09999999999999994, 0.4788174535152478] m`

It must not be called a physical fixture origin. Identity is exact, pure translation and pure rotation have known-answer tests, and the implementation verifies the SE(3) Adjoint relation. No q or timestamp is changed.

## Failure predicates and margins

| ID | Definition | Status |
|---|---|---|
| F1 robot-world collision | `minimum_signed_robot_world_distance <= 0` | Available model-based predicate from native FCL |
| F2 self collision | `minimum_signed_self_distance <= 0` | Available model-based predicate from native FCL |
| F3 low clearance | threshold crossing | `LOW_CLEARANCE_THRESHOLD=UNRESOLVED`; predicate not available |
| F4 process position | configured 6 mm gate | `ENGINEERING_DIAGNOSTIC_THRESHOLD`, not physical coating evidence |
| F5 process normal | configured 10° gate | `ENGINEERING_DIAGNOSTIC_THRESHOLD`, not physical coating evidence |
| F6 stand-off | physical process threshold | `NOT_AVAILABLE` |
| F7 coating quality | deposition/quality failure | `NOT_AVAILABLE` |

The eligible C5B margin types are therefore:

- `rho_collision_axis`: model collision boundary along a declared perturbation ray;
- `rho_position_diag` / `rho_normal_diag`: explicitly engineering-diagnostic boundaries;
- `PHYSICAL_REGISTRATION_MARGIN=NOT_AVAILABLE` until measured, authenticated manufacturer, validated calibration, or validated process evidence exists.

## Collision backends

FCL remains the primary scalar backend because it supplies signed distance, nearest pair, nearest points, and sampled critical-state localization. All finite trajectory results are labelled `adaptive_discrete_interpolation`; they are not strict continuous collision detection.

The C5A minimum sampled robot-world distance over its 25 cases was `0.0750496962097632 m` for `translation_5mm_x_minus`, pair `upperarm_link / horseshoe_wall_026`. The zero-registration minimum was `0.08004969620976321 m`. These are sampled model distances, not clearance or safety guarantees.

MoveIt2 Jazzy Bullet headers and libraries are installed. The inspected Bullet collision environment exposes a robot-versus-static-world continuous path (`BulletCastBVHManager` / `checkRobotCollisionHelperCCD`), while the inspected utility documents that continuous-to-continuous checking is unsupported. A direct Bullet synthetic fixture passed:

- state A free;
- state B free;
- crossing transition detected by `convexSweepTest`;
- clear transition not detected;
- `STRICT_SELF_CCD=NOT_AVAILABLE`.

The fixture is capability evidence, not a claim that the FR5 C5B boundary has already been replayed through Bullet: `FR5_BOUNDARY_REPLAY=NOT_RUN_C5B0`.

## Discrete convergence

The native C5A FCL evaluator was run in shadow copies at 0.25° without changing C5A, and at 0.1° for five representative cases: zero, the C5A minimum-clearance case, the maximum-position case, the maximum-normal case, and the maximum-stand-off family. The 0.5° values are the C5A authority.

For all five cases, the nearest pair remained `upperarm_link / horseshoe_wall_026`, the critical segment remained `85->86`, and robot-world collision samples remained zero. The largest absolute change from 0.5° to 0.1° was `0.00000238303284169 m`. This closes sampled-resolution bookkeeping only; it does not promote the result to CCD.

## Bounded parameterization comparison

Four small probes (`±1 mm` and `±0.1°` about x) were evaluated using the frozen q(t), C4 geometry, and native FCL. The translation probes produced the same values under both conventions for the engineering centroid because its reference orientation is identity. Rotation probes differed: at `+0.1°`, maximum process position error was `0.0011304277400754994 m` for `BASE_LEFT` versus `0.000701555101729861 m` for `WORKPIECE_LOCAL`; maximum projected stand-off error was `0.0002029610318454722 m` versus `0.00007757451238321611 m`. This confirms that parameterization changes the numerical interpretation.

`BASE_LEFT` remains primary because it is the authenticated C5A authority and preserves result comparability. `WORKPIECE_LOCAL` is retained as a sensitivity study until a physical fixture frame is authenticated.

## Validation

Focused regression suite:

```text
69 passed, 0 failed
```

It covers C5B0 semantics, C5A metrics, C3 uncertainty semantics, and the C4 scene contract. The C5B0 tests include identity equivalence, pure translation, rotation about a declared origin, Adjoint consistency, normal normalization, object ID/dimension preservation, q/timestamp invariants, fail-closed taxonomy, and convergence bookkeeping.

## Readiness and claim boundary

`P2B3_C5B_READINESS=READY` means the next study may measure:

- model collision boundaries;
- engineering diagnostic boundaries;
- process metric response surfaces.

It does not mean that a physical registration tolerance, coating-quality limit, hardware-safe bound, continuous collision guarantee, or strict self CCD is available. Those remain `NOT_AVAILABLE`, `UNRESOLVED`, or `NOT_RUN` as recorded in the Drive-only machine-readable result [`p2b3_c5b0_result.json`](https://drive.google.com/file/d/1LB7S0NRNH-MHUNnlGLJFlFLMgOkxYrEi/view?usp=drivesdk).

The GitHub checkout intentionally keeps only the minimal reproducible surface; the file-level inventory is in [P2B3_C5B0_FILE_MANIFEST.md](P2B3_C5B0_FILE_MANIFEST.md). The complete process archive, drafts, build logs, and prior C5A source snapshot are stored in the linked Drive project folder.

## Reproduction

From the repository root:

```powershell
python -m pytest -q tests/test_p2b3_c5b0_registration_spec.py tests/test_p2b3_c5a_metrics.py tests/test_p2b3_c3_physical_uncertainty.py tests/test_p2b3_c4_scene_contract.py
```

The synthetic Bullet fixture requires the installed Bullet development package and is built outside the repository:

```bash
cmake -S cpp/p2b3_c5b0_ccd -B /mnt/d/c5b0scratch/bullet_build -DCMAKE_BUILD_TYPE=Release
cmake --build /mnt/d/c5b0scratch/bullet_build --parallel 2
/mnt/d/c5b0scratch/bullet_build/p2b3_c5b0_bullet_fixture
```

The authoritative C5A replay remains the remote branch and Drive result cited above. C5B0 convergence shadow builds and logs stay outside Git; no build, install, cache, or duplicated JSONL output is committed.

The complete machine-readable closure is stored in Drive rather than the local checkout. The required next decision is authorization of a bounded model-based C5B campaign using `BASE_LEFT`; no physical threshold may be inferred from its output.
