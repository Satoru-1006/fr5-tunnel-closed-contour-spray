# D61 Final Report — articulated continuous collision certificate shadow

## Result

D61 delivered an independent, explicit `FK(q(t))` interval-certificate
implementation and measured it against real FR5 trajectories in an isolated
shadow. The implementation is not promoted. The correct final state is
`PARTIALLY_EXECUTED_MEASURED_NO_PROMOTION`.

Claim fence: D59/D60 zero-collision/zero-unresolved observations are retained
as historical conservative-model results only. D61 does not inherit them as a
complete exact external articulated self-CCD proof; its own real-trajectory
windows are coverage-incomplete and unresolved.

The new certificate proves a precise statement: if every required link pair is
separated by a positive projection gap on a fixed candidate axis for the full
interval FK enclosure, the collision meshes are separated throughout that time
interval under the recorded URDF and jerk-bound assumptions. It never turns an
overlapping outer enclosure into `PASS`.

## What was implemented

1. URDF/SRDF parser with FR5 joint-order and ACM-pair accounting.
2. Native trajectory contract for `t`, `q`, `dq`, and `ddq` with finite and
   strictly increasing time checks.
3. Endpoint-Taylor joint interval propagation using the configured jerk bound.
4. Interval axis-angle FK propagation through the complete articulated chain.
5. STL vertex support bounds, primitive-shape bounds, and fixed candidate
   separating-axis tests.
6. Recursive interval bisection with explicit coverage/resource accounting.
7. Certificate JSON containing trajectory id, checked interval, pair set,
   minimum certified gap, conservativeness assumptions, failure witness, and
   verification result.

## Verification

- D61 known-answer suite: `8/8` passed.
- Protected D50/D54/D57 plus D61 regression: `20/20` passed.
- Safe high-curvature primitive trajectory: `PASS` with complete coverage.
- Endpoint collision primitive: `COLLISION_FOUND` with exact OBB witness.
- Endpoint-safe, middle-collision primitive: `UNRESOLVED`, never a false PASS.
- q-only trajectory: blocked because `dq`/`ddq` are required for the recorded
  continuous-time assumption.
- D59 auto0 first 50 intervals: `450` certified pair regions, `50`
  unresolved enclosure regions, zero collision witnesses.
- D59 corrected auto1 first 50 intervals: same counts, zero collision
  witnesses.

The real-trajectory windows are deliberately marked coverage-incomplete. They
are evidence of execution and of the current geometric resolution boundary,
not full-trajectory safety certificates.

## Route conclusions

- Route A and A-plus are executable and mathematically explicit, but the mesh
  enclosure remains too coarse for several near self-pairs without a sound BVH
  or convex-decomposition narrow phase.
- Route B is implemented as a Bernstein cubic helper, but native D59/D60 CSVs
  do not contain explicit Ruckig segment coefficients. Reconstructing a cubic
  from samples would change trajectory semantics, so it remains unavailable
  for native certification.
- Route C is implemented as conservative interval bisection and is useful for
  reducing configuration uncertainty, but it cannot resolve a genuine overlap
  of the chosen outer geometry bounds by itself.
- Route D was investigated. Existing D60 evidence keeps MoveIt/Bullet,
  Tesseract BulletCast, and FCL rigid endpoint sweeps semantically separate
  from this articulated FK certificate.

## Boundary and handoff

The strongest inherited D60 result remains a distinct endpoint-FCL-distance
plus jerk-envelope model with two full finalist passes. D61 adds a new
explicit-FK mesh-enclosure semantics but does not replace or promote that
result. The next Stage 4B target is a compiled sound BVH/convex-decomposition
support layer for the near pairs, followed by full-trajectory recomputation,
known-answer differential testing, and the same protected regression gates.

Canonical and protected floor are unchanged. Promotion is `NO_PROMOTION`.
