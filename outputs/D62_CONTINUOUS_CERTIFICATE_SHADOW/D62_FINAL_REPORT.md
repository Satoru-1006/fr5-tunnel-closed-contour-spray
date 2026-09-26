# D62 Continuous Certification Coverage Upgrade — Final Shadow Report

Status: `PARTIALLY_EXECUTED_MEASURED_NO_PROMOTION`

D62 completed the coarse sound triangle-preserving BVH/OBB pass over every native trajectory interval for both D59 finalists. It did not obtain full continuous certification: unresolved remains unresolved, and the bounded full-set refinement was stopped before producing a valid result because the Python reference cost became hour-scale.

Claim fence: D59/D60 zero-collision/zero-unresolved observations remain
historical, scoped conservative-model results. D62's full interval accounting
supersedes those PASS-like labels for current coverage: the nonzero unresolved
sets below are not a continuous-certificate PASS and do not establish exact
external articulated `FK(q(t))` self-CCD.

## Coverage

| Case | Full intervals evaluated | Certified | Unresolved | Collision | Resource hit | Certified fraction |
|---|---:|---:|---:|---:|---:|---:|
| auto0 | 7004/7004 | 3567 | 3437 | 0 | False | 0.5093 |
| auto1 | 7029/7029 | 3628 | 3401 | 0 | False | 0.5161 |

No collision was found by this conservative shadow predicate, but that is not equivalent to a full PASS when unresolved regions remain. `minimum_clearance` and `minimum_certified_clearance` remain unavailable; projection gaps are not physical clearance.

## Route outcome

- Route A: executed a sound BVH whose leaves contain complete STL triangles; PCA OBBs tighten enclosures. No unverified convex-decomposition backend was substituted.
- Route B: adaptive time subdivision passed a 445/500 auto0 hard-case probe at depth 4; full unresolved-set depth 2 was interrupted before a valid output. This is a compute boundary, not a PASS.
- Route C: coarse BVH followed by interval-indexed refinement and fail-closed merge is implemented. The merge cannot erase collision, resource, or missing coverage.

## D63/D64 boundary

D59 motion, native execution, Profile.j, model dynamics, deterministic replay, FCL rigid endpoint cross-check, and MoveIt robot-world checks remain valid measured software evidence. D62 leaves the exact articulated continuous-certification P0 unresolved, so D64 can freeze the measured software evidence and limitations, but cannot claim physical safety closure or theorem-level full trajectory certification.

Promotion: `NO_PROMOTION`; canonical and protected floor unchanged.
