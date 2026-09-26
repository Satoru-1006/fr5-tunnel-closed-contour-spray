# D64 Project Freeze

Status: `SOFTWARE_SCOPE_FROZEN_WITH_EXPLICIT_P0_BOUNDARY`

D64 freezes the measured software scope, its reproducibility evidence, and its explicit nonclaims. It does not promote the D62 shadow and does not convert unresolved intervals or unavailable physical quantities into safety claims.

## Claim fence and D59-D64 lineage

The D59 zero-collision/zero-unresolved observation is retained as a historical fact, but it is scoped to two correctly routed candidates under D59's conservative FK-aware software model. It is superseded as the current continuous-collision authority by the later D61-D64 chain and must not be read as a complete exact external articulated `FK(q(t))` self-CCD proof.

- `D59`: historical scoped model observation; exact external articulated `FK(q(t))` self-CCD is `UNAVAILABLE_UNVERIFIED`.
- `D60`: historical conservative FK(q(t)) model and endpoint-FCL cross-check; not exact external articulated CCD.
- `D61`: explicit FK(q(t)) shadow with partial windows and unresolved coverage.
- `D62`: all native intervals evaluated, but unresolved intervals remain (therefore not a continuous-certificate PASS).
- `D63`: software evidence-chain validation; safety status remains `UNRESOLVED`.
- `D64`: freezes the measured evidence and the above boundaries; promotion remains `NO_PROMOTION`.

## Frozen floor

- Canonical: `outputs/D47_STAGE4B_ROLLING_CHAMPION_V1/PROMOTED/B3`
- Protected floor: `c4_clearance_adversarial0101_amp005_consistent_0025`
- Promotion: `NO_PROMOTION`
- Protected-path git status: `NOT_CLEAN_OR_UNTRACKED` (digest verification is separate).

## Frozen D62 boundary

- `auto0`: 3567 certified / 7004 evaluated, 3437 unresolved, 0 collision intervals, resource hit `False`.
- `auto1`: 3628 certified / 7029 evaluated, 3401 unresolved, 0 collision intervals, resource hit `False`.

The coarse route is sound as an outer-enclosure proof search and full interval coverage was evaluated. The unresolved set remains an explicit P0 boundary. The attempted full unresolved-set depth-2 refinement was interrupted before producing a valid result under the Python reference budget; it is not part of the scientific pass/fail result.

## Frozen-artifact manifest

`outputs/D64_PROJECT_FREEZE/D64_FROZEN_ARTIFACT_MANIFEST.json` is the single authority seal for 19 explicitly listed files (canonical identity, protected floor, D59-D64 state/reports, root claim-fence docs, and this checker). It records each path, byte size, and SHA-256 digest; it does not hash the repository or generated output trees recursively.

## Next authorized work

Stop optimization in this frozen scope. Paper preparation and reproduction are allowed. Any future exact articulated CCD or compiled interval/BVH backend must be a separately authorized Stage 4B shadow and must reproduce the known-answer, fail-closed, and protected regression contracts before it can affect the evidence chain.
