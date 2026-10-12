# FAIRINO FR5 — P2-B3-C5B1 Step 2: Bounded Signed Registration Boundary Search

**Date:** 2026-10-12  
**Project:** FR5 tunnel continuous surface-processing trajectories; iER8 explicitly excluded.  
**Result classification:** `SEARCH_IMPLEMENTATION_PUBLISHED`; `ARCHIVED_C5A_SMALL_AMPLITUDE_EVIDENCE_REVIEWED`; `FRESH_NATIVE_FR5_PILOT=NOT_RUN`; `FRESH_NATIVE_FR5_12_RAY_SCAN=NOT_RUN`. No physical margin, safety, continuous collision, or deposition claim.

## Frozen source of truth

The new implementation branches from the audited Step 1 freeze commit `a72fef9100892ebbd649f9be518ebb6b2efa249e`, whose parent lineage contains the 2026-10-08 Hotfix commit `ad842009562351c8b4a587fd9cd42dbfb3d03cb0`. The repository's input-freeze script is retained. The C1 trajectory, 181 C4 collision objects, 181-point q(t)/timestamps, FCL C5A evaluator source, BASE_LEFT semantics, and previous results are not modified. All C5B1 generated results must go outside the checkout.

Canonical identities: C1 trajectory semantic SHA-256 `be697bcb976cb69d349ee6c549364ba39041d9f0bda1b57d7f215129f8d5cbb3`; C4 scene semantic SHA-256 `f2a73cfe736d8379d9b5149bbd69fc98cfba09004f519c1975e2084e80eb0669`. Semantic hashes are not raw Git blob hashes.

## Implementation

- `src/p2b3_c5b1_boundary.py` implements bounded 12-ray BASE_LEFT search: tx/ty/tz/rx/ry/rz, each ±. Per predicate statuses: PASS, FAIL, UNKNOWN.
- Coarse grid samples all adjacent intervals. Every *observed* PASS↔FAIL transition (including FAIL→PASS exits) is refined independently without assuming monotonicity. UNKNOWN endpoints, UNKNOWN midpoint, evaluation-budget exhaustion, and incomplete native data produce an unresolved interval instead of a false boundary.
- Four predicates remain separate: sampled robot-world collision, sampled self-collision, >6 mm C1 position engineering diagnostic, >10° C1 projected-normal engineering diagnostic. The latter two are not validated coating failure criteria. Stand-off is not classified as FAIL without a calibrated threshold.
- No transition within sampled domain is labeled `NO_FAILURE_OBSERVED_WITHIN_DOMAIN` only if no FAIL/UNKNOWN was observed; this is *not* proof of absence in unsampled interior. `first_observed_entry` is the smallest resolved *observed* PASS→FAIL bracket, never a globally certified minimum.
- `scripts/run_p2b3_c5b1_boundary.py` orchestrates four-ray pilot followed by all twelve rays. Default is `--backend native`: accepts a prebuilt, authenticated C5A C++ FCL evaluator, checks the frozen baseline, creates new external scratch, writes frozen q-only rows and transformed authoritative C4 scene CSV per amplitude, calls the native executable for each sample, reads native C5A sample distances/collision counts and fresh FK rows, then applies C5A process metrics and C1 frozen projector.
- A deliberate `--backend fixture` option supports synthetic known-answer testing only and is labeled `SYNTHETIC_TEST_NOT_FR5`. It must never appear as a measured FR5 margin.
- `tests/test_p2b3_c5b1_boundary.py` contains tests for 12 signed rays, multiple non-monotonic boundaries, bracketing refinement, UNKNOWN at coarse/midpoint, baseline FAIL, no-failure finite-domain semantics, budget exhaustion, and pilot→12-ray fixture layout. GitHub Actions run 38212910487 completed successfully: **17 passed in 0.06s**; the read-only freeze gate also passed 8 pinned file identities.

### Frozen protocol defaults (engineering search domains, not measured error bounds)

| Stage | Rays | Domain | Coarse sampling |
|---|---|---|---|
| Pilot | ±tx, ±rx (4) | up to 1 mm translation / 0.1° rotation | 8 intervals per ray |
| Full | ±tx, ±ty, ±tz, ±rx, ±ry, ±rz (12) | up to 20 mm translation / 2° rotation | 8 intervals per ray |

Refinement tolerances: translation 0.01 mm and rotation 0.001°. Budget: 80 requests per ray. All numbers are adjustable engineering algorithm parameters, **not certified tolerance bounds**. Coarse sampling does not exclude narrow failure islands or prove global first failure.

## Archival C5A evidence: small-amplitude and 12-direction endpoint cross-check

Google Drive source: `p2b3_c5a_result.json` at https://drive.google.com/file/d/1Ublv0xnXZVbqAWloIIhIDxn1JOe37nNv/view

This is an **older independently executed 25-case C5A native FCL campaign**, not a fresh C5B1 scan. It recorded baseline plus 12 signed directions at two amplitudes: translations 1/5 mm, rotations 0.1/0.5°. All 25 archived cases reported zero sampled robot-world and self-collision samples. The largest archived process position error was `0.005652672612207367 m`; the largest projected normal error was `4.065263969342169°`. All 25 sampled cases meet the 6 mm / 10° engineering diagnostics at these amplitudes.

Four historical pilot rays at smallest nonzero amplitude (not freshly executed):

| Ray | Amplitude | Minimum robot-world sampled model distance | Maximum TCP position error | Maximum normal error |
|---|---:|---:|---:|---:|
| tx− | 1 mm | 0.0790496962 m | about 1.0014 mm | about 1.1392° |
| tx+ | 1 mm | 0.0810496962 m | about 1.0004 mm | about 1.1384° |
| rx− | 0.1° | 0.0800496962 m | about 1.1306 mm | about 1.1104° |
| rx+ | 0.1° | 0.0800496962 m | about 1.1304 mm | about 1.1110° |

All twelve signed rays have the old two nonzero-amplitude endpoint observations. These points are **not** a coarse scan over the intervening interval and are **not** failure brackets. No global margin or infinite robustness inference is allowed. Historical numeric readback is independent of new algorithm-code execution.

## Acceptance and blockers

| Gate | Evidence | Status |
|---|---|---|
| Frozen authority remains unchanged | Git compare against Step 1 and GitHub Actions frozen input checker (8 pins) | PASS; no frozen input files changed |
| Search and native adapter | Source files written to GitHub | IMPLEMENTED; fresh ROS/MoveIt runtime replay NOT_RUN |
| Adversarial tests | GitHub Actions 38212910487, full job log and test summary | CI_PASS: 17 passed, 0 failed; no ROS/native in CI |
| Pilot historical four-ray reference | C5A Drive source read | ARCHIVAL_DATA_VERIFIED, not fresh pilot |
| 12-axis historical discrete endpoint reference | C5A Drive source read | ARCHIVAL_DATA_VERIFIED, not fresh 12-ray campaign |
| Native C5B1 pilot and full 12-ray campaign | Requires existing WSL2 Jazzy/MoveIt native build and sourced runtime | NOT_RUN |
| Finite signed first-failure margin | Requires actual sampled PASS↔FAIL fronts | NOT_AVAILABLE |
| Strict continuous self collision / physical bound / hardware / deposition | Not established | NOT_AVAILABLE |

Native FCL `distanceRobot()` and `distanceSelf()` are distinct; configured collision and signed distance may not be identical under padding, so contradictory evidence is marked UNKNOWN. See MoveIt2 CollisionEnvFCL API: https://moveit.picknik.ai/main/api/html/classcollision__detection_1_1CollisionEnvFCL.html

## Reproduction (from frozen C5B1 branch, outputs outside repo)

The following pure Python tests were executed successfully by GitHub Actions (17 passed). Independently rerun them in the FR5 runtime if needed:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q -p no:cacheprovider tests/test_p2b3_c5b1_boundary.py
```

Then build or reuse `p2b3_c5a_native` using the existing C5A documented WSL2/ROS 2 Jazzy/MoveIt2/FCL runtime. Source the same ROS/MoveIt install. Run:

```bash
python3 scripts/run_p2b3_c5b1_boundary.py \
  --backend native --mode staged \
  --native-binary /absolute/path/to/p2b3_c5a_native \
  --scratch /mnt/d/c5b1_step2_new_external_scratch \
  --output-json /mnt/d/c5b1_step2_result/p2b3_c5b1_boundary_result.json
```

A fresh scratch path and a new result path are required; no historical result is overwritten. `staged` performs a 4-ray bounded pilot, blocks further native expansion on an UNKNOWN pilot, and then attempts all 12 signed directions. For fixture-only contract inspection, run `--backend fixture --mode staged` with an external `--output-json`; its output must be labeled synthetic, never FR5 measurements.

**CI evidence:** https://github.com/Satoru-1006/fr5-tunnel-closed-contour-spray/actions/runs/38212910487 — freeze check 8/8 pinned blobs PASS and C5B1 pure pytest 17 passed. 

**Required before claiming STEP2_NATIVE_COMPLETE:** native FR5 environment validation, real native 4-ray pilot PASS/FAIL/UNKNOWN ledger, real 12-ray observations and brackets, clean source tree/recorded exact Git SHA, and result JSON uploaded and read back in Drive. Do not claim full C5B1 research completion on the basis of this code-only delivery.
