# P2-B3-C5B1 Step 1 — Frozen FR5 Baseline Acceptance

Date: 2026-10-12
Project: FAIRINO FR5 tunnel continuous surface spraying. iER8 is explicitly out of scope.
Status: FROZEN_IDENTITY_AUDITED; C5B1_BOUNDARY_CAMPAIGN=NOT_RUN; CURRENT_SESSION_80_TEST_REPLAY=NOT_RUN.

## Authority and release boundary

- Source GitHub repository: https://github.com/Satoru-1006/fr5-tunnel-closed-contour-spray
- Exact source commit (October 8 Hotfix): ad842009562351c8b4a587fd9cd42dbfb3d03cb0.
- Source Hotfix branch: codex/p2b3-c5b0-hotfix.
- Separate freeze branch: codex/fr5-p2b3-c5b1-baseline-freeze-20261012. No main merge or historical input replacement is authorized.
- Frozen C1 181-point ON-state post-Ruckig path, q(t), times and derivatives must not be optimized, clipped, retimed, regenerated or overwritten.
- Frozen C4 181-object authoritative scene, ACM semantics and collision geometry must not be reconstructed from a conflicting C3/default scene.
- Primary BASE_LEFT means T_base_entity_actual = DeltaT_base @ T_base_entity_nominal; translation/rotation in base_link, rotation about the base origin; q(t) and timestamps are unchanged.

## Exact byte-identity pins (Git blob SHA-1, not physical-model certification)

| File | Git blob SHA-1 |
| --- | --- |
| outputs/p2b3_c2_c1_nominal_post_ruckig.csv | 987bc55f8d7c7b7543c64b24673c45cac5b10cf4 |
| outputs/p2b3_c4_authoritative_scene_manifest.json | 0a127c7a2042ac81825fe1a5adbeee99652ca0a1 |
| src/p2b3_c5b0_registration_spec.py | 9346789aa2c4e06c572d86c44a0e9e4d279238c6 |
| tests/test_p2b3_c5b0_registration_spec.py | 6fc475059b7b85162aee623cebdd4ca3d5ce82eb |
| tests/test_p2b3_c5a_metrics.py | b34a3005e69d67278ee864755315e7892e3be913 |
| tests/test_p2b3_c3_physical_uncertainty.py | 7f292c46152a0a27164ef33e05a9b76a1b34f962 |
| tests/test_p2b3_c4_scene_contract.py | f09ee4c5a9aaa371553ba73612b8bf919e2b594c |
| outputs/p2b3_c5b0_hotfix_result.json | 8f8411297a7dbcb119aec059603b5ae6924ad445 |

These identities were fetched and cross-checked from the GitHub connector. Both C1 and C4 blobs match the historical C5A, C5B0 pre-Hotfix, and October 8 Hotfix refs. This does not turn a Git branch into an immutable protected ref.

## Semantic reference identifiers and structural checks

- C1 frozen nominal trajectory semantic SHA-256 recorded by C1/C4: be697bcb976cb69d349ee6c549364ba39041d9f0bda1b57d7f215129f8d5cbb3.
- C4 canonical scene-manifest semantic SHA-256 from C4 and Hotfix evidence: f2a73cfe736d8379d9b5149bbd69fc98cfba09004f519c1975e2084e80eb0669.
- C1 fully read as 181 data rows, 25 columns, strictly increasing time; 0.0 to 511.878236884 s.
- C4 fully read as 181 unique ordered BOX objects (180 horseshoe wall segments plus tunnel_floor) in base_link.
- In October 8 source, BASE_FRAME = base_link, PRIMARY_PARAMETERIZATION = BASE_LEFT, with rigid transform propagated by induced @ entity.
- C4 canonical SHA-256 is a semantic hash recorded by the project's canonicalization contract; do not compare it to the raw file hash or Git blob identity as though these were the same quantity.

## Historical regression and verification classification

- October 8 Hotfix machine-readable result declares 80 passed for: python -m pytest -q tests/test_p2b3_c5b0_registration_spec.py tests/test_p2b3_c5a_metrics.py tests/test_p2b3_c3_physical_uncertainty.py tests/test_p2b3_c4_scene_contract.py.
- The exact four historical test files are pinned above. The Hotfix receipt also records native Bullet fixture PASS and a bounded FR5 robot-world CCD probe on waypoint 85->86 without detected collision.
- This freezing task independently reviewed the test-file identities and the archived execution receipt; it has NOT rerun the 80 Python tests or the ROS2/MoveIt native fixture in the current session. Treat historical regression as HISTORICAL_PASS, current-session replay as NOT_RUN.
- Historical C5B0 convergence rows are INSUFFICIENT_EVIDENCE without a declared positive distance tolerance, even though the raw model-distance samples were preserved.
- Strict self continuous collision detection, calibrated physical clearance, validated registration error distributions, physical process/coating thresholds and hardware safety remain NOT_AVAILABLE / NOT_RUN.

## Read-only fail-closed guard and execution

Script: scripts/verify_p2b3_c5b1_frozen_baseline.py.
Scope: checks eight byte-identical Git blobs, C1 rows and timestamps, C4 object/frame geometry, BASE_LEFT constants and composition, plus the archived 80-test receipt. It changes no source input and writes no result files.

Static identity gate (requires only Python 3.10+):

    python scripts/verify_p2b3_c5b1_frozen_baseline.py

Optional fresh 80-test replay (requires the original test dependencies and environment):

    python scripts/verify_p2b3_c5b1_frozen_baseline.py --run-regression

Any mismatch fails closed. --run-regression demands a passing pytest exit and the exact '80 passed' summary; it must not be reported as independently performed until run.

## Acceptance disposition and next step

FREEZE_INPUT_CONTENT_IDENTITIES=VERIFIED_BY_REMOTE_READBACK.
FROZEN_C1_C4_OBJECT_AND_WAYPOINT_COUNTS=VERIFIED.
PRIMARY_BASE_LEFT_SEMANTICS=VERIFIED_IN_HOTFIX_SOURCE.
HISTORICAL_80_TEST_RECORD=VERIFIED; CURRENT_SESSION_80_TEST_REPLAY=NOT_RUN.
C5B1_REGISTRATION_BOUNDARY_SEARCH=NOT_RUN; HARDWARE_VALIDATION=NOT_RUN.

Only after a fresh environment-capable baseline regression and recorded outcome should the separate bounded C5B1 signed-ray boundary campaign begin. Any future experiments must import the frozen artifacts rather than silently reconstructing or modifying them.

## Source receipts

- GitHub October 8 Hotfix result: https://github.com/Satoru-1006/fr5-tunnel-closed-contour-spray/blob/codex/p2b3-c5b0-hotfix/outputs/p2b3_c5b0_hotfix_result.json
- Drive October 8 Hotfix report: https://drive.google.com/file/d/1h2nRlBoOhah_U_3jlvj5JmCL0BiMKO8I/view
- Drive October 8 Hotfix result: https://drive.google.com/file/d/1kXaRvD5sgs1KOcHZ4Q6xuMuRnMSfhvT7/view
- Drive C5B0 specification: https://drive.google.com/file/d/12RZBLXBomobxk6pKJmom803GN2UVtHtr/view