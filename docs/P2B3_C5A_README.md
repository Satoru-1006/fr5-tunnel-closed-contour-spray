# P2-B3-C5A — Registration Robustness Metric Closure

**Project:** FAIRINO FR5 tunnel continuous-spray research  
**Status:** `P2B3_C5A_STATUS=PASS`; measurement pipeline `PASS`  
**Scope:** frozen C1 post-Ruckig trajectory, 181-point ON-state open arch, and the C4 authoritative 181-object scene  
**Boundary campaign:** `NOT_RUN`

## Research question

Does the project now have a trustworthy, interpretable, reproducible evaluator for measuring robot–world signed model distance and process-geometry degradation under base/workpiece registration error, with case and waypoint localization?

**Answer:** Yes for this bounded model-based measurement contract. The evaluator passed all 24 closure gates on 25 deterministic cases. This establishes a measurement tool for a later boundary study. It does not establish a physical registration tolerance, a coating-quality limit, continuous collision safety, or hardware safety.

## Frozen authority and transform convention

- Robot/project: FAIRINO FR5 only.
- C1 nominal trajectory SHA-256: `be697bcb976cb69d349ee6c549364ba39041d9f0bda1b57d7f215129f8d5cbb3`.
- C4 authoritative scene: `outputs/p2b3_c4_authoritative_scene_manifest.json`, canonical SHA-256 `f2a73cfe736d8379d9b5149bbd69fc98cfba09004f519c1975e2084e80eb0669`; 181 ordered BOX objects.
- Registration is a left-composed transform in `base_link`: `T_base_entity_actual = DeltaT_base @ T_base_entity_nominal`. It is applied to the authoritative collision objects, C1 target TCP poses, and C1 surface normals.
- C1 joint positions `q(t)` and timestamps are copied unchanged for every case. Every case retained 181 waypoints and the C4 858-sample adaptive-discrete schedule.
- The fixed case set is zero; ±1 mm and ±5 mm on each translation axis; and ±0.1° and ±0.5° about each base-frame rotation axis (25 cases total).

## Measurement definitions

### Robot–world and self distance

The native MoveIt2 C++ evaluator uses the C4 PlanningScene and ACM, with separate `distanceRobot()` and `distanceSelf()` queries. Both use `DistanceRequest::GLOBAL`, signed distance, nearest points, and the unpadded `CollisionEnvFCL`. Robot–world results identify the robot link and world object; self results identify a robot-link pair. Signed distance `<= 0` is a sampled collision under the installed MoveIt/FCL semantics.

Collision checking uses `adaptive_discrete_interpolation` with a maximum 0.5° joint interpolation step. The distance minima are also over those sampled states. These are signed model distances at sampled states, not strict continuous collision detection, a continuous clearance guarantee, a calibrated physical clearance, or a safety margin. No low-clearance acceptance threshold was frozen; it remains `UNRESOLVED_THRESHOLD`.

The current MoveItPy `PlanningScene` binding exposes collision checks, while its `CollisionResult` exposes `collision` and `distance`; it does not expose a verified robot–world distance API with nearest robot/world pair and nearest points. C5A therefore uses the native C++ FCL path and does not relabel Python `CollisionResult.distance` as robot–world clearance.

The earlier C2 evaluator was traced to `cpp/stage3_h13_d41/stage3_h13_d41_native.cpp`, called through `src/p2a_axiswise_robustness._D41BatchEvaluator`. It used separate `distanceRobot()` and `distanceSelf()` queries, supplied the ACM, enabled signed distance, and retained pair names. It did not retain nearest points, did not explicitly request an unpadded environment, and reconstructed its scene from pose rows rather than directly loading C4's authoritative manifest. C5A directly loads the frozen C4 manifest and uses the unpadded native distance environment.

### Process geometry

- `position_error_m[i]`: Euclidean distance between fresh FK TCP position for the frozen joint state and the registration-transformed C1 target TCP position. The maximum and critical waypoint are recorded per case.
- `normal_error_rad[i]`: angle between fresh FK TCP local `+Z` and the C1 projected open-arch normal returned by the exact C1 `_project_to_polyline_with_normals` function. The source convention `tcp_points_to_wall=true` is retained; error is also summarized in degrees.
- `stand_off_error_m[i]`: signed projected stand-off error using C1's authenticated `STAND_OFF=0.260 m` and the same open-arch polyline projection. The maximum absolute error and critical waypoint are recorded.

These are geometric diagnostics. No coating deposition model or new acceptance threshold was introduced.

## Validation and results

- All **24/24 measurement gates passed**. The exact gate ledger is in `outputs/p2b3_c5a_result.json`.
- Fresh native C5A measurements completed 25 cases: **858 samples per case, 4,525 fresh FK rows**. Each native shard confirmed all seven expected FR5 collision-geometry links were loaded.
- A native synthetic FCL known-answer fixture passed: clear distance `0.200 m`, translated distance `0.199 m`, and overlap distance `-0.050 m` with collision reported and pair/nearest-point checks passing.
- Zero registration: zero robot–world and self-collision samples; minimum sampled robot–world distance `0.08004969620976321 m`; minimum sampled self distance `0.016622079818704765 m`. The C1 zero-registration position, projected-normal, and stand-off arrays reconcile to the frozen C1 measurements.
- Across the 25 cases, minimum sampled robot–world distance was `0.0750496962097632 m` for `translation_5mm_x_minus` (upperarm link / `horseshoe_wall_026`). Minimum sampled self distance was `0.016622079818704765 m` for `registration_zero` (wrist2 / forearm).
- Largest TCP position error: `0.005652672612207367 m`, case `rotation_0p5deg_x_minus`, waypoint 90. Largest projected-normal error: `4.065263969342169°`, case `translation_5mm_z_minus`, waypoint 13. Largest absolute projected stand-off error: `0.005055900657355117 m`, case `translation_5mm_x_plus`, waypoint 161.
- Metric sanity showed a `0.010000000000000009 m` spread in robot–world minima across cases, a `0.004999996304532556 m` translation-related position-metric change, and a `0.009965811757371169 rad` rotation-related normal-metric change. The `1e-9` informativeness epsilon is a numerical gate only, not a physical criterion.
- Focused regression suite: **61 passed, 0 failed** across C5A metrics, C3 SE(3) uncertainty, and C4 scene-contract tests.

The positive distance values above are bounded sampled model outputs. No safety interpretation is assigned to them.

## Gate, claim boundaries, and next stage

`P2B3_C5A_STATUS=PASS` means the evaluator met its declared measurement-closure gates. The separate frozen-system observation is `NO_DISCRETE_COLLISIONS_OBSERVED` for this finite case set.

The following remain explicitly unavailable or unrun:

- `REGISTRATION_BOUNDARY_CAMPAIGN=NOT_RUN`
- `PHYSICAL_REGISTRATION_BOUND=NOT_AVAILABLE`
- `PHYSICAL_NORMALIZED_MARGIN=NOT_AVAILABLE`
- `STRICT_CONTINUOUS_CCD=NOT_AVAILABLE`
- `HARDWARE_VALIDATED=NOT_RUN`
- `COATING_QUALITY_CERTIFIED=NO`

No registration tolerance, coating-quality threshold, or hardware conclusion is implied. The next eligible gate is **P2-B3-C5B — Six-Axis Signed Registration Boundary Characterization**, subject to its own authorization and frozen protocol. C5B was not started in this task.

## Reproduction

From the execution branch, run the focused tests:

```powershell
python -m pytest -q tests/test_p2b3_c5a_metrics.py tests/test_p2b3_c3_physical_uncertainty.py tests/test_p2b3_c4_scene_contract.py
```

Run the native replay under WSL using the authenticated FAIRINO source checkout and a dedicated scratch directory:

```bash
python3 /mnt/d/fr5-p2b3-c5a-registration-metric-closure-20261002/scripts/run_p2b3_c5a_registration_metrics.py \
  --scratch /mnt/d/fr5-p2b3-c5a-run6-20261002 \
  --fairino-source /mnt/d/robotfucker/external/frcobot_ros2 \
  --resume-scratch --timeout-seconds 900 --build-timeout-seconds 1200 \
  --output-json /mnt/d/fr5-p2b3-c5a-registration-metric-closure-20261002/outputs/p2b3_c5a_result.json
```

The completed replay reported `P2B3_C5A_STATUS=PASS`. C4's previously verified identity artifact was validated and reused via `--resume-scratch`; the C5A native evaluator rebuilt and freshly measured all 25 cases. Scratch logs and intermediate CSV/JSONL files are not part of the release artifacts.

## Git and delivery provenance

- Parent C4 final commit: `f1b3840db3573142050695cb5e29702f4fd7a7de`.
- Execution code commit: `14ff903b806878005c6ff307a09fc2e69e3b0665`.
- Artifact publish commit: `6fff346aa8c859a6edd1c690c1707d935ae1bfd4`.
- The final delivery SHA is recorded in the Drive `PROJECT_INDEX.md` receipt after exact remote readback. A commit cannot contain its own Git SHA; the index and final delivery report identify the final branch tip.
- `GITHUB_CI=NOT_AVAILABLE_NO_WORKFLOW_RUNS_OR_STATUS_CHECKS`: the repository exposes zero GitHub Actions workflows, and the inspected C4 reference has zero status entries and check runs. The final C5A SHA is checked after push.
