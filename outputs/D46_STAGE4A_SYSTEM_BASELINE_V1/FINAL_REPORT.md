# D46 — Stage 4A system-level baseline measurement

TASK_STATUS: PASS
STAGE4A_STATUS: PASS_WITH_EXPLICIT_COVERAGE_GAPS
D46_MEASUREMENT_PIPELINE_STATUS: PASS
STAGE4_BASELINE_ROBOT_PERFORMANCE_STATUS: MEASURED_WITH_FAILURES_AND_LIMITATIONS
STAGE3_CANONICAL_PRESERVED: YES
FINAL_CANONICAL_UPDATE: 480
TRAINING_PERFORMED: NO
OPTIMIZATION_PERFORMED: NO
CANONICAL_MODIFIED: NO

## Scope and identity

- Scope: `Stage 0/1 ON-state open-arch only`.
- Stage 3 release: `outputs/STAGE3_FINAL_RELEASE`; status `FROZEN_AND_CLOSED`.
- Canonical checkpoint: `outputs/stage3_h13_d35_permanent_champion/checkpoints/committed_update_480.pt`; SHA-256 `37645eea5aef8ce2fe7084c83f35429a7a13fbe3e9612eb84417ccce6cba9742`.
- Authenticated H1/H32: `7.0893288390132589e-05` / `0.018491006193811731`.
- D43/D44 shadows, legacy 720-point material, OFF states, reorientation, approach, retreat, GNN/PPO/LSTM/Transformer, and closed-contour transitions were excluded.

## Benchmark execution

- `STAGE4_SYSTEM_BENCHMARK_V1`: created and frozen before execution; `1000` total cases.
- Family counts: `{"ADVERSARIAL": 200, "BOUNDARY": 200, "COLLISION_SENSITIVE": 200, "NORMAL": 200, "PERTURBATION": 150, "REGRESSION": 50}`.
- Acceptance subset: `12` fixed cases.
- Native MoveIt2 PlanningScene/Bullet/FCL geometry and native MoveIt2 FK ran on all case trajectories. Collision evidence is labelled `adaptive_discrete_interpolation`; native two-state environment Bullet evidence is separately identified and is not used to claim continuous self-collision.

## Baseline scorecard highlights

- Terminal position error P50/P95/P99/MAX (m): `3.74510906258e-08` / `0.361456919043` / `0.467321766999` / `0.467507717436`.
- Terminal orientation error P50/P95/P99/MAX (rad): `1.84063306012` / `2.502785053` / `3.0308112712` / `3.0338313769`.
- TCP trajectory error P95/P99/MAX (m): `0.887517261085` / `0.99172517067` / `1.02126362297`.
- Max velocity/acceleration/jerk ratios: `0.0917918326391` / `0.307257525526` / `0.0356850591759`.
- Hard violations: joint-limit `0`, velocity `0`, acceleration `0`, jerk `0`, continuity `0`.
- Geometry: waypoint environment collision cases `214`, waypoint self-collision cases `203`, environment CCD failures `15687`, continuous self-collision `NOT_AVAILABLE`.
- Minimum environment/self clearance (m): `-0.592559661263` / `-0.0803068499291`; physical acceptance thresholds remain `UNRESOLVED_THRESHOLD`.
- Worst singularity: sigma_min `1.36356958749e-11`, condition number `149612347097`; threshold `UNRESOLVED_THRESHOLD`.
- Model-based required torque: `UNAVAILABLE`; no torque value was imputed from collision-free states.
- Ruckig validity: `49/49` applicable actual D41 post-Ruckig controls valid; rate `1`. Audit-only perturbation/stress cases are explicitly not relabelled as newly generated Ruckig trajectories.
- PRE_RUCKIG versus POST_RUCKIG comparison: `AVAILABLE` using `regression_0001` and `regression_0000`; see the machine-readable scorecard for metric deltas.

## Findings and Stage 4B handoff

- TOP_1_BOTTLENECK: `B1_NEAR_SINGULAR_JACOBIAN`.
- TOP_2_BOTTLENECK: `B2_COLLISION_SENSITIVE_ROBUSTNESS`.
- TOP_3_BOTTLENECK: `B3_LOW_GEOMETRIC_CLEARANCE_MARGIN`.
- These are measured findings and controlled stress results; D46 did not tune or repair the frozen robot system.
- Three Type-A measurement defects were found and fixed: the FK runtime `sdformat` library path, a fresh-replay self-clearance field-semantic mismatch, and an acceptance-set hard-gate conflict. All repairs were regression-tested; unsupported torque, continuous-self-collision, TCP calibration, and physical-limit claims remain explicit coverage gaps.

## Authoritative artifacts

- `FINAL_REPORT.json` and `FINAL_REPORT.md`
- `STAGE4_SYSTEM_BENCHMARK_V1.json` and `.md`
- `STAGE4_ACCEPTANCE_SET_V1.json`
- `STAGE4_BASELINE_SCORECARD_V1.json` and `.md`
- `STAGE4_FAILURE_TAXONOMY_V1.json` and `.md`
- `STAGE4_RISK_RANKED_BOTTLENECKS_V1.json` and `.md`
- `STAGE4_LIMIT_PROVENANCE_V1.json`
- `STAGE4_COLLISION_COVERAGE_MATRIX_V1.json`
- `STAGE4_REPRODUCIBILITY_REPORT_V1.json`
- `STAGE4_CASE_RESULTS.csv` / `STAGE4_CASE_RESULTS.jsonl`

The full machine-readable scorecard, taxonomy, risk ranking, native logs, and per-case evidence are authoritative within this D46 output directory.
