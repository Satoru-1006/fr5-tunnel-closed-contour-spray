# D56 final report — Stage 4B software closure

## Overall status

`TASK_STATUS=PASS` for the Stage 0/1 ON-state open-arch software scope.

The protected Stage 3 / D45 state was not modified. Canonical promotion remains `NO`; the final result is a verified shadow champion.

| Required key | Result |
|---|---|
| `STARTING_CHAMPION` | `stateful_local_g1_w05_0025` |
| `FINAL_CHAMPION` | `c4_clearance_adversarial0101_amp005_consistent_0025` |
| `CANONICAL_PROMOTION` | `NO` |
| `SYSTEM_LEVEL_NET_GAIN` | `YES_ON_MEASURED_SOFTWARE_CHANNELS` |
| `REMAINING_SOFTWARE_ADDRESSABLE_D56_DEFECTS` | `0` |

## Phase A — Ruckig, jerk and lifecycle

- Runtime limits are explicit: velocity `3.15/3.2 rad/s`, acceleration `0.7 rad/s²`, jerk `8 rad/s³`; the jerk value is the active native Ruckig limit, not a configuration-only assumption.
- Repaired contract: `7004/7004` strict current/target validations and `7004/7004` calculations succeeded; maximum analytic `Profile.j` is `8 rad/s³`.
- Root cause: unchanged endpoint derivatives contained jerk-limited unreachable transitions; duration extension could not repair the boundary state. D56 repaired the derivative boundary and consistent timebase before native Ruckig execution.
- The accepted 12-case native post-Ruckig run completed cleanly with no `-11`; the earlier crash was reproduced, localized to MoveIt object teardown, and isolated from the accepted custom-overlay path.

Evidence: `runtime_limits/D56_RUNTIME_LIMITS.json`, `contract_repaired_0025.json`, `d56_final_native_runtime/execution_form_summary.json`.

## Phase B — Full Jacobian and sensitivity

The native full-J capture covers `96,711` state rows over all 12 cases. Consistency checks passed:

- `max ||Jv_min - sigma_min u_min|| = 1.77e-15`;
- maximum U/V normalization errors are below `2.67e-15`;
- translational/rotational split SVD residual is below `3.56e-15`;
- the controlling neighborhood tracks continuously over ±20 samples.

The final worst singularity is `sigma_min=1.6707154306500308e-05`, condition number `111175.072829886`, compared with the D55 starting values `1.9116869978233232e-06` and `971628.7757918573`. A task-weighted Jacobian was not invented because no physically justified translation/rotation weight was supplied.

Evidence: `d41_full_jacobian_final/D41_native_full_jacobian.csv`, `d41_full_jacobian_final/D41_native_svd_vectors.csv`, `d41_full_jacobian_final/D41_native_jacobian_split.csv`.

## Phase C — Clearance and system candidate

The accepted repair is a sensitivity-guided C4 `j4` bump of amplitude `0.005 rad` in `adversarial_0101`. The rejected `0.02 rad` shadow showed a nonlinear sign reversal and was not promoted.

- Minimum certified clearance improved from `3.430834805626115e-07 m` to `2.5891847838983156e-05 m`.
- The final worst pair remains `forearm_link|wrist2_link`; no equivalent new controlling pair appeared.
- The full native geometry run has 12/12 passing cases, zero nominal environment collisions, zero nominal self-collisions and zero native robot-world continuous-segment collisions.

Evidence: `c4_clearance_adversarial0101_amp005_certificate_full/D54_CONTINUOUS_SELF_COLLISION_CERTIFICATION.json`, `c4_clearance_adversarial0101_amp005_full_geometry/native/D41_native_case_summary.jsonl`, `geometry_candidates/c4_clearance_adversarial0101_amp005.definition.json`.

## Phase D — Articulated continuous self-collision

The full certificate covers all 12 trajectories, all 10 required self-pairs, and all certified intervals. It reports zero collision regions and zero unresolved regions. The certificate is a conservative FK-aware model-based lower bound over actual `FK(q(t))` bounded by the native state and jerk envelope.

This is not a claim of exact physical articulated self-CCD: an exact physical backend and calibrated hardware geometry are unavailable. Collision labels remain `adaptive_discrete_interpolation` wherever that established method is used.

## Dynamics and torque

Pinocchio RNEA+ABA is finite and self-consistent. Nominal peak torque is `51.15675525540761 Nm`; ±10% mass sensitivity gives `46.04107972986684` to `56.27243078094836 Nm`. Per-joint peaks remain below the project URDF effort fields `[150,150,150,28,28,28] Nm` in all three model variants. Nominal peak torque slew is `2.7185535626085415 Nm/s`; the +10% mass variant is `2.990408918869191 Nm/s`.

These are model-based results only. Hardware torque/current certification is `NOT_AVAILABLE`.

Evidence: `dynamics_amp005/native/native_dynamics_report.json`, `dynamics_amp005/per_joint_torque_audit.json`.

## Task accuracy measurement repair

Two Type-A measurement defects were repaired and rerun:

1. The FK executable previously read the last six CSV fields, so native files caused jerk values to be interpreted as joint positions. It now reads named `j1_q...j6_q` columns.
2. The target file's established observable is TCP normal alignment, not arbitrary quaternion roll. The D56 probe now measures nearest-point deviation to the authoritative 181-point task polyline and TCP z-axis normal error.

After repair, fixed-target cases have maximum geometric path deviation `0.006359292401327267 m` and maximum normal error `0.08906416462565228 rad`; the two boundary cases are reported separately because their endpoints are intentionally perturbed. The final candidate and its champion have identical task-accuracy metrics to machine precision; no task-performance regression was found. No universal new task threshold was invented.

Evidence: `accuracy_amp005_fixed/accuracy_final.json`, `accuracy_c4_baseline/accuracy.json`, `stage4a_fk_install_d56/`, `tools/d56_accuracy_probe.py`.

## Regression and final boundaries

The final focused regression contains `42 passed` tests, including Stage 3, D45, D54, D53, D52, execution-form and D56 accuracy tests. Python compilation of the changed measurement tools passed.

Remaining boundaries are external or threshold-policy boundaries, not unclosed software defects: physical articulated self-CCD, hardware-calibrated TCP/clearance uncertainty, hardware torque/current evidence, and universal task/singularity thresholds.

