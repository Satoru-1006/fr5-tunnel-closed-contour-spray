# P2-B1 — Solver-Policy Causal Ablation

- **Project:** FAIRINO FR5
- **Status:** COMPLETE
- **Source base:** `ce18bfda76931385e2fbb1f923d8f980b5c487a4`
- **Execution source:** `458621af8ed8d038931c12b16779debf17602b49` (clean worktree)
- **Causal classification:** `MULTIFACTOR`

## Finding

All nine variants regenerated the same 181-point ON-state open-arch task and passed the full MoveIt2, FK, Ruckig, post-Ruckig geometry, collision, continuity, joint-limit, and configured dynamics gates. A0 reproduced the frozen D41 pre/post-Ruckig trajectories, FK geometry metrics, and P2-A J6+ margin with zero measured difference.

The A1 buffer sweep tracks its configured J6 buffer: the P2-A J6+ margin rises from `0.00010034375 rad` at `1e-4 rad` to `0.00999972820 rad` at `1e-2 rad`. The original D41 full-angular formulation repeatedly proposes J6 updates beyond the configured boundary in WP80–95; A0 records 420 J6 projection activations in that window and a maximum raw hard-bound excess of `0.02162638 rad`.

Changing only the task formulation to the mathematically consistent 5-DOF process Jacobian (B0) raises the P2-A J6+ margin to `2.81169940586 rad`, or `28,020.7×` A0. The minimum post-Ruckig J6 upper slack is `2.81169923303 rad`; the minimum all-joint margin is `0.21980500643 rad`. B1's three globally uniform null-space gains (`2.5e-4`, `5e-4`, `1e-3`) retain the B0 J6 margin without a further gain. Thus the data support material buffer and task-formulation effects; they do not show an additional J6-margin benefit from the tested secondary gains.

See [`../outputs/p2b1_solver_policy_comparison.csv`](../outputs/p2b1_solver_policy_comparison.csv) for all nine rows and [`../outputs/p2b1_solver_policy_causal_ablation.json`](../outputs/p2b1_solver_policy_causal_ablation.json) for the authoritative machine-readable result.

## Measurement and validation

The B0/B1 process residual and projected Jacobian passed 30 finite-difference checks over WP80, 81, 86, 87, and 95, three perturbation directions, and two step sizes. Maximum first-order prediction error was `6.2864e-10`; all checks passed. The focused P1/P2-A/P2-B0/P2-B1 regression completed with `58 passed`.

Collision checks used `adaptive_discrete_interpolation`; all variants reported zero sampled collisions. This is not strict continuous collision detection. FCL distance samples are diagnostic only and the clearance acceptance threshold remains unresolved. Strict self-collision CCD is `NOT_AVAILABLE`; hardware validation is `NOT_RUN` and hardware safety certification is `NO`. Dynamics limits are project-configured offline limits, not vendor-certified. The 150 mm TCP remains an assumed placeholder, not a calibrated tool measurement.

## Reproduction boundary

Execution used Python 3.12.3 on WSL2 Ubuntu 24.04, ROS 2 Jazzy, MoveIt2 `2.12.4`, and Ruckig `0.9.2`. The MoveIt bridge, D41 native evaluator, and FK evaluator were source-built from the execution commit into a merged overlay. The FAIRINO description source was clean at `60755d44d521a5ad6bee8494cc19522f8801aa20`; the regenerated D41 URDF byte-matched SHA-256 `b9d20cd51aa7088b2bab5083b4919473f0e0ec94c4b0d87d4111ca5cfa270096`.

After placing the exact frozen inputs listed in `execution_manifest.identity_sha256` at their repository paths and preparing a source-built ROS overlay, run:

```powershell
python scripts/run_p2b1_solver_policy_ablation.py `
  --scratch <D-drive-scratch-directory> `
  --urdf <byte-matched-D41-derived-URDF> `
  --underlay-install <FAIRINO-ROS-install> `
  --fairino-source-repository <clean-FAIRINO-source-checkout> `
  --native-build-install <source-built-merged-ROS-overlay>
```

Focused regression command (from the repository root in WSL):

```bash
python3 -m pytest -p no:cacheprovider tests/test_process_aware_stress.py tests/test_p2a_axiswise_robustness.py tests/test_p2b0_margin_attribution.py tests/test_p2b1_solver_policy_ablation.py
```

P2-B1 ends here. The next gate is independent review of this code, execution evidence, and result. Do not start P2-B2 before that review.
