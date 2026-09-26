# FR5 隧道闭合轮廓喷涂 MoveIt2 验证

本项目用于验证 FAIRINO FR5 V6 机器人在隧道马蹄形截面内执行闭合轮廓喷涂/涂覆轨迹的可行性。验证链覆盖离线路径生成、TCP 姿态、MoveIt2 IK、Ruckig 平滑、FK 质量门、关节动力学门和碰撞门。

本项目固定使用沿法兰工具 +Z 方向伸出 `150 mm` 的虚拟 TCP（`spray_tcp_link`）。它是项目建模条件，不要求实测，也不使用 `wrist3_link` 代替。项目在指定停靠点关闭喷涂、停止并平滑调整姿态；当前 MoveIt2 桥接节点默认把这套分段轨迹作为实际执行轨迹，同时保留原始未分段轨迹用于对照审计。生产使用前仍必须替换为实测 TCP 和真实工位模型。

当前推荐的标准马蹄形主方案是内部雨刮式开口拱形扫描：从左侧下部起扫，连续经过上拱，在右侧下部结束；不沿底部闭合回转，也不依赖腕部 360° 旋转。旧的 `closed_horseshoe` 路径仅保留作对照诊断。

推荐演示命令：

```powershell
python examples/run_fr5_tunnel_spray.py --path-mode internal_wiper --base-y 0.45 --station-y 0.55 --length 0.90 --wiper-samples 181
```

## 当前状态

主要报告位于：

```text
outputs/final_quality_report.csv
outputs/audit_goal_requirements_strict.json
outputs/moveit_collision_report.csv
outputs/moveit_quality_report.csv
outputs/moveit_joint_step_report.csv
outputs/ik_continuity_segments_summary.json
outputs/ik_root_cause_matrix.json
outputs/validation_handoff_manifest.json
```

当前正式报告的关键结果：

```text
status = pass
collision_status = pass
tcp_speed_production_status = pass
tool_tcp_acceptance_status = waived
joint_continuity_status = pass
max_joint_step_deg = 145.8486084788797
process_max_joint_step_deg = 3.613547166284726
reorientation_transition_max_interpolated_step_deg = 4.979829912673047
tool_tcp_source = assumed_150mm_placeholder
```

`145.85 deg` 是相邻工艺段端点之间的总换姿量，不作为一次关节突跳执行。当前将其放在停喷区间内，以五次 smoothstep 轨迹拆分；停靠点速度和加速度均为零。

```text
process_joint_continuity_status = pass
process_collision_status = pass
process_dynamics_status = pass
spray_off_transition_status = pass
reorientation_transition_collision_status = pass
reorientation_transition_dynamics_status = pass
stop_boundary_zero_velocity_acceleration_status = pass
next_segment_entry_status = pass
no_gap_or_overlap_status = pass
```

## 已解决的问题

- 最初 `150 mm` 假设 TCP 下的 `145/145` 抽检状态全部碰撞，已经在当前简化碰撞模型中修正为 `collision_count=0`、`first_collision_index=-1`、`status=pass`。
- `150 mm` TCP 已按项目要求定义为虚拟建模参数，不再作为实测 TCP 缺失问题。
- TCP 速度已从约 `0.5 mm/s` 的安全诊断速度提升到当前正式报告约 `0.003096 m/s`，并通过 `0.003 m/s` 生产下限。
- 执行链增加硬门禁：`execute_trajectory:=true` 时，FK 质量报告不是 `pass` 就不会调用 `moveit.execute`。
- 大 IK 分支变化已转换为 12 个显式停喷换姿段；换姿使用五次 smoothstep，端点速度和加速度为零。
- 验收增加工艺段连续性/碰撞/动力学、停喷状态、换姿碰撞/动力学、下一段精确接续和无缺口/重叠硬门。

## 当前验收边界

- 工艺段必须连续、无碰撞且动力学合格。
- phase 110、210、232、237 等指定停靠点允许关闭喷涂并换姿。
- 换姿轨迹必须无碰撞、无速度/加速度/jerk 超限，且停靠端速度和加速度为零。
- 换姿终点必须与下一工艺段起点一致，禁止索引缺口、重叠喷涂和实际关节突跳。
- 当前碰撞结论只适用于项目提供的简化隧道模型；若未来用于实体工位，必须另做真实工位验证。

## 本地验证

推荐先运行一条本地收口命令：

```powershell
python scripts/verify_current_goal_state.py
```

它会刷新 action plan、production readiness、handoff manifest、commit plan、goal audit，并运行 `pytest`。当前预期结果不是生产通过，而是：

```text
overall_goal_status = active
production_readiness_status = fail
pytest_status = pass
```

单独快速回归：

```powershell
python -m pytest
```

最终生产验收时可使用严格返回码模式；当前 placeholder TCP 状态下该命令应失败：

```powershell
python scripts/verify_current_goal_state.py --require-production-ready
```

在 WSL/ROS2 Jazzy 环境中运行严格链：

```bash
bash scripts/run_moveit_strict_validation.sh
```

脚本会生成 MoveIt2 运行报告、严格审计 JSON、最终汇总和 handoff manifest。即使审计失败，最终报告仍会发布，以便保留失败证据。
# P1 — Process-Aware Stress Model

`src/process_aware_stress.py` provides deterministic base-frame, TCP, surface,
joint-state, and timing stress operators; optional trajectory fields remain
unknown when absent. Point, transition, and trajectory validity are reported
separately, and unavailable required capabilities cannot produce `PASS`.
`map_legacy_d46_case` describes the six existing D46 families without changing
case generation or stored measurements. The native D46 capability profile
keeps `adaptive_discrete_interpolation`, the native robot-world collision API, FCL
distance, and unavailable continuous self-collision distinct. This is P1
formalization only; robustness-margin search and physical/hardware validation
are not implemented or claimed.

## P2-A — Axis-wise robustness margins

`src/p2a_axiswise_robustness.py` performs deterministic signed one-axis scans
and refines only adjacent PASS-to-FAIL brackets, using the frozen D41 181-point
open-arch path with the existing MoveIt2/FCL and FK evaluators. Its nominal
geometry reference is fresh MoveIt2 FK of the frozen D41 post-Ruckig joint
trajectory, and its fixed process normals come from the matching D41
strict-replay FK trace. The authenticated Stage 0/1 pose pair remains upstream
identity evidence; it is not treated as a pairwise reference for the changed
post-Ruckig path. Process orientation is evaluated by the spray TCP +Z
direction against the D41 wall normal; TCP rotation perturbations compare the
perturbed target +Z to that same normal. Full-quaternion difference is
diagnostic-only, so tool-axis roll does not create a false spray-normal failure.
The machine-readable result is
`outputs/p2a_axiswise_robustness_margin.json`. Base transforms, stand-off and
surface-offset propagation, strict self-collision CCD, calibrated uncertainty,
torque, coating physics, hardware validation, and global robustness remain
unavailable or uncertified.

## GitHub project scope

The FR5-only source, evidence scope, reproduction boundary, and P2-A provenance note are documented in [docs/GITHUB_PROJECT_SCOPE.md](docs/GITHUB_PROJECT_SCOPE.md).

## P2-B0 — J6 margin attribution and task-constrained sensitivity

`src/p2b0_margin_attribution.py` measures the frozen D41 path and tests local
process-task null-space retreats over WP80–95. At WP86, nominal J6 slack is
`0.00010000000000021103 rad`, matching D41's configured `1e-4 rad` buffer;
P2-A's refined first-failure boundary is `0.00010034375 rad` and is a joint
limit failure. The tolerance-scaled 5×6 process Jacobian has rank 5 and
nullity 1 at WP86, with J6 null-space projection `0.9858`.

The 10 mrad local retreat increases WP86 J6 slack to `0.0101 rad` while
changing TCP position by at most `5.4 µm` and spray-normal angle by at most
`0.000179°` within the tested window. Joint bounds, continuity, adaptive
discrete-interpolation collision checks, and project-configured finite-
difference dynamics pass for the shadow candidate. These results support a
`SOLVER_POLICY_DOMINATED` local attribution; alternate-buffer full D41
regeneration and post-Ruckig candidate validation were not run. Strict
continuous self-collision, clearance acceptance, hardware validation, and
global robustness remain unavailable or unverified. This is a measured
pre-Ruckig shadow, not a release-ready trajectory. Focused regression:
`31 passed`. Full machine-readable evidence is in
`outputs/p2b0_margin_attribution.json`.

## P2-B1 — Solver-policy causal ablation

P2-B1 completed nine full 181-point regenerations with MoveIt2, Ruckig,
post-Ruckig FK/geometry, collision, continuity, joint-limit, and configured
dynamics validation. A0 exactly reproduced the frozen D41 pre/post-Ruckig
trajectories and P2-A J6+ margin. The measured classification is
`MULTIFACTOR`: A1 margins track the selected J6 buffer, while the correct
5-DOF process formulation raises J6+ margin to `2.81169940586 rad`
(`28,020.7×` A0). The three tested B1 null-space gains did not further improve
that margin. This does not optimize the frozen Stage 3 baseline.

The machine-readable result and nine-variant comparison are in
[`outputs/p2b1_solver_policy_causal_ablation.json`](outputs/p2b1_solver_policy_causal_ablation.json)
and [`outputs/p2b1_solver_policy_comparison.csv`](outputs/p2b1_solver_policy_comparison.csv).
See [`docs/P2B1_README.md`](docs/P2B1_README.md) for methods, reproduction,
validation boundaries, and the next review gate. Collision evidence is
`adaptive_discrete_interpolation`, not strict CCD; strict self-CCD is
unavailable, hardware validation is `NOT_RUN`, and hardware safety is `NO`.
