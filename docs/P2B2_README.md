# P2-B2 — Post-Reformulation Robustness Remap and Redundancy Ablation

## 结论

`P2B2_STATUS=COMPLETE_WITH_FORMAL_CAPABILITY_GAPS`。本轮对冻结 P2-A/B0 候选执行 181 点 open-arch ON-state 严格验证和鲁棒性映射；10 个变体均通过 MoveIt2 PlanningScene、FK、Ruckig 及 post-Ruckig 检查。R0 的 42 个预定义方向中，24 个已实现并完成测量，18 个仍为 `NOT_AVAILABLE / NOT_IMPLEMENTED_IN_FROZEN_P2A`。这不是 42/42 全覆盖。

P2-B1 的 B1 no-gain 判断受到冻结前缀混杂：B1 将 WP0–15 冻结，而历史最早 J6+ 失败在 WP0，旧名义 J6 最小余量在 WP2。按本轮重映射，控制方向迁移到 J3+/WP180 的末端位置误差。因此记录为 `P2B1_B1_EFFICACY_CONFOUND=CONFIRMED`；这否定的是旧图足以评估该干预的前提，不表示 R1/R2/R3 已带来可辨识的鲁棒性提升。

## 六项研究问题

1. **重整后还剩什么脆弱性？** R0 仍有小扰动即可触发的关节与任务空间末端误差边界。最小关节族方向是 `joint:j3:positive`，细化边界 `0.007212549604712551 rad`；首个失败样本为 `0.007212975400190721 rad`，WP180，失败类型 `TERMINAL_POSITION_ERROR`。对称 J3− 边界为 `0.00721269728621366 rad`。这表示扰动鲁棒性边界，不是名义轨迹碰撞或关节限位失败。
2. **为什么重新测图？** P2-B1 的干预冻结区遮住了旧图中的 WP0 控制失败点，旧图不再能代表改革后控制边界。P2-B2 对当前候选重新扫描，不沿用旧 A0 地图。
3. **各扰动族的边界是多少？** 关节族最小边界约 `0.00721255 rad`（J3+）；TCP 平移族最小边界 `0.00399609375 m`（六个方向均由终端位置误差触发，首个样本达到 `0.004 m`，WP180）；TCP 旋转族最小边界 `0.1574602871712983 rad`（绕 TCP x 正向，喷涂轴法向误差，WP172；该族 4/6 方向观察到失败，另 2 个在搜索域内未失败）。三族单位和误差定义不同，不合并成一个跨族最小值。
4. **名义候选的表现如何？** R0 严格验证通过。扰动首失败刚越过配置的 4 mm 终端位置容差，故这是可复现的边界裕度发现，不是名义 R0 的失败。样本 15→16 的名义壁面站位投影出现一次 `-0.13194315439053514 m` 回退；最大权威 TCP 路径偏差为 `0.00014126918172561212 m`，低于 6 mm 路径阈值。两者来自同一最近折线投影机制，暂列为路线顺序/投影歧义候选，不据此宣称喷涂缺陷或安全影响。
5. **冗余干预有效吗？** R1/R2/R3 的九个消融变体全部严格验证通过。最佳标记为 R1 small，但相对 R0 的全局关节边界变化仅 `+1.4547807245489375e-7 rad`，低于本轮边界分辨能力，不能作为实质提升。所有候选控制方向均迁移到 J3−/WP180。D46/P2-B2 不调参消除这些边界。
6. **Stage 4B 应先处理什么？** 按风险排序先处理 J3+/J3− 末端位置鲁棒性，再处理六个 TCP 平移方向的末端边界，第三处理四个已触发喷涂轴法向误差的 TCP 旋转方向。样本 15→16 的路线站位回退另列为中等置信度诊断候选；重映射消融也应消除 WP0–15 冻结带来的评估混杂。结构化频率、案例和证据见 [`outputs/p2b2_stage4_failure_taxonomy.json`](../outputs/p2b2_stage4_failure_taxonomy.json)。本轮没有启动 P2-B3，也没有优化冻结机器人系统。

## 测量边界

- 碰撞观测为 `adaptive_discrete_interpolation`；它不等于严格连续碰撞检测。严格自碰撞 CCD 为 `NOT_AVAILABLE`。
- 距离值仅是诊断值；验收阈值为 `UNRESOLVED_THRESHOLD`，不得据此声明安全净空。
- 动力学使用 `PROJECT_CONFIGURED_LIMITS`，不是厂商认证限值；硬件验证 `NOT_RUN`，硬件安全认证 `NO`。
- TCP 使用名义 150 mm 工具假设；未提供标定不确定度或物理涂层质量模型。
- 缺失的 18 个方向保持 unavailable。P2-B2 的 benchmark 完成状态与冻结系统表现分开记录。

## 输入与重放

GitHub 中 `outputs/p2b2_inputs/` 保存最小外部输入副本：D39 seed、P2-B1 执行清单、B0 的 Ruckig 前后轨迹、D41 身份清单所需的 D40 小型权威输入，以及 P2-B1 派生参考模型。Git 属性对这些文件关闭换行转换，暂存 blob 与正式运行来源逐字节一致；输入身份见 `outputs/p2b2_robustness_landscape.json`。官方 FAIRINO ROS2 模型来源固定为 `FAIR-INNOVATION/frcobot_ros2@60755d44d521a5ad6bee8494cc19522f8801aa20`。

运行环境：Windows Python 3.12 驱动 WSL2 Ubuntu 24.04、ROS 2 Jazzy、MoveIt 2.12.4、Ruckig 0.9.2。先在干净 checkout 中构建 `fr5_tunnel_moveit_bridge`、`stage3_h13_d41_native`、`stage4a_fk` 到独立 overlay；确保 `ros2 pkg prefix fr5_tunnel_moveit_bridge` 和导入的 `plan_closed_contour_moveit.py` 都指向该 overlay。脚本会对这两个身份做 fail-closed 检查。

全量重放命令（在已 source ROS Jazzy 和本地 underlay 的 PowerShell 中运行；scratch、官方模型源码 checkout 与 overlay 均放仓库外）：

```powershell
python scripts/run_p2b2_robustness_remapping.py `
  --scratch <TEMP_RUNTIME_SCRATCH> `
  --underlay-install <ROS_UNDERLAY_INSTALL> `
  --fairino-source-repository <FAIRINO_FRCOBOT_ROS2_CHECKOUT_AT_PINNED_COMMIT> `
  --reference-model outputs/p2b2_inputs/derived_reference_robot_model.urdf `
  --native-build-install <FRESH_NATIVE_BUILD_OVERLAY> `
  --p2b1-b0-reference outputs/p2b2_inputs/p2b1_b0_reference `
  --d39-seed outputs/p2b2_inputs/stable_velocity_residual_update.csv `
  --p2b1-execution-manifest outputs/p2b2_inputs/P2B1_execution_manifest.json `
  --frozen-input-source-root outputs/p2b2_inputs/frozen_authoritative `
  --moveit-batch-timeout-seconds 86400
```

修复与 P2-B2 汇总器的 focused 回归命令：

```powershell
python -m pytest -q tests/test_p2b1_solver_policy_ablation.py tests/test_p2b2_robustness_remapping.py
```

本轮记录为 `48 passed`。该测试结果只验证代码与汇总逻辑，不替代上述 MoveIt2、FK、Ruckig 和 formal campaign 执行。

`--reference-only --expected-execution-commit <SHA>` 仅用于干净 checkout 的 R0/P2-A 参考重放，不替代全量九项消融。所有运行数据写入外部 scratch；GitHub 只收录上述输入、核心代码、focused tests、canonical JSON/CSV 和 README。

## 产物

- `outputs/p2b2_robustness_landscape.json` / `.csv`：42 项 formal axis 定义、150 条测量记录、能力缺口和 R0/消融汇总。
- `outputs/p2b2_redundancy_ablation.json`：九项 R1/R2/R3 消融与严格验证汇总。
- `outputs/p2b2_stage4_failure_taxonomy.json`：Type-B 风险排序、频率、案例、复现信息和测量边界。
- Google Drive 阶段目录保存 CORE 字节镜像，以及筛选后的中间测量证据 ZIP 和校验清单。

代码来源基线：`e7bb161cea7396e452584f5c4ee7f158be6a9ca5`。正式 campaign 的 execution-code commit：`700212dc4d62457a6d376d42443e931e13d3ccc9`。发布 commit 与远端 SHA 在最终发布后写入 landscape JSON。
