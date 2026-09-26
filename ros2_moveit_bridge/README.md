# FR5 MoveIt2 Bridge

This folder is the ROS2-side execution-chain entry.

Prerequisites:

- ROS2 with MoveIt2 installed.
- FAIRINO `frcobot_ros2` packages in the same ROS2 workspace.
- `fairino5_v6_moveit2_config` built and sourced.
- The calibrated flange-to-nozzle transform configured as `spray_tcp_link`.

`config/fairino5_v6_spray_tcp.urdf.xacro` extends FAIRINO's official FR5 V6
description with a fixed TCP link. Its `tool_tcp_xyz` and `tool_tcp_rpy`
arguments describe the flange-to-nozzle transform. For early planning before
the nozzle is designed, `scripts/run_moveit_strict_validation.sh` supplies an
auditable placeholder:

```text
TOOL_TCP_XYZ    = 0.000 0.000 0.150
TOOL_TCP_RPY    = 0 0 0
TOOL_TCP_SOURCE = assumed_150mm_placeholder
```

Replace those values with the measured transform before any production
conclusion or real execution.

Use `config/tool_tcp_calibration_template.yaml` as the handoff checklist for
the measured flange-to-nozzle TCP. After copying and filling it, prefer passing
the YAML directly to strict validation:

```bash
TOOL_TCP_CALIBRATION_YAML=ros2_moveit_bridge/config/measured_tool_tcp.yaml \
STAND_OFF=0.26 \
TARGET_TCP_SPEED=0.0031 \
MAX_IK_JOINT_STEP_DEG=20.0 \
INCLUDE_BOTTOM_CLOSURE_COLLISION=true \
bash scripts/run_moveit_strict_validation.sh
```

`tools/load_tool_tcp_calibration.py` validates the YAML, rejects placeholder or
assumed sources, rejects an all-zero TCP, and exports the equivalent:

```bash
TOOL_TCP_XYZ="<x y z>" \
TOOL_TCP_RPY="<roll pitch yaw>" \
TOOL_TCP_SOURCE="<measured source id>" \
STAND_OFF=0.26 \
TARGET_TCP_SPEED=0.0031 \
MAX_IK_JOINT_STEP_DEG=20.0 \
INCLUDE_BOTTOM_CLOSURE_COLLISION=true \
bash scripts/run_moveit_strict_validation.sh
```

The resulting `outputs/final_quality_report.csv` must show
`tool_tcp_production_status=pass`, `joint_continuity_status=pass`,
`collision_status=pass`, `tcp_speed_production_status=pass`, and `status=pass`
before any production conclusion.

`config/fairino5_v6_spray_tcp.srdf` mirrors FAIRINO's SRDF but changes the
`fairino5_v6_group` chain tip from `wrist3_link` to `spray_tcp_link`, so MoveIt2
IK and pose goals solve for the actual nozzle TCP rather than the flange.

Build this folder as a ROS2 package (for example, symlink or copy it under a
colcon workspace `src/` directory), then launch with the measured or explicitly
assumed tool TCP:

```bash
colcon build --packages-select fairino_description fairino5_v6_moveit2_config fr5_tunnel_moveit_bridge
source install/setup.bash
REPO_ROOT=/absolute/path/to/fr5-tunnel-repo \
ROS_WS=/absolute/path/to/ros2_ws \
TOOL_TCP_XYZ="0.000 0.000 0.150" \
bash /absolute/path/to/fr5-tunnel-repo/scripts/run_moveit_strict_validation.sh
```

The current diagnostic pass for the assumed 150 mm TCP also uses:

```bash
STAND_OFF=0.26
MAX_IK_JOINT_STEP_DEG=180.0
IK_TIMEOUT=0.30
TARGET_TCP_SPEED=0.0031
INCLUDE_BOTTOM_CLOSURE_COLLISION=true
```

These settings are audit evidence, not production defaults. In particular,
`MAX_IK_JOINT_STEP_DEG=180.0` is a diagnostic relaxation; the normal 20 degree
continuity gate still exposes large branch jumps and must be solved before
production use. The current report clears the collision, FK, speed, and joint
dynamics gates under these assumptions; strict audit still fails because the TCP
source is the placeholder and the 20 degree continuity gate fails.

For continuity diagnosis, `IK_ROLL_SAMPLE_COUNT` can be set above `1` to sample
equivalent nozzle roll angles while preserving TCP position and tool-Z normal
alignment. The current `IK_ROLL_SAMPLE_COUNT=9` probe helps but still fails: the
first failure changes from a branch jump to self-colliding continuous IK
candidates, so it is not a production pass.

`IK_BEAM_WIDTH` can be set above `1` to retain multiple IK branch candidates
instead of greedily committing to the locally smallest joint step. The current
`IK_BEAM_WIDTH=4`, `IK_ROLL_SAMPLE_COUNT=5` probe improves the failure mode but
still fails at the 20 degree gate, so it is also diagnostic only.
`IK_CANDIDATES_PER_BEAM` limits per-beam expansion cost; wider beam probes
should set it deliberately because an unconstrained branch search can exceed the
strict report timeout.

The script validates `tcp_poses.csv`, launches the MoveIt2 bridge, waits for the
post-Ruckig FK/dynamics reports, terminates the long-running demo launch, and
then runs the strict repository audit. The equivalent manual sequence is:

```bash
ros2 run fr5_tunnel_moveit_bridge validate_bridge_inputs \
  --tcp-path-csv /absolute/path/to/outputs/tcp_poses.csv \
  --samples-per-loop 240
ros2 launch fr5_tunnel_moveit_bridge fr5_spray_smoke.launch.py \
  tool_tcp_xyz:="0.000 0.000 0.150" \
  tool_tcp_rpy:="0 0 0" \
  tcp_path_csv:=/absolute/path/to/outputs/tcp_poses.csv
ros2 launch fr5_tunnel_moveit_bridge fr5_spray_demo.launch.py \
  tool_tcp_xyz:="0.000 0.000 0.150" \
  tool_tcp_rpy:="0 0 0" \
  stand_off:=0.18 \
  waypoint_stride:=4 \
  velocity_scaling:=0.15 \
  acceleration_scaling:=0.15 \
  max_path_deviation:=0.005 \
  max_normal_error_deg:=10.0 \
  max_standoff_fraction:=0.05 \
  max_speed_fluctuation:=0.05 \
  execute_trajectory:=false \
  tcp_path_csv:=/absolute/path/to/outputs/tcp_poses.csv
```

Notes:

- The Windows Python demo now uses official FR5 URDF geometry by default.
- The strict script defaults to `PLANNING_MODE=ik_waypoints`, because MoveIt2
  must solve IK for `spray_tcp_link` when the tool TCP is non-zero. It refuses
  `planning_mode=seed_joint_waypoints` with a non-zero tool TCP unless
  `ALLOW_SEED_JOINT_WITH_TOOL_OFFSET=true` is set for a deliberate debug run.
- Strict validation also refuses a zero tool TCP unless
  `ALLOW_ZERO_TOOL_TCP=true` is set. A zero TCP is a wrist-link debug case, not
  evidence for an unknown spray head.
- `INCLUDE_BOTTOM_CLOSURE_COLLISION=true` includes the bottom closure in the
  simplified collision model. The current diagnostic report has zero collision
  states with this setting, but real cell floor, tunnel bottom, trolley, and
  boundary geometry still need to replace the simplified model before production
  conclusions.
- `tcp_poses.csv` contains TCP position, quaternion orientation, and wall normal
  columns. The bridge uses the quaternion orientation; it no longer sends
  identity-orientation waypoints.
- The bridge sends each waypoint as a normal-oriented `PoseStamped` goal for
  `spray_tcp_link`, so MoveIt2 constrains both nozzle position and wall-normal
  orientation. Do not substitute `wrist3_link` unless the calibrated TCP is
  exactly coincident with the flange frame.
- Every segment is planned from the previous planned endpoint. The bridge
  concatenates the complete contour, converts it to MoveIt2's bound
  `RobotTrajectory`, runs native TOTG, then runs native
  `RobotTrajectory.apply_ruckig_smoothing()` once before a single execution.
  `use_ruckig_smoothing:=false` is only acceptable for offline debugging; the
  node refuses controller execution unless native Ruckig smoothing is enabled.
- By default, `segmented_execution:=true` converts large IK branch changes into
  executable spray-off stops: it holds the boundary at zero velocity and
  acceleration, follows a quintic joint transition, holds the next boundary,
  and sends the resulting `moveit_executed_segmented_joint_trajectory.csv` to
  the controller. The original post-Ruckig trajectory remains in
  `moveit_smoothed_joint_trajectory.csv` for raw-jump comparison.
  Process FK is checked against the coating path on the raw process points;
  the executed segmented trajectory is checked independently for dynamics and
  collision, because a spray-off reorientation is intentionally not required
  to remain on the coating path between its two boundary poses.
- `config/joint_limits_with_jerk.yaml` preserves FAIRINO's official velocity
  and acceleration limits and adds explicit 8 rad/s^3 jerk limits. The launch
  file loads this override into `robot_description_planning`.
- After Ruckig, every trajectory sample is recomputed with MoveIt `RobotState`
  FK. The bridge refuses execution when TCP speed fluctuation exceeds 5%, wall
  normal error exceeds 10 degrees, stand-off error exceeds 5%, or FK TCP leaves
  the target closed contour by more than `max_path_deviation`. Results are
  written to `outputs/moveit_quality_report.csv`, including
  `moveit_ruckig_smoothing_used=true` and `moveit_ee_link=spray_tcp_link` for
  runtime audit. Per-sample FK TCP evidence is written to
  `outputs/moveit_fk_tcp_trace.csv`.
- By default the launch is safe for offline validation: it plans, smooths,
  validates, and exports `outputs/moveit_smoothed_joint_trajectory.csv` without
  sending the controller command. Set `execute_trajectory:=true` only after the
  FK quality report passes and the cell is ready for motion. The node refuses
  execution when the FK quality report status is not `pass`, including
  `fail:joint continuity`.
- The exported trajectory includes `q/dq/ddq/jerk` columns. A separate
  `outputs/moveit_joint_dynamics_report.csv` checks post-Ruckig velocity,
  acceleration, and jerk against MoveIt joint limits before execution.
  This check is controlled by `validate_joint_dynamics` and is enabled by
  default.
- After running the ROS2 bridge, run the repository audit in strict mode:
  `python tools/audit_goal_requirements.py --strict-runtime --moveit-quality-report outputs/moveit_quality_report.csv --moveit-dynamics-report outputs/moveit_joint_dynamics_report.csv --moveit-fk-trace outputs/moveit_fk_tcp_trace.csv --json-report outputs/audit_goal_requirements_strict.json`.
  Strict mode fails unless both MoveIt runtime reports exist and pass their FK
  and joint-dynamics gates.
- The installed MoveIt2 version must expose the MoveItPy
  `RobotTrajectory.apply_ruckig_smoothing()` binding. This is verified against
  current upstream MoveIt2 source; validate availability in the target ROS2
  distribution before deployment.
