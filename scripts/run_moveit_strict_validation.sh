#!/usr/bin/env bash
set -eo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ROS_WS="${ROS_WS:-$REPO_ROOT}"
TCP_POSES_CSV="${TCP_POSES_CSV:-$REPO_ROOT/outputs/tcp_poses_base_link.csv}"
SAMPLES_PER_LOOP="${SAMPLES_PER_LOOP:-240}"
TOOL_TCP_XYZ="${TOOL_TCP_XYZ:-0.000 0.000 0.000}"
TOOL_TCP_RPY="${TOOL_TCP_RPY:-0 0 0}"
STAND_OFF="${STAND_OFF:-0.18}"
WAYPOINT_STRIDE="${WAYPOINT_STRIDE:-4}"
PLANNING_MODE="${PLANNING_MODE:-seed_joint_waypoints}"
IK_TIMEOUT="${IK_TIMEOUT:-0.10}"
MAX_IK_POSITION_ERROR="${MAX_IK_POSITION_ERROR:-0.002}"
MAX_IK_JOINT_STEP_DEG="${MAX_IK_JOINT_STEP_DEG:-20.0}"
SEED_JOINT_CSV="${SEED_JOINT_CSV:-$REPO_ROOT/outputs/ik_waypoints.csv}"
VALIDATION_STRIDE="${VALIDATION_STRIDE:-1}"
VELOCITY_SCALING="${VELOCITY_SCALING:-0.15}"
ACCELERATION_SCALING="${ACCELERATION_SCALING:-0.15}"
TIME_PARAMETERIZATION="${TIME_PARAMETERIZATION:-tcp_arclength}"
TARGET_TCP_SPEED="${TARGET_TCP_SPEED:-0.003}"
ZERO_BOUNDARY_STATE="${ZERO_BOUNDARY_STATE:-true}"
MAX_PATH_DEVIATION="${MAX_PATH_DEVIATION:-0.005}"
MAX_NORMAL_ERROR_DEG="${MAX_NORMAL_ERROR_DEG:-10.0}"
MAX_STANDOFF_FRACTION="${MAX_STANDOFF_FRACTION:-0.05}"
MAX_SPEED_FLUCTUATION="${MAX_SPEED_FLUCTUATION:-0.05}"
VALIDATE_COLLISION="${VALIDATE_COLLISION:-true}"
COLLISION_CHECK_STRIDE="${COLLISION_CHECK_STRIDE:-5}"
COLLISION_SEGMENT_STRIDE="${COLLISION_SEGMENT_STRIDE:-4}"
TUNNEL_WALL_THICKNESS="${TUNNEL_WALL_THICKNESS:-0.025}"
TUNNEL_Y_THICKNESS="${TUNNEL_Y_THICKNESS:-0.08}"
FAST_EXIT_AFTER_REPORTS="${FAST_EXIT_AFTER_REPORTS:-true}"
ROS_EXIT_GRACE_SEC="${ROS_EXIT_GRACE_SEC:-10}"
ROS_REPORT_TIMEOUT_SEC="${ROS_REPORT_TIMEOUT_SEC:-300}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
WRITE_FINAL_VISUALS="${WRITE_FINAL_VISUALS:-false}"
WRITE_FINAL_ANIMATION="${WRITE_FINAL_ANIMATION:-false}"

QUALITY_REPORT="$REPO_ROOT/outputs/moveit_quality_report.csv"
DYNAMICS_REPORT="$REPO_ROOT/outputs/moveit_joint_dynamics_report.csv"
COLLISION_REPORT="$REPO_ROOT/outputs/moveit_collision_report.csv"
FK_TRACE="$REPO_ROOT/outputs/moveit_fk_tcp_trace.csv"
TRAJECTORY_CSV="$REPO_ROOT/outputs/moveit_smoothed_joint_trajectory.csv"
STRICT_JSON="$REPO_ROOT/outputs/audit_goal_requirements_strict.json"
RUNTIME_LOG="$REPO_ROOT/outputs/moveit_runtime.log"
FILTER_KNOWN_RUCKIG_WARNING="${FILTER_KNOWN_RUCKIG_WARNING:-true}"

if [[ ! -f "$TCP_POSES_CSV" ]]; then
  echo "Missing TCP pose CSV: $TCP_POSES_CSV" >&2
  echo "Run: python3 examples/run_fr5_tunnel_spray.py --no-animation" >&2
  exit 2
fi
if [[ -n "$SEED_JOINT_CSV" && ! -f "$SEED_JOINT_CSV" ]]; then
  echo "Missing seed joint CSV: $SEED_JOINT_CSV" >&2
  echo "Run: python3 examples/run_fr5_tunnel_spray.py --no-animation" >&2
  exit 2
fi

if [[ -f "$ROS_WS/install/setup.bash" ]]; then
  # shellcheck source=/dev/null
  source "$ROS_WS/install/setup.bash"
fi

"$PYTHON_BIN" "$REPO_ROOT/ros2_moveit_bridge/validate_bridge_inputs.py" \
  --tcp-path-csv "$TCP_POSES_CSV" \
  --samples-per-loop "$SAMPLES_PER_LOOP"

ros2 run fr5_tunnel_moveit_bridge validate_bridge_inputs \
  --tcp-path-csv "$TCP_POSES_CSV" \
  --samples-per-loop "$SAMPLES_PER_LOOP"

rm -f "$QUALITY_REPORT" "$DYNAMICS_REPORT" "$COLLISION_REPORT" "$FK_TRACE" "$TRAJECTORY_CSV" "$STRICT_JSON" "$RUNTIME_LOG"

set +e
setsid ros2 launch fr5_tunnel_moveit_bridge fr5_spray_plan_only.launch.py \
  tool_tcp_xyz:="$TOOL_TCP_XYZ" \
  tool_tcp_rpy:="$TOOL_TCP_RPY" \
  tcp_path_csv:="$TCP_POSES_CSV" \
  stand_off:="$STAND_OFF" \
  waypoint_stride:="$WAYPOINT_STRIDE" \
  planning_mode:="$PLANNING_MODE" \
  ik_timeout:="$IK_TIMEOUT" \
  max_ik_position_error:="$MAX_IK_POSITION_ERROR" \
  max_ik_joint_step_deg:="$MAX_IK_JOINT_STEP_DEG" \
  seed_joint_csv:="$SEED_JOINT_CSV" \
  validation_stride:="$VALIDATION_STRIDE" \
  velocity_scaling:="$VELOCITY_SCALING" \
  acceleration_scaling:="$ACCELERATION_SCALING" \
  time_parameterization:="$TIME_PARAMETERIZATION" \
  target_tcp_speed:="$TARGET_TCP_SPEED" \
  zero_boundary_state:="$ZERO_BOUNDARY_STATE" \
  max_path_deviation:="$MAX_PATH_DEVIATION" \
  max_normal_error_deg:="$MAX_NORMAL_ERROR_DEG" \
  max_standoff_fraction:="$MAX_STANDOFF_FRACTION" \
  max_speed_fluctuation:="$MAX_SPEED_FLUCTUATION" \
  validate_collision:="$VALIDATE_COLLISION" \
  collision_check_stride:="$COLLISION_CHECK_STRIDE" \
  collision_segment_stride:="$COLLISION_SEGMENT_STRIDE" \
  tunnel_wall_thickness:="$TUNNEL_WALL_THICKNESS" \
  tunnel_y_thickness:="$TUNNEL_Y_THICKNESS" \
  fast_exit_after_reports:="$FAST_EXIT_AFTER_REPORTS" \
  execute_trajectory:=false \
  quality_report_csv:="$QUALITY_REPORT" \
  fk_trace_csv:="$FK_TRACE" \
  trajectory_csv:="$TRAJECTORY_CSV" \
  dynamics_report_csv:="$DYNAMICS_REPORT" \
  collision_report_csv:="$COLLISION_REPORT" > "$RUNTIME_LOG" 2>&1 &
launch_pid=$!
set -e

deadline=$((SECONDS + ROS_REPORT_TIMEOUT_SEC))
while (( SECONDS < deadline )); do
  if [[ -s "$QUALITY_REPORT" && -s "$DYNAMICS_REPORT" && -s "$COLLISION_REPORT" && -s "$FK_TRACE" ]]; then
    break
  fi
  if ! kill -0 "$launch_pid" 2>/dev/null; then
    echo "ROS2 launch exited before all runtime reports were produced." >&2
    cat "$RUNTIME_LOG" >&2 || true
    wait "$launch_pid" || true
    exit 3
  fi
  sleep 2
done

if [[ ! -s "$QUALITY_REPORT" || ! -s "$DYNAMICS_REPORT" || ! -s "$COLLISION_REPORT" || ! -s "$FK_TRACE" ]]; then
  echo "Timed out waiting for MoveIt runtime reports after ${ROS_REPORT_TIMEOUT_SEC}s." >&2
  cat "$RUNTIME_LOG" >&2 || true
  kill "-$launch_pid" 2>/dev/null || kill "$launch_pid" 2>/dev/null || true
  exit 4
fi

exit_deadline=$((SECONDS + ROS_EXIT_GRACE_SEC))
while (( SECONDS < exit_deadline )); do
  if ! kill -0 "$launch_pid" 2>/dev/null; then
    break
  fi
  sleep 1
done

if kill -0 "$launch_pid" 2>/dev/null; then
  echo "ROS2 launch did not exit after reports; terminating process group." >&2
  kill "-$launch_pid" 2>/dev/null || kill "$launch_pid" 2>/dev/null || true
fi
wait "$launch_pid" 2>/dev/null || true

if [[ "$FILTER_KNOWN_RUCKIG_WARNING" == "true" ]]; then
  grep -v "Ruckig extended the trajectory duration to its maximum and still did not find a solution" "$RUNTIME_LOG" || true
else
  cat "$RUNTIME_LOG"
fi

"$PYTHON_BIN" "$REPO_ROOT/tools/audit_goal_requirements.py" \
  --strict-runtime \
  --moveit-quality-report "$QUALITY_REPORT" \
  --moveit-dynamics-report "$DYNAMICS_REPORT" \
  --moveit-fk-trace "$FK_TRACE" \
  --moveit-collision-report "$COLLISION_REPORT" \
  --json-report "$STRICT_JSON"

publish_args=(
  "$REPO_ROOT/tools/publish_final_outputs.py"
  --moveit-quality-report "$QUALITY_REPORT"
  --moveit-dynamics-report "$DYNAMICS_REPORT"
  --moveit-collision-report "$COLLISION_REPORT"
  --strict-audit-json "$STRICT_JSON"
  --moveit-trajectory-csv "$TRAJECTORY_CSV"
  --runtime-log "$RUNTIME_LOG"
  --out-dir "$REPO_ROOT/outputs"
)
if [[ "$WRITE_FINAL_VISUALS" == "true" ]]; then
  publish_args+=(--write-visuals)
fi
if [[ "$WRITE_FINAL_ANIMATION" == "true" ]]; then
  publish_args+=(--write-animation)
fi
"$PYTHON_BIN" "${publish_args[@]}"

echo "Strict MoveIt2/Ruckig validation passed."
