#!/usr/bin/env bash
set -eo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ROS_WS="${ROS_WS:-$REPO_ROOT}"
TCP_POSES_CSV="${TCP_POSES_CSV:-$REPO_ROOT/outputs/tcp_poses_base_link.csv}"
SAMPLES_PER_LOOP="${SAMPLES_PER_LOOP:-240}"
OPEN_PATH="${OPEN_PATH:-false}"
TCP_POINTS_TO_WALL="${TCP_POINTS_TO_WALL:-false}"
TCP_POINTS_TO_WALL="${TCP_POINTS_TO_WALL:-false}"
TOOL_TCP_CALIBRATION_YAML="${TOOL_TCP_CALIBRATION_YAML:-}"
TOOL_TCP_XYZ="${TOOL_TCP_XYZ:-0.000 0.000 0.150}"
TOOL_TCP_RPY="${TOOL_TCP_RPY:-0 0 0}"
TOOL_TCP_SOURCE="${TOOL_TCP_SOURCE:-assumed_150mm_placeholder}"
TOOL_TCP_MEASURED_BY="${TOOL_TCP_MEASURED_BY:-simulation}"
TOOL_TCP_MEASURED_DATE="${TOOL_TCP_MEASURED_DATE:-not_applicable_virtual_design}"
TOOL_TCP_CALIBRATION_METHOD="${TOOL_TCP_CALIBRATION_METHOD:-virtual_design_parameter}"
ALLOW_ZERO_TOOL_TCP="${ALLOW_ZERO_TOOL_TCP:-false}"
ALLOW_SEED_JOINT_WITH_TOOL_OFFSET="${ALLOW_SEED_JOINT_WITH_TOOL_OFFSET:-false}"
STAND_OFF="${STAND_OFF:-0.18}"
WAYPOINT_STRIDE="${WAYPOINT_STRIDE:-4}"
PLANNING_MODE="${PLANNING_MODE:-ik_waypoints}"
IK_TIMEOUT="${IK_TIMEOUT:-0.10}"
MAX_IK_POSITION_ERROR="${MAX_IK_POSITION_ERROR:-0.002}"
MAX_IK_JOINT_STEP_DEG="${MAX_IK_JOINT_STEP_DEG:-20.0}"
IK_ROLL_SAMPLE_COUNT="${IK_ROLL_SAMPLE_COUNT:-1}"
IK_BEAM_WIDTH="${IK_BEAM_WIDTH:-1}"
IK_CANDIDATES_PER_BEAM="${IK_CANDIDATES_PER_BEAM:-8}"
IK_BEAM_DIVERSITY_JOINT_DEG="${IK_BEAM_DIVERSITY_JOINT_DEG:-0.0}"
IK_MAX_SOLVE_SECONDS="${IK_MAX_SOLVE_SECONDS:-0.0}"
SEED_JOINT_CSV="${SEED_JOINT_CSV:-}"
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
COLLISION_INTERPOLATION_STEP_DEG="${COLLISION_INTERPOLATION_STEP_DEG:-1.0}"
INCLUDE_TUNNEL_FLOOR_COLLISION="${INCLUDE_TUNNEL_FLOOR_COLLISION:-false}"
TUNNEL_FLOOR_Z="${TUNNEL_FLOOR_Z:--0.20}"
TUNNEL_WALL_THICKNESS="${TUNNEL_WALL_THICKNESS:-0.025}"
TUNNEL_Y_THICKNESS="${TUNNEL_Y_THICKNESS:-0.08}"
INCLUDE_BOTTOM_CLOSURE_COLLISION="${INCLUDE_BOTTOM_CLOSURE_COLLISION:-true}"
FAST_EXIT_AFTER_REPORTS="${FAST_EXIT_AFTER_REPORTS:-true}"
ROS_EXIT_GRACE_SEC="${ROS_EXIT_GRACE_SEC:-10}"
ROS_REPORT_TIMEOUT_SEC="${ROS_REPORT_TIMEOUT_SEC:-300}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
WRITE_FINAL_VISUALS="${WRITE_FINAL_VISUALS:-false}"
WRITE_FINAL_ANIMATION="${WRITE_FINAL_ANIMATION:-false}"
SEGMENTED_EXECUTION="${SEGMENTED_EXECUTION:-true}"

QUALITY_REPORT="${QUALITY_REPORT:-$REPO_ROOT/outputs/moveit_quality_report.csv}"
DYNAMICS_REPORT="${DYNAMICS_REPORT:-$REPO_ROOT/outputs/moveit_joint_dynamics_report.csv}"
COLLISION_REPORT="${COLLISION_REPORT:-$REPO_ROOT/outputs/moveit_collision_report.csv}"
FK_TRACE="${FK_TRACE:-$REPO_ROOT/outputs/moveit_fk_tcp_trace.csv}"
TRAJECTORY_CSV="${TRAJECTORY_CSV:-$REPO_ROOT/outputs/moveit_smoothed_joint_trajectory.csv}"
SEGMENTED_TRAJECTORY_CSV="${SEGMENTED_TRAJECTORY_CSV:-$REPO_ROOT/outputs/moveit_executed_segmented_joint_trajectory.csv}"
WAYPOINT_TRAJECTORY_CSV="${WAYPOINT_TRAJECTORY_CSV:-$REPO_ROOT/outputs/moveit_waypoint_joint_trajectory.csv}"
IK_SEARCH_DIAGNOSTICS_CSV="${IK_SEARCH_DIAGNOSTICS_CSV:-$REPO_ROOT/outputs/moveit_ik_search_diagnostics.csv}"
IK_CANDIDATE_BANK_CSV="${IK_CANDIDATE_BANK_CSV:-$REPO_ROOT/outputs/ik_candidate_bank.csv}"
IK_FAILURE_SUMMARY_JSON="${IK_FAILURE_SUMMARY_JSON:-$FINAL_OUT_DIR/moveit_ik_failure_summary.json}"
IK_SEARCH_DIAGNOSTICS_SUMMARY_JSON="${IK_SEARCH_DIAGNOSTICS_SUMMARY_JSON:-$FINAL_OUT_DIR/moveit_ik_search_diagnostics_summary.json}"
STRICT_JSON="${STRICT_JSON:-$REPO_ROOT/outputs/audit_goal_requirements_strict.json}"
PRODUCTION_READINESS_JSON="${PRODUCTION_READINESS_JSON:-$FINAL_OUT_DIR/production_readiness_check.json}"
RUNTIME_LOG="${RUNTIME_LOG:-$REPO_ROOT/outputs/moveit_runtime.log}"
FINAL_OUT_DIR="${FINAL_OUT_DIR:-$REPO_ROOT/outputs}"
FILTER_KNOWN_RUCKIG_WARNING="${FILTER_KNOWN_RUCKIG_WARNING:-true}"

mkdir -p \
  "$(dirname "$QUALITY_REPORT")" \
  "$(dirname "$DYNAMICS_REPORT")" \
  "$(dirname "$COLLISION_REPORT")" \
  "$(dirname "$FK_TRACE")" \
  "$(dirname "$TRAJECTORY_CSV")" \
  "$(dirname "$SEGMENTED_TRAJECTORY_CSV")" \
  "$(dirname "$WAYPOINT_TRAJECTORY_CSV")" \
  "$(dirname "$IK_SEARCH_DIAGNOSTICS_CSV")" \
  "$(dirname "$IK_CANDIDATE_BANK_CSV")" \
  "$(dirname "$IK_FAILURE_SUMMARY_JSON")" \
  "$(dirname "$IK_SEARCH_DIAGNOSTICS_SUMMARY_JSON")" \
  "$(dirname "$STRICT_JSON")" \
  "$(dirname "$PRODUCTION_READINESS_JSON")" \
  "$(dirname "$RUNTIME_LOG")" \
  "$FINAL_OUT_DIR"

if [[ -n "$TOOL_TCP_CALIBRATION_YAML" ]]; then
  if [[ ! -f "$TOOL_TCP_CALIBRATION_YAML" ]]; then
    echo "Missing TOOL_TCP_CALIBRATION_YAML: $TOOL_TCP_CALIBRATION_YAML" >&2
    exit 2
  fi
  calibration_exports="$("$PYTHON_BIN" "$REPO_ROOT/tools/load_tool_tcp_calibration.py" "$TOOL_TCP_CALIBRATION_YAML")"
  eval "$calibration_exports"
fi

is_zero_vector() {
  "$PYTHON_BIN" - "$1" <<'PY'
import math
import sys

values = [float(part) for part in sys.argv[1].replace(",", " ").split()]
raise SystemExit(0 if values and all(math.isclose(value, 0.0, abs_tol=1e-12) for value in values) else 1)
PY
}

tcp_xyz_is_zero=false
tcp_rpy_is_zero=false
if is_zero_vector "$TOOL_TCP_XYZ"; then
  tcp_xyz_is_zero=true
fi
if is_zero_vector "$TOOL_TCP_RPY"; then
  tcp_rpy_is_zero=true
fi

if [[ "$tcp_xyz_is_zero" == "true" && "$tcp_rpy_is_zero" == "true" && "$ALLOW_ZERO_TOOL_TCP" != "true" ]]; then
  echo "Refusing strict validation with a zero tool TCP." >&2
  echo "Set TOOL_TCP_XYZ to a measured or assumed flange-to-nozzle offset, for example: 0.000 0.000 0.150." >&2
  echo "Only set ALLOW_ZERO_TOOL_TCP=true for an explicit wrist-link debug run." >&2
  exit 2
fi

if [[ "$PLANNING_MODE" == "seed_joint_waypoints" && ( "$tcp_xyz_is_zero" != "true" || "$tcp_rpy_is_zero" != "true" ) && "$ALLOW_SEED_JOINT_WITH_TOOL_OFFSET" != "true" ]]; then
  echo "Refusing planning_mode=seed_joint_waypoints with a non-zero tool TCP." >&2
  echo "Use PLANNING_MODE=ik_waypoints so MoveIt2 solves IK for spray_tcp_link, or set ALLOW_SEED_JOINT_WITH_TOOL_OFFSET=true for a deliberate debug run." >&2
  exit 2
fi

if [[ "$PLANNING_MODE" == "seed_joint_waypoints" && -z "$SEED_JOINT_CSV" ]]; then
  SEED_JOINT_CSV="$REPO_ROOT/outputs/ik_waypoints.csv"
fi

SEED_JOINT_LAUNCH_ARG=()
if [[ -n "$SEED_JOINT_CSV" ]]; then
  SEED_JOINT_LAUNCH_ARG=(seed_joint_csv:="$SEED_JOINT_CSV")
fi

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

input_validation_args=(--tcp-path-csv "$TCP_POSES_CSV" --samples-per-loop "$SAMPLES_PER_LOOP")
if [[ "$OPEN_PATH" == "true" ]]; then
  input_validation_args+=(--open-path)
fi
"$PYTHON_BIN" "$REPO_ROOT/ros2_moveit_bridge/validate_bridge_inputs.py" "${input_validation_args[@]}"

ros2 run fr5_tunnel_moveit_bridge validate_bridge_inputs "${input_validation_args[@]}"

rm -f "$QUALITY_REPORT" "$DYNAMICS_REPORT" "$COLLISION_REPORT" "$FK_TRACE" "$TRAJECTORY_CSV" "$SEGMENTED_TRAJECTORY_CSV" "$WAYPOINT_TRAJECTORY_CSV" "$IK_SEARCH_DIAGNOSTICS_CSV" "$IK_FAILURE_SUMMARY_JSON" "$IK_SEARCH_DIAGNOSTICS_SUMMARY_JSON" "$STRICT_JSON" "$PRODUCTION_READINESS_JSON" "$RUNTIME_LOG"

summarize_ik_early_exit() {
  if [[ -s "$RUNTIME_LOG" ]]; then
    "$PYTHON_BIN" "$REPO_ROOT/tools/summarize_moveit_ik_failure.py" \
      "$RUNTIME_LOG" \
      --out-json "$IK_FAILURE_SUMMARY_JSON" || true
  fi
  if [[ -s "$IK_SEARCH_DIAGNOSTICS_CSV" ]]; then
    "$PYTHON_BIN" "$REPO_ROOT/tools/summarize_ik_search_diagnostics.py" \
      "$IK_SEARCH_DIAGNOSTICS_CSV" \
      --out-json "$IK_SEARCH_DIAGNOSTICS_SUMMARY_JSON" || true
  fi
}

set +e
setsid ros2 launch fr5_tunnel_moveit_bridge fr5_spray_plan_only.launch.py \
  tool_tcp_xyz:="$TOOL_TCP_XYZ" \
  tool_tcp_rpy:="$TOOL_TCP_RPY" \
  tool_tcp_source:="$TOOL_TCP_SOURCE" \
  tool_tcp_measured_by:="$TOOL_TCP_MEASURED_BY" \
  tool_tcp_measured_date:="$TOOL_TCP_MEASURED_DATE" \
  tool_tcp_calibration_method:="$TOOL_TCP_CALIBRATION_METHOD" \
  tcp_path_csv:="$TCP_POSES_CSV" \
  stand_off:="$STAND_OFF" \
  waypoint_stride:="$WAYPOINT_STRIDE" \
  planning_mode:="$PLANNING_MODE" \
  ik_timeout:="$IK_TIMEOUT" \
  max_ik_position_error:="$MAX_IK_POSITION_ERROR" \
  max_ik_joint_step_deg:="$MAX_IK_JOINT_STEP_DEG" \
  ik_roll_sample_count:="$IK_ROLL_SAMPLE_COUNT" \
  ik_beam_width:="$IK_BEAM_WIDTH" \
  ik_candidates_per_beam:="$IK_CANDIDATES_PER_BEAM" \
  ik_beam_diversity_joint_deg:="$IK_BEAM_DIVERSITY_JOINT_DEG" \
  ik_search_diagnostics_csv:="$IK_SEARCH_DIAGNOSTICS_CSV" \
  ik_candidate_bank_csv:="$IK_CANDIDATE_BANK_CSV" \
  ik_max_solve_seconds:="$IK_MAX_SOLVE_SECONDS" \
  "${SEED_JOINT_LAUNCH_ARG[@]}" \
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
  collision_interpolation_step_deg:="$COLLISION_INTERPOLATION_STEP_DEG" \
  include_tunnel_floor_collision:="$INCLUDE_TUNNEL_FLOOR_COLLISION" \
  tunnel_floor_z:="$TUNNEL_FLOOR_Z" \
  tunnel_wall_thickness:="$TUNNEL_WALL_THICKNESS" \
  tunnel_y_thickness:="$TUNNEL_Y_THICKNESS" \
  include_bottom_closure_collision:="$INCLUDE_BOTTOM_CLOSURE_COLLISION" \
  open_path:="$OPEN_PATH" \
  tcp_points_to_wall:="$TCP_POINTS_TO_WALL" \
  fast_exit_after_reports:="$FAST_EXIT_AFTER_REPORTS" \
  execute_trajectory:=false \
  quality_report_csv:="$QUALITY_REPORT" \
  fk_trace_csv:="$FK_TRACE" \
  trajectory_csv:="$TRAJECTORY_CSV" \
  segmented_execution:="$SEGMENTED_EXECUTION" \
  segmented_trajectory_csv:="$SEGMENTED_TRAJECTORY_CSV" \
  waypoint_trajectory_csv:="$WAYPOINT_TRAJECTORY_CSV" \
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
    summarize_ik_early_exit
    wait "$launch_pid" || true
    exit 3
  fi
  sleep 2
done

if [[ ! -s "$QUALITY_REPORT" || ! -s "$DYNAMICS_REPORT" || ! -s "$COLLISION_REPORT" || ! -s "$FK_TRACE" ]]; then
  echo "Timed out waiting for MoveIt runtime reports after ${ROS_REPORT_TIMEOUT_SEC}s." >&2
  cat "$RUNTIME_LOG" >&2 || true
  kill "-$launch_pid" 2>/dev/null || kill "$launch_pid" 2>/dev/null || true
  summarize_ik_early_exit
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

set +e
"$PYTHON_BIN" "$REPO_ROOT/tools/audit_goal_requirements.py" \
  --strict-runtime \
  --moveit-quality-report "$QUALITY_REPORT" \
  --moveit-dynamics-report "$DYNAMICS_REPORT" \
  --moveit-fk-trace "$FK_TRACE" \
  --moveit-collision-report "$COLLISION_REPORT" \
  --tcp-pose-csv "$TCP_POSES_CSV" \
  --json-report "$STRICT_JSON"
audit_rc=$?
set -e

publish_args=(
  "$REPO_ROOT/tools/publish_final_outputs.py"
  --moveit-quality-report "$QUALITY_REPORT"
  --moveit-dynamics-report "$DYNAMICS_REPORT"
  --moveit-collision-report "$COLLISION_REPORT"
  --strict-audit-json "$STRICT_JSON"
  --moveit-trajectory-csv "$TRAJECTORY_CSV"
  --segmented-execution-trajectory-csv "$SEGMENTED_TRAJECTORY_CSV"
  --moveit-waypoint-csv "$WAYPOINT_TRAJECTORY_CSV"
  --runtime-log "$RUNTIME_LOG"
  --out-dir "$FINAL_OUT_DIR"
)
if [[ "$WRITE_FINAL_VISUALS" == "true" ]]; then
  publish_args+=(--write-visuals)
fi
if [[ "$WRITE_FINAL_ANIMATION" == "true" ]]; then
  publish_args+=(--write-animation)
fi
"$PYTHON_BIN" "${publish_args[@]}"

set +e
"$PYTHON_BIN" "$REPO_ROOT/tools/check_production_readiness.py" \
  --final-quality-csv "$FINAL_OUT_DIR/final_quality_report.csv" \
  --strict-audit-json "$STRICT_JSON" \
  --collision-report-csv "$COLLISION_REPORT" \
  --out "$PRODUCTION_READINESS_JSON"
readiness_rc=$?
set -e

if [[ "$audit_rc" -ne 0 ]]; then
  echo "Strict MoveIt2/Ruckig validation failed; final reports were still published." >&2
  exit "$audit_rc"
fi

if [[ "$readiness_rc" -ne 0 ]]; then
  echo "Production readiness check failed; final reports were still published." >&2
  exit "$readiness_rc"
fi

echo "Strict MoveIt2/Ruckig validation passed."
