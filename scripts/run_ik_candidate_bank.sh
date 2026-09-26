#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
FINAL_OUT_DIR="${FINAL_OUT_DIR:-$REPO_ROOT/outputs/internal_wiper_moveit_rl_candidate_bank}"
TCP_POSES_CSV="${TCP_POSES_CSV:-$REPO_ROOT/outputs/internal_wiper_moveit_inputs/open_arch_tcp_poses_base_link.csv}"
SEED_JOINT_CSV="${SEED_JOINT_CSV:-$REPO_ROOT/outputs/internal_wiper_moveit_inputs/open_arch_seed_joints.csv}"
IK_CANDIDATE_BANK_CSV="${IK_CANDIDATE_BANK_CSV:-$FINAL_OUT_DIR/ik_candidate_bank.csv}"
IK_CANDIDATE_COLLECTION_REPORT_JSON="${IK_CANDIDATE_COLLECTION_REPORT_JSON:-$FINAL_OUT_DIR/ik_candidate_bank_report.json}"
RUNTIME_LOG="${RUNTIME_LOG:-$FINAL_OUT_DIR/ik_candidate_bank_runtime.log}"

mkdir -p "$FINAL_OUT_DIR" "$(dirname "$IK_CANDIDATE_BANK_CSV")" "$(dirname "$IK_CANDIDATE_COLLECTION_REPORT_JSON")"
rm -f "$IK_CANDIDATE_BANK_CSV" "$IK_CANDIDATE_COLLECTION_REPORT_JSON" "$RUNTIME_LOG"

# Match the strict internal-wiper path gate: the verified seed route allows
# up to 6 mm of TCP path deviation after time parameterization.

if [[ -f "$REPO_ROOT/install/setup.bash" ]]; then
  # shellcheck source=/dev/null
  set +u
  source "$REPO_ROOT/install/setup.bash"
  set -u
fi

set +e
ros2 launch fr5_tunnel_moveit_bridge fr5_spray_plan_only.launch.py \
  tool_tcp_xyz:="0.000 0.000 0.150" \
  tool_tcp_rpy:="0 0 0" \
  tool_tcp_source:=assumed_150mm_placeholder \
  tool_tcp_measured_by:=simulation \
  tool_tcp_measured_date:=not_applicable_virtual_design \
  tool_tcp_calibration_method:=virtual_design_parameter \
  tcp_path_csv:="$TCP_POSES_CSV" \
  seed_joint_csv:="$SEED_JOINT_CSV" \
  stand_off:=0.260 \
  waypoint_stride:=1 \
  planning_mode:=ik_global_roll_backtracking \
  ik_timeout:=0.05 \
  max_ik_position_error:=0.006 \
  max_ik_joint_step_deg:=20.0 \
  ik_roll_sample_count:="${IK_ROLL_SAMPLE_COUNT:-36}" \
  ik_beam_width:=1 \
  ik_candidates_per_beam:="${IK_CANDIDATES_PER_BEAM:-8}" \
  global_roll_step_deg:="${GLOBAL_ROLL_STEP_DEG:-5.0}" \
  global_roll_bias_limit_deg:="${GLOBAL_ROLL_BIAS_LIMIT_DEG:-45.0}" \
  global_tilt_step_deg:="${GLOBAL_TILT_STEP_DEG:-0.5}" \
  global_tilt_limit_deg:="${GLOBAL_TILT_LIMIT_DEG:-2.0}" \
  global_max_backtracks:="${GLOBAL_MAX_BACKTRACKS:-1200}" \
  global_reparameterization_window:="${GLOBAL_REPARAMETERIZATION_WINDOW:-5}" \
  ik_beam_diversity_joint_deg:=0.0 \
  ik_search_diagnostics_csv:="$FINAL_OUT_DIR/ik_candidate_search_diagnostics.csv" \
  ik_candidate_bank_csv:="$IK_CANDIDATE_BANK_CSV" \
  ik_candidate_collection_report_json:="$IK_CANDIDATE_COLLECTION_REPORT_JSON" \
  ik_max_solve_seconds:=0.0 \
  max_normal_error_deg:=10.0 \
  validate_collision:=true \
  collision_check_stride:=1 \
  collision_segment_stride:=1 \
  collision_interpolation_step_deg:=1.0 \
  include_tunnel_floor_collision:=true \
  tunnel_floor_z:=-0.20 \
  tunnel_wall_thickness:=0.04 \
  tunnel_y_thickness:=1.10 \
  include_bottom_closure_collision:=false \
  open_path:=true \
  tcp_points_to_wall:=true \
  fast_exit_after_reports:=true \
  execute_trajectory:=false >"$RUNTIME_LOG" 2>&1
rc=$?
set -e

cat "$RUNTIME_LOG"
if [[ "$rc" -ne 0 ]]; then
  echo "IK candidate-bank collection failed with exit code $rc." >&2
  exit "$rc"
fi
if [[ ! -s "$IK_CANDIDATE_BANK_CSV" || ! -s "$IK_CANDIDATE_COLLECTION_REPORT_JSON" ]]; then
  echo "IK candidate-bank collection exited without both CSV and report." >&2
  exit 6
fi
echo "IK candidate bank written: $IK_CANDIDATE_BANK_CSV"
