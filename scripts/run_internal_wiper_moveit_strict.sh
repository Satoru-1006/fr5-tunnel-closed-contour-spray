#!/usr/bin/env bash
set -eo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
FINAL_OUT_DIR="${FINAL_OUT_DIR:-$REPO_ROOT/outputs/internal_wiper_moveit_strict}"

export REPO_ROOT
export ROS_WS="${ROS_WS:-$REPO_ROOT}"
export TCP_POSES_CSV="${TCP_POSES_CSV:-$REPO_ROOT/outputs/internal_wiper_moveit_inputs/open_arch_tcp_poses_base_link.csv}"
export SEED_JOINT_CSV="${SEED_JOINT_CSV:-$REPO_ROOT/outputs/internal_wiper_moveit_inputs/open_arch_seed_joints.csv}"
export FINAL_OUT_DIR
export QUALITY_REPORT="$FINAL_OUT_DIR/moveit_quality_report.csv"
export DYNAMICS_REPORT="$FINAL_OUT_DIR/moveit_joint_dynamics_report.csv"
export COLLISION_REPORT="$FINAL_OUT_DIR/moveit_collision_report.csv"
export FK_TRACE="$FINAL_OUT_DIR/moveit_fk_tcp_trace.csv"
export TRAJECTORY_CSV="$FINAL_OUT_DIR/moveit_smoothed_joint_trajectory.csv"
export SEGMENTED_TRAJECTORY_CSV="$FINAL_OUT_DIR/moveit_executed_segmented_joint_trajectory.csv"
export WAYPOINT_TRAJECTORY_CSV="$FINAL_OUT_DIR/moveit_waypoint_joint_trajectory.csv"
export IK_SEARCH_DIAGNOSTICS_CSV="$FINAL_OUT_DIR/moveit_ik_search_diagnostics.csv"
export IK_CANDIDATE_BANK_CSV="$FINAL_OUT_DIR/ik_candidate_bank.csv"
export STRICT_JSON="$FINAL_OUT_DIR/audit_goal_requirements_strict.json"
export PRODUCTION_READINESS_JSON="$FINAL_OUT_DIR/production_readiness_check.json"
export RUNTIME_LOG="$FINAL_OUT_DIR/moveit_runtime.log"

export OPEN_PATH=true
export TCP_POINTS_TO_WALL=true
export SAMPLES_PER_LOOP=181
export WAYPOINT_STRIDE="${WAYPOINT_STRIDE:-1}"
export PLANNING_MODE="${PLANNING_MODE:-seed_joint_waypoints}"
export ALLOW_SEED_JOINT_WITH_TOOL_OFFSET="${ALLOW_SEED_JOINT_WITH_TOOL_OFFSET:-true}"
export IK_TIMEOUT="${IK_TIMEOUT:-0.20}"
export IK_ROLL_SAMPLE_COUNT="${IK_ROLL_SAMPLE_COUNT:-1}"
export IK_BEAM_WIDTH="${IK_BEAM_WIDTH:-1}"
export IK_CANDIDATES_PER_BEAM="${IK_CANDIDATES_PER_BEAM:-8}"
export IK_BEAM_DIVERSITY_JOINT_DEG="${IK_BEAM_DIVERSITY_JOINT_DEG:-0.0}"
export IK_MAX_SOLVE_SECONDS="${IK_MAX_SOLVE_SECONDS:-0.0}"
export VALIDATION_STRIDE=1
export COLLISION_CHECK_STRIDE=1
export COLLISION_SEGMENT_STRIDE=1
export COLLISION_INTERPOLATION_STEP_DEG=0.5

# The target section is 0.10 m from the robot base in base_link Y. A 1.10 m
# wall span centered there conservatively covers the full 0.90 m tunnel.
export TUNNEL_Y_THICKNESS=1.10
export TUNNEL_WALL_THICKNESS=0.04
export INCLUDE_BOTTOM_CLOSURE_COLLISION=false
export INCLUDE_TUNNEL_FLOOR_COLLISION=true
export TUNNEL_FLOOR_Z=-0.20
export STAND_OFF=0.260

export MAX_PATH_DEVIATION=0.006
export MAX_IK_POSITION_ERROR=0.003
export ROS_REPORT_TIMEOUT_SEC=300
export WRITE_FINAL_VISUALS=false
export WRITE_FINAL_ANIMATION=false

bash "$REPO_ROOT/scripts/run_moveit_strict_validation.sh"
