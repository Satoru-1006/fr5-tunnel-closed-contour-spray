#!/usr/bin/env bash
set -o pipefail

root=/mnt/d/robotfucker
output_root="$root/outputs/stage3_h7_8_moveitpy_clean_teardown_20260809T154616Z"
h77="$root/outputs/stage3_h7_7_tier_b_remediation_20260809T143600Z"

source /opt/ros/jazzy/setup.bash
source "$root/install/setup.bash"
source "$root/install/stage3_h7_4_native/setup.bash"
source "$root/stage3_h7_8_overlay/install/final/setup.bash"
set -u

pids=()
for replay in 1 2 3; do
  replay_dir="$output_root/formal_replay_${replay}"
  mkdir -p "$replay_dir/native_probe"
  (
    export LD_PRELOAD="$h77/libstage3_h7_4_ruckig_interposer.so"
    export STAGE25R_NATIVE_DIR="$replay_dir/native_probe"
    ros2 launch fr5_tunnel_moveit_bridge stage3_h7_7_native.launch.py \
      output_dir:="$replay_dir" \
      h6_4_final_validation:="$root/outputs/stage3_h6_4_native_valid_exact_ik_graph_20260809T150000Z/stage3_h6_4_final_process_validation.jsonl" \
      h6_4_segments:="$root/outputs/stage3_h6_4_native_valid_exact_ik_graph_20260809T150000Z/stage3_h6_4_spray_on_off_segments.json" \
      process_contract:="$root/config/stage3/stage3_h6_1_spray_process_tolerance_contract.json" \
      fixture_mesh:="$root/outputs/stage3_h4_5_robot_fixture_geometry_codesign_20260809T000000Z/curved_fixture_stage3_h4_5_selected.obj" \
      totg_parameters:="$root/outputs/stage3_h7_3_target55_controlled_stop_20260809T170900Z/stage3_h7_3_totg_parameters.json" \
      tier_b_plan:="$h77/tier_b_alpha_plan.json"
  ) >"$replay_dir/launch.log" 2>&1 &
  pids+=("$!")
done

status=0
for index in 0 1 2; do
  replay=$((index + 1))
  if wait "${pids[$index]}"; then
    launch_rc=0
  else
    launch_rc=$?
    status=1
  fi
  printf '%s\n' "$launch_rc" >"$output_root/formal_replay_${replay}/launch_returncode.txt"
done

for replay in 1 2 3; do
  replay_dir="$output_root/formal_replay_${replay}"
  if ! grep -q 'process has finished cleanly' "$replay_dir/launch.log"; then
    status=1
  fi
  if grep -q 'exit code -11\|Segmentation fault' "$replay_dir/launch.log"; then
    status=1
  fi
done

exit "$status"
