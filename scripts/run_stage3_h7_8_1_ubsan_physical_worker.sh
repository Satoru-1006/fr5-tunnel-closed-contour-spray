#!/usr/bin/env bash
# Run exactly one complete frozen H7.7 physical-validation shard against the
# isolated UBSan overlay.  This has no FJT/action-client or motion surface.

set -eo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 OUTPUT_DIRECTORY" >&2
  exit 64
fi

root=/mnt/d/robotfucker
out="$1"
h77="$root/outputs/stage3_h7_7_tier_b_remediation_20260809T143600Z"
mkdir -p "$out"

source /opt/ros/jazzy/setup.bash
source "$root/install/setup.bash"
source "$root/install/stage3_h7_4_native/setup.bash"
source /home/robot/stage3_h7_8_1_sanitizers/ubsan/install/setup.bash
set -u

export UBSAN_OPTIONS='print_stacktrace=1:halt_on_error=1:report_error_type=1'
ulimit -c 0
set +e
timeout 3600s ros2 launch fr5_tunnel_moveit_bridge stage3_h7_7_physical_native.launch.py \
  "output_dir:=$out" \
  "native_samples:=$h77/formal_candidate_2/native_probe/stage25r_native_samples.jsonl" \
  "h6_4_final_validation:=$root/outputs/stage3_h6_4_native_valid_exact_ik_graph_20260809T150000Z/stage3_h6_4_final_process_validation.jsonl" \
  "h6_4_segments:=$root/outputs/stage3_h6_4_native_valid_exact_ik_graph_20260809T150000Z/stage3_h6_4_spray_on_off_segments.json" \
  "process_contract:=$root/config/stage3/stage3_h6_1_spray_process_tolerance_contract.json" \
  "fixture_mesh:=$root/outputs/stage3_h4_5_robot_fixture_geometry_codesign_20260809T000000Z/curved_fixture_stage3_h4_5_selected.obj" \
  shard_index:=0 shard_count:=4 >"$out/launch.log" 2>&1
launch_rc=$?
set -e
printf '%s\n' "$launch_rc" >"$out/launch_returncode.txt"

if [[ "$launch_rc" -ne 0 ]] || grep -q 'process has died\|runtime error:\|exit code -11\|Segmentation fault' "$out/launch.log" || ! grep -q 'process has finished cleanly' "$out/launch.log"; then
  exit 1
fi
