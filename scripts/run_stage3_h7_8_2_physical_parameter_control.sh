#!/usr/bin/env bash
set -eo pipefail
if [[ $# -ne 1 ]]; then
  echo "usage: $0 OUTPUT_DIRECTORY" >&2
  exit 64
fi
root=/mnt/d/robotfucker
out="$1"
h77="$root/outputs/stage3_h7_7_tier_b_remediation_20260809T143600Z"
launch_file="$root/scripts/stage3_h7_8_2_physical_parameter_control.launch.py"
mkdir -p "$out"
source /opt/ros/jazzy/setup.bash
source "$root/install/setup.bash"
source "$root/install/stage3_h7_4_native/setup.bash"
source /home/robot/stage3_h7_8_1_sanitizers/asan/install/setup.bash
source "$root/stage3_h7_8_overlay/install/mre2/setup.bash"
set -u
ulimit -c 0
for n in 1 2 3; do
  printf -v log '%s/C2PHYS_%03d.log' "$out" "$n"
  printf -v rcfile '%s/C2PHYS_%03d.exitcode' "$out" "$n"
  set +e
  timeout 120s ros2 launch "$launch_file" \
    "output_dir:=$out" \
    "native_samples:=$h77/formal_candidate_2/native_probe/stage25r_native_samples.jsonl" \
    "h6_4_final_validation:=$root/outputs/stage3_h6_4_native_valid_exact_ik_graph_20260809T150000Z/stage3_h6_4_final_process_validation.jsonl" \
    "h6_4_segments:=$root/outputs/stage3_h6_4_native_valid_exact_ik_graph_20260809T150000Z/stage3_h6_4_spray_on_off_segments.json" \
    "process_contract:=$root/config/stage3/stage3_h6_1_spray_process_tolerance_contract.json" \
    "fixture_mesh:=$root/outputs/stage3_h4_5_robot_fixture_geometry_codesign_20260809T000000Z/curved_fixture_stage3_h4_5_selected.obj" \
    shard_index:=0 shard_count:=4 >"$log" 2>&1
  status=$?
  set -e
  printf '%s\n' "$status" >"$rcfile"
done
