#!/usr/bin/env bash
# Run C4-C8 as fresh child processes; ASan is injected only into the MRE node.

set -eo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 OUTPUT_DIRECTORY" >&2
  exit 64
fi

root=/mnt/d/robotfucker
out="$1"
mkdir -p "$out"
samples="$root/outputs/stage3_h7_7_tier_b_remediation_20260809T143600Z/formal_candidate_2/native_probe/stage25r_native_samples.jsonl"
launch_file="$root/scripts/stage3_h7_8_2_asan_mre.launch.py"

source /opt/ros/jazzy/setup.bash
source "$root/install/setup.bash"
asan_overlay="${H782_ASAN_OVERLAY:-/home/robot/stage3_h7_8_1_sanitizers/asan}"
source "$asan_overlay/install/setup.bash"
source "$root/stage3_h7_8_overlay/install/mre2/setup.bash"
set -u
ulimit -c 0

{
  printf 'Python: '; ldd /usr/bin/python3
  while IFS= read -r library; do
    printf '\nExtension: %s\n' "$library"
    ldd "$library"
  done < <(find "$asan_overlay/install" -type f \
    \( -name 'planning*.so' -o -name 'moveit_cpp*.so' -o -name 'libmoveit_trajectory_execution_manager*.so' \) | LC_ALL=C sort)
} >"$out/C4_C8_loaded_libraries.txt"

run_control() {
  local id="$1"
  local mode="$2"
  local variant="$3"
  local count="$4"
  local n log rcfile envfile
  for ((n=1; n<=count; n++)); do
    printf -v log '%s/%s_%03d.log' "$out" "$id" "$n"
    printf -v rcfile '%s/%s_%03d.exitcode' "$out" "$id" "$n"
    printf -v envfile '%s/%s_%03d.env' "$out" "$id" "$n"
    {
      declare -px
      printf 'COMMAND=ros2 launch %s control_mode:=%s mre_variant:=%s teardown_mode:=explicit native_samples:=%s\n' \
        "$launch_file" "$mode" "$variant" "$samples"
      printf 'CHILD_LD_PRELOAD=%s\n' \
        '/usr/lib/gcc/x86_64-linux-gnu/13/libasan.so:/usr/lib/x86_64-linux-gnu/libstdc++.so.6'
      printf 'CHILD_ASAN_OPTIONS=%s\n' \
        'detect_leaks=1:halt_on_error=1:abort_on_error=1:verify_asan_link_order=1'
    } >"$envfile"
    set +e
    timeout 120s ros2 launch "$launch_file" \
      "control_mode:=$mode" \
      "mre_variant:=$variant" \
      teardown_mode:=explicit \
      "native_samples:=$samples" >"$log" 2>&1
    status=$?
    set -e
    printf '%s\n' "$status" >"$rcfile"
  done
}

run_control C4 C4 A 3
run_control C5 C5 A 3
run_control C6 C6 A 3
run_control C7 C6 B 3
run_control C8 C6 C 3
