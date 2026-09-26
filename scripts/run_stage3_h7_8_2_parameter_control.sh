#!/usr/bin/env bash
set -eo pipefail
if [[ $# -ne 1 ]]; then
  echo "usage: $0 OUTPUT_DIRECTORY" >&2
  exit 64
fi
root=/mnt/d/robotfucker
out="$1"
mkdir -p "$out"
samples="$root/outputs/stage3_h7_7_tier_b_remediation_20260809T143600Z/formal_candidate_2/native_probe/stage25r_native_samples.jsonl"
launch_file="$root/scripts/stage3_h7_8_2_parameter_control.launch.py"
source /opt/ros/jazzy/setup.bash
source "$root/install/setup.bash"
source /home/robot/stage3_h7_8_1_sanitizers/asan/install/setup.bash
source "$root/stage3_h7_8_overlay/install/mre2/setup.bash"
set -u
ulimit -c 0
for n in 1 2 3; do
  printf -v log '%s/C2P_%03d.log' "$out" "$n"
  printf -v rcfile '%s/C2P_%03d.exitcode' "$out" "$n"
  printf -v envfile '%s/C2P_%03d.env' "$out" "$n"
  {
    declare -px
    printf 'COMMAND=ros2 launch %s native_samples:=%s\n' "$launch_file" "$samples"
    printf 'MOVEITPY_CONSTRUCTED=NO\n'
    printf 'CHILD_LD_PRELOAD=%s\n' \
      '/usr/lib/gcc/x86_64-linux-gnu/13/libasan.so:/usr/lib/x86_64-linux-gnu/libstdc++.so.6'
    printf 'CHILD_ASAN_OPTIONS=%s\n' \
      'detect_leaks=1:halt_on_error=1:abort_on_error=1:verify_asan_link_order=1'
  } >"$envfile"
  set +e
  timeout 120s ros2 launch "$launch_file" "native_samples:=$samples" >"$log" 2>&1
  status=$?
  set -e
  printf '%s\n' "$status" >"$rcfile"
done
