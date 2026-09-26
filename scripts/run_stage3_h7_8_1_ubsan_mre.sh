#!/usr/bin/env bash
# Run fresh UBSan-instrumented H7.8 MRE-A/B/C processes and retain all logs.

set -eo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 OUTPUT_DIRECTORY" >&2
  exit 64
fi

root=/mnt/d/robotfucker
raw_dir="$1"
samples="$root/outputs/stage3_h7_7_tier_b_remediation_20260809T143600Z/formal_candidate_2/native_probe/stage25r_native_samples.jsonl"
mkdir -p "$raw_dir"

source /opt/ros/jazzy/setup.bash
source "$root/install/setup.bash"
source /home/robot/stage3_h7_8_1_sanitizers/ubsan/install/setup.bash
source "$root/stage3_h7_8_overlay/install/mre2/setup.bash"
set -u

export UBSAN_OPTIONS='print_stacktrace=1:halt_on_error=1:report_error_type=1'
ulimit -c 0
overall=0

for variant in A B C; do
  log="$raw_dir/ubsan_mre_${variant}_001.log"
  set +e
  timeout 120s ros2 launch stage3_h7_8_mre mre.launch.py \
    "mre_variant:=$variant" \
    teardown_mode:=explicit \
    "native_samples:=$samples" >"$log" 2>&1
  launch_rc=$?
  set -e
  printf '%s\n' "$launch_rc" >"$raw_dir/ubsan_mre_${variant}_001_launch_returncode.txt"
  if [[ "$launch_rc" -ne 0 ]] || grep -q 'process has died\|runtime error:' "$log" || ! grep -q 'process has finished cleanly' "$log"; then
    overall=1
  fi
done

printf '%s\n' "$overall" >"$raw_dir/ubsan_mre_matrix_returncode.txt"
exit "$overall"
