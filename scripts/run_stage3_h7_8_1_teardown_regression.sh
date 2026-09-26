#!/usr/bin/env bash
# Fresh non-sanitizer regression for the three H7.8 MoveItPy shutdown modes.

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
source "$root/stage3_h7_8_overlay/install/final/setup.bash"
source "$root/stage3_h7_8_overlay/install/mre2/setup.bash"
set -u

ulimit -c 0
overall=0
for mode in explicit destructor_only normal_exit; do
  log="$raw_dir/teardown_${mode}_001.log"
  set +e
  timeout 120s ros2 launch stage3_h7_8_mre mre.launch.py \
    mre_variant:=A \
    "teardown_mode:=$mode" \
    "native_samples:=$samples" >"$log" 2>&1
  launch_rc=$?
  set -e
  printf '%s\n' "$launch_rc" >"$raw_dir/teardown_${mode}_001_launch_returncode.txt"
  if [[ "$launch_rc" -ne 0 ]] || grep -q 'process has died\|exit code -11\|Segmentation fault\|Traceback (most recent call last)' "$log" || ! grep -q 'process has finished cleanly' "$log"; then
    overall=1
  fi
done
printf '%s\n' "$overall" >"$raw_dir/teardown_regression_returncode.txt"
exit "$overall"
