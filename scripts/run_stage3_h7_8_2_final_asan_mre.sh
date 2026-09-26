#!/usr/bin/env bash
# Run a numbered final-patch ASan MRE shard.  LSan remains fully enabled.

set -eo pipefail
if [[ $# -ne 5 ]]; then
  echo "usage: $0 OUTPUT_DIRECTORY ID VARIANT COUNT START_INDEX" >&2
  exit 64
fi
root=/mnt/d/robotfucker
out="$1"
id="$2"
variant="$3"
count="$4"
start="$5"
mkdir -p "$out"
samples="$root/outputs/stage3_h7_7_tier_b_remediation_20260809T143600Z/formal_candidate_2/native_probe/stage25r_native_samples.jsonl"
launch_file="$root/scripts/stage3_h7_8_2_asan_mre.launch.py"
asan_overlay="${H782_ASAN_OVERLAY:-/home/robot/stage3_h7_8_1_sanitizers/asan}"
source /opt/ros/jazzy/setup.bash
source "$root/install/setup.bash"
source "$asan_overlay/install/setup.bash"
source "$root/stage3_h7_8_overlay/install/mre2/setup.bash"
set -u
ulimit -c 0
for ((offset=0; offset<count; offset++)); do
  n=$((start + offset))
  printf -v log '%s/%s_%03d.log' "$out" "$id" "$n"
  printf -v rcfile '%s/%s_%03d.exitcode' "$out" "$id" "$n"
  printf -v envfile '%s/%s_%03d.env' "$out" "$id" "$n"
  {
    declare -px
    printf 'COMMAND=ros2 launch %s control_mode:=C6 mre_variant:=%s teardown_mode:=explicit native_samples:=%s\n' \
      "$launch_file" "$variant" "$samples"
    printf 'ASAN_OVERLAY=%s\n' "$asan_overlay"
    printf 'CHILD_LD_PRELOAD=%s\n' \
      '/usr/lib/gcc/x86_64-linux-gnu/13/libasan.so:/usr/lib/x86_64-linux-gnu/libstdc++.so.6'
    printf 'CHILD_ASAN_OPTIONS=%s\n' \
      'detect_leaks=1:halt_on_error=1:abort_on_error=1:verify_asan_link_order=1'
  } >"$envfile"
  set +e
  timeout 120s ros2 launch "$launch_file" \
    control_mode:=C6 \
    "mre_variant:=$variant" \
    teardown_mode:=explicit \
    "native_samples:=$samples" >"$log" 2>&1
  status=$?
  set -e
  printf '%s\n' "$status" >"$rcfile"
done
