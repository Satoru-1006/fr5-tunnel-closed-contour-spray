#!/usr/bin/env bash
# Fresh-process C0/C1/C2 controls under the exact H7.8.1 ASan environment.

set -eo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 OUTPUT_DIRECTORY" >&2
  exit 64
fi

root=/mnt/d/robotfucker
out="$1"
mkdir -p "$out"

source /opt/ros/jazzy/setup.bash
source "$root/install/setup.bash"
source /home/robot/stage3_h7_8_1_sanitizers/asan/install/setup.bash
source "$root/stage3_h7_8_overlay/install/mre2/setup.bash"
set -u

asan_runtime=/usr/lib/gcc/x86_64-linux-gnu/13/libasan.so:/usr/lib/x86_64-linux-gnu/libstdc++.so.6
asan_options=detect_leaks=1:halt_on_error=1:abort_on_error=1:verify_asan_link_order=1
ulimit -c 0

run_control() {
  local id="$1"
  local code="$2"
  local n log rcfile envfile
  for n in 1 2 3; do
    printf -v log '%s/%s_%03d.log' "$out" "$id" "$n"
    printf -v rcfile '%s/%s_%03d.exitcode' "$out" "$id" "$n"
    printf -v envfile '%s/%s_%03d.env' "$out" "$id" "$n"
    {
      declare -px
      printf 'COMMAND=python3 -c %q\n' "$code"
      printf 'PYTHON=%s\n' "$(command -v python3)"
      printf 'ASAN_SYMBOLIZER_PATH=%s\n' "${ASAN_SYMBOLIZER_PATH:-}"
      printf 'LD_PRELOAD=%s\n' "$asan_runtime"
      printf 'ASAN_OPTIONS=%s\n' "$asan_options"
      printf 'LOADED_LIBRARIES_SOURCE=/proc/self/maps emitted by child\n'
    } >"$envfile"
    set +e
    LD_PRELOAD="$asan_runtime" ASAN_OPTIONS="$asan_options" python3 -c \
      "${code}; print('H782_LOADED_LIBRARIES_BEGIN'); print(''.join(sorted({line.split()[-1] for line in open('/proc/self/maps', encoding='utf-8') if '/' in line}))); print('H782_LOADED_LIBRARIES_END')" \
      >"$log" 2>&1
    status=$?
    set -e
    printf '%s\n' "$status" >"$rcfile"
  done
}

run_control C0 "pass"
run_control C1 "import moveit.planning"
run_control C2 "import rclpy; rclpy.init(); rclpy.shutdown()"
