#!/usr/bin/env bash
# C2M: exact MRE imports plus rclpy init/shutdown, without MoveItPy construction.

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
code="import rclpy; from moveit.core.collision_detection import CollisionRequest, CollisionResult; from moveit.core.robot_state import RobotState; from moveit.planning import MoveItPy; rclpy.init(); rclpy.shutdown()"
ulimit -c 0
for n in 1 2 3; do
  printf -v log '%s/C2M_%03d.log' "$out" "$n"
  printf -v rcfile '%s/C2M_%03d.exitcode' "$out" "$n"
  printf -v envfile '%s/C2M_%03d.env' "$out" "$n"
  {
    declare -px
    printf 'COMMAND=python3 -c %q\n' "$code"
    printf 'LD_PRELOAD=%s\nASAN_OPTIONS=%s\n' "$asan_runtime" "$asan_options"
    printf 'MOVEITPY_CONSTRUCTED=NO\n'
  } >"$envfile"
  set +e
  LD_PRELOAD="$asan_runtime" ASAN_OPTIONS="$asan_options" python3 -c \
    "${code}; print('H782_LOADED_LIBRARIES_BEGIN'); print(''.join(sorted({line.split()[-1] for line in open('/proc/self/maps', encoding='utf-8') if '/' in line}))); print('H782_LOADED_LIBRARIES_END')" \
    >"$log" 2>&1
  status=$?
  set -e
  printf '%s\n' "$status" >"$rcfile"
done
