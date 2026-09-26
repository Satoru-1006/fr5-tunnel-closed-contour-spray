#!/usr/bin/env bash
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
source /home/robot/stage3_h7_8_2_controls/install/setup.bash
set -u

exe=/home/robot/stage3_h7_8_2_controls/install/lib/stage3_h7_8_controls/rclcpp_action_control
asan_options=detect_leaks=1:halt_on_error=1:abort_on_error=1:verify_asan_link_order=1
ulimit -c 0
ldd "$exe" >"$out/C3_loaded_libraries.txt"

for n in 1 2 3; do
  printf -v log '%s/C3_%03d.log' "$out" "$n"
  printf -v rcfile '%s/C3_%03d.exitcode' "$out" "$n"
  printf -v envfile '%s/C3_%03d.env' "$out" "$n"
  {
    declare -px
    printf 'COMMAND=%s\n' "$exe"
    printf 'ASAN_OPTIONS=%s\n' "$asan_options"
    printf 'NEW_FJT_GOALS_SENT=0\n'
  } >"$envfile"
  set +e
  ASAN_OPTIONS="$asan_options" "$exe" >"$log" 2>&1
  status=$?
  set -e
  printf '%s\n' "$status" >"$rcfile"
done
