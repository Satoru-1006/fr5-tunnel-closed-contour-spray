#!/usr/bin/env bash
set -eo pipefail
if [[ $# -ne 1 ]]; then
  echo "usage: $0 OUTPUT_DIRECTORY" >&2
  exit 64
fi
root=/mnt/d/robotfucker
out="$1"
mkdir -p "$out"
launch_file="$root/scripts/stage3_h7_8_2_plugin_control.launch.py"
source /opt/ros/jazzy/setup.bash
source "$root/install/setup.bash"
source /home/robot/stage3_h7_8_2_controls/install/setup.bash
set -u
ulimit -c 0
for n in 1 2 3; do
  printf -v log '%s/C3P_%03d.log' "$out" "$n"
  printf -v rcfile '%s/C3P_%03d.exitcode' "$out" "$n"
  printf -v envfile '%s/C3P_%03d.env' "$out" "$n"
  {
    declare -px
    printf 'COMMAND=ros2 launch %s\n' "$launch_file"
    printf 'MOVEITPY_CONSTRUCTED=NO\nTEM_CONSTRUCTED=NO\nNEW_FJT_GOALS_SENT=0\n'
    printf 'CHILD_ASAN_OPTIONS=%s\n' \
      'detect_leaks=1:halt_on_error=1:abort_on_error=1:verify_asan_link_order=1'
  } >"$envfile"
  set +e
  timeout 120s ros2 launch "$launch_file" >"$log" 2>&1
  status=$?
  set -e
  printf '%s\n' "$status" >"$rcfile"
done
