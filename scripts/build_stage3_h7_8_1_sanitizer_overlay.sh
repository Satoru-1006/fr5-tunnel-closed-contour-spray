#!/usr/bin/env bash
# Build an isolated, instrumented copy of only the locally patched MoveIt packages.
# This script deliberately never writes to /opt/ros/jazzy or the formal H7.8 overlay.

set -eo pipefail

if [[ $# -ne 1 || ( "$1" != "asan" && "$1" != "ubsan" ) ]]; then
  echo "usage: $0 {asan|ubsan}" >&2
  exit 64
fi

kind="$1"
root=/mnt/d/robotfucker
source_tree="$root/stage3_h7_8_overlay/src/moveit2_jazzy"
sanitizer_root="/home/robot/stage3_h7_8_1_sanitizers/$kind"

if [[ ! -d "$source_tree/.git" ]]; then
  echo "missing patched MoveIt source: $source_tree" >&2
  exit 66
fi

if [[ -e "$sanitizer_root" ]]; then
  echo "refusing to overwrite an existing sanitizer overlay: $sanitizer_root" >&2
  exit 73
fi

mkdir -p "$sanitizer_root/src" "$sanitizer_root/log"
rsync -a --exclude=.git "$source_tree/" "$sanitizer_root/src/moveit2_jazzy/"

case "$kind" in
  asan)
    sanitize_flag='-fsanitize=address'
    ;;
  ubsan)
    sanitize_flag='-fsanitize=undefined -fno-sanitize-recover=undefined'
    ;;
esac

cmake_flags="$sanitize_flag -fno-omit-frame-pointer -g"

source /opt/ros/jazzy/setup.bash
source "$root/install/setup.bash"
set -u

set +e
set -o pipefail
colcon --log-base "$sanitizer_root/log" build \
  --base-paths "$sanitizer_root/src/moveit2_jazzy" \
  --packages-select moveit_py moveit_ros_planning \
  --allow-overriding moveit_py moveit_ros_planning \
  --build-base "$sanitizer_root/build" \
  --install-base "$sanitizer_root/install" \
  --merge-install \
  --event-handlers console_direct+ \
  --cmake-args \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DBUILD_TESTING=OFF \
    "-DCMAKE_CXX_FLAGS=$cmake_flags" \
    "-DCMAKE_SHARED_LINKER_FLAGS=$sanitize_flag" \
    "-DCMAKE_MODULE_LINKER_FLAGS=$sanitize_flag" \
    "-DCMAKE_EXE_LINKER_FLAGS=$sanitize_flag" \
  2>&1 | tee "$sanitizer_root/build_console.log"
build_status=${PIPESTATUS[0]}
set -e

printf '%s\n' "$build_status" > "$sanitizer_root/build_exit_code.txt"
exit "$build_status"
