#!/usr/bin/env bash
set -eo pipefail

root=/mnt/d/robotfucker
out=/home/robot/stage3_h7_8_2_controls
mkdir -p "$out"
source /opt/ros/jazzy/setup.bash
source "$root/install/setup.bash"
colcon --log-base "$out/log" build \
  --base-paths "$root/stage3_h7_8_overlay/src/stage3_h7_8_controls" \
  --packages-select stage3_h7_8_controls \
  --build-base "$out/build" \
  --install-base "$out/install" \
  --merge-install \
  --event-handlers console_direct+ \
  --cmake-args \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    "-DCMAKE_CXX_FLAGS=-fsanitize=address -fno-omit-frame-pointer -g" \
    "-DCMAKE_EXE_LINKER_FLAGS=-fsanitize=address" \
  2>&1 | tee "$out/build_console.log"
