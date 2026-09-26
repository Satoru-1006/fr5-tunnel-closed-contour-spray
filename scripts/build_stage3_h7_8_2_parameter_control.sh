#!/usr/bin/env bash
set -eo pipefail
root=/mnt/d/robotfucker
source /opt/ros/jazzy/setup.bash
colcon --log-base "$root/stage3_h7_8_overlay/log/mre2" build \
  --base-paths "$root/stage3_h7_8_overlay/src/stage3_h7_8_mre" \
  --packages-select stage3_h7_8_mre \
  --build-base "$root/stage3_h7_8_overlay/build/mre2" \
  --install-base "$root/stage3_h7_8_overlay/install/mre2" \
  --event-handlers console_direct+
