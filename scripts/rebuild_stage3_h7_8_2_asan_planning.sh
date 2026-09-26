#!/usr/bin/env bash
set -eo pipefail
root=/mnt/d/robotfucker
out=/home/robot/stage3_h7_8_2_sanitizers/asan
cp "$root/stage3_h7_8_overlay/src/moveit2_jazzy/moveit_ros/planning/trajectory_execution_manager/src/trajectory_execution_manager.cpp" \
  "$out/src/moveit2_jazzy/moveit_ros/planning/trajectory_execution_manager/src/trajectory_execution_manager.cpp"
source /opt/ros/jazzy/setup.bash
source "$root/install/setup.bash"
sanitize_flag='-fsanitize=address'
cmake_flags="$sanitize_flag -fno-omit-frame-pointer -g"
colcon --log-base "$out/log-rebuild" build \
  --base-paths "$out/src/moveit2_jazzy" \
  --packages-select moveit_ros_planning \
  --allow-overriding moveit_ros_planning \
  --build-base "$out/build" \
  --install-base "$out/install" \
  --merge-install \
  --event-handlers console_direct+ \
  --cmake-args \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DBUILD_TESTING=OFF \
    "-DCMAKE_CXX_FLAGS=$cmake_flags" \
    "-DCMAKE_SHARED_LINKER_FLAGS=$sanitize_flag" \
    "-DCMAKE_MODULE_LINKER_FLAGS=$sanitize_flag" \
    "-DCMAKE_EXE_LINKER_FLAGS=$sanitize_flag" \
  2>&1 | tee "$out/rebuild_console.log"
