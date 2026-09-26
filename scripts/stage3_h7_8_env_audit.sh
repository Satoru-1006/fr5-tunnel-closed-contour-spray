#!/usr/bin/env bash
source /opt/ros/jazzy/setup.bash
set -u
printf 'ROS_DISTRO=%s\n' "${ROS_DISTRO:-not_set}"
printf 'RMW_IMPLEMENTATION=%s\n' "${RMW_IMPLEMENTATION:-default}"
printf 'GDB=%s\n' "$(command -v gdb || printf not_available)"
printf 'COREDUMPCTL=%s\n' "$(command -v coredumpctl || printf not_available)"
dpkg-query -W -f='${Package}\t${Version}\n' 'ros-jazzy-moveit*' 'ros-jazzy-rclcpp' 'ros-jazzy-rclpy' 2>/dev/null | sort
gcc --version | head -1
python3 - <<'PY'
import platform
import moveit.planning
import rclpy
print(f"python={platform.python_version()}")
print(f"moveit_planning={moveit.planning.__file__}")
print(f"rclpy={rclpy.__file__}")
print("moveitpy_api=" + ",".join(sorted(name for name in dir(moveit.planning.MoveItPy) if not name.startswith("__"))))
PY
