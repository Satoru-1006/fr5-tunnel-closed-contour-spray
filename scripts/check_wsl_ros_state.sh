#!/usr/bin/env bash
set -eo pipefail

echo "== processes =="
ps aux | grep -E "apt|dpkg|setup_ros|colcon" | grep -v grep || true

echo "== ros2 apt source =="
test -f /etc/apt/sources.list.d/ros2.list && cat /etc/apt/sources.list.d/ros2.list || true

echo "== installed packages =="
dpkg -l | grep -E "ros-jazzy-(desktop|moveit|moveit-py)|python3-colcon|python3-rosdep" || true

echo "== ros command =="
if test -f /opt/ros/jazzy/setup.bash; then
  source /opt/ros/jazzy/setup.bash
  ros2 --help >/dev/null && echo "ros2 command ok" || true
else
  echo "/opt/ros/jazzy/setup.bash missing"
fi

echo "== moveit_py =="
python3 - <<'PY'
try:
    from moveit.core.robot_trajectory import RobotTrajectory
    print("moveit_py import ok")
    print("apply_ruckig_smoothing:", hasattr(RobotTrajectory, "apply_ruckig_smoothing"))
except Exception as exc:
    print("moveit_py import failed:", repr(exc))
PY
