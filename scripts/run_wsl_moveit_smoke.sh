#!/usr/bin/env bash
set -eo pipefail

source /opt/ros/jazzy/setup.bash
source "$HOME/fr5_ros2_ws/install/setup.bash"

repo_root="${REPO_ROOT:-/mnt/c/Users/86198/Desktop/robotfucker}"
timeout_sec="${SMOKE_TIMEOUT_SEC:-90}"

timeout "$timeout_sec" ros2 launch fr5_tunnel_moveit_bridge fr5_spray_smoke.launch.py \
  tool_tcp_xyz:="0.000 0.000 0.150" \
  tool_tcp_rpy:="0 0 0" \
  tcp_path_csv:="$repo_root/outputs/tcp_poses.csv"
