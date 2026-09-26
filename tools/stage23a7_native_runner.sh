#!/usr/bin/env bash
set -eo pipefail

backend="${1:?backend required}"
run_index="${2:?run index required}"
output_dir="${3:?output directory required}"
root="/mnt/c/Users/86198/Desktop/robotfucker"
base="$root/outputs/ik_graph_stage23a4_backend_equivalent/fr5_scaled_horseshoe_demo_v44"
manifest="$root/outputs/ik_graph_stage23a7_1_runtime_audit/disputed_node_ids.txt"

source /opt/ros/jazzy/setup.bash
set -u
export AMENT_PREFIX_PATH="$root/tmp/stage23a7_install2:$root/install/fairino5_v6_moveit2_config:$root/install/fairino_description:$root/install/fr5_tunnel_moveit_bridge:/opt/ros/jazzy"
export CMAKE_PREFIX_PATH="$AMENT_PREFIX_PATH"

exec ros2 launch "$root/tools/stage23a7_runtime_audit_launch.py" \
  "candidate_csv:=$base/deterministic_ik_candidates.csv" \
  "parts_dir:=$base/tunnel_collision_parts" \
  "backend:=$backend" \
  "run_index:=$run_index" \
  "disputed_manifest:=$manifest" \
  "output_dir:=$output_dir"
