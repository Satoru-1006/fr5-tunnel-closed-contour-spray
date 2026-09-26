#!/usr/bin/env bash
set -eo pipefail

output_dir="$1"
native_dir="${output_dir}/native_probe"
mkdir -p "${native_dir}"

source /opt/ros/jazzy/setup.bash
source /mnt/d/robotfucker/install/setup.bash
source /mnt/d/robotfucker/install/stage3_h7_4_native/setup.bash
export LD_PRELOAD=/mnt/d/robotfucker/outputs/stage3_h7_7_tier_b_remediation_20260809T143600Z/libstage3_h7_4_ruckig_interposer.so
export STAGE25R_NATIVE_DIR="${native_dir}"

ros2 launch fr5_tunnel_moveit_bridge stage3_h7_7_native.launch.py \
  output_dir:="${output_dir}" \
  h6_4_final_validation:=/mnt/d/robotfucker/outputs/stage3_h6_4_native_valid_exact_ik_graph_20260809T150000Z/stage3_h6_4_final_process_validation.jsonl \
  h6_4_segments:=/mnt/d/robotfucker/outputs/stage3_h6_4_native_valid_exact_ik_graph_20260809T150000Z/stage3_h6_4_spray_on_off_segments.json \
  process_contract:=/mnt/d/robotfucker/config/stage3/stage3_h6_1_spray_process_tolerance_contract.json \
  fixture_mesh:=/mnt/d/robotfucker/outputs/stage3_h4_5_robot_fixture_geometry_codesign_20260809T000000Z/curved_fixture_stage3_h4_5_selected.obj \
  totg_parameters:=/mnt/d/robotfucker/outputs/stage3_h7_3_target55_controlled_stop_20260809T170900Z/stage3_h7_3_totg_parameters.json \
  tier_b_plan:=/mnt/d/robotfucker/outputs/stage3_h7_7_tier_b_remediation_20260809T143600Z/tier_b_alpha_plan.json
