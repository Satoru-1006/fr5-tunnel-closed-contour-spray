#!/usr/bin/env bash
set -eo pipefail

source /opt/ros/jazzy/setup.bash
source /mnt/d/robotfucker/ros2_overlay/install_h3/setup.bash
set -u
export STAGE28SR2_RUNTIME_INSTANCE_ID="${STAGE28SR2_RUNTIME_INSTANCE_ID:-h3-probe-runtime}"

probe_dir="/tmp/stage28sr2_h3_probe_${BASHPID}"
probe_log="/tmp/stage28sr2_h3_probe_${BASHPID}.log"
node_name="stage28sr2_recorder_h3probe_${BASHPID}"
ros2 bag record \
  --storage mcap \
  --output "${probe_dir}" \
  --max-cache-size 0 \
  --node-name "${node_name}" \
  --disable-keyboard-controls \
  --include-unpublished-topics \
  --topics /joint_states >"${probe_log}" 2>&1 &
recorder_pid=$!

finish() {
  if kill -0 "${recorder_pid}" 2>/dev/null; then
    kill -TERM "${recorder_pid}" || true
    wait "${recorder_pid}" || true
  fi
}
trap finish EXIT

sleep 4
echo "recorder_pid=${recorder_pid}"
timeout 10s python3 /mnt/d/robotfucker/scripts/stage28sr2_recorder_live_loss_query.py \
  --service "/${node_name}/stage28sr2_live_transport_loss" \
  --timeout 5
loaded_library=$(awk '/librosbag2_transport[.]so/ {print $6; exit}' "/proc/${recorder_pid}/maps")
echo "loaded_library=${loaded_library}"
sha256sum "${loaded_library}"
finish
trap - EXIT
grep -E 'Recording stopped|Number of messages lost' "${probe_log}" || true
