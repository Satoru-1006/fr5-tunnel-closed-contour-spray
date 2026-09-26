#!/usr/bin/env bash
set -eo pipefail
if [[ $# -ne 1 ]]; then
  echo "usage: $0 OUTPUT_DIRECTORY" >&2
  exit 64
fi
root=/mnt/d/robotfucker
out="$1"
mkdir -p "$out"
source /opt/ros/jazzy/setup.bash
source "$root/install/setup.bash"
set +e
python3 -m pytest -q \
  tests/test_stage3_h7_7.py tests/test_stage3_h7_6.py tests/test_stage3_h7_5.py \
  tests/test_stage3_h7_4.py tests/test_stage3_h7_3.py tests/test_stage3_h7_2.py \
  tests/test_stage3_h7_1.py tests/test_stage3_h7.py tests/test_stage3_h6_4.py \
  tests/test_stage3_h6_3.py tests/test_stage3_h6_2.py tests/test_stage3_h6_1.py \
  tests/test_stage3_h6.py >"$out/regression_raw.log" 2>&1
status=$?
set -e
printf '%s\n' "$status" >"$out/regression_returncode.txt"
exit "$status"
