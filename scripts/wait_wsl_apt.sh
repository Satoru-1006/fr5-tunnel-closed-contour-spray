#!/usr/bin/env bash
set -euo pipefail

while ps aux | grep -E "apt-get install|/usr/bin/dpkg" | grep -v grep >/dev/null; do
  date
  ps aux | grep -E "apt-get install|/usr/bin/dpkg" | grep -v grep
  sleep 30
done

echo "apt_done"
