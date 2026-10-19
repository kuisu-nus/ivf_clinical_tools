#!/usr/bin/env bash
set -euo pipefail
for pid_file in /tmp/omni_task/service-*.pid; do
  [ -e "$pid_file" ] || continue
  kill "$(cat "$pid_file")" 2>/dev/null || true
done
