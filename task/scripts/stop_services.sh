#!/usr/bin/env bash
set -euo pipefail
TASK_DIR="${OMNI_TASK_DIR:-/tmp/omni_task}"
for pid_file in "$TASK_DIR"/service-*.pid; do
  [ -e "$pid_file" ] || continue
  kill "$(cat "$pid_file")" 2>/dev/null || true
  rm -f "$pid_file"
done
