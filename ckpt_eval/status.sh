#!/usr/bin/env bash
# Progress of run_eval.sh: finished and failed cases per model.
#
#   bash ckpt_eval/status.sh <samples_name> <name> [<name> ...]      e.g. status.sh test283 step0 step100
set -uo pipefail
source "$(dirname "$0")/common.sh"
SET="${1:?usage: status.sh <samples_name> <name>...}"; shift
for name in "$@"; do
  H="$OC_EVAL_ROOT/$name"
  total=$(ls -d "$REPO"/runs/"$SET"/cases/*/ 2>/dev/null | wc -l)
  done_=$(ls "$H/$SET"/cases/*/score.json 2>/dev/null | wc -l)
  failed=$(grep -c FAILED "$H/run_$SET.log" 2>/dev/null || true)
  state=running; pgrep -f "run_all.sh .*$H/$SET" >/dev/null || state=stopped
  [ "$total" -gt 0 ] || total="?"
  echo "$name: $done_/$total done, ${failed:-0} failed, $state"
done
