#!/usr/bin/env bash
# Run every case: before_agent -> agent -> after_agent, then summarise.
#
#   bash openclaw_eval/run_all.sh <samples_dir> [eval_dir]
#
# <samples_dir> comes from dataset/build_samples.py (cases/<id>/prompt.md, metadata.json, service.env).
# Results: <eval_dir>/cases/<id>/ (default eval_dir: <samples_dir>/eval_openclaw) and
# <eval_dir>/summary.json. Finished cases are skipped, so rerunning resumes.
# Run openclaw_eval/prepare.sh once before.
set -uo pipefail
source "$(dirname "$0")/common.sh"
SAMPLES="$(cd "${1:?usage: run_all.sh <samples_dir> [eval_dir]}" && pwd)"
OUT="${2:-$SAMPLES/eval_openclaw}"
mkdir -p "$OUT/cases"
HERE="$(cd "$(dirname "$0")" && pwd)"
[ -d "$TOOLS_DIR" ] || die "run openclaw_eval/prepare.sh first"

failed=0
for case_dir in "$SAMPLES"/cases/*/; do
  id="$(basename "$case_dir")"; res="$OUT/cases/$id"
  if [ -f "$res/score.json" ]; then echo "$id: done, skipped"; continue; fi
  rm -rf "$res"; mkdir -p "$res"
  echo "== $id"
  bash "$HERE/before_agent.sh" "$case_dir" "$res" \
    && bash "$HERE/agent.sh" "$case_dir" "$res" \
    && bash "$HERE/after_agent.sh" "$case_dir" "$res" \
    || { echo "$id: FAILED (logs in $res)"; failed=$((failed + 1)); }
done
bash "$REPO/task/scripts/stop_services.sh" >/dev/null 2>&1 || true

ls "$OUT"/cases/*/score.json >/dev/null 2>&1 && "$PYTHON" "$REPO/dataset/summarize_eval.py" "$OUT"
[ "$failed" = 0 ] || { echo "$failed case(s) failed"; exit 1; }
