#!/usr/bin/env bash
# 3. Evaluation: run a sample set through several models at once, each in its own OpenClaw home.
#
#   bash ckpt_eval/run_eval.sh <samples_dir> <name> [<name> ...]
#   e.g. bash ckpt_eval/run_eval.sh runs/test283 step0 step100 step400
#
# <samples_dir> comes from dataset/build_samples.py; each <name> from oc_home.sh. Every model runs
# openclaw_eval/run_all.sh (before_agent -> OpenClaw -> after_agent per case) in the background, writing
# $OC_EVAL_ROOT/<name>/<samples>/cases/<id>/ and summary.json; finished cases are skipped on a rerun.
# Follow it with ckpt_eval/status.sh, then compare with ckpt_eval/compare.py.
set -euo pipefail
source "$(dirname "$0")/common.sh"
SAMPLES="$(cd "${1:?usage: run_eval.sh <samples_dir> <name>...}" && pwd)"; shift
[ $# -gt 0 ] || die "usage: run_eval.sh <samples_dir> <name>..."
[ -d "$SAMPLES/cases" ] || die "$SAMPLES has no cases/ (build it with dataset/build_samples.py)"
SET="$(basename "$SAMPLES")"
for name in "$@"; do
  H="$OC_EVAL_ROOT/$name"
  [ -f "$H/env.sh" ] || die "$H/env.sh not found (run oc_home.sh $name ... first)"
  (source "$H/env.sh" && cd "$REPO" && setsid nohup bash openclaw_eval/run_all.sh "$SAMPLES" "$H/$SET" \
     > "$H/run_$SET.log" 2>&1 &)
  echo "started $name -> $H/$SET (log $H/run_$SET.log)"
done
echo "progress: bash ckpt_eval/status.sh $SET $*"
