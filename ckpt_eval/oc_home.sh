#!/usr/bin/env bash
# 2a. OpenClaw for evaluation: an isolated OpenClaw home per model, so several models run side by side.
#
#   bash ckpt_eval/oc_home.sh <name> <model_port> <tool_port>
#   e.g. bash ckpt_eval/oc_home.sh step100 30001 9321
#
# Copies $OC_SOURCE_HOME/.openclaw to $OC_EVAL_ROOT/<name>/.openclaw, points it at the model on
# 127.0.0.1:<model_port> (served name <name>, or MODEL_ID), puts the task's short workspace files in,
# writes $OC_EVAL_ROOT/<name>/env.sh (HOME, ports, work dir) and runs openclaw_eval/prepare.sh in it:
# system prompt + tool instructions in AGENTS.md, MCP server claw-tools on tool ports <tool_port>/<tool_port+1>.
# The original ~/.openclaw is not changed. FORCE=1 rebuilds an existing home.
set -euo pipefail
source "$(dirname "$0")/common.sh"
NAME="${1:?usage: oc_home.sh <name> <model_port> <tool_port>}"
MODEL_PORT="${2:?usage: oc_home.sh <name> <model_port> <tool_port>}"
TOOL_PORT="${3:?usage: oc_home.sh <name> <model_port> <tool_port>}"
MODEL_ID="${MODEL_ID:-$NAME}"
H="$OC_EVAL_ROOT/$NAME"
[ -f "$OC_SOURCE_HOME/.openclaw/openclaw.json" ] || die "no OpenClaw config at $OC_SOURCE_HOME/.openclaw"
if [ -d "$H/.openclaw" ]; then
  [ "${FORCE:-0}" = 1 ] || die "$H already exists (FORCE=1 rebuilds it)"
  rm -rf "$H/.openclaw" "$H/work"
fi
[ -n "$(models_at "$MODEL_PORT")" ] || echo "warning: nothing answers on port $MODEL_PORT yet (start it with serve.sh)" >&2

mkdir -p "$H"
cp -r "$OC_SOURCE_HOME/.openclaw" "$H/"
cp "$REPO"/task/workspace/*.md "$H/.openclaw/workspace/"
rm -f "$H/.openclaw/workspace/BOOTSTRAP.md"
"$PYTHON" "$HERE/oc_config.py" "$H/.openclaw/openclaw.json" --provider "$OC_PROVIDER" \
  --base-url "http://127.0.0.1:$MODEL_PORT/v1" --model "$MODEL_ID" --workspace "$H/.openclaw/workspace"

# HOME moves to the copy; PYTHONUSERBASE keeps the Python packages installed under the real home.
cat > "$H/env.sh" <<ENV
export HOME=$H PYTHONUSERBASE=$OC_SOURCE_HOME/.local OC_LOCAL=1 OC_EVAL_WORK=$H/work
export IVF_SIMILAR_PORT=$TOOL_PORT IVF_PREDICT_PORT=$((TOOL_PORT + 1)) OMNI_TASK_DIR=$H/work/run
export IVF_CASES_PORT=$((TOOL_PORT + 2)) IVF_CASE_SQL_PORT=$((TOOL_PORT + 3))
ENV
(source "$H/env.sh" && "${OPENCLAW_BIN:-openclaw}" config validate >/dev/null && bash "$REPO/openclaw_eval/prepare.sh") 2>&1 | tail -n 2
echo "ready: $H (model $OC_PROVIDER/$MODEL_ID on port $MODEL_PORT, tools on $TOOL_PORT–$((TOOL_PORT + 3)))"
echo "try it: (source $H/env.sh && openclaw agent --local --session-id test --message '你好')"
