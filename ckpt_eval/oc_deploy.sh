#!/usr/bin/env bash
# 2b. OpenClaw for trying a model by hand: switch the main OpenClaw ($OC_SOURCE_HOME/.openclaw) to a
#     local model, with the task's AGENTS.md and the two IVF tools, after backing everything up.
#
#   bash ckpt_eval/oc_deploy.sh <model_port> <model_id>
#   e.g. bash ckpt_eval/oc_deploy.sh 30001 ivf-step100
#
# The other providers stay in the config. A running gateway picks the change up by itself.
# Undo with: bash ckpt_eval/oc_restore.sh <backup_dir>   (the backup directory is printed below)
set -euo pipefail
source "$(dirname "$0")/common.sh"
MODEL_PORT="${1:?usage: oc_deploy.sh <model_port> <model_id>}"
MODEL_ID="${2:?usage: oc_deploy.sh <model_port> <model_id>}"
OC="$OC_SOURCE_HOME/.openclaw"
[ -f "$OC/openclaw.json" ] || die "no OpenClaw config at $OC"
[ -n "$(models_at "$MODEL_PORT")" ] || die "nothing answers on port $MODEL_PORT (start it with serve.sh)"

BACKUP="$OC_SOURCE_HOME/openclaw_backup_$(date +%m%d%H%M%S)"
mkdir -p "$BACKUP"
cp "$OC/openclaw.json" "$BACKUP/"
cp "$OC"/workspace/*.md "$BACKUP/" 2>/dev/null || true
echo "backup: $BACKUP"

"$PYTHON" "$HERE/oc_config.py" "$OC/openclaw.json" --provider "$OC_PROVIDER" \
  --base-url "http://127.0.0.1:$MODEL_PORT/v1" --model "$MODEL_ID"
(export HOME="$OC_SOURCE_HOME"; "${OPENCLAW_BIN:-openclaw}" config validate >/dev/null && bash "$REPO/openclaw_eval/prepare.sh" | tail -n 2 \
  && source "$REPO/openclaw_eval/common.sh" && bash "$REPO/task/scripts/start_services.sh" | tail -n 2 | cut -c1-80)
echo "deployed: default model $OC_PROVIDER/$MODEL_ID; try: openclaw agent --session-id demo --message '你好'"
echo "undo:     bash ckpt_eval/oc_restore.sh $BACKUP"
