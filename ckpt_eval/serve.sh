#!/usr/bin/env bash
# Serve one Hugging Face model with SGLang in the background and wait until it answers.
#
#   bash ckpt_eval/serve.sh <model_dir> <served_name> <port> <gpu>
#   e.g. bash ckpt_eval/serve.sh ~/Dressage/models/Qwen3-4B-ivf-step100-hf ivf-step100 30001 1
#
# Environment: HOST (default 127.0.0.1; 0.0.0.0 lets other machines use it), TOOL_PARSER (default qwen25),
# MEM_FRACTION (default 0.8), WAIT (seconds, default 600). Log: $SGLANG_LOG_DIR/<served_name>.log
# Stop it with: pkill -f "served-model-name <served_name>"
set -euo pipefail
source "$(dirname "$0")/common.sh"
MODEL="${1:?usage: serve.sh <model_dir> <served_name> <port> <gpu>}"
NAME="${2:?usage: serve.sh <model_dir> <served_name> <port> <gpu>}"
PORT="${3:?usage: serve.sh <model_dir> <served_name> <port> <gpu>}"
GPU="${4:?usage: serve.sh <model_dir> <served_name> <port> <gpu>}"
[ -f "$MODEL/config.json" ] || die "$MODEL is not a Hugging Face model (no config.json)"
[ -z "$(models_at "$PORT")" ] || die "port $PORT already serves a model: $(models_at "$PORT" | head -c 150)"
mkdir -p "$SGLANG_LOG_DIR"
LOG="$SGLANG_LOG_DIR/$NAME.log"

CUDA_VISIBLE_DEVICES="$GPU" SGLANG_ENABLE_SPEC_V2=True setsid nohup "$PYTHON" -m sglang.launch_server \
  --model "$MODEL" --served-model-name "$NAME" --tool-call-parser "${TOOL_PARSER:-qwen25}" \
  --host "${HOST:-127.0.0.1}" --port "$PORT" --mem-fraction-static "${MEM_FRACTION:-0.8}" > "$LOG" 2>&1 &
pid=$!
echo "starting $NAME on port $PORT (GPU $GPU, pid $pid, log $LOG)"
deadline=$(( $(date +%s) + ${WAIT:-600} ))
until models_at "$PORT" | grep -q "\"$NAME\""; do
  kill -0 "$pid" 2>/dev/null || { tail -n 20 "$LOG" >&2; die "SGLang exited; see $LOG"; }
  [ "$(date +%s)" -lt "$deadline" ] || die "not ready after ${WAIT:-600}s; see $LOG"
  sleep 5
done
echo "ready: http://${HOST:-127.0.0.1}:$PORT/v1  model $NAME"
