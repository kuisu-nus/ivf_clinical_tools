#!/usr/bin/env bash
# Stage 2 (agent): one OpenClaw turn on the case text.
#
#   bash openclaw_eval/agent.sh <case_dir> <result_dir>
#
# Environment: OC_TIMEOUT (seconds, default 600), OC_LOCAL=1 to run the embedded agent
# (openclaw agent --local) instead of going through the running Gateway.
# Writes <result_dir>/answer.txt (the reply, decoration lines removed), openclaw_raw.txt,
# openclaw_stderr.txt, dispatches.jsonl (the tool calls) and agent_run.json.
set -euo pipefail
source "$(dirname "$0")/common.sh"
CASE="$(cd "${1:?usage: agent.sh <case_dir> <result_dir>}" && pwd)"
RES="${2:?usage: agent.sh <case_dir> <result_dir>}"
mkdir -p "$RES"
id="$(basename "$CASE")"
session="ivf-$id-$(date +%s)"                 # a new session per run: no earlier conversation
LOCAL=(); [ "${OC_LOCAL:-0}" = 1 ] && LOCAL=(--local)

started=$(date +%s)
code=0
"$OPENCLAW" agent ${LOCAL[@]+"${LOCAL[@]}"} --session-id "$session" --message "$(cat "$CASE/prompt.md")" \
  --timeout "${OC_TIMEOUT:-600}" > "$RES/openclaw_raw.txt" 2> "$RES/openclaw_stderr.txt" || code=$?
duration=$(( $(date +%s) - started ))

# OpenClaw prints "│" / "◇" progress lines before the reply; keep only the reply.
grep -v -E '^[[:space:]]*[│◇]([[:space:]]|[│◇])*$' "$RES/openclaw_raw.txt" | sed '/./,$!d' > "$RES/answer.txt" || true
if [ -f "$TRACE_DIR/dispatches.jsonl" ]; then cp "$TRACE_DIR/dispatches.jsonl" "$RES/dispatches.jsonl"
else : > "$RES/dispatches.jsonl"; fi
calls=$(grep -c . "$RES/dispatches.jsonl" || true)

"$PYTHON" - "$RES/agent_run.json" "$session" "$duration" "$code" "$calls" <<'PY'
import json, sys
out, session, duration, code, calls = sys.argv[1:]
json.dump({"session_id": session, "duration_ms": int(duration) * 1000, "openclaw_exit": int(code),
           "tool_calls": int(calls)}, open(out, "w"), indent=2)
PY
[ "$code" = 0 ] || echo "warning: openclaw exited $code; see $RES/openclaw_stderr.txt" >&2
echo "answer: $(tail -n 1 "$RES/answer.txt")  (tool calls: $calls, ${duration}s)"
