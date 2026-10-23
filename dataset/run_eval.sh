#!/usr/bin/env bash
# Local evaluation of the staged outcome task: one agent run per case, scored by the task grader.
#
#   bash dataset/run_eval.sh <samples_dir> [eval_dir]
#
# <samples_dir> comes from dataset/build_samples.py (cases/<id>/prompt.md, metadata.json, service.env).
# The agent is the Claude Code CLI with only the two task tools (MCP server "claw-tools"); any backend
# works through the usual ANTHROPIC_* variables (e.g. DeepSeek). Finished cases are skipped, so an
# interrupted run can be restarted with the same command.
#
# Environment:
#   EVAL_LIMIT     evaluate at most N cases (default: all)
#   EVAL_MODEL     model passed to claude --model (default: the CLI's default)
#   EVAL_TIMEOUT   seconds per case (default 600)
#   CLAUDE_BIN     claude executable (default: claude)
#   PYTHON         python executable (default: python3)
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
SAMPLES="$(cd "${1:?usage: bash dataset/run_eval.sh <samples_dir> [eval_dir]}" && pwd)"
OUT="${2:-$SAMPLES/eval}"
mkdir -p "$OUT/cases"
OUT="$(cd "$OUT" && pwd)"
PYTHON="${PYTHON:-python3}"
CLAUDE="${CLAUDE_BIN:-claude}"
LIMIT="${EVAL_LIMIT:-0}"
export ENABLE_TOOL_SEARCH="${ENABLE_TOOL_SEARCH:-false}"
export OMNI_TASK_DIR="$OUT/services"            # pid files and service logs
SYSTEM_PROMPT="$(cat "$REPO/task/runtime/system_prompt_prefix.txt")"
MODEL_ARGS=()
[ -n "${EVAL_MODEL:-}" ] && MODEL_ARGS=(--model "$EVAL_MODEL")

stop() { bash "$REPO/task/scripts/stop_services.sh" >/dev/null 2>&1 || true; }
trap stop EXIT
running_env="none"
done_count=0

for case_dir in "$SAMPLES"/cases/*/; do
  id="$(basename "$case_dir")"
  res="$OUT/cases/$id"
  if [ -f "$res/score.json" ]; then continue; fi
  if [ "$LIMIT" -gt 0 ] && [ "$done_count" -ge "$LIMIT" ]; then break; fi

  # (Re)start the services when this case needs different settings (fold held out for training cases).
  case_env="$(cat "$case_dir/service.env")"
  if [ "$case_env" != "$running_env" ]; then
    stop
    (set -a; [ -n "$case_env" ] && . "$case_dir/service.env"; set +a
     bash "$REPO/task/scripts/start_services.sh" >/dev/null)
    running_env="$case_env"
  fi

  rm -rf "$res"; mkdir -p "$res/trace"
  cat > "$res/mcp.json" <<JSON
{"mcpServers": {"claw-tools": {"type": "stdio", "command": "$PYTHON",
  "args": ["$REPO/task/runtime/claw_tools_mcp.py"],
  "env": {"OMNI_MCP_SERVER_NAME": "claw-tools", "OMNI_TOOL_SPECS_DIR": "$REPO/task/tools",
          "OMNI_TRACE_DIR": "$res/trace", "NO_PROXY": "127.0.0.1,localhost", "no_proxy": "127.0.0.1,localhost"}}}}
JSON
  (cd "$res" && timeout "${EVAL_TIMEOUT:-600}" "$CLAUDE" -p "$(cat "$case_dir/prompt.md")" \
      --append-system-prompt "$SYSTEM_PROMPT" ${MODEL_ARGS[@]+"${MODEL_ARGS[@]}"} \
      --mcp-config "$res/mcp.json" --strict-mcp-config --tools "" \
      --allowedTools "mcp__claw-tools__ivf_similar_patients" "mcp__claw-tools__ivf_predict_outcome" \
      --output-format stream-json --verbose --no-session-persistence \
      < /dev/null > "$res/stream.jsonl" 2> "$res/stderr.log") || echo "$id: agent run failed or timed out (scored as is)"
  "$PYTHON" "$REPO/dataset/eval_case.py" --case-dir "$case_dir" --result-dir "$res"
  done_count=$((done_count + 1))
done

"$PYTHON" "$REPO/dataset/summarize_eval.py" "$OUT"
