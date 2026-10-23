#!/usr/bin/env bash
# One-time setup: inject the IVF tools into OpenClaw.
#   = register_mcp + expose_tool_instructions of the Dressage sample, on this machine's OpenClaw.
#
#   bash openclaw_eval/prepare.sh
#
# 1. writes tool specs that point at our ports ($IVF_SIMILAR_PORT / $IVF_PREDICT_PORT)
# 2. registers the MCP server "claw-tools" (openclaw mcp set)
# 3. backs up AGENTS.md (and moves BOOTSTRAP.md away), then writes the task system prompt,
#    the no-questions policy and the tool instructions into AGENTS.md
# Undo everything with openclaw_eval/restore.sh.
set -euo pipefail
source "$(dirname "$0")/common.sh"

command -v "$OPENCLAW" >/dev/null || die "openclaw not found (set OPENCLAW_BIN)"
"$PYTHON" -c "import numpy, fastapi, pydantic, uvicorn" 2>/dev/null || die "missing packages: pip install -r $REPO/requirements.txt"
[ -d "$OC_WORKSPACE" ] || die "OpenClaw workspace not found: $OC_WORKSPACE (set OC_WORKSPACE)"
mkdir -p "$TOOLS_DIR" "$TRACE_DIR" "$BACKUP_DIR" "$OMNI_TASK_DIR"

echo "== 1. tool specs -> $TOOLS_DIR"
PYTHONPATH="$REPO/task/services" "$PYTHON" - "$TOOLS_DIR" <<'PY'
import os, sys
from pathlib import Path
from ivf_services.specs import write_specs
urls = {"ivf_similar_patients": f"http://127.0.0.1:{os.environ['IVF_SIMILAR_PORT']}",
        "ivf_predict_outcome": f"http://127.0.0.1:{os.environ['IVF_PREDICT_PORT']}"}
for path in write_specs(Path(sys.argv[1]), urls):
    print("  ", path)
PY

echo "== 2. register MCP server $MCP_NAME"
config="$("$PYTHON" - "$REPO" "$TOOLS_DIR" "$TRACE_DIR" "$PYTHON" <<'PY'
import json, sys
repo, tools, trace, python = sys.argv[1:]
print(json.dumps({"command": python, "args": [f"{repo}/task/runtime/claw_tools_mcp.py"], "cwd": repo,
                  "env": {"OMNI_TOOL_SPECS_DIR": tools, "OMNI_TRACE_DIR": trace,
                          "NO_PROXY": "127.0.0.1,localhost", "no_proxy": "127.0.0.1,localhost"}}))
PY
)"
"$OPENCLAW" mcp set "$MCP_NAME" "$config"

echo "== 3. AGENTS.md in $OC_WORKSPACE"
if [ ! -e "$BACKUP_DIR/.saved" ]; then              # back up only the user's originals, never our own file
  [ -f "$OC_WORKSPACE/AGENTS.md" ] && cp "$OC_WORKSPACE/AGENTS.md" "$BACKUP_DIR/AGENTS.md"
  [ -f "$OC_WORKSPACE/BOOTSTRAP.md" ] && mv "$OC_WORKSPACE/BOOTSTRAP.md" "$BACKUP_DIR/BOOTSTRAP.md"
  touch "$BACKUP_DIR/.saved"
  echo "   originals saved in $BACKUP_DIR"
fi
rm -f "$OC_WORKSPACE/BOOTSTRAP.md"                   # its first-run dialogue would replace the task
{ cat "$REPO/task/runtime/system_prompt_prefix.txt"
  printf '\n## Blackbox Rollout Interaction Policy\n\nDo not ask the user questions during this rollout.\n'
  printf 'Do not request confirmation or wait for external input.\n'
  printf 'If required information is missing, choose reasonable defaults and continue.\n'
} > "$OC_WORKSPACE/AGENTS.md"
OMNI_TOOL_SPECS_DIR="$TOOLS_DIR" OMNI_AGENTS_MD="$OC_WORKSPACE/AGENTS.md" \
  "$PYTHON" "$REPO/task/runtime/claw_tools_mcp.py" --append-agents-md

"$OPENCLAW" mcp reload >/dev/null 2>&1 || true
echo "== check"
"$OPENCLAW" mcp probe 2>&1 | grep -E "$MCP_NAME" || die "openclaw mcp probe does not list $MCP_NAME"
echo "done. Next: bash openclaw_eval/run_all.sh <samples_dir>"
