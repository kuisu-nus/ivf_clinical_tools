#!/usr/bin/env bash
# Register the task's MCP server "claw-tools" with the OpenClaw that BlackboxServer started in this
# sandbox (Dressage before_agent command "register_mcp").
#
# OpenClaw's HOME is the one BlackboxServer prepared: $BBS_RUNTIME_ROOT/bbs-*/home. When the tool
# service runs on another port than the specs' 9201 (see ports.sh), OMNI_TOOL_PORT_MAP tells the
# MCP bridge where to send the calls.
#
# OpenClaw settings for this task: the minimal tool profile and no skills keep the context short;
# tools.alsoAllow lets the MCP tools (bundle-mcp) and file/runtime tools through that profile,
# which otherwise drops the claw-tools. The final line prints tools.alsoAllow into the rollout log.
set -euo pipefail
SCRIPTS="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPTS/ports.sh"
TASK_DIR="${OMNI_TASK_DIR:-/tmp/omni_task}"
OC_HOME="$(ls -d "${BBS_RUNTIME_ROOT:-/workspace_sandbox/blackbox_server_runtime}"/bbs-*/home 2>/dev/null | head -1)"
[ -n "$OC_HOME" ] || { echo "no OpenClaw home under ${BBS_RUNTIME_ROOT:-/workspace_sandbox/blackbox_server_runtime}" >&2; exit 1; }

config="$("${PYTHON:-python3}" - "$TASK_DIR" "$IVF_TOOLS_PORT" <<'PY'
import json, sys
task_dir, port = sys.argv[1:]
local = "127.0.0.1,localhost,0.0.0.0,::1"
env = {"OMNI_TOOL_SPECS_DIR": f"{task_dir}/tools", "OMNI_TRACE_DIR": f"{task_dir}/trace",
       "NO_PROXY": local, "no_proxy": local}
if port != "9201":
    env["OMNI_TOOL_PORT_MAP"] = f"9201:{port}"
print(json.dumps({"command": "python3", "args": [f"{task_dir}/runtime/claw_tools_mcp.py"], "env": env,
                  "cwd": task_dir}))
PY
)"
HOME="$OC_HOME" "${OPENCLAW_BIN:-openclaw}" config set tools.profile minimal
HOME="$OC_HOME" "${OPENCLAW_BIN:-openclaw}" config set agents.list '[{"id":"main","skills":[]}]'
HOME="$OC_HOME" "${OPENCLAW_BIN:-openclaw}" config set tools.alsoAllow '["group:fs", "group:runtime","bundle-mcp"]'

HOME="$OC_HOME" "${OPENCLAW_BIN:-openclaw}" mcp set claw-tools "$config"
echo "registered claw-tools for $OC_HOME (tools :$IVF_TOOLS_PORT)"
HOME="$OC_HOME" "${OPENCLAW_BIN:-openclaw}" config get tools.alsoAllow
