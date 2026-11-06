# Shared settings for the openclaw_eval scripts (sourced, not run).
# Override any of these in the environment before calling a script.
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK="${OC_EVAL_WORK:-$REPO/outputs/openclaw}"          # tool specs, trace, service pids and logs, backups
PYTHON="${PYTHON:-python3}"
OPENCLAW="${OPENCLAW_BIN:-openclaw}"
OC_WORKSPACE="${OC_WORKSPACE:-$HOME/.openclaw/workspace}"   # the OpenClaw workspace whose AGENTS.md is used
MCP_NAME=claw-tools

# Our own port, so a tool service someone else started on 9201 is never used by mistake.
export IVF_TOOLS_PORT="${IVF_TOOLS_PORT:-9301}"
export OMNI_TASK_DIR="$WORK/run"                          # service pid files and logs (start/stop_services.sh)
TOOLS_DIR="$WORK/tools"                                   # tool specs pointing at the ports above
TRACE_DIR="$WORK/trace"                                   # dispatches.jsonl written by the MCP bridge
BACKUP_DIR="$WORK/backup"                                 # original AGENTS.md / BOOTSTRAP.md

die() { echo "error: $*" >&2; exit 1; }
