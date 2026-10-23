#!/usr/bin/env bash
# Undo prepare.sh: remove the MCP server, restore the original AGENTS.md / BOOTSTRAP.md, stop the services.
#
#   bash openclaw_eval/restore.sh
set -euo pipefail
source "$(dirname "$0")/common.sh"
bash "$REPO/task/scripts/stop_services.sh" >/dev/null 2>&1 || true
"$OPENCLAW" mcp unset "$MCP_NAME" || true
"$OPENCLAW" mcp reload >/dev/null 2>&1 || true
if [ -e "$BACKUP_DIR/.saved" ]; then
  for name in AGENTS.md BOOTSTRAP.md; do
    [ -f "$BACKUP_DIR/$name" ] && cp "$BACKUP_DIR/$name" "$OC_WORKSPACE/$name" && echo "restored $name"
  done
  rm -f "$BACKUP_DIR/.saved"
else
  echo "no backup in $BACKUP_DIR; AGENTS.md left as it is"
fi
