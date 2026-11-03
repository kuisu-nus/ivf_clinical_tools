#!/usr/bin/env bash
# Undo oc_deploy.sh: put back the backed-up openclaw.json and workspace files and stop the IVF tools.
#
#   bash ckpt_eval/oc_restore.sh <backup_dir>
set -euo pipefail
source "$(dirname "$0")/common.sh"
BACKUP="${1:?usage: oc_restore.sh <backup_dir>}"
OC="$OC_SOURCE_HOME/.openclaw"
[ -f "$BACKUP/openclaw.json" ] || die "$BACKUP has no openclaw.json"
cp "$BACKUP/openclaw.json" "$OC/openclaw.json"
for f in "$BACKUP"/*.md; do [ -f "$f" ] && cp "$f" "$OC/workspace/"; done
(export HOME="$OC_SOURCE_HOME"; source "$REPO/openclaw_eval/common.sh" && bash "$REPO/task/scripts/stop_services.sh") >/dev/null 2>&1 || true
echo "restored $OC from $BACKUP; the model server keeps running (pkill -f \"served-model-name <name>\")"
