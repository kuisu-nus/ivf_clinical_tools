#!/usr/bin/env bash
# Convenience wrapper: run python3 -m e2b_eval from the repo root and load
# e2b_eval/.env when it exists (does not override variables already exported).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
cd "$REPO"

ARGS=("$@")
if [ -f "$HERE/.env" ]; then
  ARGS=(--env-file "$HERE/.env" "${ARGS[@]}")
fi
exec python3 -m e2b_eval "${ARGS[@]}"
