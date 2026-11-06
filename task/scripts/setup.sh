#!/usr/bin/env bash
# Make sure the services' Python packages are present (numpy, fastapi, pydantic, uvicorn); install them if not.
# The Dressage E2B template has fastapi/pydantic/uvicorn but no numpy and no pip; it has uv
# (/home/blackbox/.local/bin/uv), which installs into the system Python instead.
set -euo pipefail
PYTHON="${PYTHON:-python3}"
PACKAGES=("numpy>=1.26" "fastapi>=0.110" "pydantic>=2.5" "uvicorn>=0.27")
check() { "$PYTHON" -c "import numpy, fastapi, pydantic, uvicorn" 2>/dev/null; }
if ! check; then
  echo "installing ${PACKAGES[*]}" >&2
  if "$PYTHON" -m pip --version >/dev/null 2>&1; then
    "$PYTHON" -m pip install -q "${PACKAGES[@]}" \
      || "$PYTHON" -m pip install -q --break-system-packages "${PACKAGES[@]}" || true
  fi
  if ! check; then
    UV="$(command -v uv || true)"
    for candidate in /home/blackbox/.local/bin/uv "$HOME/.local/bin/uv"; do
      [ -z "$UV" ] && [ -x "$candidate" ] && UV="$candidate"
    done
    [ -n "$UV" ] || { echo "cannot install packages: no pip and no uv for $PYTHON" >&2; exit 1; }
    "$UV" pip install --system --break-system-packages --python "$(command -v "$PYTHON")" "${PACKAGES[@]}"
  fi
fi
"$PYTHON" -c "import numpy, fastapi, pydantic, uvicorn; print('python packages ok')"
