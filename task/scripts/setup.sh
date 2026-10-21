#!/usr/bin/env bash
# Make sure the services' Python packages are present (numpy, fastapi, pydantic, uvicorn); install them if not.
set -euo pipefail
PYTHON="${PYTHON:-python3}"
PACKAGES=("numpy>=1.26" "fastapi>=0.110" "pydantic>=2.5" "uvicorn>=0.27")
if ! "$PYTHON" -c "import numpy, fastapi, pydantic, uvicorn" 2>/dev/null; then
  echo "installing ${PACKAGES[*]}" >&2
  "$PYTHON" -m pip install -q "${PACKAGES[@]}" \
    || "$PYTHON" -m pip install -q --break-system-packages "${PACKAGES[@]}"
fi
"$PYTHON" -c "import numpy, fastapi, pydantic, uvicorn; print('python packages ok')"
