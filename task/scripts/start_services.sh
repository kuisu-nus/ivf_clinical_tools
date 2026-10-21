#!/usr/bin/env bash
# Start the two IVF tool services (FastAPI, one process each) and wait until both answer /health.
#   similar patients  port ${IVF_SIMILAR_PORT:-9201}  needs IVF_TRAIN_CSV (cleaned training CSV, 1132 cases; not in the repo)
#   pregnancy predict port ${IVF_PREDICT_PORT:-9202}  needs IVF_MODEL (default: services/models/lr_35_indicators.json)
#   IVF_EXCLUDE_CASE  training case under evaluation; its fold is hidden from the similar-patient tool
#   IVF_LOG_DIR       rotating service logs (default: $OMNI_TASK_DIR/logs); see services/ivf_services/_base.py
#   PYTHON            default python3 (needs numpy, fastapi, uvicorn: requirements.txt)
set -euo pipefail
if [ -d /tmp/omni_services/ivf_services ]; then
  SERVICES_DIR=/tmp/omni_services            # task container layout (task.json file mapping)
else
  SERVICES_DIR="$(cd "$(dirname "$0")/../services" && pwd)"   # repository checkout
fi
TASK_DIR="${OMNI_TASK_DIR:-/tmp/omni_task}"
PYTHON="${PYTHON:-python3}"
export IVF_SIMILAR_PORT="${IVF_SIMILAR_PORT:-9201}" IVF_PREDICT_PORT="${IVF_PREDICT_PORT:-9202}"
export IVF_TRAIN_CSV="${IVF_TRAIN_CSV:-/tmp/omni_services/data/train_clean.csv}"
export IVF_MODEL="${IVF_MODEL:-$SERVICES_DIR/models/lr_35_indicators.json}"
export IVF_LOG_DIR="${IVF_LOG_DIR:-$TASK_DIR/logs}"
export PYTHONPATH="$SERVICES_DIR${PYTHONPATH:+:$PYTHONPATH}"
[ -f "$IVF_TRAIN_CSV" ] || { echo "training CSV not found: $IVF_TRAIN_CSV (set IVF_TRAIN_CSV)" >&2; exit 1; }
[ -f "$IVF_MODEL" ] || { echo "model artifact not found: $IVF_MODEL (set IVF_MODEL)" >&2; exit 1; }
"$PYTHON" -c "import numpy, fastapi, pydantic, uvicorn" 2>/dev/null \
  || { echo "missing packages: pip install -r requirements.txt (numpy, fastapi, uvicorn)" >&2; exit 1; }
mkdir -p "$TASK_DIR" "$IVF_LOG_DIR"

start() {  # name server-directory
  (exec nohup "$PYTHON" "$SERVICES_DIR/ivf_services/$2/server.py") > "$TASK_DIR/service-$1.log" 2>&1 &
  echo "$!" > "$TASK_DIR/service-$1.pid"
  disown "$!" 2>/dev/null || true
}
start ivf_similar similar
start ivf_predict predict

"$PYTHON" - "$IVF_SIMILAR_PORT" "$IVF_PREDICT_PORT" <<'PY'
import sys, time, urllib.request
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
for port in sys.argv[1:]:
    url = f"http://127.0.0.1:{port}/health"
    deadline = time.time() + 60
    while True:
        try:
            request = urllib.request.Request(url, headers={"X-Health-Check": "1"})
            with opener.open(request, timeout=2) as resp:
                print(resp.read().decode("utf-8"))
                break
        except OSError:
            if time.time() > deadline:
                sys.stderr.write(f"service did not become ready at {url}\n")
                sys.exit(1)
            time.sleep(0.5)
PY
