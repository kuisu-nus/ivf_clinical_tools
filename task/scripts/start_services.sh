#!/usr/bin/env bash
# Start the IVF tool service (FastAPI, one process, one port) and wait until it answers /health.
# All four tools are served on port ${IVF_TOOLS_PORT:-9201}, each at /tools/<tool name>:
#   ivf_similar_patients  reads IVF_TRAIN_CSV (default: services/data/train_age_amh_gn.csv, in the repo)
#   ivf_predict_outcome   reads IVF_MODEL (default: services/models/$IVF_MODEL_NAME,
#                         IVF_MODEL_NAME default lr_35_indicators.json; fold models for RL)
#   ivf_query_cases, ivf_case_sql   read IVF_CASE_TABLE (default: services/data/train_cases_35.csv, no IDs)
#   IVF_EXCLUDE_CASE  training case under evaluation; its fold is hidden from the similar-patient and case tools
#   IVF_LOG_DIR       rotating service log (default: $OMNI_TASK_DIR/logs); see services/ivf_services/_base.py
#   PYTHON            default python3 (needs numpy, fastapi, uvicorn: requirements.txt)
set -euo pipefail
if [ -d /tmp/omni_services/ivf_services ]; then
  SERVICES_DIR=/tmp/omni_services            # task container layout (task.json file mapping)
else
  SERVICES_DIR="$(cd "$(dirname "$0")/../services" && pwd)"   # repository checkout
fi
TASK_DIR="${OMNI_TASK_DIR:-/tmp/omni_task}"
PYTHON="${PYTHON:-python3}"
export IVF_TOOLS_PORT="${IVF_TOOLS_PORT:-9201}"
export IVF_TRAIN_CSV="${IVF_TRAIN_CSV:-$SERVICES_DIR/data/train_age_amh_gn.csv}"
export IVF_MODEL="${IVF_MODEL:-$SERVICES_DIR/models/${IVF_MODEL_NAME:-lr_35_indicators.json}}"
export IVF_CASE_TABLE="${IVF_CASE_TABLE:-$SERVICES_DIR/data/train_cases_35.csv}"
export IVF_LOG_DIR="${IVF_LOG_DIR:-$TASK_DIR/logs}"
export PYTHONPATH="$SERVICES_DIR${PYTHONPATH:+:$PYTHONPATH}"
[ -f "$IVF_TRAIN_CSV" ] || { echo "training CSV not found: $IVF_TRAIN_CSV (set IVF_TRAIN_CSV)" >&2; exit 1; }
[ -f "$IVF_MODEL" ] || { echo "model artifact not found: $IVF_MODEL (set IVF_MODEL)" >&2; exit 1; }
[ -f "$IVF_CASE_TABLE" ] || { echo "case table not found: $IVF_CASE_TABLE (set IVF_CASE_TABLE)" >&2; exit 1; }
"$PYTHON" -c "import numpy, fastapi, pydantic, uvicorn" 2>/dev/null \
  || { echo "missing packages: pip install -r requirements.txt (numpy, fastapi, uvicorn)" >&2; exit 1; }
mkdir -p "$TASK_DIR" "$IVF_LOG_DIR"

(exec nohup "$PYTHON" "$SERVICES_DIR/ivf_services/server.py") > "$TASK_DIR/service-ivf_tools.log" 2>&1 &
echo "$!" > "$TASK_DIR/service-ivf_tools.pid"
disown "$!" 2>/dev/null || true

"$PYTHON" - "$IVF_TOOLS_PORT" <<'PY'
import sys, time, urllib.request
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
url = f"http://127.0.0.1:{sys.argv[1]}/health"
deadline = time.time() + 90
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
