#!/usr/bin/env bash
# Start the two IVF tool services as independent processes and wait until both answer /health.
#   similar patients  port ${IVF_SIMILAR_PORT:-9201}  needs IVF_TRAIN_CSV (training CSV, 1132 cases; not in the repo)
#   pregnancy predict port ${IVF_PREDICT_PORT:-9202}  needs IVF_MODEL (default: services/models/lr_35_indicators.json)
#   IVF_EXCLUDE_CASE  training case under evaluation; its fold is hidden from the similar-patient tool
#   PYTHON            default python3
set -euo pipefail
if [ -d /tmp/omni_services/clinical_tools ]; then
  SERVICES_DIR=/tmp/omni_services            # task container layout (task.json file mapping)
else
  SERVICES_DIR="$(cd "$(dirname "$0")/../services" && pwd)"   # repository checkout
fi
TASK_DIR="${OMNI_TASK_DIR:-/tmp/omni_task}"
PYTHON="${PYTHON:-python3}"
export IVF_SIMILAR_PORT="${IVF_SIMILAR_PORT:-9201}" IVF_PREDICT_PORT="${IVF_PREDICT_PORT:-9202}"
export IVF_TRAIN_CSV="${IVF_TRAIN_CSV:-/tmp/omni_services/data/train_clean.csv}"
export IVF_MODEL="${IVF_MODEL:-$SERVICES_DIR/models/lr_35_indicators.json}"
[ -f "$IVF_TRAIN_CSV" ] || { echo "training CSV not found: $IVF_TRAIN_CSV (set IVF_TRAIN_CSV)" >&2; exit 1; }
[ -f "$IVF_MODEL" ] || { echo "model artifact not found: $IVF_MODEL (set IVF_MODEL)" >&2; exit 1; }
mkdir -p "$TASK_DIR"

start() {  # name module
  (
    export PYTHONPATH="$SERVICES_DIR${PYTHONPATH:+:$PYTHONPATH}"
    exec nohup "$PYTHON" -m "clinical_tools.$2"
  ) > "$TASK_DIR/service-$1.log" 2>&1 &
  echo "$!" > "$TASK_DIR/service-$1.pid"
  disown "$!" 2>/dev/null || true
}
start ivf_similar similar_service
start ivf_predict predict_service

"$PYTHON" - "$IVF_SIMILAR_PORT" "$IVF_PREDICT_PORT" <<'PY'
import sys, time, urllib.request
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
for port in sys.argv[1:]:
    url = f"http://127.0.0.1:{port}/health"
    deadline = time.time() + 60
    while True:
        try:
            with opener.open(url, timeout=2) as resp:
                print(resp.read().decode("utf-8"))
                break
        except OSError:
            if time.time() > deadline:
                sys.stderr.write(f"service did not become ready at {url}\n")
                sys.exit(1)
            time.sleep(0.5)
PY
