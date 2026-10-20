#!/usr/bin/env bash
# Start the IVF clinical tools HTTP service and wait until /health answers.
#   IVF_TRAIN_CSV     training CSV (1132 cases; never the sealed test set). Not in the repo.
#                     Default inside the task container: /tmp/omni_services/data/train_clean.csv
#   IVF_TOOLS_PORT    default 9201 (must match task/tools/*.json)
#   IVF_EXCLUDE_CASE  training case under evaluation; its fold is hidden from the tool (agentic RL)
#   PYTHON            default python3
set -euo pipefail
if [ -d /tmp/omni_services/clinical_tools ]; then
  SERVICES_DIR=/tmp/omni_services            # task container layout (task.json file mapping)
else
  SERVICES_DIR="$(cd "$(dirname "$0")/../services" && pwd)"   # repository checkout
fi
TASK_DIR="${OMNI_TASK_DIR:-/tmp/omni_task}"
PORT="${IVF_TOOLS_PORT:-9201}"
PYTHON="${PYTHON:-python3}"
export IVF_TRAIN_CSV="${IVF_TRAIN_CSV:-/tmp/omni_services/data/train_clean.csv}"
[ -f "$IVF_TRAIN_CSV" ] || { echo "training CSV not found: $IVF_TRAIN_CSV (set IVF_TRAIN_CSV)" >&2; exit 1; }
mkdir -p "$TASK_DIR"

(
  export PYTHONPATH="$SERVICES_DIR${PYTHONPATH:+:$PYTHONPATH}"
  export IVF_TOOLS_PORT="$PORT"
  exec nohup "$PYTHON" -m clinical_tools.service
) > "$TASK_DIR/service-ivf_clinical.log" 2>&1 &
pid=$!
echo "$pid" > "$TASK_DIR/service-ivf_clinical.pid"
disown "$pid" 2>/dev/null || true

"$PYTHON" - "$PORT" <<'PY'
import sys, time, urllib.request
url = f"http://127.0.0.1:{sys.argv[1]}/health"
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
deadline = time.time() + 60
while time.time() < deadline:
    try:
        with opener.open(url, timeout=2) as resp:
            print(resp.read().decode("utf-8"))
            sys.exit(0)
    except OSError:
        time.sleep(0.5)
sys.stderr.write(f"service ivf_clinical did not become ready at {url}\n")
sys.exit(1)
PY
