#!/usr/bin/env bash
# Start the IVF tool service (FastAPI, one process, one port) and wait until it answers /health.
# All four tools are served on port $IVF_TOOLS_PORT (ports.sh: default 9201, one port per Dressage
# sandbox slot), each at /tools/<tool name>:
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
source "$(dirname "$0")/ports.sh"                # one port per Dressage sandbox slot
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

# An IVF service still holding our port (left by an aborted run in the same sandbox slot, whose pid
# file is gone with its /tmp) would answer with another case's settings: stop it first.
"$PYTHON" - "$IVF_TOOLS_PORT" <<'PY'
import os, signal, sys, time
port = sys.argv[1]
stale = []
for pid in filter(str.isdigit, os.listdir("/proc") if os.path.isdir("/proc") else []):
    try:
        cmd = open(f"/proc/{pid}/cmdline", "rb").read().split(b"\0")
        if not any(a.endswith(b"ivf_services/server.py") for a in cmd):
            continue
        env = dict(e.split(b"=", 1) for e in open(f"/proc/{pid}/environ", "rb").read().split(b"\0") if b"=" in e)
    except OSError:                       # gone, or another user's process: leave it
        continue
    if env.get(b"IVF_TOOLS_PORT", b"").decode() == port:
        stale.append(int(pid))
for pid in stale:
    os.kill(pid, signal.SIGTERM)
    print(f"stopped an earlier IVF service on port {port} (pid {pid})", file=sys.stderr)
deadline = time.time() + 5
while stale and time.time() < deadline:
    stale = [p for p in stale if os.path.exists(f"/proc/{p}")]
    time.sleep(0.1)
for pid in stale:
    os.kill(pid, signal.SIGKILL)
PY
export IVF_SERVICE_INSTANCE="$(date +%s%N)-$$-$RANDOM"     # /health must echo this: proves it is our process

(exec nohup "$PYTHON" "$SERVICES_DIR/ivf_services/server.py") > "$TASK_DIR/service-ivf_tools.log" 2>&1 &
SERVICE_PID=$!
echo "$SERVICE_PID" > "$TASK_DIR/service-ivf_tools.pid"
disown "$SERVICE_PID" 2>/dev/null || true
# Not ready, or the port is someone else's: stop the process just started instead of leaving it behind.
trap 'kill "$SERVICE_PID" 2>/dev/null || true; rm -f "$TASK_DIR/service-ivf_tools.pid"' EXIT

# The answer must come from the service just started: another service already on the port
# (someone else's, which we cannot stop) would serve the wrong data.
"$PYTHON" - "$IVF_SERVICE_INSTANCE" "$IVF_TOOLS_PORT" <<'PY'
import json, sys, time, urllib.error, urllib.request
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
instance, port = sys.argv[1:]
url = f"http://127.0.0.1:{port}/health"
deadline = time.time() + 90
while True:
    try:
        request = urllib.request.Request(url, headers={"X-Health-Check": "1"})
        with opener.open(request, timeout=2) as resp:
            text = resp.read().decode("utf-8")
        try:
            ours = json.loads(text).get("instance") == instance
        except ValueError:
            ours = False
        if not ours:
            raise urllib.error.HTTPError(url, 200, "not this service", None, None)
        print(text)
        break
    except urllib.error.HTTPError:        # something else answers on the port
        sys.exit(f"port {port} is held by another service; free it or set IVF_TOOLS_PORT")
    except OSError:                       # not listening yet
        if time.time() > deadline:
            sys.stderr.write(f"service did not become ready at {url}\n")
            sys.exit(1)
        time.sleep(0.5)
PY
trap - EXIT
