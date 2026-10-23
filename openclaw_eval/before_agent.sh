#!/usr/bin/env bash
# Stage 1 (before_agent): start the tool services with this case's settings.
#
#   bash openclaw_eval/before_agent.sh <case_dir> <result_dir>
#
# <case_dir>/service.env holds the case's service settings (empty for test cases; for training
# cases IVF_EXCLUDE_CASE and IVF_MODEL_NAME hide the case's fold from both tools).
# Writes <result_dir>/before_agent.json (health of both services).
set -euo pipefail
source "$(dirname "$0")/common.sh"
CASE="$(cd "${1:?usage: before_agent.sh <case_dir> <result_dir>}" && pwd)"
RES="${2:?usage: before_agent.sh <case_dir> <result_dir>}"
mkdir -p "$RES" "$TRACE_DIR" "$OMNI_TASK_DIR"

bash "$REPO/task/scripts/stop_services.sh" >/dev/null 2>&1 || true
started=$(date +%s)
(set -a; [ -s "$CASE/service.env" ] && . "$CASE/service.env"; set +a
 bash "$REPO/task/scripts/start_services.sh") > "$RES/before_agent.log" 2>&1 \
  || die "services did not start; see $RES/before_agent.log and $OMNI_TASK_DIR/service-*.log"

# The health answers must come from the services just started, with this case's settings.
"$PYTHON" - "$CASE/service.env" "$RES/before_agent.json" "$started" <<'PY'
import json, os, sys, time, urllib.request
env_file, out, started = sys.argv[1], sys.argv[2], int(sys.argv[3])
settings = dict(line.split("=", 1) for line in open(env_file, encoding="utf-8").read().split() if "=" in line)
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
health = {}
for name, port in (("similar", os.environ["IVF_SIMILAR_PORT"]), ("predict", os.environ["IVF_PREDICT_PORT"])):
    with opener.open(f"http://127.0.0.1:{port}/health", timeout=5) as resp:
        health[name] = json.loads(resp.read())
    if health[name]["uptime_seconds"] > time.time() - started + 5:
        sys.exit(f"port {port} answers from a service started earlier (uptime {health[name]['uptime_seconds']} s); "
                 "another service holds the port: set IVF_SIMILAR_PORT / IVF_PREDICT_PORT and rerun prepare.sh")
model = settings.get("IVF_MODEL_NAME", "lr_35_indicators.json")
if health["predict"]["model"] != model:
    sys.exit(f"prediction service uses {health['predict']['model']}, expected {model}")
if "IVF_EXCLUDE_CASE" in settings and "排除" not in health["similar"]["scope"]:
    sys.exit(f"similar-patient service does not hold out the case: {health['similar']['scope']}")
json.dump({"service_env": settings, "health": health}, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print(f"services ready: {health['similar']['scope']} ({health['similar']['cases']} cases), model {model}")
PY

rm -f "$TRACE_DIR/dispatches.jsonl"        # this case's tool calls start from an empty trace
