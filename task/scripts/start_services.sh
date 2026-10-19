#!/usr/bin/env bash
set -euo pipefail
mkdir -p /tmp/omni_task

# start service: config
(
  export PYTHONPATH="/tmp/omni_services/python_deps:/tmp/omni_services"
  export CONFIG_FIXTURES="/tmp/omni_services/service_fixtures/fixtures/config/integrations.json"
  export MOCK_TODAY="2026-03-25"
  export PORT=9111
  exec nohup python3 /tmp/omni_services/mock_services/config/server.py
) > /tmp/omni_task/service-config.log 2>&1 &
pid=$!
echo "$pid" > /tmp/omni_task/service-config.pid
disown "$pid" 2>/dev/null || true

# start service: helpdesk
(
  export PYTHONPATH="/tmp/omni_services/python_deps:/tmp/omni_services"
  export HELPDESK_FIXTURES="/tmp/omni_services/service_fixtures/fixtures/helpdesk/tickets.json"
  export MOCK_TODAY="2026-03-25"
  export PORT=9107
  exec nohup python3 /tmp/omni_services/mock_services/helpdesk/server.py
) > /tmp/omni_task/service-helpdesk.log 2>&1 &
pid=$!
echo "$pid" > /tmp/omni_task/service-helpdesk.pid
disown "$pid" 2>/dev/null || true

# start service: contacts
(
  export PYTHONPATH="/tmp/omni_services/python_deps:/tmp/omni_services"
  export CONTACTS_FIXTURES="/tmp/omni_services/service_fixtures/fixtures/contacts/contacts.json"
  export MOCK_TODAY="2026-03-25"
  export PORT=9103
  exec nohup python3 /tmp/omni_services/mock_services/contacts/server.py
) > /tmp/omni_task/service-contacts.log 2>&1 &
pid=$!
echo "$pid" > /tmp/omni_task/service-contacts.pid
disown "$pid" 2>/dev/null || true

python3 - <<'PY'
import time, urllib.request, json, sys
checks = [
    ("config", "http://localhost:9111/config/integrations", "POST", 10),
    ("helpdesk", "http://localhost:9107/helpdesk/tickets", "POST", 10),
    ("contacts", "http://localhost:9103/contacts/search", "POST", 10),
]
from urllib.error import HTTPError, URLError
for name, url, method, ready_timeout in checks:
    deadline = time.time() + ready_timeout
    body = b'{}' if method == 'POST' else None
    headers = {'Content-Type': 'application/json'} if body is not None else {}
    last_err = None
    ok = False
    while time.time() < deadline:
        try:
            req = urllib.request.Request(url, data=body, headers=headers, method=method)
            with urllib.request.urlopen(req, timeout=2) as resp:
                if resp.status < 500:
                    ok = True
                    break
        except HTTPError as exc:
            # 4xx means the HTTP server is up but rejected the probe body — treat as ready.
            if exc.code < 500:
                ok = True
                break
            last_err = exc
            time.sleep(0.2)
        except (URLError, ConnectionError, OSError) as exc:
            last_err = exc
            time.sleep(0.2)
    if not ok:
        sys.stderr.write(f'service {name} did not become ready: {last_err}\n')
        sys.exit(1)
PY
