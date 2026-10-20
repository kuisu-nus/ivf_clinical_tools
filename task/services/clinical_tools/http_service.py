"""Minimal JSON-over-HTTP server shared by the two tool services (standard library only).

Each service exposes POST routes plus GET /health, GET /ivf/audit (every call
since start) and POST /ivf/reset. Bad arguments return 400 with {"error": ...}
so the agent can correct itself.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Callable
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from clinical_tools.common import InputError


def make_server(routes: dict[str, Callable[[dict], dict]], health: dict, host: str = "127.0.0.1",
                port: int = 0) -> ThreadingHTTPServer:
    audit: list[dict] = []
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def _send(self, status: int, body: dict) -> None:
            data = json.dumps(body, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:
            if self.path == "/health":
                self._send(200, {"status": "ok", **health})
            elif self.path == "/ivf/audit":
                with lock:
                    self._send(200, {"calls": list(audit)})
            else:
                self._send(404, {"error": f"not found: {self.path}"})

        def do_POST(self) -> None:
            if self.path == "/ivf/reset":
                with lock:
                    audit.clear()
                self._send(200, {"status": "reset"})
                return
            handler = routes.get(self.path)
            if handler is None:
                self._send(404, {"error": f"not found: {self.path}"})
                return
            arguments = None
            try:
                length = int(self.headers.get("Content-Length") or 0)
                arguments = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
                if not isinstance(arguments, dict):
                    raise InputError("请求体必须是 JSON 对象")
                status, body = 200, handler(arguments)
            except (InputError, json.JSONDecodeError, UnicodeDecodeError) as exc:
                status, body = 400, {"error": str(exc)}
            with lock:
                audit.append({"path": self.path, "arguments": arguments, "status": status,
                              "timestamp": datetime.now(timezone.utc).isoformat()})
            self._send(status, body)

        def log_message(self, fmt: str, *args) -> None:  # keep the agent harness output quiet
            pass

    return ThreadingHTTPServer((host, port), Handler)


def serve(server: ThreadingHTTPServer, banner: dict) -> None:
    host, port = server.server_address[:2]
    print(json.dumps({"listening": f"http://{host}:{port}", **banner}, ensure_ascii=False), flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
