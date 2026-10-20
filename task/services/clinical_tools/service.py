"""HTTP service for the IVF clinical tools (standard library only).

Endpoints (JSON in, JSON out):
  POST /ivf/similar_patients   {"age": 38, "amh": 0.9, "age_window": 1}
  POST /ivf/predict_outcome    {"age": 38, "amh": 0.9}
  GET  /health                 status, visible training cases, scope
  GET  /ivf/audit              every call since start (arguments and status)
  POST /ivf/reset              clear the audit log

Bad arguments return 400 with {"error": ...} so the agent can correct itself;
a cohort below the minimum size (拒绝统计) or an input outside the model's
training range (拒绝预测) is a normal 200 answer with that flag set to true.

  python -m clinical_tools.service --csv train_clean.csv [--port 9201] [--exclude-case ID]
  python -m clinical_tools.service --write-specs DIR [--base-url URL]
"""

from __future__ import annotations

import argparse
import json
import os
import threading
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from clinical_tools import predict, similar
from clinical_tools.cleaning import load_training
from clinical_tools.common import InputError
from clinical_tools.predict import OutcomePredictor
from clinical_tools.similar import SimilarCohort


def tool_specs(base_url: str) -> list[dict]:
    return [similar.tool_spec(base_url), predict.tool_spec(base_url)]


def make_server(cohort: SimilarCohort, predictor: OutcomePredictor, host: str = "127.0.0.1",
                port: int = 9201) -> ThreadingHTTPServer:
    routes = {"/ivf/similar_patients": cohort.query, "/ivf/predict_outcome": predictor.query}
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
                self._send(200, {"status": "ok", "tools": ["ivf_similar_patients", "ivf_predict_outcome"],
                                 "scope": cohort.scope, "searchable_cases": int(cohort.searchable.sum())})
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
                              "refused": body.get("拒绝统计", body.get("拒绝预测")), "timestamp": datetime.now(timezone.utc).isoformat()})
            self._send(status, body)

        def log_message(self, fmt: str, *args) -> None:  # keep the agent harness output quiet
            pass

    return ThreadingHTTPServer((host, port), Handler)


def write_specs(directory: Path, base_url: str) -> list[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for spec in tool_specs(base_url):
        path = directory / f"{spec['name']}.json"
        path.write_text(json.dumps(spec, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        paths.append(path)
    return paths


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--csv", default=os.environ.get("IVF_TRAIN_CSV"),
                        help="cleaned training CSV (never the sealed test set)")
    parser.add_argument("--host", default=os.environ.get("IVF_TOOLS_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("IVF_TOOLS_PORT", "9201")))
    parser.add_argument("--exclude-case", default=os.environ.get("IVF_EXCLUDE_CASE") or None,
                        help="training case under evaluation; its fold is hidden from both tools")
    parser.add_argument("--write-specs", type=Path, help="write the tool spec JSON files here and exit")
    parser.add_argument("--base-url", default="http://127.0.0.1:9201")
    args = parser.parse_args(argv)
    if args.write_specs:
        for path in write_specs(args.write_specs, args.base_url):
            print(path)
        return
    if not args.csv:
        parser.error("--csv or IVF_TRAIN_CSV is required")
    training = load_training(args.csv)
    cohort = SimilarCohort(training, exclude_case=args.exclude_case)
    predictor = OutcomePredictor(training, exclude_case=args.exclude_case)
    server = make_server(cohort, predictor, args.host, args.port)
    print(json.dumps({"listening": f"http://{args.host}:{server.server_address[1]}", "scope": cohort.scope,
                      "searchable_cases": int(cohort.searchable.sum()),
                      "prediction_cv_auc": round(predictor.with_amh.cv_auc, 3)}, ensure_ascii=False), flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
