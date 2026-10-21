"""Shared plumbing for the IVF tool services (FastAPI), so each server.py only
declares its request model, its tool endpoint and how it loads its data.

``ServiceKit(app, service, tool)`` adds to an app:

* GET /health (status, uptime, request counts, load), GET /ivf/audit (tool calls
  since start or the last reset) and POST /ivf/reset, as in the template services;
* ``kit.compute(func, ...)``: runs the tool's computation in a bounded worker
  pool so the event loop never blocks. At most IVF_MAX_CONCURRENCY computations
  run at once and at most IVF_MAX_PENDING more wait; further calls get 429 with
  Retry-After instead of piling up threads;
* ``kit.record(...)``: one audit entry and one log line per tool call;
* logging: a line per request (method, path, status, milliseconds) and per tool
  call (arguments, short result summary); unexpected errors are logged with the
  traceback and answered with 500; invalid arguments are answered with 422 and
  a readable message;
* optional error injection for robustness tests (ERROR_RATE, default 0).

``run(app, port_env, default_port)`` starts uvicorn (one process, connection cap).

Environment:
  IVF_LOG_LEVEL        INFO       DEBUG also logs /health probes
  IVF_LOG_DIR          (unset)    write <service>.log there, rotated; unset = stderr
  IVF_LOG_MAX_BYTES    10485760   rotate size;  IVF_LOG_BACKUPS 5 rotated files kept
  IVF_MAX_CONCURRENCY  4          computations running at once
  IVF_MAX_PENDING      512        computations allowed to wait (cheap: no thread each); beyond that 429
  IVF_MAX_CONNECTIONS  1024       open connections uvicorn accepts; beyond that 503
  IVF_AUDIT_MAX        2000       audit entries kept in memory (oldest dropped)
  IVF_TOOLS_HOST       127.0.0.1  listen address
  ERROR_RATE           0          probability of an injected 429/500/slow reply
"""

from __future__ import annotations

import asyncio
import functools
import json
import logging
import logging.handlers
import os
import random
import time
from collections import Counter, deque
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TypeVar

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

T = TypeVar("T")
LOGGED_ARGS_MAX = 2000                          # characters of request arguments written per log line
QUIET_PATHS = ("/health",)                      # logged at DEBUG only
EXEMPT_SUFFIXES = ("/audit", "/reset", "/health", "/docs", "/openapi.json")   # never error-injected


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def log_level() -> str:
    level = os.environ.get("IVF_LOG_LEVEL", "INFO").upper()
    return level if level in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL") else "INFO"


def get_logger(service: str) -> logging.Logger:
    logger = logging.getLogger(f"ivf.{service}")
    if logger.handlers:
        return logger
    log_dir = os.environ.get("IVF_LOG_DIR")
    if log_dir:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        handler: logging.Handler = logging.handlers.RotatingFileHandler(
            Path(log_dir) / f"{service}.log", maxBytes=_env_int("IVF_LOG_MAX_BYTES", 10 * 1024 * 1024),
            backupCount=_env_int("IVF_LOG_BACKUPS", 5), encoding="utf-8")
    else:
        handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(log_level())
    logger.propagate = False
    return logger


class Busy(Exception):
    """Too many computations running or waiting."""


class ComputeGate:
    """Runs blocking work in a fixed-size thread pool and caps how much may wait."""

    def __init__(self, service: str, max_concurrency: int, max_pending: int):
        self.service, self.max_concurrency, self.max_pending = service, max_concurrency, max_pending
        self.pending = 0                       # running + waiting; only touched on the event loop
        self._executor: ThreadPoolExecutor | None = None   # created on first use, again after a restart

    async def run(self, func: Callable[..., T], /, *args: Any, **kwargs: Any) -> T:
        if self.pending >= self.max_concurrency + self.max_pending:
            raise Busy
        if self._executor is None:
            self._executor = ThreadPoolExecutor(max_workers=self.max_concurrency,
                                                thread_name_prefix=f"ivf-{self.service}")
        self.pending += 1
        try:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(self._executor, functools.partial(func, *args, **kwargs))
        finally:
            self.pending -= 1

    def shutdown(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=False, cancel_futures=True)
            self._executor = None


class ServiceKit:
    def __init__(self, app: FastAPI, service: str, tool: str):
        self.service, self.tool = service, tool
        self.log = get_logger(service)
        self.gate = ComputeGate(service, _env_int("IVF_MAX_CONCURRENCY", 4), _env_int("IVF_MAX_PENDING", 512))
        self.audit: deque[dict[str, Any]] = deque(maxlen=_env_int("IVF_AUDIT_MAX", 2000))
        self.status_counts: Counter[int] = Counter()
        self.health_info: dict[str, Any] = {}      # filled by the service once its data is loaded
        self.started = time.time()
        self._install(app)

    async def compute(self, func: Callable[..., T], /, *args: Any, **kwargs: Any) -> T:
        return await self.gate.run(func, *args, **kwargs)

    def record(self, endpoint: str, request_body: Any, response_body: Any, *, status: int = 200,
               summary: str = "") -> None:
        self.audit.append({"endpoint": endpoint, "request_body": request_body, "response_body": response_body,
                           "status": status, "timestamp": datetime.now(timezone.utc).isoformat()})
        args = json.dumps(request_body, ensure_ascii=False, default=str)
        if len(args) > LOGGED_ARGS_MAX:
            args = args[:LOGGED_ARGS_MAX] + f"...(+{len(args) - LOGGED_ARGS_MAX} chars)"
        self.log.info("call %s status=%d args=%s %s", endpoint, status, args, summary)

    def shutdown(self) -> None:
        self.gate.shutdown()
        self.log.info("stopped after %.0fs; responses by status %s", time.time() - self.started,
                      dict(self.status_counts))

    def _install(self, app: FastAPI) -> None:
        add_error_injection(app)            # added first so the access log below wraps it

        @app.middleware("http")
        async def access_log(request: Request, call_next):
            start = time.perf_counter()
            try:
                response = await call_next(request)
            except Exception:
                self.log.exception("unhandled error on %s %s", request.method, request.url.path)
                response = JSONResponse(status_code=500, content={
                    "error": "internal_server_error", "message": "服务内部错误，请稍后重试"})
            self.status_counts[response.status_code] += 1
            level = logging.DEBUG if request.url.path in QUIET_PATHS else logging.INFO
            self.log.log(level, "%s %s -> %d %.1fms", request.method, request.url.path, response.status_code,
                         (time.perf_counter() - start) * 1000)
            return response

        @app.exception_handler(RequestValidationError)
        async def invalid_arguments(request: Request, exc: RequestValidationError):
            details = [{"field": ".".join(str(p) for p in e["loc"] if p != "body") or "body", "message": e["msg"].removeprefix("Value error, ")}
                       for e in exc.errors()]
            body = {"error": "invalid_arguments",
                    "message": "参数校验失败：" + "；".join(f"{d['field']}: {d['message']}" for d in details),
                    "details": details}
            received = exc.body.decode("utf-8", "replace") if isinstance(exc.body, bytes) else exc.body
            self.record(request.url.path, received, body, status=422)
            return JSONResponse(status_code=422, content=body)

        @app.exception_handler(Busy)
        async def busy(request: Request, exc: Busy):
            self.log.warning("busy: %d computations running or waiting, refused %s", self.gate.pending,
                             request.url.path)
            return JSONResponse(status_code=429, headers={"Retry-After": "1"}, content={
                "error": "busy", "message": "服务繁忙，请稍后重试", "retry_after_seconds": 1})

        @app.get("/health")
        async def health() -> dict[str, Any]:
            return {"status": "ok", "service": self.service, "tool": self.tool,
                    "uptime_seconds": round(time.time() - self.started, 1),
                    "responses_by_status": {str(k): v for k, v in sorted(self.status_counts.items())},
                    "computations_in_flight": self.gate.pending,
                    "max_concurrency": self.gate.max_concurrency, "max_pending": self.gate.max_pending,
                    **self.health_info}

        @app.get("/ivf/audit")
        async def get_audit() -> dict[str, Any]:
            return {"calls": list(self.audit)}

        @app.post("/ivf/reset")
        async def reset_state() -> dict[str, str]:
            self.audit.clear()
            return {"status": "reset"}


def add_error_injection(app: FastAPI) -> None:
    """Random 429/500 or slow replies on tool calls when ERROR_RATE > 0 (default off)."""

    @app.middleware("http")
    async def inject_errors(request: Request, call_next):
        rate = float(os.environ.get("ERROR_RATE", "0"))
        if (rate <= 0 or request.method != "POST" or request.url.path.endswith(EXEMPT_SUFFIXES)
                or request.headers.get("X-Health-Check") == "1" or random.random() >= rate):
            return await call_next(request)
        kind = random.choices(["rate_limit", "server_error", "slow"], weights=[0.35, 0.35, 0.30])[0]
        if kind == "rate_limit":
            return JSONResponse(status_code=429, headers={"Retry-After": "2"}, content={
                "error": "rate_limit_exceeded", "message": "Too many requests. Please retry after a short delay.",
                "retry_after_seconds": 2})
        if kind == "server_error":
            return JSONResponse(status_code=500, content={
                "error": "internal_server_error", "message": "An unexpected error occurred. Please try again."})
        await asyncio.sleep(random.uniform(2.0, 4.0))      # non-blocking delay
        return await call_next(request)


def run(app: FastAPI, port_env: str, default_port: int) -> None:
    import uvicorn

    uvicorn.run(app, host=os.environ.get("IVF_TOOLS_HOST", "127.0.0.1"),
                port=int(os.environ.get(port_env, default_port)), workers=1, access_log=False,
                limit_concurrency=_env_int("IVF_MAX_CONNECTIONS", 1024),
                log_level=log_level().lower())
