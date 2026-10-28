"""HTTP client for the Dressage BlackboxServer running inside the sandbox.

The self-hosted E2B client proxy selects a sandbox with the ``Host`` header
(``<port>-<sandbox_id>.e2b.app``), so an :class:`Endpoint` carries both the
proxy URL and that header.  No Dressage package is imported here.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
import uuid
from typing import Any

import httpx

from .config import AgentLLMConfig
from .sandbox import Endpoint

log = logging.getLogger(__name__)

DEFAULT_RUNTIME_ROOT = "/workspace_sandbox/blackbox_server_runtime"
TERMINAL_TURN_ERRORS = frozenset({"failed", "unknown", "cancelled"})


class BlackboxError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        status: int | None = None,
        payload: Any = None,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.payload = payload


class BlackboxClient:
    def __init__(
        self,
        endpoint: Endpoint,
        *,
        request_timeout: float = 120.0,
        runtime_root: str | None = None,
    ) -> None:
        self.endpoint = endpoint
        self.runtime_root = (
            runtime_root
            or os.environ.get("E2B_BBS_RUNTIME_ROOT")
            or DEFAULT_RUNTIME_ROOT
        )
        self._client = httpx.AsyncClient(
            base_url=endpoint.url.rstrip("/"),
            headers=endpoint.headers,
            timeout=httpx.Timeout(request_timeout, connect=30.0),
            trust_env=False,
        )

    async def close(self) -> None:
        await self._client.aclose()

    async def health(self) -> dict[str, Any]:
        response = await self._request("GET", "/health", timeout=10.0)
        data = response.json()
        if not data.get("healthy"):
            raise BlackboxError(f"unexpected health response: {data}")
        return data

    async def wait_healthy(self, *, timeout_sec: float = 90.0) -> dict[str, Any]:
        deadline = time.monotonic() + timeout_sec
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            try:
                return await self.health()
            except Exception as exc:  # noqa: BLE001 - retry until ready
                last_error = exc
                await asyncio.sleep(1.0)
        raise BlackboxError(f"blackbox server did not become healthy: {last_error}")

    async def register(
        self,
        *,
        session_id: str,
        instance_id: str,
        system_prompt_file: str | None,
        agent: AgentLLMConfig,
        sandbox_id: str | None = None,
    ) -> dict[str, Any]:
        backend_options: dict[str, Any] = {
            "agent_id": "default",
            "provider_id": agent.provider_id,
            "model_id": agent.model,
            "model_name": agent.model,
            "api_key": agent.api_key or "mock",
            "context_window": int(agent.context_window),
            "max_tokens": int(agent.max_tokens),
        }
        if agent.max_steps is not None:
            backend_options["proxy"] = {"max_steps": int(agent.max_steps)}

        server_config = {
            "runtime_root": self.runtime_root,
            "backend_timeout": max(300, int(agent.turn_timeout_sec) + 120),
            "execute_cmd_timeout": 600,
        }
        payload: dict[str, Any] = {
            "blackbox_type": "openclaw",
            "router": agent.base_url,
            "router_api_path": "/v1",
            "bound_instance_id": instance_id,
            "bound_session_id": session_id,
            "backend_options": backend_options,
            "server_config": server_config,
        }
        if system_prompt_file:
            payload["system_prompt_file"] = system_prompt_file
        if sandbox_id:
            payload["bound_sandbox_id"] = sandbox_id

        log.info(
            "registering OpenClaw via BBS (model=%s provider=%s)",
            agent.model,
            agent.provider_id,
        )
        # OpenClaw starts inside this call; a cold template can take ~30s.
        response = await self._request(
            "POST",
            "/v1/rollout/register",
            json=payload,
            timeout=agent.register_timeout_sec,
            retries=2,
        )
        return response.json()

    async def send_message(
        self,
        *,
        session_id: str,
        prompt: str,
        agent: AgentLLMConfig,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        turn_id = f"turn-{uuid.uuid4().hex}"
        payload = {
            "turn_id": turn_id,
            "mode": "async",
            "messages": [{"role": "user", "content": prompt}],
            "metadata": metadata or {},
        }
        log.info("submitting agent turn %s", turn_id)
        response = await self._request(
            "POST",
            f"/v1/sessions/{session_id}/messages",
            json=payload,
            timeout=120.0,
            retries=2,
        )
        if response.status_code != 202:
            # A sync-mode response; the body already contains outputs.
            return response.json()
        return await self._poll_turn(
            session_id=session_id,
            turn_id=turn_id,
            timeout_sec=agent.turn_timeout_sec,
        )

    async def fetch_session(
        self,
        session_id: str,
        *,
        include_history: bool = True,
        include_trace: bool = True,
    ) -> dict[str, Any]:
        params = {
            "include_history": str(include_history).lower(),
            "include_trace": str(include_trace).lower(),
        }
        response = await self._request(
            "GET", f"/v1/sessions/{session_id}", params=params, timeout=60.0, retries=2
        )
        return response.json()

    async def execute_cmd(
        self,
        session_id: str,
        cmd: str,
        *,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        if "\n" in cmd or "\r" in cmd or "\x00" in cmd:
            raise BlackboxError(
                "blackbox execute_cmd only accepts a single-line command; "
                "join setup steps with ';' or '&&'"
            )
        execute_timeout = 600.0 if timeout is None else float(timeout)
        payload: dict[str, Any] = {"cmd": cmd}
        if timeout is not None:
            payload["timeout"] = float(timeout)
        response = await self._request(
            "POST",
            f"/v1/sessions/{session_id}/execute_cmd",
            json=payload,
            timeout=execute_timeout + 30.0,
        )
        return response.json()

    async def abort(self, session_id: str) -> None:
        try:
            await self._request(
                "POST", f"/v1/sessions/{session_id}/abort", timeout=30.0
            )
        except Exception as exc:  # noqa: BLE001 - cleanup is best effort
            log.debug("abort session %s failed: %s", session_id, exc)

    async def _poll_turn(
        self,
        *,
        session_id: str,
        turn_id: str,
        timeout_sec: float,
    ) -> dict[str, Any]:
        deadline = time.monotonic() + timeout_sec
        last_error: Exception | None = None
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise BlackboxError(
                    f"agent turn {turn_id} did not finish within {timeout_sec:.0f}s"
                    + (f" (last error: {last_error})" if last_error else "")
                )
            wait = min(30.0, max(0.0, remaining))
            try:
                response = await self._request(
                    "GET",
                    f"/v1/sessions/{session_id}/turns/{turn_id}",
                    params={"wait": wait},
                    timeout=wait + 30.0,
                )
                data = response.json()
            except BlackboxError as exc:
                # The server may briefly restart its monitor; keep polling
                # until the overall turn deadline.
                last_error = exc
                log.warning("polling turn %s failed: %s", turn_id, exc)
                await asyncio.sleep(min(2.0, max(0.0, deadline - time.monotonic())))
                continue

            status = str(data.get("status") or "")
            if status == "committed":
                return data
            if status in TERMINAL_TURN_ERRORS:
                error = data.get("error") if isinstance(data.get("error"), dict) else {}
                raise BlackboxError(
                    f"agent turn {turn_id} {status}: "
                    f"{error.get('message') or error or 'no error details'}",
                    status=error.get("http_status"),
                    payload=data,
                )
            # queued / inflight: keep waiting.

    async def _request(
        self,
        method: str,
        path: str,
        *,
        timeout: float | None = None,
        retries: int = 0,
        **kwargs: Any,
    ) -> httpx.Response:
        last_error: Exception | None = None
        for attempt in range(retries + 1):
            try:
                response = await self._client.request(
                    method,
                    path,
                    timeout=httpx.Timeout(timeout, connect=30.0) if timeout else None,
                    **kwargs,
                )
            except (httpx.ConnectError, httpx.ReadTimeout, httpx.RemoteProtocolError) as exc:
                last_error = exc
                if attempt < retries:
                    await asyncio.sleep(2 ** attempt)
                    continue
                raise BlackboxError(f"{method} {path} failed: {exc}") from exc

            if response.status_code >= 400:
                try:
                    payload: Any = response.json()
                except ValueError:
                    payload = response.text
                message = payload.get("message") if isinstance(payload, dict) else payload
                raise BlackboxError(
                    f"{method} {path} -> HTTP {response.status_code}: {message}",
                    status=response.status_code,
                    payload=payload,
                )
            return response
        raise BlackboxError(f"{method} {path} failed: {last_error}")
