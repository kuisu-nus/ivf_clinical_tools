"""Thin E2B sandbox wrapper.

Only this module imports the ``e2b`` SDK, so sample validation and unit tests
work without E2B installed.  The wrapper owns the sandbox lifecycle, bulk file
upload, file reads and the public endpoint used by the blackbox HTTP client.
"""

from __future__ import annotations

import inspect
import logging
from dataclasses import dataclass
from typing import Any

from .config import E2BConfig
from .samples import Upload
from .util import truncate

log = logging.getLogger(__name__)


class SandboxError(RuntimeError):
    pass


@dataclass(frozen=True)
class Endpoint:
    url: str
    headers: dict[str, str]

    def describe(self) -> dict[str, str]:
        return {"url": self.url, "host": self.headers.get("Host", "")}


@dataclass(frozen=True)
class CommandResult:
    cmd: str
    stdout: str
    stderr: str
    exit_code: int | None
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "cmd": self.cmd,
            "stdout": truncate(self.stdout),
            "stderr": truncate(self.stderr),
            "returncode": self.exit_code,
            "error": self.error,
            "timed_out": False,
        }


class Sandbox:
    def __init__(
        self,
        raw: Any,
        *,
        sandbox_proxy_url: str | None,
        blackbox_port: int,
    ) -> None:
        self.raw = raw
        self.sandbox_id = _sandbox_id(raw)
        self._sandbox_proxy_url = sandbox_proxy_url.rstrip("/") if sandbox_proxy_url else None
        self.blackbox_port = blackbox_port

    @classmethod
    async def create(
        cls,
        config: E2BConfig,
        *,
        template: str,
        timeout_sec: int,
        metadata: dict[str, str] | None = None,
    ) -> "Sandbox":
        try:
            from e2b import AsyncSandbox
        except ImportError as exc:  # pragma: no cover - depends on env
            raise SandboxError(
                "the e2b Python SDK is not installed; pip install -r e2b_eval/requirements.txt"
            ) from exc

        config.export_env()
        log.info("creating E2B sandbox template=%s timeout=%ss", template, timeout_sec)
        raw = await AsyncSandbox.create(
            template=template,
            timeout=int(timeout_sec),
            metadata=metadata or {},
        )
        sandbox = cls(
            raw,
            sandbox_proxy_url=config.sandbox_url,
            blackbox_port=config.blackbox_port,
        )
        log.info("sandbox ready id=%s", sandbox.sandbox_id)
        return sandbox

    async def endpoint(self, port: int | None = None) -> Endpoint:
        port = self.blackbox_port if port is None else int(port)
        host = await _maybe_await(self.raw.get_host(port))
        host = str(host)
        if self._sandbox_proxy_url:
            # Self-hosted E2B serves every sandbox port through one proxy and
            # selects the sandbox/port from the Host header.
            return Endpoint(url=self._sandbox_proxy_url, headers={"Host": host})
        return Endpoint(url=f"https://{host}", headers={})

    async def write_files(self, uploads: list[Upload]) -> list[dict[str, Any]]:
        if not uploads:
            return []
        files_api = getattr(self.raw, "files", None)
        if files_api is None:
            raise SandboxError("E2B sandbox does not expose files")

        entries = []
        for upload in uploads:
            try:
                data = upload.local_path.read_bytes()
            except OSError as exc:
                raise SandboxError(f"cannot read {upload.local_path}: {exc}") from exc
            entries.append({"path": upload.sandbox_path, "data": data})

        write_files = getattr(files_api, "write_files", None)
        if callable(write_files):
            await _maybe_await(write_files(entries))
        else:  # older SDKs
            write = getattr(files_api, "write", None)
            if not callable(write):
                raise SandboxError("E2B sandbox does not expose files.write_files or files.write")
            for entry in entries:
                await _maybe_await(write(entry["path"], entry["data"]))
        log.info("uploaded %d file(s)", len(entries))
        return [upload.describe() for upload in uploads]

    async def read_text(self, path: str, *, max_bytes: int | None = None) -> str | None:
        files_api = getattr(self.raw, "files", None)
        read = getattr(files_api, "read", None) if files_api is not None else None
        if not callable(read):
            raise SandboxError("E2B sandbox does not expose files.read")
        try:
            data = await _maybe_await(read(path, format="text"))
        except Exception as exc:  # noqa: BLE001 - missing artifacts are normal
            if _looks_missing(exc):
                log.debug("artifact missing in sandbox: %s", path)
                return None
            raise
        if isinstance(data, bytes):
            text = data.decode("utf-8", errors="replace")
        else:
            text = str(data)
        if max_bytes is not None and len(text.encode("utf-8")) > max_bytes:
            return text.encode("utf-8")[:max_bytes].decode("utf-8", errors="replace")
        return text

    async def read_bytes(self, path: str, *, max_bytes: int | None = None) -> bytes | None:
        files_api = getattr(self.raw, "files", None)
        read = getattr(files_api, "read", None) if files_api is not None else None
        if not callable(read):
            raise SandboxError("E2B sandbox does not expose files.read")
        try:
            data = await _maybe_await(read(path, format="bytes"))
        except Exception as exc:  # noqa: BLE001
            if _looks_missing(exc):
                return None
            raise
        raw = bytes(data)
        return raw if max_bytes is None else raw[:max_bytes]

    async def run(
        self,
        cmd: str,
        *,
        timeout: float = 60.0,
        envs: dict[str, str] | None = None,
        cwd: str | None = None,
    ) -> CommandResult:
        commands = getattr(self.raw, "commands", None)
        run = getattr(commands, "run", None) if commands is not None else None
        if not callable(run):
            raise SandboxError("E2B sandbox does not expose commands.run")
        try:
            result = await _maybe_await(
                run(cmd, timeout=timeout, envs=envs, cwd=cwd)
            )
        except Exception as exc:  # noqa: BLE001
            return CommandResult(cmd=cmd, stdout="", stderr="", exit_code=None, error=str(exc))
        return CommandResult(
            cmd=cmd,
            stdout=str(getattr(result, "stdout", "") or ""),
            stderr=str(getattr(result, "stderr", "") or ""),
            exit_code=getattr(result, "exit_code", None),
            error=getattr(result, "error", None),
        )

    async def kill(self) -> None:
        kill = getattr(self.raw, "kill", None)
        if not callable(kill):
            return
        try:
            await _maybe_await(kill())
            log.info("sandbox killed id=%s", self.sandbox_id)
        except Exception as exc:  # noqa: BLE001 - cleanup is best effort
            log.warning("failed to kill sandbox id=%s: %s", self.sandbox_id, exc)


async def _maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


def _sandbox_id(raw: Any) -> str:
    for attr in ("sandbox_id", "sandboxId", "id"):
        value = getattr(raw, attr, None)
        if value:
            return str(value)
    return ""


def _looks_missing(exc: Exception) -> bool:
    name = type(exc).__name__.lower()
    text = str(exc).lower()
    return (
        "notfound" in name
        or "not_found" in name
        or "404" in text
        or "not found" in text
        or "no such file" in text
        or "does not exist" in text
    )
