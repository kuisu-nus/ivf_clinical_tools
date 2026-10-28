"""Environment configuration for the E2B runner."""

from __future__ import annotations

import os
from dataclasses import dataclass


DEFAULT_TEMPLATE = "e2b-dressage-claw-blackbox"
DEFAULT_AGENT_BASE_URL = "https://api.deepseek.com/v1"
DEFAULT_AGENT_MODEL = "deepseek-flash"
DEFAULT_SANDBOX_TIMEOUT_SEC = 3600


class ConfigError(RuntimeError):
    pass


def _env_str(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return default
    return value.strip()


def _env_int(name: str, default: int) -> int:
    raw = _env_str(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be an integer, got {raw!r}") from exc


def _env_float(name: str, default: float) -> float:
    raw = _env_str(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be a number, got {raw!r}") from exc


@dataclass(frozen=True)
class E2BConfig:
    """E2B endpoint settings.

    ``sandbox_url`` is the client proxy (e.g. ``http://host:3002``) of a
    self-hosted E2B install.  The Python SDK uses it to route ``commands.run``
    to the sandbox; without it a self-hosted sandbox appears to be created and
    then immediately "not found".
    """

    api_key: str
    api_url: str | None
    sandbox_url: str | None
    template: str = DEFAULT_TEMPLATE
    timeout_sec: int = DEFAULT_SANDBOX_TIMEOUT_SEC
    blackbox_port: int = 31000

    @classmethod
    def from_env(
        cls,
        *,
        template: str | None = None,
        timeout_sec: int | None = None,
        blackbox_port: int | None = None,
        require_key: bool = True,
    ) -> "E2BConfig":
        api_key = _env_str("E2B_API_KEY") or _env_str("DRESSAGE_E2B_API_KEY") or ""
        api_url = _env_str("E2B_API_URL")
        sandbox_url = _env_str("E2B_SANDBOX_URL")
        if require_key and not api_key:
            raise ConfigError(
                "E2B_API_KEY is not set (DRESSAGE_E2B_API_KEY also accepted).\n"
                "Example:\n"
                "  export E2B_API_KEY=e2b_...\n"
                "  export E2B_API_URL=http://<e2b-host>:3000\n"
                "  export E2B_SANDBOX_URL=http://<e2b-host>:3002"
            )
        if api_url and not sandbox_url:
            raise ConfigError(
                "E2B_API_URL is set but E2B_SANDBOX_URL is not.  For a self-hosted E2B "
                "install the SDK also needs the client proxy URL, otherwise commands.run "
                "fails with 'The sandbox was not found'.\n"
                "  export E2B_SANDBOX_URL=http://<e2b-host>:3002"
            )
        resolved_template = template or _env_str("E2B_TEMPLATE") or DEFAULT_TEMPLATE
        resolved_port = (
            blackbox_port
            if blackbox_port is not None
            else _env_int("DRESSAGE_E2B_BLACKBOX_PORT", 31000)
        )
        if resolved_template == DEFAULT_TEMPLATE and resolved_port != 31000:
            raise ConfigError(
                f"{DEFAULT_TEMPLATE} starts BlackboxServer on the fixed port 31000; "
                f"DRESSAGE_E2B_BLACKBOX_PORT={resolved_port} cannot be used with this template"
            )
        resolved_timeout = (
            timeout_sec
            if timeout_sec is not None
            else _env_int("E2B_SANDBOX_TIMEOUT_SEC", DEFAULT_SANDBOX_TIMEOUT_SEC)
        )
        if resolved_timeout <= 0:
            raise ConfigError("sandbox timeout must be positive")
        return cls(
            api_key=api_key or "dry-run",
            api_url=api_url,
            sandbox_url=sandbox_url,
            template=resolved_template,
            timeout_sec=resolved_timeout,
            blackbox_port=resolved_port,
        )

    def export_env(self) -> None:
        """The E2B SDK reads its connection settings from the environment."""
        os.environ["E2B_API_KEY"] = self.api_key
        if self.api_url:
            os.environ["E2B_API_URL"] = self.api_url
        if self.sandbox_url:
            os.environ["E2B_SANDBOX_URL"] = self.sandbox_url


@dataclass(frozen=True)
class AgentLLMConfig:
    """The OpenAI-compatible model endpoint OpenClaw uses inside the sandbox.

    The blackbox server starts OpenClaw with a local rollout proxy and forwards
    its ``/chat/completions`` calls to ``base_url``.  The model key is written
    into the sandbox's OpenClaw config, so treat the sandbox and this key as
    trusted.
    """

    mode: str = "openclaw"
    base_url: str = DEFAULT_AGENT_BASE_URL
    model: str = DEFAULT_AGENT_MODEL
    api_key: str | None = None
    provider_id: str = "deepseek"
    context_window: int = 65_536
    max_tokens: int = 4_096
    max_steps: int | None = 40
    register_timeout_sec: float = 300.0
    turn_timeout_sec: float = 900.0

    @classmethod
    def from_env(
        cls,
        *,
        mode: str = "openclaw",
        base_url: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        provider_id: str | None = None,
        context_window: int | None = None,
        max_tokens: int | None = None,
        max_steps: int | None = None,
        register_timeout_sec: float | None = None,
        turn_timeout_sec: float | None = None,
        require_key: bool = True,
    ) -> "AgentLLMConfig":
        mode = (mode or "openclaw").strip().lower()
        if mode not in {"openclaw", "mock"}:
            raise ConfigError(f"agent mode must be 'openclaw' or 'mock', got {mode!r}")

        resolved_key = (
            api_key
            or _env_str("AGENT_LLM_API_KEY")
            or _env_str("OPENAI_API_KEY")
        )
        if mode == "mock":
            resolved_key = resolved_key or "mock"
        elif require_key and not resolved_key:
            raise ConfigError(
                "AGENT_LLM_API_KEY is not set.  The sandbox's OpenClaw needs an "
                "OpenAI-compatible endpoint and key for the agent phase.\n"
                "Example:\n"
                "  export AGENT_LLM_BASE_URL=https://api.deepseek.com/v1\n"
                "  export AGENT_LLM_API_KEY=sk-...\n"
                "  export AGENT_LLM_MODEL=deepseek-flash\n"
                "For a pipeline check without an LLM use --agent-mode mock."
            )

        env_max_steps = _env_str("AGENT_MAX_STEPS")
        if max_steps is None and env_max_steps is not None:
            try:
                max_steps = int(env_max_steps)
            except ValueError as exc:
                raise ConfigError(
                    f"AGENT_MAX_STEPS must be an integer, got {env_max_steps!r}"
                ) from exc
        if max_steps is not None and max_steps <= 0:
            max_steps = None

        resolved_context = (
            context_window
            if context_window is not None
            else _env_int("AGENT_LLM_CONTEXT_WINDOW", 65_536)
        )
        resolved_max_tokens = (
            max_tokens
            if max_tokens is not None
            else _env_int("AGENT_LLM_MAX_TOKENS", 4_096)
        )
        if resolved_context <= 0:
            raise ConfigError("AGENT_LLM_CONTEXT_WINDOW must be positive")
        if resolved_max_tokens <= 0:
            raise ConfigError("AGENT_LLM_MAX_TOKENS must be positive")

        return cls(
            mode=mode,
            base_url=base_url
            or _env_str("AGENT_LLM_BASE_URL")
            or DEFAULT_AGENT_BASE_URL,
            model=model
            or _env_str("AGENT_LLM_MODEL")
            or DEFAULT_AGENT_MODEL,
            api_key=resolved_key,
            provider_id=provider_id
            or _env_str("AGENT_LLM_PROVIDER_ID")
            or "deepseek",
            context_window=resolved_context,
            max_tokens=resolved_max_tokens,
            max_steps=max_steps,
            register_timeout_sec=register_timeout_sec
            if register_timeout_sec is not None
            else _env_float("AGENT_REGISTER_TIMEOUT_SEC", 300.0),
            turn_timeout_sec=turn_timeout_sec
            if turn_timeout_sec is not None
            else _env_float("AGENT_TURN_TIMEOUT_SEC", 900.0),
        )
