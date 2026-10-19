"""Runtime-injected judge for V4 graders.

This module is copied into the sandbox grader mount next to
``omni_grader_runner.py``. It intentionally keeps provider selection out of
task graders:

- text judge calls use the existing reward/judge model endpoint;
- vision/audio judge calls use OpenRouter when configured;
- capability absence means task graders should skip that dimension;
- configured judge API failures are retried, then raised as infrastructure
  errors so the rollout can be dropped instead of trained as score 0.
"""

from __future__ import annotations

import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable


DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_OPENROUTER_VISION_MODEL = "openai/gpt-5.4"
DEFAULT_OPENROUTER_AUDIO_MODEL = "google/gemini-3.5-flash"
DEFAULT_TEXT_MODEL = "judge"
DEFAULT_TEXT_API_KEY = "no-auth"
DEAD_MODEL_ROUTES = {"openai/gpt-4o-mini-audio-preview": "audio"}
RETRY_STATUS_CODES = {408, 409, 429, 500, 502, 503, 504}


class JudgeRuntimeError(RuntimeError):
    """Infrastructure failure from a configured judge provider."""

    omni_judge_runtime_error = True


@dataclass(frozen=True)
class JudgeProviderConfig:
    base_url: str
    api_key: str
    model_id: str


@dataclass(frozen=True)
class JudgeRetryConfig:
    max_retries: int = 3
    initial_delay: float = 2.0
    max_delay: float = 30.0
    timeout: float = 180.0
    max_calls: int = 500


def _truthy(value: Any) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _first_env(*names: str) -> str:
    for name in names:
        value = os.environ.get(name)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _openrouter_api_key() -> str:
    key = _first_env("OMNI_OPENROUTER_API_KEY", "OPENROUTER_API_KEY")
    if key:
        return key
    keyfile = Path(
        _first_env("OMNI_OPENROUTER_KEY_FILE", "OPENROUTER_KEY_FILE")
        or "~/.config/openrouter/key"
    ).expanduser()
    if keyfile.is_file():
        return keyfile.read_text(encoding="utf-8").strip()
    return ""


def _text_judge_config_from_env() -> JudgeProviderConfig | None:
    base_url = _first_env(
        "OMNI_TEXT_JUDGE_BASE_URL",
        "OMNI_JUDGE_TEXT_BASE_URL",
        "OMNI_CLAW_EVAL_JUDGE_BASE_URL",
        "JUDGE_BASE_URL",
        "OPENAI_BASE_URL",
    )
    # Older launch scripts used OPENROUTER_BASE_URL for the text reward model.
    # Use it only when no OpenRouter key is configured, so real OpenRouter
    # routing stays reserved for vision/audio.
    if not base_url and not _openrouter_api_key():
        base_url = _first_env("OPENROUTER_BASE_URL")
    if not base_url:
        return None
    api_key = (
        _first_env(
            "OMNI_TEXT_JUDGE_API_KEY",
            "OMNI_JUDGE_TEXT_API_KEY",
            "OMNI_CLAW_EVAL_JUDGE_API_KEY",
            "JUDGE_API_KEY",
            "OPENAI_API_KEY",
        )
        or DEFAULT_TEXT_API_KEY
    )
    model_id = (
        _first_env(
            "OMNI_TEXT_JUDGE_MODEL",
            "OMNI_JUDGE_TEXT_MODEL",
            "OMNI_CLAW_EVAL_JUDGE_MODEL",
            "JUDGE_MODEL",
            "REWARD_MODEL_ID",
        )
        or DEFAULT_TEXT_MODEL
    )
    return JudgeProviderConfig(base_url=base_url, api_key=api_key, model_id=model_id)


def _openrouter_config_from_env() -> JudgeProviderConfig | None:
    api_key = _openrouter_api_key()
    if not api_key:
        return None
    text_base_url = _first_env(
        "OMNI_TEXT_JUDGE_BASE_URL",
        "OMNI_JUDGE_TEXT_BASE_URL",
        "OMNI_CLAW_EVAL_JUDGE_BASE_URL",
        "JUDGE_BASE_URL",
        "OPENAI_BASE_URL",
    )
    base_url = _first_env("OMNI_OPENROUTER_BASE_URL")
    legacy_openrouter_base = _first_env("OPENROUTER_BASE_URL")
    if not base_url and legacy_openrouter_base and legacy_openrouter_base != text_base_url:
        base_url = legacy_openrouter_base
    if not base_url:
        base_url = DEFAULT_OPENROUTER_BASE_URL
    model_id = (
        _first_env("OMNI_JUDGE_VISION_MODEL", "OMNI_OPENROUTER_VISION_MODEL")
        or DEFAULT_OPENROUTER_VISION_MODEL
    )
    return JudgeProviderConfig(base_url=base_url, api_key=api_key, model_id=model_id)


def _retry_config_from_env() -> JudgeRetryConfig:
    return JudgeRetryConfig(
        max_retries=_env_int("OMNI_JUDGE_MAX_RETRIES", 3),
        initial_delay=_env_float("OMNI_JUDGE_INITIAL_DELAY", 2.0),
        max_delay=_env_float("OMNI_JUDGE_MAX_DELAY", 30.0),
        timeout=_env_float("OMNI_JUDGE_TIMEOUT", 180.0),
        max_calls=_env_int("OMNI_JUDGE_MAX_CALLS", 500),
    )


def _status_code(exc: BaseException) -> int | None:
    value = getattr(exc, "status_code", None)
    return value if isinstance(value, int) else None


def _is_retryable_exception(exc: BaseException) -> bool:
    status = _status_code(exc)
    if status is not None:
        return status in RETRY_STATUS_CODES
    name = type(exc).__name__
    return name in {"APIConnectionError", "APITimeoutError", "TimeoutError"}


def _log_judge_record(record: dict[str, Any], *, usage_log: Path | None) -> None:
    log = usage_log
    if log is None:
        raw = os.environ.get("OMNI_JUDGE_USAGE_LOG", "")
        if not raw:
            return
        log = Path(raw)
    if not log:
        return
    try:
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        pass


class RetryingOpenAICompatibleClient:
    """Small OpenAI-compatible chat client wrapper with explicit retry policy."""

    def __init__(
        self,
        *,
        client: Any,
        model_id: str,
        retry_config: JudgeRetryConfig,
        usage_log: str | os.PathLike | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._client = client
        self.model_id = model_id
        self.model = model_id
        self.retry_config = retry_config
        self.calls_made = 0
        self.usage_log = Path(usage_log) if usage_log else None
        self._sleep = sleep
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    @property
    def client(self) -> "RetryingOpenAICompatibleClient":
        return self

    def _create(self, **kwargs: Any) -> Any:
        if self.calls_made >= self.retry_config.max_calls:
            raise JudgeRuntimeError(
                f"judge call budget exhausted ({self.retry_config.max_calls})"
            )
        kwargs.setdefault("model", self.model_id)
        delay = self.retry_config.initial_delay
        last_exc: BaseException | None = None
        for attempt in range(self.retry_config.max_retries + 1):
            try:
                self.calls_made += 1
                resp = self._client.chat.completions.create(**kwargs)
                self._log_usage(kwargs.get("model"), resp)
                return resp
            except Exception as exc:  # noqa: BLE001 - provider SDKs vary here
                last_exc = exc
                if not _is_retryable_exception(exc):
                    break
                if attempt < self.retry_config.max_retries:
                    self._sleep(delay + random.uniform(0, 1))
                    delay = min(delay * 2.5, self.retry_config.max_delay)
        self._log_error(kwargs.get("model"), last_exc)
        raise JudgeRuntimeError(
            f"judge API failed after retries: {type(last_exc).__name__}: {last_exc}"
        ) from last_exc

    def _log_usage(self, model: Any, resp: Any) -> None:
        usage = getattr(resp, "usage", None)
        _log_judge_record(
            {
                "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "event": "usage",
                "model": model,
                "prompt_tokens": getattr(usage, "prompt_tokens", None),
                "completion_tokens": getattr(usage, "completion_tokens", None),
            },
            usage_log=self.usage_log,
        )

    def _log_error(self, model: Any, exc: BaseException | None) -> None:
        _log_judge_record(
            {
                "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "event": "error",
                "model": model,
                "error": f"{type(exc).__name__}: {exc}"[:400],
            },
            usage_log=self.usage_log,
        )


class CompositeJudge:
    """OpenAI-compatible judge that routes text to local judge, media to OpenRouter."""

    def __init__(
        self,
        *,
        text_client: Any | None,
        openrouter_client: Any | None,
        text_model: str = DEFAULT_TEXT_MODEL,
        vision_model: str = DEFAULT_OPENROUTER_VISION_MODEL,
        audio_model: str = DEFAULT_OPENROUTER_AUDIO_MODEL,
    ) -> None:
        self.text_client = text_client
        self.openrouter_client = openrouter_client
        self.text_model = text_model
        self.vision_model = vision_model
        self.audio_model = audio_model
        self.model_id = text_model if text_client is not None else vision_model
        self.model = self.model_id
        self.capability_models = {}
        if text_client is not None:
            self.capability_models["text"] = text_model
        if openrouter_client is not None:
            self.capability_models["vision"] = vision_model
            self.capability_models["audio"] = audio_model
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    @property
    def client(self) -> "CompositeJudge":
        return self

    @property
    def capabilities(self) -> set[str]:
        return set(self.capability_models)

    @staticmethod
    def _payload_has(messages: Any, part_types: set[str]) -> bool:
        try:
            for message in messages or []:
                content = message.get("content") if isinstance(message, dict) else None
                if isinstance(content, list):
                    for part in content:
                        if isinstance(part, dict) and part.get("type") in part_types:
                            return True
        except Exception:
            return False
        return False

    def _route_capability(self, requested_model: str | None, messages: Any) -> str:
        if self._payload_has(messages, {"input_audio", "audio", "audio_url"}):
            return "audio"
        if self._payload_has(messages, {"image_url", "input_image"}):
            return "vision"
        if requested_model in DEAD_MODEL_ROUTES:
            return DEAD_MODEL_ROUTES[requested_model]
        return "text"

    def _create(self, **kwargs: Any) -> Any:
        requested_model = kwargs.get("model")
        capability = self._route_capability(requested_model, kwargs.get("messages"))
        if capability == "text":
            if self.text_client is None:
                raise JudgeRuntimeError("text judge requested but no text client configured")
            kwargs["model"] = (
                requested_model
                if requested_model and requested_model != "injected-judge"
                else self.text_model
            )
            return self.text_client.chat.completions.create(**kwargs)
        if self.openrouter_client is None:
            raise JudgeRuntimeError(
                f"{capability} judge requested but no OpenRouter client configured"
            )
        kwargs["model"] = self.audio_model if capability == "audio" else self.vision_model
        return self.openrouter_client.chat.completions.create(**kwargs)


def _openai_client_from_config(config: JudgeProviderConfig, retry: JudgeRetryConfig) -> Any:
    import openai  # local import keeps offline tests importable

    return openai.OpenAI(
        base_url=config.base_url,
        api_key=config.api_key,
        timeout=retry.timeout,
        max_retries=0,
    )


def build_judge_from_env() -> CompositeJudge | None:
    enabled = _truthy(os.environ.get("OMNI_JUDGE_ENABLE")) or _truthy(
        os.environ.get("OMNI_CLAW_EVAL_ENABLE_JUDGE")
    )
    if not enabled:
        return None

    retry = _retry_config_from_env()
    usage_log = os.environ.get("OMNI_JUDGE_USAGE_LOG") or None
    text_config = _text_judge_config_from_env()
    openrouter_config = _openrouter_config_from_env()

    text_client = None
    if text_config is not None:
        text_client = RetryingOpenAICompatibleClient(
            client=_openai_client_from_config(text_config, retry),
            model_id=text_config.model_id,
            retry_config=retry,
            usage_log=usage_log,
        )

    openrouter_client = None
    if openrouter_config is not None:
        openrouter_client = RetryingOpenAICompatibleClient(
            client=_openai_client_from_config(openrouter_config, retry),
            model_id=openrouter_config.model_id,
            retry_config=retry,
            usage_log=usage_log,
        )

    if text_client is None and openrouter_client is None:
        return None

    return CompositeJudge(
        text_client=text_client,
        openrouter_client=openrouter_client,
        text_model=text_config.model_id if text_config is not None else DEFAULT_TEXT_MODEL,
        vision_model=(
            _first_env("OMNI_JUDGE_VISION_MODEL", "OMNI_OPENROUTER_VISION_MODEL")
            or DEFAULT_OPENROUTER_VISION_MODEL
        ),
        audio_model=(
            _first_env("OMNI_JUDGE_AUDIO_MODEL", "OMNI_OPENROUTER_AUDIO_MODEL")
            or DEFAULT_OPENROUTER_AUDIO_MODEL
        ),
    )


def adapt_for_claw_eval(judge: Any) -> Any:
    """Wrap the runtime judge with claw_eval's vendored LLMJudge when present."""
    if judge is None:
        return None
    try:
        from claw_eval.graders.llm_judge import LLMJudge  # type: ignore
    except ImportError:
        return judge
    adapted = LLMJudge.__new__(LLMJudge)
    adapted.client = judge
    adapted.model_id = getattr(judge, "model_id", None) or getattr(
        judge, "model", "injected-judge"
    )
    adapted.model = adapted.model_id
    adapted.capabilities = getattr(judge, "capabilities", {"text"})
    adapted.capability_models = getattr(judge, "capability_models", {})
    return adapted
