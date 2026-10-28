"""Small helpers shared by the E2B runner."""

from __future__ import annotations

import json
import os
import re
import shlex
from pathlib import Path
from typing import Any


SECRET_KEYS = {
    "api_key",
    "apikey",
    "authorization",
    "auth_token",
    "access_token",
    "secret",
    "password",
}


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_text(path: str | Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text if text.endswith("\n") else text + "\n", encoding="utf-8")


def redact(value: Any) -> Any:
    """Return a copy of *value* with API keys/tokens replaced.

    The blackbox server's register response echoes ``backend_options``, which
    contains the agent LLM key.  Never write that to disk or print it.
    """
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, item in value.items():
            lowered = str(key).lower().replace("-", "_")
            if lowered in SECRET_KEYS or lowered.endswith("_api_key"):
                out[key] = "<redacted>"
            else:
                out[key] = redact(item)
        return out
    if isinstance(value, list):
        return [redact(item) for item in value]
    return value


def case_key(instance_id: str, *, index: int) -> str:
    key = re.sub(r"[^A-Za-z0-9._-]+", "-", str(instance_id or "")).strip("-._")
    return (key or f"case-{index:04d}")[:96]


def truncate(text: str, limit: int = 200_000) -> str:
    if text is None:
        return ""
    text = str(text)
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n... [truncated {len(text) - limit} bytes]\n"


def command_ok(result: dict[str, Any]) -> bool:
    """A normalized command result counts as success only when it ran and exited 0."""
    if result.get("error"):
        return False
    if result.get("timed_out"):
        return False
    return result.get("returncode") == 0


def shell_quote(value: str) -> str:
    return shlex.quote(value)


def load_env_file(path: str | Path, *, override: bool = False) -> None:
    """Load ``KEY=VALUE`` / ``export KEY=VALUE`` lines without python-dotenv."""
    for raw_line in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        if key and (override or key not in os.environ):
            os.environ[key] = value
