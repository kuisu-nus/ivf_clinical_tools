"""Helpers for Omni/CogitoRL grader evidence files.

This module is intentionally dependency-free.  The sandbox runner copies it to
``/tmp/omni_grader/omni_grader_helpers.py`` so task graders can import a stable
helper API without packaging the whole CogitoRL repo inside the sandbox.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


VALID_SCORE_STATUSES = {"passed", "failed", "skipped", "error"}
VALID_TOOL_STATUSES = {"ok", "pending", "orphan_tool_result", "error"}
DEFAULT_GRADER_PATH = Path("/tmp/omni_grader")
DEFAULT_WORKSPACE_PATH = Path("/workspace")
IMAGE_MIME_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
}
AUDIO_FORMATS = {
    ".wav": "wav",
    ".mp3": "mp3",
    ".m4a": "m4a",
    ".aac": "aac",
    ".flac": "flac",
    ".ogg": "ogg",
    ".opus": "opus",
}


def _ctx_value(ctx: Any, *names: str, default: Any = None) -> Any:
    if ctx is None:
        return default
    for name in names:
        if isinstance(ctx, dict) and name in ctx:
            return ctx[name]
        if hasattr(ctx, name):
            return getattr(ctx, name)
    return default


def _grader_path(ctx: Any) -> Path:
    return Path(
        _ctx_value(
            ctx,
            "grader_path",
            "grader_dir",
            "grader_mount_path",
            default=DEFAULT_GRADER_PATH,
        )
    )


def read_json(path: str | Path, *, default: Any = None, required: bool = True) -> Any:
    path = Path(path)
    if not path.exists():
        if required:
            raise FileNotFoundError(str(path))
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        if required:
            raise
        return default


def read_tool_trajectory(ctx: Any) -> dict[str, Any]:
    path = _ctx_value(ctx, "tool_trajectory_path", default=None)
    data = read_json(
        Path(path) if path else _grader_path(ctx) / "tool_trajectory.json",
        default={
            "schema_version": "omni_tool_trajectory.v1",
            "source": {"runner": "openclaw", "adapter": "missing"},
            "calls": [],
        },
        required=False,
    )
    validate_tool_trajectory(data)
    return data


def read_llm_messages(ctx: Any) -> list[dict[str, Any]]:
    path = _ctx_value(ctx, "llm_messages_path", default=None)
    data = read_json(
        Path(path) if path else _grader_path(ctx) / "llm_messages.json",
        default=[],
        required=False,
    )
    validate_llm_messages(data)
    return data


def read_audit_data(ctx: Any) -> dict[str, Any]:
    path = _ctx_value(ctx, "audit_data_path", default=None)
    data = read_json(
        Path(path) if path else _grader_path(ctx) / "audit_data.json",
        default=empty_audit_data(),
        required=False,
    )
    validate_audit_data(data)
    return data


def final_assistant_text(messages: list[dict[str, Any]]) -> str:
    for message in reversed(messages):
        if not isinstance(message, dict):
            continue
        if message.get("role") != "assistant" or message.get("tool_calls"):
            continue
        content = message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict) and isinstance(item.get("text"), str):
                    parts.append(item["text"])
            return "\n".join(parts)
        return "" if content is None else str(content)
    return ""


def iter_tool_calls(
    trajectory: dict[str, Any],
    *,
    tool_name: str | None = None,
    tool_server: str | None = None,
    status: str | None = None,
) -> list[dict[str, Any]]:
    validate_tool_trajectory(trajectory)
    calls = [call for call in trajectory.get("calls", []) if isinstance(call, dict)]
    if tool_name is not None:
        calls = [call for call in calls if call.get("tool_name") == tool_name]
    if tool_server is not None:
        calls = [call for call in calls if call.get("tool_server") == tool_server]
    if status is not None:
        calls = [call for call in calls if call.get("status") == status]

    if tool_name is not None and tool_server is None:
        servers = {
            str(call.get("tool_server"))
            for call in calls
            if call.get("tool_server") not in (None, "")
        }
        if len(servers) > 1:
            raise ValueError(
                f"tool_name {tool_name!r} matches multiple tool_server values: "
                f"{sorted(servers)}"
            )
    return calls


def count_tool_calls(
    trajectory: dict[str, Any],
    *,
    tool_name: str | None = None,
    tool_server: str | None = None,
    status: str | None = None,
) -> int:
    return len(
        iter_tool_calls(
            trajectory,
            tool_name=tool_name,
            tool_server=tool_server,
            status=status,
        )
    )


def tool_called(
    trajectory: dict[str, Any],
    *,
    tool_name: str | None = None,
    tool_server: str | None = None,
    status: str | None = None,
) -> bool:
    return bool(
        iter_tool_calls(
            trajectory,
            tool_name=tool_name,
            tool_server=tool_server,
            status=status,
        )
    )


def get_service(audit_data: dict[str, Any], service_id: str) -> dict[str, Any]:
    validate_audit_data(audit_data)
    services = audit_data.get("services") or {}
    if service_id not in services:
        raise KeyError(f"missing audit service: {service_id}")
    service = services[service_id]
    if not isinstance(service, dict):
        raise ValueError(f"audit service {service_id!r} is not an object")
    return service


def iter_audit_actions(
    audit_data: dict[str, Any],
    *,
    service_id: str | None = None,
    action_type: str | None = None,
    status: str | None = None,
) -> list[dict[str, Any]]:
    validate_audit_data(audit_data)
    services = audit_data.get("services") or {}
    rows: list[dict[str, Any]] = []
    for key, service in services.items():
        if service_id is not None and key != service_id:
            continue
        if not isinstance(service, dict):
            continue
        for action in service.get("actions") or []:
            if not isinstance(action, dict):
                continue
            if action_type is not None and action.get("action_type") != action_type:
                continue
            if status is not None and action.get("status") != status:
                continue
            row = dict(action)
            row.setdefault("service_id", key)
            rows.append(row)
    return rows


def workspace_path(ctx: Any, relpath: str) -> Path:
    rel = Path(relpath)
    if rel.is_absolute():
        raise ValueError("workspace relative path must not be absolute")
    base = Path(
        _ctx_value(
            ctx,
            "workspace_path",
            "workspace_dir",
            "workspace_mount_path",
            default=DEFAULT_WORKSPACE_PATH,
        )
    )
    return base / rel


def workspace_file_exists(ctx: Any, relpath: str) -> bool:
    return workspace_path(ctx, relpath).is_file()


def score_item(
    name: str,
    score: float | None,
    *,
    weight: float = 1.0,
    status: str | None = None,
    reason: str = "",
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if status is None:
        if score is None:
            status = "skipped"
        else:
            status = "passed" if score >= 1.0 else "failed"
    item: dict[str, Any] = {
        "name": name,
        "score": score,
        "weight": weight,
        "status": status,
        "reason": reason,
    }
    if metadata is not None:
        item["metadata"] = metadata
    return item


def skipped_item(
    name: str,
    *,
    weight: float = 1.0,
    reason: str = "",
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return score_item(
        name,
        None,
        weight=weight,
        status="skipped",
        reason=reason,
        metadata=metadata,
    )


def overall_item(
    score: float,
    *,
    reason: str = "",
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return score_item(
        "overall",
        score,
        weight=1.0,
        status="passed" if score >= 1.0 else "failed",
        reason=reason,
        metadata=metadata,
    )


def zero_task_gate(
    name: str,
    passed: bool,
    *,
    reason: str,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    item = score_item(
        name,
        1.0 if passed else 0.0,
        weight=0.0,
        status="passed" if passed else "failed",
        reason=reason,
        metadata=metadata,
    )
    item["gate"] = "zero_task"
    return item


# ---------------------------------------------------------------------------
# LLM judge: shared rubric-scoring machinery.
#
# A per-grader ``llm_judge(ctx, judge)`` should only DECLARE what to judge
# (rubric items, which workspace files to read, the LLM half's weight) and
# delegate the mechanics to ``judge_rubric``. This is the single place that
# pins the judge-client interface, prompt shape, JSON parsing and retry/skip
# policy, so changing the judge interface touches ONE function instead of every
# task grader.
#
# Judge-client contract (PROVISIONAL -- to be confirmed in the LLM phase):
# an OpenAI-compatible client reachable as ``judge`` or ``judge.client`` that
# exposes ``.chat.completions.create(model=..., messages=[...], temperature=...,
# max_tokens=...)`` and returns ``resp.choices[0].message.content``. The model
# id is read from ``judge.model_id`` / ``judge.model`` when present.
#
# When ``judge`` is None (e.g. the workspace-only pilot has no judge runtime)
# every rubric item is returned as ``skipped`` so the task reports
# ``partial_evaluation`` instead of pretending the LLM half passed.
# ---------------------------------------------------------------------------

JUDGE_DEFAULT_RETRIES = 3
# Token budget for ONE judge reply. Default is generous because reasoning
# ("thinking") judge models spend most of the budget on reasoning_content
# before emitting the JSON verdict in `content`; with a small cap they finish
# on `length` with empty content, which would otherwise be (mis)read as a 0.
# Override per-run with OMNI_JUDGE_MAX_TOKENS. Non-reasoning judges stop early
# and are unaffected by the higher cap.
JUDGE_DEFAULT_MAX_TOKENS = 2000


def _judge_max_tokens() -> int:
    import os

    raw = os.environ.get("OMNI_JUDGE_MAX_TOKENS")
    if raw:
        try:
            value = int(raw)
            if value > 0:
                return value
        except ValueError:
            pass
    return JUDGE_DEFAULT_MAX_TOKENS


def _is_judge_runtime_error(exc: BaseException) -> bool:
    return bool(getattr(exc, "omni_judge_runtime_error", False)) or (
        exc.__class__.__name__ == "JudgeRuntimeError"
    )


def _normalize_rubric(rubric: Any) -> list[tuple[str, str]]:
    norm: list[tuple[str, str]] = []
    for i, item in enumerate(rubric):
        if isinstance(item, (tuple, list)) and len(item) == 2:
            name, criterion = item
        else:
            name, criterion = f"item_{i}", item
        norm.append((str(name), str(criterion)))
    return norm


def _read_files_payload(ctx: Any, read_files: list[str] | None) -> tuple[str, list[dict[str, Any]]]:
    import base64

    parts: list[str] = []
    media_parts: list[dict[str, Any]] = []
    for rel in read_files or []:
        path = workspace_path(ctx, rel)
        suffix = path.suffix.lower()
        if suffix in IMAGE_MIME_TYPES:
            try:
                data = path.read_bytes()
                encoded = base64.b64encode(data).decode("ascii")
                media_parts.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{IMAGE_MIME_TYPES[suffix]};base64,{encoded}",
                        },
                    }
                )
                parts.append(
                    f"===== {rel} =====\n"
                    f"<attached image: {IMAGE_MIME_TYPES[suffix]}, {len(data)} bytes>"
                )
            except OSError:
                parts.append(f"===== {rel} =====\n<missing>")
            continue
        if suffix in AUDIO_FORMATS:
            try:
                data = path.read_bytes()
                encoded = base64.b64encode(data).decode("ascii")
                media_parts.append(
                    {
                        "type": "input_audio",
                        "input_audio": {
                            "data": encoded,
                            "format": AUDIO_FORMATS[suffix],
                        },
                    }
                )
                parts.append(
                    f"===== {rel} =====\n"
                    f"<attached audio: {AUDIO_FORMATS[suffix]}, {len(data)} bytes>"
                )
            except OSError:
                parts.append(f"===== {rel} =====\n<missing>")
            continue
        try:
            body = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            body = "<missing>"
        parts.append(f"===== {rel} =====\n{body}")
    return "\n\n".join(parts), media_parts


def _read_files_context(ctx: Any, read_files: list[str] | None) -> str:
    context, _media_parts = _read_files_payload(ctx, read_files)
    return context


def _judge_user_content(prompt: str, media_parts: list[dict[str, Any]]) -> str | list[dict[str, Any]]:
    if not media_parts:
        return prompt
    return [{"type": "text", "text": prompt}, *media_parts]


def _resolve_grader_file(ctx: Any, relpath: str) -> Path:
    """Resolve a grader-private file (e.g. rubric.md): try grader_assets_path,
    then grader_mount_path, then the path as given."""
    p = Path(relpath)
    if p.is_absolute():
        return p
    candidates: list[Path] = []
    assets = _ctx_value(ctx, "grader_assets_path", default=None)
    mount = _ctx_value(ctx, "grader_mount_path", "grader_dir", "grader_path", default=None)
    if assets:
        candidates.append(Path(assets) / relpath)
    if mount:
        candidates.append(Path(mount) / relpath)
    candidates.append(p)
    for c in candidates:
        if c.exists():
            return c
    return candidates[0]


def _parse_rubric_md(text: str) -> list[tuple[str, str]]:
    """Parse a rubric.md into [(name, criterion)] items.

    Supported (in priority order):
      1. ``## name`` h2 sections -> named criteria (preferred).
      2. markdown list items (``- ``/``* ``/``1. ``); ``**name**: text`` keeps the name.
      3. blank-line-separated paragraphs (headings skipped) -> auto-named.
    """
    import re

    lines = text.splitlines()

    # 1) h2 sections
    sections: list[tuple[str, list[str]]] = []
    cur_name: str | None = None
    cur_body: list[str] = []
    for line in lines:
        m = re.match(r"^##\s+(.+?)\s*$", line)
        if m:
            if cur_name is not None:
                sections.append((cur_name, cur_body))
            cur_name, cur_body = m.group(1).strip(), []
        elif cur_name is not None:
            cur_body.append(line)
    if cur_name is not None:
        sections.append((cur_name, cur_body))
    named = [(n, "\n".join(b).strip()) for n, b in sections if "\n".join(b).strip()]
    if named:
        return named

    # 2) list items
    items = [
        m.group(1).strip()
        for line in lines
        if (m := re.match(r"^\s*(?:[-*]|\d+\.)\s+(.+)$", line))
    ]
    if items:
        out: list[tuple[str, str]] = []
        for i, it in enumerate(items):
            m = re.match(r"^\*\*(.+?)\*\*\s*[:：]\s*(.+)$", it)
            if m:
                out.append((m.group(1).strip(), m.group(2).strip()))
            else:
                out.append((f"item_{i}", it))
        return out

    # 3) paragraphs
    paras: list[str] = []
    buf: list[str] = []
    for line in lines:
        if line.strip().startswith("#"):
            continue
        if line.strip() == "":
            if buf:
                paras.append(" ".join(buf).strip())
                buf = []
        else:
            buf.append(line.strip())
    if buf:
        paras.append(" ".join(buf).strip())
    return [(f"item_{i}", p) for i, p in enumerate(paras) if p]


def _load_rubric(ctx: Any, rubric: Any, rubric_file: str | None) -> list[tuple[str, str]]:
    if rubric is not None:
        return _normalize_rubric(rubric)
    if rubric_file:
        text = _resolve_grader_file(ctx, rubric_file).read_text(encoding="utf-8", errors="ignore")
        parsed = _parse_rubric_md(text)
        if not parsed:
            raise ValueError(f"rubric file {rubric_file!r} parsed to zero criteria")
        return parsed
    raise ValueError("judge_rubric requires either rubric= or rubric_file=")


def judge_rubric(
    judge: Any,
    ctx: Any,
    *,
    rubric: Any = None,
    rubric_file: str | None = None,
    read_files: list[str] | None = None,
    total_weight: float = 1.0,
    name_prefix: str = "llm::",
    extra_context: str = "",
    retries: int = JUDGE_DEFAULT_RETRIES,
) -> list[dict[str, Any]]:
    """Score a list of rubric criteria with the injected judge client.

    Pass the rubric ONE of two ways:
      - ``rubric_file="rubric.md"`` (preferred): keep criterion prose out of
        Python. Resolved against grader_assets_path / grader_mount_path and
        parsed by ``_parse_rubric_md`` (``## name`` sections recommended).
      - ``rubric=[(name, text), ...]`` or ``["text", ...]`` for programmatic use.

    Returns one ScoreItem per criterion. Weights default to ``total_weight/n``;
    per-criterion weights may be declared via ``## name (weight: 0.3)`` headings
    in rubric.md or ``(name, text, weight)`` tuples, normalized to ``total_weight``.
    On no judge, the item is ``skipped`` (never ``failed``). Runtime judge
    infrastructure failures are re-raised so the rollout can be dropped instead
    of trained as score 0; malformed per-item responses still skip that item.
    """
    rubric = _load_rubric(ctx, rubric, rubric_file)
    # Per-criterion weights (2026-06-10, user-approved contract extension):
    #   - rubric.md headings may declare ``## name (weight: 0.3)`` (or ``30%``);
    #   - programmatic rubrics may pass 3-tuples ``(name, text, weight)``.
    # Declared weights are normalized so they sum to ``total_weight``; criteria
    # without a declared weight share the mean declared weight; when nothing is
    # declared, the historical uniform ``total_weight/n`` split applies.
    import re as _re_w

    _names: list[str] = []
    _texts: list[Any] = []
    _raw_w: list[float | None] = []
    for _item in rubric:
        if isinstance(_item, (list, tuple)) and len(_item) == 3:
            _n, _t, _w = _item
            _w = float(_w)
        else:
            _n, _t = _item
            _w = None
        _m = _re_w.match(r"^(.*?)\s*\(\s*weight\s*[:=]?\s*([0-9.]+\s*%?)\s*\)\s*$", str(_n), _re_w.I)
        if _m is not None and _w is None:
            _n = _m.group(1).strip()
            _ws = _m.group(2).replace(" ", "")
            _w = float(_ws[:-1]) / 100.0 if _ws.endswith("%") else float(_ws)
        _names.append(str(_n))
        _texts.append(_t)
        _raw_w.append(_w)
    if any(_w is not None for _w in _raw_w):
        _declared = [_w for _w in _raw_w if _w is not None]
        _fill = sum(_declared) / len(_declared)
        _ws_full = [_w if _w is not None else _fill for _w in _raw_w]
        _scale = sum(_ws_full) or 1.0
        pers = [total_weight * _w / _scale for _w in _ws_full]
    else:
        pers = [total_weight / (len(rubric) or 1)] * len(rubric)
    rubric = list(zip(_names, _texts))

    if judge is None:
        return [
            skipped_item(
                f"{name_prefix}{name}",
                weight=pers[idx],
                reason="No judge runtime available; LLM rubric item not evaluated.",
            )
            for idx, (name, _) in enumerate(rubric)
        ]

    context, media_parts = _read_files_payload(ctx, read_files)
    if extra_context:
        context = (
            f"{context}\n\n===== additional context =====\n{extra_context}"
            if context
            else extra_context
        )
    model = getattr(judge, "model_id", None) or getattr(judge, "model", None) or "default"
    client = getattr(judge, "client", judge)

    import json as _json
    import re as _re
    import time as _time

    items: list[dict[str, Any]] = []
    for idx, (name, criterion) in enumerate(rubric):
        prompt = (
            "You are grading ONE rubric criterion for an agent's output.\n"
            'Respond with ONLY a JSON object: {"score": <float 0..1>, "reason": "..."}.\n'
            "1.0 = criterion fully met, 0.0 = not met, values in between for partial.\n\n"
            f"Criterion:\n{criterion}\n\n"
            f"Agent output / context:\n{context}\n"
        )
        last_err: Exception | None = None
        scored: dict[str, Any] | None = None
        for attempt in range(max(1, retries)):
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=[
                        {
                            "role": "user",
                            "content": _judge_user_content(prompt, media_parts),
                        }
                    ],
                    temperature=0.0,
                    max_tokens=_judge_max_tokens(),
                )
                choice = resp.choices[0]
                content = getattr(choice.message, "content", None)
                # Empty content => the model spent its whole budget on reasoning
                # and never emitted the verdict (finish_reason "length"). Treat
                # as a failed attempt so it retries and, if still empty, ends up
                # SKIPPED -- never a silent 0.0 that would punish a good answer.
                if content is None or not str(content).strip():
                    raise ValueError(
                        "judge returned empty content (likely truncated by "
                        "max_tokens / reasoning budget; finish_reason="
                        f"{getattr(choice, 'finish_reason', None)!r}). Raise "
                        "OMNI_JUDGE_MAX_TOKENS or use a judge that fits the cap."
                    )
                raw = _re.sub(r"^```(?:json)?\s*|\s*```$", "", str(content).strip())
                parsed = _json.loads(raw)
                score = max(0.0, min(1.0, float(parsed.get("score", 0.0))))
                reason = str(parsed.get("reason", ""))[:400]
                scored = score_item(f"{name_prefix}{name}", score, weight=pers[idx], reason=reason)
                break
            except Exception as exc:  # noqa: BLE001 - provider/parse surfaces vary
                if _is_judge_runtime_error(exc):
                    raise
                last_err = exc
                if attempt + 1 < max(1, retries):
                    _time.sleep(min(2 ** attempt, 8))
        if scored is None:
            scored = skipped_item(
                f"{name_prefix}{name}",
                weight=pers[idx],
                reason=f"Judge call failed after retries, item not evaluated: {last_err!r}",
            )
        items.append(scored)
    return items


def validate_score_items(items: list[dict[str, Any]], *, function_name: str) -> None:
    if not isinstance(items, list):
        raise ValueError("score items must be a list")
    has_overall = False
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("score item must be an object")
        name = item.get("name")
        score = item.get("score")
        status = item.get("status")
        weight = item.get("weight", 0.0)
        if not isinstance(name, str) or not name:
            raise ValueError("score item missing name")
        if score is not None and (
            not isinstance(score, (int, float))
            or isinstance(score, bool)
            or score < 0.0
            or score > 1.0
        ):
            raise ValueError(f"score item {name!r} has invalid score")
        if status not in VALID_SCORE_STATUSES:
            raise ValueError(f"score item {name!r} has invalid status")
        if not isinstance(weight, (int, float)) or isinstance(weight, bool) or weight < 0:
            raise ValueError(f"score item {name!r} has invalid weight")
        if function_name == "llm_judge" and "gate" in item:
            raise ValueError("llm_judge must not return gate items")
        if name == "overall":
            has_overall = True

    if has_overall:
        for item in items:
            if item.get("name") != "overall" and float(item.get("weight", 0.0)) != 0.0:
                raise ValueError("overall item requires all other item weights to be 0")


def aggregate_score_items(items: list[dict[str, Any]]) -> dict[str, Any]:
    validate_score_items(items, function_name="grade")
    total_declared_weight = sum(float(item.get("weight", 0.0)) for item in items)
    skipped_weight = sum(
        float(item.get("weight", 0.0))
        for item in items
        if item.get("status") == "skipped" and float(item.get("weight", 0.0)) > 0
    )
    gate_failed = any(
        item.get("gate") == "zero_task" and item.get("status") == "failed"
        for item in items
    )
    overall = next((item for item in items if item.get("name") == "overall"), None)

    def result(
        *,
        evaluation_status: str,
        outcome_status: str,
        available_score: float | None,
        scored_weight: float,
    ) -> dict[str, Any]:
        return {
            "status": evaluation_status,
            "evaluation_status": evaluation_status,
            "outcome_status": outcome_status,
            "gate_failed": gate_failed,
            "available_score": available_score,
            "scored_weight": scored_weight,
            "skipped_weight": skipped_weight,
            "total_declared_weight": total_declared_weight,
            "items": items,
        }

    if gate_failed:
        return result(
            evaluation_status="fully_evaluated",
            outcome_status="gate_failed",
            available_score=0.0,
            scored_weight=0.0,
        )
    if overall is not None:
        score = float(overall["score"])
        return result(
            evaluation_status="fully_evaluated",
            outcome_status="passed" if score >= 1.0 else "score_below_threshold",
            available_score=score,
            scored_weight=float(overall.get("weight", 0.0)),
        )

    scored = [
        item
        for item in items
        if item.get("score") is not None
        and item.get("status") != "skipped"
        and float(item.get("weight", 0.0)) > 0
    ]
    scored_weight = sum(float(item.get("weight", 0.0)) for item in scored)
    if scored_weight <= 0:
        return result(
            evaluation_status="skipped",
            outcome_status="not_scored",
            available_score=None,
            scored_weight=0.0,
        )
    available_score = sum(
        float(item["score"]) * float(item.get("weight", 0.0)) for item in scored
    ) / scored_weight
    has_skipped_score_weight = skipped_weight > 0
    if any(item.get("status") == "error" for item in scored):
        evaluation_status = "error"
    else:
        evaluation_status = "partial_evaluation" if has_skipped_score_weight else "fully_evaluated"
    return result(
        evaluation_status=evaluation_status,
        outcome_status="passed" if available_score >= 1.0 else "score_below_threshold",
        available_score=available_score,
        scored_weight=scored_weight,
    )


def validate_tool_trajectory(data: dict[str, Any]) -> None:
    if not isinstance(data, dict):
        raise ValueError("tool trajectory must be an object")
    if data.get("schema_version") != "omni_tool_trajectory.v1":
        raise ValueError("invalid tool trajectory schema_version")
    source = data.get("source")
    if not isinstance(source, dict):
        raise ValueError("tool trajectory source must be an object")
    if not source.get("runner") or not source.get("adapter"):
        raise ValueError("tool trajectory source missing runner/adapter")
    calls = data.get("calls")
    if not isinstance(calls, list):
        raise ValueError("tool trajectory calls must be a list")
    for index, call in enumerate(calls):
        if not isinstance(call, dict):
            raise ValueError("tool trajectory call must be an object")
        if call.get("call_index") != index:
            raise ValueError("tool trajectory call_index must be sequential")
        if not isinstance(call.get("tool_name"), str) or not call.get("tool_name"):
            raise ValueError("tool trajectory call missing tool_name")
        if "arguments" not in call:
            raise ValueError("tool trajectory call missing arguments")
        if call.get("status") not in VALID_TOOL_STATUSES:
            raise ValueError("tool trajectory call has invalid status")
        if not isinstance(call.get("source"), str) or not call.get("source"):
            raise ValueError("tool trajectory call missing source")


def validate_llm_messages(data: list[dict[str, Any]]) -> None:
    if not isinstance(data, list):
        raise ValueError("llm_messages must be a list")
    for message in data:
        if not isinstance(message, dict):
            raise ValueError("llm message must be an object")
        role = message.get("role")
        if role not in {"system", "user", "assistant", "tool"}:
            raise ValueError(f"invalid llm message role: {role!r}")
        if role == "assistant" and "tool_calls" in message:
            calls = message.get("tool_calls")
            if not isinstance(calls, list):
                raise ValueError("assistant tool_calls must be a list")
            for call in calls:
                if not isinstance(call, dict):
                    raise ValueError("assistant tool_call must be an object")
                function = call.get("function")
                if not isinstance(function, dict):
                    raise ValueError("assistant tool_call missing function")
                if not isinstance(function.get("name"), str) or not function.get("name"):
                    raise ValueError("assistant tool_call missing function.name")
                if "arguments" in function and not isinstance(function["arguments"], str):
                    raise ValueError("assistant tool_call function.arguments must be a string")
        if role == "tool":
            if not isinstance(message.get("tool_call_id"), str) or not message.get("tool_call_id"):
                raise ValueError("tool message missing tool_call_id")
            if "name" in message and not isinstance(message.get("name"), str):
                raise ValueError("tool message name must be a string")
            if "content" in message and not isinstance(message.get("content"), str):
                raise ValueError("tool message content must be a string")


def validate_audit_data(data: dict[str, Any]) -> None:
    if not isinstance(data, dict):
        raise ValueError("audit data must be an object")
    if data.get("schema_version") != "omni_audit_data.v1":
        raise ValueError("invalid audit data schema_version")
    source = data.get("source")
    if not isinstance(source, dict) or not source.get("runner"):
        raise ValueError("audit data source missing runner")
    services = data.get("services")
    if not isinstance(services, dict):
        raise ValueError("audit data services must be an object")
    for service_id, service in services.items():
        if not isinstance(service_id, str) or not service_id:
            raise ValueError("audit service id must be a non-empty string")
        if not isinstance(service, dict):
            raise ValueError(f"audit service {service_id!r} must be an object")
        if service.get("service_id") != service_id:
            raise ValueError(f"audit service {service_id!r} service_id mismatch")
        if not isinstance(service.get("actions"), list):
            raise ValueError(f"audit service {service_id!r} actions must be a list")
        if not isinstance(service.get("state"), dict):
            raise ValueError(f"audit service {service_id!r} state must be an object")
        for action in service.get("actions") or []:
            if not isinstance(action, dict):
                raise ValueError(f"audit service {service_id!r} action must be an object")
            if not isinstance(action.get("action_type"), str) or not action.get("action_type"):
                raise ValueError(f"audit service {service_id!r} action missing action_type")
            if not isinstance(action.get("arguments", {}), dict):
                raise ValueError(f"audit service {service_id!r} action arguments must be an object")
            if action.get("status") not in VALID_TOOL_STATUSES:
                raise ValueError(f"audit service {service_id!r} action has invalid status")
    if "artifacts" in data and not isinstance(data["artifacts"], dict):
        raise ValueError("audit data artifacts must be an object")
    if "metadata" in data and not isinstance(data["metadata"], dict):
        raise ValueError("audit data metadata must be an object")


def empty_audit_data(*, runner: str = "openclaw") -> dict[str, Any]:
    return {
        "schema_version": "omni_audit_data.v1",
        "source": {"runner": runner, "adapter": "service_audit_collector"},
        "services": {},
        "artifacts": {},
        "metadata": {},
    }


def canonical_audit_data(
    audits: dict[str, Any] | None,
    *,
    runner: str = "openclaw",
    session_id: str | None = None,
) -> dict[str, Any]:
    payload = empty_audit_data(runner=runner)
    if session_id:
        payload["source"]["session_id"] = session_id
    if not isinstance(audits, dict):
        return payload
    for service_id, wrapped in audits.items():
        if not isinstance(service_id, str) or not service_id:
            continue
        if isinstance(wrapped, dict) and "audit_data" in wrapped:
            audit = wrapped.get("audit_data")
            raw = wrapped.get("raw_audit_data", audit)
            audit_url = wrapped.get("audit_url") or ""
        else:
            audit = wrapped
            raw = wrapped
            audit_url = ""
        state = audit if isinstance(audit, dict) else {}
        actions = _audit_actions_from_state(state)
        payload["services"][service_id] = {
            "service_id": service_id,
            "source_paths": [f"/workspace/audit_{service_id}.json"],
            "actions": actions,
            "state": state,
            "raw_reports": [
                {
                    "path": f"/workspace/audit_{service_id}.json",
                    "audit_url": audit_url,
                    "data": raw,
                }
            ],
        }
    return payload


def _audit_actions_from_state(state: dict[str, Any]) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    flat = state.get("expected_actions_flat")
    if isinstance(flat, list):
        for row in flat:
            if not isinstance(row, dict):
                continue
            action_type = str(row.get("action_type") or row.get("action_key") or "action")
            arguments = {
                key: value
                for key, value in row.items()
                if key not in {"action_type", "action_key", "status", "timestamp", "result"}
            }
            actions.append(
                {
                    "action_type": action_type,
                    "arguments": arguments,
                    "result": row.get("result"),
                    "status": str(row.get("status") or "ok"),
                    "timestamp": row.get("timestamp"),
                }
            )
        return actions

    for key, value in state.items():
        if key in {"calls", "expected_actions", "expected_actions_flat"}:
            continue
        if not isinstance(value, list):
            continue
        for row in value:
            arguments = row if isinstance(row, dict) else {"value": row}
            actions.append(
                {
                    "action_type": key,
                    "arguments": dict(arguments),
                    "result": None,
                    "status": str(arguments.get("status") or "ok")
                    if isinstance(arguments, dict)
                    else "ok",
                    "timestamp": arguments.get("timestamp")
                    if isinstance(arguments, dict)
                    else None,
                }
            )
    return actions
