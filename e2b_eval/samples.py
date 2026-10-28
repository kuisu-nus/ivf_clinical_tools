"""Load and validate Dressage-claw ``samples.jsonl`` files.

A sample is one JSON object per line::

    {
      "prompt": "...",
      "label": "",
      "reward_fn": "claw_grader",
      "metadata": {
        "instance_id": "...",
        "sandbox_image": "e2b-dressage-claw-blackbox",
        "system_prompt_file": "/tmp/omni_task/runtime/system_prompt_prefix.txt",
        "before_agent_files": [{"path": "/tmp/...", "local_path": "/host/..."}],
        "after_agent_files":  [{"path": "/tmp/omni_grader/...", "local_path": "/host/..."}],
        "blackbox_execute_cmds": {
          "before_agent": [{"name": ..., "cmd": ..., "required": true, "timeout": 120}],
          "after_agent":  [...]
        }
      }
    }

The runner only needs these fields.  Local paths in a file produced on
another machine are remapped with ``--task-root`` and ``--path-map``.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .util import case_key

GRADER_PREFIX = "/tmp/omni_grader"
STAGES = ("before_agent", "after_agent")
_PATH_MARKERS = ("/task/", "/grader/", "/outputs/", "/runs/")


class SampleError(RuntimeError):
    pass


@dataclass(frozen=True)
class Upload:
    sandbox_path: str
    local_path: Path
    raw_local_path: str = ""

    def describe(self) -> dict[str, Any]:
        return {
            "path": self.sandbox_path,
            "local_path": str(self.local_path),
            "source": self.raw_local_path or str(self.local_path),
            "bytes": self.local_path.stat().st_size if self.local_path.is_file() else None,
        }


@dataclass
class Sample:
    index: int
    raw: dict[str, Any]
    source: Path
    prompt: str
    label: str
    reward_fn: str
    metadata: dict[str, Any]

    @property
    def instance_id(self) -> str:
        return str(self.metadata["instance_id"])

    @property
    def task_id(self) -> str:
        return str(self.metadata.get("task_id") or "")

    @property
    def case_key(self) -> str:
        return case_key(self.instance_id, index=self.index)

    @property
    def template(self) -> str | None:
        value = self.metadata.get("sandbox_image")
        return str(value) if value else None

    @property
    def sandbox_timeout_sec(self) -> int | None:
        value = self.metadata.get("sandbox_timeout_sec")
        if value is None:
            return None
        try:
            timeout = int(value)
        except (TypeError, ValueError):
            return None
        return timeout if timeout > 0 else None

    @property
    def system_prompt_file(self) -> str | None:
        value = self.metadata.get("system_prompt_file")
        return str(value) if value else None

    def files(self, stage: str) -> list[dict[str, Any]]:
        if stage not in STAGES:
            raise SampleError(f"unknown stage: {stage}")
        value = self.metadata.get(f"{stage}_files") or []
        return list(value)

    def commands(self, stage: str) -> list[dict[str, Any]]:
        if stage not in STAGES:
            raise SampleError(f"unknown stage: {stage}")
        schedule = self.metadata.get("blackbox_execute_cmds") or {}
        return list(schedule.get(stage) or [])

    def summary(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "instance_id": self.instance_id,
            "case_key": self.case_key,
            "task_id": self.task_id,
            "template": self.template,
            "sandbox_timeout_sec": self.sandbox_timeout_sec,
            "before_files": len(self.files("before_agent")),
            "after_files": len(self.files("after_agent")),
            "before_commands": [c.get("name") for c in self.commands("before_agent")],
            "after_commands": [c.get("name") for c in self.commands("after_agent")],
        }


def load_samples(
    path: str | Path,
    *,
    limit: int | None = None,
    only: Iterable[str] = (),
) -> list[Sample]:
    path = Path(path)
    if not path.is_file():
        raise SampleError(f"samples file not found: {path}")

    samples: list[Sample] = []
    seen: dict[str, int] = {}
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SampleError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            sample = _build_sample(raw, path, line_no, len(samples))
            if sample.case_key in seen:
                raise SampleError(
                    f"{path}:{line_no}: duplicate case key {sample.case_key!r} "
                    f"(first seen on line {seen[sample.case_key]})"
                )
            seen[sample.case_key] = line_no
            samples.append(sample)

    if not samples:
        raise SampleError(f"no samples found in {path}")

    tokens = [str(token).strip() for token in only if str(token).strip()]
    if tokens:
        selected: list[Sample] = []
        missing: list[str] = []
        for token in tokens:
            matched = [sample for sample in samples if _matches(sample, token)]
            if not matched:
                missing.append(token)
            for sample in matched:
                if sample not in selected:
                    selected.append(sample)
        if missing:
            available = ", ".join(sample.instance_id for sample in samples)
            raise SampleError(
                "no sample matched: " + ", ".join(missing) + f"\navailable: {available}"
            )
        samples = selected

    if limit is not None:
        if limit <= 0:
            raise SampleError("--limit must be positive")
        samples = samples[:limit]
    if not samples:
        raise SampleError("sample selection is empty")
    return samples


def resolve_uploads(
    sample: Sample,
    stage: str,
    *,
    samples_path: Path,
    task_root: Path | None,
    path_maps: Iterable[tuple[str, str]] = (),
) -> list[Upload]:
    uploads: list[Upload] = []
    seen: dict[str, Path] = {}
    path_maps = tuple(path_maps)
    for entry in sample.files(stage):
        sandbox_path = str(entry["path"])
        raw_local = str(entry["local_path"])
        local = resolve_local_path(
            raw_local,
            samples_path=Path(samples_path),
            task_root=task_root,
            path_maps=path_maps,
        )
        if sandbox_path in seen:
            raise SampleError(
                f"{sample.instance_id}: two {stage} files target {sandbox_path}: "
                f"{seen[sandbox_path]} and {local}"
            )
        seen[sandbox_path] = local
        uploads.append(Upload(sandbox_path=sandbox_path, local_path=local, raw_local_path=raw_local))
    return uploads


def resolve_local_path(
    raw_path: str,
    *,
    samples_path: Path,
    task_root: Path | None = None,
    path_maps: Iterable[tuple[str, str]] = (),
) -> Path:
    """Resolve a sample's ``local_path`` on this machine.

    Resolution order:
      1. the path itself, if it exists;
      2. an explicit ``--path-map OLD=NEW`` prefix;
      3. ``--task-root`` + the suffix after ``/task/``, ``/grader/``,
         ``/outputs/`` or ``/runs/``;
      4. ``<samples dir>/cases/...`` for files under a ``/cases/`` directory.
    """
    raw = os.path.expanduser(str(raw_path))
    path_maps = tuple(path_maps)
    direct = Path(raw)
    if direct.is_file():
        return direct.resolve()

    candidates: list[Path] = []
    for old, new in path_maps:
        old_prefix = str(old).rstrip("/")
        if raw == old_prefix:
            candidates.append(Path(new))
        elif raw.startswith(old_prefix + "/"):
            candidates.append(Path(new) / raw[len(old_prefix) + 1:])

    if task_root is not None:
        for marker in _PATH_MARKERS:
            index = raw.find(marker)
            if index >= 0:
                candidates.append(Path(task_root) / raw[index + 1:])

    marker = "/cases/"
    index = raw.find(marker)
    if index >= 0:
        candidates.append(Path(samples_path).parent / raw[index + 1:])

    existing: list[Path] = []
    for candidate in candidates:
        try:
            resolved = candidate.expanduser().resolve()
        except OSError:
            continue
        if resolved.is_file() and resolved not in existing:
            existing.append(resolved)

    if existing:
        return existing[0]

    hint = "\n".join(f"  tried: {candidate}" for candidate in candidates) or "  no remap rules matched"
    raise SampleError(
        f"local file not found: {raw}\n{hint}\n"
        "Pass --task-root <repo checkout> or --path-map OLD_PREFIX=NEW_PREFIX."
    )


def validate_upload_plan(
    sample: Sample,
    before: list[Upload],
    after: list[Upload],
) -> None:
    """Guard the grader isolation rule: no grader file may exist before the agent."""
    before_paths = {upload.sandbox_path for upload in before}
    after_paths = {upload.sandbox_path for upload in after}

    leaked = sorted(path for path in before_paths if _is_under(path, GRADER_PREFIX))
    if leaked:
        raise SampleError(
            f"{sample.instance_id}: before_agent_files contain grader paths: {leaked}; "
            "grader files must only be uploaded after the agent finishes"
        )
    overlap = sorted(before_paths & after_paths)
    if overlap:
        raise SampleError(
            f"{sample.instance_id}: paths uploaded in both stages: {overlap}"
        )

    prompt_file = sample.system_prompt_file
    if prompt_file and prompt_file not in before_paths:
        raise SampleError(
            f"{sample.instance_id}: system_prompt_file {prompt_file!r} is not in "
            "before_agent_files"
        )


def _build_sample(raw: Any, source: Path, line_no: int, index: int) -> Sample:
    if not isinstance(raw, dict):
        raise SampleError(f"{source}:{line_no}: sample must be a JSON object")
    prompt = raw.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise SampleError(f"{source}:{line_no}: prompt must be a non-empty string")
    metadata = raw.get("metadata")
    if not isinstance(metadata, dict):
        raise SampleError(f"{source}:{line_no}: metadata must be an object")
    instance_id = metadata.get("instance_id")
    if not isinstance(instance_id, str) or not instance_id.strip():
        raise SampleError(f"{source}:{line_no}: metadata.instance_id is required")
    blackbox_type = metadata.get("blackbox_type")
    if blackbox_type and str(blackbox_type).strip().lower() != "openclaw":
        raise SampleError(
            f"{source}:{line_no}: only blackbox_type=openclaw is supported, "
            f"got {blackbox_type!r}"
        )

    for stage in STAGES:
        files = metadata.get(f"{stage}_files") or []
        if not isinstance(files, list):
            raise SampleError(f"{source}:{line_no}: metadata.{stage}_files must be a list")
        for file_index, entry in enumerate(files):
            where = f"metadata.{stage}_files[{file_index}]"
            if not isinstance(entry, dict):
                raise SampleError(f"{source}:{line_no}: {where} must be an object")
            path = entry.get("path")
            local_path = entry.get("local_path")
            if not isinstance(path, str) or not path.startswith("/"):
                raise SampleError(f"{source}:{line_no}: {where}.path must be an absolute path")
            if not isinstance(local_path, str) or not local_path.strip():
                raise SampleError(f"{source}:{line_no}: {where}.local_path is required")

    schedule = metadata.get("blackbox_execute_cmds")
    if schedule is not None:
        if not isinstance(schedule, dict):
            raise SampleError(
                f"{source}:{line_no}: metadata.blackbox_execute_cmds must be an object"
            )
        unknown = sorted(str(stage) for stage in schedule if stage not in STAGES)
        if unknown:
            raise SampleError(
                f"{source}:{line_no}: unsupported blackbox_execute_cmds stage(s): {unknown}"
            )
        for stage in STAGES:
            commands = schedule.get(stage) or []
            if not isinstance(commands, list):
                raise SampleError(
                    f"{source}:{line_no}: metadata.blackbox_execute_cmds.{stage} must be a list"
                )
            for command_index, command in enumerate(commands):
                where = f"metadata.blackbox_execute_cmds.{stage}[{command_index}]"
                if not isinstance(command, dict):
                    raise SampleError(f"{source}:{line_no}: {where} must be an object")
                for key in ("name", "cmd", "required"):
                    if key not in command:
                        raise SampleError(f"{source}:{line_no}: {where}.{key} is required")
                if not isinstance(command["name"], str) or not command["name"].strip():
                    raise SampleError(f"{source}:{line_no}: {where}.name must be a string")
                if not isinstance(command["cmd"], str) or not command["cmd"].strip():
                    raise SampleError(f"{source}:{line_no}: {where}.cmd must be a string")
                if not isinstance(command["required"], bool):
                    raise SampleError(f"{source}:{line_no}: {where}.required must be a bool")
                if "\n" in command["cmd"] or "\r" in command["cmd"]:
                    raise SampleError(
                        f"{source}:{line_no}: {where}.cmd must be a single line "
                        "(blackbox execute_cmd rejects newlines)"
                    )

    return Sample(
        index=index,
        raw=raw,
        source=source,
        prompt=prompt,
        label=str(raw.get("label") or ""),
        reward_fn=str(raw.get("reward_fn") or ""),
        metadata=metadata,
    )


def _matches(sample: Sample, token: str) -> bool:
    return (
        token == sample.instance_id
        or token == sample.case_key
        or sample.instance_id.endswith(f"-{token}")
        or sample.instance_id.endswith(f"_{token}")
    )


def _is_under(path: str, prefix: str) -> bool:
    prefix = prefix.rstrip("/")
    return path == prefix or path.startswith(prefix + "/")
