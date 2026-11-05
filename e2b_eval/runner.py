"""Three-stage orchestration of one sample inside one E2B sandbox."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shutil
import time
import traceback
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .blackbox import BlackboxClient
from .config import AgentLLMConfig, E2BConfig
from .samples import (
    Sample,
    Upload,
    load_samples,
    resolve_uploads,
    validate_upload_plan,
)
from .sandbox import Sandbox
from .util import (
    command_ok,
    read_json,
    redact,
    shell_quote as _shell_quote,
    truncate,
    write_json,
    write_text,
)

log = logging.getLogger(__name__)

DISPATCHES_PATH = "/tmp/omni_task/trace/dispatches.jsonl"
AGENT_MESSAGES_PATH = "/tmp/omni_grader/agent_messages.json"
GRADER_METADATA_PATH = "/tmp/omni_grader/metadata.json"
GRADER_FILES = (
    "/tmp/omni_grader/tool_trajectory.json",
    "/tmp/omni_grader/llm_messages.json",
    "/tmp/omni_grader/agent_messages.json",
)


class StageError(RuntimeError):
    pass


@dataclass
class RunOptions:
    samples_path: Path
    out_dir: Path
    limit: int | None = None
    only: tuple[str, ...] = ()
    workers: int = 1
    resume: bool = True
    keep_sandbox: bool = False
    dry_run: bool = False
    task_root: Path | None = None
    path_maps: tuple[tuple[str, str], ...] = ()
    template_override: str | None = None
    sandbox_timeout_override: int | None = None
    e2b: E2BConfig | None = None
    agent: AgentLLMConfig | None = None


@dataclass
class CaseOutcome:
    case_key: str
    instance_id: str
    status: str
    score: float | None = None
    prediction: str | None = None
    truth: str | None = None
    error: str | None = None
    duration_ms: int = 0
    agent_mode: str = ""
    tool_calls: int | None = None
    case_dir: str | None = None
    skipped: bool = False

    def compact(self) -> str:
        if self.skipped:
            return f"{self.case_key}: already done, skipped"
        if self.status != "ok":
            return f"{self.case_key}: FAILED {self.error or ''}".strip()
        score = "?" if self.score is None else f"{self.score:.3f}"
        return (
            f"{self.case_key}: ok score {score} "
            f"predicted {self.prediction or '-'} truth {self.truth or '-'} "
            f"tool_calls {self.tool_calls if self.tool_calls is not None else '-'}"
        )


def run_all(options: RunOptions) -> int:
    if options.e2b is None or options.agent is None:
        raise ValueError("RunOptions.e2b and RunOptions.agent are required")
    samples = load_samples(options.samples_path, limit=options.limit, only=options.only)

    if options.dry_run:
        return _dry_run(samples, options)

    options.out_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    outcomes: list[CaseOutcome] = []

    if options.workers <= 1:
        for index, sample in enumerate(samples, 1):
            log.info("[%d/%d] case %s", index, len(samples), sample.instance_id)
            outcomes.append(asyncio.run(run_case(sample, options)))
    else:
        outcomes = asyncio.run(_run_concurrent(samples, options))

    summary = _build_summary(outcomes, elapsed_sec=time.monotonic() - started, options=options)
    write_json(options.out_dir / "summary.json", summary)
    _print_summary(outcomes, summary)
    return 0 if not any(outcome.status == "failed" for outcome in outcomes) else 1


async def _run_concurrent(samples: list[Sample], options: RunOptions) -> list[CaseOutcome]:
    semaphore = asyncio.Semaphore(max(1, options.workers))

    async def one(index: int, sample: Sample) -> CaseOutcome:
        async with semaphore:
            log.info("[%d/%d] case %s", index, len(samples), sample.instance_id)
            return await run_case(sample, options)

    return list(await asyncio.gather(*(one(i, s) for i, s in enumerate(samples, 1))))


async def run_case(sample: Sample, options: RunOptions) -> CaseOutcome:
    assert options.e2b is not None and options.agent is not None
    case_dir = options.out_dir / "cases" / sample.case_key
    started = time.monotonic()
    started_at = _now()

    if options.resume and (case_dir / "score.json").is_file():
        outcome = _outcome_from_existing(sample, case_dir)
        if outcome is not None:
            log.info("%s: already has score.json, skipped", sample.case_key)
            return outcome

    if case_dir.exists():
        shutil.rmtree(case_dir)
    case_dir.mkdir(parents=True, exist_ok=True)

    status = "failed"
    stage = "prepare"
    error: str | None = None
    sandbox: Sandbox | None = None
    client: BlackboxClient | None = None
    session_id: str | None = None
    answer = ""
    grader: dict[str, Any] | None = None
    agent_data: dict[str, Any] | None = None
    agent_run: dict[str, Any] = {}
    tool_calls: int | None = None
    prediction: str | None = None
    truth: str | None = None
    score_value: float | None = None
    uploads_before: list[Upload] = []
    uploads_after: list[Upload] = []
    dispatches_text: str | None = None

    try:
        stage = "plan"
        uploads_before = resolve_uploads(
            sample,
            "before_agent",
            samples_path=options.samples_path,
            task_root=options.task_root,
            path_maps=options.path_maps,
        )
        uploads_after = resolve_uploads(
            sample,
            "after_agent",
            samples_path=options.samples_path,
            task_root=options.task_root,
            path_maps=options.path_maps,
        )
        validate_upload_plan(sample, uploads_before, uploads_after)
        write_json(case_dir / "sample.json", sample.raw)
        template = (
            options.template_override
            or sample.template
            or options.e2b.template
        )
        timeout_sec = (
            options.sandbox_timeout_override
            or sample.sandbox_timeout_sec
            or options.e2b.timeout_sec
        )
        write_json(
            case_dir / "plan.json",
            {
                "instance_id": sample.instance_id,
                "case_key": sample.case_key,
                "template": template,
                "sandbox_timeout_sec": timeout_sec,
                "before_files": [upload.describe() for upload in uploads_before],
                "after_files": [upload.describe() for upload in uploads_after],
                "system_prompt_file": sample.system_prompt_file,
            },
        )

        stage = "sandbox"
        metadata = {
            "instance_id": sample.instance_id,
            "task_id": sample.task_id,
            "runner": "e2b_eval",
        }
        sandbox = await Sandbox.create(
            options.e2b,
            template=template,
            timeout_sec=timeout_sec,
            metadata=metadata,
        )
        endpoint = await sandbox.endpoint()
        session_id = _new_session_id(sample)
        client = BlackboxClient(
            endpoint,
            request_timeout=max(120.0, options.agent.turn_timeout_sec + 60.0),
        )

        # ------------------------------------------------------------------
        # Stage 1: before_agent
        # ------------------------------------------------------------------
        stage = "before_agent"
        uploaded_before = await sandbox.write_files(uploads_before)
        write_json(case_dir / "before_files.json", uploaded_before)
        health = await client.wait_healthy(timeout_sec=120.0)
        register = await client.register(
            session_id=session_id,
            instance_id=_bbs_identifier(sample.instance_id),
            system_prompt_file=sample.system_prompt_file,
            agent=options.agent,
            sandbox_id=sandbox.sandbox_id,
        )
        write_json(
            case_dir / "before_agent.json",
            {
                "stage": "before_agent",
                "sandbox_id": sandbox.sandbox_id,
                "template": template,
                "endpoint": endpoint.describe(),
                "session_id": session_id,
                "health": health,
                "register": redact(register),
                "files": uploaded_before,
            },
        )
        await _run_commands(
            client,
            session_id,
            sample.commands("before_agent"),
            case_dir,
            "before_agent",
        )
        tool_health = await _probe_tools(client, session_id)
        write_json(case_dir / "tools_health.json", tool_health)
        if not command_ok(tool_health):
            raise StageError(
                "tool services are not healthy after before_agent; "
                "see tools_health.json and before_agent.log"
            )

        # ------------------------------------------------------------------
        # Stage 2: agent
        # ------------------------------------------------------------------
        stage = "agent"
        agent_started = time.monotonic()
        if options.agent.mode == "mock":
            agent_data, history = await _run_mock_agent(
                client, session_id, sample, uploads_after
            )
        else:
            agent_data = await _send_agent_turn(client, session_id, sample, options.agent)
            history = await client.fetch_session(
                session_id, include_history=True, include_trace=True
            )
        agent_ms = int((time.monotonic() - agent_started) * 1000)

        write_json(case_dir / "agent_response.json", agent_data)
        write_json(case_dir / "history.json", history)
        answer = _extract_answer(agent_data) or _extract_answer(history)
        if not answer.strip():
            raise StageError("agent finished but returned no assistant text")
        write_text(case_dir / "answer.txt", answer)

        dispatches_text = await sandbox.read_text(DISPATCHES_PATH)
        if dispatches_text is not None:
            write_text(case_dir / "dispatches.jsonl", dispatches_text)
        tool_calls = _count_dispatch_lines(dispatches_text)

        messages = history.get("conversation_history") or agent_data.get("outputs") or []
        if not isinstance(messages, list):
            messages = []
        agent_run = {
            "mode": options.agent.mode,
            "model": options.agent.model,
            "provider_id": options.agent.provider_id,
            "session_id": session_id,
            "turn_id": agent_data.get("turn_id"),
            "duration_ms": agent_ms,
            "usage": agent_data.get("usage"),
            "tool_calls": tool_calls,
            "finished_at": _now(),
        }
        write_json(case_dir / "agent_run.json", agent_run)

        agent_messages = {
            "prompt": sample.prompt,
            "response": answer,
            "messages": messages,
        }
        write_json(case_dir / "agent_messages.json", agent_messages)
        await sandbox.write_files(
            [
                Upload(
                    sandbox_path=AGENT_MESSAGES_PATH,
                    local_path=case_dir / "agent_messages.json",
                )
            ]
        )

        # ------------------------------------------------------------------
        # Stage 3: after_agent (grader files appear only now)
        # ------------------------------------------------------------------
        stage = "after_agent"
        uploaded_after = await sandbox.write_files(uploads_after)
        write_json(case_dir / "after_files.json", uploaded_after)
        after_results = await _run_commands(
            client,
            session_id,
            sample.commands("after_agent"),
            case_dir,
            "after_agent",
        )

        grader = _parse_grader_output(after_results)
        if grader is None:
            grader = await _read_grader_file(sandbox)
        if grader is None:
            raise StageError(
                "grader produced no JSON on stdout and no grader_output.json was found; "
                "see after_agent.log"
            )
        write_json(case_dir / "grader_output.json", grader)

        raw_score = grader.get("overall_score")
        score_value = float(raw_score) if isinstance(raw_score, (int, float)) else None
        prediction, truth = _extract_outcome(grader)
        score_payload = {
            "case_key": sample.case_key,
            "instance_id": sample.instance_id,
            "task_id": sample.task_id,
            "status": "ok",
            "score": score_value,
            "prediction": prediction,
            "truth": truth,
            "agent": agent_run,
            "grader": grader,
            "finished_at": _now(),
        }
        write_json(case_dir / "score.json", score_payload)
        status = "ok"

    except Exception as exc:  # noqa: BLE001 - one case must not stop the batch
        error = f"{stage}: {type(exc).__name__}: {exc}"
        log.error("%s: %s", sample.case_key, error)
        write_text(case_dir / "error.txt", traceback.format_exc())
        if client is not None and session_id is not None:
            await client.abort(session_id)

    finally:
        if sandbox is not None:
            await _collect_artifacts(sandbox, case_dir, dispatches_already=dispatches_text)
        if client is not None:
            await client.close()
        if sandbox is not None:
            if options.keep_sandbox:
                log.warning(
                    "keeping sandbox id=%s (template timeout will kill it later)",
                    sandbox.sandbox_id,
                )
            else:
                await sandbox.kill()

        duration_ms = int((time.monotonic() - started) * 1000)
        outcome = CaseOutcome(
            case_key=sample.case_key,
            instance_id=sample.instance_id,
            status=status,
            score=score_value,
            prediction=prediction,
            truth=truth,
            error=error,
            duration_ms=duration_ms,
            agent_mode=options.agent.mode,
            tool_calls=tool_calls,
            case_dir=str(case_dir),
            skipped=False,
        )
        write_json(
            case_dir / "case.json",
            {
                **asdict(outcome),
                "status": status,
                "started_at": started_at,
                "finished_at": _now(),
                "sandbox_id": sandbox.sandbox_id if sandbox else None,
                "answer_chars": len(answer),
                "agent_run": agent_run,
            },
        )
        return outcome


async def _send_agent_turn(
    client: BlackboxClient,
    session_id: str,
    sample: Sample,
    agent: AgentLLMConfig,
) -> dict[str, Any]:
    return await client.send_message(
        session_id=session_id,
        prompt=sample.prompt,
        agent=agent,
        metadata={"source": "e2b_eval", "instance_id": sample.instance_id},
    )


async def _run_mock_agent(
    client: BlackboxClient,
    session_id: str,
    sample: Sample,
    uploads_after: list[Upload],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """A deterministic stand-in for OpenClaw, for pipeline checks without an LLM.

    It calls the same MCP tools an agent would call (so dispatches.jsonl and the
    tool trajectory are real) and fills the required answer template with the
    tool values.  It is not a model and its score says nothing about model
    quality.
    """
    metadata_local = _metadata_local_path(uploads_after)
    if metadata_local is None:
        raise StageError(
            "mock agent needs the after_agent metadata.json to know age/amh/gn_days"
        )
    metadata = read_json(metadata_local)
    inputs = metadata.get("inputs") if isinstance(metadata, dict) else None
    if not isinstance(inputs, dict):
        raise StageError(f"{metadata_local} has no inputs object")

    def args(include_gn: bool) -> dict[str, Any]:
        values: dict[str, Any] = {}
        for key in ("age", "amh") + (("gn_days",) if include_gn else ()):
            value = inputs.get(key)
            if value is not None:
                values[key] = value
        return values

    stage1_args = args(include_gn=False)
    stage2_args = args(include_gn=True)
    if not stage1_args:
        raise StageError("mock agent has no age/amh/gn_days to call the tools with")

    predict1 = await _mcp_call(client, session_id, "ivf_predict_outcome", stage1_args)
    similar = None
    if all(key in stage1_args for key in ("age", "amh")):
        similar = await _mcp_call(
            client, session_id, "ivf_similar_patients", stage1_args
        )
        # Mirror the system-prompt advice: widen the age window when the tool
        # refuses to report a cohort.
        if similar.get("拒绝统计") is True:
            for window in (2, 3):
                wider = dict(stage1_args, age_window=window)
                wider_result = await _mcp_call(
                    client, session_id, "ivf_similar_patients", wider
                )
                similar = wider_result
                if not wider_result.get("拒绝统计"):
                    break
    predict2 = await _mcp_call(
        client, session_id, "ivf_predict_outcome", stage2_args
    )

    p1 = _probability(predict1)
    p2 = _probability(predict2)
    sim_rate = None
    sim_n = None
    if isinstance(similar, dict) and not similar.get("拒绝统计"):
        distribution = similar.get("妊娠结局分布") or {}
        if isinstance(distribution, dict):
            sim_rate = distribution.get("妊娠率")
        sim_n = similar.get("符合条件例数")

    conclusion = _mock_conclusion(p1, p2, sim_rate)
    stage1 = f"【初诊】年龄 {_num(inputs.get('age'))} 岁，AMH {_num(inputs.get('amh'))}"
    if p1 is not None:
        stage1 += f"；模型 {_pct(p1)}"
    if sim_rate is not None and sim_n is not None:
        stage1 += f"，相似人群 {_pct(sim_rate)}（{int(sim_n)} 例）"
    if p1 is not None or sim_rate is not None:
        stage1 += "，低于 39%" if _below_reference(p1, sim_rate) else "，高于 39%"
    stage1 += "。"
    stage2 = f"【促排】Gn {_num(inputs.get('gn_days'))} 天"
    if p2 is not None:
        stage2 += f"；模型 {_pct(p2)}，判断不变"
    stage2 += "。"
    answer = "\n".join([stage1, stage2, "【胚胎】未提供。", f"结论：{conclusion}"])

    response = {
        "mock": True,
        "session_id": session_id,
        "turn_id": "mock",
        "outputs": [
            {
                "role": "assistant",
                "content": answer,
            }
        ],
        "usage": {"steps": 3, "tool_calls": 3 + (1 if similar and similar.get("拒绝统计") else 0)},
    }
    history = {
        "session_id": session_id,
        "conversation_history": [
            {"role": "user", "content": sample.prompt},
            {"role": "assistant", "content": answer},
        ],
    }
    return response, history


async def _mcp_call(
    client: BlackboxClient,
    session_id: str,
    tool: str,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    args_json = json.dumps(arguments, ensure_ascii=False, separators=(",", ":"))
    cmd = (
        "OMNI_TOOL_SPECS_DIR=/tmp/omni_task/tools OMNI_TRACE_DIR=/tmp/omni_task/trace "
        "NO_PROXY=127.0.0.1,localhost,0.0.0.0,::1 "
        "no_proxy=127.0.0.1,localhost,0.0.0.0,::1 "
        f"python3 /tmp/omni_task/runtime/claw_tools_mcp.py --call-tool {tool} "
        f"--arguments {_shell_quote(args_json)}"
    )
    result = await client.execute_cmd(session_id, cmd, timeout=60)
    if not command_ok(result):
        raise StageError(
            f"MCP call {tool} failed: rc={result.get('returncode')} "
            f"stderr={truncate(str(result.get('stderr') or ''), 2000)}"
        )
    stdout = str(result.get("stdout") or "")
    try:
        envelope = json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise StageError(f"MCP call {tool} returned non-JSON: {stdout[:1000]}") from exc
    if envelope.get("isError"):
        raise StageError(f"MCP call {tool} returned an error: {envelope}")
    content = envelope.get("content") or []
    if not content or not isinstance(content[0], dict):
        raise StageError(f"MCP call {tool} returned no content: {envelope}")
    text = content[0].get("text")
    try:
        return json.loads(text)
    except (TypeError, json.JSONDecodeError) as exc:
        raise StageError(f"MCP call {tool} returned invalid text content: {text!r}") from exc


async def _run_commands(
    client: BlackboxClient,
    session_id: str,
    commands: list[dict[str, Any]],
    case_dir: Path,
    stage: str,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    log_lines = [f"# {stage}"]
    failure: StageError | None = None

    for command in commands:
        name = str(command.get("name") or "command")
        cmd = str(command.get("cmd") or "")
        required = bool(command.get("required", True))
        timeout = command.get("timeout")
        started = time.monotonic()
        log.info("%s: run %s", stage, name)
        try:
            raw = await client.execute_cmd(
                session_id,
                cmd,
                timeout=float(timeout) if timeout is not None else 600.0,
            )
            result = {
                "name": name,
                "cmd": cmd,
                "required": required,
                "timeout": timeout,
                "stdout": truncate(str(raw.get("stdout") or "")),
                "stderr": truncate(str(raw.get("stderr") or "")),
                "returncode": raw.get("returncode"),
                "timed_out": bool(raw.get("timed_out")),
                "duration_ms": int(float(raw.get("duration_seconds") or 0) * 1000)
                if raw.get("duration_seconds") is not None
                else int((time.monotonic() - started) * 1000),
                "error": None,
            }
        except Exception as exc:  # noqa: BLE001
            result = {
                "name": name,
                "cmd": cmd,
                "required": required,
                "timeout": timeout,
                "stdout": "",
                "stderr": "",
                "returncode": None,
                "timed_out": False,
                "duration_ms": int((time.monotonic() - started) * 1000),
                "error": f"{type(exc).__name__}: {exc}",
            }
        results.append(result)
        log_lines.append(
            f"\n## {name} (rc={result['returncode']}, timed_out={result['timed_out']}, "
            f"{result['duration_ms']} ms)\n$ {cmd}\n{result['stdout']}"
            + (f"\n[stderr]\n{result['stderr']}" if result["stderr"] else "")
            + (f"\n[error]\n{result['error']}" if result["error"] else "")
        )
        if not command_ok(result) and required:
            failure = StageError(
                f"{stage} command {name!r} failed "
                f"(rc={result['returncode']}, timed_out={result['timed_out']}"
                + (f", error={result['error']}" if result["error"] else "")
                + "); see "
                f"{stage}.log"
            )
            break

    write_json(case_dir / f"{stage}_commands.json", results)
    write_text(case_dir / f"{stage}.log", "\n".join(log_lines))
    if failure is not None:
        raise failure
    return results


async def _probe_tools(client: BlackboxClient, session_id: str) -> dict[str, Any]:
    """Confirm both tool services answer from the sandbox after start_services."""
    # Read the ports at call time so an --env-file is honoured.
    similar_port = os.environ.get("IVF_SIMILAR_PORT", "9201")
    predict_port = os.environ.get("IVF_PREDICT_PORT", "9202")
    cases_port = os.environ.get("IVF_CASES_PORT", "9203")
    sql_port = os.environ.get("IVF_CASE_SQL_PORT", "9204")
    cmd = (
        f"echo '== ivf_similar'; curl -sf --noproxy '*' --max-time 10 "
        f"http://127.0.0.1:{similar_port}/health; "
        f"echo; echo '== ivf_predict'; curl -sf --noproxy '*' --max-time 10 "
        f"http://127.0.0.1:{predict_port}/health; "
        f"echo; echo '== ivf_cases'; curl -sf --noproxy '*' --max-time 10 "
        f"http://127.0.0.1:{cases_port}/health; "
        f"echo; echo '== ivf_case_sql'; curl -sf --noproxy '*' --max-time 10 "
        f"http://127.0.0.1:{sql_port}/health; echo"
    )
    started = time.monotonic()
    try:
        raw = await client.execute_cmd(session_id, cmd, timeout=30.0)
        return {
            "cmd": cmd,
            "stdout": str(raw.get("stdout") or ""),
            "stderr": str(raw.get("stderr") or ""),
            "returncode": raw.get("returncode"),
            "timed_out": bool(raw.get("timed_out")),
            "duration_ms": int((time.monotonic() - started) * 1000),
            "error": None,
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "cmd": cmd,
            "stdout": "",
            "stderr": "",
            "returncode": None,
            "timed_out": False,
            "duration_ms": int((time.monotonic() - started) * 1000),
            "error": f"{type(exc).__name__}: {exc}",
        }


async def _collect_artifacts(
    sandbox: Sandbox,
    case_dir: Path,
    *,
    dispatches_already: str | None,
) -> None:
    """Best-effort download of grader inputs and logs; never raises."""
    try:
        if dispatches_already is None:
            dispatches = await sandbox.read_text(DISPATCHES_PATH, max_bytes=4_000_000)
            if dispatches:
                write_text(case_dir / "dispatches.jsonl", dispatches)
    except Exception as exc:  # noqa: BLE001
        log.debug("could not collect dispatches: %s", exc)

    for remote in GRADER_FILES:
        safe_name = remote.rsplit("/", 1)[-1]
        try:
            text = await sandbox.read_text(remote, max_bytes=4_000_000)
            if text:
                write_text(case_dir / f"grader_{safe_name}", text)
        except Exception as exc:  # noqa: BLE001
            log.debug("could not collect %s: %s", remote, exc)

    service_cmd = (
        "for f in /tmp/omni_task/service-*.log; do "
        '[ -f "$f" ] || continue; echo "===== $f"; tail -c 40000 "$f"; echo; done'
    )
    openclaw_cmd = (
        "for d in /workspace_sandbox/blackbox_server_runtime/bbs-*; do "
        '[ -d "$d" ] || continue; echo "===== $d"; '
        'for f in "$d"/logs/*.log; do [ -f "$f" ] || continue; '
        'echo "----- $f"; tail -c 30000 "$f"; echo; done; done'
    )
    for name, cmd in (("service_logs.txt", service_cmd), ("openclaw_logs.txt", openclaw_cmd)):
        try:
            result = await sandbox.run(cmd, timeout=60.0)
            if result.stdout or result.stderr:
                payload = result.as_dict()
                write_text(case_dir / name, payload["stdout"] + payload["stderr"])
        except Exception as exc:  # noqa: BLE001
            log.debug("could not collect %s: %s", name, exc)


def _parse_grader_output(results: list[dict[str, Any]]) -> dict[str, Any] | None:
    chosen = None
    for result in results:
        if result.get("name") == "run_grader":
            chosen = result
    if chosen is None and results:
        chosen = results[-1]
    if chosen is None:
        return None
    text = str(chosen.get("stdout") or "")
    return _parse_grader_text(text)


def _parse_grader_text(text: str) -> dict[str, Any] | None:
    text = (text or "").strip()
    if not text:
        return None
    for candidate in (text, text.splitlines()[-1] if text.splitlines() else ""):
        try:
            data = json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(data, dict) and ("overall_score" in data or "raw_scores" in data):
            return data
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            data = json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            return None
        if isinstance(data, dict) and ("overall_score" in data or "raw_scores" in data):
            return data
    return None


async def _read_grader_file(sandbox: Sandbox) -> dict[str, Any] | None:
    for path in ("/tmp/omni_grader/grader_output.json", "/tmp/omni_grader/score.json"):
        text = await sandbox.read_text(path, max_bytes=4_000_000)
        if text:
            data = _parse_grader_text(text)
            if data is not None:
                return data
    return None


def _extract_answer(payload: Any) -> str:
    if not isinstance(payload, dict):
        return ""
    for key in ("outputs", "conversation_history", "messages"):
        messages = payload.get(key)
        if not isinstance(messages, list):
            continue
        for message in reversed(messages):
            if not isinstance(message, dict) or message.get("role") != "assistant":
                continue
            if message.get("tool_calls"):
                continue
            content = message.get("content")
            if isinstance(content, list):
                parts = [
                    part if isinstance(part, str) else (part.get("text") if isinstance(part, dict) else "")
                    for part in content
                ]
                content = "\n".join(part for part in parts if isinstance(part, str))
            if isinstance(content, str) and content.strip():
                return content.strip()
    content = payload.get("answer") or payload.get("response")
    if isinstance(content, str) and content.strip():
        return content.strip()
    return ""


def _extract_outcome(grader: dict[str, Any]) -> tuple[str | None, str | None]:
    raw = grader.get("raw_scores")
    items = raw.get("items") if isinstance(raw, dict) else None
    if not isinstance(items, list):
        return None, None
    for item in items:
        if not isinstance(item, dict) or item.get("name") != "outcome":
            continue
        metadata = item.get("metadata")
        if not isinstance(metadata, dict):
            return None, None
        prediction = metadata.get("predicted")
        truth = metadata.get("truth")
        return (
            str(prediction) if prediction is not None else None,
            str(truth) if truth is not None else None,
        )
    return None, None


def _count_dispatch_lines(text: str | None) -> int:
    if not text:
        return 0
    return sum(1 for line in text.splitlines() if line.strip())


def _metadata_local_path(uploads: list[Upload]) -> Path | None:
    for upload in uploads:
        if upload.sandbox_path == GRADER_METADATA_PATH:
            return upload.local_path
    return None


def _outcome_from_existing(sample: Sample, case_dir: Path) -> CaseOutcome | None:
    try:
        score = read_json(case_dir / "score.json")
    except Exception as exc:  # noqa: BLE001
        log.warning("%s: score.json is unreadable, rerunning: %s", sample.case_key, exc)
        return None
    if not isinstance(score, dict):
        return None
    value = score.get("score")
    agent = score.get("agent") if isinstance(score.get("agent"), dict) else {}
    return CaseOutcome(
        case_key=sample.case_key,
        instance_id=sample.instance_id,
        status="ok",
        score=float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None,
        prediction=score.get("prediction"),
        truth=score.get("truth"),
        error=None,
        duration_ms=0,
        agent_mode=str(agent.get("mode") or ""),
        tool_calls=agent.get("tool_calls"),
        case_dir=str(case_dir),
        skipped=True,
    )


def _build_summary(
    outcomes: list[CaseOutcome],
    *,
    elapsed_sec: float,
    options: RunOptions,
) -> dict[str, Any]:
    completed = [o for o in outcomes if o.status == "ok"]
    failed = [o for o in outcomes if o.status == "failed"]
    scores = [o.score for o in completed if o.score is not None]
    answered = [
        o for o in completed if o.prediction in {"妊娠", "未妊娠"}
    ]
    correct = [o for o in answered if o.truth and o.prediction == o.truth]
    uncertain = [o for o in completed if o.prediction == "不确定"]
    predictions: dict[str, int] = {}
    truths: dict[str, int] = {}
    for outcome in completed:
        if outcome.prediction:
            predictions[outcome.prediction] = predictions.get(outcome.prediction, 0) + 1
        if outcome.truth:
            truths[outcome.truth] = truths.get(outcome.truth, 0) + 1
    return {
        "samples": str(options.samples_path),
        "out_dir": str(options.out_dir),
        "agent_mode": options.agent.mode if options.agent else None,
        "agent_model": options.agent.model if options.agent else None,
        "elapsed_sec": round(elapsed_sec, 1),
        "cases": len(outcomes),
        "ok": len(completed),
        "failed": len(failed),
        "skipped": sum(1 for o in outcomes if o.skipped),
        "mean_score": round(sum(scores) / len(scores), 4) if scores else None,
        "answered": len(answered),
        "correct": len(correct),
        "uncertain": len(uncertain),
        "accuracy_excluding_uncertain": round(len(correct) / len(answered), 4) if answered else None,
        "predictions": predictions,
        "truths": truths,
        "results": [asdict(outcome) for outcome in outcomes],
    }


def _print_summary(outcomes: list[CaseOutcome], summary: dict[str, Any]) -> None:
    for outcome in outcomes:
        print(outcome.compact())
    mean_score = summary.get("mean_score")
    mean_text = "n/a" if mean_score is None else f"{mean_score:.3f}"
    print(
        f"== {summary['cases']} case(s): {summary['ok']} ok, "
        f"{summary['failed']} failed, {summary['skipped']} skipped; "
        f"mean score {mean_text}; elapsed {summary['elapsed_sec']}s"
    )


def _dry_run(samples: list[Sample], options: RunOptions) -> int:
    errors: list[str] = []
    print(f"samples: {options.samples_path} ({len(samples)} case(s))")
    default_template = options.e2b.template if options.e2b else "(default)"
    default_timeout = options.e2b.timeout_sec if options.e2b else "(default)"
    for sample in samples:
        template = options.template_override or sample.template or default_template
        timeout = (
            options.sandbox_timeout_override
            or sample.sandbox_timeout_sec
            or default_timeout
        )
        print(f"\n[{sample.case_key}] {sample.instance_id}")
        print(
            f"  template: {template}  "
            f"timeout: {timeout}s  "
            f"system prompt: {sample.system_prompt_file or '(none)'}"
        )
        try:
            before = resolve_uploads(
                sample,
                "before_agent",
                samples_path=options.samples_path,
                task_root=options.task_root,
                path_maps=options.path_maps,
            )
            after = resolve_uploads(
                sample,
                "after_agent",
                samples_path=options.samples_path,
                task_root=options.task_root,
                path_maps=options.path_maps,
            )
            validate_upload_plan(sample, before, after)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{sample.case_key}: {exc}")
            print(f"  ERROR: {exc}")
            continue
        before_names = ", ".join(str(c.get("name")) for c in sample.commands("before_agent"))
        after_names = ", ".join(str(c.get("name")) for c in sample.commands("after_agent"))
        print(f"  before_agent: {len(before)} file(s), {len(sample.commands('before_agent'))} cmd(s): {before_names}")
        print(f"  after_agent:  {len(after)} file(s), {len(sample.commands('after_agent'))} cmd(s): {after_names}")
        leaked = [u.sandbox_path for u in before if u.sandbox_path.startswith("/tmp/omni_grader")]
        if leaked:
            errors.append(f"{sample.case_key}: grader path in before files: {leaked}")
    if errors:
        print("\nERRORS:")
        for error in errors:
            print(f"  {error}")
        return 1
    print("\ndry run ok; no sandbox was created")
    return 0


def _new_session_id(sample: Sample) -> str:
    base = _bbs_identifier(sample.instance_id)[:72]
    return f"bbs-{base}-{uuid.uuid4().hex[:8]}"


def _bbs_identifier(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._:-]+", "-", str(value or "")).strip("-._:")
    if not cleaned or not cleaned[0].isalnum():
        cleaned = f"case-{cleaned}"
    return cleaned[:128]


def _probability(payload: Any) -> float | None:
    if not isinstance(payload, dict):
        return None
    value = payload.get("妊娠概率")
    return float(value) if isinstance(value, (int, float)) else None


def _pct(value: float) -> str:
    return f"{float(value) * 100:.1f}%"


def _num(value: Any) -> str:
    if value is None:
        return "未提供"
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _below_reference(p1: float | None, sim_rate: float | None) -> bool:
    values = [v for v in (p1, sim_rate) if v is not None]
    return bool(values) and sum(values) / len(values) < 0.39


def _mock_conclusion(p1: float | None, p2: float | None, sim_rate: float | None) -> str:
    values = [v for v in (p1, p2, sim_rate) if v is not None]
    if not values:
        return "不确定"
    if all(v < 0.39 for v in values):
        return "未妊娠"
    if all(v >= 0.39 for v in values):
        return "妊娠"
    return "不确定"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
