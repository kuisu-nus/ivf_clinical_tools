#!/usr/bin/env python3
"""Generate tool_trajectory.json and llm_messages.json for V4 grader.

Reads:
  /tmp/omni_grader/agent_messages.json  — {"prompt": ..., "response": ..., "messages": [...]}
  /tmp/omni_task/trace/dispatches.jsonl  — tool dispatch events

Writes:
  /tmp/omni_grader/tool_trajectory.json  — omni_tool_trajectory.v1 schema
  /tmp/omni_grader/llm_messages.json     — raw message list [{role, content}]
"""

import json
import os
import pathlib

GRADER_DIR = pathlib.Path("/tmp/omni_grader")
TRACE_DIR = pathlib.Path("/tmp/omni_task/trace")

TOOL_PREFIX = os.environ.get("OMNI_MCP_SERVER_NAME", "claw-tools") + "__"


def _read_agent_messages() -> dict:
    """Read agent_messages.json; return {} when missing or invalid."""
    path = GRADER_DIR / "agent_messages.json"
    if not path.exists():
        print(f"[gen_trace] WARNING: agent_messages.json not found at {path}")
        return {}
    raw = path.read_text(encoding="utf-8")
    if not raw.strip():
        print(f"[gen_trace] WARNING: agent_messages.json is empty at {path}")
        return {}
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        print(f"[gen_trace] WARNING: agent_messages.json is invalid JSON: {exc}")
        return {}
    if not isinstance(data, dict):
        print(f"[gen_trace] WARNING: agent_messages.json is not a JSON object")
        return {}
    return data


def _read_dispatches():
    """Read dispatches.jsonl written by the MCP dispatcher."""
    path = TRACE_DIR / "dispatches.jsonl"
    records = []
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                records.append(json.loads(line))
    return records


def _strip_tool_prefix(tool_name: str) -> str:
    """Remove server-name__ prefix from tool name."""
    if tool_name.startswith(TOOL_PREFIX):
        return tool_name[len(TOOL_PREFIX):]
    return tool_name


def write_tool_trajectory(dispatches: list) -> None:
    """Write tool_trajectory.json in omni_tool_trajectory.v1 schema."""
    calls = []
    for dispatch in dispatches:
        tool_name = dispatch.get("tool_name", "")
        tool_name = _strip_tool_prefix(tool_name)
        response_status = dispatch.get("response_status", 0)
        status = "ok" if isinstance(response_status, int) and response_status < 400 else "error"
        calls.append({
            "call_index": len(calls),
            "tool_name": tool_name,
            "arguments": dispatch.get("request_body", {}),
            "status": status,
            "source": "mcp_dispatch",
            "response": dispatch.get("response_body", ""),
        })

    trajectory = {
        "schema_version": "omni_tool_trajectory.v1",
        "source": {"runner": "openclaw", "adapter": "mcp"},
        "calls": calls,
    }
    (GRADER_DIR / "tool_trajectory.json").write_text(
        json.dumps(trajectory, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _clean_message(msg: dict) -> dict:
    """Remove fields with None values that fail grader validation."""
    out = {}
    for k, v in msg.items():
        if v is None:
            continue
        if k == "tool_calls" and not isinstance(v, list):
            continue
        out[k] = v
    return out


def write_llm_messages(agent_data: dict) -> None:
    """Write llm_messages.json as raw message list.

    Uses the full multi-turn ``messages`` array from agent_messages.json when
    available. Falls back to prompt+response for older agent_messages.json
    files that lack the ``messages`` key.
    """
    messages = agent_data.get("messages")
    if not isinstance(messages, list) or not messages:
        messages = []
        prompt = agent_data.get("prompt", "")
        response = agent_data.get("response", "")
        if prompt:
            messages.append({"role": "user", "content": str(prompt)})
        if response:
            messages.append({"role": "assistant", "content": response})
    else:
        messages = [_clean_message(m) for m in messages if isinstance(m, dict)]
    (GRADER_DIR / "llm_messages.json").write_text(
        json.dumps(messages, ensure_ascii=False),
        encoding="utf-8",
    )


def main() -> None:
    agent_data = _read_agent_messages()
    dispatches = _read_dispatches()

    write_tool_trajectory(dispatches)
    write_llm_messages(agent_data)

    msg_count = len(agent_data.get("messages") or [])
    if msg_count == 0 and (agent_data.get("prompt") or agent_data.get("response")):
        msg_count = 1 + (1 if agent_data.get("response") else 0)
    print(f"[gen_trace] wrote tool_trajectory.json ({len(dispatches)} calls)")
    print(f"[gen_trace] wrote llm_messages.json ({msg_count} messages)")


if __name__ == "__main__":
    main()
