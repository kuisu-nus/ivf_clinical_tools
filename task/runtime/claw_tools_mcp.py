"""Task-local MCP tool dispatcher for Claw tasks.

The server intentionally has no third-party dependencies. It implements the
small MCP stdio surface OpenClaw needs for task-local tools:

* ``tools/list`` exposes each ``*.json`` spec from OMNI_TOOL_SPECS_DIR.
* ``tools/call`` dispatches the call to the spec's HTTP backend.
* every call is appended as a claw-eval-like ``tool_dispatch`` JSONL event.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
import traceback
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, BinaryIO, Optional


# MCP server name — must match MCP_SERVER_NAME in claw_to_dressage.py.
# Can be overridden via OMNI_MCP_SERVER_NAME env var.
SERVER_NAME = os.environ.get("OMNI_MCP_SERVER_NAME", "claw-tools")
SERVER_VERSION = "0.1.0"
DEFAULT_PROTOCOL_VERSION = "2024-11-05"


@dataclass(frozen=True)
class ToolBackend:
    type: str
    url: str
    method: str = "POST"
    service: Optional[str] = None


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    input_schema: dict[str, Any]
    backend: Optional[ToolBackend]


def main() -> None:
    args = parse_args()
    if args.list_tools:
        print(
            json.dumps(
                [tool_to_mcp(tool) for tool in load_tools()],
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    if args.call_tool:
        arguments = json.loads(args.arguments or "{}")
        result = call_tool(load_tools_by_name(), args.call_tool, arguments)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    if args.export_dispatches:
        export_dispatches(Path(args.export_dispatches))
        return
    if args.append_agents_md:
        append_agents_md()
        return
    serve_stdio()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Omni task-local MCP dispatcher.")
    parser.add_argument("--list-tools", action="store_true", help="Print MCP tool specs and exit.")
    parser.add_argument("--call-tool", help="Call one task tool directly and exit.")
    parser.add_argument("--arguments", default="{}", help="JSON object for --call-tool.")
    parser.add_argument(
        "--export-dispatches",
        help="Convert OMNI_TRACE_DIR/dispatches.jsonl into a JSON array at this path.",
    )
    parser.add_argument(
        "--append-agents-md",
        action="store_true",
        help="Append MCP server prefixed tool instructions to AGENTS.md in OpenClaw CWD.",
    )
    return parser.parse_args()


def serve_stdio() -> None:
    tools = load_tools_by_name()
    while True:
        item = read_stdio_message(sys.stdin.buffer)
        if item is None:
            return
        message, framing = item
        response = handle_jsonrpc(message, tools)
        if response is not None:
            write_stdio_message(sys.stdout.buffer, response, framing)


def handle_jsonrpc(message: dict[str, Any], tools: dict[str, ToolSpec]) -> Optional[dict[str, Any]]:
    request_id = message.get("id")
    method = message.get("method")
    params = message.get("params") or {}
    if request_id is None:
        return None
    try:
        if method == "initialize":
            result = {
                "protocolVersion": params.get("protocolVersion") or DEFAULT_PROTOCOL_VERSION,
                "capabilities": {"tools": {}},
                "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
            }
        elif method == "ping":
            result = {}
        elif method == "tools/list":
            result = {"tools": [tool_to_mcp(tool) for tool in tools.values()]}
        elif method == "tools/call":
            name = str(params.get("name") or "")
            arguments = params.get("arguments") or {}
            if not isinstance(arguments, dict):
                raise ValueError("tools/call arguments must be a JSON object")
            result = call_tool(tools, name, arguments)
        elif method == "resources/list":
            result = {"resources": []}
        elif method == "prompts/list":
            result = {"prompts": []}
        else:
            return jsonrpc_error(request_id, -32601, f"Method not found: {method}")
        return {"jsonrpc": "2.0", "id": request_id, "result": result}
    except Exception as exc:  # pragma: no cover - defensive server boundary
        print(traceback.format_exc(), file=sys.stderr, flush=True)
        return jsonrpc_error(request_id, -32000, str(exc))


def jsonrpc_error(request_id: Any, code: int, message: str) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}


def read_stdio_message(stream: BinaryIO) -> Optional[tuple[dict[str, Any], str]]:
    first = stream.readline()
    if first == b"":
        return None
    stripped = first.strip()
    if not stripped:
        return None
    if stripped.startswith(b"{"):
        return json.loads(stripped.decode("utf-8")), "jsonl"

    headers: dict[str, str] = {}
    line_text = first.decode("ascii", errors="replace").strip()
    key, _, value = line_text.partition(":")
    headers[key.lower()] = value.strip()
    while True:
        line = stream.readline()
        if line == b"":
            return None
        line_text = line.decode("ascii", errors="replace").strip()
        if not line_text:
            break
        key, _, value = line_text.partition(":")
        headers[key.lower()] = value.strip()
    length = int(headers.get("content-length") or "0")
    if length <= 0:
        return None
    body = stream.read(length)
    if not body:
        return None
    return json.loads(body.decode("utf-8")), "framed"


def write_stdio_message(stream: BinaryIO, payload: dict[str, Any], framing: str) -> None:
    body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    if framing == "jsonl":
        stream.write(body + b"\n")
    else:
        stream.write(f"Content-Length: {len(body)}\r\n\r\n".encode("ascii"))
        stream.write(body)
    stream.flush()


def read_framed_message(stream: BinaryIO) -> Optional[dict[str, Any]]:
    headers: dict[str, str] = {}
    while True:
        line = stream.readline()
        if line == b"":
            return None
        line_text = line.decode("ascii", errors="replace").strip()
        if not line_text:
            break
        key, _, value = line_text.partition(":")
        headers[key.lower()] = value.strip()
    length = int(headers.get("content-length") or "0")
    if length <= 0:
        return None
    body = stream.read(length)
    if not body:
        return None
    return json.loads(body.decode("utf-8"))


def write_framed_message(stream: BinaryIO, payload: dict[str, Any]) -> None:
    body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    stream.write(f"Content-Length: {len(body)}\r\n\r\n".encode("ascii"))
    stream.write(body)
    stream.flush()


def load_tools_by_name() -> dict[str, ToolSpec]:
    tools = load_tools()
    return {tool.name: tool for tool in tools}


def load_tools() -> list[ToolSpec]:
    """Load all tool specs from OMNI_TOOL_SPECS_DIR.

    Scans the directory for *.json files and parses each as a tool spec.
    This replaces the old task.json-based loading — task.json is no longer
    injected into the container to prevent reward hacking.
    """
    tool_specs_dir = Path(os.environ.get("OMNI_TOOL_SPECS_DIR", "/data/tools"))
    tools: list[ToolSpec] = []
    for spec_path in sorted(tool_specs_dir.glob("*.json")):
        raw = json.loads(spec_path.read_text(encoding="utf-8"))
        tools.append(parse_tool(raw))
    return tools


def parse_tool(raw: dict[str, Any]) -> ToolSpec:
    backend_raw = raw.get("backend")
    backend = None
    if isinstance(backend_raw, dict):
        backend = ToolBackend(
            type=str(backend_raw.get("type") or "http"),
            url=str(backend_raw.get("url") or ""),
            method=str(backend_raw.get("method") or "POST").upper(),
            service=str(backend_raw["service"]) if backend_raw.get("service") else None,
        )
    return ToolSpec(
        name=str(raw["name"]),
        description=str(raw.get("description") or ""),
        input_schema=dict(raw.get("input_schema") or raw.get("inputSchema") or {"type": "object"}),
        backend=backend,
    )


def tool_to_mcp(tool: ToolSpec) -> dict[str, Any]:
    return {
        "name": tool.name,
        "description": tool.description,
        "inputSchema": tool.input_schema,
    }


def append_agents_md() -> None:
    """Append MCP server prefixed tool instructions to AGENTS.md in OpenClaw CWD."""
    tools = load_tools()
    if not tools:
        print("append_agents_md: no tools found, skipping")
        return
    lines = [
        "",
        "# Claw Tool Bridge",
        "",
        f"This task provides tools through the OpenClaw MCP server `{SERVER_NAME}`.",
        f"Use the `{SERVER_NAME}__<tool_name>` prefixed names when calling tools.",
        "",
        "Available task tools:",
    ]
    for tool in tools:
        exposed = f"{SERVER_NAME}__{tool.name}"
        lines.append(f"- `{exposed}` (task tool `{tool.name}`): {tool.description}")
    lines += [
        "",
        "Do not call native web_search, web_fetch, exec, read, or OCR tools",
        f"when a {SERVER_NAME}__ tool is available for the same purpose.",
        "",
    ]
    block = "\n".join(lines) + "\n"
    pattern = "/workspace_sandbox/blackbox_server_runtime/bbs-*/home/.openclaw/workspace/AGENTS.md"
    targets = glob.glob(pattern)
    if not targets:
        print(f"append_agents_md: no AGENTS.md found at {pattern}")
        return
    for path in targets:
        with open(path, "a") as f:
            f.write(block)
        print(f"append_agents_md: appended {len(tools)} tool(s) to {path}")


def call_tool(tools: dict[str, ToolSpec], name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    tool = tools.get(name)
    if tool is None:
        body = {"error": f"unknown tool: {name}"}
        return tool_error_response(body)
    if tool.backend is None:
        body = {"error": f"tool has no backend: {name}"}
        write_dispatch(tool, arguments, "", 500, body, 0.0)
        return tool_error_response(body)
    if tool.backend.type != "http":
        body = {"error": f"unsupported backend type: {tool.backend.type}"}
        write_dispatch(tool, arguments, tool.backend.url, 500, body, 0.0)
        return tool_error_response(body)
    status, body, latency_ms = dispatch_http(tool.backend, arguments)
    write_dispatch(tool, arguments, tool.backend.url, status, body, latency_ms)
    return {
        "content": [{"type": "text", "text": json.dumps(body, ensure_ascii=False)}],
        "isError": status >= 400,
    }


def tool_error_response(body: dict[str, Any]) -> dict[str, Any]:
    return {
        "content": [{"type": "text", "text": json.dumps(body, ensure_ascii=False)}],
        "isError": True,
    }


def dispatch_http(backend: ToolBackend, arguments: dict[str, Any]) -> tuple[int, Any, float]:
    data = json.dumps(arguments, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        backend.url,
        data=data if backend.method != "GET" else None,
        method=backend.method,
        headers={"Content-Type": "application/json"},
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    start = time.monotonic()
    try:
        timeout = float(os.environ.get("OMNI_TOOL_HTTP_TIMEOUT", "30"))
        with opener.open(request, timeout=timeout) as response:
            raw = response.read()
            return response.status, decode_response_body(raw), (time.monotonic() - start) * 1000
    except urllib.error.HTTPError as exc:
        raw = exc.read()
        return exc.code, decode_response_body(raw), (time.monotonic() - start) * 1000
    except Exception as exc:
        return 500, {"error": str(exc)}, (time.monotonic() - start) * 1000


def decode_response_body(raw: bytes) -> Any:
    text = raw.decode("utf-8", errors="replace")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"text": text}


def write_dispatch(
    tool: ToolSpec,
    arguments: dict[str, Any],
    endpoint_url: str,
    status: int,
    response_body: Any,
    latency_ms: float,
) -> None:
    trace_dir = Path(os.environ.get("OMNI_TRACE_DIR", "/tmp/omni_task/trace"))
    trace_dir.mkdir(parents=True, exist_ok=True)
    trace_id = os.environ.get("OMNI_TRACE_ID") or "omni-" + uuid.uuid4().hex
    event = {
        "type": "tool_dispatch",
        "trace_id": trace_id,
        "tool_use_id": "mcp-" + uuid.uuid4().hex,
        "tool_name": tool.name,
        "endpoint_url": endpoint_url,
        "request_body": arguments,
        "response_status": status,
        "response_body": response_body,
        "latency_ms": latency_ms,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    with (trace_dir / "dispatches.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(event, ensure_ascii=False) + "\n")


def export_dispatches(target: Path) -> None:
    trace_dir = Path(os.environ.get("OMNI_TRACE_DIR", "/tmp/omni_task/trace"))
    source = trace_dir / "dispatches.jsonl"
    records = []
    if source.exists():
        for line in source.read_text(encoding="utf-8").splitlines():
            if line.strip():
                records.append(json.loads(line))
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
