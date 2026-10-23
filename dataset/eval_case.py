"""Score one finished case of a local evaluation (called by run_eval.sh).

  python dataset/eval_case.py --case-dir <samples>/cases/<id> --result-dir <eval>/cases/<id>

Reads the agent's stream (Claude Code --output-format stream-json) and the MCP bridge's
dispatches.jsonl, writes llm_messages.json and tool_trajectory.json in the grader's format,
runs the task grader, and writes score.json (grader output plus run statistics).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TOOL_PREFIXES = ("mcp__claw-tools__", "claw-tools__")


def _strip(name: str) -> str:
    for prefix in TOOL_PREFIXES:
        if name.startswith(prefix):
            return name[len(prefix):]
    return name


def read_stream(path: Path) -> tuple[str, dict]:
    """Final answer text and run statistics from a stream-json transcript."""
    final, stats = "", {}
    if not path.exists():
        return final, {"error": "no stream output"}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if event.get("type") == "result":
            final = event.get("result") or ""
            usage = event.get("usage") or {}
            stats = {"is_error": event.get("is_error"), "num_turns": event.get("num_turns"),
                     "duration_ms": event.get("duration_ms"), "output_tokens": usage.get("output_tokens"),
                     "input_tokens": usage.get("input_tokens")}
    return final, stats


def read_trajectory(path: Path) -> dict:
    calls = []
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            d = json.loads(line)
            status = d.get("response_status", 0)
            calls.append({"call_index": len(calls), "tool_name": _strip(str(d.get("tool_name", ""))),
                          "arguments": d.get("request_body", {}),
                          "status": "ok" if isinstance(status, int) and status < 400 else "error",
                          "source": "mcp_dispatch", "response": d.get("response_body", "")})
    return {"schema_version": "omni_tool_trajectory.v1",
            "source": {"runner": "claude-code", "adapter": "mcp"}, "calls": calls}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--case-dir", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    res = args.result_dir

    final, stats = read_stream(res / "stream.jsonl")
    prompt = (args.case_dir / "prompt.md").read_text(encoding="utf-8")
    (res / "llm_messages.json").write_text(json.dumps(
        [{"role": "user", "content": prompt}, {"role": "assistant", "content": final}], ensure_ascii=False),
        encoding="utf-8")
    (res / "tool_trajectory.json").write_text(json.dumps(
        read_trajectory(res / "trace" / "dispatches.jsonl"), ensure_ascii=False), encoding="utf-8")
    shutil.copy(args.case_dir / "metadata.json", res / "metadata.json")

    env = {**os.environ, "OMNI_GRADER_MOUNT_PATH": str(res), "OMNI_GRADER_METADATA_PATH": str(res / "metadata.json"),
           "OMNI_TOOL_TRAJECTORY_PATH": str(res / "tool_trajectory.json"),
           "OMNI_LLM_MESSAGES_PATH": str(res / "llm_messages.json"), "OMNI_WORKSPACE_PATH": str(res)}
    out = subprocess.run([sys.executable, "omni_grader_runner.py"], cwd=REPO / "grader", env=env,
                         capture_output=True, text=True, encoding="utf-8")
    if out.returncode != 0:
        raise SystemExit(f"grader failed for {args.case_dir.name}: {out.stderr[-2000:]}")
    graded = json.loads(out.stdout)
    (res / "answer.txt").write_text(final, encoding="utf-8")
    (res / "score.json").write_text(json.dumps({"case_id": args.case_dir.name, "run": stats, "grader": graded},
                                               ensure_ascii=False, indent=2), encoding="utf-8")
    items = {i["name"]: i for i in graded["raw_scores"]["items"]}
    print(f"{args.case_dir.name}: reward={graded['overall_score']:.3f} "
          f"answer={items.get('outcome', {}).get('metadata', {}).get('predicted')} "
          f"truth={items.get('outcome', {}).get('metadata', {}).get('truth')}")


if __name__ == "__main__":
    main()
