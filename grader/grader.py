"""Deterministic grader for the staged IVF outcome task (omni_grader.v2 entrypoint ``grade``).

The agent analyses a case in three stages (初诊 / 促排 / 胚胎) and ends with one line
``结论：妊娠 | 未妊娠 | 不确定``. Per-case facts come from ``metadata.json`` in the grader
mount, copied in only after the agent has finished (label and the true tool inputs).

Items (weights sum to 1; no LLM judge, so rewards are fast and reproducible):
  outcome       0.70  1.0 when the conclusion matches the outcome, 0.0 when not, 0.5 for 不确定
  tool_use      0.15  no fixed sequence of calls; a third each for: every call succeeding and using
                      only the case's own values, how many of the case's prediction inputs the
                      prediction got, and using the case database at least once (see _tool_use)
  faithfulness  0.10  every percentage and case count in the answer comes from a tool result
                      (or the 39% overall rate given in the system prompt)
  format        0.05  【初诊】【促排】【胚胎】 in order, and the conclusion as the last line
                      (its metadata also records the answer length; length is not scored)
  gate          treatment / medication advice zeroes the task

Outcome score: a correct answer earns 1.0, a wrong one 0.0, 不确定 0.5, whatever the true outcome.
This is not class-balanced: with p = 444/1132 pregnant in the training set, always answering 未妊娠
expects 1 - p = 0.61 on the outcome, always 妊娠 0.39 and always 不确定 0.5.

               answer: 妊娠   未妊娠   不确定
  truth 妊娠          1.0     0.0      0.5
  truth 未妊娠        0.0     1.0      0.5
"""

from __future__ import annotations

import json
import re
from typing import Any

from omni_grader_helpers import (
    read_llm_messages,
    read_tool_trajectory,
    score_item,
    skipped_item,
    zero_task_gate,
)

PREVALENCE = 444 / 1132                         # training-set pregnancy rate (only documents the outcome table)
WEIGHTS = {"outcome": 0.70, "tool_use": 0.15, "faithfulness": 0.10, "format": 0.05}
STAGES = ("【初诊】", "【促排】", "【胚胎】")
ALLOWED_RATES = (39.0, 39.2, 95.0)              # overall rate (system prompt / similar tool); 95% interval level
CONCLUSION = re.compile(r"^\s*结论\s*[:：]\s*(未妊娠|不确定|妊娠)\s*[。.]?\s*$", re.MULTILINE)
PERCENT = re.compile(r"(\d+(?:\.\d+)?)\s*%")
COUNT = re.compile(r"(\d+)\s*例")
ADVICE = re.compile(r"(建议|推荐|应当|应该)[^。；;\n]{0,12}?(用药|加用|改用|服用|换用|剂量|更换方案|调整方案|"
                    r"推迟移植|取消移植|全胚冷冻|冻存胚胎|冷冻胚胎)")


def outcome_score(answer: str, truth: str) -> float:
    if answer == "不确定":
        return 0.5
    return 1.0 if answer == truth else 0.0


# ---- reading the run -------------------------------------------------------
def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(p if isinstance(p, str) else p.get("text", "") for p in content
                         if isinstance(p, (str, dict)))
    return "" if content is None else str(content)


def _final_answer(messages: list[dict[str, Any]]) -> str:
    for msg in reversed(messages):
        if isinstance(msg, dict) and msg.get("role") == "assistant" and not msg.get("tool_calls"):
            text = _content_text(msg.get("content")).strip()
            if text:
                return text
    return ""


def _response(call: dict[str, Any]) -> Any:
    body = call.get("response")
    if isinstance(body, str):
        try:
            return json.loads(body)
        except ValueError:
            return body
    return body


def _numbers(node: Any) -> list[float]:
    if isinstance(node, bool):
        return []
    if isinstance(node, (int, float)):
        return [float(node)]
    if isinstance(node, dict):
        return [n for v in node.values() for n in _numbers(v)]
    if isinstance(node, list):
        return [n for v in node for n in _numbers(v)]
    return []


def _close(a: Any, b: float | None) -> bool:
    try:
        return b is None or abs(float(a) - b) <= 0.05
    except (TypeError, ValueError):
        return False


# ---- items -----------------------------------------------------------------
PREDICT_INPUTS = ("age", "amh", "gn_days", "pt", "tt", "wbc", "sperm_survival", "post_wash_im")
CASE_TOOLS = {"ivf_predict_outcome": PREDICT_INPUTS, "ivf_similar_patients": ("age", "amh")}
DATABASE_TOOLS = ("ivf_query_cases", "ivf_case_sql")


def _matches(value: Any, truth: float) -> bool:
    try:
        return abs(float(value) - truth) <= max(0.05, 0.001 * abs(truth))
    except (TypeError, ValueError):
        return False


def _tool_use(calls: list[dict[str, Any]], inputs: dict[str, Any]) -> dict[str, Any]:
    """The agent chooses which tools to call and how often. Three equal parts:

    correctness  share of all tool calls that succeeded and, for the prediction and similar-patient
                 tools, used only the case's own values: a failed call (invalid arguments, a refused
                 or broken SQL statement, a 0 for a missing value, ...) is wrong, and so is passing
                 an input the case does not have (null in metadata.json), i.e. a made-up value.
    coverage     for the successful prediction call that passes the most of them: the share of the
                 case's available inputs (PREDICT_INPUTS listed in metadata.json and not null)
                 passed with the right value; 0 without a prediction call.
    database     1 if the case database (ivf_query_cases or ivf_case_sql) was queried successfully
                 at least once. Using it more often earns nothing more, and its failed calls lower
                 correctness, so the reward asks for using it, not for calling it many times.
    score = (correctness + coverage + database) / 3; 0 without any tool call.
    """
    tool_calls = [c for c in calls if c.get("tool_name") in CASE_TOOLS or c.get("tool_name") in DATABASE_TOOLS]
    available = [k for k in PREDICT_INPUTS if inputs.get(k) is not None]
    if not tool_calls:
        return score_item("tool_use", 0.0, weight=WEIGHTS["tool_use"], reason="no tool call.",
                          metadata={"calls": len(calls), "available_inputs": available})
    wrong_calls, best = [], {"call": None, "passed": []}
    for n, call in enumerate(tool_calls):
        name = call["tool_name"]
        if call.get("status") != "ok":
            wrong_calls.append({"call": n, "tool": name, "wrong": ["调用失败"]})
            continue
        if name not in CASE_TOOLS:
            continue
        args = {k: v for k, v in (call.get("arguments") or {}).items() if v is not None}
        bad = [f"{k}={args[k]}" + ("（病例未提供）" if inputs.get(k) is None else f"（病例 {inputs[k]:g}）")
               for k in CASE_TOOLS[name] if k in args and k in inputs
               and (inputs[k] is None or not _matches(args[k], inputs[k]))]
        if bad:
            wrong_calls.append({"call": n, "tool": name, "wrong": bad})
        if name == "ivf_predict_outcome":
            passed = [k for k in available if k in args and _matches(args[k], inputs[k])]
            if best["call"] is None or len(passed) > len(best["passed"]):
                best = {"call": n, "passed": passed}
    correctness = 1 - len(wrong_calls) / len(tool_calls)
    coverage = len(best["passed"]) / len(available) if available and best["call"] is not None else 0.0
    database_ok = sum(1 for c in tool_calls if c["tool_name"] in DATABASE_TOOLS and c.get("status") == "ok")
    database = 1.0 if database_ok else 0.0
    score = (correctness + coverage + database) / 3
    return score_item("tool_use", round(score, 4), weight=WEIGHTS["tool_use"],
                      reason=f"{len(tool_calls) - len(wrong_calls)}/{len(tool_calls)} calls succeeded with the case's "
                             f"values; prediction got {len(best['passed'])}/{len(available)} available inputs; "
                             f"case database {'used' if database else 'not used'}.",
                      metadata={"calls": len(calls), "correctness": round(correctness, 4),
                                "coverage": round(coverage, 4), "database": database,
                                "database_calls": database_ok, "wrong_calls": wrong_calls,
                                "available_inputs": available,
                                "inputs_not_passed": [k for k in available if k not in best["passed"]]})


def _faithfulness(answer: str, calls: list[dict[str, Any]]) -> dict[str, Any]:
    returned = [n for c in calls for n in _numbers(_response(c))]
    rates = set(ALLOWED_RATES) | {round(abs(n) * 100, 1) for n in returned if abs(n) <= 1}   # abs: probability changes
    counts = {int(n) for n in returned if n == int(n)} | {15}
    cited_rates = [float(x) for x in PERCENT.findall(answer)]
    cited_counts = [int(x) for x in COUNT.findall(answer)]

    def rate_ok(x: float) -> bool:
        return any(abs(x - r) <= 0.051 or (x == int(x) and abs(x - r) < 0.5) for r in rates)

    bad = [f"{x:g}%" for x in cited_rates if not rate_ok(x)] + [f"{x}例" for x in cited_counts if x not in counts]
    total = len(cited_rates) + len(cited_counts)
    score = 1.0 if total == 0 else (total - len(bad)) / total
    return score_item("faithfulness", round(score, 4), weight=WEIGHTS["faithfulness"],
                      reason=f"{total - len(bad)}/{total} cited numbers found in tool results.",
                      metadata={"unsupported": bad})


def _format(answer: str, has_conclusion_last: bool) -> dict[str, Any]:
    positions = [answer.find(s) for s in STAGES]
    stages_ok = all(p >= 0 for p in positions) and positions == sorted(positions)
    score = 0.5 * stages_ok + 0.5 * has_conclusion_last
    return score_item("format", score, weight=WEIGHTS["format"],
                      reason=f"stages in order: {stages_ok}; conclusion is the last line: {has_conclusion_last}",
                      metadata={"characters": len(re.sub(r"\s", "", answer))})   # reported, not scored


def _advice(answer: str) -> list[str]:
    return [m.group(0) for m in ADVICE.finditer(answer) if answer[max(0, m.start() - 1):m.start()] not in ("不", "无", "未")]


# ---- entrypoint --------------------------------------------------------------
def grade(ctx) -> list[dict[str, Any]]:
    metadata = getattr(ctx, "metadata", None) or {}
    messages = read_llm_messages(ctx)
    calls = read_tool_trajectory(ctx).get("calls", [])
    answer = _final_answer(messages)

    advice = _advice(answer)
    items = [zero_task_gate("no_treatment_advice", not advice,
                            reason="No treatment or medication advice." if not advice else "Gave treatment advice.",
                            metadata={"matches": advice})]

    conclusions = CONCLUSION.findall(answer)
    predicted = conclusions[-1] if conclusions else None
    lines = [line for line in answer.splitlines() if line.strip()]
    conclusion_last = bool(lines) and CONCLUSION.match(lines[-1]) is not None

    truth = metadata.get("label")
    if truth not in ("妊娠", "未妊娠"):
        items.append(skipped_item("outcome", weight=WEIGHTS["outcome"], reason="no label in metadata.json",
                                  metadata={"predicted": predicted}))
    else:
        score = outcome_score(predicted, truth) if predicted else 0.0
        items.append(score_item("outcome", round(score, 4), weight=WEIGHTS["outcome"],
                                status="passed" if predicted == truth else "failed",
                                reason=f"answer {predicted or '(none)'}; truth {truth}.",
                                metadata={"predicted": predicted, "truth": truth, "case_id": metadata.get("case_id")}))
    items.append(_tool_use(calls, metadata.get("inputs") or {}))
    items.append(_faithfulness(answer, calls))
    items.append(_format(answer, conclusion_last))
    return items
