"""Deterministic grader for the staged IVF outcome task (omni_grader.v2 entrypoint ``grade``).

The agent analyses a case in three stages (初诊 / 促排 / 胚胎) and ends with one line
``结论：妊娠 | 未妊娠 | 不确定``. Per-case facts come from ``metadata.json`` in the grader
mount, copied in only after the agent has finished (label and the true tool inputs).

Items (weights sum to 1; no LLM judge, so rewards are fast and reproducible):
  outcome       0.60  class-balanced outcome score (table below)
  tool_use      0.15  stage-1 prediction (age, AMH), stage-2 prediction (with Gn days) and one
                      similar-patient query, each with the case's own values
  faithfulness  0.10  every percentage and case count in the answer comes from a tool result
                      (or the 39% overall rate given in the system prompt)
  format        0.05  【初诊】【促排】【胚胎】 in order, and the conclusion as the last line
  concise       0.10  full score up to CONCISE_FULL characters, zero at CONCISE_ZERO
  gate          treatment / medication advice zeroes the task

Outcome score: correct and wrong answers are weighted by the inverse class frequency (p = 444/1132
pregnant in the training set) and mapped to [0, 1], so that always answering 妊娠, always 未妊娠, or
always 不确定 each earn an expected 0.5; only answers that separate the cases score above 0.5.

               answer: 妊娠   未妊娠   不确定
  truth 妊娠          1.000   0.000    0.5
  truth 未妊娠        0.177   0.823    0.5
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

PREVALENCE = 444 / 1132
WEIGHTS = {"outcome": 0.60, "tool_use": 0.15, "faithfulness": 0.10, "format": 0.05, "concise": 0.10}
CONCISE_FULL, CONCISE_ZERO = 300, 600          # characters of the final answer, whitespace excluded
STAGES = ("【初诊】", "【促排】", "【胚胎】")
ALLOWED_RATES = (39.0, 39.2, 95.0)              # overall rate (system prompt / similar tool); 95% interval level
CONCLUSION = re.compile(r"^\s*结论\s*[:：]\s*(未妊娠|不确定|妊娠)\s*[。.]?\s*$", re.MULTILINE)
PERCENT = re.compile(r"(\d+(?:\.\d+)?)\s*%")
COUNT = re.compile(r"(\d+)\s*例")
ADVICE = re.compile(r"(建议|推荐|应当|应该)[^。；;\n]{0,12}?(用药|加用|改用|服用|换用|剂量|更换方案|调整方案|"
                    r"推迟移植|取消移植|全胚冷冻|冻存胚胎|冷冻胚胎)")


def outcome_score(answer: str, truth: str) -> float:
    w_pos, w_neg = 1 / (2 * PREVALENCE), 1 / (2 * (1 - PREVALENCE))
    if answer == "不确定":
        return 0.5
    weight = w_pos if truth == "妊娠" else w_neg
    sign = 1 if answer == truth else -1
    return 0.5 + sign * weight / (2 * max(w_pos, w_neg))


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
def _tool_use(calls: list[dict[str, Any]], inputs: dict[str, Any]) -> dict[str, Any]:
    age, amh, gn = inputs.get("age"), inputs.get("amh"), inputs.get("gn_days")
    ok = [c for c in calls if c.get("status") == "ok"]
    predict = [c.get("arguments") or {} for c in ok if c.get("tool_name") == "ivf_predict_outcome"]
    similar = [c.get("arguments") or {} for c in ok if c.get("tool_name") == "ivf_similar_patients"]
    checks = {"stage1_predict_age_amh": any(_close(a.get("age"), age) and _close(a.get("amh"), amh)
                                            and a.get("gn_days") is None for a in predict)}
    if not (inputs and gn is None):                  # stage 2 needs the case's Gn days
        checks["stage2_predict_with_gn"] = any(_close(a.get("age"), age) and a.get("gn_days") is not None
                                               and _close(a.get("gn_days"), gn) for a in predict)
    if not (inputs and amh is None):                 # the similar tool needs AMH
        checks["similar_patients"] = any(_close(a.get("age"), age) and _close(a.get("amh"), amh) for a in similar)
    passed = sum(checks.values())
    return score_item("tool_use", round(passed / len(checks), 4), weight=WEIGHTS["tool_use"],
                      reason=f"{passed}/{len(checks)} required tool calls with the case's values.",
                      metadata={"checks": checks, "calls": len(calls)})


def _faithfulness(answer: str, calls: list[dict[str, Any]]) -> dict[str, Any]:
    returned = [n for c in calls for n in _numbers(_response(c))]
    rates = set(ALLOWED_RATES) | {round(n * 100, 1) for n in returned if 0 <= n <= 1}
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
                      reason=f"stages in order: {stages_ok}; conclusion is the last line: {has_conclusion_last}")


def _concise(answer: str) -> dict[str, Any]:
    n = len(re.sub(r"\s", "", answer))
    score = 1.0 if n <= CONCISE_FULL else max(0.0, (CONCISE_ZERO - n) / (CONCISE_ZERO - CONCISE_FULL))
    return score_item("concise", round(score, 4), weight=WEIGHTS["concise"],
                      reason=f"{n} characters (full score up to {CONCISE_FULL}, zero at {CONCISE_ZERO}).",
                      metadata={"characters": n})


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
    items.append(_concise(answer))
    return items
