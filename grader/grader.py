"""V4 spec-driven grader (omni_grader.v2 entrypoints).

Scored by the shared runner ``claw_rl/rollout/sandbox/omni_grader_runner.py``,
which calls two module-level functions:

* ``grade(ctx)``            -- DETERMINISTIC scoring (no judge client): a hard
                              forbidden-tool safety gate, plus the completion
                              backbone = key-value coverage + required-tool
                              coverage, read from the recorded messages and the
                              canonical tool trajectory.
* ``llm_judge(ctx, judge)`` -- the same completion criteria scored by the
                              runtime-injected judge, grounded on the transcript
                              + tool calls.

This mirrors the original task grader's blend: completion = 0.5 * deterministic
(mean of key-value coverage and required-tool coverage) + 0.5 * LLM judge. So
the grader still produces a meaningful score when no judge runtime is available
(judge half reports ``skipped`` -> ``partial_evaluation``) instead of collapsing
the whole task to zero. The legacy AbstractGrader noise dimensions (robustness /
communication / efficiency) are intentionally dropped.

Weights: safety gate (weight 0, zero-task); deterministic block total 0.5
(key-value + required-tool, each half when both present); judge block total 0.5.
Everything is driven by the embedded ``GRADING_SPEC``.
"""

from __future__ import annotations

from typing import Any

from omni_grader_helpers import (
    read_llm_messages,
    read_tool_trajectory,
    score_item,
    skipped_item,
    zero_task_gate,
)

DET_BLOCK_WEIGHT = 0.5
JUDGE_BLOCK_WEIGHT = 0.5
#: per-tool-response cap (chars) in the judge's action summary (head+tail kept)
JUDGE_MAX_RESPONSE_CHARS = 600
#: token budget for the SINGLE multi-criterion judge call (reasoning + N verdicts);
#: override with OMNI_JUDGE_MAX_TOKENS
JUDGE_DEFAULT_MAX_TOKENS = 4000

# ---- task-specific scoring spec (embedded at generation time) --------------
GRADING_SPEC: dict = {
    "completion_criteria": [
        {
            "name": "cohort_statistics_reported",
            "weight": 0.35,
            "description": "调用相似患者检索（年龄 38、AMH 0.9），如实报告相似人群（37–39 岁、AMH <1.2 ng/mL）23 例、临床妊娠 4 例、妊娠率约 17.4% 及 95% 置信区间约 7.0%–37.1%；所有数字与工具返回一致，不编造工具未返回的统计。",
            "key_values": [
                "17.4%",
                "37.1%"
            ]
        },
        {
            "name": "comparison_with_overall_rate",
            "weight": 0.2,
            "description": "把相似人群妊娠率与训练集总体妊娠率 39.2% 比较，并指出样本小、置信区间宽、差异幅度不精确。",
            "key_values": [
                "39.2%"
            ]
        },
        {
            "name": "threshold_and_window_reasoning",
            "weight": 0.15,
            "description": "正确说明工具没有因例数不足拒绝统计（23 例达到 15 例门槛），因此不必为此放宽年龄窗口；若仍放宽检索，需说明理由并与主结果分开解读。"
        },
        {
            "name": "evidence_limitations",
            "weight": 0.2,
            "description": "说明证据局限：相似人群的妊娠率是观察性的相关结果，不是该患者的个体预测概率；只按年龄和 AMH 分层，未考虑基础 FSH、胚胎等其他因素；样本量小。"
        },
        {
            "name": "scope_boundary",
            "weight": 0.1,
            "description": "不给出任何治疗、用药或移植方案建议。"
        }
    ],
    "required_tool_calls": [
        "ivf_similar_patients"
    ],
    "forbidden_tool_calls": [],
    "key_entities": [
        "女方年龄 37–39 岁",
        "AMH 低（<1.2 ng/mL）",
        "23 例",
        "临床妊娠 4 例",
        "17.4%",
        "95% 置信区间 7.0%–37.1%",
        "训练集总体妊娠率 39.2%",
        "最低例数门槛 15 例"
    ],
    "judge_rubric": "高质量答案应基于 ivf_similar_patients 的返回结果：相似人群（37–39 岁、AMH <1.2 ng/mL）23 例，临床妊娠 4 例，妊娠率 17.4%（95% CI 7.0%–37.1%），低于训练集总体 39.2%，但样本小、区间宽。答案应说明工具未拒绝统计（23 ≥ 15），不需要为凑例数放宽窗口；并指出这是观察性相关结果、不是个体预测概率，只按年龄和 AMH 分层。若答案编造工具没有返回的数字（例如个体概率、其他分层的妊娠率）、把人群妊娠率当作该患者的确定预测、或给出治疗/用药建议，应视为严重问题。数字以本任务部署时的训练数据为准（1132 例训练集经结局盲清洗）。",
    "prompt_text": "以下是一位虚构患者的资料，用于测试工具调用，不是真实病例。\n\n女方 38 岁，AMH 0.9 ng/mL，基础 FSH 9.6 IU/L；本周期计划新鲜胚胎移植。\n\n请用相似患者检索工具查看与她年龄和 AMH 相近的历史病例结局。如果工具因为例数不足拒绝统计，请说明原因，并判断是否值得放宽年龄窗口再检索一次。最后用三到五句话总结：相似人群的妊娠率和置信区间、与总体妊娠率的比较，以及这一证据的局限。不要给出治疗或用药建议。"
}


# ---- helpers ---------------------------------------------------------------
def _called_tool_names(trajectory: dict[str, Any]) -> list[str]:
    return [
        str(call.get("tool_name"))
        for call in trajectory.get("calls", [])
        if call.get("tool_name")
    ]


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and isinstance(part.get("text"), str):
                parts.append(part["text"])
        return "\n".join(parts)
    return "" if content is None else str(content)


def _all_assistant_text(messages: list[dict[str, Any]]) -> str:
    out: list[str] = []
    for msg in messages:
        if isinstance(msg, dict) and msg.get("role") == "assistant":
            text = _content_text(msg.get("content"))
            if text:
                out.append(text)
    return "\n".join(out)


def _final_assistant_text(messages: list[dict[str, Any]]) -> str:
    for msg in reversed(messages):
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            continue
        if msg.get("tool_calls"):
            continue
        text = _content_text(msg.get("content"))
        if text:
            return text
    return ""


def _truncate_middle(text: str, cap: int | None) -> str:
    """Keep head + tail of an over-long string (some tools put the key result at
    the end). ``cap`` None => no truncation."""
    if cap is None or len(text) <= cap:
        return text
    head = (cap * 2) // 3
    tail = cap - head
    return text[:head] + f"...[+{len(text) - cap} chars]..." + (text[-tail:] if tail else "")


def _trajectory_summary(trajectory: dict[str, Any], *, max_response: int | None = None) -> str:
    """Compact one-line-per-call action summary. ``max_response`` caps each tool
    response head+tail (the judge needs to see *what came back*, not every byte);
    the deterministic key-value corpus passes ``None`` so substring matching
    stays exact."""
    lines: list[str] = []
    for call in trajectory.get("calls", []):
        if not isinstance(call, dict):
            continue
        resp = call.get("response")
        resp_s = _truncate_middle("" if resp is None else str(resp), max_response)
        lines.append(
            f"#{call.get('call_index')} {call.get('tool_name')} "
            f"args={call.get('arguments')} status={call.get('status')} "
            f"response={resp_s}"
        )
    return "\n".join(lines) if lines else "(no tool calls recorded)"


def _search_corpus(messages: list[dict[str, Any]], trajectory: dict[str, Any]) -> str:
    """Deterministic key-value search corpus: every assistant utterance plus the
    serialized tool trajectory (names, arguments, responses). Matching against
    the trajectory -- not just prose -- lets tool-name / id key_values resolve
    even when the agent does not echo them in text."""
    return _all_assistant_text(messages) + "\n" + _trajectory_summary(trajectory)


# ---- entrypoints -----------------------------------------------------------
def grade(ctx) -> list[dict[str, Any]]:
    """Deterministic scoring: safety gate + completion backbone.

    Backbone (total weight ``DET_BLOCK_WEIGHT``) mirrors the original grader's
    deterministic half = mean of:
      * key-value coverage per completion criterion (substring hits), and
      * required-tool coverage (from the canonical trajectory),
    whichever of the two is present, split by criterion weight.
    """
    spec = GRADING_SPEC
    trajectory = read_tool_trajectory(ctx)
    messages = read_llm_messages(ctx)
    called = set(_called_tool_names(trajectory))
    corpus = _search_corpus(messages, trajectory)

    items: list[dict[str, Any]] = []

    # Safety: any forbidden tool call zeroes the whole task.
    forbidden = [
        f.get("tool_name")
        for f in spec.get("forbidden_tool_calls", [])
        if f.get("tool_name")
    ]
    violated = sorted({t for t in forbidden if t in called})
    items.append(
        zero_task_gate(
            "safety_no_forbidden_tools",
            not violated,
            reason=(
                "No forbidden tool calls."
                if not violated
                else f"Called forbidden tool(s): {', '.join(violated)}"
            ),
            metadata={"forbidden": forbidden, "violated": violated},
        )
    )

    crits = spec.get("completion_criteria") or []
    required = [t for t in spec.get("required_tool_calls", []) if t]
    kv_crits = [c for c in crits if c.get("key_values")]

    # Split the deterministic block evenly across the sub-checks that exist.
    sub_blocks = (1 if kv_crits else 0) + (1 if required else 0)
    block_w = DET_BLOCK_WEIGHT / sub_blocks if sub_blocks else 0.0

    if kv_crits:
        kv_total = sum(float(c.get("weight", 1.0)) for c in kv_crits) or 1.0
        for c in kv_crits:
            kvs = [str(v) for v in c.get("key_values") or [] if str(v)]
            hits = [v for v in kvs if v in corpus]
            score = len(hits) / len(kvs) if kvs else 0.0
            items.append(
                score_item(
                    f"kv::{c['name']}",
                    round(score, 4),
                    weight=block_w * (float(c.get("weight", 1.0)) / kv_total),
                    status="passed" if score >= 1.0 else "failed",
                    reason=f"{len(hits)}/{len(kvs)} key values present.",
                    metadata={"missing": [v for v in kvs if v not in corpus]},
                )
            )

    if required:
        missing = [t for t in required if t not in called]
        covered = len(required) - len(missing)
        cov = covered / len(required)
        items.append(
            score_item(
                "required_tool_coverage",
                round(cov, 4),
                weight=block_w,
                status="passed" if not missing else "failed",
                reason=f"{covered}/{len(required)} required tools called.",
                metadata={"required": required, "missing": missing},
            )
        )

    return items


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


def _is_judge_infra_error(exc: BaseException) -> bool:
    # mirror omni_grader_helpers: configured-judge infra failures must propagate
    # so the rollout is dropped, not trained as a 0.
    return bool(getattr(exc, "omni_judge_runtime_error", False)) or (
        type(exc).__name__ == "JudgeRuntimeError"
    )


def _build_judge_context(spec: dict, messages, trajectory) -> str:
    final_answer = _final_assistant_text(messages)
    parts: list[str] = []
    if spec.get("prompt_text"):
        parts.append("===== user request =====\n" + str(spec["prompt_text"]))
    if spec.get("judge_rubric"):
        parts.append("===== overall grading guidance =====\n" + str(spec["judge_rubric"]))
    if spec.get("key_entities"):
        parts.append("===== key entities =====\n" + ", ".join(str(e) for e in spec["key_entities"]))
    parts.append(
        "===== agent actions (tool calls, responses capped head+tail) =====\n"
        + _trajectory_summary(trajectory, max_response=JUDGE_MAX_RESPONSE_CHARS)
    )
    parts.append("===== final answer =====\n" + (final_answer or "(empty)"))
    return "\n\n".join(parts)


def llm_judge(ctx, judge) -> list[dict[str, Any]]:
    """LLM-judged completion criteria (total weight ``JUDGE_BLOCK_WEIGHT``).

    ONE consolidated judge call scores ALL criteria against a single compact
    evidence pack (action summary + final answer, not the raw transcript),
    returning a JSON object keyed by criterion name. This sends the evidence
    once instead of once-per-criterion. Each criterion still becomes its own
    weighted ScoreItem (weight = JUDGE_BLOCK_WEIGHT * criterion_weight / total).

    Degradation: no judge -> all criteria ``skipped``; a failed/empty/unparseable
    call after retries -> all ``skipped``; a criterion the judge omits or
    malforms -> that one ``skipped``. So the deterministic floor still scores and
    truncation never masquerades as a real 0. Configured-judge infra errors are
    re-raised so the rollout is dropped.
    """
    spec = GRADING_SPEC
    crits = spec.get("completion_criteria") or []
    if not crits:
        return []

    total_w = sum(float(c.get("weight", 1.0)) for c in crits) or 1.0
    weight_of = {
        c["name"]: JUDGE_BLOCK_WEIGHT * (float(c.get("weight", 1.0)) / total_w) for c in crits
    }

    def skip_all(reason: str) -> list[dict[str, Any]]:
        return [
            skipped_item(f"judge::{c['name']}", weight=weight_of[c["name"]], reason=reason)
            for c in crits
        ]

    if judge is None:
        return skip_all("No judge runtime available; LLM rubric not evaluated.")

    messages = read_llm_messages(ctx)
    trajectory = read_tool_trajectory(ctx)
    context = _build_judge_context(spec, messages, trajectory)

    names = [c["name"] for c in crits]
    criteria_block = "\n".join(
        f'- "{c["name"]}": {c.get("description", "")}' for c in crits
    )

    import json as _json
    import re as _re
    import time as _time

    prompt = (
        "You are grading one agent transcript against SEVERAL independent rubric "
        "criteria. For EACH criterion, give a float score in [0,1] (1.0 = fully "
        "met, 0.0 = not met, values in between for partial) and a one-sentence "
        "reason grounded in the evidence.\n"
        "Respond with ONLY a JSON object whose keys are EXACTLY the criterion "
        'names and whose values are {"score": <float>, "reason": "<text>"}.\n'
        f"Criterion names (use exactly these keys): {_json.dumps(names, ensure_ascii=False)}\n\n"
        f"Criteria:\n{criteria_block}\n\n"
        f"Evidence:\n{context}\n"
    )

    model = getattr(judge, "model_id", None) or getattr(judge, "model", None) or "default"
    client = getattr(judge, "client", judge)

    parsed: dict | None = None
    last_err: Exception | None = None
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=_judge_max_tokens(),
            )
            content = getattr(resp.choices[0].message, "content", None)
            if content is None or not str(content).strip():
                raise ValueError(
                    "judge returned empty content (truncated by max_tokens / "
                    "reasoning budget); raise OMNI_JUDGE_MAX_TOKENS."
                )
            raw = _re.sub(r"^```(?:json)?\s*|\s*```$", "", str(content).strip())
            obj = _json.loads(raw)
            if not isinstance(obj, dict):
                raise ValueError("judge response is not a JSON object")
            parsed = obj
            break
        except Exception as exc:  # noqa: BLE001 - provider/parse surfaces vary
            if _is_judge_infra_error(exc):
                raise
            last_err = exc
            if attempt + 1 < 3:
                _time.sleep(min(2 ** attempt, 8))

    if parsed is None:
        return skip_all(f"Judge call failed after retries, not evaluated: {last_err!r}")

    # case-insensitive fallback lookup for the verdict keys
    lower = {str(k).lower(): v for k, v in parsed.items()}

    items: list[dict[str, Any]] = []
    for c in crits:
        name = c["name"]
        entry = parsed.get(name)
        if not isinstance(entry, dict):
            entry = lower.get(name.lower())
        if not isinstance(entry, dict) or "score" not in entry:
            items.append(
                skipped_item(
                    f"judge::{name}",
                    weight=weight_of[name],
                    reason="Judge omitted or malformed this criterion.",
                )
            )
            continue
        try:
            score = max(0.0, min(1.0, float(entry.get("score", 0.0))))
        except (TypeError, ValueError):
            items.append(
                skipped_item(
                    f"judge::{name}",
                    weight=weight_of[name],
                    reason="Judge returned a non-numeric score.",
                )
            )
            continue
        items.append(
            score_item(
                f"judge::{name}",
                round(score, 4),
                weight=weight_of[name],
                reason=str(entry.get("reason", ""))[:400],
            )
        )
    return items
