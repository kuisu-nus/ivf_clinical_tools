#!/usr/bin/env python3
"""Patch claw_eval/llm_judge.py in the container for WAF safety, fallback parsing, and exception throwing.

This script is injected into the sandbox at /data/runtime/patch_llm_judge.py
and executed as an after_agent execute_cmd for tasks that require LLM judge.

Usage:
    python3 patch_llm_judge.py <grader_mount>

    grader_mount: path where grader files are located (e.g. /tmp/omni_grader)
                  Grader files are injected via ossutil in after_agent stage.
                  The script looks for claw_eval/src/claw_eval/graders/llm_judge.py
                  under this path.

Exit codes:
    0 on success (whether or not the file was patched)
    0 if the target file doesn't exist (prints {"patched": false, "reason": "missing"})
"""

import json
import sys
from pathlib import Path


def main() -> None:
    if len(sys.argv) < 2:
        print(json.dumps({"patched": False, "reason": "no grader_mount argument"}))
        raise SystemExit(0)

    grader_mount = Path(sys.argv[1])
    path = grader_mount / "claw_eval" / "src" / "claw_eval" / "graders" / "llm_judge.py"
    if not path.exists():
        print(json.dumps({"patched": False, "reason": "missing"}))
        raise SystemExit(0)

    text = path.read_text(encoding="utf-8")
    original = text

    # --- Insert WAF-safe helper ---
    helper_marker = "def _accio_waf_safe_judge_text("
    if helper_marker not in text:
        insert_marker = "\n\nclass LLMJudge:\n"
        helper = r'''

_ACCIO_WAF_SAFE_NOTE = (
    "Transport note: security payload examples below may be WAF-sanitized "
    "with harmless spaces, e.g. '$ {' for '${', 'jndi :' for 'jndi:', "
    "and 'ldap : //' for 'ldap://'. Treat them as the original security "
    "strings when grading."
)


def _accio_waf_safe_judge_text(value: str) -> str:
    # Avoid ingress WAF false positives on Log4Shell examples.
    replacements = (
        ("${", "$ {"),
        ("jndi:", "jndi :"),
        ("JNDI:", "JNDI :"),
        ("ldap://", "ldap : //"),
        ("ldaps://", "ldaps : //"),
        ("rmi://", "rmi : //"),
        ("dns://", "dns : //"),
    )
    for old, new in replacements:
        value = value.replace(old, new)
    return value
'''
        if insert_marker in text:
            text = text.replace(insert_marker, helper + insert_marker, 1)

    # --- Fix broken fallback parsing ---
    marker = "\n    def evaluate_visual(\n"
    if marker in text:
        evaluate_text, visual_suffix = text.split(marker, 1)
        visual_text = marker + visual_suffix
    else:
        evaluate_text, visual_text = text, ""

    broken_fallback = '''                        parsed = {
                            "score": float(score_m.group(1)),
                            "reasoning": reason_m.group(1) if reason_m else "",
                        }
'''
    fixed_fallback = '''                        score = float(score_m.group(1))
                        reasoning = reason_m.group(1) if reason_m else ""
'''
    if broken_fallback in evaluate_text:
        evaluate_text = evaluate_text.replace(broken_fallback, fixed_fallback, 1)

    # --- Apply WAF-safe judge text ---
    judge_user_msg = '''        user_msg = (
            f"## Task Prompt\\n{task_prompt}\\n\\n"
            f"## Conversation\\n{conversation}\\n\\n"
            f"## Actions Taken\\n{actions_summary}\\n\\n"
            f"## Rubric\\n{rubric}"
        )
'''
    safe_judge_user_msg = judge_user_msg + '''        user_msg = _ACCIO_WAF_SAFE_NOTE + "\\n\\n" + _accio_waf_safe_judge_text(user_msg)
'''
    if "_accio_waf_safe_judge_text(user_msg)" not in evaluate_text and judge_user_msg in evaluate_text:
        evaluate_text = evaluate_text.replace(judge_user_msg, safe_judge_user_msg, 1)

    # --- Add exception throwing after retries (evaluate) ---
    if "LLM judge failed after" not in evaluate_text:
        sleep_line = "                time.sleep(delay)\n"
        pos = evaluate_text.rfind(sleep_line)
        if pos != -1:
            pos += len(sleep_line)
            evaluate_text = (
                evaluate_text[:pos]
                + '        raise RuntimeError(\n'
                + '            f"LLM judge failed after {max_retries + 1} attempts: {last_exc}"\n'
                + '        )\n'
                + evaluate_text[pos:]
            )

    # --- Add exception throwing after retries (visual) ---
    if visual_text and "LLM visual judge failed after" not in visual_text:
        sleep_line = "                time.sleep(delay)\n"
        pos = visual_text.rfind(sleep_line)
        if pos != -1:
            pos += len(sleep_line)
            visual_text = (
                visual_text[:pos]
                + '        raise RuntimeError(\n'
                + '            f"LLM visual judge failed after {max_retries + 1} attempts: {last_exc}"\n'
                + '        )\n'
                + visual_text[pos:]
            )

    text = evaluate_text + visual_text
    if text != original:
        path.write_text(text, encoding="utf-8")
    print(json.dumps({"patched": text != original, "path": str(path)}))


if __name__ == "__main__":
    main()
