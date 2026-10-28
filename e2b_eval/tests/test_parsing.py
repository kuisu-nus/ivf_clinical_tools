"""Answer/grader parsing and summary construction."""

from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from pathlib import Path

from e2b_eval.runner import (
    CaseOutcome,
    RunOptions,
    StageError,
    _build_summary,
    _count_dispatch_lines,
    _extract_answer,
    _extract_outcome,
    _mock_conclusion,
    _parse_grader_output,
    _parse_grader_text,
    _run_commands,
)


class ExtractAnswerTests(unittest.TestCase):
    def test_prefers_last_plain_assistant_output(self):
        payload = {
            "outputs": [
                {"role": "assistant", "content": "intermediate", "tool_calls": [{"id": "1"}]},
                {"role": "assistant", "content": "final answer"},
            ]
        }
        self.assertEqual(_extract_answer(payload), "final answer")

    def test_reads_history_fallback(self):
        payload = {
            "conversation_history": [
                {"role": "system", "content": "system prompt"},
                {"role": "user", "content": "case"},
                {"role": "assistant", "content": "answer from history"},
            ]
        }
        self.assertEqual(_extract_answer(payload), "answer from history")

    def test_reads_content_parts(self):
        payload = {"outputs": [{"role": "assistant", "content": [{"type": "text", "text": "part"}]}]}
        self.assertEqual(_extract_answer(payload), "part")

    def test_missing_answer_is_empty(self):
        self.assertEqual(_extract_answer({"outputs": []}), "")
        self.assertEqual(_extract_answer(None), "")
        self.assertEqual(_extract_answer({"outputs": [{"role": "assistant", "content": ""}]}), "")


class GraderParsingTests(unittest.TestCase):
    def test_plain_json(self):
        data = {"overall_score": 0.75, "raw_scores": {"items": []}}
        self.assertEqual(_parse_grader_text(json.dumps(data)), data)

    def test_noisy_stdout(self):
        data = {"overall_score": 0.5, "raw_scores": {"items": []}}
        text = "[gen_trace] wrote tool_trajectory.json\n" + json.dumps(data) + "\n"
        self.assertEqual(_parse_grader_text(text), data)

    def test_json_with_trailing_log(self):
        data = {"overall_score": 0.5}
        text = "prefix " + json.dumps(data) + " trailing log"
        self.assertEqual(_parse_grader_text(text), data)

    def test_invalid(self):
        self.assertIsNone(_parse_grader_text(""))
        self.assertIsNone(_parse_grader_text("not json"))
        self.assertIsNone(_parse_grader_text('{"other": 1}'))

    def test_selects_last_run_grader_result(self):
        results = [
            {"name": "generate_trace", "stdout": '{"overall_score": 0.1}'},
            {"name": "run_grader", "stdout": '{"overall_score": 0.9}'},
        ]
        self.assertEqual(_parse_grader_output(results)["overall_score"], 0.9)

    def test_outcome_metadata(self):
        grader = {
            "raw_scores": {
                "items": [
                    {"name": "tool_use", "metadata": {}},
                    {"name": "outcome", "metadata": {"predicted": "妊娠", "truth": "未妊娠"}},
                ]
            }
        }
        self.assertEqual(_extract_outcome(grader), ("妊娠", "未妊娠"))
        self.assertEqual(_extract_outcome({}), (None, None))

    def test_dispatch_line_count(self):
        self.assertEqual(_count_dispatch_lines('{"a":1}\n\n{"b":2}\n'), 2)
        self.assertEqual(_count_dispatch_lines(None), 0)


class CommandRunnerTests(unittest.TestCase):
    class FakeClient:
        async def execute_cmd(self, session_id, cmd, timeout=None):
            if "fail" in cmd:
                return {
                    "stdout": "partial",
                    "stderr": "boom",
                    "returncode": 1,
                    "timed_out": False,
                    "duration_seconds": 0.01,
                }
            return {
                "stdout": "ok",
                "stderr": "",
                "returncode": 0,
                "timed_out": False,
                "duration_seconds": 0.02,
            }

    def test_writes_results_and_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            case_dir = Path(tmp)
            results = asyncio.run(
                _run_commands(
                    self.FakeClient(),
                    "sid",
                    [
                        {"name": "a", "cmd": "echo ok", "required": True},
                        {"name": "b", "cmd": "echo ok", "required": False},
                    ],
                    case_dir,
                    "before_agent",
                )
            )
            self.assertEqual([r["returncode"] for r in results], [0, 0])
            self.assertTrue((case_dir / "before_agent_commands.json").is_file())
            log = (case_dir / "before_agent.log").read_text(encoding="utf-8")
            self.assertIn("## a", log)
            self.assertIn("## b", log)

    def test_required_failure_raises_after_writing(self):
        with tempfile.TemporaryDirectory() as tmp:
            case_dir = Path(tmp)
            with self.assertRaisesRegex(StageError, "fail"):
                asyncio.run(
                    _run_commands(
                        self.FakeClient(),
                        "sid",
                        [
                            {"name": "fail", "cmd": "echo fail", "required": True},
                            {"name": "never", "cmd": "echo ok", "required": True},
                        ],
                        case_dir,
                        "before_agent",
                    )
                )
            results = json.loads((case_dir / "before_agent_commands.json").read_text(encoding="utf-8"))
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0]["returncode"], 1)
            self.assertTrue((case_dir / "before_agent.log").is_file())


class SummaryTests(unittest.TestCase):
    def test_summary_metrics(self):
        outcomes = [
            CaseOutcome("a", "a", "ok", score=1.0, prediction="妊娠", truth="妊娠"),
            CaseOutcome("b", "b", "ok", score=0.5, prediction="未妊娠", truth="妊娠"),
            CaseOutcome("c", "c", "ok", score=0.7, prediction="不确定", truth="未妊娠"),
            CaseOutcome("d", "d", "failed", error="boom"),
            CaseOutcome("e", "e", "ok", score=0.6, prediction="妊娠", truth="未妊娠", skipped=True),
        ]
        summary = _build_summary(
            outcomes,
            elapsed_sec=12.3,
            options=RunOptions(samples_path=Path("samples.jsonl"), out_dir=Path("out")),
        )
        self.assertEqual(summary["cases"], 5)
        self.assertEqual(summary["ok"], 4)
        self.assertEqual(summary["failed"], 1)
        self.assertEqual(summary["skipped"], 1)
        self.assertAlmostEqual(summary["mean_score"], (1.0 + 0.5 + 0.7 + 0.6) / 4, places=4)
        self.assertEqual(summary["answered"], 3)
        self.assertEqual(summary["correct"], 1)
        self.assertEqual(summary["uncertain"], 1)
        self.assertAlmostEqual(summary["accuracy_excluding_uncertain"], 1 / 3, places=4)
        self.assertEqual(summary["predictions"], {"妊娠": 2, "未妊娠": 1, "不确定": 1})

    def test_mock_conclusion(self):
        self.assertEqual(_mock_conclusion(0.2, 0.3, 0.25), "未妊娠")
        self.assertEqual(_mock_conclusion(0.5, 0.6, 0.55), "妊娠")
        self.assertEqual(_mock_conclusion(0.5, 0.6, 0.2), "不确定")
        self.assertEqual(_mock_conclusion(None, None, None), "不确定")


if __name__ == "__main__":
    unittest.main()
