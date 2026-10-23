"""openclaw_eval: answer extraction, scoring of the example run, AGENTS.md tool instructions."""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "openclaw_eval"
CASE = SCRIPTS / "examples" / "cases" / "demo"
OUTPUT = SCRIPTS / "examples" / "output" / "demo"
BASH = shutil.which("bash")


@unittest.skipUnless(BASH, "needs bash")
class OpenClawEvalTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)
        self.env = {**os.environ, "PYTHON": sys.executable, "OC_EVAL_WORK": str(self.tmp / "work")}

    def run_script(self, name, *args, **env):
        out = subprocess.run([BASH, str(SCRIPTS / name), *map(str, args)], env={**self.env, **env},
                             capture_output=True, text=True, encoding="utf-8")
        self.assertEqual(out.returncode, 0, out.stderr + out.stdout)
        return out.stdout

    def test_agent_keeps_only_the_reply(self):
        stub = self.tmp / "openclaw"                  # prints OpenClaw's recorded output, progress lines included
        stub.write_text(f"#!/usr/bin/env bash\ncat '{OUTPUT / 'openclaw_raw.txt'}'\n", encoding="utf-8")
        stub.chmod(0o755)
        res = self.tmp / "res"
        self.run_script("agent.sh", CASE, res, OPENCLAW_BIN=str(stub))
        answer = (res / "answer.txt").read_text(encoding="utf-8")
        self.assertEqual(answer, (OUTPUT / "answer.txt").read_text(encoding="utf-8"))
        self.assertTrue(answer.startswith("【初诊】"))
        self.assertNotIn("│", answer)
        self.assertEqual(json.loads((res / "agent_run.json").read_text())["openclaw_exit"], 0)

    def test_after_agent_reproduces_the_example_score(self):
        res = self.tmp / "res"
        res.mkdir()
        for name in ("answer.txt", "dispatches.jsonl", "agent_run.json"):
            shutil.copy(OUTPUT / name, res / name)
        self.run_script("after_agent.sh", CASE, res)
        graded = json.loads((res / "grader_output.json").read_text(encoding="utf-8"))
        expected = json.loads((OUTPUT / "grader_output.json").read_text(encoding="utf-8"))
        self.assertEqual(graded["overall_score"], expected["overall_score"])
        items = {i["name"]: i for i in graded["raw_scores"]["items"]}
        self.assertEqual(items["tool_use"]["score"], 1.0)
        self.assertEqual(items["outcome"]["metadata"]["predicted"], "未妊娠")
        trajectory = json.loads((res / "grader_mount" / "tool_trajectory.json").read_text(encoding="utf-8"))
        self.assertEqual(len(trajectory["calls"]), 3)
        self.assertEqual(json.loads((res / "score.json").read_text(encoding="utf-8"))["case_id"], "demo")

    def test_tool_instructions_go_to_the_named_agents_md(self):
        agents = self.tmp / "AGENTS.md"
        agents.write_text("task prompt\n", encoding="utf-8")
        subprocess.run([sys.executable, str(ROOT / "task" / "runtime" / "claw_tools_mcp.py"), "--append-agents-md"],
                       env={**os.environ, "OMNI_TOOL_SPECS_DIR": str(ROOT / "task" / "tools"),
                            "OMNI_AGENTS_MD": str(agents)}, check=True, capture_output=True)
        text = agents.read_text(encoding="utf-8")
        self.assertTrue(text.startswith("task prompt"))
        self.assertIn("claw-tools__ivf_predict_outcome", text)
        self.assertIn("claw-tools__ivf_similar_patients", text)


if __name__ == "__main__":
    unittest.main()
