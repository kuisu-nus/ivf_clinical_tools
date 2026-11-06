"""ckpt_eval: OpenClaw config editing, isolated OpenClaw homes, weight check and model comparison."""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CKPT = ROOT / "ckpt_eval"
sys.path.insert(0, str(CKPT))
import check_weights  # noqa: E402
import compare  # noqa: E402
import oc_config  # noqa: E402

BASH = shutil.which("bash")
OPENCLAW_CONFIG = {
    "models": {"providers": {"sglang": {"baseUrl": "http://10.0.0.1:30000/v1", "api": "openai-completions",
                                        "models": [{"id": "Hy3", "name": "Hy3", "contextWindow": 128000,
                                                    "maxTokens": 8192}]}}},
    "agents": {"defaults": {"model": {"primary": "sglang/Hy3"}, "workspace": "/old/workspace"}},
}


class OcConfigTests(unittest.TestCase):
    def test_adds_a_local_provider_and_keeps_the_others(self):
        new = oc_config.point_to_model(OPENCLAW_CONFIG, "http://127.0.0.1:30001/v1", "ivf-step100",
                                       workspace="/new/workspace")
        providers = new["models"]["providers"]
        self.assertEqual(providers["sglang"], OPENCLAW_CONFIG["models"]["providers"]["sglang"])
        self.assertEqual(providers["ivf"]["baseUrl"], "http://127.0.0.1:30001/v1")
        self.assertEqual(providers["ivf"]["apiKey"], oc_config.PLACEHOLDER_KEY)     # no auth-store entry
        self.assertEqual(providers["ivf"]["models"][0]["id"], "ivf-step100")
        self.assertEqual(providers["ivf"]["models"][0]["contextWindow"], 128000)   # follows the existing model
        self.assertEqual(new["agents"]["defaults"]["model"], {"primary": "ivf/ivf-step100"})
        self.assertEqual(new["agents"]["defaults"]["workspace"], "/new/workspace")
        self.assertEqual(OPENCLAW_CONFIG["agents"]["defaults"]["model"]["primary"], "sglang/Hy3")  # input untouched

    def test_empty_config(self):
        new = oc_config.point_to_model({}, "http://127.0.0.1:30000/v1", "m")
        self.assertEqual(new["models"]["providers"]["ivf"]["api"], "openai-completions")
        self.assertEqual(new["agents"]["defaults"]["model"]["primary"], "ivf/m")


class CheckWeightsTests(unittest.TestCase):
    def test_names_must_match(self):
        with tempfile.TemporaryDirectory() as tmp:
            a, b = Path(tmp, "a"), Path(tmp, "b")
            for d, names in ((a, ["x", "y"]), (b, ["x"])):
                d.mkdir()
                (d / "config.json").write_text("{}")
                (d / "model.safetensors.index.json").write_text(json.dumps({"weight_map": dict.fromkeys(names, "f")}))
            self.assertEqual(check_weights.main([str(a), str(a)]), 0)
            self.assertEqual(check_weights.main([str(a), str(b)]), 1)


def fake_eval(root: Path, answers: list[tuple[str, str]]) -> Path:
    for i, (predicted, truth) in enumerate(answers):
        case = root / "cases" / f"{i:04d}"
        case.mkdir(parents=True)
        right = predicted == truth
        items = [{"name": "outcome", "score": 1.0 if right else 0.0, "reason": "r",
                  "metadata": {"predicted": predicted, "truth": truth}},
                 {"name": "tool_use", "score": 1.0, "reason": "r"}]
        (case / "score.json").write_text(json.dumps({"grader": {"overall_score": 0.5, "raw_scores": {"items": items}}}))
        (case / "dispatches.jsonl").write_text('{"a": 1}\n{"a": 2}\n')
        (case / "answer.txt").write_text(f"结论：{predicted}\n", encoding="utf-8")
    return root


class CompareTests(unittest.TestCase):
    def test_metrics_and_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            a = fake_eval(tmp / "a", [("妊娠", "妊娠"), ("未妊娠", "未妊娠"), ("妊娠", "未妊娠"), ("不确定", "妊娠")])
            b = fake_eval(tmp / "b", [("未妊娠", "妊娠"), ("未妊娠", "未妊娠"), ("未妊娠", "未妊娠"), ("未妊娠", "妊娠")])
            m = compare.metrics(compare.load(a))
            self.assertEqual(m["answers"], "2 / 1 / 1")
            self.assertEqual(m["sensitivity"], 0.5)
            self.assertEqual(m["specificity"], 0.5)
            self.assertEqual(m["right_when_妊娠"], "1/2")
            self.assertEqual(m["tool_calls"], 2)
            self.assertEqual(compare.metrics(compare.load(b))["balanced_accuracy"], 0.5)   # one answer for all
            out = tmp / "rows.jsonl"
            compare.main([str(a), str(b), "--names", "A", "B", "--export", str(out)])
            rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(rows), 8)
            self.assertEqual(rows[0]["answer"], "结论：妊娠")


@unittest.skipUnless(BASH, "needs bash")
class OcHomeTests(unittest.TestCase):
    def test_builds_an_isolated_home_and_leaves_the_original(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            src = tmp / "home"
            (src / ".openclaw" / "workspace").mkdir(parents=True)
            (src / ".openclaw" / "openclaw.json").write_text(json.dumps(OPENCLAW_CONFIG))
            (src / ".openclaw" / "workspace" / "BOOTSTRAP.md").write_text("first run " * 100)
            (src / ".openclaw" / "workspace" / "SOUL.md").write_text("long default " * 100)
            stub = tmp / "bin" / "openclaw"                    # records mcp set, answers mcp probe
            stub.parent.mkdir()
            stub.write_text('#!/usr/bin/env bash\necho "$HOME $*" >> "$STUB_LOG"\n'
                            '[ "$1 $2" = "mcp probe" ] && echo "- claw-tools: 2 tools"\nexit 0\n')
            stub.chmod(0o755)
            env = {**os.environ, "OC_SOURCE_HOME": str(src), "OC_EVAL_ROOT": str(tmp / "eval"),
                   "PATH": f"{stub.parent}:{os.environ['PATH']}", "OPENCLAW_BIN": str(stub), "STUB_LOG": str(tmp / "log"), "PYTHON": sys.executable}
            run = subprocess.run([BASH, str(CKPT / "oc_home.sh"), "step100", "30001", "9321"], env=env,
                                 capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stderr)
            home = tmp / "eval" / "step100"
            config = json.loads((home / ".openclaw" / "openclaw.json").read_text())
            self.assertEqual(config["agents"]["defaults"]["model"]["primary"], "ivf/step100")
            self.assertEqual(config["models"]["providers"]["ivf"]["baseUrl"], "http://127.0.0.1:30001/v1")
            ws = home / ".openclaw" / "workspace"
            self.assertFalse((ws / "BOOTSTRAP.md").exists())
            self.assertEqual((ws / "SOUL.md").read_text(), "# SOUL.md\n")
            self.assertTrue((ws / "AGENTS.md").read_text(encoding="utf-8").startswith("你是生殖医学临床决策支持 agent"))
            self.assertIn("claw-tools__ivf_predict_outcome", (ws / "AGENTS.md").read_text(encoding="utf-8"))
            env_sh = (home / "env.sh").read_text()
            self.assertIn(f"HOME={home}", env_sh)
            self.assertIn("IVF_TOOLS_PORT=9321", env_sh)
            self.assertIn(f"{home} mcp set claw-tools", (tmp / "log").read_text())     # registered in the copy
            self.assertEqual(json.loads((src / ".openclaw" / "openclaw.json").read_text()), OPENCLAW_CONFIG)
            self.assertTrue((src / ".openclaw" / "workspace" / "BOOTSTRAP.md").exists())
            again = subprocess.run([BASH, str(CKPT / "oc_home.sh"), "step100", "30001", "9321"], env=env,
                                   capture_output=True, text=True)
            self.assertNotEqual(again.returncode, 0)                                 # no silent overwrite
            self.assertIn("FORCE=1", again.stderr)


if __name__ == "__main__":
    unittest.main()
