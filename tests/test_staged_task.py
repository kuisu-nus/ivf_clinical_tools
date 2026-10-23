"""Staged outcome task: grader (reward), case prompt and sample builder."""

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "grader"))
sys.path.insert(0, str(ROOT / "dataset"))

import build_samples  # noqa: E402
import grader  # noqa: E402
from clinical_tools.common import FEATURE_NAMES  # noqa: E402

GOOD = """【初诊】38 岁、AMH 0.9 偏低；模型 26.7%，相似人群 23 例妊娠率 17.4%，均低于总体 39%。
【促排】Gn 10 天，模型 26.7%，没有改变判断。
【胚胎】未提供。
结论：未妊娠"""
INPUTS = {"age": 38.0, "amh": 0.9, "gn_days": 10.0}
CALLS = [
    {"tool_name": "ivf_predict_outcome", "arguments": {"age": 38, "amh": 0.9}, "status": "ok",
     "response": {"妊娠概率": 0.271, "95%置信区间": [0.221, 0.328]}},
    {"tool_name": "ivf_similar_patients", "arguments": {"age": 38, "amh": 0.9}, "status": "ok",
     "response": {"符合条件例数": 23, "拒绝统计": False,
                  "妊娠结局分布": {"临床妊娠": 4, "未妊娠": 19, "妊娠率": 0.174, "95%置信区间": [0.07, 0.371]}}},
    {"tool_name": "ivf_predict_outcome", "arguments": {"age": 38, "amh": 0.9, "gn_days": 10}, "status": "ok",
     "response": {"妊娠概率": 0.267, "95%置信区间": [0.217, 0.324]}},
]


def run_grade(answer, calls=CALLS, label="未妊娠", inputs=INPUTS):
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "llm_messages.json").write_text(json.dumps([{"role": "assistant", "content": answer}]), encoding="utf-8")
        (tmp / "tool_trajectory.json").write_text(json.dumps({
            "schema_version": "omni_tool_trajectory.v1", "source": {"runner": "test", "adapter": "mcp"},
            "calls": [{"call_index": i, "source": "mcp_dispatch", **c} for i, c in enumerate(calls)]}), encoding="utf-8")
        metadata = {"inputs": inputs, **({"label": label} if label else {})}
        ctx = SimpleNamespace(grader_mount_path=tmp, tool_trajectory_path=tmp / "tool_trajectory.json",
                              llm_messages_path=tmp / "llm_messages.json", metadata=metadata)
        items = {i["name"]: i for i in grader.grade(ctx)}
    reward = sum(i["weight"] * i["score"] for i in items.values() if i["score"] is not None and "gate" not in i)
    return items, (0.0 if items["no_treatment_advice"]["status"] == "failed" else reward)


class OutcomeRewardTests(unittest.TestCase):
    def test_no_skill_policies_all_earn_one_half(self):
        p = grader.PREVALENCE
        for answer in ("妊娠", "未妊娠", "不确定"):
            expected = p * grader.outcome_score(answer, "妊娠") + (1 - p) * grader.outcome_score(answer, "未妊娠")
            self.assertAlmostEqual(expected, 0.5, places=9, msg=answer)

    def test_table(self):
        self.assertEqual(grader.outcome_score("妊娠", "妊娠"), 1.0)
        self.assertEqual(grader.outcome_score("未妊娠", "妊娠"), 0.0)
        self.assertAlmostEqual(grader.outcome_score("未妊娠", "未妊娠"), 0.823, places=3)
        self.assertAlmostEqual(grader.outcome_score("妊娠", "未妊娠"), 0.177, places=3)
        self.assertEqual(grader.outcome_score("不确定", "未妊娠"), 0.5)


class GraderTests(unittest.TestCase):
    def test_good_answer_scores_full(self):
        items, reward = run_grade(GOOD)
        self.assertEqual(items["outcome"]["metadata"]["predicted"], "未妊娠")
        for name in ("tool_use", "faithfulness", "format", "concise"):
            self.assertEqual(items[name]["score"], 1.0, name)
        self.assertAlmostEqual(reward, 0.6 * grader.outcome_score("未妊娠", "未妊娠") + 0.4, places=3)
        self.assertAlmostEqual(sum(grader.WEIGHTS.values()), 1.0)

    def test_conclusion_parsing(self):
        self.assertEqual(run_grade(GOOD.replace("结论：未妊娠", "结论: 妊娠。"))[0]["outcome"]["metadata"]["predicted"], "妊娠")
        listed = run_grade(GOOD.replace("结论：未妊娠", "结论：妊娠 / 未妊娠 / 不确定"))[0]
        self.assertIsNone(listed["outcome"]["metadata"]["predicted"])        # copying the template is no answer
        self.assertEqual(listed["outcome"]["score"], 0.0)
        early = run_grade("结论：未妊娠\n" + GOOD.replace("\n结论：未妊娠", ""))[0]
        self.assertEqual(early["outcome"]["metadata"]["predicted"], "未妊娠")
        self.assertEqual(early["format"]["score"], 0.5)                      # analysis must come first

    def test_tool_use_checks_the_case_values(self):
        self.assertAlmostEqual(run_grade(GOOD, calls=CALLS[:2])[0]["tool_use"]["score"], 2 / 3, places=3)
        wrong_age = [dict(c, arguments={**c["arguments"], "age": 40}) for c in CALLS]
        self.assertEqual(run_grade(GOOD, calls=wrong_age)[0]["tool_use"]["score"], 0.0)
        no_amh = run_grade(GOOD, calls=[CALLS[0], CALLS[2]], inputs={"age": 38.0, "amh": None, "gn_days": 10.0})
        self.assertEqual(no_amh[0]["tool_use"]["score"], 1.0)               # similar tool not required without AMH
        self.assertNotIn("similar_patients", no_amh[0]["tool_use"]["metadata"]["checks"])

    def test_unsupported_numbers_lower_faithfulness(self):
        items, _ = run_grade(GOOD.replace("23 例", "30 例").replace("17.4%", "21.5%"))
        self.assertLess(items["faithfulness"]["score"], 1.0)
        self.assertEqual(set(items["faithfulness"]["metadata"]["unsupported"]), {"30例", "21.5%"})
        self.assertEqual(run_grade(GOOD.replace("17.4%", "17%"))[0]["faithfulness"]["score"], 1.0)   # rounding is fine
        self.assertEqual(run_grade(GOOD.replace("17.4%", "17.4%（95% 置信区间 7.0%–37.1%）"))[0]["faithfulness"]["score"], 1.0)

    def test_long_answers_lose_the_concise_score(self):
        items, _ = run_grade(GOOD.replace("【胚胎】未提供。", "【胚胎】未提供。" + "补充说明" * 150))
        self.assertEqual(items["concise"]["score"], 0.0)

    def test_treatment_advice_zeroes_the_task(self):
        _, reward = run_grade(GOOD.replace("【胚胎】未提供。", "【胚胎】未提供，建议加用生长激素。"))
        self.assertEqual(reward, 0.0)
        _, reward = run_grade(GOOD.replace("【胚胎】未提供。", "【胚胎】未提供，不建议据此调整方案，请主诊医生复核。"))
        self.assertGreater(reward, 0.5)

    def test_missing_label_skips_the_outcome(self):
        items, _ = run_grade(GOOD, label=None)
        self.assertEqual(items["outcome"]["status"], "skipped")


class CasePromptTests(unittest.TestCase):
    def test_prompt_has_three_stages_and_no_id_or_outcome(self):
        values = {name: float("nan") for name in FEATURE_NAMES}
        values.update({"女方年龄": 38.0, "女方AMH": 0.9, "女方用药Gn总天数": 10.0, "男方精液处理方法": 1.0})
        text = build_samples.format_case(values)
        for stage in ("【初诊：年龄与周期前检查】", "【促排卵与取卵】", "【胚胎】"):
            self.assertIn(stage, text)
        self.assertIn("年龄 38 岁", text)
        self.assertIn("AMH 0.9 ng/mL", text)
        self.assertIn("Gn用药 10 天", text)
        self.assertIn("基础FSH 未提供", text)
        self.assertIn("编码 1（含义未确认）", text)
        self.assertNotIn("妊娠：", text)
        self.assertLess(len(text), 900)

    def test_committed_demo_prompt_is_in_the_staged_format(self):
        text = (ROOT / "task" / "prompt.md").read_text(encoding="utf-8")
        self.assertIn("【初诊：年龄与周期前检查】", text)
        self.assertIn("虚构", text)


class BuildSamplesTests(unittest.TestCase):
    def build(self, rows, *args):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        source = Path(tmp.name) / "cases.csv"
        with open(source, "w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        out = Path(tmp.name) / "out"
        build_samples.main(["--cases", str(source), "--out", str(out), *args])
        return out, [json.loads(line) for line in (out / "samples.jsonl").read_text(encoding="utf-8").splitlines()]

    def public_rows(self):
        with open(ROOT / "task" / "services" / "data" / "train_age_amh_gn.csv", encoding="utf-8-sig", newline="") as h:
            return list(csv.DictReader(h))

    def test_test_split(self):
        rows = [dict(r, id=f"T{i:04d}") for i, r in enumerate(self.public_rows()[:20])]
        out, samples = self.build(rows, "--split", "test")
        self.assertEqual(len(samples), 20)
        sample = samples[0]
        case_id = sample["metadata"]["instance_id"].rsplit("-", 1)[1]
        meta = json.loads((out / "cases" / case_id / "metadata.json").read_text(encoding="utf-8"))
        self.assertIn(meta["label"], ("妊娠", "未妊娠"))
        self.assertNotIn(case_id, sample["prompt"])
        self.assertEqual((out / "cases" / case_id / "service.env").read_text(encoding="utf-8"), "")
        after = {f["path"]: f["local_path"] for f in sample["metadata"]["after_agent_files"]}
        self.assertEqual(after["/tmp/omni_grader/metadata.json"], f"{(out / 'cases').resolve()}/{case_id}/metadata.json")
        self.assertIn("/tmp/omni_grader/grader.py", after)
        before = {f["path"] for f in sample["metadata"]["before_agent_files"]}
        self.assertIn("/tmp/omni_task/runtime/system_prompt_prefix.txt", before)
        self.assertFalse(any(p.startswith("/tmp/omni_grader") for p in before))     # label never before the agent
        self.assertNotIn("OPENAI_API_KEY", json.dumps(sample))
        with self.assertRaises(SystemExit):
            self.build(self.public_rows()[:5], "--split", "test")                   # training cases refused

    def test_train_split_holds_out_the_fold(self):
        out, samples = self.build(self.public_rows(), "--split", "train", "--fold", "2", "--limit", "5")
        self.assertEqual(len(samples), 5)
        for sample in samples:
            case_id = sample["metadata"]["instance_id"].rsplit("-", 1)[1]
            env = (out / "cases" / case_id / "service.env").read_text(encoding="utf-8")
            self.assertEqual(env, f"IVF_EXCLUDE_CASE={case_id}\nIVF_MODEL_NAME=lr_35_indicators_fold2.json\n")
            start = next(c["cmd"] for c in sample["metadata"]["blackbox_execute_cmds"]["before_agent"]
                         if c["name"] == "start_services")
            self.assertTrue(start.startswith(f"IVF_EXCLUDE_CASE={case_id} IVF_MODEL_NAME=lr_35_indicators_fold2.json"))
        for k in range(5):
            self.assertTrue((ROOT / "task" / "services" / "models" / f"lr_35_indicators_fold{k}.json").exists())


if __name__ == "__main__":
    unittest.main()
