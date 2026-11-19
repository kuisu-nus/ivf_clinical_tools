"""ovu.report: the population-statistics Word document built from report.json (synthetic data)."""

from __future__ import annotations

import json
import subprocess
import sys
import unittest
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

try:
    import docx  # noqa: F401
    import matplotlib  # noqa: F401
    AVAILABLE = True
except ImportError:
    AVAILABLE = False

from test_ovu_pipeline import synthetic  # noqa: E402


@unittest.skipUnless(AVAILABLE, "python-docx and matplotlib are only needed for the Word report")
class PopulationReportTest(unittest.TestCase):
    def test_document_is_built_from_the_aggregates_alone(self) -> None:
        from ovu.report.population import age_bands
        with TemporaryDirectory() as tmp:
            raw, out = Path(tmp) / "raw.csv", Path(tmp) / "out"
            synthetic(1500).to_csv(raw, index=False, encoding="utf-8-sig")
            subprocess.run([sys.executable, "-m", "ovu.pipeline", str(raw), "--out", str(out)], cwd=ROOT, check=True, capture_output=True)
            only = Path(tmp) / "only_report"
            only.mkdir()
            (only / "report.json").write_bytes((out / "report.json").read_bytes())      # no patient files next to it
            done = subprocess.run([sys.executable, "-m", "ovu.report.population", "--report", str(only / "report.json")],
                                  cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(done.returncode, 0, done.stderr)
            with zipfile.ZipFile(only / "人群统计分析.docx") as archive:
                body = archive.read("word/document.xml").decode("utf-8")
                images = [n for n in archive.namelist() if n.startswith("word/media/")]
            report = json.loads((out / "report.json").read_text(encoding="utf-8"))
            for heading in ("研究人群与数据清洗", "年龄分布", "获卵数分布", "基础情况与内分泌指标", "用药方案与用药情况", "不孕因素与诊断"):
                self.assertIn(heading, body)
            self.assertIn(f"{report['rows']['rows_clean']:,}", body)
            self.assertIn("男方因素（精液、睾丸、精索异常）", body)
            self.assertIn("关键词表的覆盖情况", body)
            self.assertIn("top_uncovered", report["diagnosis"])
            self.assertGreaterEqual(len(images), 5)
            self.assertGreaterEqual(body.count("<w:tbl>"), 8)
            bands = age_bands(report["fields"]["age"]["histogram"])
            self.assertEqual(sum(c for _, c in bands), report["rows"]["rows_clean"])

            # names come from data.yaml, and a field added there shows up in the table of its stage
            self.assertEqual(report["fields"]["fsh"]["title"], "FSH")
            report["fields"]["fsh"]["title"] = "卵泡刺激素"
            report["fields"]["new_marker"] = {**report["fields"]["lh"], "title": "新增指标"}
            (only / "report.json").write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")
            done = subprocess.run([sys.executable, "-m", "ovu.report.population", "--report", str(only / "report.json")],
                                  cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(done.returncode, 0, done.stderr)
            with zipfile.ZipFile(only / "人群统计分析.docx") as archive:
                body = archive.read("word/document.xml").decode("utf-8")
            self.assertIn("卵泡刺激素", body)
            self.assertIn("新增指标", body)

    def test_model_document_before_and_after_the_test_set(self) -> None:
        from test_ovu_ml import prepared
        with TemporaryDirectory() as tmp:
            data = prepared(Path(tmp), 2000)
            subprocess.run([sys.executable, "-m", "ovu.ml.train", "--data", str(data)], cwd=ROOT, check=True, capture_output=True)
            document = data / "model" / "模型性能与可解释分析.docx"
            for evaluate_first, wording in ((False, "测试集尚未评估"), (True, "测试集只评估一次")):
                if evaluate_first:
                    subprocess.run([sys.executable, "-m", "ovu.ml.evaluate_test", "--data", str(data)], cwd=ROOT, check=True, capture_output=True)
                done = subprocess.run([sys.executable, "-m", "ovu.report.model", "--model-dir", str(data / "model")],
                                      cwd=ROOT, capture_output=True, text=True)
                self.assertEqual(done.returncode, 0, done.stderr)
                with zipfile.ZipFile(document) as archive:
                    body = archive.read("word/document.xml").decode("utf-8")
                    images = [n for n in archive.namelist() if n.startswith("word/media/")]
                self.assertIn(wording, body)
                for heading in ("模型与数据", "模型性能", "可解释分析", "5 折交叉验证", "置换重要性", "部分依赖"):
                    self.assertIn(heading, body)
                self.assertIn("用药方案：", body)                                         # readable input names
                self.assertGreaterEqual(len(images), 3)
            # retrained after the test evaluation: the old test numbers belong to another model and are not used
            model = data / "model" / "model.json"
            model.write_text(model.read_text(encoding="utf-8") + "\n", encoding="utf-8")
            done = subprocess.run([sys.executable, "-m", "ovu.report.model", "--model-dir", str(data / "model")],
                                  cwd=ROOT, capture_output=True, text=True)
            self.assertIn("旧模型的测试结果", done.stdout)
            with zipfile.ZipFile(document) as archive:
                self.assertIn("当前模型尚未在测试集上评估", archive.read("word/document.xml").decode("utf-8"))


if __name__ == "__main__":
    unittest.main()
