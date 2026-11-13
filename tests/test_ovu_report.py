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
            self.assertIn("男方因素（精液异常）", body)
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


if __name__ == "__main__":
    unittest.main()
