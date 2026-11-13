"""ovu.db: read-only SQL on the training cases, the overview and the query examples (synthetic data)."""

from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from ovu.db.database import CaseDatabase  # noqa: E402
from ovu.ml.train import read_rows  # noqa: E402
from test_ovu_ml import prepared  # noqa: E402


class DatabaseTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = TemporaryDirectory()
        cls.data = prepared(Path(cls.tmp.name), 2000)
        done = subprocess.run([sys.executable, "-m", "ovu.db.build", "--data", str(cls.data)], cwd=ROOT, capture_output=True, text=True)
        assert done.returncode == 0, done.stderr
        cls.spec = json.loads((cls.data / "db" / "db.json").read_text(encoding="utf-8"))
        cls.rows = read_rows(cls.data / "train.csv")
        cls.db = CaseDatabase(cls.spec, cls.rows)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.tmp.cleanup()

    def test_select_works_and_missing_is_null(self) -> None:
        result = self.db.sql("SELECT COUNT(*), SUM(amh IS NULL), AVG(oocytes) FROM cases")
        n, amh_missing, mean = result["rows"][0]
        self.assertEqual(n, len(self.rows))
        self.assertEqual(amh_missing, sum(r["amh"] == "" for r in self.rows))
        self.assertAlmostEqual(mean, sum(int(r["oocytes"]) for r in self.rows) / n, places=3)
        groups = dict(self.db.sql("SELECT oocytes_group, COUNT(*) FROM cases GROUP BY 1")["rows"])
        self.assertEqual(sum(groups.values()), n)
        self.assertLessEqual(set(groups), set(self.spec["groups"]))

    def test_anything_but_one_select_is_refused(self) -> None:
        for statement in ("DELETE FROM cases", "DROP TABLE cases", "UPDATE cases SET age = 1", "PRAGMA table_info(cases)",
                          "ATTACH DATABASE 'x.db' AS x", "SELECT 1; SELECT 2", "CREATE TABLE t (a)", "not sql"):
            with self.assertRaises(ValueError, msg=statement):
                self.db.sql(statement)
        self.assertEqual(self.db.sql("SELECT COUNT(*) FROM cases")["rows"][0][0], len(self.rows))

    def test_ids_folds_and_free_text_are_not_reachable(self) -> None:
        names = [c["name"] for c in self.spec["columns"]]
        for hidden in ("cv_fold", "diagnosis_text", "周期号", "_key"):
            self.assertNotIn(hidden, names)
            with self.assertRaises(ValueError):
                self.db.sql(f"SELECT {hidden} FROM cases")
        self.assertIn("gn_total_dose", names)                       # during-stimulation fields are there to explore

    def test_row_limit_and_timeout(self) -> None:
        result = self.db.sql("SELECT age FROM cases")
        self.assertEqual((result["返回行数"], result["已截断"]), (self.spec["max_rows"], True))
        slow = CaseDatabase({**self.spec, "timeout_seconds": 0.05}, self.rows)
        with self.assertRaisesRegex(ValueError, "被中止"):
            slow.sql("WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM c) SELECT COUNT(*) FROM c")

    def test_hidden_fold(self) -> None:
        hidden = CaseDatabase(self.spec, self.rows, hide_fold=3)
        expected = sum(r["cv_fold"] != "3" for r in self.rows)
        self.assertEqual(hidden.sql("SELECT COUNT(*) FROM cases")["rows"][0][0], expected)
        self.assertEqual(hidden.overview()["病例数"], expected)
        self.assertIn("不含", hidden.overview()["检索范围"])

    def test_overview_and_examples(self) -> None:
        overview = self.db.overview()
        self.assertEqual(sum(g["例数"] for g in overview["获卵数分组"].values()), len(self.rows))
        by_name = {c["列名"]: c for c in overview["列"]}
        self.assertEqual(by_name["gn_total_dose"]["时点"], "during")
        self.assertIn("拮抗剂方案", by_name["protocol"]["取值"])
        self.assertIn("为 1 的比例", by_name["dx_pcos"])
        self.assertIn("多囊", by_name["dx_pcos"]["含义"])
        self.assertIn("关键词", by_name["dx_tubal"]["含义"])           # described from the keyword list
        self.assertGreater(by_name["amh"]["缺失率"], 0.2)
        text = json.dumps(overview, ensure_ascii=False)
        self.assertLess(len(text), 20_000)                           # an overview, not a dump
        examples = self.db.examples()["示例"]
        self.assertGreaterEqual(len(examples), 5)
        for example in examples:
            self.assertGreater(self.db.sql(example["sql"])["返回行数"], 0, example["title"])


if __name__ == "__main__":
    unittest.main()
