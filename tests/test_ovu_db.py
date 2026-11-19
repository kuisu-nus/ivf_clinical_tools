"""ovu.db: read-only SQL on the training cases, the overview and the query examples (synthetic data)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("OVU_DB_BACKEND", "sqlite")            # MySQL is tested separately (MySQLTest)
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
        self.assertIn("拮抗剂方案", by_name["protocol"]["取值"])
        self.assertIn("为 1 的比例", by_name["dxg_ovulation"])
        self.assertIn("多囊", by_name["dxg_ovulation"]["含义"])
        self.assertIn("关键词", by_name["dx_primary"]["含义"])           # described from the keyword list
        self.assertGreater(by_name["amh"]["缺失率"], 0.2)
        text = json.dumps(overview, ensure_ascii=False)
        self.assertLess(len(text), 20_000)                           # an overview, not a dump
        examples = self.db.examples()["示例"]
        self.assertGreaterEqual(len(examples), 5)
        for example in examples:
            self.assertGreater(self.db.sql(example["sql"])["返回行数"], 0, example["title"])


ADMIN = {"OVU_MYSQL_ADMIN_USER": os.environ.get("OVU_TEST_MYSQL_ADMIN_USER", ""),
         "OVU_MYSQL_ADMIN_PASSWORD": os.environ.get("OVU_TEST_MYSQL_ADMIN_PASSWORD", "")}


@unittest.skipUnless(ADMIN["OVU_MYSQL_ADMIN_USER"], "set OVU_TEST_MYSQL_ADMIN_USER/PASSWORD (and OVU_MYSQL_HOST/PORT) to test MySQL")
class MySQLTest(unittest.TestCase):
    """Against a real server: loads the synthetic training set into throw-away databases (ovu_test*)."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = TemporaryDirectory()
        cls.data = prepared(Path(cls.tmp.name), 2000)
        config = Path(cls.tmp.name) / "db.yaml"
        text = (ROOT / "task" / "services" / "ovu" / "config" / "db.yaml").read_text(encoding="utf-8")
        config.write_text(text.replace("database: ovu ", "database: ovu_test ").replace("reader: ovu_reader ", "reader: ovu_test_reader "),
                          encoding="utf-8")
        cls.env = {**os.environ, **ADMIN, "OVU_DB_BACKEND": "mysql", "OVU_MYSQL_PASSWORD": "reader-test-pw"}
        done = subprocess.run([sys.executable, "-m", "ovu.db.build", "--data", str(cls.data), "--config", str(config)],
                              cwd=ROOT, capture_output=True, text=True, env=cls.env)
        assert done.returncode == 0, done.stderr + done.stdout
        cls.output = done.stdout
        cls.spec = json.loads((cls.data / "db" / "db.json").read_text(encoding="utf-8"))
        cls.rows = read_rows(cls.data / "train.csv")
        os.environ["OVU_MYSQL_PASSWORD"] = "reader-test-pw"
        cls.db = CaseDatabase.connect(cls.spec)
        cls.memory = CaseDatabase(cls.spec, cls.rows)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.tmp.cleanup()

    def test_same_answers_as_the_in_memory_copy(self) -> None:
        self.assertIn("示例都能执行", self.output)
        self.assertEqual(self.db.n, len(self.rows))
        for statement in ["SELECT COUNT(*), SUM(amh IS NULL), ROUND(AVG(oocytes), 3) FROM cases",
                          "SELECT oocytes_group, COUNT(*) FROM cases GROUP BY oocytes_group ORDER BY oocytes_group",
                          "SELECT protocol, COUNT(*) FROM cases GROUP BY protocol ORDER BY protocol"]:
            self.assertEqual(self.db.sql(statement)["rows"], self.memory.sql(statement)["rows"], statement)
        self.assertEqual(self.db.overview()["病例数"], len(self.rows))
        types = {c["列名"]: c["类型"] for c in self.db.overview()["列"]}
        self.assertEqual((types["age"], types["amh"], types["protocol"], types["dxg_ovulation"]), ("INT", "DOUBLE", "VARCHAR(100)", "TINYINT"))
        self.assertNotIn("dx_pcos", types)                                   # details stay out: db.yaml diagnosis_level is groups
        self.assertIn("MySQL", self.db.overview()["说明"])

    def test_types_and_comments_come_from_the_config(self) -> None:
        fields = dict((r[0], r[2]) for r in self.db.sql("SELECT column_name, type, description FROM fields")["rows"])
        self.assertEqual(fields["amh"], "AMH")
        self.assertIn("多囊", fields["dxg_ovulation"])
        schema = (self.data / "db" / "schema.sql").read_text(encoding="utf-8")
        self.assertIn("`age` INT NULL COMMENT '女方年龄（岁）'", schema)
        self.assertNotIn("reader-test-pw", schema)

    def test_only_select_on_its_own_view(self) -> None:
        for statement in ("DELETE FROM cases", "UPDATE cases SET age = 1", "DROP VIEW cases", "SELECT 1; SELECT 2",
                          "SELECT * FROM ovu_test_base.cases_all", "SELECT COUNT(*) FROM ovu_test_f1.cases",
                          "SELECT cv_fold FROM cases", "SHOW DATABASES", "SELECT * FROM mysql.user", "not sql"):
            with self.assertRaises(ValueError, msg=statement):
                self.db.sql(statement)
        with self.assertRaisesRegex(ValueError, "被中止"):
            CaseDatabase.connect({**self.spec, "timeout_seconds": 0.5}).sql("SELECT SLEEP(3) FROM cases LIMIT 1")
        result = self.db.sql("SELECT age FROM cases")
        self.assertEqual((result["返回行数"], result["已截断"]), (self.spec["max_rows"], True))

    def test_hidden_fold_uses_its_own_account_and_view(self) -> None:
        hidden = CaseDatabase.connect(self.spec, hide_fold=2)
        self.assertEqual(hidden.n, sum(r["cv_fold"] != "2" for r in self.rows))
        with self.assertRaises(ValueError):
            hidden.sql("SELECT COUNT(*) FROM ovu_test.cases")                            # the full view is not its own


if __name__ == "__main__":
    unittest.main()
