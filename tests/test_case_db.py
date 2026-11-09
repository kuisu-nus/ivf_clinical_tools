"""Case database tool: the ID-free table, hiding the evaluated case's fold, read-only SQL."""

import csv
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from clinical_tools.case_db import COLUMNS, TABLE_HEADER, CaseDatabase, fold_of
from clinical_tools.cleaning import load_training
from clinical_tools.common import FEATURE_NAMES
from ivf_services import specs
from ivf_services import server as tools_server
from test_clinical_tools import LiveServer

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "task" / "services" / "data"
TABLE = DATA / "train_cases_35.csv"
PUBLIC = DATA / "train_age_amh_gn.csv"


class TableTests(unittest.TestCase):
    def test_table_has_no_ids_and_matches_the_training_set(self):
        with open(TABLE, encoding="utf-8", newline="") as handle:
            rows = list(csv.reader(handle))
        self.assertEqual(tuple(rows[0]), TABLE_HEADER)
        self.assertNotIn("id", rows[0])
        body = rows[1:]
        self.assertEqual(len(body), 1132)
        self.assertEqual(sum(int(r[-1]) for r in body), 444)
        # the three public columns agree with the repository's age/AMH/Gn file, case for case (as multisets)
        public = load_training(PUBLIC)
        cols = [FEATURE_NAMES.index(n) for n in ("女方年龄", "女方AMH", "女方用药Gn总天数")]
        key = lambda m: sorted(tuple(np.nan_to_num(r, nan=-1).round(4)) for r in m)
        table = np.array([[float(v) if v else np.nan for v in r[1:-1]] for r in body])
        self.assertEqual(key(table[:, cols]), key(public.matrix[:, cols]))
        # row order is not the ID order: outcomes are mixed along the file (IDs 0001-0860 are all 未妊娠)
        labels = [int(r[-1]) for r in body]
        self.assertGreater(sum(labels[:300]), 60)

    def test_hiding_a_case_hides_the_same_fold_as_the_similar_tool(self):
        public = load_training(PUBLIC)
        case = public.case_ids[17]
        visible, _ = public.visible(case)
        db = CaseDatabase(TABLE, exclude_fold=fold_of(case, PUBLIC))
        self.assertEqual(len(db.labels), len(visible.case_ids))
        self.assertEqual(sum(db.labels), int(visible.labels.sum()))
        self.assertIn("排除", db.scope)
        with self.assertRaises(ValueError):
            fold_of("T0001", PUBLIC)


class SqlTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.db = CaseDatabase(TABLE)

    def test_table_content_matches_the_csv(self):
        with open(TABLE, encoding="utf-8", newline="") as handle:
            body = list(csv.reader(handle))[1:]
        age = [float(r[1 + FEATURE_NAMES.index("女方年龄")]) for r in body]
        missing_amh = sum(1 for r in body if not r[1 + FEATURE_NAMES.index("女方AMH")])
        self.assertEqual(self.db.sql("SELECT COUNT(*), SUM(pregnant) FROM cases WHERE age >= 38")["rows"],
                         [[sum(a >= 38 for a in age), sum(int(r[-1]) for r, a in zip(body, age) if a >= 38)]])
        self.assertEqual(self.db.sql("SELECT COUNT(*) FROM cases WHERE amh IS NULL")["rows"], [[missing_amh]])

    def test_sql_is_read_only_and_bounded(self):
        r = self.db.sql("SELECT pregnant, COUNT(*) FROM cases GROUP BY pregnant ORDER BY pregnant")
        self.assertEqual(r["rows"], [[0, 688], [1, 444]])
        self.assertEqual(len(self.db.sql("SELECT * FROM fields")["rows"]), len(COLUMNS) + 1)
        big = self.db.sql("SELECT age FROM cases")
        self.assertTrue(big["已截断"])
        self.assertEqual(big["返回行数"], 200)
        for bad in ("DELETE FROM cases", "DROP TABLE cases", "ATTACH DATABASE ':memory:' AS x",
                    "PRAGMA table_info(cases)", "SELECT nope FROM cases",
                    "WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM c) SELECT COUNT(*) FROM c"):
            with self.assertRaises(ValueError, msg=bad):
                self.db.sql(bad)
        self.assertEqual(self.db.sql("SELECT COUNT(*) FROM cases")["rows"], [[1132]])   # nothing changed


class ServiceTests(unittest.TestCase):
    def setUp(self):
        self.saved = {k: os.environ.get(k) for k in ("IVF_EXCLUDE_CASE", "IVF_EXCLUDE_FOLD", "IVF_TRAIN_CSV")}
        for k in self.saved:
            os.environ.pop(k, None)

    def tearDown(self):
        for k, v in self.saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def test_endpoint(self):
        with TestClient(tools_server.app) as client:
            self.assertEqual(client.post("/tools/ivf_case_sql", json={"sql": "SELECT COUNT(*) FROM cases"}).json()["rows"],
                             [[1132]])
            bad = client.post("/tools/ivf_case_sql", json={"sql": "DELETE FROM cases"})
            self.assertEqual((bad.status_code, bad.json()["error"]), (400, "invalid_sql"))
            for invalid in ({}, {"sql": ""}, {"sql": "SELECT 1", "filters": []}):
                self.assertEqual(client.post("/tools/ivf_case_sql", json=invalid).status_code, 422, invalid)
            self.assertEqual(client.get("/health").json()["tool_info"]["ivf_case_sql"]["cases"], 1132)
            self.assertEqual(client.post("/tools/ivf_query_cases", json={}).status_code, 404)   # removed

    def test_evaluated_case_fold_is_hidden(self):
        case = load_training(PUBLIC).case_ids[5]
        os.environ["IVF_EXCLUDE_CASE"] = case
        with TestClient(tools_server.app) as client:
            health = client.get("/health").json()["tool_info"]["ivf_case_sql"]
        self.assertIn("排除", health["scope"])
        self.assertLess(health["cases"], 1132)
        with TestClient(tools_server.app) as client:
            n = client.post("/tools/ivf_case_sql", json={"sql": "SELECT COUNT(*) FROM cases"}).json()["rows"][0][0]
        self.assertEqual(n, health["cases"])

    def test_mcp_bridge_calls_the_sql_tool(self):
        with LiveServer(tools_server.app) as live, \
                tempfile.TemporaryDirectory() as tmp:
            specs.write_specs(Path(tmp) / "tools", {"ivf_case_sql": live.url})
            env = {**os.environ, "OMNI_TOOL_SPECS_DIR": str(Path(tmp) / "tools"), "OMNI_TRACE_DIR": str(Path(tmp) / "trace")}
            calls = [
                {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "ivf_case_sql", "arguments": {
                    "sql": "SELECT AVG(pregnant) FROM cases WHERE age < 35"}}},
                {"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {"name": "ivf_case_sql", "arguments": {
                    "sql": "UPDATE cases SET pregnant = 1"}}},
            ]
            out = subprocess.run([sys.executable, str(ROOT / "task" / "runtime" / "claw_tools_mcp.py")], env=env,
                                 timeout=60, input="".join(json.dumps(m) + "\n" for m in calls),
                                 capture_output=True, encoding="utf-8")
            replies = {r["id"]: r for r in map(json.loads, out.stdout.splitlines())}
            avg = json.loads(replies[1]["result"]["content"][0]["text"])["rows"][0][0]
            self.assertEqual(avg, CaseDatabase(TABLE).sql("SELECT AVG(pregnant) FROM cases WHERE age < 35")["rows"][0][0])
            self.assertTrue(replies[2]["result"]["isError"])

if __name__ == "__main__":
    unittest.main()
