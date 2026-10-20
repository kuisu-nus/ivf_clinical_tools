import csv
import json
import os
import random
import re
import subprocess
import sys
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

from clinical_tools.cleaning import clean_file, clean_matrix, load_training, rows_to_training
from clinical_tools.service import make_server, write_specs
from clinical_tools.similar import MIN_COHORT, InputError, SimilarCohort, amh_band, tool_spec
from clinical_tools.common import FEATURE_NAMES

ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / "task"
COL = {n: i for i, n in enumerate(FEATURE_NAMES)}


def synthetic_rows(n=400, seed=5):
    """Pregnancy falls with age; AMH carries per-outcome fills 3.11/3.39 in some rows."""
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        age = rng.randint(24, 44)
        label = int(rng.random() < 1 / (1 + np.exp(-(6.5 - 0.2 * age))))
        row = {"id": f"{i:04d}", "label": str(label), **{f: f"{rng.uniform(1, 50):.2f}" for f in FEATURE_NAMES}}
        row.update({"女方年龄": str(age), "女方AMH": f"{rng.uniform(0.1, 6):.2f}", "男方血肌酐": "70",
                    "男方精液处理方法": str(rng.randint(0, 1))})
        if i % 7 == 0:
            row["女方AMH"] = "3.39" if label else "3.11"
        rows.append(row)
    return rows


class CleaningTests(unittest.TestCase):
    def test_rules_never_read_the_label(self):
        rows = synthetic_rows()
        flipped = [dict(r, label=str(1 - int(r["label"]))) for r in rows]
        np.testing.assert_array_equal(rows_to_training(rows).matrix, rows_to_training(flipped).matrix)

    def test_both_fill_values_and_outliers_become_missing(self):
        x = np.full((4, len(FEATURE_NAMES)), 50.0)
        x[:, COL["男方PR"]] = [28.63, 29.11, 29.1, 40]
        x[:, COL["女方AMH"]] = [3.11, 3.39, 3.4, 0.02]
        x[2, COL["男方浓度"]] = 0
        x[3, COL["男方血肌酐"]] = 5.6
        x[0, COL["男方存活率"]] = 104
        clean, counts = clean_matrix(x)
        pr, amh = clean[:, COL["男方PR"]], clean[:, COL["女方AMH"]]
        self.assertTrue(np.isnan(pr[:2]).all() and np.isnan(amh[:2]).all())
        self.assertEqual((amh[2], amh[3]), (3.4, 0.02))  # neighbours of fills and the detection floor stay
        self.assertTrue(np.isnan(clean[2, [COL[c] for c in ("男方浓度", "男方PR", "男方NP", "男方存活率")]]).all())
        self.assertTrue(np.isnan(clean[3, COL["男方血肌酐"]]))
        self.assertTrue(np.isnan(clean[0, COL["男方存活率"]]))
        self.assertEqual(counts[("结局回填值", "男方PR")], 2)
        again, _ = clean_matrix(clean)
        np.testing.assert_array_equal(again, clean)

    def test_clean_file_refuses_test_set_size_and_existing_output(self):
        rows = synthetic_rows(50)
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "train.csv"
            with open(source, "w", encoding="utf-8-sig", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            with self.assertRaises(SystemExit):
                clean_file(source, Path(tmp) / "out")
            summary = clean_file(source, Path(tmp) / "out", expected_rows=50)
            self.assertFalse(summary["rules_read_labels"])
            self.assertEqual(summary["changes_by_rule"]["结局回填值"], sum(i % 7 == 0 for i in range(50)))
            self.assertNotIn("0001", json.dumps(summary, ensure_ascii=False))
            cleaned = load_training(Path(tmp) / "out" / "train_clean.csv")
            np.testing.assert_array_equal(cleaned.matrix, rows_to_training(rows).matrix)
            with self.assertRaises(FileExistsError):
                clean_file(source, Path(tmp) / "out", expected_rows=50)


class SimilarCohortTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = synthetic_rows()
        cls.training = rows_to_training(cls.rows)
        cls.cohort = SimilarCohort(cls.training)

    def test_statistics_match_a_manual_count(self):
        result = self.cohort.query({"age": 30, "amh": 2.0})
        age, amh = self.training.matrix[:, COL["女方年龄"]], self.training.matrix[:, COL["女方AMH"]]
        mask = (np.abs(age - 30) <= 1) & (amh >= 1.2) & (amh < 3.5)
        self.assertGreaterEqual(mask.sum(), MIN_COHORT)
        self.assertFalse(result["拒绝统计"])
        self.assertEqual(result["符合条件例数"], int(mask.sum()))
        self.assertEqual(result["妊娠结局分布"]["临床妊娠"], int(self.training.labels[mask].sum()))
        self.assertEqual(len(result["相似患者"]), min(int(mask.sum()), 50))
        self.assertTrue(all(abs(p["女方年龄"] - 30) <= 1 for p in result["相似患者"]))

    def test_small_cohorts_are_refused_without_outcomes(self):
        result = self.cohort.query({"age": 44, "amh": 0.15})
        self.assertLess(result["符合条件例数"], MIN_COHORT)
        self.assertTrue(result["拒绝统计"])
        text = json.dumps(result, ensure_ascii=False)
        self.assertNotIn("妊娠结局分布", result)
        self.assertNotIn("相似患者", result)
        self.assertNotIn("未妊娠", text)

    def test_ids_never_appear_and_row_order_does_not_matter(self):
        result = self.cohort.query({"age": 30, "amh": 2.0})
        text = json.dumps(result, ensure_ascii=False)
        self.assertFalse(re.search(r"\b0\d{3}\b", text))
        shuffled = list(self.rows)
        random.Random(1).shuffle(shuffled)
        self.assertEqual(SimilarCohort(rows_to_training(shuffled)).query({"age": 30, "amh": 2.0}), result)

    def test_excluding_a_case_hides_its_fold(self):
        hidden = SimilarCohort(self.training, exclude_case="0007")
        self.assertLess(len(hidden.labels), len(self.training.labels) * 0.85)
        self.assertIn("排除当前病例", hidden.query({"age": 30, "amh": 2.0})["检索范围"])
        with self.assertRaises(ValueError):
            SimilarCohort(self.training, exclude_case="nope")

    def test_bands_and_bad_arguments(self):
        self.assertEqual(amh_band(1.2)[0], 1.2)
        self.assertEqual(amh_band(3.5)[0], 3.5)
        self.assertEqual(amh_band(1.19)[0], 0.0)
        for bad in ({"age": 30}, {"age": "x", "amh": 1}, {"age": 30, "amh": -1}, {"age": 30, "amh": 1, "age_window": 4},
                    {"age": True, "amh": 1}, {"age": 30, "amh": 1, "age_window": 1.5}):
            with self.assertRaises(InputError):
                self.cohort.query(bad)


class ServiceAndBridgeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = make_server(SimilarCohort(rows_to_training(synthetic_rows())), port=0)
        cls.base = f"http://127.0.0.1:{cls.server.server_address[1]}"
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()
        cls.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def post(self, path, body):
        request = urllib.request.Request(self.base + path, data=json.dumps(body).encode(), method="POST",
                                         headers={"Content-Type": "application/json"})
        try:
            with self.opener.open(request, timeout=5) as response:
                return response.status, json.loads(response.read())
        except urllib.error.HTTPError as exc:
            return exc.code, json.loads(exc.read())

    def test_http_endpoints_and_audit(self):
        self.post("/ivf/reset", {})
        status, body = self.post("/ivf/similar_patients", {"age": 30, "amh": 2.0})
        self.assertEqual(status, 200)
        self.assertFalse(body["拒绝统计"])
        status, body = self.post("/ivf/similar_patients", {"age": 30})
        self.assertEqual(status, 400)
        self.assertIn("amh", body["error"])
        with self.opener.open(self.base + "/ivf/audit", timeout=5) as response:
            calls = json.loads(response.read())["calls"]
        self.assertEqual([c["status"] for c in calls], [200, 400])

    def test_spec_matches_generator_and_uses_api_safe_names(self):
        committed = json.loads((TASK / "tools" / "ivf_similar_patients.json").read_text(encoding="utf-8"))
        self.assertEqual(committed, tool_spec())
        for key in committed["input_schema"]["properties"]:
            self.assertRegex(key, r"^[a-zA-Z0-9_.-]{1,64}$")

    def test_mcp_bridge_lists_and_calls_the_tool(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_specs(Path(tmp) / "tools", self.base)
            env = {**os.environ, "OMNI_TOOL_SPECS_DIR": str(Path(tmp) / "tools"),
                   "OMNI_TRACE_DIR": str(Path(tmp) / "trace"), "OMNI_MCP_SERVER_NAME": "ivf-clinical"}
            messages = [
                {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2025-06-18"}},
                {"jsonrpc": "2.0", "method": "notifications/initialized"},
                {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
                {"jsonrpc": "2.0", "id": 3, "method": "tools/call",
                 "params": {"name": "ivf_similar_patients", "arguments": {"age": 44, "amh": 0.15}}},
            ]
            out = subprocess.run([sys.executable, str(TASK / "runtime" / "claw_tools_mcp.py")], env=env, timeout=30,
                                 input="".join(json.dumps(m) + "\n" for m in messages), capture_output=True, text=True)
            replies = {r["id"]: r for r in map(json.loads, out.stdout.splitlines())}
            self.assertEqual([t["name"] for t in replies[2]["result"]["tools"]], ["ivf_similar_patients"])
            call = replies[3]["result"]
            self.assertFalse(call["isError"])
            self.assertTrue(json.loads(call["content"][0]["text"])["拒绝统计"])
            trace = [json.loads(line) for line in (Path(tmp) / "trace" / "dispatches.jsonl").read_text().splitlines()]
            self.assertEqual(trace[0]["request_body"], {"age": 44, "amh": 0.15})


if __name__ == "__main__":
    unittest.main()
