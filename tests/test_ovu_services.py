"""ovu_services: the oocyte-yield tools over HTTP, and their MCP specs (synthetic data)."""

from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "task" / "services"))
sys.path.insert(0, str(ROOT / "tests"))

from fastapi.testclient import TestClient  # noqa: E402

from test_ovu_ml import prepared  # noqa: E402


def load_service(data: Path, hide_fold: str = ""):
    """(Re)import the service with the given data directory and hidden fold."""
    os.environ["OVU_DATA_DIR"], os.environ["OVU_HIDE_FOLD"] = str(data), hide_fold
    from ovu_services import server, specs, tools
    importlib.reload(tools)
    importlib.reload(server)
    importlib.reload(specs)
    return server, specs


class ServiceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = TemporaryDirectory()
        cls.data = prepared(Path(cls.tmp.name), 2500)
        for module in ("ovu.ml.train", "ovu.similar.build", "ovu.db.build"):
            done = subprocess.run([sys.executable, "-m", module, "--data", str(cls.data)], cwd=ROOT, capture_output=True, text=True)
            assert done.returncode == 0, done.stderr
        cls.server, cls.specs = load_service(cls.data)
        cls.client = TestClient(cls.server.app)
        cls.client.__enter__()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.client.__exit__(None, None, None)
        cls.tmp.cleanup()
        os.environ.pop("OVU_DATA_DIR", None)
        os.environ.pop("OVU_HIDE_FOLD", None)

    def post(self, tool: str, body: dict):
        return self.client.post(f"/tools/{tool}", json=body)

    def test_health_lists_the_five_tools(self) -> None:
        health = self.client.get("/health").json()
        self.assertEqual(health["tools"], ["ovu_predict_oocytes", "ovu_similar_cases", "ovu_case_sql",
                                           "ovu_data_overview", "ovu_sql_examples"])
        self.assertEqual(health["tool_info"]["ovu_predict_oocytes"]["model"], "model.json")
        self.assertIsNone(health["tool_info"]["ovu_similar_cases"]["hidden_fold"])

    def test_predict(self) -> None:
        resp = self.post("ovu_predict_oocytes", {"age": 42, "amh": 0.6, "protocol": "拮抗剂方案", "dx_pcos": 0})
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        self.assertAlmostEqual(sum(body["各分组概率"].values()), 1.0, places=2)
        self.assertIn("fsh", body["未提供的输入"])
        young = self.post("ovu_predict_oocytes", {"age": 25, "amh": 6}).json()
        self.assertGreater(young["预期获卵数"], body["预期获卵数"])

    def test_similar(self) -> None:
        resp = self.post("ovu_similar_cases", {"age": 30, "amh": 3.0, "k": 30})
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        self.assertEqual(sum(g["例数"] for g in body["获卵数分组"].values()), 30)
        self.assertEqual(self.post("ovu_similar_cases", {"age": 30}).json()["检索条件"]["取最相似的例数"], 50)

    def test_bad_arguments_get_a_readable_refusal(self) -> None:
        cases = [("ovu_predict_oocytes", {"age": 5}, 422), ("ovu_predict_oocytes", {"gn_total_dose": 3000}, 422),
                 ("ovu_predict_oocytes", {"dx_pcos": 2}, 422), ("ovu_predict_oocytes", {}, 400),
                 ("ovu_similar_cases", {"age": 30, "k": 100000}, 422), ("ovu_similar_cases", {"k": 30}, 400),
                 ("ovu_case_sql", {"sql": "DELETE FROM cases"}, 400), ("ovu_data_overview", {"x": 1}, 422)]
        for tool, body, status in cases:
            resp = self.post(tool, body)
            self.assertEqual(resp.status_code, status, f"{tool} {body}: {resp.text}")
            self.assertTrue(resp.json()["message"])

    def test_database_tools(self) -> None:
        overview = self.post("ovu_data_overview", {}).json()
        self.assertIn("age", [c["列名"] for c in overview["列"]])
        examples = self.post("ovu_sql_examples", {}).json()["示例"]
        for example in examples:
            resp = self.post("ovu_case_sql", {"sql": example["sql"]})
            self.assertEqual(resp.status_code, 200, example["title"])
        count = self.post("ovu_case_sql", {"sql": "SELECT COUNT(*) FROM cases"}).json()["rows"][0][0]
        self.assertEqual(count, overview["病例数"])

    def test_calls_are_audited_without_empty_arguments(self) -> None:
        self.client.post("/ivf/reset")
        self.post("ovu_predict_oocytes", {"age": 33})
        calls = self.client.get("/ivf/audit").json()["calls"]
        self.assertEqual((calls[-1]["endpoint"], calls[-1]["request_body"]), ("/tools/ovu_predict_oocytes", {"age": 33.0}))

    def test_specs_follow_the_model_and_the_retrieval_config(self) -> None:
        specs = {s["name"]: s for s in self.specs.tool_specs()}
        model = json.loads((self.data / "model" / "model.json").read_text(encoding="utf-8"))
        self.assertEqual(list(specs["ovu_predict_oocytes"]["input_schema"]["properties"]), [i["name"] for i in model["inputs"]])
        similar = specs["ovu_similar_cases"]["input_schema"]["properties"]
        self.assertEqual(similar["age"]["minimum"], 18)
        self.assertEqual(similar["dx_pcos"]["enum"], [0, 1])
        self.assertIn("k", similar)
        self.assertNotIn("gn_total_dose", specs["ovu_predict_oocytes"]["input_schema"]["properties"])
        self.assertEqual(specs["ovu_data_overview"]["input_schema"]["properties"], {})
        self.assertTrue(specs["ovu_case_sql"]["backend"]["url"].endswith(":9211/tools/ovu_case_sql"))
        self.assertNotIn("$ref", json.dumps(specs))


class HiddenFoldTest(unittest.TestCase):
    def test_every_tool_leaves_the_hidden_fold_out(self) -> None:
        with TemporaryDirectory() as tmp:
            data = prepared(Path(tmp), 2500)
            for module in ("ovu.ml.train", "ovu.similar.build", "ovu.db.build"):
                subprocess.run([sys.executable, "-m", module, "--data", str(data)], cwd=ROOT, capture_output=True, check=True)
            try:
                server, _ = load_service(data, hide_fold="2")
                with TestClient(server.app) as client:
                    info = client.get("/health").json()["tool_info"]
                    self.assertEqual(info["ovu_predict_oocytes"]["model"], "model_fold2.json")
                    self.assertEqual(info["ovu_similar_cases"]["hidden_fold"], 2)
                    visible = client.post("/tools/ovu_case_sql", json={"sql": "SELECT COUNT(*) FROM cases"}).json()["rows"][0][0]
                    self.assertEqual(visible, info["ovu_case_sql"]["cases"])
                    self.assertLess(visible, info["ovu_similar_cases"]["cases"])
                    similar = client.post("/tools/ovu_similar_cases", json={"age": 30}).json()
                    self.assertEqual(similar["检索条件"]["可比病例数"], visible)
            finally:
                os.environ.pop("OVU_DATA_DIR", None)
                os.environ.pop("OVU_HIDE_FOLD", None)


if __name__ == "__main__":
    unittest.main()
