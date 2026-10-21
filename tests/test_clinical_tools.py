import csv
import json
import logging
import os
import random
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

os.environ.setdefault("IVF_LOG_LEVEL", "WARNING")       # keep test output quiet; set before the services load

import uvicorn
from fastapi.testclient import TestClient

from clinical_tools import train_predictor
from clinical_tools.cleaning import clean_file, clean_matrix, load_training, rows_to_training
from clinical_tools.common import FEATURE_NAMES
from clinical_tools.predict import Predictor, auc, fit_logistic, train_model
from clinical_tools.similar import (MIN_COHORT, STRATEGIES, AgeAmhBand, Query, Selection, SimilarCohort,
                                    SimilarityStrategy, amh_band)
from ivf_services import specs
from ivf_services._base import ComputeGate, get_logger
from ivf_services.predict import server as predict_server
from ivf_services.similar import server as similar_server

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


def write_csv(path, rows):
    with open(path, "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class AgeOnly(SimilarityStrategy):
    """A second strategy, defined outside the service, to show that one class is all it takes."""
    name, description = "age_only_test", "只按年龄 ±2 岁"

    def select(self, cohort, query):
        age = cohort.column("女方年龄")
        searchable = ~np.isnan(age)
        with np.errstate(invalid="ignore"):
            matched = searchable & (np.abs(age - query.age) <= 2)
        return Selection("女方年龄 ±2 岁", searchable, matched, np.abs(age - query.age), "无")


class SimilarCohortTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = synthetic_rows()
        cls.training = rows_to_training(cls.rows)
        cls.cohort = SimilarCohort(cls.training)

    def test_statistics_match_a_manual_count(self):
        result = self.cohort.query(Query(30, 2.0))
        age, amh = self.training.matrix[:, COL["女方年龄"]], self.training.matrix[:, COL["女方AMH"]]
        mask = (np.abs(age - 30) <= 1) & (amh >= 1.2) & (amh < 3.5)
        self.assertGreaterEqual(mask.sum(), MIN_COHORT)
        self.assertFalse(result["拒绝统计"])
        self.assertEqual(result["检索策略"], "age_amh_band")
        self.assertEqual(result["符合条件例数"], int(mask.sum()))
        self.assertEqual(result["妊娠结局分布"]["临床妊娠"], int(self.training.labels[mask].sum()))
        self.assertEqual(len(result["相似患者"]), min(int(mask.sum()), 50))
        self.assertTrue(all(abs(p["女方年龄"] - 30) <= 1 for p in result["相似患者"]))
        self.assertEqual(set(result["相似患者"][0]), {"女方年龄", "女方AMH", "妊娠结局"})

    def test_small_cohorts_are_refused_without_outcomes(self):
        result = self.cohort.query(Query(44, 0.15))
        self.assertLess(result["符合条件例数"], MIN_COHORT)
        self.assertTrue(result["拒绝统计"])
        text = json.dumps(result, ensure_ascii=False)
        self.assertNotIn("妊娠结局分布", result)
        self.assertNotIn("相似患者", result)
        self.assertNotIn("未妊娠", text)
        self.assertIn("放宽", result["建议"])
        self.assertIn("最大年龄窗口", self.cohort.query(Query(54, 0.15, 3))["建议"])

    def test_ids_never_appear_and_row_order_does_not_matter(self):
        result = self.cohort.query(Query(30, 2.0), extra_fields=["女方基础FSH"])
        text = json.dumps(result, ensure_ascii=False)
        self.assertFalse([cid for cid in self.training.case_ids if f'"{cid}"' in text])
        shuffled = list(self.rows)
        random.Random(1).shuffle(shuffled)
        again = SimilarCohort(rows_to_training(shuffled)).query(Query(30, 2.0), extra_fields=["女方基础FSH"])
        self.assertEqual(again, result)

    def test_extra_fields_are_listed_per_patient(self):
        result = self.cohort.query(Query(30, 2.0), extra_fields=["女方基础FSH", "男方DFI", "女方基础FSH"])
        first = result["相似患者"][0]
        self.assertEqual(list(first), ["女方年龄", "女方AMH", "女方基础FSH", "男方DFI", "妊娠结局"])
        # values come from the cleaned matrix; a cleaned-away value is null
        self.assertTrue(all(p["女方基础FSH"] is None or isinstance(p["女方基础FSH"], float) for p in result["相似患者"]))
        self.assertEqual(result["妊娠结局分布"], self.cohort.query(Query(30, 2.0))["妊娠结局分布"])

    def test_a_new_strategy_plugs_in_without_touching_the_service(self):
        STRATEGIES["age_only_test"] = AgeOnly()
        try:
            result = self.cohort.query(Query(30, 2.0), strategy="age_only_test")
            age = self.training.matrix[:, COL["女方年龄"]]
            self.assertEqual(result["符合条件例数"], int((np.abs(age - 30) <= 2).sum()))
            self.assertEqual(result["相似人群临床信息"], {})
        finally:
            del STRATEGIES["age_only_test"]

    def test_excluding_a_case_hides_its_fold(self):
        hidden = SimilarCohort(self.training, exclude_case="0007")
        self.assertLess(len(hidden.labels), len(self.training.labels) * 0.85)
        self.assertIn("排除当前病例", hidden.query(Query(30, 2.0))["检索范围"])
        with self.assertRaises(ValueError):
            SimilarCohort(self.training, exclude_case="nope")

    def test_bands(self):
        self.assertEqual(amh_band(1.2)[0], 1.2)
        self.assertEqual(amh_band(3.5)[0], 3.5)
        self.assertEqual(amh_band(1.19)[0], 0.0)
        self.assertIsInstance(STRATEGIES["age_amh_band"], AgeAmhBand)


class PredictorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.training = rows_to_training(synthetic_rows())
        cls.artifact = json.loads(json.dumps(train_model(cls.training)))  # as read back from disk
        cls.predictor = Predictor(cls.artifact)

    def test_fit_recovers_known_coefficients(self):
        rng = np.random.default_rng(0)
        x = np.column_stack([np.ones(20000), rng.uniform(22, 45, 20000), rng.uniform(0.1, 8, 20000), rng.integers(8, 13, 20000)])
        truth = np.array([0.5, -0.06, 0.03, 0.2])
        y = (rng.random(20000) < 1 / (1 + np.exp(-(x @ truth)))).astype(int)
        beta, cov = fit_logistic(x, y)
        np.testing.assert_allclose(beta, truth, atol=4 * np.sqrt(np.diag(cov)).max())

    def test_auc_matches_the_pairwise_definition(self):
        scores = np.array([0.1, 0.4, 0.4, 0.35, 0.8, 0.4])
        labels = np.array([0, 0, 1, 1, 1, 0])
        pos, neg = scores[labels == 1], scores[labels == 0]
        expected = np.mean([(p > n) + 0.5 * (p == n) for p in pos for n in neg])
        self.assertAlmostEqual(auc(scores, labels), expected)

    def test_artifact_holds_only_aggregates(self):
        self.assertEqual(self.artifact["features"], list(FEATURE_NAMES))
        self.assertEqual(len(self.artifact["weights"]), len(FEATURE_NAMES))
        self.assertEqual(np.array(self.artifact["covariance"]).shape, (len(FEATURE_NAMES) + 1,) * 2)
        self.assertEqual(self.artifact["training"]["n"], len(self.training.labels))
        self.assertGreater(self.artifact["training"]["cv_auc_three_inputs"], 0.5)
        text = json.dumps(self.artifact, ensure_ascii=False)
        self.assertFalse([cid for cid in self.training.case_ids if f'"{cid}"' in text])

    def test_output_is_only_probability_and_interval(self):
        young = self.predictor.predict(age=26, amh=2.0, gn_days=10)
        old = self.predictor.predict(age=43, amh=2.0, gn_days=10)
        for r in (young, old):
            self.assertEqual(set(r), {"妊娠概率", "95%置信区间"})
            self.assertLessEqual(r["95%置信区间"][0], r["妊娠概率"])
            self.assertLessEqual(r["妊娠概率"], r["95%置信区间"][1])
        self.assertGreater(young["妊娠概率"], old["妊娠概率"])

    def test_other_indicators_and_missing_inputs_use_training_means(self):
        means = dict(zip(self.artifact["features"], self.artifact["means"]))
        self.assertEqual(self.predictor.predict(age=33),
                         self.predictor.predict(age=33, amh=means["女方AMH"], gn_days=means["女方用药Gn总天数"]))
        at_means = self.predictor.predict(age=means["女方年龄"])   # every input at its mean: the intercept alone
        self.assertAlmostEqual(at_means["妊娠概率"], 1 / (1 + np.exp(-self.artifact["intercept"])), places=3)
        for bad in ({}, {"age": None}, {"fsh": 8}):
            with self.assertRaises(ValueError):
                self.predictor.predict(**bad)

    def test_training_cli_writes_once_and_predictor_needs_only_the_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "train.csv"
            write_csv(source, synthetic_rows())
            out = Path(tmp) / "model.json"
            train_predictor.main(["--csv", str(source), "--out", str(out)])
            source.unlink()  # prediction must not need the training data
            loaded = Predictor.from_file(out)
            self.assertEqual(loaded.predict(age=30, amh=2, gn_days=10), self.predictor.predict(age=30, amh=2, gn_days=10))
            with self.assertRaises(SystemExit):
                train_predictor.main(["--csv", str(source), "--out", str(out)])
        with self.assertRaises(ValueError):
            Predictor(dict(self.artifact, inputs={"age": "女方年龄"}))

    def test_excluding_a_case_retrains_without_its_fold(self):
        subset, scope = self.training.visible("0007")
        self.assertLess(train_model(subset, scope=scope)["training"]["n"], self.artifact["training"]["n"])


class ServiceEnv(unittest.TestCase):
    """Training CSV and model artifact on disk, pointed to by the services' environment."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        rows = synthetic_rows()
        cls.csv = Path(cls.tmp.name) / "train.csv"
        write_csv(cls.csv, rows)
        cls.model = Path(cls.tmp.name) / "model.json"
        cls.model.write_text(json.dumps(train_model(rows_to_training(rows))), encoding="utf-8")
        cls.saved = {k: os.environ.get(k) for k in ("IVF_TRAIN_CSV", "IVF_MODEL", "IVF_EXCLUDE_CASE")}
        os.environ.update(IVF_TRAIN_CSV=str(cls.csv), IVF_MODEL=str(cls.model))
        os.environ.pop("IVF_EXCLUDE_CASE", None)

    @classmethod
    def tearDownClass(cls):
        for key, value in cls.saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        cls.tmp.cleanup()


class ApiTests(ServiceEnv):
    def test_similar_endpoint_and_validation(self):
        with TestClient(similar_server.app) as client:
            client.post("/ivf/reset")
            ok = client.post("/ivf/similar_patients", json={"age": 30, "amh": 2.0})
            self.assertEqual(ok.status_code, 200)
            self.assertFalse(ok.json()["拒绝统计"])
            for bad in ({"age": 30}, {"age": "x", "amh": 1}, {"age": 30, "amh": -1}, {"age": 30, "amh": 1, "age_window": 4},
                        {"age": 30, "amh": 1, "age_window": 1.5}, {"age": 30, "amh": True}, {"age": 30, "amh": 1, "id": "0001"},
                        {"age": 30, "amh": 1, "strategy": "nope"}, {"age": 30, "amh": 1, "extra_fields": ["label"]},
                        {"age": 30, "amh": 1, "extra_fields": ["女方年龄"]}):
                response = client.post("/ivf/similar_patients", json=bad)
                self.assertEqual(response.status_code, 422, bad)
                self.assertTrue(response.json()["message"].startswith("参数校验失败"))
            self.assertEqual(client.post("/ivf/similar_patients", content=b"{not json",
                                         headers={"Content-Type": "application/json"}).status_code, 422)
            calls = client.get("/ivf/audit").json()["calls"]
            self.assertEqual([c["status"] for c in calls][:2], [200, 422])
            self.assertEqual(calls[0]["request_body"]["age"], 30)
            health = client.get("/health").json()
            self.assertEqual((health["status"], health["tool"], health["cases"]), ("ok", "ivf_similar_patients", 400))
            self.assertEqual(client.post("/ivf/reset").json(), {"status": "reset"})
            self.assertEqual(client.get("/ivf/audit").json()["calls"], [])

    def test_a_registered_strategy_is_served_without_service_changes(self):
        STRATEGIES["age_only_test"] = AgeOnly()
        try:
            with TestClient(similar_server.app) as client:
                body = client.post("/ivf/similar_patients", json={"age": 30, "amh": 2.0, "strategy": "age_only_test"}).json()
            self.assertEqual((body["检索策略"], body["筛选条件"]), ("age_only_test", "女方年龄 ±2 岁"))
        finally:
            del STRATEGIES["age_only_test"]

    def test_predict_endpoint_and_validation(self):
        with TestClient(predict_server.app) as client:
            ok = client.post("/ivf/predict_outcome", json={"age": 30, "amh": 2.0, "gn_days": 10})
            self.assertEqual((ok.status_code, set(ok.json())), (200, {"妊娠概率", "95%置信区间"}))
            self.assertEqual(client.post("/ivf/predict_outcome", json={"amh": 2.0}).status_code, 200)
            for bad in ({}, {"age": None}, {"age": 80}, {"gn_days": -1}, {"age": 30, "fsh": 8}, {"gn_days": True}):
                self.assertEqual(client.post("/ivf/predict_outcome", json=bad).status_code, 422, bad)
            self.assertEqual(client.get("/health").json()["tool"], "ivf_predict_outcome")

    def test_each_service_serves_only_its_own_tool(self):
        with TestClient(similar_server.app) as similar, TestClient(predict_server.app) as predict:
            self.assertEqual(similar.post("/ivf/predict_outcome", json={"age": 30}).status_code, 404)
            self.assertEqual(predict.post("/ivf/similar_patients", json={"age": 30, "amh": 2}).status_code, 404)

    def test_requests_and_calls_are_logged(self):
        with TestClient(similar_server.app) as client, self.assertLogs("ivf.ivf_similar", level="INFO") as logs:
            client.post("/ivf/similar_patients", json={"age": 30, "amh": 2.0})
        text = "\n".join(logs.output)
        self.assertRegex(text, r"POST /ivf/similar_patients -> 200 [\d.]+ms")
        self.assertIn('call /ivf/similar_patients status=200 args={"age": 30.0, "amh": 2.0', text)
        self.assertNotIn("未妊娠", text)        # results stay out of the log, only a summary

    def test_log_files_rotate_in_the_log_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            saved = {k: os.environ.get(k) for k in ("IVF_LOG_DIR", "IVF_LOG_MAX_BYTES", "IVF_LOG_BACKUPS")}
            os.environ.update(IVF_LOG_DIR=tmp, IVF_LOG_MAX_BYTES="2000", IVF_LOG_BACKUPS="2")
            try:
                logger = get_logger("rotation_test")
                logger.setLevel(logging.INFO)
                for i in range(200):
                    logger.info("line %d %s", i, "x" * 40)
            finally:
                for handler in list(logger.handlers):
                    handler.close()
                    logger.removeHandler(handler)
                for key, value in saved.items():
                    os.environ.pop(key, None) if value is None else os.environ.__setitem__(key, value)
            self.assertEqual(sorted(p.name for p in Path(tmp).iterdir()),
                             ["rotation_test.log", "rotation_test.log.1", "rotation_test.log.2"])


class LiveServer:
    """A service on a free port in a background thread (real HTTP, as the MCP bridge uses)."""

    def __init__(self, app):
        self.server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=0, log_level="warning"))
        self.thread = threading.Thread(target=self.server.run, daemon=True)

    def __enter__(self):
        self.thread.start()
        deadline = time.time() + 20
        while not self.server.started:
            if time.time() > deadline:
                raise RuntimeError("server did not start")
            time.sleep(0.02)
        self.url = f"http://127.0.0.1:{self.server.servers[0].sockets[0].getsockname()[1]}"
        return self

    def __exit__(self, *exc):
        self.server.should_exit = True
        self.thread.join(10)


OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def post(url, body, timeout=30):
    request = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST",
                                     headers={"Content-Type": "application/json"})
    try:
        with OPENER.open(request, timeout=timeout) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


class LiveServiceTests(ServiceEnv):
    def test_a_flood_of_requests_is_bounded_and_the_service_stays_up(self):
        kit, state = similar_server.kit, similar_server.app.state
        original_gate = kit.gate
        kit.gate = ComputeGate("flood", max_concurrency=2, max_pending=3)
        release, worker_counts = threading.Event(), []

        def held_query(*args, **kwargs):          # admitted work waits until every refusal has come back
            worker_counts.append(sum(t.name.startswith("ivf-flood") for t in threading.enumerate()))
            release.wait(30)
            return state.real_query(*args, **kwargs)

        try:
            with LiveServer(similar_server.app) as live:
                state.real_query, state.cohort.query = state.cohort.query, held_query
                with ThreadPoolExecutor(max_workers=40) as pool:
                    futures = [pool.submit(post, live.url + "/ivf/similar_patients", {"age": 30, "amh": 2.0})
                               for _ in range(40)]
                    deadline = time.time() + 30
                    while sum(f.done() for f in futures) < 35 and time.time() < deadline:
                        time.sleep(0.02)
                    with OPENER.open(live.url + "/health", timeout=5) as response:
                        health = json.loads(response.read())
                    release.set()
                    results = [f.result() for f in futures]
                statuses = [s for s, _ in results]
                self.assertEqual((statuses.count(200), statuses.count(429)), (5, 35))   # 2 running + 3 waiting
                self.assertTrue(all(b["error"] == "busy" for s, b in results if s == 429))
                self.assertEqual((health["status"], health["computations_in_flight"]), ("ok", 5))
                self.assertLessEqual(max(worker_counts), 2)        # never more worker threads than the cap
                del state.cohort.query
                self.assertEqual(post(live.url + "/ivf/similar_patients", {"age": 30, "amh": 2.0})[0], 200)
        finally:
            release.set()
            kit.gate.shutdown()
            kit.gate = original_gate

    def test_startup_failures_are_logged(self):
        saved = os.environ["IVF_TRAIN_CSV"]
        os.environ["IVF_TRAIN_CSV"] = str(Path(self.tmp.name) / "missing.csv")
        try:
            with self.assertLogs("ivf.ivf_similar", level="ERROR") as logs, self.assertRaises(FileNotFoundError):
                with TestClient(similar_server.app):
                    pass
        finally:
            os.environ["IVF_TRAIN_CSV"] = saved
        self.assertIn("failed to start: could not load training data", "\n".join(logs.output))

    def test_mcp_bridge_lists_and_calls_both_tools(self):
        with LiveServer(similar_server.app) as similar, LiveServer(predict_server.app) as predict, \
                tempfile.TemporaryDirectory() as tmp:
            specs.write_specs(Path(tmp) / "tools", {"ivf_similar_patients": similar.url, "ivf_predict_outcome": predict.url})
            env = {**os.environ, "OMNI_TOOL_SPECS_DIR": str(Path(tmp) / "tools"),
                   "OMNI_TRACE_DIR": str(Path(tmp) / "trace"), "OMNI_MCP_SERVER_NAME": "ivf-clinical"}
            messages = [
                {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2025-06-18"}},
                {"jsonrpc": "2.0", "method": "notifications/initialized"},
                {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
                {"jsonrpc": "2.0", "id": 3, "method": "tools/call",
                 "params": {"name": "ivf_similar_patients", "arguments": {"age": 44, "amh": 0.15}}},
                {"jsonrpc": "2.0", "id": 4, "method": "tools/call",
                 "params": {"name": "ivf_predict_outcome", "arguments": {"age": 38, "amh": 0.9, "gn_days": 10}}},
                {"jsonrpc": "2.0", "id": 5, "method": "tools/call",
                 "params": {"name": "ivf_predict_outcome", "arguments": {"age": 80}}},
            ]
            out = subprocess.run([sys.executable, str(TASK / "runtime" / "claw_tools_mcp.py")], env=env, timeout=60,
                                 input="".join(json.dumps(m) + "\n" for m in messages), capture_output=True,
                                 encoding="utf-8")   # the bridge speaks UTF-8; Windows defaults to GBK
            replies = {r["id"]: r for r in map(json.loads, out.stdout.splitlines())}
            self.assertEqual([t["name"] for t in replies[2]["result"]["tools"]],
                             ["ivf_predict_outcome", "ivf_similar_patients"])
            self.assertTrue(json.loads(replies[3]["result"]["content"][0]["text"])["拒绝统计"])
            self.assertIn("妊娠概率", json.loads(replies[4]["result"]["content"][0]["text"]))
            self.assertTrue(replies[5]["result"]["isError"])
            self.assertIn("参数校验失败", replies[5]["result"]["content"][0]["text"])
            trace = [json.loads(line) for line in (Path(tmp) / "trace" / "dispatches.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual([t["tool_name"] for t in trace],
                             ["ivf_similar_patients", "ivf_predict_outcome", "ivf_predict_outcome"])


class SpecTests(unittest.TestCase):
    def test_committed_specs_and_prompt_match_the_request_models(self):
        for spec in specs.tool_specs():
            committed = json.loads((TASK / "tools" / f"{spec['name']}.json").read_text(encoding="utf-8"))
            self.assertEqual(committed, spec)
            for key in committed["input_schema"]["properties"]:
                self.assertRegex(key, r"^[a-zA-Z0-9_.-]{1,64}$")   # Claude Code drops tools with other names
        ports = {s["name"]: s["backend"]["url"].split(":")[2].split("/")[0] for s in specs.tool_specs()}
        self.assertEqual(ports, {"ivf_similar_patients": "9201", "ivf_predict_outcome": "9202"})
        today = json.loads((ROOT / "task.json").read_text(encoding="utf-8"))["runtime"]["mock_today"]
        self.assertEqual((TASK / "runtime" / "system_prompt_prefix.txt").read_text(encoding="utf-8"),
                         specs.system_prompt_prefix(today))


if __name__ == "__main__":
    unittest.main()
