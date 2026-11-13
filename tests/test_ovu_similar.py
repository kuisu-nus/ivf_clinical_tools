"""ovu.similar: weighted nearest-case retrieval set by similar.yaml (synthetic data)."""

from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from ovu.ml.train import read_rows  # noqa: E402
from ovu.pipeline import clean as cleaning  # noqa: E402
from ovu.similar.index import SimilarIndex, resolve  # noqa: E402
from test_ovu_ml import DATA_CONFIG, prepared  # noqa: E402

CONFIG = ROOT / "task" / "services" / "ovu" / "config" / "similar.yaml"


def case(age, amh, protocol, oocytes, fold=0, pcos=0):
    return {"age": age, "amh": amh, "protocol": protocol, "dx_pcos": pcos, "oocytes": oocytes, "cv_fold": fold}


SPEC = {"features": {"age": {"type": "numeric", "weight": 2.0, "transform": None, "scale": 2.0},
                     "amh": {"type": "numeric", "weight": 1.0, "transform": None, "scale": 1.0},
                     "protocol": {"type": "category", "weight": 1.0}, "dx_pcos": {"type": "binary", "weight": 1.0}},
        "cap": 3.0, "missing_distance": 1.0, "k_default": 3, "k_max": 5, "min_cohort": 2, "max_listed": 2,
        "listed_fields": ["age", "amh", "protocol"], "label": "oocytes", "bins": [3, 9], "groups": ["0–3", "4–9", "≥10"]}
ROWS = [case(30, 2.0, "A", 12), case(32, 2.0, "A", 8, fold=1), case(30, "", "B", 2), case(50, 9.0, "B", 20, pcos=1),
        case(31, 2.5, "A", 11, fold=1)]


class DistanceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.index = SimilarIndex(SPEC, ROWS)

    def test_weighted_mean_cap_and_missing(self) -> None:
        d = self.index.distances({"age": 30, "amh": 2.0, "protocol": "A"})
        self.assertEqual(d[0], 0.0)
        self.assertAlmostEqual(d[1], (2 * 1.0 + 0 + 0) / 4)                 # 2 years = 1 unit, weight 2
        self.assertAlmostEqual(d[2], (0 + 1.0 + 1.0) / 4)                   # AMH missing in the case, protocol differs
        self.assertAlmostEqual(d[3], (2 * 3.0 + 3.0 + 1.0) / 4)             # both numeric differences capped at 3
        only_age = self.index.distances({"age": 30})
        self.assertEqual(only_age[2], 0.0)                                  # features not in the query do not count

    def test_binary_values_compare_as_numbers(self) -> None:
        self.assertEqual(self.index.distances({"dx_pcos": "1"}).tolist(), [1, 1, 1, 0, 1])
        self.assertEqual(self.index.distances({"dx_pcos": 0.0}).tolist(), [0, 0, 0, 1, 0])

    def test_query_summarises_the_nearest_cases(self) -> None:
        result = self.index.query({"age": 30, "amh": 2.0, "protocol": "A"})
        self.assertEqual(result["获卵数分组"], {"0–3": {"例数": 0, "占比": 0.0}, "4–9": {"例数": 1, "占比": 0.333},
                                           "≥10": {"例数": 2, "占比": 0.667}})
        self.assertEqual(result["获卵数"]["中位数"], 11.0)
        self.assertEqual([p["获卵数"] for p in result["最相似的患者"]], [12, 11])
        self.assertEqual(set(result["最相似的患者"][0]), {"age", "amh", "protocol", "获卵数", "距离"})
        self.assertEqual(result["检索条件"]["未用于比对的特征"], ["dx_pcos"])

    def test_hidden_fold_never_comes_back(self) -> None:
        result = self.index.query({"age": 31, "amh": 2.5, "protocol": "A"}, k=5, hide_fold=1)
        self.assertEqual(result["检索条件"]["可比病例数"], 3)
        self.assertEqual(result["检索条件"]["取最相似的例数"], 3)
        self.assertNotIn(11, [p["获卵数"] for p in result["最相似的患者"]])

    def test_small_cohorts_and_bad_arguments(self) -> None:
        self.assertIn("不给统计", self.index.query({"age": 30}, k=1)["结果"])
        for bad in ({"gn_days": 10}, {}, {"age": None}):
            with self.assertRaises(ValueError):
                self.index.query(bad)
        with self.assertRaises(ValueError):
            self.index.query({"age": 30}, k=6)


class ConfigTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
        self.stages = cleaning.feature_stages(cleaning.load_config(DATA_CONFIG))

    def test_std_scale_comes_from_training_and_stages_are_enforced(self) -> None:
        data_config = cleaning.load_config(DATA_CONFIG)
        rows = [{"age": 30, "amh": 1.0, "protocol": "A", "oocytes": 5}, {"age": 40, "amh": 3.0, "protocol": "B", "oocytes": 9}]
        config = {"stages": ["baseline", "plan"], "features": {"age": {"type": "numeric", "weight": 1, "scale": "std"},
                                                               "protocol": {"type": "category", "weight": 1}}}
        spec = resolve(config, rows, data_config, [3, 9], ["0–3", "4–9", "≥10"])
        self.assertEqual(spec["features"]["age"]["scale"], 5.0)
        by_name = {i["name"]: i for i in spec["inputs"]}
        self.assertEqual((by_name["age"]["kind"], by_name["age"]["minimum"], by_name["age"]["maximum"]), ("integer", 18, 55))
        self.assertEqual(by_name["protocol"]["levels"], ["A", "B"])
        config["features"]["gn_total_dose"] = {"type": "numeric", "weight": 1}
        with self.assertRaises(ValueError):                                 # a during-stimulation field
            resolve(config, rows, data_config, [3, 9], [])
        config["features"] = {"nope": {"type": "numeric", "weight": 1}, "age": {"type": "numeric", "weight": 1}}
        self.assertEqual(list(resolve(config, rows, data_config, [3, 9], [])["features"]), ["age"])   # unknown field skipped

    def test_build_and_query_on_pipeline_output(self) -> None:
        with TemporaryDirectory() as tmp:
            data = prepared(Path(tmp), 2000)
            done = subprocess.run([sys.executable, "-m", "ovu.similar.build", "--data", str(data)], cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(done.returncode, 0, done.stderr)
            spec = json.loads((data / "similar" / "similar.json").read_text(encoding="utf-8"))
            rows = read_rows(data / "train.csv")
            self.assertEqual({r["cv_fold"] for r in rows}, {"0", "1", "2", "3", "4"})
            index = SimilarIndex(spec, rows)
            young = index.query({"age": 25, "amh": 6.0})
            old = index.query({"age": 44, "amh": 0.5})
            self.assertGreater(young["获卵数"]["中位数"], old["获卵数"]["中位数"])
            self.assertEqual(sum(g["例数"] for g in young["获卵数分组"].values()), spec["k_default"])
            self.assertNotIn("cv_fold", json.dumps(young, ensure_ascii=False))


if __name__ == "__main__":
    unittest.main()
