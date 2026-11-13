"""ovu.ml: config-driven encoding, the softmax model, metrics, explanations and the predictor (synthetic data)."""

from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from ovu.ml import train as training  # noqa: E402
from ovu.ml.features import encode, fit_encoder, selected_features  # noqa: E402
from ovu.ml.model import fit_softmax, metrics, softmax  # noqa: E402
from ovu.ml.predict import Predictor  # noqa: E402
from ovu.pipeline import clean as cleaning  # noqa: E402
from ovu.pipeline import split  # noqa: E402
from test_ovu_pipeline import synthetic  # noqa: E402

DATA_CONFIG = ROOT / "task" / "services" / "ovu" / "config" / "data.yaml"
MODEL_CONFIG = ROOT / "task" / "services" / "ovu" / "config" / "model.yaml"


def prepared(tmp: Path, n: int = 3000) -> Path:
    config = cleaning.load_config(DATA_CONFIG)
    data, _ = cleaning.clean(synthetic(n), config)
    split.write_split(data, split.assign(data, config), tmp)
    return tmp


class EncodingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = yaml.safe_load(MODEL_CONFIG.read_text(encoding="utf-8"))
        self.stages = cleaning.feature_stages(cleaning.load_config(DATA_CONFIG))

    def test_only_allowed_stages_enter_the_model(self) -> None:
        self.config["features"]["numeric"]["gn_total_dose"] = {"transform": "standard"}
        self.assertNotIn("gn_total_dose", selected_features(self.config, self.stages)["numeric"])
        self.config["stages"].append("during")
        self.assertIn("gn_total_dose", selected_features(self.config, self.stages)["numeric"])
        self.config["features"]["binary"].append("no_such_field")      # removed from data.yaml: skipped, not an error
        self.assertNotIn("no_such_field", selected_features(self.config, self.stages)["binary"])

    def test_missing_unseen_and_rare_values(self) -> None:
        config = {"stages": ["baseline", "plan"], "features": {
            "numeric": {"amh": {"transform": "log", "missing_indicator": True}, "age": {"transform": "standard"}},
            "categorical": {"protocol": {"min_count": 2}}, "binary": ["dx_pcos"]}}
        rows = [{"age": 30, "amh": 2.0, "protocol": "A", "dx_pcos": 1}, {"age": 40, "amh": "", "protocol": "A", "dx_pcos": 0},
                {"age": 35, "amh": 4.0, "protocol": "B", "dx_pcos": 0}]
        encoder = fit_encoder(rows, config, self.stages)
        self.assertEqual(encoder["categorical"]["protocol"]["levels"], ["A", "其他"])       # B is rare
        self.assertEqual([c["column"] for c in encoder["columns"]], ["amh", "amh（缺失）", "age", "protocol=A", "protocol=其他", "dx_pcos"])
        x = encode(rows + [{"age": None, "protocol": "never seen"}], encoder)
        self.assertEqual(x[:, 1].tolist(), [0, 1, 0, 1])                                  # AMH missing flag
        self.assertEqual(x[1, 0], x[3, 0])                                                # both imputed with the median
        self.assertEqual(x[:, 4].tolist(), [0, 0, 1, 1])                                  # rare and unseen -> 其他
        self.assertAlmostEqual(x[3, 2], (35 - 35) / np.std([30, 40, 35]))                 # missing age -> median
        self.assertEqual(json.loads(json.dumps(encoder)), encoder)                        # plain JSON


class ModelTest(unittest.TestCase):
    def test_softmax_recovers_a_known_relation(self) -> None:
        rng = np.random.default_rng(0)
        x = rng.normal(size=(4000, 3))
        y = np.digitize(x[:, 0] + 0.3 * rng.normal(size=4000), [-0.8, 0.0, 0.8])
        weights, bias, fit = fit_softmax(x, y, 4, l2=1.0)
        self.assertTrue(fit["converged"])
        p = softmax(x @ weights + bias)
        m = metrics(p, y, np.array([1.0, 2.0, 3.0, 4.0]))
        self.assertGreater(m["quadratic_kappa"], 0.8)
        self.assertLess(np.abs(weights[1:]).max(), 0.3 * np.abs(weights[0]).max())       # noise features stay small

    def test_metrics_at_the_extremes(self) -> None:
        truth = np.array([0, 1, 2, 2])
        perfect = metrics(np.eye(3)[truth], truth, np.array([1.0, 5.0, 9.0]), np.array([1, 5, 9, 9]))
        self.assertEqual((perfect["accuracy"], perfect["quadratic_kappa"], perfect["rps"], perfect["mae_count"]), (1.0, 1.0, 0.0, 0.0))
        far = metrics(np.eye(3)[[2, 2, 0, 0]], truth, np.array([1.0, 5.0, 9.0]))
        near = metrics(np.eye(3)[[1, 2, 1, 1]], truth, np.array([1.0, 5.0, 9.0]))
        self.assertEqual(far["accuracy"], near["accuracy"])                               # both always wrong
        self.assertLess(near["rps"], far["rps"])                                          # but near misses cost less
        self.assertEqual(near["within_one"], 1.0)


class TrainAndPredictTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = TemporaryDirectory()
        cls.data = prepared(Path(cls.tmp.name))
        done = subprocess.run([sys.executable, "-m", "ovu.ml.train", "--data", str(cls.data)], cwd=ROOT, capture_output=True, text=True)
        assert done.returncode == 0, done.stderr
        cls.artifact = json.loads((cls.data / "model" / "model.json").read_text(encoding="utf-8"))
        cls.report = json.loads((cls.data / "model" / "model_report.json").read_text(encoding="utf-8"))

    @classmethod
    def tearDownClass(cls) -> None:
        cls.tmp.cleanup()

    def test_model_beats_the_baseline_and_finds_the_real_drivers(self) -> None:
        val = self.report["val"]
        self.assertLess(val["model"]["rps"], val["baseline"]["rps"])
        self.assertEqual(val["baseline"]["quadratic_kappa"], 0.0)
        self.assertEqual({i["feature"] for i in self.report["permutation_importance"][:2]}, {"age", "amh"})
        self.assertTrue(self.artifact["training"]["converged"])
        self.assertIn("## 特征重要性", (self.data / "model" / "model_report.md").read_text(encoding="utf-8"))

    def test_model_file_holds_no_patient_rows_and_no_during_features(self) -> None:
        self.assertLess((self.data / "model" / "model.json").stat().st_size, 60_000)
        names = set(self.artifact["encoder"]["numeric"]) | set(self.artifact["encoder"]["categorical"])
        self.assertFalse(names & {"gn_total_dose", "gn_days", "gnrha_days", "gnrha_total_dose"})
        self.assertNotIn("diagnosis_text", json.dumps(self.artifact, ensure_ascii=False))
        by_name = {i["name"]: i for i in self.artifact["inputs"]}
        self.assertEqual(list(by_name), list(Predictor(self.artifact).inputs))
        self.assertEqual(by_name["amh"]["kind"], "number")
        self.assertIn("拮抗剂方案", by_name["protocol"]["levels"])
        self.assertEqual(by_name["dx_pcos"]["kind"], "binary")

    def test_fold_models_never_saw_their_fold(self) -> None:
        for fold in range(5):
            part = json.loads((self.data / "model" / f"model_fold{fold}.json").read_text(encoding="utf-8"))
            rows = training.read_rows(self.data / "train.csv")
            self.assertEqual(part["training"]["n"], sum(r["cv_fold"] != str(fold) for r in rows))
            self.assertEqual(part["training"]["hidden_fold"], fold)

    def test_predictor_matches_training_and_explains(self) -> None:
        predictor = Predictor(self.artifact)
        rows = training.read_rows(self.data / "val.csv")[:50]
        np.testing.assert_allclose(predictor.probabilities(rows), training._probabilities(rows, self.artifact))
        young = predictor.predict(age=26, amh=6.0)
        old = predictor.predict(age=43, amh=0.5, protocol="拮抗剂方案")
        self.assertAlmostEqual(sum(young["各分组概率"].values()), 1.0, places=2)
        self.assertGreater(young["预期获卵数"], old["预期获卵数"])
        self.assertLess({e["输入"]: e["预期获卵数变化"] for e in old["各输入的影响"]}["age"], 0)
        self.assertIn("fsh", old["未提供的输入"])
        with self.assertRaises(ValueError):
            predictor.predict(gn_total_dose=3000)                                         # not a model input
        with self.assertRaises(ValueError):
            predictor.predict(age=None)

    def test_training_never_reads_the_sealed_test_set(self) -> None:
        source = (ROOT / "task" / "services" / "ovu" / "ml" / "train.py").read_text(encoding="utf-8")
        self.assertNotIn("test.csv", source.split('"""', 2)[2])
        self.assertNotIn("SEALED", source)


if __name__ == "__main__":
    unittest.main()
