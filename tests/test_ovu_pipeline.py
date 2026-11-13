"""ovu.pipeline: config-driven cleaning, the sealed split and the aggregate report (synthetic data)."""

from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ovu.pipeline import clean as cleaning  # noqa: E402
from ovu.pipeline import split, stats  # noqa: E402

CONFIG = ROOT / "task" / "services" / "ovu" / "config" / "data.yaml"


def synthetic(n: int = 1200, seed: int = 0) -> pd.DataFrame:
    """A table with the real file's 26 column names and made-up values."""
    r = np.random.default_rng(seed)
    age = r.integers(21, 48, n)
    amh = np.where(r.random(n) < 0.4, np.nan, r.gamma(2, 1.6, n).round(2))
    oocytes = r.poisson(np.clip(22 - 0.35 * age + 1.5 * np.nan_to_num(amh, nan=3), 1, None))
    return pd.DataFrame({
        "周期号": 1_000_000 + np.arange(n), "女方年龄": age,
        "用药方案明细": r.choice(["拮抗剂方案", "长效长方案", "短效长方案", "罕见方案"], n, p=[0.5, 0.3, 0.19, 0.01]),
        "不孕年限": r.integers(0, 10, n),
        "诊断": r.choice(["原发不孕；多囊卵巢综合征", "继发不孕;输卵管因素", "男方少弱精症", "特殊情况"], n, p=[0.3, 0.4, 0.25, 0.05]),
        "内分泌FSH": r.normal(6, 1.5, n).round(2), "内分泌LH": r.normal(3.5, 1, n).round(2),
        "内分泌PRL": r.normal(15, 4, n).round(2), "内分泌E2": r.normal(35, 8, n).round(0),
        "内分泌T": r.normal(0.3, 0.05, n).round(2), "血压": [f"{a}/{b}" for a, b in zip(r.integers(95, 135, n), r.integers(60, 88, n))],
        "脉搏": r.integers(60, 100, n), "身高(cm)": r.integers(148, 172, n), "体重(kg)": r.integers(42, 75, n),
        "体重指数": r.normal(21.5, 2.5, n).round(2), "AMH": amh, "降调日当天窦卵泡数": r.choice([0, 8, 12, 16], n),
        "GnRha总用量": r.choice([0.8, 1.0, 1.5], n), "GnRha总天数": r.integers(1, 30, n), "GnRha开始量": r.choice([0.25, 0.8], n),
        "GnRha开始用药": r.choice(["达必佳", "达菲林", "加尼瑞克", "加尼瑞克（晴乐）"], n), "Gn总天数": r.integers(7, 14, n),
        "Gn总用量": r.integers(12, 40, n) * 75, "启动Gn用量": r.choice([150, 225, 300], n), "启动Gn时机": r.integers(2, 6, n),
        "卵子总数": oocytes,
    }).astype(str).replace("nan", np.nan)


class CleaningTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = cleaning.load_config(CONFIG)
        self.raw = synthetic()

    def test_zero_and_out_of_range_become_missing_without_dropping_rows(self) -> None:
        self.raw.loc[0, "内分泌FSH"] = "0"
        self.raw.loc[1, "女方年龄"] = "99"
        self.raw.loc[2, "AMH"] = "<0.01"
        data, log = cleaning.clean(self.raw, self.config)
        self.assertEqual(len(data), len(self.raw))
        self.assertEqual(log["fields"]["fsh"]["zero_as_missing"], 1)
        self.assertEqual(log["fields"]["age"]["out_of_range"], 1)
        self.assertEqual(log["fields"]["amh"]["non_numeric"], 1)
        self.assertEqual(int(data["afc_downreg"].eq(0).sum()), 0)

    def test_duplicate_ids_and_missing_labels_are_dropped(self) -> None:
        self.raw.loc[5, "周期号"] = self.raw.loc[4, "周期号"]
        self.raw.loc[9, "卵子总数"] = np.nan
        data, log = cleaning.clean(self.raw, self.config)
        self.assertEqual(log["rows_duplicate_id"], 2)
        self.assertEqual(len(data), len(self.raw) - 3)

    def test_fields_are_renamed_split_and_flagged(self) -> None:
        data, _ = cleaning.clean(self.raw, self.config)
        self.assertNotIn("周期号", data.columns)
        self.assertTrue(data["bp_systolic"].between(95, 135).all())
        self.assertTrue(data["bp_diastolic"].between(60, 88).all())
        pcos = data["diagnosis_text"].str.contains("多囊")
        self.assertTrue((data["dx_pcos"] == pcos.astype(int)).all())
        self.assertTrue(data.loc[data["diagnosis_text"].str.contains("少弱精"), "dx_male"].eq(1).all())
        self.assertTrue(data.loc[data["diagnosis_text"] == "特殊情况", "dx_other"].eq(1).all())
        self.assertEqual(int(data["dx_other"].sum()), int((data["diagnosis_text"] == "特殊情况").sum()))
        self.assertIn("其他", set(data["protocol"]))                  # the rare protocol
        self.assertNotIn("加尼瑞克（晴乐）", set(data["gnrha_drug"]))   # merged by `replace`

    def test_male_factor_means_semen_findings_not_male_genetics(self) -> None:
        self.raw.loc[0, "诊断"] = "继发不孕 男方染色体平衡易位46XYt(1;2)"
        self.raw.loc[1, "诊断"] = "原发不孕 畸形精子症 双方α地贫杂合子"
        data, _ = cleaning.clean(self.raw, self.config)
        self.assertEqual((int(data.loc[0, "dx_male"]), int(data.loc[0, "dx_genetic"]), int(data.loc[0, "dx_other"])), (0, 1, 0))
        self.assertEqual((int(data.loc[1, "dx_male"]), int(data.loc[1, "dx_genetic"])), (1, 1))

    def test_removing_a_field_from_the_config_removes_it_everywhere(self) -> None:
        import contextlib, io
        from ovu.db.database import CaseDatabase, resolve as resolve_db
        from ovu.ml.features import selected_features
        from ovu.similar.index import resolve as resolve_similar
        import yaml
        folder = CONFIG.parent
        for name in ("bmi", "protocol"):
            del self.config["columns"][name]
        del self.config["split_columns"]["bp_systolic"]
        data, _ = cleaning.clean(self.raw, self.config)
        self.assertFalse({"bmi", "protocol", "bp_systolic"} & set(data.columns))
        stages = cleaning.feature_stages(self.config)
        rows = data.drop(columns=[c for c in data.columns if c.startswith("_")]).assign(cv_fold=0).astype(object).to_dict("records")
        with contextlib.redirect_stderr(io.StringIO()) as notices:
            chosen = selected_features(yaml.safe_load((folder / "model.yaml").read_text(encoding="utf-8")), stages)
            similar = resolve_similar(yaml.safe_load((folder / "similar.yaml").read_text(encoding="utf-8")), rows, self.config,
                                      [3, 9, 15, 25], cleaning.bin_labels([3, 9, 15, 25]))
            database = resolve_db(yaml.safe_load((folder / "db.yaml").read_text(encoding="utf-8")), self.config, stages,
                                  cleaning.bin_labels([3, 9, 15, 25]), [3, 9, 15, 25])
        self.assertNotIn("bmi", chosen["numeric"])
        self.assertNotIn("protocol", chosen["categorical"])
        self.assertFalse({"bmi", "protocol"} & set(similar["features"]))
        self.assertFalse({"bmi", "protocol", "bp_systolic"} & {c["name"] for c in database["columns"]})
        self.assertIn("已跳过", notices.getvalue())
        self.assertEqual(CaseDatabase(database, rows).n, len(rows))

    def test_label_bins(self) -> None:
        self.assertEqual(cleaning.bin_labels([3, 9, 15, 25]), ["0–3", "4–9", "10–15", "16–25", "≥26"])
        values = pd.Series([0, 3, 4, 9, 10, 25, 26, 60])
        self.assertEqual(list(cleaning.bin_index(values, [3, 9, 15, 25])), [0, 0, 1, 1, 2, 3, 4, 4])

    def test_every_field_has_a_stage(self) -> None:
        stages = cleaning.feature_stages(self.config)
        self.assertEqual(stages["gn_total_dose"], "during")
        self.assertEqual(stages["gn_start_dose"], "plan")
        self.assertEqual(stages["dx_pcos"], "baseline")


class SplitTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = cleaning.load_config(CONFIG)
        self.data, self.log = cleaning.clean(synthetic(), self.config)

    def test_split_ignores_row_order_and_is_stratified(self) -> None:
        parts = split.assign(self.data, self.config)
        shuffled = self.data.sample(frac=1, random_state=1)
        again = split.assign(shuffled, self.config)
        self.assertTrue((parts == again.sort_index()).all())
        self.assertAlmostEqual((parts == "test").mean(), 0.2, delta=0.01)
        self.assertAlmostEqual((parts == "val").mean(), 0.1, delta=0.01)
        overall = self.data["oocytes_bin"].value_counts(normalize=True)
        in_test = self.data.loc[parts == "test", "oocytes_bin"].value_counts(normalize=True)
        self.assertLess((overall - in_test).abs().max(), 0.01)

    def test_sealed_test_set_survives_a_changed_config(self) -> None:
        with TemporaryDirectory() as tmp:
            out = Path(tmp)
            parts = split.assign(self.data, self.config)
            split.write_split(self.data, parts, out)
            sealed = split.read_sealed(out)
            self.config["label"]["bins"] = [5, 10, 20]                 # new bins -> new strata
            self.config["split"]["val"] = 0.2
            data, _ = cleaning.clean(synthetic(), self.config)
            again = split.assign(data, self.config, sealed)
            self.assertEqual(set(data.loc[again == "test", "_key"]), sealed)
            self.assertAlmostEqual((again == "val").mean(), 0.2, delta=0.01)
            written = pd.read_csv(out / "train.csv")
            self.assertFalse([c for c in written.columns if c.startswith("_")])

    def test_report_holds_aggregates_only(self) -> None:
        parts = split.assign(self.data, self.config)
        report = stats.build_report(self.data, parts, self.config, self.log)
        text = stats.to_markdown(report)
        self.assertIn("## 获卵数分布", text)
        self.assertLess(report["fields"]["age"]["spearman_with_label"], -0.1)
        self.assertEqual(sum(report["label"]["bin_count"]), len(self.data))
        self.assertIn("## 数据自检", text)
        age = report["fields"]["age"]["histogram"]
        self.assertTrue(age["unit"])
        self.assertEqual(sum(age["counts"]), len(self.data))
        self.assertEqual(len(age["edges"]), len(age["counts"]) + 1)
        self.assertEqual(sum(report["diagnosis"]["factors_per_cycle"].values()), len(self.data))
        self.assertEqual(report["fields"]["dx_male"]["title"], "男方因素（精液异常）")
        self.assertEqual({(i["a"], i["b"]) for i in report["sanity"]["order"]},
                         {("_row", "oocytes"), ("_row", "age"), ("_id_number", "oocytes"), ("_id_number", "age")})
        for key in self.data["_key"].head(20):
            self.assertNotIn(key, text)

    def test_command_line(self) -> None:
        with TemporaryDirectory() as tmp:
            raw = Path(tmp) / "raw.csv"
            synthetic(400).to_csv(raw, index=False, encoding="utf-8-sig")
            for expected in ("本次新封存", "沿用已封存的"):
                done = subprocess.run([sys.executable, "-m", "ovu.pipeline", str(raw), "--out", str(Path(tmp) / "out")],
                                      cwd=ROOT, capture_output=True, text=True)
                self.assertEqual(done.returncode, 0, done.stderr)
                self.assertIn(expected, done.stdout)
            self.assertTrue((Path(tmp) / "out" / "sealed" / "test.csv").exists())


if __name__ == "__main__":
    unittest.main()
