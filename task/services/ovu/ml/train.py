"""Train and evaluate the oocyte-count group model, and explain it.

    python -m ovu.ml.train [--data outputs/ovu] [--config ovu/config/model.yaml]
                           [--data-config ovu/config/data.yaml] [--out outputs/ovu/model]

Reads train.csv and val.csv written by ``python -m ovu.pipeline``; never opens
the sealed test set (that is ``python -m ovu.ml.evaluate_test``, run once at the end).

1. For each L2 strength in model.yaml, train on the training set and score on the
   validation set; keep the strength with the lowest validation RPS.
2. 5-fold cross-validation inside the training set: for each fold k, train without
   fold k and score on fold k.
3. Explain the final model: permutation importance on the validation set, the
   per-column effect at the average patient, and partial dependence of each feature.

Writes to --out:
    model.json          coefficients and encoding statistics only (no patient rows)
    model_fold<k>.json  the model trained without fold k (also used when training an agent on
                        training cases: the tool answers with a model that never saw the case)
    model_report.md     everything above, against the "always predict the training
    model_report.json   distribution" baseline; aggregates only
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from ovu.inputs import describe
from ovu.ml.evaluation import (METRICS, coefficient_table, evaluate, feature_titles, labels, partial_dependence,
                               permutation_importance, probabilities, read_rows, summarise_folds)
from ovu.ml.features import encode, feature_names, fit_encoder
from ovu.ml.model import fit_softmax
from ovu.ml.predict import ARTIFACT_VERSION
from ovu.pipeline.clean import available_fields, bin_labels, load_config

ROOT = Path(__file__).resolve().parents[1]


def train(train_rows: list[dict[str, Any]], config: dict[str, Any], data_config: dict[str, Any], bins: list[int],
          l2: float | None = None) -> dict[str, Any]:
    counts, y = labels(train_rows, config["label"]["column"], bins)
    groups = bin_labels(bins)
    present = np.bincount(y, minlength=len(groups))
    if (present == 0).any():
        raise ValueError(f"训练集中这些分组没有病例：{[g for g, c in zip(groups, present) if c == 0]}")
    encoder = fit_encoder(train_rows, config, available_fields(data_config))
    options = config.get("model", {})
    if l2 is None:
        l2 = l2_candidates(config)[0]
    weights, bias, fit = fit_softmax(encode(train_rows, encoder), y, len(groups), l2=float(l2),
                                     max_iterations=int(options.get("max_iterations", 3000)))
    return {"artifact_version": ARTIFACT_VERSION, "model": "softmax", "label": config["label"]["column"], "bins": bins,
            "groups": groups, "group_means": [round(float(counts[y == k].mean()), 4) for k in range(len(groups))],
            "group_share": [round(float(s), 4) for s in present / present.sum()], "encoder": encoder,
            "inputs": describe(feature_names(encoder), data_config, {n: c["levels"] for n, c in encoder["categorical"].items()}),
            "weights": weights.round(8).tolist(), "intercepts": bias.round(8).tolist(),
            "training": {"n": len(train_rows), "l2": float(l2), **fit,
                         "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}}


def l2_candidates(config: dict[str, Any]) -> list[float]:
    value = config.get("model", {}).get("l2", 1.0)
    return [float(v) for v in (value if isinstance(value, list) else [value])]


def _probabilities(rows: list[dict[str, Any]], artifact: dict[str, Any]) -> np.ndarray:
    return probabilities(rows, artifact)


def run(train_rows: list[dict[str, Any]], val_rows: list[dict[str, Any]], config: dict[str, Any], data_config: dict[str, Any],
        bins: list[int]) -> tuple[dict[str, Any], dict[int, dict[str, Any]], dict[str, Any]]:
    """Train, choose L2 on the validation set, cross-validate and explain. Returns the final
    model, the fold models and the report."""
    search = []
    for l2 in l2_candidates(config):
        candidate = train(train_rows, config, data_config, bins, l2)
        scored = evaluate(val_rows, candidate)["model"]
        search.append({"l2": l2, "val_rps": scored["rps"], "val_log_loss": scored["log_loss"],
                       "converged": candidate["training"]["converged"], "_artifact": candidate})
    best = min(search, key=lambda s: s["val_rps"])
    artifact = best["_artifact"]
    for s in search:
        del s["_artifact"]

    fold_models, folds = {}, []
    for fold in sorted({int(r["cv_fold"]) for r in train_rows if r.get("cv_fold", "") != ""}):
        inside = [r for r in train_rows if r["cv_fold"] != str(fold)]
        held_out = [r for r in train_rows if r["cv_fold"] == str(fold)]
        part = train(inside, config, data_config, bins, best["l2"])
        part["training"]["hidden_fold"] = fold
        fold_models[fold] = part
        scored = evaluate(held_out, part)
        folds.append({"fold": fold, "n_train": len(inside), "n_test": len(held_out),
                      "model": {k: v for k, v in scored["model"].items() if k != "confusion"},
                      "baseline": {k: v for k, v in scored["baseline"].items() if k != "confusion"}})

    explain = config.get("explain", {})
    features = feature_names(artifact["encoder"])
    report = {
        "setup": {"model": "多分类逻辑回归（softmax），L2 正则", "groups": artifact["groups"], "bins": bins,
                  "n_train": len(train_rows), "n_val": len(val_rows), "features": features,
                  "titles": feature_titles(features + list(artifact["encoder"].get("binary_dropped") or {}), data_config), "encoder": artifact["encoder"],
                  "l2_search": search, "l2": best["l2"], "iterations": artifact["training"]["iterations"],
                  "converged": artifact["training"]["converged"]},
        "train": evaluate(train_rows, artifact), "val": evaluate(val_rows, artifact),
        "cv": {"folds": folds, "summary": summarise_folds(folds)} if folds else None,
        "permutation_importance": permutation_importance(val_rows, artifact, int(explain.get("permutation_repeats", 5)),
                                                         int(explain.get("seed", 42))),
        "coefficients": coefficient_table(artifact),
        "partial_dependence": partial_dependence(train_rows, artifact, int(explain.get("grid_points", 15))),
    }
    artifact["training"]["val"] = {k: v for k, v in report["val"]["model"].items() if k != "confusion"}
    return artifact, fold_models, report


def _metric_rows(columns: list[tuple[str, dict[str, Any]]]) -> list[str]:
    md = ["| 指标 | " + " | ".join(name for name, _ in columns) + " |", "|---|" + "---|" * len(columns)]
    for key, (name, higher) in METRICS.items():
        md.append(f"| {name}（{'越高越好' if higher else '越低越好'}） | " + " | ".join(
            (f"{m[key]['mean']} ± {m[key]['sd']}" if isinstance(m.get(key), dict) else str(m.get(key, "—"))) for _, m in columns) + " |")
    return md


def to_markdown(report: dict[str, Any]) -> str:
    setup = report["setup"]
    groups, titles = setup["groups"], setup["titles"]
    md = ["# 获卵数分组模型报告", "",
          f"模型：{setup['model']}。分组：{'、'.join(groups)}。训练 {setup['n_train']} 例，验证 {setup['n_val']} 例；"
          f"L2 = {setup['l2']:g}（按验证集 RPS 选出），迭代 {setup['iterations']} 次"
          f"（{'已收敛' if setup['converged'] else '未收敛，可调大 max_iterations'}）。", "",
          "基线 = 对每位患者都给出训练集的分组占比。测试集没有使用（见 python -m ovu.ml.evaluate_test）。", "",
          "## 特征", "", "、".join(f"{titles[f]}（{f}）" for f in setup["features"]), "",
          *([f"训练集中为 1 的周期少于 {setup['encoder']['binary_min_positive']} 例、没有进入模型的 0/1 特征："
             + "、".join(f"{titles.get(n, n)}（{n}，{c} 例）" for n, c in setup["encoder"]["binary_dropped"].items()), ""]
            if setup["encoder"].get("binary_dropped") else []),
          "## L2 选择（验证集）", "", "| L2 | 验证集 RPS | 验证集对数损失 | 收敛 |", "|---|---|---|---|"]
    md += [f"| {s['l2']:g} | {s['val_rps']} | {s['val_log_loss']} | {'是' if s['converged'] else '否'} |" for s in setup["l2_search"]]
    columns = [("训练集：模型", report["train"]["model"]), ("训练集：基线", report["train"]["baseline"]),
               ("验证集：模型", report["val"]["model"]), ("验证集：基线", report["val"]["baseline"])]
    if report.get("cv"):
        columns += [("5 折交叉验证：模型", report["cv"]["summary"]["model"]), ("5 折交叉验证：基线", report["cv"]["summary"]["baseline"])]
    md += ["", "## 指标", ""] + _metric_rows(columns)
    md += ["", "## 验证集混淆矩阵（行 = 真实分组，列 = 预测分组）", "", "| | " + " | ".join(groups) + " |", "|---|" + "---|" * len(groups)]
    md += [f"| {g} | " + " | ".join(str(c) for c in row) + " |" for g, row in zip(groups, report["val"]["model"]["confusion"])]
    cal = report["val"]["calibration"]
    md += ["", "## 验证集校准（各分组的平均预测概率 vs 实际占比）", "", "| 分组 | 平均预测概率 | 实际占比 |", "|---|---|---|"]
    md += [f"| {g} | {p:.3f} | {o:.3f} |" for g, p, o in zip(groups, cal["predicted"], cal["observed"])]
    md += ["", "## 特征重要性（置换重要性，验证集）", "",
           "把某个特征在患者之间打乱后，模型变差多少。数值越大，模型越依赖这个特征；接近 0 或为负说明没有用上。", "",
           "| 特征 | RPS 上升 | ± | 对数损失上升 | 预期获卵数误差上升 |", "|---|---|---|---|---|"]
    md += [f"| {titles.get(i['feature'], i['feature'])} | {i['rps_increase']:+.5f} | {i['rps_increase_sd']:.5f} | "
           f"{i['log_loss_increase']:+.5f} | {i['mae_count_increase']:+.4f} |" for i in report["permutation_importance"]]
    md += ["", "## 各列对预期获卵数的影响（模型系数）", "",
           "在平均患者的基础上：数值特征增加一个标准差，或 0/1、类别列取 1 时，预期获卵数的变化。是关联，不是因果。", "",
           "| 列 | 所属特征 | 预期获卵数变化 |", "|---|---|---|"]
    md += [f"| {c['column']} | {titles.get(c['feature'], c['feature'])} | {c['expected_count_change']:+.3f} |" for c in report["coefficients"]]
    md += ["", "## 部分依赖（其他输入不变，只改这一项时的平均预期获卵数）", ""]
    for feature, pd_ in report["partial_dependence"].items():
        pairs = "，".join(f"{v}: {e}" for v, e in zip(pd_["values"], pd_["expected_count"]))
        md.append(f"- {titles.get(feature, feature)}：{pairs}")
    return "\n".join(md) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default="outputs/ovu")
    parser.add_argument("--config", default=str(ROOT / "config" / "model.yaml"))
    parser.add_argument("--data-config", default=str(ROOT / "config" / "data.yaml"))
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    data_config = load_config(args.data_config)
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    bins = config["label"].get("bins") or data_config["label"]["bins"]
    data_dir, out_dir = Path(args.data), Path(args.out or Path(args.data) / "model")
    train_rows, val_rows = read_rows(data_dir / "train.csv"), read_rows(data_dir / "val.csv")

    artifact, fold_models, report = run(train_rows, val_rows, config, data_config, bins)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "model.json").write_text(json.dumps(artifact, ensure_ascii=False), encoding="utf-8")
    for fold, part in fold_models.items():
        (out_dir / f"model_fold{fold}.json").write_text(json.dumps(part, ensure_ascii=False), encoding="utf-8")
    (out_dir / "model_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    (out_dir / "model_report.md").write_text(to_markdown(report), encoding="utf-8")
    v = report["val"]
    print(f"训练 {len(train_rows)} 例，验证 {len(val_rows)} 例；L2 = {report['setup']['l2']:g}；"
          f"验证集 RPS {v['model']['rps']}（基线 {v['baseline']['rps']}），加权 kappa {v['model']['quadratic_kappa']}")
    if report["cv"]:
        cv = report["cv"]["summary"]["model"]["rps"]
        print(f"{len(fold_models)} 折交叉验证 RPS {cv['mean']} ± {cv['sd']}")
    print(f"模型：{out_dir / 'model.json'}（另有 {len(fold_models)} 个分折模型）；报告：{out_dir / 'model_report.md'}")


if __name__ == "__main__":
    main()
