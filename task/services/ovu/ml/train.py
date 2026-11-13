"""Train and evaluate the oocyte-count group model, and explain it.

    python -m ovu.ml.train [--data outputs/ovu] [--config ovu/config/model.yaml]
                           [--data-config ovu/config/data.yaml] [--out outputs/ovu/model]

Reads train.csv and val.csv written by ``python -m ovu.pipeline``; never opens
the sealed test set. Writes to --out:
    model.json          coefficients and encoding statistics only (no patient rows)
    model_fold<k>.json  the same model trained without fold k (for agent training on training cases)
    model_report.md     metrics on train / val against the "always predict the training
    model_report.json   distribution" baseline, permutation importance, coefficients
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from ovu.inputs import describe
from ovu.ml.features import column_names, encode, feature_names, fit_encoder
from ovu.ml.model import fit_softmax, metrics, softmax
from ovu.ml.predict import ARTIFACT_VERSION
from ovu.pipeline.clean import bin_labels, feature_stages, load_config

ROOT = Path(__file__).resolve().parents[1]


def read_rows(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def labels(rows: list[dict[str, Any]], column: str, bins: list[int]) -> tuple[np.ndarray, np.ndarray]:
    counts = np.array([int(float(r[column])) for r in rows])
    return counts, np.searchsorted(np.asarray(bins), counts, side="left")


def train(train_rows: list[dict[str, Any]], config: dict[str, Any], data_config: dict[str, Any], bins: list[int]) -> dict[str, Any]:
    stages = feature_stages(data_config)
    counts, y = labels(train_rows, config["label"]["column"], bins)
    groups = bin_labels(bins)
    present = np.bincount(y, minlength=len(groups))
    if (present == 0).any():
        raise ValueError(f"训练集中这些分组没有病例：{[g for g, c in zip(groups, present) if c == 0]}")
    encoder = fit_encoder(train_rows, config, stages)
    options = config.get("model", {})
    weights, bias, fit = fit_softmax(encode(train_rows, encoder), y, len(groups), l2=float(options.get("l2", 1.0)),
                                     max_iterations=int(options.get("max_iterations", 3000)))
    return {"artifact_version": ARTIFACT_VERSION, "model": "softmax", "label": config["label"]["column"], "bins": bins,
            "groups": groups, "group_means": [round(float(counts[y == k].mean()), 4) for k in range(len(groups))],
            "group_share": [round(float(s), 4) for s in present / present.sum()],
            "stages": list(config["stages"]), "encoder": encoder,
            "inputs": describe(feature_names(encoder), data_config, {n: c["levels"] for n, c in encoder["categorical"].items()}),
            "weights": weights.round(8).tolist(), "intercepts": bias.round(8).tolist(),
            "training": {"n": len(train_rows), "l2": float(options.get("l2", 1.0)), **fit,
                         "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}}


def _probabilities(rows: list[dict[str, Any]], artifact: dict[str, Any]) -> np.ndarray:
    return softmax(encode(rows, artifact["encoder"]) @ np.array(artifact["weights"]) + np.array(artifact["intercepts"]))


def evaluate(rows: list[dict[str, Any]], artifact: dict[str, Any]) -> dict[str, Any]:
    counts, y = labels(rows, artifact["label"], artifact["bins"])
    means = np.array(artifact["group_means"])
    baseline = np.tile(np.array(artifact["group_share"]), (len(rows), 1))
    return {"model": metrics(_probabilities(rows, artifact), y, means, counts), "baseline": metrics(baseline, y, means, counts)}


def permutation_importance(rows: list[dict[str, Any]], artifact: dict[str, Any], repeats: int, seed: int) -> list[dict[str, Any]]:
    """How much worse the model gets on ``rows`` when one feature's values are shuffled
    between patients (all of its columns together). Larger = the model relies on it more."""
    counts, y = labels(rows, artifact["label"], artifact["bins"])
    means = np.array(artifact["group_means"])
    x = encode(rows, artifact["encoder"])
    weights, bias = np.array(artifact["weights"]), np.array(artifact["intercepts"])
    base = metrics(softmax(x @ weights + bias), y, means, counts)
    owner = [c["feature"] for c in column_names(artifact["encoder"])]
    rng = np.random.default_rng(seed)
    out = []
    for feature in feature_names(artifact["encoder"]):
        columns = [i for i, f in enumerate(owner) if f == feature]
        deltas = {"rps": [], "log_loss": [], "mae_count": []}
        for _ in range(repeats):
            shuffled = x.copy()
            shuffled[:, columns] = x[rng.permutation(len(x))][:, columns]
            m = metrics(softmax(shuffled @ weights + bias), y, means, counts)
            for key in deltas:
                deltas[key].append(m[key] - base[key])
        out.append({"feature": feature, **{f"{key}_increase": round(float(np.mean(v)), 5) for key, v in deltas.items()},
                    "rps_increase_sd": round(float(np.std(deltas["rps"])), 5)})
    return sorted(out, key=lambda item: -item["rps_increase"])


def coefficient_table(artifact: dict[str, Any]) -> list[dict[str, Any]]:
    """Per column: the change in the expected count, at the average patient, for one
    standard deviation more (numeric) or for having it (0/1 and category columns)."""
    weights, bias, means = np.array(artifact["weights"]), np.array(artifact["intercepts"]), np.array(artifact["group_means"])
    reference = float(softmax(bias[None, :])[0] @ means)
    table = []
    for i, column in enumerate(column_names(artifact["encoder"])):
        shifted = float(softmax((weights[i] + bias)[None, :])[0] @ means)
        table.append({**column, "expected_count_change": round(shifted - reference, 3)})
    return sorted(table, key=lambda item: -abs(item["expected_count_change"]))


def to_markdown(report: dict[str, Any], artifact: dict[str, Any]) -> str:
    groups = artifact["groups"]
    names = {"accuracy": "准确率", "within_one": "相邻组以内的准确率", "quadratic_kappa": "加权 kappa（二次）",
             "rps": "RPS（越低越好）", "log_loss": "对数损失（越低越好）", "mae_count": "预期获卵数的平均绝对误差"}
    md = ["# 获卵数分组模型报告", "", f"分组：{'、'.join(groups)}。使用的预测时点：{'、'.join(artifact['stages'])}。"
          f"训练 {artifact['training']['n']} 例，迭代 {artifact['training']['iterations']} 次"
          f"（{'已收敛' if artifact['training']['converged'] else '未收敛，可调大 max_iterations'}）。", "",
          "基线 = 对每位患者都给出训练集的分组占比。测试集没有使用。", "",
          "## 指标", "", "| 指标 | 训练集：模型 | 训练集：基线 | 验证集：模型 | 验证集：基线 |", "|---|---|---|---|---|"]
    for key, name in names.items():
        md.append(f"| {name} | " + " | ".join(str(report[part][who][key]) for part in ("train", "val") for who in ("model", "baseline")) + " |")
    md += ["", "## 验证集混淆矩阵（行 = 真实分组，列 = 预测分组）", "", "| | " + " | ".join(groups) + " |", "|---|" + "---|" * len(groups)]
    md += [f"| {g} | " + " | ".join(str(c) for c in row) + " |" for g, row in zip(groups, report["val"]["model"]["confusion"])]
    md += ["", "## 特征重要性（置换重要性，验证集）", "",
           "把某个特征在患者之间打乱后，模型变差多少。数值越大，模型越依赖这个特征；接近 0 或为负说明没有用上。", "",
           "| 特征 | RPS 上升 | ± | 对数损失上升 | 预期获卵数误差上升 |", "|---|---|---|---|---|"]
    md += [f"| {i['feature']} | {i['rps_increase']:+.5f} | {i['rps_increase_sd']:.5f} | {i['log_loss_increase']:+.5f} | {i['mae_count_increase']:+.4f} |"
           for i in report["permutation_importance"]]
    md += ["", "## 各列对预期获卵数的影响（模型系数）", "",
           "在平均患者的基础上：数值特征增加一个标准差，或 0/1、类别列取 1 时，预期获卵数的变化。是关联，不是因果。", "",
           "| 列 | 所属特征 | 预期获卵数变化 |", "|---|---|---|"]
    md += [f"| {c['column']} | {c['feature']} | {c['expected_count_change']:+.3f} |" for c in report["coefficients"]]
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

    artifact = train(train_rows, config, data_config, bins)
    explain = config.get("explain", {})
    report = {"train": evaluate(train_rows, artifact), "val": evaluate(val_rows, artifact),
              "permutation_importance": permutation_importance(val_rows, artifact, int(explain.get("permutation_repeats", 5)),
                                                               int(explain.get("seed", 42))),
              "coefficients": coefficient_table(artifact)}
    artifact["training"]["val"] = {k: v for k, v in report["val"]["model"].items() if k != "confusion"}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "model.json").write_text(json.dumps(artifact, ensure_ascii=False), encoding="utf-8")
    # One model per fold, trained without that fold: when an agent is trained on a training case,
    # the tool answers with the model that never saw the case (and its fold).
    folds = sorted({r["cv_fold"] for r in train_rows if r.get("cv_fold", "") != ""})
    for fold in folds:
        part = train([r for r in train_rows if r["cv_fold"] != fold], config, data_config, bins)
        part["training"]["hidden_fold"] = int(fold)
        (out_dir / f"model_fold{fold}.json").write_text(json.dumps(part, ensure_ascii=False), encoding="utf-8")
    (out_dir / "model_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    (out_dir / "model_report.md").write_text(to_markdown(report, artifact), encoding="utf-8")
    v = report["val"]
    print(f"训练 {len(train_rows)} 例，验证 {len(val_rows)} 例；验证集 RPS {v['model']['rps']}（基线 {v['baseline']['rps']}），"
          f"加权 kappa {v['model']['quadratic_kappa']}")
    print(f"模型：{out_dir / 'model.json'}（另有 {len(folds)} 个分折模型）；报告：{out_dir / 'model_report.md'}")


if __name__ == "__main__":
    main()
