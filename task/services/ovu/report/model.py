"""Model performance and explanation as a Word document, built from the model reports only.

    python -m ovu.report.model [--model-dir outputs/ovu/model] [--out outputs/ovu/model/模型性能与可解释分析.docx]

Reads model_report.json (written by ``python -m ovu.ml.train``) and, when it exists,
test_report.json (written once by ``python -m ovu.ml.evaluate_test``). Both hold aggregates
only, so this step never opens patient rows. Without test_report.json the document reports
the validation set and the cross-validation and says that the test set has not been used yet.
Needs python-docx and matplotlib.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from docx import Document

from ovu.ml.evaluation import METRICS
from ovu.report.population import BLUE, GRID, INK, MUTED, Builder, _axes, _g, _n, _png, _pct, chart_font

LIGHT = "#a9b4c2"
TRANSFORMS = {"standard": "标准化", "log": "取 log(1+x) 后标准化"}


# --- figures ------------------------------------------------------------------------------------
def calibration_figure(groups: list[str], predicted: list[float], observed: list[float], labels: tuple[str, str, str]) -> io.BytesIO:
    fig, ax = _axes(6.2, 2.8)
    x = np.arange(len(groups))
    ax.bar(x - 0.19, predicted, width=0.36, color=BLUE, linewidth=0, label=labels[0])
    ax.bar(x + 0.19, observed, width=0.36, color=LIGHT, linewidth=0, label=labels[1])
    ax.set_xticks(x)
    ax.set_xticklabels(groups, color=INK, fontsize=8.5)
    ax.yaxis.grid(True, color=GRID, linewidth=0.6)
    ax.yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_xlabel(labels[2], color=INK, fontsize=9)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    return _png(fig)


def importance_figure(names: list[str], values: list[float], errors: list[float], xlabel: str) -> io.BytesIO:
    fig, ax = _axes(6.2, 0.32 * len(names) + 0.9)
    positions = list(range(len(names)))[::-1]
    ax.barh(positions, values, xerr=errors, height=0.6, color=BLUE, linewidth=0, error_kw={"ecolor": MUTED, "lw": 0.8, "capsize": 2})
    ax.axvline(0, color=MUTED, linewidth=0.8)
    ax.set_yticks(positions)
    ax.set_yticklabels(names, color=INK, fontsize=8.5)
    ax.xaxis.grid(True, color=GRID, linewidth=0.6)
    ax.spines["bottom"].set_visible(False)
    ax.set_xlabel(xlabel, color=INK, fontsize=9)
    return _png(fig)


def dependence_figure(panels: list[tuple[str, list[float], list[float]]], ylabel: str) -> io.BytesIO:
    columns = 3
    rows = (len(panels) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(6.4, 2.0 * rows + 0.2), dpi=200, squeeze=False, sharey=True)
    for ax, (title, xs, ys) in zip(axes.flat, panels):
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.spines["left"].set_color(MUTED)
        ax.spines["bottom"].set_color(MUTED)
        ax.tick_params(colors=MUTED, length=0, labelsize=7)
        ax.plot(xs, ys, color=BLUE, linewidth=1.6, marker="o", markersize=2.5)
        ax.yaxis.grid(True, color=GRID, linewidth=0.6)
        ax.set_title(title, color=INK, fontsize=8.5)
    values = [y for _, _, ys in panels for y in ys]
    low, high = min(values), max(values)
    if high - low < 4:                                              # a flat curve should look flat, not like a trend
        middle = (high + low) / 2
        low, high = middle - 2, middle + 2
    axes[0, 0].set_ylim(low - 0.05 * (high - low), high + 0.05 * (high - low))
    for ax in list(axes.flat)[len(panels):]:
        ax.set_visible(False)
    for ax in axes[:, 0]:
        ax.set_ylabel(ylabel, color=INK, fontsize=8)
    return _png(fig)


# --- content ------------------------------------------------------------------------------------
def _value(metric: dict[str, Any] | None, key: str) -> str:
    if metric is None or key not in metric:
        return "—"
    value = metric[key]
    return f"{value['mean']:.3f} ± {value['sd']:.3f}" if isinstance(value, dict) else f"{value:.3f}"


def _column_title(column: dict[str, str], titles: dict[str, str]) -> str:
    """protocol=拮抗剂方案 -> 用药方案：拮抗剂方案；amh（缺失） -> AMH（缺失）."""
    feature, name = column["feature"], column["column"]
    title = titles.get(feature, feature)
    if name.startswith(feature + "="):
        return f"{title}：{name[len(feature) + 1:]}"
    return title + name[len(feature):]


def _feature_rows(setup: dict[str, Any]) -> list[list[str]]:
    encoder, titles = setup["encoder"], setup["titles"]
    rows = []
    for name, spec in encoder["numeric"].items():
        how = TRANSFORMS.get(spec["transform"], spec["transform"]) + "；缺失用训练集中位数填补"
        if spec["missing_indicator"]:
            how += "，另加“是否缺失”标记"
        rows.append([titles.get(name, name), "数值", how, _pct(spec.get("missing_rate", 0))])
    for name, spec in encoder["categorical"].items():
        kept = [level for level in spec["levels"] if level != "其他"]
        rows.append([titles.get(name, name), "类别", f"独热编码，保留 {len(kept)} 个常见取值，其余与缺失归为“其他”", "—"])
    for name in encoder["binary"]:
        rows.append([titles.get(name, name), "0/1", "诊断类别，原样使用", "—"])
    return rows


def build(report: dict[str, Any], test: dict[str, Any] | None, cjk: bool, stale_test: bool = False) -> Document:
    def label(chinese: str, english: str) -> str:
        return chinese if cjk else english

    setup = report["setup"]
    groups, titles = setup["groups"], setup["titles"]
    cv = report.get("cv")
    final = test["test"] if test else report["val"]
    final_name = "测试集" if test else "验证集"
    b = Builder()
    b.title("获卵数分组模型：性能与可解释分析",
            f"获卵数预测项目 · 训练集 {_n(setup['n_train'])} 例，验证集 {_n(setup['n_val'])} 例"
            + (f"，测试集 {_n(test['record']['n'])} 例" if test else ""))

    b.heading("一、模型与数据")
    b.text(f"预测目标是获卵数所在的分组（{'、'.join(groups)}），模型输出每个分组的概率。模型为{setup['model']}，"
           f"在训练集上拟合；正则强度在验证集上选择（按 RPS 最低），最终 L2 = {setup['l2']:g}。"
           "测试集在划分时单独封存，不参与训练和任何选择。")
    if test:
        b.text(f"测试集只评估一次（{test['record']['evaluated_utc'][:10]}，第 {test['record']['times_used']} 次使用）。")
    b.table("输入特征及处理方式", ["特征", "类型", "处理方式", "训练集缺失率"], _feature_rows(setup), [3.6, 1.4, 8.0, 2.4],
            "特征及其处理方式由配置文件 model.yaml 决定；字段名称来自 data.yaml。")
    dropped = setup["encoder"].get("binary_dropped") or {}
    if dropped:
        b.text(f"以下 0/1 特征在 model.yaml 里列出，但训练集中为 1 的周期少于 {setup['encoder']['binary_min_positive']} 例，"
               "系数估不准，没有进入模型：" + "、".join(f"{titles.get(n, n)}（{_n(c)} 例）" for n, c in dropped.items()) + "。")
    b.table("正则强度选择（验证集）", ["L2", "验证集 RPS", "验证集对数损失", "是否收敛"],
            [[f"{s['l2']:g}", f"{s['val_rps']:.4f}", f"{s['val_log_loss']:.4f}", "是" if s["converged"] else "否"] for s in setup["l2_search"]],
            [3.0, 3.5, 3.5, 3.0])

    b.heading("二、模型性能")
    b.text("基线是不看任何特征、对每位患者都给出训练集分组占比的预测，模型至少要优于它才有意义。"
           "RPS（排序概率评分）同时考虑整个预测分布和分组的先后顺序，是主要指标；加权 kappa 衡量预测分组与真实分组的一致程度，"
           "相差越远扣分越多；预期获卵数 = 各分组概率 × 训练集中该分组的平均获卵数。")
    if not test:
        b.text(("测试集评估过的是之前的模型，当前模型尚未在测试集上评估" if stale_test else "测试集尚未评估")
               + "（python -m ovu.ml.evaluate_test），下表暂以验证集代替。")
    header = ["指标", f"{final_name}：模型", f"{final_name}：基线"] + (["验证集：模型"] if test else []) + (["5 折交叉验证：模型"] if cv else [])
    rows = []
    for key, (name, higher) in METRICS.items():
        row = [f"{name}（{'越高越好' if higher else '越低越好'}）", _value(final["model"], key), _value(final["baseline"], key)]
        if test:
            row.append(_value(report["val"]["model"], key))
        if cv:
            row.append(_value(cv["summary"]["model"], key))
        rows.append(row)
    widths = [5.2] + [(15.4 - 5.2) / (len(header) - 1)] * (len(header) - 1)
    b.table(f"模型性能（{final_name}）", header, rows, widths, "交叉验证一列为 5 折的均值 ± 标准差。" if cv else None)
    confusion = final["model"]["confusion"]
    b.table(f"{final_name}混淆矩阵（行为真实分组，列为预测分组）", ["真实 \\ 预测"] + groups + ["合计"],
            [[g] + [_n(c) for c in row] + [_n(sum(row))] for g, row in zip(groups, confusion)],
            [2.6] + [12.8 / (len(groups) + 1)] * (len(groups) + 1))
    cal = final["calibration"]
    b.table(f"各分组的平均预测概率与实际占比（{final_name}）", ["获卵数分组", "平均预测概率", "实际占比", "差值"],
            [[g, _pct(p), _pct(o), f"{(p - o) * 100:+.1f} 个百分点"] for g, p, o in zip(groups, cal["predicted"], cal["observed"])],
            [3.5, 3.5, 3.5, 3.5])
    b.figure(f"各分组的平均预测概率与实际占比（{final_name}）",
             calibration_figure(groups, cal["predicted"], cal["observed"],
                                (label("平均预测概率", "Mean predicted"), label("实际占比", "Observed"), label("获卵数分组", "Oocyte group"))))
    if cv:
        b.table("5 折交叉验证各折结果（训练集内）", ["折", "训练例数", "验证例数", "RPS", "基线 RPS", "加权 kappa", "准确率"],
                [[str(f["fold"] + 1), _n(f["n_train"]), _n(f["n_test"]), f"{f['model']['rps']:.4f}", f"{f['baseline']['rps']:.4f}",
                  f"{f['model']['quadratic_kappa']:.3f}", f"{f['model']['accuracy']:.3f}"] for f in cv["folds"]],
                [1.4, 2.3, 2.3, 2.3, 2.3, 2.4, 2.4], "每一折用其余 4 折训练、在这一折上评估，正则强度与最终模型相同。")

    b.heading("三、可解释分析")
    importance = (test or report)["permutation_importance"]
    b.text(f"置换重要性：把某个特征的取值在{final_name}患者之间随机打乱（其余特征不变），看 RPS 上升多少；"
           "上升越多，说明模型越依赖这个特征，接近 0 或为负说明模型没有用上它。每个特征打乱 5 次取平均。")
    top = importance[:15]
    b.table(f"特征重要性（置换重要性，{final_name}）", ["特征", "RPS 上升", "标准差", "预期获卵数误差上升"],
            [[titles.get(i["feature"], i["feature"]), f"{i['rps_increase']:+.5f}", f"{i['rps_increase_sd']:.5f}", f"{i['mae_count_increase']:+.3f}"]
             for i in top], [5.6, 3.2, 3.0, 3.6], f"按 RPS 上升从大到小，列出前 {len(top)} 个。" if len(importance) > len(top) else None)
    b.figure(f"特征重要性（置换重要性，{final_name}）",
             importance_figure([titles.get(i["feature"], i["feature"]).split("（")[0] if cjk else i["feature"] for i in top],
                               [i["rps_increase"] for i in top], [i["rps_increase_sd"] for i in top], label("RPS 上升", "Increase in RPS")))
    coefficients = report["coefficients"][:15]
    b.text("系数效应：以训练集的平均患者为基准，数值特征增加一个标准差，或某个 0/1、类别取值成立时，预期获卵数的变化。")
    b.table("各输入对预期获卵数的影响（模型系数）", ["输入", "预期获卵数变化"],
            [[_column_title(c, titles), f"{c['expected_count_change']:+.2f}"] for c in coefficients],
            [9.0, 4.0], f"按影响大小列出前 {len(coefficients)} 个；数值特征为增加一个标准差时的变化。")
    dependence = report["partial_dependence"]
    ranked = [i["feature"] for i in importance if dependence.get(i["feature"], {}).get("kind") == "numeric"][:6]
    if ranked:
        b.text("部分依赖：让训练集中每位患者的某一个特征都取同一个值（其余特征保持原样），计算平均预期获卵数，"
               "再在该特征 P5 到 P95 的范围内逐点重复，得到这一特征与预期获卵数的关系形状。")
        b.figure("主要数值特征的部分依赖（纵轴为平均预期获卵数）",
                 dependence_figure([(titles.get(f, f).split("（")[0] if cjk else f, dependence[f]["values"], dependence[f]["expected_count"])
                                    for f in ranked], label("预期获卵数", "Expected oocytes")))
    for feature, pd_ in dependence.items():
        if pd_["kind"] == "category":
            b.table(f"{titles.get(feature, feature)}的部分依赖", ["取值", "平均预期获卵数"] + groups,
                    [[str(v), f"{e:.2f}"] + [_pct(p) for p in probs] for v, e, probs in zip(pd_["values"], pd_["expected_count"], pd_["group_probabilities"])],
                    [3.6, 2.6] + [9.2 / len(groups)] * len(groups), "后几列为该取值下各分组的平均预测概率。")

    b.heading("四、说明")
    b.bullets(["以上影响和依赖关系都是模型学到的统计关联，不代表因果。",
               "模型文件只保存系数和编码所需的统计量（中位数、均值、标准差、类别取值），不含任何患者记录。",
               "周期号只用于划分数据集，不进入任何特征。",
               "Gn 总用量、Gn 总天数、GnRHa 总用量和总天数是促排结束后的累计量，默认不作为输入；需要时在 model.yaml 中加入。"])
    return b.doc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-dir", default="outputs/ovu/model")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    folder = Path(args.model_dir)
    report = json.loads((folder / "model_report.json").read_text(encoding="utf-8"))
    test_path = folder / "test_report.json"
    test = json.loads(test_path.read_text(encoding="utf-8")) if test_path.exists() else None
    model_path, stale = folder / "model.json", False
    if test and model_path.exists():
        digest = hashlib.sha256(model_path.read_bytes()).hexdigest()[:16]
        if test.get("record", {}).get("model_sha256") != digest:      # the model was retrained after the test evaluation
            print("提示：test_report.json 是旧模型的测试结果（模型文件已变），文档不使用它，性能部分改用验证集。"
                  "新模型要在测试集上评估，需要 python -m ovu.ml.evaluate_test --again（会记为又一次使用测试集）。")
            test, stale = None, True
    font = chart_font()
    if font:
        plt.rcParams["font.sans-serif"] = [font, "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False
    else:
        print("提示：这台机器上没有中文字体，图中的文字改用英文（表格和正文不受影响）")
    out = Path(args.out or folder / "模型性能与可解释分析.docx")
    build(report, test, cjk=font is not None, stale_test=stale).save(out)
    print(f"已生成：{out}" + ("" if test else "（测试集尚未评估，性能部分用的是验证集）"))


if __name__ == "__main__":
    main()
