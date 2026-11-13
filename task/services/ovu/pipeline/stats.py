"""Statistics report for the cleaned data: aggregates only, never a patient row.

Descriptive statistics (field distributions, the label distribution) use every
cleaned cycle. Anything that relates a field to the label (correlations, label
by category) uses the training set only, so the sealed test set never informs
a modelling choice.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any

import numpy as np
import pandas as pd

from ovu.pipeline.clean import bin_labels, feature_stages

STAGE_NAMES = {"baseline": "进周期前", "plan": "启动方案", "during": "促排过程中"}
AGE_BANDS = [(0, 29, "<30"), (30, 34, "30–34"), (35, 37, "35–37"), (38, 40, "38–40"), (41, 99, "≥41")]


def _r(value: float, digits: int = 2) -> float:
    return round(float(value), digits)


def _numeric_summary(values: pd.Series) -> dict[str, Any]:
    v = values.dropna().astype(float)
    if v.empty:
        return {"n": 0}
    q = v.quantile([0.05, 0.25, 0.5, 0.75, 0.95])
    return {"n": int(len(v)), "mean": _r(v.mean()), "std": _r(v.std()), "p5": _r(q[0.05]), "p25": _r(q[0.25]),
            "median": _r(q[0.5]), "p75": _r(q[0.75]), "p95": _r(q[0.95])}


def _histogram(values: pd.Series, bins: int = 24) -> dict[str, Any]:
    """Counts over the central 99% of the values (the tails are added to the end bars).
    Whole-number fields with a short range get one bar per value."""
    v = values.dropna().astype(float)
    if v.empty:
        return {"edges": [], "counts": [], "unit": False}
    low, high = float(v.quantile(0.005)), float(v.quantile(0.995))
    unit = bool((v % 1 == 0).all()) and high - low <= 60
    if unit:
        edges = np.arange(math.floor(low) - 0.5, math.ceil(high) + 1.0, 1.0)
    else:
        edges = np.linspace(low, high if high > low else low + 1.0, bins + 1)
    counts, _ = np.histogram(v.clip(edges[0], edges[-1]), bins=edges)
    return {"edges": [round(float(e), 4) for e in edges], "counts": [int(c) for c in counts], "unit": unit}


def _spearman(table: pd.DataFrame, a: str, b: str) -> dict[str, Any]:
    """Spearman correlation as Pearson on ranks (no scipy needed)."""
    pair = table[[a, b]].dropna().astype(float)
    usable = len(pair) > 30 and pair[a].nunique() > 1 and pair[b].nunique() > 1
    return {"rho": _r(pair[a].rank().corr(pair[b].rank()), 3) if usable else None, "n": int(len(pair))}


def _counts(values: pd.Series, min_count: int) -> dict[str, int]:
    counts = values.dropna().astype(str).value_counts()
    shown = {k: int(c) for k, c in counts.items() if c >= min_count}
    hidden = int(counts[counts < min_count].sum())
    if hidden:
        shown[f"（少于 {min_count} 次的取值合计）"] = hidden
    return shown


def _label_by_group(train: pd.DataFrame, groups: pd.Series, label: str, n_bins: int, min_count: int) -> dict[str, Any]:
    out = {}
    for value, rows in train.groupby(groups, observed=True):
        if len(rows) < min_count:
            continue
        shares = rows[label + "_bin"].value_counts(normalize=True).reindex(range(n_bins), fill_value=0.0)
        out[str(value)] = {"n": int(len(rows)), "mean": _r(rows[label].mean()), "median": _r(rows[label].median()),
                           "bin_share": [_r(s, 3) for s in shares]}
    return out


def build_report(data: pd.DataFrame, parts: pd.Series, config: dict[str, Any], log: dict[str, Any]) -> dict[str, Any]:
    label = config["label"]["name"]
    bins = config["label"]["bins"]
    names = bin_labels(bins)
    min_count = config.get("report", {}).get("min_count", 10)
    train = data[parts == "train"]
    report: dict[str, Any] = {"rows": {k: log[k] for k in ("rows_raw", "rows_duplicate_id", "rows_label_missing", "rows_clean")},
                              "label_cleaning": log["label"], "bin_names": names}

    report["split"] = {}
    for part in ("train", "val", "test"):
        rows = data[parts == part]
        shares = rows[label + "_bin"].value_counts(normalize=True).reindex(range(len(names)), fill_value=0.0)
        report["split"][part] = {"n": int(len(rows)), "label_mean": _r(rows[label].mean()) if len(rows) else None,
                                 "bin_share": [_r(s, 3) for s in shares]}

    counts = data[label].value_counts().sort_index()
    report["label"] = {"summary": _numeric_summary(data[label]), "histogram": _histogram(data[label]),
                       "bin_count": [int(c) for c in data[label + "_bin"].value_counts().reindex(range(len(names)), fill_value=0)],
                       "value_count": {str(k): int(c) for k, c in counts.items() if c >= min_count},
                       "value_count_hidden": int(counts[counts < min_count].sum())}

    stages = feature_stages(config)
    specs = {**config.get("columns", {}), **config.get("split_columns", {})}
    report["fields"] = {}
    for name, stage in stages.items():
        spec = specs.get(name, {})
        kind = spec.get("type", "flag" if name not in specs else "float")
        entry: dict[str, Any] = {"source": spec.get("source", config.get("diagnosis", {}).get("source")), "stage": stage,
                                 "type": kind, "missing_rate": _r(data[name].isna().mean(), 4),
                                 "cleaning": log["fields"].get(name, {}), "note": spec.get("note", "")}
        if spec.get("name"):
            entry["title"] = spec["name"]
        if kind == "category":
            entry["counts"] = _counts(data[name], min_count)
            entry["label_by_value"] = _label_by_group(train, train[name], label, len(names), min_count)
        elif kind == "flag":
            entry["title"] = (config.get("diagnosis") or {}).get("names", {}).get(name, name)
            entry["positive"] = int(data[name].sum())
            entry["positive_rate"] = _r(data[name].mean(), 4)
            entry["label_by_value"] = _label_by_group(train, train[name], label, len(names), min_count)
        else:
            entry["summary"] = _numeric_summary(data[name])
            entry["histogram"] = _histogram(data[name])
            rho = _spearman(train, name, label)
            entry["spearman_with_label"], entry["spearman_n"] = rho["rho"], rho["n"]
        report["fields"][name] = entry

    report["sanity"] = {"pairs": [], "order": []}
    for a, b in config.get("report", {}).get("sanity_pairs", []):
        report["sanity"]["pairs"].append({"a": a, "b": b, **_spearman(data, a, b)})
    for a in ("_row", "_id_number"):
        for b in (label, "age"):
            if b in data.columns:
                report["sanity"]["order"].append({"a": a, "b": b, **_spearman(data, a, b)})

    if "age" in data.columns:
        ages = train["age"].astype(float)
        band = pd.Series(pd.NA, index=train.index, dtype=object)
        for low, high, name in AGE_BANDS:
            band[(ages >= low) & (ages <= high)] = name
        by_band = _label_by_group(train, band, label, len(names), min_count)
        report["label_by_age_band"] = {name: by_band[name] for _, _, name in AGE_BANDS if name in by_band}

    diagnosis = config.get("diagnosis")
    if diagnosis:
        splitter = "|".join([r"\s+"] + [re.escape(s) for s in diagnosis["separators"]])
        tokens: Counter[str] = Counter()
        for text in data["diagnosis_text"].drop_duplicates():       # one count per distinct diagnosis, not per cycle
            text = re.sub(r"[(（][^()（）]*[)）]", " ", text)           # karyotypes etc. inside brackets
            text = re.sub(r"[(（][^()（）]*[)）]", " ", text)           # one level of nesting
            tokens.update({t.strip() for t in re.split(splitter, text) if len(t.strip()) > 1})
        top = config.get("report", {}).get("top_tokens", 80)
        specific = [f for f in diagnosis["flags"] if f not in ("dx_primary", "dx_secondary")]
        per_cycle = data[specific].sum(axis=1).clip(upper=5).value_counts().sort_index()
        report["diagnosis"] = {"empty": log["diagnosis_empty"], "unmatched": log["diagnosis_unmatched"],
                               "factors_per_cycle": {("≥5" if k == 5 else str(int(k))): int(c) for k, c in per_cycle.items()},
                               "unmatched_rate": _r(log["diagnosis_unmatched"] / max(log["rows_raw"], 1), 4),
                               "distinct_tokens": len(tokens), "distinct_texts": int(data["diagnosis_text"].nunique()),
                               "repeated_text_rows": int(data["diagnosis_text"].duplicated(keep=False).sum()),
                               "top_tokens": {t: c for t, c in tokens.most_common(top) if c >= min_count}}
    return report


def _shares(shares: list[float]) -> str:
    return " | ".join(f"{s:.1%}" for s in shares)


def to_markdown(report: dict[str, Any]) -> str:
    names = report["bin_names"]
    rows = report["rows"]
    md = ["# 获卵数数据：清洗与统计报告", "",
          "本报告只含汇总统计，不含任何一行病例。字段与结局的关系（相关系数、分组均值）只用训练集计算。", "",
          "## 行数", "",
          f"原始 {rows['rows_raw']} 行；周期号重复去掉 {rows['rows_duplicate_id']} 行；"
          f"获卵数缺失或超范围 {rows['rows_label_missing']} 行；清洗后 {rows['rows_clean']} 行。", "",
          "## 划分", "", "| 集合 | 例数 | 平均获卵数 | " + " | ".join(names) + " |", "|---|---|---|" + "---|" * len(names)]
    for part, s in report["split"].items():
        md.append(f"| {part} | {s['n']} | {s['label_mean']} | {_shares(s['bin_share'])} |")
    md += ["", "数据里没有患者编号，只能按周期划分：同一位患者如果做过多个周期，可能分别落在训练集和测试集。"
           "测试集已封存（sealed/test_ids.txt），重跑不会变。"]

    def rho(item: dict[str, Any]) -> str:
        return "—" if item["rho"] is None else f"{item['rho']:+.3f}"

    sanity = report["sanity"]
    md += ["", "## 数据自检", "",
           "下面这些字段对在临床上必然相关。如果它们彼此相关，而所有字段都和获卵数无关，说明获卵数这一列没有和其他列对齐。", "",
           "| 字段对 | Spearman 相关 | 例数 |", "|---|---|---|"]
    md += [f"| {i['a']} × {i['b']} | {rho(i)} | {i['n']} |" for i in sanity["pairs"]]
    names_of = {"_row": "文件中的行号", "_id_number": "周期号"}
    md += ["", "行顺序和周期号（都不进入任何特征，这里只检查它们是否带有信息）：", "", "| 字段对 | Spearman 相关 | 例数 |", "|---|---|---|"]
    md += [f"| {names_of[i['a']]} × {i['b']} | {rho(i)} | {i['n']} |" for i in sanity["order"]]

    lab = report["label"]
    s = lab["summary"]
    md += ["", "## 获卵数分布", "",
           f"均值 {s['mean']}，标准差 {s['std']}，中位数 {s['median']}（P25–P75 {s['p25']}–{s['p75']}，P5–P95 {s['p5']}–{s['p95']}）。", "",
           "| 分组 | 例数 | 占比 |", "|---|---|---|"]
    total = sum(lab["bin_count"])
    md += [f"| {n} | {c} | {c / total:.1%} |" for n, c in zip(names, lab["bin_count"])]
    md += ["", "逐值例数：" + "，".join(f"{k}: {c}" for k, c in lab["value_count"].items())
           + (f"；其余取值合计 {lab['value_count_hidden']} 例" if lab["value_count_hidden"] else "")]

    md += ["", "## 数值字段", "",
           "| 字段 | 原始列 | 时点 | 缺失率 | 中位数（P25–P75） | P5–P95 | 与获卵数的 Spearman 相关 | 清洗 | 备注 |",
           "|---|---|---|---|---|---|---|---|---|"]
    numeric = {k: f for k, f in report["fields"].items() if "summary" in f}
    for name, f in sorted(numeric.items(), key=lambda kv: -abs(kv[1]["spearman_with_label"] or 0)):
        s = f["summary"]
        changed = "，".join(f"{label} {f['cleaning'][key]}" for key, label in
                           (("zero_as_missing", "0 记缺失"), ("out_of_range", "超范围"), ("non_numeric", "非数字"))
                           if f["cleaning"].get(key))
        dist = f"{s['median']}（{s['p25']}–{s['p75']}）| {s['p5']}–{s['p95']}" if s["n"] else "— | —"
        rho = "—" if f["spearman_with_label"] is None else f"{f['spearman_with_label']:+.3f}"
        md.append(f"| {name} | {f['source']} | {STAGE_NAMES[f['stage']]} | {f['missing_rate']:.1%} | {dist} | {rho} | {changed} | {f['note']} |")

    for name, f in report["fields"].items():
        if f["type"] != "category":
            continue
        md += ["", f"## {name}（{f['source']}，{STAGE_NAMES[f['stage']]}）", "",
               f"缺失率 {f['missing_rate']:.1%}。例数为全部数据，获卵数为训练集。", "",
               "| 取值 | 例数 | 训练集例数 | 平均获卵数 | 中位数 | " + " | ".join(names) + " |", "|---|---|---|---|---|" + "---|" * len(names)]
        for value, count in f["counts"].items():
            g = f["label_by_value"].get(value)
            md.append(f"| {value} | {count} | " + (f"{g['n']} | {g['mean']} | {g['median']} | {_shares(g['bin_share'])} |"
                                                   if g else "— | — | — |" + " — |" * len(names)))

    if "label_by_age_band" in report:
        md += ["", "## 年龄段与获卵数（训练集）", "", "| 年龄 | 例数 | 平均获卵数 | 中位数 | " + " | ".join(names) + " |",
               "|---|---|---|---|" + "---|" * len(names)]
        md += [f"| {band} | {g['n']} | {g['mean']} | {g['median']} | {_shares(g['bin_share'])} |"
               for band, g in report["label_by_age_band"].items()]

    flags = {k: f for k, f in report["fields"].items() if f["type"] == "flag"}
    if flags:
        d = report["diagnosis"]
        md += ["", "## 诊断类别", "",
               f"诊断为空 {d['empty']} 例；有诊断但没有匹配到任何类别 {d['unmatched']} 例（{d['unmatched_rate']:.1%}）。"
               "一位患者可以属于多个类别。", "",
               "| 类别 | 例数 | 占比 | 训练集：有 / 无该诊断的平均获卵数 |", "|---|---|---|---|"]
        for name, f in sorted(flags.items(), key=lambda kv: -kv[1]["positive"]):
            g = f["label_by_value"]
            pair = f"{g['1']['mean']} / {g['0']['mean']}" if "1" in g and "0" in g else "—"
            md.append(f"| {f['title']}（{name}） | {f['positive']} | {f['positive_rate']:.1%} | {pair} |")
        md += ["", f"诊断共 {d['distinct_texts']} 种不同写法；{d['repeated_text_rows']} 个周期的诊断与别的周期完全相同（可能是同一患者的多个周期）。",
               "", f"诊断高频词（去掉括号内容后按空白和分隔符切开，每种诊断写法只计一次，共 {d['distinct_tokens']} 种；用来修订关键词表）：", "",
               "，".join(f"{t.replace('|', '/')}: {c}" for t, c in d["top_tokens"].items())]
    return "\n".join(md) + "\n"
