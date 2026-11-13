"""Weighted nearest-case retrieval. Depends on numpy only.

``resolve`` turns ovu/config/similar.yaml plus the training rows into a plain
dict (features, weights, numeric scales) that ``python -m ovu.similar.build``
writes to similar.json. ``SimilarIndex`` loads that dict and the training rows
and answers queries; the tool service needs nothing else.

A query gives any subset of the configured features. The distance to a case is
the weighted mean, over the features given, of: |difference| / scale capped at
``cap`` (numeric), 0 or 1 (category, binary), ``missing_distance`` when the
case lacks the value. The answer describes the k nearest cases: how their
oocyte counts fall into the groups, summary statistics, and a short list of
the closest cases without any identifier.

``hide_fold``: when an agent is trained on a training case, the tools hide that
case's ``cv_fold`` so its own outcome cannot come back.
"""

from __future__ import annotations

import math
import sys
from typing import Any

import numpy as np

from ovu.inputs import describe
from ovu.ml.features import is_missing

KINDS = ("numeric", "category", "binary")
Row = dict[str, Any]


def _number(value: Any) -> float:
    return float("nan") if is_missing(value) else float(value)


def _transform(values: np.ndarray, kind: str | None) -> np.ndarray:
    return np.log1p(np.clip(values, 0, None)) if kind == "log" else values


def resolve(config: dict[str, Any], rows: list[Row], data_config: dict[str, Any], bins: list[int], groups: list[str]) -> dict[str, Any]:
    """Check the config against the data config and replace ``scale: std`` by the training value."""
    from ovu.pipeline.clean import feature_stages                   # offline only (needs pandas); queries never get here
    stages, label = feature_stages(data_config), data_config["label"]["name"]
    allowed = set(config["stages"])
    features: dict[str, Any] = {}
    for name, spec in config["features"].items():
        if name not in stages:                                       # removed from data.yaml: leave it out
            print(f"提示：similar.yaml 中的字段 {name} 不在 data.yaml 里，已跳过", file=sys.stderr)
            continue
        if stages[name] not in allowed:
            raise ValueError(f"{name} 的预测时点是 {stages[name]}，不在允许的 {sorted(allowed)} 中")
        if spec.get("type") not in KINDS or not float(spec.get("weight", 0)) > 0:
            raise ValueError(f"{name}：type 必须是 {KINDS} 之一，weight 必须大于 0")
        entry: dict[str, Any] = {"type": spec["type"], "weight": float(spec["weight"])}
        if spec["type"] == "numeric":
            entry["transform"] = spec.get("transform")
            scale = spec.get("scale", "std")
            if scale == "std":
                values = _transform(np.array([_number(r.get(name)) for r in rows]), entry["transform"])
                scale = float(np.nanstd(values))
            entry["scale"] = float(scale) if float(scale) > 0 else 1.0
        features[name] = entry
    if not features:
        raise ValueError("similar.yaml 中没有可用的比对特征")
    listed = [f for f in config.get("listed_fields", []) if f in stages]
    levels = {name: sorted({str(r[name]) for r in rows if not is_missing(r.get(name))})
              for name, f in features.items() if f["type"] == "category"}
    return {"features": features, "inputs": describe(list(features), data_config, levels), "cap": float(config.get("cap", 3.0)), "missing_distance": float(config.get("missing_distance", 1.0)),
            "k_default": int(config.get("k_default", 50)), "k_max": int(config.get("k_max", 200)),
            "min_cohort": int(config.get("min_cohort", 15)), "max_listed": int(config.get("max_listed", 10)),
            "listed_fields": listed, "label": label, "bins": bins, "groups": groups}


class SimilarIndex:
    def __init__(self, spec: dict[str, Any], rows: list[Row]):
        self.spec = spec
        self.rows = rows
        self.features = spec["features"]
        self.counts = np.array([int(float(r[spec["label"]])) for r in rows])
        self.group = np.searchsorted(np.asarray(spec["bins"]), self.counts, side="left")
        self.fold = np.array([int(r["cv_fold"]) if not is_missing(r.get("cv_fold")) else -1 for r in rows])
        self.numeric = {name: _transform(np.array([_number(r.get(name)) for r in rows]), f["transform"])
                        for name, f in self.features.items() if f["type"] == "numeric"}
        self.discrete = {name: np.array(["" if is_missing(r.get(name)) else _label(r[name]) for r in rows])
                         for name, f in self.features.items() if f["type"] != "numeric"}

    def distances(self, query: Row) -> np.ndarray:
        total, weight_sum = np.zeros(len(self.rows)), 0.0
        for name, value in query.items():
            feature = self.features[name]
            if feature["type"] == "numeric":
                target = float(_transform(np.array([float(value)]), feature["transform"])[0])
                d = np.minimum(np.abs(self.numeric[name] - target) / feature["scale"], self.spec["cap"])
                d = np.where(np.isnan(d), self.spec["missing_distance"], d)
            else:
                column = self.discrete[name]
                d = np.where(column == "", self.spec["missing_distance"], (column != _label(value)).astype(float))
            total += feature["weight"] * d
            weight_sum += feature["weight"]
        return total / weight_sum

    def query(self, values: Row, *, k: int | None = None, hide_fold: int | None = None) -> dict[str, Any]:
        unknown = sorted(set(values) - set(self.features))
        if unknown:
            raise ValueError(f"不支持的检索特征：{unknown}；可用的特征：{list(self.features)}")
        given = {name: value for name, value in values.items() if not is_missing(value)}
        if not given:
            raise ValueError(f"至少提供一项检索特征：{list(self.features)}")
        k = self.spec["k_default"] if k is None else int(k)
        if not 1 <= k <= self.spec["k_max"]:
            raise ValueError(f"k 必须在 1 到 {self.spec['k_max']} 之间")
        visible = np.ones(len(self.rows), dtype=bool) if hide_fold is None else self.fold != hide_fold
        distance = np.where(visible, self.distances(given), np.inf)
        order = np.argsort(distance, kind="mergesort")[:min(k, int(visible.sum()))]
        criteria = {"比对的特征与权重": {name: self.features[name]["weight"] for name in given},
                    "未用于比对的特征": [name for name in self.features if name not in given],
                    "可比病例数": int(visible.sum()), "取最相似的例数": int(len(order))}
        if len(order) < self.spec["min_cohort"]:
            return {"检索条件": criteria, "结果": f"相似病例不足 {self.spec['min_cohort']} 例，不给统计；请增大 k。"}
        counts = self.counts[order]
        groups = np.bincount(self.group[order], minlength=len(self.spec["groups"]))
        listed = []
        for i in order[:self.spec["max_listed"]]:
            row = self.rows[i]
            entry = {field: _shown(row.get(field)) for field in self.spec["listed_fields"]}
            entry.update({"获卵数": int(self.counts[i]), "距离": round(float(distance[i]), 3)})
            listed.append(entry)
        return {
            "检索条件": criteria,
            "获卵数分组": {g: {"例数": int(c), "占比": round(float(c) / len(order), 3)} for g, c in zip(self.spec["groups"], groups)},
            "获卵数": {"中位数": float(np.median(counts)), "P25": float(np.percentile(counts, 25)),
                    "P75": float(np.percentile(counts, 75)), "均值": round(float(counts.mean()), 1)},
            "距离范围": [round(float(distance[order[0]]), 3), round(float(distance[order[-1]]), 3)],
            "最相似的患者": listed,
            "说明": "距离是各比对特征差异的加权平均，越小越相似。统计来自训练集中的历史病例，是关联，不是对当前患者的保证。",
        }


def _label(value: Any) -> str:
    """'1', 1 and 1.0 are the same level."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value).strip()
    return str(int(number)) if number.is_integer() else str(number)


def _shown(value: Any) -> Any:
    if is_missing(value):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return value
    return int(number) if math.isfinite(number) and number.is_integer() else round(number, 2)
