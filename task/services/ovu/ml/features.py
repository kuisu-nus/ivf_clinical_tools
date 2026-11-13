"""Feature encoding set by ovu/config/model.yaml. Depends on numpy only.

``fit_encoder`` learns the encoding from training rows and returns a plain dict
(medians, means, scales, category levels) that goes into the model file.
``encode`` turns rows (dicts of unified field name -> value) into the design
matrix with that dict. Training and prediction both go through ``encode``, so
they cannot drift apart.

A missing value is ``None``, NaN or an empty string. Numeric features: missing
-> training median (plus a 0/1 "missing" column when asked for), optional
log(1+x), then standardised. Categorical features: one column per level kept
at training time; rare, unseen and missing values share the level "其他".
Binary features: used as given, missing -> 0.
"""

from __future__ import annotations

import math
import sys
from collections import Counter
from typing import Any

import numpy as np

OTHER = "其他"
Row = dict[str, Any]


def is_missing(value: Any) -> bool:
    return value is None or value == "" or (isinstance(value, float) and math.isnan(value))


def _number(value: Any) -> float:
    return float("nan") if is_missing(value) else float(value)


def _transform(values: np.ndarray, kind: str) -> np.ndarray:
    if kind == "log":
        return np.log1p(np.clip(values, 0, None))
    if kind == "standard":
        return values
    raise ValueError(f"transform 必须是 standard 或 log，不是 {kind}")


def selected_features(config: dict[str, Any], stages: dict[str, str]) -> dict[str, list[str]]:
    """The configured features whose stage is allowed, by kind. A field that is not in the data
    config (removed there, so no longer loaded) is left out with a notice."""
    spec = config["features"]
    allowed = set(config["stages"])
    chosen: dict[str, list[str]] = {}
    for kind in ("numeric", "categorical", "binary"):
        names = list(spec.get(kind) or [])
        unknown = [n for n in names if n not in stages]
        if unknown:
            print(f"提示：model.yaml 中这些字段不在 data.yaml 里，已跳过：{unknown}", file=sys.stderr)
        chosen[kind] = [n for n in names if n in stages and stages[n] in allowed]
    return chosen


def fit_encoder(rows: list[Row], config: dict[str, Any], stages: dict[str, str]) -> dict[str, Any]:
    chosen = selected_features(config, stages)
    spec = config["features"]
    encoder: dict[str, Any] = {"numeric": {}, "categorical": {}, "binary": chosen["binary"]}
    for name in chosen["numeric"]:
        options = spec["numeric"][name] or {}
        kind = options.get("transform", "standard")
        raw = np.array([_number(r.get(name)) for r in rows])
        present = raw[~np.isnan(raw)]
        if present.size == 0:
            raise ValueError(f"{name}：训练集中全部缺失")
        median = float(np.median(present))
        transformed = _transform(np.where(np.isnan(raw), median, raw), kind)
        scale = float(transformed.std())
        encoder["numeric"][name] = {"transform": kind, "median": median, "mean": float(transformed.mean()),
                                    "scale": scale if scale > 0 else 1.0,
                                    "missing_indicator": bool(options.get("missing_indicator", False)),
                                    "missing_rate": round(float(np.isnan(raw).mean()), 4)}
    for name in chosen["categorical"]:
        options = spec["categorical"][name] or {}
        counts = Counter(str(r[name]) for r in rows if not is_missing(r.get(name)))
        levels = sorted(v for v, c in counts.items() if c >= options.get("min_count", 1) and v != OTHER)
        encoder["categorical"][name] = {"levels": levels + [OTHER]}
    encoder["columns"] = column_names(encoder)
    return encoder


def column_names(encoder: dict[str, Any]) -> list[dict[str, str]]:
    """One entry per design-matrix column: its name and the feature it belongs to."""
    columns = []
    for name, spec in encoder["numeric"].items():
        columns.append({"column": name, "feature": name})
        if spec["missing_indicator"]:
            columns.append({"column": f"{name}（缺失）", "feature": name})
    for name, spec in encoder["categorical"].items():
        columns += [{"column": f"{name}={level}", "feature": name} for level in spec["levels"]]
    columns += [{"column": name, "feature": name} for name in encoder["binary"]]
    return columns


def encode(rows: list[Row], encoder: dict[str, Any]) -> np.ndarray:
    blocks = []
    for name, spec in encoder["numeric"].items():
        raw = np.array([_number(r.get(name)) for r in rows])
        missing = np.isnan(raw)
        values = _transform(np.where(missing, spec["median"], raw), spec["transform"])
        blocks.append((values - spec["mean"]) / spec["scale"])
        if spec["missing_indicator"]:
            blocks.append(missing.astype(float))
    for name, spec in encoder["categorical"].items():
        levels = spec["levels"]
        index = {level: i for i, level in enumerate(levels)}
        onehot = np.zeros((len(rows), len(levels)))
        for i, row in enumerate(rows):
            value = row.get(name)
            onehot[i, index.get(OTHER if is_missing(value) else str(value), index[OTHER])] = 1.0
        blocks += list(onehot.T)
    for name in encoder["binary"]:
        blocks.append(np.array([0.0 if is_missing(r.get(name)) else float(r[name]) for r in rows]))
    return np.column_stack(blocks) if blocks else np.zeros((len(rows), 0))


def feature_names(encoder: dict[str, Any]) -> list[str]:
    return list(encoder["numeric"]) + list(encoder["categorical"]) + list(encoder["binary"])
