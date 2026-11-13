"""Prediction from a trained model file. Loads only the file (coefficients and
encoding statistics, no patient rows); depends on numpy only.

``predict`` returns the probability of each oocyte-count group, the most likely
group, the expected count, and each given input's effect: the change in the
expected count when only that input goes from "not provided" to the patient's
value while every other input stays as the patient gave it. Effects describe
the model's associations, not causes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from ovu.ml.features import encode, feature_names, is_missing
from ovu.ml.model import softmax

ARTIFACT_VERSION = 1


class Predictor:
    def __init__(self, artifact: dict[str, Any]):
        if artifact.get("artifact_version") != ARTIFACT_VERSION or artifact.get("model") != "softmax":
            raise ValueError("不支持的模型文件")
        self.encoder = artifact["encoder"]
        self.weights = np.array(artifact["weights"])
        self.bias = np.array(artifact["intercepts"])
        self.groups: list[str] = artifact["groups"]
        self.group_means = np.array(artifact["group_means"])
        self.inputs = feature_names(self.encoder)
        self.info = artifact["training"]

    @classmethod
    def from_file(cls, path: str | Path) -> "Predictor":
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    def probabilities(self, rows: list[dict[str, Any]]) -> np.ndarray:
        return softmax(encode(rows, self.encoder) @ self.weights + self.bias)

    def predict(self, **values: Any) -> dict[str, Any]:
        unknown = sorted(set(values) - set(self.inputs))
        if unknown:
            raise ValueError(f"不支持的输入：{unknown}；可用的输入：{self.inputs}")
        given = {name: value for name, value in values.items() if not is_missing(value)}
        if not given:
            raise ValueError(f"至少提供一项输入：{self.inputs}")
        without = [{k: v for k, v in given.items() if k != name} for name in given]
        p = self.probabilities([given] + without)
        expected = p @ self.group_means
        effects = [{"输入": name, "取值": given[name], "预期获卵数变化": round(float(expected[0] - expected[1 + i]), 2)}
                   for i, name in enumerate(given)]
        effects.sort(key=lambda e: -abs(e["预期获卵数变化"]))
        return {
            "各分组概率": {group: round(float(q), 3) for group, q in zip(self.groups, p[0])},
            "最可能的分组": self.groups[int(p[0].argmax())],
            "预期获卵数": round(float(expected[0]), 1),
            "各输入的影响": effects,
            "未提供的输入": [name for name in self.inputs if name not in given],
            "说明": "预期获卵数 = 各分组概率 × 训练集中该分组的平均获卵数。各输入的影响 = 提供该项与不提供该项相比，"
                  "预期获卵数的变化（其余输入不变）；是模型中的关联，不是因果。未提供的数值按训练集中位数处理。",
        }
