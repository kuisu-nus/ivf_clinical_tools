"""Clinical pregnancy prediction: logistic regression on age, AMH and Gn days.

Training and prediction are separate. ``train_model`` fits the model on the
cleaned training set and returns a JSON-serialisable artifact (coefficients,
their covariance, the training means used for imputation and training metrics);
``clinical_tools.train_predictor`` writes it to disk. ``Predictor`` loads only
that artifact, never the training data, and returns the probability with a 95%
interval (Wald interval on the logit, from the coefficient covariance).

Features enter linearly. A missing feature, in training or at prediction time,
is replaced by its training mean.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from clinical_tools.cleaning import TrainingSet
from clinical_tools.common import FEATURE_NAMES, InputError, assign_folds, number_argument

# (argument name, training column, valid input range)
FEATURES = (("age", "女方年龄", (18, 55)), ("amh", "女方AMH", (0, 50)), ("gn_days", "女方用药Gn总天数", (1, 30)))
CV_FOLDS, CV_REPEATS = 5, 5
Z95 = 1.959964
ARTIFACT_VERSION = 1


def fit_logistic(x: np.ndarray, y: np.ndarray, *, ridge: float = 1e-6, iterations: int = 100) -> tuple[np.ndarray, np.ndarray]:
    """Newton-Raphson maximum likelihood (x includes the intercept column);
    returns coefficients and their covariance."""
    beta = np.zeros(x.shape[1])
    penalty = ridge * np.eye(x.shape[1])
    penalty[0, 0] = 0.0
    for _ in range(iterations):
        p = 1 / (1 + np.exp(-(x @ beta)))
        hessian = x.T @ (x * (p * (1 - p))[:, None]) + penalty
        step = np.linalg.solve(hessian, x.T @ (y - p) - penalty @ beta)
        beta += step
        if np.max(np.abs(step)) < 1e-10:
            break
    p = 1 / (1 + np.exp(-(x @ beta)))
    return beta, np.linalg.inv(x.T @ (x * (p * (1 - p))[:, None]) + penalty)


def auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Rank-based AUC with ties counted as one half."""
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores))
    sorted_scores = scores[order]
    i = 0
    while i < len(scores):
        j = i
        while j + 1 < len(scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    positives = labels.sum()
    negatives = len(labels) - positives
    return float((ranks[labels == 1].sum() - positives * (positives + 1) / 2) / (positives * negatives))


def _design(raw: np.ndarray, means: np.ndarray) -> np.ndarray:
    filled = np.where(np.isnan(raw), means, raw)
    return np.column_stack([np.ones(len(filled)), filled])


def train_model(training: TrainingSet, *, scope: str = "全部训练病例") -> dict:
    raw = np.column_stack([training.matrix[:, FEATURE_NAMES.index(column)] for _, column, _ in FEATURES])
    y = training.labels
    means = np.nanmean(raw, axis=0)
    beta, cov = fit_logistic(_design(raw, means), y)
    aucs, briers = [], []
    for repeat in range(CV_REPEATS):  # means and coefficients refitted inside each training fold
        folds = np.array(assign_folds(training.case_ids, y.tolist(), CV_FOLDS, seed=f"predict-cv-{repeat}"))
        oof = np.zeros(len(y))
        for f in range(CV_FOLDS):
            train = folds != f
            fold_means = np.nanmean(raw[train], axis=0)
            b, _ = fit_logistic(_design(raw[train], fold_means), y[train])
            oof[~train] = 1 / (1 + np.exp(-(_design(raw[~train], fold_means) @ b)))
        aucs.append(auc(oof, y))
        briers.append(float(np.mean((oof - y) ** 2)))
    names = [name for name, _, _ in FEATURES]
    return {
        "artifact_version": ARTIFACT_VERSION,
        "model": "logistic_regression",
        "features": [{"name": name, "column": column} for name, column, _ in FEATURES],
        "coefficients": dict(zip(["intercept", *names], beta.round(10).tolist())),
        "covariance": cov.round(12).tolist(),
        "imputation_means": dict(zip(names, means.round(10).tolist())),
        "training": {
            "scope": scope, "n": int(len(y)), "positives": int(y.sum()),
            "missing": {name: int(np.isnan(raw[:, i]).sum()) for i, name in enumerate(names)},
            "observed_range": {name: [float(np.nanmin(raw[:, i])), float(np.nanmax(raw[:, i]))]
                               for i, name in enumerate(names)},
            "cv_auc": round(float(np.mean(aucs)), 4), "cv_brier": round(float(np.mean(briers)), 4),
            "base_rate_brier": round(float(np.mean((y.mean() - y) ** 2)), 4),
            "cv": f"{CV_REPEATS} x {CV_FOLDS}-fold, imputation means refitted per fold",
        },
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


class Predictor:
    def __init__(self, artifact: dict):
        if artifact.get("artifact_version") != ARTIFACT_VERSION or artifact.get("model") != "logistic_regression":
            raise ValueError("Unsupported model artifact")
        names = [name for name, _, _ in FEATURES]
        if [f["name"] for f in artifact["features"]] != names:
            raise ValueError("Model artifact features do not match this predictor")
        self.beta = np.array([artifact["coefficients"][k] for k in ("intercept", *names)])
        self.cov = np.array(artifact["covariance"])
        self.means = np.array([artifact["imputation_means"][n] for n in names])
        self.training = artifact["training"]

    @classmethod
    def from_file(cls, path: str | Path) -> "Predictor":
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    def query(self, arguments: dict) -> dict:
        names = [name for name, _, _ in FEATURES]
        unknown = set(arguments) - set(names)
        if unknown:
            raise InputError(f"不支持的参数：{'、'.join(sorted(unknown))}；只接受 {', '.join(names)}")
        values = [number_argument(arguments, name, *bounds, required=False) for name, _, bounds in FEATURES]
        if all(v is None for v in values):
            raise InputError(f"至少提供 {', '.join(names)} 中的一个")
        x = np.array([1.0, *[m if v is None else v for v, m in zip(values, self.means)]])
        logit = float(x @ self.beta)
        half = Z95 * math.sqrt(float(x @ self.cov @ x))
        expit = lambda z: 1 / (1 + math.exp(-z))
        return {"妊娠概率": round(expit(logit), 3), "95%置信区间": [round(expit(logit - half), 3), round(expit(logit + half), 3)]}


def tool_spec(base_url: str = "http://127.0.0.1:9202") -> dict:
    return {
        "name": "ivf_predict_outcome",
        "description": "预测当前患者本次移植的临床妊娠概率及 95% 置信区间。模型为逻辑回归，只使用女方年龄、AMH、"
                       "Gn 用药总天数三项；未提供的指标按训练集均值代替。模型区分能力有限（交叉验证 AUC 约 0.6），"
                       "置信区间只反映模型系数的不确定性，不是个体结局的范围。",
        "input_schema": {
            "type": "object",
            "properties": {
                "age": {"type": "number", "description": "女方年龄（岁）"},
                "amh": {"type": "number", "description": "女方 AMH（ng/mL）"},
                "gn_days": {"type": "number", "description": "女方 Gn 用药总天数（天）"},
            },
            "required": [],
            "additionalProperties": False,
        },
        "backend": {"type": "http", "service": "ivf_predict", "url": f"{base_url}/ivf/predict_outcome",
                    "method": "POST"},
    }
