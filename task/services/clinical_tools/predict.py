"""Clinical pregnancy prediction: logistic regression trained on all 35 indicators,
queried with age, AMH and Gn days only.

Training and prediction are separate. ``train_model`` fits the model on the
cleaned training set as before: all 35 indicators, missing values replaced by
training means, standardised, L2-penalised logistic regression (C = 1). It
returns a JSON-serialisable artifact (means, scales, coefficients, their
covariance and training metrics); ``clinical_tools.train_predictor`` writes it
to disk. ``Predictor`` loads only that artifact, never the training data. At
prediction time only age, AMH and Gn days are accepted; every other indicator,
and any of the three that is not given, is set to its training mean. The answer
is the probability and a 95% interval (Wald interval on the logit from the
coefficient covariance).
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from clinical_tools.cleaning import TrainingSet
from clinical_tools.common import FEATURE_NAMES, InputError, assign_folds, number_argument

# Prediction inputs: (argument name, training column, valid input range)
INPUTS = (("age", "女方年龄", (18, 55)), ("amh", "女方AMH", (0, 50)), ("gn_days", "女方用药Gn总天数", (1, 30)))
C = 1.0                          # inverse L2 strength on standardised coefficients, as the project LR
CV_FOLDS, CV_REPEATS = 5, 5
Z95 = 1.959964
ARTIFACT_VERSION = 2


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


class _Standardiser:
    def __init__(self, raw: np.ndarray):
        self.means = np.nanmean(raw, axis=0)
        filled = np.where(np.isnan(raw), self.means, raw)
        self.scales = filled.std(axis=0)
        self.scales[~(self.scales > 0)] = 1.0

    def design(self, raw: np.ndarray, only: list[int] | None = None) -> np.ndarray:
        """Intercept plus standardised values; missing values, and with ``only`` every
        column outside it, take the training mean."""
        filled = np.where(np.isnan(raw), self.means, raw)
        if only is not None:
            keep = np.zeros(raw.shape[1], dtype=bool)
            keep[only] = True
            filled = np.where(keep, filled, self.means)
        return np.column_stack([np.ones(len(raw)), (filled - self.means) / self.scales])


def train_model(training: TrainingSet, *, scope: str = "全部训练病例") -> dict:
    raw, y = training.matrix, training.labels
    inputs = [FEATURE_NAMES.index(column) for _, column, _ in INPUTS]
    standardiser = _Standardiser(raw)
    beta, cov = fit_logistic(standardiser.design(raw), y, ridge=1 / C)
    scores = {"all": ([], []), "inputs": ([], [])}
    for repeat in range(CV_REPEATS):  # means, scales and coefficients refitted inside each training fold
        folds = np.array(assign_folds(training.case_ids, y.tolist(), CV_FOLDS, seed=f"predict-cv-{repeat}"))
        oof = {"all": np.zeros(len(y)), "inputs": np.zeros(len(y))}
        for f in range(CV_FOLDS):
            train = folds != f
            fold = _Standardiser(raw[train])
            b, _ = fit_logistic(fold.design(raw[train]), y[train], ridge=1 / C)
            oof["all"][~train] = 1 / (1 + np.exp(-(fold.design(raw[~train]) @ b)))
            oof["inputs"][~train] = 1 / (1 + np.exp(-(fold.design(raw[~train], inputs) @ b)))
        for key, (aucs, briers) in scores.items():
            aucs.append(auc(oof[key], y))
            briers.append(float(np.mean((oof[key] - y) ** 2)))
    metric = lambda key, i: round(float(np.mean(scores[key][i])), 4)
    return {
        "artifact_version": ARTIFACT_VERSION,
        "model": "logistic_regression",
        "features": list(FEATURE_NAMES),
        "inputs": {name: column for name, column, _ in INPUTS},
        "means": standardiser.means.round(10).tolist(),
        "scales": standardiser.scales.round(10).tolist(),
        "intercept": round(float(beta[0]), 10),
        "weights": beta[1:].round(10).tolist(),
        "covariance": cov.round(12).tolist(),
        "training": {
            "scope": scope, "n": int(len(y)), "positives": int(y.sum()), "penalty": f"L2, C={C}, standardised",
            "imputation": "training mean", "missing": {n: int(np.isnan(raw[:, i]).sum()) for i, n in enumerate(FEATURE_NAMES)},
            "cv_auc_all_indicators": metric("all", 0), "cv_brier_all_indicators": metric("all", 1),
            "cv_auc_three_inputs": metric("inputs", 0), "cv_brier_three_inputs": metric("inputs", 1),
            "base_rate_brier": round(float(np.mean((y.mean() - y) ** 2)), 4),
            "cv": f"{CV_REPEATS} x {CV_FOLDS}-fold; three_inputs = held-out cases given only age, AMH, Gn days",
        },
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


class Predictor:
    def __init__(self, artifact: dict):
        if artifact.get("artifact_version") != ARTIFACT_VERSION or artifact.get("model") != "logistic_regression":
            raise ValueError("Unsupported model artifact")
        if list(artifact["features"]) != list(FEATURE_NAMES) or artifact["inputs"] != {n: c for n, c, _ in INPUTS}:
            raise ValueError("Model artifact features do not match this predictor")
        self.means = np.array(artifact["means"])
        self.scales = np.array(artifact["scales"])
        self.beta = np.array([artifact["intercept"], *artifact["weights"]])
        self.cov = np.array(artifact["covariance"])
        self.training = artifact["training"]

    @classmethod
    def from_file(cls, path: str | Path) -> "Predictor":
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    def query(self, arguments: dict) -> dict:
        names = [name for name, _, _ in INPUTS]
        unknown = set(arguments) - set(names)
        if unknown:
            raise InputError(f"不支持的参数：{'、'.join(sorted(unknown))}；只接受 {', '.join(names)}")
        values = [number_argument(arguments, name, *bounds, required=False) for name, _, bounds in INPUTS]
        if all(v is None for v in values):
            raise InputError(f"至少提供 {', '.join(names)} 中的一个")
        raw = self.means.copy()
        for (_, column, _), value in zip(INPUTS, values):
            if value is not None:
                raw[FEATURE_NAMES.index(column)] = value
        x = np.concatenate([[1.0], (raw - self.means) / self.scales])
        logit = float(x @ self.beta)
        half = Z95 * math.sqrt(float(x @ self.cov @ x))
        expit = lambda z: 1 / (1 + math.exp(-z))
        return {"妊娠概率": round(expit(logit), 3), "95%置信区间": [round(expit(logit - half), 3), round(expit(logit + half), 3)]}


def tool_spec(base_url: str = "http://127.0.0.1:9202") -> dict:
    return {
        "name": "ivf_predict_outcome",
        "description": "预测当前患者本次移植的临床妊娠概率及 95% 置信区间。模型为在 35 项指标上训练的逻辑回归，"
                       "预测时只接收女方年龄、AMH、Gn 用药总天数，其余指标及未提供的指标按训练集均值代替。"
                       "模型区分能力有限（交叉验证 AUC 约 0.6），置信区间只反映模型系数的不确定性，不是个体结局的范围。",
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
