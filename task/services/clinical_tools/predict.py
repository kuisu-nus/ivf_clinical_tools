"""Clinical pregnancy prediction from female age and AMH (logistic regression, numpy only).

Model: logit p = b0 + b1 (age - 33) + b2 (age - 33)^2 + b3 log(AMH), with AMH
floored at 0.02 ng/mL (the most common lowest value, likely the assay limit).
Chosen on the cleaned training set by repeated 5-fold cross-validation over
age only, age + log AMH, age + age^2 + log AMH and an age x AMH interaction;
the quadratic age term let the decline steepen in the late thirties and gave
the best AUC and Brier score. Without AMH the age-only quadratic model is used.

The 95% interval reflects uncertainty in the fitted coefficients (Wald interval
on the logit), not the range of outcomes for one patient. Inputs outside the
training range are refused rather than extrapolated. Cross-validated AUC and
Brier score are recomputed at start-up on the visible cases and reported with
every prediction so the agent can weigh the model.
"""

from __future__ import annotations

import math

import numpy as np

from clinical_tools.cleaning import TrainingSet
from clinical_tools.common import FEATURE_NAMES, InputError, assign_folds, number_argument

AGE, AMH = FEATURE_NAMES.index("女方年龄"), FEATURE_NAMES.index("女方AMH")
AGE_CENTRE = 33.0
AMH_FLOOR = 0.02
CV_FOLDS, CV_REPEATS = 5, 5
Z95 = 1.959964


def design(age: np.ndarray, amh: np.ndarray | None) -> np.ndarray:
    a = np.asarray(age, dtype=float) - AGE_CENTRE
    columns = [np.ones_like(a), a, a ** 2]
    if amh is not None:
        columns.append(np.log(np.maximum(np.asarray(amh, dtype=float), AMH_FLOOR)))
    return np.column_stack(columns)


def fit_logistic(x: np.ndarray, y: np.ndarray, *, ridge: float = 1e-6, iterations: int = 100) -> tuple[np.ndarray, np.ndarray]:
    """Newton-Raphson maximum likelihood; returns coefficients and their covariance."""
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


class _Model:
    def __init__(self, ids: list[str], age: np.ndarray, amh: np.ndarray | None, labels: np.ndarray):
        self.uses_amh = amh is not None
        x = design(age, amh)
        self.beta, self.cov = fit_logistic(x, labels)
        self.n = int(len(labels))
        self.age_range = (float(age.min()), float(age.max()))
        self.amh_range = (float(amh.min()), float(amh.max())) if amh is not None else None
        self.reference = x.mean(axis=0)
        aucs, briers = [], []
        for repeat in range(CV_REPEATS):
            folds = np.array(assign_folds(ids, labels.tolist(), CV_FOLDS, seed=f"predict-cv-{repeat}"))
            oof = np.zeros(self.n)
            for f in range(CV_FOLDS):
                train = folds != f
                beta, _ = fit_logistic(x[train], labels[train])
                oof[~train] = 1 / (1 + np.exp(-(x[~train] @ beta)))
            aucs.append(auc(oof, labels))
            briers.append(float(np.mean((oof - labels) ** 2)))
        self.cv_auc, self.cv_brier = float(np.mean(aucs)), float(np.mean(briers))
        self.base_rate = float(labels.mean())
        self.base_brier = float(np.mean((self.base_rate - labels) ** 2))

    def predict(self, age: float, amh: float | None) -> dict:
        x = design(np.array([age]), None if amh is None else np.array([amh]))[0]
        logit = float(x @ self.beta)
        se = float(math.sqrt(x @ self.cov @ x))
        expit = lambda z: 1 / (1 + math.exp(-z))
        terms = [("女方年龄（含年龄²）", float((x[1:3] - self.reference[1:3]) @ self.beta[1:3]))]
        if self.uses_amh:
            terms.append(("AMH（log）", float((x[3] - self.reference[3]) * self.beta[3])))
        return {"point": expit(logit), "low": expit(logit - Z95 * se), "high": expit(logit + Z95 * se), "terms": terms}


class OutcomePredictor:
    def __init__(self, training: TrainingSet, *, exclude_case: str | None = None, k_folds: int = 5):
        training, self.scope = training.visible(exclude_case, k_folds)
        age, amh, y = training.matrix[:, AGE], training.matrix[:, AMH], training.labels
        has_age, has_both = ~np.isnan(age), ~np.isnan(age) & ~np.isnan(amh)
        ids = training.case_ids
        self.with_amh = _Model([c for c, k in zip(ids, has_both) if k], age[has_both], amh[has_both], y[has_both])
        self.age_only = _Model([c for c, k in zip(ids, has_age) if k], age[has_age], None, y[has_age])

    def query(self, arguments: dict) -> dict:
        unknown = set(arguments) - {"age", "amh"}
        if unknown:
            raise InputError(f"不支持的参数：{'、'.join(sorted(unknown))}；只接受 age 和 amh")
        age = number_argument(arguments, "age", 18, 55)
        amh = number_argument(arguments, "amh", 0, 50, required=False)
        model = self.with_amh if amh is not None else self.age_only
        inputs = {"age": age, **({"amh": amh} if amh is not None else {})}
        out_of_range = []
        if not model.age_range[0] <= age <= model.age_range[1]:
            out_of_range.append(f"年龄 {age:g} 超出训练数据范围 {model.age_range[0]:g}–{model.age_range[1]:g} 岁")
        if amh is not None and amh > model.amh_range[1]:
            out_of_range.append(f"AMH {amh:g} 超出训练数据上限 {model.amh_range[1]:g} ng/mL")
        if out_of_range:
            return {"输入": inputs, "拒绝预测": True, "原因": "；".join(out_of_range) + "，模型不外推"}
        result = model.predict(age, amh)
        notes = []
        if amh is None:
            notes.append("未提供 AMH，使用仅含年龄的模型")
        elif amh < AMH_FLOOR:
            notes.append(f"AMH 低于 {AMH_FLOOR} ng/mL，按 {AMH_FLOOR} 计算（可能为检测下限）")
        return {
            "输入": inputs, "拒绝预测": False,
            "预测结局": "本次移植的临床妊娠概率",
            "概率": round(result["point"], 3),
            "95%置信区间": [round(result["low"], 3), round(result["high"], 3)],
            "区间含义": "模型系数估计的不确定性，不是该患者个体结局的范围",
            "各因素作用（相对训练集平均患者的对数几率，正为升高）": [
                {"因素": name, "作用": "升高" if value > 0 else "降低", "对数几率": round(value, 3)}
                for name, value in result["terms"]],
            "模型卡片": {
                "模型": "逻辑回归：年龄、年龄²、log(AMH)" if model.uses_amh else "逻辑回归：年龄、年龄²",
                "训练范围": self.scope, "训练样本数": model.n,
                "交叉验证AUC": round(model.cv_auc, 3),
                "交叉验证Brier": round(model.cv_brier, 4),
                "按总体妊娠率预测的Brier（对照）": round(model.base_brier, 4),
                "本模型训练样本妊娠率": round(model.base_rate, 3),
                "适用范围": f"年龄 {model.age_range[0]:g}–{model.age_range[1]:g} 岁"
                            + (f"，AMH ≤ {model.amh_range[1]:g} ng/mL" if model.uses_amh else ""),
                "数据处理": "结局回填值和异常值按结局盲规则还原为缺失"},
            **({"备注": notes} if notes else {}),
            "说明": "模型区分能力有限（交叉验证 AUC 约 0.6），只能作为参考；未纳入胚胎、内膜、男方等因素；"
                  "对数几率是模型权重，不代表因果。",
        }


def tool_spec(base_url: str = "http://127.0.0.1:9201") -> dict:
    return {
        "name": "ivf_predict_outcome",
        "description": "用去泄露训练集上的逻辑回归（年龄、年龄²、log AMH）预测当前患者本次移植的临床妊娠概率，"
                       "返回概率、95% 置信区间、年龄与 AMH 各自的作用和模型卡片（交叉验证 AUC、Brier）。"
                       "不提供 AMH 时只用年龄；输入超出训练数据范围时拒绝预测。模型区分能力有限，作为参考。",
        "input_schema": {
            "type": "object",
            "properties": {
                "age": {"type": "number", "description": "当前患者女方年龄（岁）"},
                "amh": {"type": "number", "description": "当前患者女方 AMH（ng/mL）；未检查可不填"},
            },
            "required": ["age"],
            "additionalProperties": False,
        },
        "backend": {"type": "http", "service": "ivf_clinical", "url": f"{base_url}/ivf/predict_outcome",
                    "method": "POST"},
    }
