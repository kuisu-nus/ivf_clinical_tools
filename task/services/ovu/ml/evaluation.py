"""Evaluation and explanation of a trained oocyte-count group model. numpy only.

Shared by ``ovu.ml.train`` (training, validation and cross-validation folds) and
``ovu.ml.evaluate_test`` (the one-time evaluation on the sealed test set), so every
set is scored the same way. Everything returned here is an aggregate: no patient rows.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np

from ovu.ml.features import column_names, encode, feature_names, is_missing
from ovu.ml.model import metrics, softmax

# Metric -> (name in reports, True when higher is better)
METRICS = {"accuracy": ("准确率", True), "within_one": ("相邻组以内的准确率", True),
           "quadratic_kappa": ("加权 kappa（二次）", True), "rps": ("RPS", False),
           "log_loss": ("对数损失", False), "mae_count": ("预期获卵数的平均绝对误差", False)}


def read_rows(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def labels(rows: list[dict[str, Any]], column: str, bins: list[int]) -> tuple[np.ndarray, np.ndarray]:
    counts = np.array([int(float(r[column])) for r in rows])
    return counts, np.searchsorted(np.asarray(bins), counts, side="left")


def probabilities(rows: list[dict[str, Any]], artifact: dict[str, Any]) -> np.ndarray:
    return softmax(encode(rows, artifact["encoder"]) @ np.array(artifact["weights"]) + np.array(artifact["intercepts"]))


def evaluate(rows: list[dict[str, Any]], artifact: dict[str, Any]) -> dict[str, Any]:
    """Metrics of the model and of the baseline (every patient gets the training group shares),
    and the calibration by group: the average predicted probability against the observed share."""
    counts, y = labels(rows, artifact["label"], artifact["bins"])
    means = np.array(artifact["group_means"])
    p = probabilities(rows, artifact)
    baseline = np.tile(np.array(artifact["group_share"]), (len(rows), 1))
    k = len(artifact["groups"])
    return {"model": metrics(p, y, means, counts), "baseline": metrics(baseline, y, means, counts),
            "calibration": {"predicted": [round(float(v), 4) for v in p.mean(axis=0)],
                            "observed": [round(float(v), 4) for v in np.bincount(y, minlength=k) / len(y)]}}


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


def partial_dependence(rows: list[dict[str, Any]], artifact: dict[str, Any], points: int = 15) -> dict[str, Any]:
    """For each feature: set it to one value for every patient in ``rows`` (all other inputs as
    they are) and average the predicted expected count and group probabilities; repeat over a
    grid of values (P5 to P95 for numbers, every level for categories, 0 and 1 for flags).
    Shows the shape of each feature's association in the model, not a cause."""
    encoder = artifact["encoder"]
    weights, bias, means = np.array(artifact["weights"]), np.array(artifact["intercepts"]), np.array(artifact["group_means"])
    x = encode(rows, encoder)
    owner = [c["feature"] for c in column_names(encoder)]
    out: dict[str, Any] = {}
    for feature in feature_names(encoder):
        columns = [i for i, f in enumerate(owner) if f == feature]
        if feature in encoder["numeric"]:
            values = np.array([float(r[feature]) for r in rows if not is_missing(r.get(feature))])
            if values.size == 0:
                continue
            grid = np.unique(np.round(np.quantile(values, np.linspace(0.05, 0.95, points)), 4)).tolist()
            kind = "numeric"
        elif feature in encoder["categorical"]:
            grid, kind = list(encoder["categorical"][feature]["levels"]), "category"
        else:
            grid, kind = [0, 1], "binary"
        expected, probs = [], []
        for value in grid:
            changed = x.copy()
            changed[:, columns] = encode([{feature: value}], encoder)[0, columns]
            p = softmax(changed @ weights + bias)
            expected.append(round(float((p @ means).mean()), 3))
            probs.append([round(float(v), 4) for v in p.mean(axis=0)])
        out[feature] = {"kind": kind, "values": grid, "expected_count": expected, "group_probabilities": probs}
    return out


def feature_titles(names: list[str], data_config: dict[str, Any]) -> dict[str, str]:
    """Field -> the Chinese name from data.yaml (diagnosis categories from diagnosis.names)."""
    specs = {**data_config.get("columns", {}), **data_config.get("split_columns", {})}
    flags = (data_config.get("diagnosis") or {}).get("names", {})
    return {n: specs.get(n, {}).get("name") or flags.get(n) or n for n in names}


def summarise_folds(folds: list[dict[str, Any]]) -> dict[str, Any]:
    """Mean and standard deviation of each metric over the cross-validation folds."""
    out = {}
    for who in ("model", "baseline"):
        out[who] = {key: {"mean": round(float(np.mean([f[who][key] for f in folds])), 4),
                          "sd": round(float(np.std([f[who][key] for f in folds])), 4)}
                    for key in METRICS if all(key in f[who] for f in folds)}
    return out
