"""Softmax (multinomial logistic) regression and the metrics for ordered groups. numpy only."""

from __future__ import annotations

import numpy as np


def softmax(scores: np.ndarray) -> np.ndarray:
    scores = scores - scores.max(axis=1, keepdims=True)
    e = np.exp(scores)
    return e / e.sum(axis=1, keepdims=True)


def fit_softmax(x: np.ndarray, y: np.ndarray, n_classes: int, *, l2: float = 1.0,
                max_iterations: int = 3000, tolerance: float = 1e-8) -> tuple[np.ndarray, np.ndarray, dict]:
    """Minimise mean log loss + l2/(2n) * |W|^2 by gradient descent with a backtracking
    line search (deterministic; the intercepts are not penalised). Returns the weights
    (features x classes), the intercepts and how the fit ended."""
    n, d = x.shape
    onehot = np.eye(n_classes)[y]
    prior = np.clip(onehot.mean(axis=0), 1e-9, None)
    weights, bias = np.zeros((d, n_classes)), np.log(prior)

    def loss(w: np.ndarray, b: np.ndarray) -> float:
        p = softmax(x @ w + b)
        return float(-np.log(np.clip(p[np.arange(n), y], 1e-300, None)).mean() + l2 / (2 * n) * (w ** 2).sum())

    current, step, used = loss(weights, bias), 1.0, 0
    for used in range(1, max_iterations + 1):
        error = (softmax(x @ weights + bias) - onehot) / n
        grad_w, grad_b = x.T @ error + l2 / n * weights, error.sum(axis=0)
        norm = float((grad_w ** 2).sum() + (grad_b ** 2).sum())
        step *= 2.0
        while True:
            candidate = loss(weights - step * grad_w, bias - step * grad_b)
            if candidate <= current - 0.5 * step * norm or step < 1e-12:
                break
            step /= 2.0
        weights, bias = weights - step * grad_w, bias - step * grad_b
        improvement, current = current - candidate, candidate
        if improvement < tolerance:
            break
    return weights, bias, {"iterations": used, "converged": used < max_iterations, "objective": round(current, 6)}


def quadratic_kappa(truth: np.ndarray, predicted: np.ndarray, n_classes: int) -> float:
    observed = np.zeros((n_classes, n_classes))
    np.add.at(observed, (truth, predicted), 1)
    expected = np.outer(observed.sum(axis=1), observed.sum(axis=0)) / observed.sum()
    i, j = np.indices((n_classes, n_classes))
    penalty = (i - j) ** 2
    denominator = (penalty * expected).sum()
    return float(1 - (penalty * observed).sum() / denominator) if denominator > 0 else 0.0


def metrics(probabilities: np.ndarray, truth: np.ndarray, group_means: np.ndarray, counts: np.ndarray | None = None) -> dict:
    """Metrics for ordered groups. ``rps`` (ranked probability score, lower is better)
    and ``log_loss`` judge the whole predicted distribution; ``within_one`` counts a
    prediction in a neighbouring group as right; ``mae_count`` compares the expected
    count (sum of p_k x the training mean count of group k) with the real count."""
    n, k = probabilities.shape
    predicted = probabilities.argmax(axis=1)
    onehot = np.eye(k)[truth]
    rps = ((np.cumsum(probabilities, axis=1) - np.cumsum(onehot, axis=1))[:, :-1] ** 2).sum(axis=1).mean() / (k - 1)
    out = {"n": int(n), "accuracy": float((predicted == truth).mean()),
           "within_one": float((np.abs(predicted - truth) <= 1).mean()),
           "quadratic_kappa": quadratic_kappa(truth, predicted, k), "rps": float(rps),
           "log_loss": float(-np.log(np.clip(probabilities[np.arange(n), truth], 1e-12, None)).mean()),
           "confusion": np.histogram2d(truth, predicted, bins=(np.arange(k + 1), np.arange(k + 1)))[0].astype(int).tolist()}
    if counts is not None:
        out["mae_count"] = float(np.abs(probabilities @ group_means - counts).mean())
    return {key: round(v, 4) if isinstance(v, float) else v for key, v in out.items()}
