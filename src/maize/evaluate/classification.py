"""Stage 1 metrics: macro-F1, per-class recall, ECE, bootstrap CI, McNemar, retake threshold."""
from __future__ import annotations

import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score

from ..constants import CLASSES


def summarize(y, probs) -> dict:
    yhat = probs.argmax(1)
    k = range(len(CLASSES))
    return {
        "accuracy": float(accuracy_score(y, yhat)),
        "macro_f1": float(f1_score(y, yhat, average="macro")),
        "recall": dict(zip(CLASSES, map(float, recall_score(y, yhat, average=None, labels=k)))),
        "confusion": confusion_matrix(y, yhat, labels=k).tolist(),
        "ece": ece(y, probs),
    }


def ece(y, probs, bins: int = 15) -> float:
    conf, pred = probs.max(1), probs.argmax(1)
    correct = (pred == y).astype(float)
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(total)


def bootstrap_ci(y, probs, fn=lambda y, p: f1_score(y, p.argmax(1), average="macro"),
                 n: int = 1000, seed: int = 0, alpha: float = 0.05):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y))
    vals = [fn(y[s], probs[s]) for s in (rng.choice(idx, len(idx)) for _ in range(n))]
    return float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2))


def mcnemar_exact(y, probs_a, probs_b) -> dict:
    a, b = probs_a.argmax(1) == y, probs_b.argmax(1) == y
    n01, n10 = int((~a & b).sum()), int((a & ~b).sum())  # discordant pairs
    p = 1.0 if n01 + n10 == 0 else float(binomtest(min(n01, n10), n01 + n10, 0.5).pvalue)
    return {"a_wrong_b_right": n01, "a_right_b_wrong": n10, "p_value": p}


def pick_confidence_threshold(y, probs, target_acc: float = 0.95) -> dict:
    """Lowest confidence cut-off such that accuracy on the ACCEPTED predictions >= target.
    Below it the tool asks for a retake instead of guessing."""
    conf, correct = probs.max(1), probs.argmax(1) == y
    for t in np.unique(np.round(conf, 3)):
        m = conf >= t
        if m.sum() >= 20 and correct[m].mean() >= target_acc:
            return {"threshold": float(t), "accepted": float(m.mean()), "accuracy_on_accepted": float(correct[m].mean())}
    return {"threshold": 0.9, "accepted": float((conf >= 0.9).mean()), "accuracy_on_accepted": None}
