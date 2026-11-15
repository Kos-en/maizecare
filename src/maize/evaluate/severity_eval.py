"""Stage 2 metrics (Table 3.3): AUROC, false-high rate, Spearman and MAE vs symptomatic area,
and stability within near-duplicate groups. Reported separately per disease."""
from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score


def auroc(healthy_s: np.ndarray, diseased_s: np.ndarray) -> float:
    y = np.r_[np.zeros(len(healthy_s)), np.ones(len(diseased_s))]
    return float(roc_auc_score(y, np.r_[healthy_s, diseased_s]))


def false_high_rate(healthy_e: np.ndarray, e_healthy_p95: float) -> float:
    """Share of held-out healthy leaves scored ABOVE the healthy range."""
    return float((healthy_e > e_healthy_p95).mean())


def _minmax(v):
    v = np.asarray(v, float)
    r = v.max() - v.min()
    return (v - v.min()) / r if r > 0 else np.zeros_like(v)


def vs_symptomatic_area(score: np.ndarray, area: np.ndarray) -> dict:
    rho, p = spearmanr(score, area)
    return {"spearman_rho": float(rho), "p_value": float(p),
            "mae_01": float(np.abs(_minmax(score) - _minmax(area)).mean()), "n": int(len(score))}


def group_stability(score: np.ndarray, group_id: np.ndarray) -> dict:
    """Mean within-group std of s(x) over groups with >=2 photos, next to the overall std."""
    stds = [score[group_id == g].std() for g in np.unique(group_id) if (group_id == g).sum() >= 2]
    return {"mean_within_group_std": float(np.mean(stds)) if stds else None,
            "overall_std": float(score.std()), "n_groups": len(stds)}
