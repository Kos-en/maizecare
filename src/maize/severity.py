"""Severity score s(x) from reconstruction error (proposal Eq. 3.2). numpy only.

e(x)  = mean squared error over the leaf mask M
s(x)  = clip((e - e_H) / (e_D99 - e_H), 0, 1)
e_H   = median error of healthy validation leaves
e_D99 = 99th percentile of error over diseased validation leaves
Below the 95th percentile of healthy errors -> "within healthy range".
Mild/Moderate/Severe = per-disease tertiles of training scores above that range.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

GRADES = ["Mild", "Moderate", "Severe"]


def masked_error(x: np.ndarray, x_hat: np.ndarray, mask: np.ndarray) -> float:
    """x, x_hat: HxWx3 float; mask: HxW {0,1}. Mean squared error over mask pixels."""
    m = mask > 0
    if not m.any():
        return 0.0
    return float(((x - x_hat) ** 2).mean(axis=-1)[m].mean())


def error_map(x: np.ndarray, x_hat: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return ((x - x_hat) ** 2).mean(axis=-1) * (mask > 0)


def fit_calibration(healthy_val: np.ndarray, diseased_val: np.ndarray,
                    diseased_train: dict[str, np.ndarray]) -> dict:
    """All inputs are 1-D arrays of raw errors e(x). Returns a JSON-able dict."""
    e_h = float(np.median(healthy_val))
    e_d99 = float(np.percentile(diseased_val, 99))
    if e_d99 <= e_h:
        raise ValueError("e_D99 <= e_H: the autoencoder does not separate diseased from healthy.")
    cal = {
        "e_healthy_median": e_h,
        "e_disease_p99": e_d99,
        "e_healthy_p95": float(np.percentile(healthy_val, 95)),
        "tertiles": {},
    }
    for disease, errs in diseased_train.items():
        s = np.array([score(e, cal) for e in errs])
        s = s[np.array(errs) > cal["e_healthy_p95"]]  # only leaves above the healthy range
        cal["tertiles"][disease] = [float(np.quantile(s, 1 / 3)), float(np.quantile(s, 2 / 3))] if len(s) >= 3 else None
    return cal


def score(e: float, cal: dict) -> float:
    d = cal["e_disease_p99"] - cal["e_healthy_median"]
    return float(min(1.0, max(0.0, (e - cal["e_healthy_median"]) / d)))


def grade(e: float, disease: str, cal: dict) -> str:
    if e <= cal["e_healthy_p95"]:
        return "Within healthy range"
    t = cal["tertiles"].get(disease)
    if not t:
        return "Unknown"
    s = score(e, cal)
    return GRADES[0] if s <= t[0] else GRADES[1] if s <= t[1] else GRADES[2]


def save_calibration(cal: dict, path: str | Path) -> None:
    Path(path).write_text(json.dumps(cal, indent=2))


def load_calibration(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())
