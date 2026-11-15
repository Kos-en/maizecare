"""Symptomatic leaf area A(x) = 100 * |S| / |M|  (proposal Eq. 3.3): the external check on s(x).

FREEZE RULE: the colour ranges below must be tuned on a few images by eye and then
committed BEFORE any autoencoder score is computed, so A(x) cannot be tuned to agree
with the score. Bump SYMPTOM_VERSION on any change; evaluation reports it.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

SYMPTOM_VERSION = "0.1.0-UNFROZEN"  # rename to "1.0.0" once frozen and committed


@dataclass(frozen=True)
class SymptomParams:
    # (hue_lo, hue_hi, sat_min, val_min, val_max)  OpenCV hue 0-179
    yellow: tuple = (20, 38, 70, 100, 255)   # chlorosis / mottling
    brown: tuple = (3, 20, 60, 30, 190)      # dead tissue / lesions
    # MSV streaks are fine and pale: this measure will under-estimate them (report it).


DEFAULT = SymptomParams()


def symptom_mask(rgb: np.ndarray, leaf: np.ndarray, params: SymptomParams = DEFAULT) -> np.ndarray:
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    out = np.zeros(h.shape, bool)
    for hl, hh, sm, vl, vh in (params.yellow, params.brown):
        out |= (h >= hl) & (h <= hh) & (s >= sm) & (v >= vl) & (v <= vh)
    return out & (leaf > 0)


def symptomatic_area(rgb: np.ndarray, leaf: np.ndarray, params: SymptomParams = DEFAULT) -> float:
    m = int((leaf > 0).sum())
    return 100.0 * int(symptom_mask(rgb, leaf, params).sum()) / m if m else 0.0
