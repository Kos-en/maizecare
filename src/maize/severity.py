"""Symptomatic area A(x) (proposal 3.6, eq. 3.3): the share of the leaf that shows symptoms.

    A(x) = 100 * |S| / |M|    M = leaf pixels (the leaf mask), S = leaf pixels in a symptom colour range

A(x) is the reference the autoencoder score is checked against; it is never used to train anything.
The colour ranges below were set before any autoencoder score was seen and must not be tuned on them.
"""
from __future__ import annotations

import cv2
import numpy as np

# OpenCV HSV: hue 0-179 (degrees / 2), saturation and value 0-255. Healthy green is hue ~35-85.
SYMPTOM_RANGES = {
    "yellowing": ((20, 60, 80), (34, 255, 255)),     # chlorosis: yellow, saturated, fairly bright
    "brown lesion": ((5, 50, 30), (19, 255, 200)),   # necrosis: orange-brown to dark brown
    "dead tissue": ((10, 0, 120), (30, 60, 255)),    # dried, straw-coloured tissue: pale, low saturation
}


def symptom_pixels(rgb: np.ndarray) -> np.ndarray:
    """True where a pixel falls in any symptom colour range. rgb: H x W x 3 uint8."""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    found = np.zeros(rgb.shape[:2], bool)
    for low, high in SYMPTOM_RANGES.values():
        found |= cv2.inRange(hsv, low, high) > 0
    return found


def symptomatic_area(rgb: np.ndarray, mask: np.ndarray) -> float:
    """A(x) in percent (0-100): symptom pixels inside the leaf / leaf pixels."""
    leaf = mask > 0
    return float(100 * (symptom_pixels(rgb) & leaf).sum() / max(leaf.sum(), 1))
