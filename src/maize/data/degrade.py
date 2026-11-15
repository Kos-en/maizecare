"""Deterministic degraded copies of the (clean) test set for robustness checks."""
from __future__ import annotations

import cv2
import numpy as np


def blur(rgb: np.ndarray, sigma: float = 3.0) -> np.ndarray:
    return cv2.GaussianBlur(rgb, (0, 0), sigma)


def low_light(rgb: np.ndarray, factor: float = 0.35) -> np.ndarray:
    return np.clip(rgb.astype(np.float32) * factor, 0, 255).astype(np.uint8)


DEGRADATIONS = {"clean": lambda x: x, "blur": blur, "low_light": low_light}
