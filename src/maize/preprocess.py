"""Photo -> model input. Shared by training and the web app so both see identical pixels.
Only numpy + OpenCV, because the web app has no torch or Keras."""
from __future__ import annotations

import cv2
import numpy as np

CLS_SIZE = 224  # classifier input (ImageNet backbones)
AE_SIZE = 128   # autoencoder input


def decode(data: bytes) -> np.ndarray | None:
    """Image bytes -> RGB uint8 array, or None if unreadable."""
    bgr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    return None if bgr is None else cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def resize(rgb: np.ndarray, size: int) -> np.ndarray:
    """Pad to a square with black, then resize, so the leaf keeps its shape (proposal 3.4)."""
    h, w = rgb.shape[:2]
    side = max(h, w)
    top, left = (side - h) // 2, (side - w) // 2
    square = cv2.copyMakeBorder(rgb, top, side - h - top, left, side - w - left, cv2.BORDER_CONSTANT, value=0)
    return cv2.resize(square, (size, size), interpolation=cv2.INTER_AREA)
