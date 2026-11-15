"""Photo -> model input. Shared by training and the web app so both see identical pixels.
Only numpy + OpenCV, because the serving image has no torch."""
from __future__ import annotations

import cv2
import numpy as np

IMG_SIZE = 128


def decode(data: bytes) -> np.ndarray | None:
    """Image bytes -> RGB uint8 array, or None if unreadable."""
    bgr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    return None if bgr is None else cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def resize(rgb: np.ndarray) -> np.ndarray:
    """RGB uint8 HxWx3 -> RGB uint8 128x128x3 (this is what the image cache stores)."""
    return cv2.resize(rgb, (IMG_SIZE, IMG_SIZE))


def to_tensor(img: np.ndarray) -> np.ndarray:
    """Resized uint8 HxWx3 -> float32 3x128x128 in [0,1]."""
    return np.ascontiguousarray((img.astype(np.float32) / 255.0).transpose(2, 0, 1))


def to_input(rgb: np.ndarray) -> np.ndarray:
    """Full-size RGB photo -> model input."""
    return to_tensor(resize(rgb))
