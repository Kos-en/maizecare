"""File checks and per-image quality stats. Also used by the web service quality gate,
so blur/brightness are computed identically in EDA and in production."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

VALID_EXT = {".jpg", ".jpeg", ".png"}
STATS_SIDE = 512  # stats are computed at this long-side size (blur is resolution dependent)


def read_rgb(path: str | Path) -> np.ndarray | None:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    return None if img is None else cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def image_stats(rgb: np.ndarray) -> dict:
    h, w = rgb.shape[:2]
    s = STATS_SIDE / max(h, w)
    small = cv2.resize(rgb, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA) if s < 1 else rgb
    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
    return {
        "w": w, "h": h,
        "blur": float(cv2.Laplacian(gray, cv2.CV_64F).var()),          # low = blurry
        "brightness": float(cv2.cvtColor(small, cv2.COLOR_RGB2HSV)[..., 2].mean()),  # mean V
    }


def check_file(path: str | Path, min_side: int = 50) -> np.ndarray | None:
    """Return the RGB image if it is a valid JPEG/PNG of at least min_side px, else None."""
    p = Path(path)
    if p.suffix.lower() not in VALID_EXT:
        return None
    rgb = read_rgb(p)
    if rgb is None or min(rgb.shape[:2]) < min_side:
        return None
    return rgb
