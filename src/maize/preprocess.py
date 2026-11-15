"""Single entry point that turns a raw RGB photo into the model-ready (image, mask) pair.

Used by scripts/04_preprocess.py (training cache) and serving/pipeline.py (web service),
so a field photo is processed exactly like the training images.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .constants import CACHE_SIZE
from .masking import DEFAULT_PARAMS, MASK_VERSION, MaskParams, leaf_fraction, leaf_mask

MAX_SIDE = 1024  # photos larger than this are downscaled first (same in train and serve)


def downscale(rgb: np.ndarray, max_side: int = MAX_SIDE) -> np.ndarray:
    h, w = rgb.shape[:2]
    s = max_side / max(h, w)
    if s >= 1:
        return rgb
    return cv2.resize(rgb, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)


def pad_square_resize(img: np.ndarray, size: int, interp: int = cv2.INTER_AREA) -> np.ndarray:
    """Pad with zeros to a square (leaf is not stretched), then resize."""
    h, w = img.shape[:2]
    s = max(h, w)
    top, left = (s - h) // 2, (s - w) // 2
    pad = ((top, s - h - top), (left, s - w - left)) + ((0, 0),) * (img.ndim - 2)
    return cv2.resize(np.pad(img, pad), (size, size), interpolation=interp)


@dataclass
class Prepared:
    image: np.ndarray        # size x size x 3 uint8 RGB (background NOT removed)
    mask: np.ndarray         # size x size uint8 {0,1}
    leaf_fraction: float
    mask_version: str = MASK_VERSION


def prepare(rgb: np.ndarray, size: int = CACHE_SIZE, params: MaskParams = DEFAULT_PARAMS) -> Prepared:
    rgb = downscale(rgb)
    mask = leaf_mask(rgb, params)
    return Prepared(
        image=pad_square_resize(rgb, size, cv2.INTER_AREA),
        mask=pad_square_resize(mask, size, cv2.INTER_NEAREST),
        leaf_fraction=leaf_fraction(mask),
    )
