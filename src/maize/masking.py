"""Leaf mask (proposal 3.4). ONE implementation, imported by training AND the web service.

Pipeline: Excess-Green + Otsu -> add yellow/brown tissue that touches green -> close,
fill holes -> largest component -> (optional GrabCut). numpy + OpenCV only.

Bump MASK_VERSION whenever behaviour changes; it is saved next to every model.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

MASK_VERSION = "1.0.0"


@dataclass(frozen=True)
class MaskParams:
    work_side: int = 512          # mask is computed at this long-side resolution
    exg_floor: float = 0.02       # never call a pixel green if ExG is below this
    hue_lo: int = 8               # OpenCV hue 0-179: brown/orange ... yellow
    hue_hi: int = 35
    sat_min: int = 50
    val_min: int = 40
    touch_frac: float = 0.04      # diseased tissue is kept only this close to green
    close_frac: float = 0.015
    grabcut: bool = False


DEFAULT_PARAMS = MaskParams()


def excess_green(rgb: np.ndarray) -> np.ndarray:
    """ExG = 2g - r - b on chromatic (sum-normalised) coordinates."""
    f = rgb.astype(np.float32)
    s = f.sum(axis=2) + 1e-6
    r, g, b = f[..., 0] / s, f[..., 1] / s, f[..., 2] / s
    return 2 * g - r - b


def _odd(n: int) -> int:
    n = max(3, int(n))
    return n if n % 2 else n + 1


def _ellipse(k: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))


def fill_holes(m: np.ndarray) -> np.ndarray:
    """Fill background regions not connected to the image border (lesions inside the leaf)."""
    m8 = (m > 0).astype(np.uint8)
    padded = np.pad(m8, 1)
    flood = (padded * 255).astype(np.uint8)
    ff_mask = np.zeros((flood.shape[0] + 2, flood.shape[1] + 2), np.uint8)
    cv2.floodFill(flood, ff_mask, (0, 0), 255)
    holes = (flood == 0)[1:-1, 1:-1]
    return (m8 | holes.astype(np.uint8)).astype(bool)


def largest_component(m: np.ndarray) -> np.ndarray:
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=8)
    if n <= 1:
        return m.astype(bool)
    idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == idx


def _grabcut(rgb: np.ndarray, m: np.ndarray) -> np.ndarray:
    gc = np.where(m, cv2.GC_PR_FGD, cv2.GC_PR_BGD).astype(np.uint8)
    core = cv2.erode(m.astype(np.uint8), _ellipse(_odd(0.03 * min(m.shape)))) > 0
    gc[core] = cv2.GC_FGD
    if not (gc == cv2.GC_FGD).any() or not (gc == cv2.GC_PR_BGD).any():
        return m
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    cv2.grabCut(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), gc, None, bgd, fgd, 3, cv2.GC_INIT_WITH_MASK)
    out = (gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD)
    return largest_component(out) if out.any() else m


def leaf_mask(rgb: np.ndarray, params: MaskParams = DEFAULT_PARAMS) -> np.ndarray:
    """rgb: HxWx3 uint8. Returns HxW uint8 {0,1} at the input resolution."""
    h, w = rgb.shape[:2]
    scale = params.work_side / max(h, w)
    small = cv2.resize(rgb, (max(1, round(w * scale)), max(1, round(h * scale))),
                       interpolation=cv2.INTER_AREA) if scale < 1 else rgb
    sh, sw = small.shape[:2]

    # 1. green tissue: Otsu on ExG, but never below an absolute floor
    exg = excess_green(small)
    lo, hi = float(exg.min()), float(exg.max())
    exg8 = cv2.normalize(exg, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    t8, _ = cv2.threshold(exg8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thr = max(lo + (t8 / 255.0) * (hi - lo), params.exg_floor)
    green = exg > thr

    # 2. candidate diseased tissue: yellow-to-brown, saturated enough
    hsv = cv2.cvtColor(small, cv2.COLOR_RGB2HSV)
    hch, sch, vch = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    cand = (hch >= params.hue_lo) & (hch <= params.hue_hi) & (sch >= params.sat_min) & (vch >= params.val_min)

    # 3. keep candidates only close to green (limits brown soil leaking in)
    k = _odd(params.touch_frac * min(sh, sw))
    near_green = cv2.dilate(green.astype(np.uint8), _ellipse(k)) > 0
    m = green | (cand & near_green)

    # 4. close gaps, fill holes, keep the leaf in focus
    kc = _odd(params.close_frac * min(sh, sw))
    m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_CLOSE, _ellipse(kc)) > 0
    m = fill_holes(m)
    m = largest_component(m)
    if params.grabcut:
        m = _grabcut(small, m)

    m8 = m.astype(np.uint8)
    if scale < 1:
        m8 = cv2.resize(m8, (w, h), interpolation=cv2.INTER_NEAREST)
    return m8


def apply_mask(rgb: np.ndarray, mask: np.ndarray, fill: int = 0) -> np.ndarray:
    out = rgb.copy()
    out[mask == 0] = fill
    return out


def leaf_fraction(mask: np.ndarray) -> float:
    return float((mask > 0).mean())


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a > 0, b > 0
    union = (a | b).sum()
    return float((a & b).sum() / union) if union else 1.0
