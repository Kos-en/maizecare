"""Photo -> model input and leaf mask. Shared by training, evaluation and the web app, so all three
see identical pixels. Only numpy + OpenCV, because the web app has no torch or Keras.

Resizing (proposal 3.4): scale the longest side to the model size, then pad to a square with black,
so the leaf keeps its shape. The leaf mask is computed on the scaled photo before padding, so the
black padding is never "leaf" and the mask lines up pixel for pixel with the model input.
"""
from __future__ import annotations

import cv2
import numpy as np

CLS_SIZE = 224  # classifier input (ImageNet backbones)
AE_SIZE = 128   # autoencoder input

# Yellow-to-brown tissue (chlorosis, lesions) that may belong to the leaf. OpenCV hue runs 0-179,
# so 5-35 is roughly 10-70 degrees: orange-brown through yellow. Low saturation/value is soil or shadow.
LESION_HSV_LOW = (5, 40, 30)
LESION_HSV_HIGH = (35, 255, 255)
TOUCH_KERNEL = 7  # lesion pixels this close to green tissue count as leaf
CLOSE_KERNEL = 5  # closes small gaps along the leaf edge
MASK_OK = (0.10, 0.95)  # leaf fraction outside this range: no leaf found, or the background was kept

# How the autoencoder's error is measured (proposal eq. 3.2), by name -> blur in pixels at AE_SIZE.
# "plain" compares every pixel. "smoothed" first blurs the photo and the reconstruction inside the leaf,
# so fine texture the autoencoder cannot rebuild anyway (veins, edges) stops counting and colour and tone
# (yellowing, browning, dead tissue) remain. 4 px is about the detail an 8 x 8 code can carry at 128 px.
ERRORS = {"plain": 0.0, "smoothed": 4.0}


def decode(data: bytes) -> np.ndarray | None:
    """Image bytes -> RGB uint8 array, or None if unreadable."""
    bgr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    return None if bgr is None else cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _fit(rgb: np.ndarray, size: int) -> np.ndarray:
    """Scale so the longest side is `size` pixels, keeping the aspect ratio."""
    h, w = rgb.shape[:2]
    scale = size / max(h, w)
    new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
    return cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)


def _pad(image: np.ndarray, size: int) -> np.ndarray:
    """Pad with zeros (black / not leaf) to size x size, image centred."""
    h, w = image.shape[:2]
    top, left = (size - h) // 2, (size - w) // 2
    return cv2.copyMakeBorder(image, top, size - h - top, left, size - w - left, cv2.BORDER_CONSTANT, value=0)


def resize(rgb: np.ndarray, size: int) -> np.ndarray:
    """RGB photo -> size x size RGB uint8 model input."""
    return _pad(_fit(rgb, size), size)


def prepare(rgb: np.ndarray, size: int) -> tuple[np.ndarray, np.ndarray]:
    """RGB photo -> (model input like resize(), leaf mask 0/1 of the same size; padding is 0)."""
    scaled = _fit(rgb, size)
    return _pad(scaled, size), _pad(leaf_mask(scaled), size)


def _ellipse(k: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))


def green_mask(rgb: np.ndarray) -> np.ndarray:
    """Excess Green on chromatic coordinates (ExG = 2g - r - b), thresholded with Otsu. 0/1."""
    f = rgb.astype(np.float32)
    total = f.sum(axis=-1) + 1e-6
    r, g, b = f[..., 0] / total, f[..., 1] / total, f[..., 2] / total
    exg = 2 * g - r - b                                             # between -1 and 2
    exg8 = np.clip((exg + 1) / 3 * 255, 0, 255).astype(np.uint8)
    _, mask = cv2.threshold(exg8, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return mask


def fill_holes(mask: np.ndarray) -> np.ndarray:
    """Background not reachable from the border is a hole in the leaf (e.g. a lesion): fill it."""
    padded = np.pad(mask, 1)  # a background frame, so the flood always starts outside the leaf
    flooded = padded.copy()
    cv2.floodFill(flooded, np.zeros((padded.shape[0] + 2, padded.shape[1] + 2), np.uint8), (0, 0), 1)
    holes = flooded == 0
    return (padded | holes)[1:-1, 1:-1].astype(np.uint8)


def largest_component(mask: np.ndarray) -> np.ndarray:
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:  # nothing found
        return mask
    biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (labels == biggest).astype(np.uint8)


def leaf_mask(rgb: np.ndarray) -> np.ndarray:
    """Leaf pixels (uint8 0/1, same H x W): green tissue plus yellow/brown tissue touching it (proposal 3.4)."""
    green = green_mask(rgb)
    lesion = (cv2.inRange(cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV), LESION_HSV_LOW, LESION_HSV_HIGH) > 0).astype(np.uint8)
    near_green = cv2.dilate(green, _ellipse(TOUCH_KERNEL))
    mask = green | (lesion & near_green)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _ellipse(CLOSE_KERNEL))
    return largest_component(fill_holes(mask))


def leaf_fraction(mask: np.ndarray) -> float:
    """Share of the image that is leaf. Below 0.10: no leaf found; above 0.95: background not removed."""
    return float(mask.mean())


def usable_masks(masks: np.ndarray) -> np.ndarray:
    """True per image whose leaf fraction is inside MASK_OK. masks (N,H,W). Others are left out of scoring."""
    fractions = masks.mean(axis=(1, 2))
    return (fractions >= MASK_OK[0]) & (fractions <= MASK_OK[1])


def mask_background(images: np.ndarray, masks: np.ndarray) -> np.ndarray:
    """Autoencoder input: pixels outside the leaf set to 0. images (N,H,W,3), masks (N,H,W)."""
    return images * masks[..., None]


def blur_inside(images: np.ndarray, masks: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian blur that only mixes leaf pixels with leaf pixels, so the leaf edge does not fade into
    the background. images float (N,H,W,3), masks 0/1 (N,H,W)."""
    out = np.empty_like(images)
    for i, (image, mask) in enumerate(zip(images, masks.astype(np.float32))):
        weight = cv2.GaussianBlur(mask, (0, 0), sigma)[..., None]
        out[i] = cv2.GaussianBlur(image * mask[..., None], (0, 0), sigma) / np.maximum(weight, 1e-6)
    return out


def error_map(images: np.ndarray, recons: np.ndarray, masks: np.ndarray, sigma: float = 0.0) -> np.ndarray:
    """Squared error per pixel (mean of R, G, B), 0 outside the leaf. sigma > 0: compare blurred versions.
    images uint8 0-255, recons 0-1, masks 0/1; all batched (N, ...). Returns (N,H,W)."""
    images, recons = (images / 255.0).astype(np.float32), recons.astype(np.float32)
    if sigma:
        images, recons = blur_inside(images, masks, sigma), blur_inside(recons, masks, sigma)
    return ((images - recons) ** 2).mean(axis=-1) * masks


def reconstruction_error(images: np.ndarray, recons: np.ndarray, masks: np.ndarray, sigma: float = 0.0) -> np.ndarray:
    """e(x) per image (proposal eq. 3.2): mean squared error over leaf pixels only."""
    return error_map(images, recons, masks, sigma).sum(axis=(1, 2)) / np.maximum(masks.sum(axis=(1, 2)), 1)
