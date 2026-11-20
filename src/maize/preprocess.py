"""Photo -> model input and leaf mask, shared by training, evaluation and the web app so all three see the
same pixels. Only numpy + OpenCV."""
from __future__ import annotations

import cv2
import numpy as np

CLS_SIZE = 224  # classifier input
AE_SIZE = 128   # autoencoder input

# Leaf mask: GrabCut grown out from the middle of the photo, where the farmer aims the camera.
SEED_FRACTION = 0.12     # ellipse around the middle that is surely leaf
LIKELY_FRACTION = 0.30   # larger ellipse that starts as "probably leaf"
GRABCUT_ITERATIONS = 10
MASK_VERSION = 2         # bump when leaf_mask changes, so cached masks are recomputed
MASK_OK = (0.10, 0.95)  # leaf fraction outside this range: no leaf found, or the background was kept

# Error blur in pixels. "smoothed" ignores fine texture (veins, edges) and keeps colour and tone.
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
    """Pad with black to size x size, image centred."""
    h, w = image.shape[:2]
    top, left = (size - h) // 2, (size - w) // 2
    return cv2.copyMakeBorder(image, top, size - h - top, left, size - w - left, cv2.BORDER_CONSTANT, value=0)


def resize(rgb: np.ndarray, size: int) -> np.ndarray:
    """RGB photo -> size x size model input, aspect ratio kept."""
    return _pad(_fit(rgb, size), size)


def prepare(rgb: np.ndarray, size: int) -> tuple[np.ndarray, np.ndarray]:
    """RGB photo -> (model input, leaf mask 0/1 of the same size)."""
    image = resize(rgb, size)
    return image, padded_leaf_mask(image)


def padded_leaf_mask(image: np.ndarray) -> np.ndarray:
    """Leaf mask of a resized image, found inside the black padding; 0 on the padding."""
    rows = np.flatnonzero(image.any(axis=(1, 2)))
    cols = np.flatnonzero(image.any(axis=(0, 2)))
    mask = np.zeros(image.shape[:2], np.uint8)
    if len(rows):
        photo = (slice(rows[0], rows[-1] + 1), slice(cols[0], cols[-1] + 1))
        mask[photo] = leaf_mask(image[photo])
    return mask


def _ellipse(k: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))


def fill_holes(mask: np.ndarray) -> np.ndarray:
    """Fill background the border cannot reach (e.g. a lesion inside the leaf)."""
    padded = np.pad(mask, 1)
    flooded = padded.copy()
    cv2.floodFill(flooded, np.zeros((padded.shape[0] + 2, padded.shape[1] + 2), np.uint8), (0, 0), 1)
    holes = flooded == 0
    return (padded | holes)[1:-1, 1:-1].astype(np.uint8)


def central_component(mask: np.ndarray) -> np.ndarray:
    """The connected piece that holds the middle of the image (empty if none)."""
    labels = cv2.connectedComponents(mask, connectivity=8)[1]
    middle = labels[mask.shape[0] // 2, mask.shape[1] // 2]
    return ((labels == middle) & (middle > 0)).astype(np.uint8)


def _centre_ellipse(labels: np.ndarray, fraction: float, value: int) -> None:
    h, w = labels.shape
    axes = (max(1, int(w * fraction)), max(1, int(h * fraction)))
    cv2.ellipse(labels, (w // 2, h // 2), axes, 0, 0, 360, value, -1)


def grabcut_from_centre(rgb: np.ndarray) -> np.ndarray:
    """GrabCut seeded with the middle as leaf and the rest as probably background. 0/1."""
    labels = np.full(rgb.shape[:2], cv2.GC_PR_BGD, np.uint8)
    _centre_ellipse(labels, LIKELY_FRACTION, cv2.GC_PR_FGD)
    _centre_ellipse(labels, SEED_FRACTION, cv2.GC_FGD)
    cv2.setRNGSeed(0)  # same photo, same mask
    cv2.grabCut(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), labels, None, np.zeros((1, 65)), np.zeros((1, 65)),
                GRABCUT_ITERATIONS, cv2.GC_INIT_WITH_MASK)
    return np.isin(labels, (cv2.GC_FGD, cv2.GC_PR_FGD)).astype(np.uint8)


def leaf_mask(rgb: np.ndarray) -> np.ndarray:
    """The photographed leaf as a 0/1 mask: GrabCut, specks removed, central piece kept, holes filled."""
    mask = cv2.morphologyEx(grabcut_from_centre(rgb), cv2.MORPH_OPEN, _ellipse(3))
    return fill_holes(central_component(mask))


def leaf_fraction(mask: np.ndarray) -> float:
    """Share of the image that is leaf."""
    return float(mask.mean())


def usable_masks(masks: np.ndarray) -> np.ndarray:
    """True per image whose leaf fraction is inside MASK_OK. masks (N,H,W)."""
    fractions = masks.mean(axis=(1, 2))
    return (fractions >= MASK_OK[0]) & (fractions <= MASK_OK[1])


def mask_background(images: np.ndarray, masks: np.ndarray) -> np.ndarray:
    """Pixels outside the leaf set to 0. images (N,H,W,3), masks (N,H,W)."""
    return images * masks[..., None]


def blur_inside(images: np.ndarray, masks: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian blur that mixes leaf pixels only with leaf pixels."""
    out = np.empty_like(images)
    for i, (image, mask) in enumerate(zip(images, masks.astype(np.float32))):
        weight = cv2.GaussianBlur(mask, (0, 0), sigma)[..., None]
        out[i] = cv2.GaussianBlur(image * mask[..., None], (0, 0), sigma) / np.maximum(weight, 1e-6)
    return out


def error_map(images: np.ndarray, recons: np.ndarray, masks: np.ndarray, sigma: float = 0.0) -> np.ndarray:
    """Squared error per pixel, 0 outside the leaf. images 0-255, recons 0-1; batched. sigma > 0 blurs both first."""
    images, recons = (images / 255.0).astype(np.float32), recons.astype(np.float32)
    if sigma:
        images, recons = blur_inside(images, masks, sigma), blur_inside(recons, masks, sigma)
    return ((images - recons) ** 2).mean(axis=-1) * masks


def reconstruction_error(images: np.ndarray, recons: np.ndarray, masks: np.ndarray, sigma: float = 0.0) -> np.ndarray:
    """Mean squared error over leaf pixels only, per image."""
    return error_map(images, recons, masks, sigma).sum(axis=(1, 2)) / np.maximum(masks.sum(axis=(1, 2)), 1)
