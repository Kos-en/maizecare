"""Photo -> model input and leaf mask. Shared by training, evaluation and the web app, so all three
see identical pixels. Only numpy + OpenCV, because the web app has no torch or Keras.

Resizing (proposal 3.4): scale the longest side to the model size, then pad to a square with black,
so the leaf keeps its shape. The leaf mask is found on the photo inside the padding (padded_leaf_mask),
so the black padding is never "leaf", the mask lines up pixel for pixel with the model input, and it
can be recomputed from a cached model input without the original photo.
"""
from __future__ import annotations

import cv2
import numpy as np

CLS_SIZE = 224  # classifier input (ImageNet backbones)
AE_SIZE = 128   # autoencoder input

# Leaf mask (proposal 3.4): GrabCut (Rother et al. 2004) grown out from the middle of the photo, because
# the farmer aims the camera at the leaf. Colour rules (Excess Green) cannot do this: the background is
# often other green leaves and weeds, and a yellowed diseased leaf is less green than the weeds around it.
SEED_FRACTION = 0.12     # an ellipse this share of the width/height around the middle is leaf for sure
LIKELY_FRACTION = 0.30   # a larger ellipse starts as "probably leaf"; everything else as "probably background"
GRABCUT_ITERATIONS = 10
MASK_VERSION = 2         # bump when leaf_mask changes, so cached masks are recomputed
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
    image = resize(rgb, size)
    return image, padded_leaf_mask(image)


def padded_leaf_mask(image: np.ndarray) -> np.ndarray:
    """Leaf mask of a model input from resize(): found on the photo inside the black padding (the
    outermost rows and columns that are all black), 0 on the padding."""
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
    """Background not reachable from the border is a hole in the leaf (e.g. a lesion): fill it."""
    padded = np.pad(mask, 1)  # a background frame, so the flood always starts outside the leaf
    flooded = padded.copy()
    cv2.floodFill(flooded, np.zeros((padded.shape[0] + 2, padded.shape[1] + 2), np.uint8), (0, 0), 1)
    holes = flooded == 0
    return (padded | holes)[1:-1, 1:-1].astype(np.uint8)


def central_component(mask: np.ndarray) -> np.ndarray:
    """The connected piece that holds the middle of the image (none: an empty mask, flagged as doubtful)."""
    labels = cv2.connectedComponents(mask, connectivity=8)[1]
    middle = labels[mask.shape[0] // 2, mask.shape[1] // 2]
    return ((labels == middle) & (middle > 0)).astype(np.uint8)


def _centre_ellipse(labels: np.ndarray, fraction: float, value: int) -> None:
    h, w = labels.shape
    axes = (max(1, int(w * fraction)), max(1, int(h * fraction)))
    cv2.ellipse(labels, (w // 2, h // 2), axes, 0, 0, 360, value, -1)


def grabcut_from_centre(rgb: np.ndarray) -> np.ndarray:
    """GrabCut learns a colour model of the leaf from the middle and of the background from the rest,
    then moves the boundary to where the colours change most. 0/1."""
    labels = np.full(rgb.shape[:2], cv2.GC_PR_BGD, np.uint8)
    _centre_ellipse(labels, LIKELY_FRACTION, cv2.GC_PR_FGD)
    _centre_ellipse(labels, SEED_FRACTION, cv2.GC_FGD)
    cv2.setRNGSeed(0)  # GrabCut starts from k-means: same photo, same mask (training, app, any thread)
    cv2.grabCut(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), labels, None, np.zeros((1, 65)), np.zeros((1, 65)),
                GRABCUT_ITERATIONS, cv2.GC_INIT_WITH_MASK)
    return np.isin(labels, (cv2.GC_FGD, cv2.GC_PR_FGD)).astype(np.uint8)


def leaf_mask(rgb: np.ndarray) -> np.ndarray:
    """The photographed leaf (uint8 0/1, same H x W): GrabCut from the middle, specks removed, the piece
    holding the middle kept, holes (e.g. lesions GrabCut left out) filled."""
    mask = cv2.morphologyEx(grabcut_from_centre(rgb), cv2.MORPH_OPEN, _ellipse(3))
    return fill_holes(central_component(mask))


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
