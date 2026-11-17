"""Cache of preprocessed images and leaf masks, so training does not re-read full-size photos every run.

Per size (128 for the autoencoder, 224 for the classifier), two files in the data root:
    cache_<size>.npz   images, uint8 (N, size, size, 3)
    mask_<size>.npz    leaf masks, uint8 0/1 (N, size, size)
both keyed by the image's sha1 from the manifest. Each photo is read once and gives both. New images
are added on demand; cached ones are never re-read. If you change preprocess.py, delete the cache files.
"""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from .preprocess import prepare


class ArrayFile:
    """One .npz cache: arrays keyed by sha1, loaded into memory, saved atomically."""

    def __init__(self, path: Path, item_shape: tuple):
        self.path = path
        self.keys, self.arrays = [], np.zeros((0, *item_shape), np.uint8)
        if path.exists():
            with np.load(path) as z:
                self.keys, self.arrays = list(z["sha1"]), z["x"]

    def replace(self, keys: list[str], arrays: np.ndarray) -> None:
        """Add these entries, dropping any older entry with the same key, and save."""
        keep = ~np.isin(self.keys, keys)
        self.keys = [k for k, kept in zip(self.keys, keep) if kept] + list(keys)
        self.arrays = np.concatenate([self.arrays[keep], arrays])
        tmp = self.path.with_name(self.path.stem + ".tmp.npz")  # write then rename: a crash cannot corrupt it
        np.savez(tmp, sha1=np.array(self.keys), x=self.arrays)
        os.replace(tmp, self.path)

    def get(self, keys: pd.Series) -> np.ndarray:
        position = {k: i for i, k in enumerate(self.keys)}
        return self.arrays[[position[k] for k in keys]]


def _read(path: str, size: int) -> tuple[np.ndarray, np.ndarray]:
    bgr = cv2.imread(path)
    if bgr is None:
        raise FileNotFoundError(f"Could not read {path}")
    return prepare(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), size)


def load(rows: pd.DataFrame, root: Path, size: int, workers: int = 8) -> tuple[np.ndarray, np.ndarray]:
    """(images, masks) for the manifest rows, in the same order. Reads only photos missing from either cache."""
    images = ArrayFile(Path(root) / f"cache_{size}.npz", (size, size, 3))
    masks = ArrayFile(Path(root) / f"mask_{size}.npz", (size, size))

    missing = rows[~rows["sha1"].isin(set(images.keys) & set(masks.keys))].drop_duplicates("sha1")
    if len(missing):
        new_images = np.empty((len(missing), size, size, 3), np.uint8)
        new_masks = np.empty((len(missing), size, size), np.uint8)
        paths = [str(Path(root) / p) for p in missing["path"]]
        with ThreadPoolExecutor(workers) as pool:
            done = pool.map(lambda p: _read(p, size), paths)
            for i, (image, mask) in enumerate(tqdm(done, total=len(paths), desc=f"caching {size}px")):
                new_images[i], new_masks[i] = image, mask
        images.replace(list(missing["sha1"]), new_images)
        masks.replace(list(missing["sha1"]), new_masks)

    return images.get(rows["sha1"]), masks.get(rows["sha1"])


def load_images(rows: pd.DataFrame, root: Path, size: int) -> np.ndarray:
    return load(rows, root, size)[0]


def load_masks(rows: pd.DataFrame, root: Path, size: int) -> np.ndarray:
    return load(rows, root, size)[1]
