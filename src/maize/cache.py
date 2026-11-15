"""Cache of preprocessed images, so training does not re-read full-size photos every run.

One file, <data root>/cache_128.npz, holds every image resized to 128x128 (uint8), keyed by the
image's sha1 from the manifest. New images are added on demand; existing ones are never re-read.
If you change the preprocessing in preprocess.resize, delete the cache file.
"""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from .preprocess import IMG_SIZE, resize


def _read(path: str) -> np.ndarray:
    bgr = cv2.imread(path)
    if bgr is None:
        raise FileNotFoundError(f"Could not read {path}")
    return resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


def load_images(rows: pd.DataFrame, root: Path, workers: int = 8) -> np.ndarray:
    """(N,128,128,3) uint8 for the manifest rows, in the same order. Reads only uncached images."""
    f = root / f"cache_{IMG_SIZE}.npz"
    if f.exists():
        with np.load(f) as z:
            keys, imgs = list(z["sha1"]), z["x"]
    else:
        keys, imgs = [], np.zeros((0, IMG_SIZE, IMG_SIZE, 3), np.uint8)
    pos = {k: i for i, k in enumerate(keys)}

    missing = rows[~rows["sha1"].isin(pos)]
    if len(missing):
        paths = [str(root / p) for p in missing["path"]]
        with ThreadPoolExecutor(workers) as pool:
            new = np.stack(list(tqdm(pool.map(_read, paths), total=len(paths), desc="caching images")))
        keys += list(missing["sha1"])
        imgs = np.concatenate([imgs, new])
        tmp = f.with_name(f.stem + ".tmp.npz")  # write then rename so a crash cannot corrupt the cache
        np.savez(tmp, sha1=np.array(keys), x=imgs)
        os.replace(tmp, f)
        pos = {k: i for i, k in enumerate(keys)}

    return imgs[[pos[s] for s in rows["sha1"]]]
