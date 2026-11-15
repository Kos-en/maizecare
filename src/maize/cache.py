"""Cache of preprocessed images, so training does not re-read full-size photos every run.

One file per size, <data root>/cache_<size>.npz (128 for the autoencoder, 224 for the classifier),
holds every image resized to that size (uint8), keyed by the image's sha1 from the manifest. New
images are added on demand; existing ones are never re-read. If you change preprocess.resize,
delete the cache files.
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


def _read(path: str, size: int) -> np.ndarray:
    bgr = cv2.imread(path)
    if bgr is None:
        raise FileNotFoundError(f"Could not read {path}")
    return resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), size)


def load_images(rows: pd.DataFrame, root: Path, size: int = IMG_SIZE, workers: int = 8) -> np.ndarray:
    """(N,size,size,3) uint8 for the manifest rows, in the same order. Reads only uncached images."""
    f = root / f"cache_{size}.npz"
    if f.exists():
        with np.load(f) as z:
            keys, imgs = list(z["sha1"]), z["x"]
    else:
        keys, imgs = [], np.zeros((0, size, size, 3), np.uint8)
    pos = {k: i for i, k in enumerate(keys)}

    missing = rows[~rows["sha1"].isin(pos)]
    if len(missing):
        paths = [str(root / p) for p in missing["path"]]
        with ThreadPoolExecutor(workers) as pool:
            new = list(tqdm(pool.map(lambda p: _read(p, size), paths), total=len(paths), desc=f"caching {size}px"))
            new = np.stack(new)
        keys += list(missing["sha1"])
        imgs = np.concatenate([imgs, new])
        tmp = f.with_name(f.stem + ".tmp.npz")  # write then rename so a crash cannot corrupt the cache
        np.savez(tmp, sha1=np.array(keys), x=imgs)
        os.replace(tmp, f)
        pos = {k: i for i, k in enumerate(keys)}

    return imgs[[pos[s] for s in rows["sha1"]]]
