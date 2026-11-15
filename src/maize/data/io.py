"""Manifest + cache loading shared by scripts. The manifest stores paths RELATIVE to
$MAIZE_DATA_ROOT so the same file works locally, on Drive and on Colab's /content."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from ..config import data_root
from ..constants import AE_SIZE
from ..masking import apply_mask
from .degrade import DEGRADATIONS


def manifest_path() -> Path:
    return data_root() / "manifest.csv"


def load_manifest(path: str | Path | None = None) -> pd.DataFrame:
    df = pd.read_csv(path or manifest_path())
    root = data_root()
    df["abs_path"] = df["path"].map(lambda p: str(root / p))
    if "rel_proc" in df:
        ok = df["rel_proc"].notna()
        df.loc[ok, "proc_path"] = df.loc[ok, "rel_proc"].map(lambda p: str(root / p))
        df.loc[ok, "mask_path"] = df.loc[ok, "rel_mask"].map(lambda p: str(root / p))
    return df


def save_manifest(df: pd.DataFrame, path: str | Path | None = None) -> None:
    drop = [c for c in ("abs_path", "proc_path", "mask_path") if c in df]
    df.drop(columns=drop).to_csv(path or manifest_path(), index=False)


def load_proc(row) -> tuple[np.ndarray, np.ndarray]:
    img = cv2.cvtColor(cv2.imread(row["proc_path"], cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    msk = cv2.imread(row["mask_path"], cv2.IMREAD_GRAYSCALE)
    return img, msk


def load_clf_array(df: pd.DataFrame, masked: bool, degradation: str = "clean") -> np.ndarray:
    """(N,224,224,3) float32 raw 0-255, exactly what the classifier / TFLite model receives."""
    deg = DEGRADATIONS[degradation]
    out = []
    for _, r in df.iterrows():
        img, msk = load_proc(r)
        img = deg(img)
        out.append(apply_mask(img, msk) if masked else img)
    return np.stack(out).astype(np.float32)


def load_ae_arrays(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """x (N,128,128,3) float in [0,1] with background zeroed, m (N,128,128) {0,1}.
    Same cv2 resizing as serving/pipeline.py."""
    xs, ms = [], []
    for _, r in df.iterrows():
        img, msk = load_proc(r)
        x = cv2.resize(apply_mask(img, msk), (AE_SIZE, AE_SIZE), interpolation=cv2.INTER_AREA) / 255.0
        m = cv2.resize(msk, (AE_SIZE, AE_SIZE), interpolation=cv2.INTER_NEAREST)
        xs.append((x * m[..., None]).astype(np.float32))
        ms.append(m)
    return np.stack(xs), np.stack(ms)
