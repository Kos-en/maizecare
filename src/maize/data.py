"""Read/write data/manifest.csv: one row per image (class, near-duplicate group, train/val split)."""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

HEALTHY = "Healthy"

COLUMNS = ["path", "label", "sha1", "phash", "ahash", "group", "split", "status", "added_at"]
# status: ok | duplicate_exact | bad_image | too_small | label_conflict | split_conflict
# Only "ok" rows are used for training and evaluation.


def data_root(override: str | None = None) -> Path:
    return Path(override or os.environ.get("MAIZE_DATA_ROOT", "data"))


def load_manifest(root: Path) -> pd.DataFrame:
    f = root / "manifest.csv"
    if not f.exists():
        return pd.DataFrame(columns=COLUMNS)
    return pd.read_csv(f, dtype={"phash": str, "ahash": str, "sha1": str})


def save_manifest(df: pd.DataFrame, root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    df[COLUMNS].to_csv(root / "manifest.csv", index=False)


def select(df: pd.DataFrame, root: Path, label: str | None = None, split: str | None = None) -> list[str]:
    """Absolute paths of accepted images, optionally filtered."""
    m = df["status"] == "ok"
    if label is not None:
        m &= df["label"] == label
    if split is not None:
        m &= df["split"] == split
    return [str(root / p) for p in df.loc[m, "path"]]
