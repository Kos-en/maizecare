"""Manifest access. The manifest (data/manifest.csv) is the single source of truth for which
images exist, what class they are, which near-duplicate group they belong to, and whether
they are train or val. Torch-free on purpose."""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

IMG_SIZE = 128
HEALTHY = "Healthy"

COLUMNS = ["path", "label", "sha1", "phash", "ahash", "group", "split", "status", "added_at"]
# status: ok | duplicate_exact | bad_image | too_small | label_conflict | split_conflict
# Only status == "ok" is ever used for training / evaluation.


def data_root(override: str | None = None) -> Path:
    return Path(override or os.environ.get("MAIZE_DATA_ROOT", "data"))


def manifest_file(root: Path) -> Path:
    return root / "manifest.csv"


def load_manifest(root: Path) -> pd.DataFrame:
    f = manifest_file(root)
    if not f.exists():
        return pd.DataFrame(columns=COLUMNS)
    return pd.read_csv(f, dtype={"phash": str, "ahash": str, "sha1": str})


def save_manifest(df: pd.DataFrame, root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    df[COLUMNS].to_csv(manifest_file(root), index=False)


def select(df: pd.DataFrame, root: Path, label: str | None = None, split: str | None = None,
           exclude_label: str | None = None) -> list[str]:
    """Absolute paths of accepted images, optionally filtered."""
    m = df["status"] == "ok"
    if label is not None:
        m &= df["label"] == label
    if exclude_label is not None:
        m &= df["label"] != exclude_label
    if split is not None:
        m &= df["split"] == split
    return [str(root / p) for p in df.loc[m, "path"]]
