"""The manifest: one row per image with its class, near-copy group, split and status.

prepare() follows the proposal (sections 3.3-3.4). Only reading new files is incremental;
the cleaning below is redone on the whole dataset every run, so the result never depends on
the order images arrived.

  1. File check     opens as JPEG/PNG, shortest side >= 50 px      else bad_image / too_small
  2. Near copies    min(pHash, aHash) Hamming distance <= 6; chains of near copies form one group
  3. Mixed labels   a group holding two classes is dropped whole     label_conflict
  4. Exact copies   same file contents: keep one                     duplicate_exact
  5. Split          whole groups, from the group's hash: 15% test, 15% val, 70% train

Only rows with status "ok" are used for training and evaluation.
"""
from __future__ import annotations

import hashlib
import io
import os
from datetime import datetime, timezone
from pathlib import Path

import imagehash
import numpy as np
import pandas as pd
from PIL import Image
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from tqdm import tqdm

HEALTHY = "Healthy"
COLUMNS = ["path", "label", "sha1", "phash", "ahash", "group", "split", "status", "added_at"]
FILE_FAULTS = ["bad_image", "too_small"]  # facts about the file itself; every other status is recomputed

IMAGE_TYPES = {".jpg", ".jpeg", ".png"}
MIN_SIDE = 50
HAMMING = 6
TEST_FRAC = 0.15
VAL_FRAC = 0.15
SAVE_EVERY = 1000  # save while reading, so an interrupted run resumes instead of starting over


# ---- manifest file ------------------------------------------------------------------------------
def load_manifest(root: Path) -> pd.DataFrame:
    path = Path(root) / "manifest.csv"
    if not path.exists():
        return pd.DataFrame(columns=COLUMNS)
    return pd.read_csv(path, dtype={"sha1": str, "phash": str, "ahash": str})


def save_manifest(df: pd.DataFrame, root: Path) -> None:
    Path(root).mkdir(parents=True, exist_ok=True)
    df[COLUMNS].to_csv(Path(root) / "manifest.csv", index=False)


def select(df: pd.DataFrame, label: str | None = None, split: str | None = None) -> pd.DataFrame:
    """Usable images, optionally only one class and/or one split."""
    keep = df["status"] == "ok"
    if label is not None:
        keep &= df["label"] == label
    if split is not None:
        keep &= df["split"] == split
    return df[keep]


def data_version(df: pd.DataFrame) -> str:
    """Short hash of which images are usable and their split, to trace a model back to its data."""
    ok = select(df)
    lines = sorted(ok["sha1"] + ":" + ok["split"])
    return hashlib.sha1("\n".join(lines).encode()).hexdigest()[:12]


# ---- step 1: read files -------------------------------------------------------------------------
def inspect(path: Path) -> dict:
    """Content hash, perceptual hashes and file-check status of one image."""
    try:
        data = path.read_bytes()
        with Image.open(io.BytesIO(data)) as im:
            im.load()
            status = "ok" if min(im.size) >= MIN_SIDE else "too_small"
            return {"sha1": hashlib.sha1(data).hexdigest(), "phash": str(imagehash.phash(im)),
                    "ahash": str(imagehash.average_hash(im)), "status": status}
    except Exception:
        return {"sha1": "", "phash": "", "ahash": "", "status": "bad_image"}


def scan(raw: Path, root: Path) -> pd.DataFrame:
    """The manifest plus a row for every image in raw/<Class>/ that is not in it yet.
    Only those new files are read; the class is the folder name."""
    old = load_manifest(root)
    known = set(old["path"])
    files = sorted(p for p in raw.rglob("*") if p.suffix.lower() in IMAGE_TYPES and p.parent != raw)
    new_files = [p for p in files if relative(p, root) not in known]
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")

    rows = old.to_dict("records")
    for i, p in enumerate(tqdm(new_files, desc="reading new images", unit="img"), 1):
        rows.append({"path": relative(p, root), "label": p.relative_to(raw).parts[0], "added_at": now, **inspect(p)})
        if i % SAVE_EVERY == 0:
            save_manifest(pd.DataFrame(rows, columns=COLUMNS), root)
    print(f"{len(files)} images in {raw}, {len(new_files)} new.")
    return pd.DataFrame(rows, columns=COLUMNS)


def relative(path: Path, root: Path) -> str:
    """Manifest paths are relative to the data root, so the data folder can move."""
    return os.path.relpath(path.resolve(), Path(root).resolve())


# ---- steps 2-5: clean and split -----------------------------------------------------------------
def popcount(x: np.ndarray) -> np.ndarray:
    """Number of 1-bits in each uint64 (the Hamming distance, when x is an XOR of two hashes)."""
    bits_in_byte = np.array([bin(i).count("1") for i in range(256)], np.uint8)
    return bits_in_byte[x.view(np.uint8).reshape(*x.shape, 8)].sum(-1)


def near_pairs(df: pd.DataFrame, hamming: int = HAMMING, chunk: int = 512) -> np.ndarray:
    """Row-position pairs (i, j), i < j, of images that are near copies of each other."""
    ph = np.array([int(h, 16) for h in df["phash"]], np.uint64)
    ah = np.array([int(h, 16) for h in df["ahash"]], np.uint64)
    pairs = [np.empty((0, 2), np.int64)]
    for start in range(0, len(df), chunk):  # compare a chunk of images against all images
        dist = np.minimum(popcount(ph[start:start + chunk, None] ^ ph), popcount(ah[start:start + chunk, None] ^ ah))
        i, j = np.nonzero(dist <= hamming)
        i += start
        pairs.append(np.stack([i, j], axis=1)[i < j])
    return np.concatenate(pairs)


def group_ids(n: int, pairs: np.ndarray) -> np.ndarray:
    """Connected components: if A~B and B~C, then A, B and C are one group."""
    graph = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n))
    return connected_components(graph, directed=False)[1]


def split_of(sha1: str) -> str:
    """Split from a hash, so a group lands in the same split on every run."""
    u = int(sha1[:8], 16) / 16 ** 8  # uniform in [0, 1)
    if u < TEST_FRAC:
        return "test"
    if u < TEST_FRAC + VAL_FRAC:
        return "val"
    return "train"


def clean(df: pd.DataFrame, hamming: int = HAMMING) -> pd.DataFrame:
    """Steps 2-5 on the whole manifest: recompute group, status and split for every row."""
    df = df.sort_values("path", ignore_index=True)
    df["status"] = df["status"].where(df["status"].isin(FILE_FAULTS), "ok")

    readable = df.index[df["status"] == "ok"]
    df["group"] = -1
    df.loc[readable, "group"] = group_ids(len(readable), near_pairs(df.loc[readable], hamming))

    labels_in_group = df.loc[readable].groupby("group")["label"].transform("nunique")
    df.loc[labels_in_group[labels_in_group > 1].index, "status"] = "label_conflict"
    df.loc[(df["status"] == "ok") & df.duplicated("sha1"), "status"] = "duplicate_exact"

    usable = df["status"] == "ok"
    group_key = df[usable].groupby("group")["sha1"].transform("min")  # one hash per group
    df["split"] = ""
    df.loc[usable, "split"] = group_key.map(split_of)
    return df


def prepare(raw: Path, root: Path, hamming: int = HAMMING) -> pd.DataFrame:
    df = clean(scan(Path(raw), Path(root)), hamming)
    save_manifest(df, root)
    return df


def count_leaks(df: pd.DataFrame, hamming: int = HAMMING) -> int:
    """Near-copy pairs of usable images that sit in different splits. Must be 0."""
    ok = select(df)
    pairs = near_pairs(ok, hamming)
    split = ok["split"].to_numpy()
    return int((split[pairs[:, 0]] != split[pairs[:, 1]]).sum())


def report(df: pd.DataFrame) -> None:
    print("\nImages by class and status:")
    print(df.groupby(["label", "status"]).size().unstack(fill_value=0))
    ok = select(df)
    table = ok.groupby(["label", "split"]).size().unstack(fill_value=0)
    table["groups"] = ok.groupby("label")["group"].nunique()
    print("\nUsable images by class and split:")
    print(table)
    print(f"\nLeak check: {count_leaks(df)} near-copy pairs cross splits (must be 0).")
