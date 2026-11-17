"""The data pipeline (proposal 3.3-3.4) and the manifest it writes. Every image goes through it
before any training.

    maize-prepare RAW ROOT        RAW/<Class>/*.jpg in, ROOT/manifest.csv and image caches out

  1. Read      new files only: content hash, perceptual hashes; must open and be >= 50 px   else bad_image / too_small
  2. Group     a near copy is within 6 bits on BOTH pHash and aHash; chains of near copies form one group.
               (Either hash alone chained 18,624 unrelated photos into one group on the real data, because
               aHash only sees rough brightness layout. With both required: largest group 9 photos, and
               cut-offs 2-6 give almost the same pairs, 5,171-5,293.)
  3. Conflicts a group holding two classes is dropped whole (label_conflict); exact copies keep one (duplicate_exact)
  4. Split     whole groups, stratified by class: within each class, groups in a fixed (hash) order fill
               test, then val, then train until each holds 15% / 15% / 70% of that class's images
  5. Cache     each usable image resized to 128 and 224 px, with its leaf mask (preprocess.prepare), in
               ROOT/cache_<size>.npz and ROOT/mask_<size>.npz keyed by sha1. Training reads these with
               load(), so photos are never re-read. If you change preprocess.py, delete the cache files.

Re-run after adding photos: only the new files are read. Steps 2-4 are redone on the whole dataset,
so the result never depends on the order images arrived (adding photos can move groups between
splits, so retrain after adding data).

The manifest (ROOT/manifest.csv) has one row per image: class, near-copy group, split and status.
Training and evaluation read it with load_usable_manifest() and select(); only status "ok" rows are used.
"""
from __future__ import annotations

import hashlib
import io
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import cv2
import imagehash
import numpy as np
import pandas as pd
from PIL import Image
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from tqdm import tqdm

from maize.preprocess import AE_SIZE, CLS_SIZE, prepare

HEALTHY = "Healthy"
COLUMNS = ["path", "label", "sha1", "phash", "ahash", "group", "split", "status", "added_at"]
IMAGE_TYPES = {".jpg", ".jpeg", ".png"}
FILE_FAULTS = ["bad_image", "too_small"]  # facts about the file itself; every other status is recomputed
MIN_SIDE = 50
HAMMING = 6  # bits, on both hashes
TEST_FRAC = 0.15
VAL_FRAC = 0.15
SAVE_EVERY = 1000  # save while reading, so an interrupted run resumes instead of starting over


# ---- the manifest -------------------------------------------------------------------------------
def load_manifest(root: Path) -> pd.DataFrame:
    path = Path(root) / "manifest.csv"
    if not path.exists():
        return pd.DataFrame(columns=COLUMNS)
    return pd.read_csv(path, dtype={"sha1": str, "phash": str, "ahash": str})


def save_manifest(df: pd.DataFrame, root: Path) -> None:
    Path(root).mkdir(parents=True, exist_ok=True)
    df[COLUMNS].to_csv(Path(root) / "manifest.csv", index=False)


def load_usable_manifest(root: Path) -> pd.DataFrame:
    """The manifest, or a clear message if the data pipeline has not produced any usable images yet."""
    df = load_manifest(root)
    if select(df).empty:
        raise SystemExit(f"No usable images in {root}/manifest.csv: run maize-prepare first.")
    return df


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


def command_args(usage: str, required: int) -> list[str]:
    """The words typed after a maize-* command, or the usage line if too few were given."""
    args = sys.argv[1:]
    if len(args) < required:
        raise SystemExit(f"usage: {usage}")
    return args


# ---- the pipeline -------------------------------------------------------------------------------
class DataPipeline:
    """Runs the steps in the module docstring in order on raw/<Class>/ images, keeping the manifest in self.df."""

    def __init__(self, raw: Path, root: Path, hamming: int = HAMMING):
        self.raw, self.root, self.hamming = Path(raw), Path(root), hamming
        self.df = load_manifest(self.root)
        self.pairs = np.empty((0, 2), np.int64)  # near-copy row pairs, kept for the leak check

    def run(self) -> pd.DataFrame:
        self.read_new_images()
        self.clean()
        save_manifest(self.df, self.root)
        self.report()
        self.cache()
        return self.df

    def clean(self) -> None:
        """Steps 2-4 on the whole manifest (no files needed)."""
        self.group_near_copies()
        self.drop_conflicts()
        self.split()

    # step 1
    def read_new_images(self) -> None:
        known = set(self.df["path"])
        files = sorted(p for p in self.raw.rglob("*") if p.suffix.lower() in IMAGE_TYPES and p.parent != self.raw)
        new_files = [p for p in files if self.relative(p) not in known]
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")

        rows = self.df.to_dict("records")
        for i, path in enumerate(tqdm(new_files, desc="reading new images", unit="img"), 1):
            label = path.relative_to(self.raw).parts[0]  # the class is the folder name
            rows.append({"path": self.relative(path), "label": label, "added_at": now, **inspect(path)})
            if i % SAVE_EVERY == 0:
                save_manifest(pd.DataFrame(rows, columns=COLUMNS), self.root)
        self.df = pd.DataFrame(rows, columns=COLUMNS)
        print(f"{len(files)} images in {self.raw}, {len(new_files)} new.")

    def relative(self, path: Path) -> str:
        """Manifest paths are relative to the data root, so the data folder can move."""
        return os.path.relpath(path.resolve(), self.root.resolve())

    # step 2
    def group_near_copies(self) -> None:
        df = self.df.sort_values("path", ignore_index=True)
        df["status"] = df["status"].where(df["status"].isin(FILE_FAULTS), "ok")
        readable = df.index[df["status"] == "ok"]
        pairs = near_pairs(df.loc[readable], self.hamming)

        n = len(readable)
        graph = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n))
        df["group"] = -1
        df.loc[readable, "group"] = connected_components(graph, directed=False)[1]  # A~B and B~C: one group
        self.df, self.pairs = df, readable.to_numpy()[pairs]

    # step 3
    def drop_conflicts(self) -> None:
        df = self.df
        grouped = df[df["group"] >= 0]
        labels_in_group = grouped.groupby("group")["label"].transform("nunique")
        df.loc[labels_in_group[labels_in_group > 1].index, "status"] = "label_conflict"
        df.loc[(df["status"] == "ok") & df.duplicated("sha1"), "status"] = "duplicate_exact"

    # step 4
    def split(self) -> None:
        df = self.df
        usable = df[df["status"] == "ok"]
        groups = usable.groupby("group").agg(label=("label", "first"), key=("sha1", "min"), size=("sha1", "size"))
        groups = groups.sort_values(["label", "key"])  # the hash order is effectively random, but fixed
        placed_before = groups.groupby("label")["size"].cumsum() - groups["size"]
        share_before = placed_before / groups.groupby("label")["size"].transform("sum")
        df["split"] = ""
        df.loc[usable.index, "split"] = usable["group"].map(share_before.map(split_for))

    # step 5
    def cache(self) -> None:
        for size in (AE_SIZE, CLS_SIZE):
            load(select(self.df), self.root, size)  # only photos not cached yet are read

    # checks
    def count_leaks(self) -> int:
        """Near-copy pairs of usable images that sit in different splits. Must be 0."""
        usable = (self.df["status"] == "ok").to_numpy()
        both_usable = self.pairs[usable[self.pairs[:, 0]] & usable[self.pairs[:, 1]]]
        split = self.df["split"].to_numpy()
        return int((split[both_usable[:, 0]] != split[both_usable[:, 1]]).sum())

    def report(self) -> None:
        df = self.df
        print("\nImages by class and status:")
        print(df.groupby(["label", "status"]).size().unstack(fill_value=0))
        ok = select(df)
        table = ok.groupby(["label", "split"]).size().unstack(fill_value=0)
        table["groups"] = ok.groupby("label")["group"].nunique()
        print("\nUsable images by class and split:")
        print(table)
        print(f"\nImages dropped as label_conflict: {int((df['status'] == 'label_conflict').sum())}")
        groups = df[df["group"] >= 0].groupby("group")
        largest = pd.DataFrame({"images": groups.size(), "classes": groups["label"].unique().map(", ".join)})
        print("5 largest near-copy groups (a huge one means chains joined unrelated photos):")
        print(largest.nlargest(5, "images").to_string())
        print(f"\nLeak check: {self.count_leaks()} near-copy pairs cross splits (must be 0).")


# ---- steps 1-4: helpers ------------------------------------------------------------------------
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


def popcount(x: np.ndarray) -> np.ndarray:
    """Number of 1-bits in each uint64 (the Hamming distance, when x is an XOR of two hashes)."""
    bits_in_byte = np.array([bin(i).count("1") for i in range(256)], np.uint8)
    return bits_in_byte[x.view(np.uint8).reshape(*x.shape, 8)].sum(-1)


def near_pairs(df: pd.DataFrame, hamming: int = HAMMING, chunk: int = 512) -> np.ndarray:
    """Row-position pairs (i, j), i < j, of near copies: within `hamming` bits on both pHash and aHash."""
    ph = np.array([int(h, 16) for h in df["phash"]], np.uint64)
    ah = np.array([int(h, 16) for h in df["ahash"]], np.uint64)
    pairs = [np.empty((0, 2), np.int64)]
    for start in range(0, len(df), chunk):  # compare a chunk of images against all images
        dist = np.maximum(popcount(ph[start:start + chunk, None] ^ ph), popcount(ah[start:start + chunk, None] ^ ah))
        i, j = np.nonzero(dist <= hamming)
        i += start
        pairs.append(np.stack([i, j], axis=1)[i < j])
    return np.concatenate(pairs)


def split_for(share_before: float) -> str:
    """Split of a group, from the share of its class's images placed before it."""
    if share_before < TEST_FRAC:
        return "test"
    if share_before < TEST_FRAC + VAL_FRAC:
        return "val"
    return "train"


# ---- step 5: the image + leaf mask cache, also read by training ----------------------------------
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


def read_photo(path: str, size: int) -> tuple[np.ndarray, np.ndarray]:
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
            done = pool.map(lambda p: read_photo(p, size), paths)
            for i, (image, mask) in enumerate(tqdm(done, total=len(paths), desc=f"caching {size}px")):
                new_images[i], new_masks[i] = image, mask
        images.replace(list(missing["sha1"]), new_images)
        masks.replace(list(missing["sha1"]), new_masks)

    return images.get(rows["sha1"]), masks.get(rows["sha1"])


def load_images(rows: pd.DataFrame, root: Path, size: int) -> np.ndarray:
    return load(rows, root, size)[0]


def load_masks(rows: pd.DataFrame, root: Path, size: int) -> np.ndarray:
    return load(rows, root, size)[1]


def main():
    raw, root = command_args("maize-prepare RAW_FOLDER DATA_ROOT", required=2)[:2]
    if not Path(raw).is_dir():
        raise SystemExit(f"{raw} is not a folder. It must hold Healthy/ MLB/ MLND/ MSV/.")
    DataPipeline(Path(raw), Path(root)).run()
