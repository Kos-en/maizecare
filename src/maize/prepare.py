"""Clean new images and add them to data/manifest.csv. Safe to re-run.

    maize-prepare --raw /path/to/raw     # raw/<Class>/*.jpg, class = folder name
    maize-prepare --verify               # check the manifest for leakage between splits
    maize-prepare --resplit              # one-off: re-derive splits (e.g. to add the test split)

Splits by group: about 70% train, 15% val (early stopping, calibration), 15% test (final scores only).

Files already in the manifest are skipped, and an existing image never changes group or split.
Unusable images are kept in the manifest with a status instead of being dropped.
Near-copy rule (from the notebook): min(pHash, aHash Hamming distance) <= 6. A new image joins
an existing group if it is near ANY member (the notebook used complete linkage); this is
stricter, which is the safe direction for leakage.
"""
from __future__ import annotations

import argparse
import hashlib
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from .data import COLUMNS, data_root, load_manifest, save_manifest

EXTS = {".jpg", ".jpeg", ".png"}
MIN_SIDE = 50
HAMMING = 6
VAL_FRAC = 0.15
TEST_FRAC = 0.15
SAVE_EVERY = 1000  # save progress so an interrupted run resumes instead of starting over
_POP8 = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def popcount(x: np.ndarray) -> np.ndarray:
    """Number of set bits in each uint64 (= Hamming distance when x is an XOR of two hashes)."""
    return _POP8[x.view(np.uint8).reshape(*x.shape, 8)].sum(-1)


def hamming_to_all(arr: np.ndarray, q: np.uint64) -> np.ndarray:
    return popcount(np.bitwise_xor(arr, q))


class Index:
    """Hashes, group, split and label of every accepted image, held in memory during a run."""

    def __init__(self, ok: pd.DataFrame):
        self.ph = np.array([int(h, 16) for h in ok["phash"]], dtype=np.uint64)
        self.ah = np.array([int(h, 16) for h in ok["ahash"]], dtype=np.uint64)
        self.group = ok["group"].to_numpy(dtype=np.int64)
        self.split = ok["split"].to_numpy(dtype=object)
        self.label = ok["label"].to_numpy(dtype=object)
        self.next_group = int(self.group.max()) + 1 if len(self.group) else 0

    def add(self, ph, ah, group, split, label):
        self.ph = np.append(self.ph, np.uint64(ph))
        self.ah = np.append(self.ah, np.uint64(ah))
        self.group = np.append(self.group, group)
        self.split = np.append(self.split, split)
        self.label = np.append(self.label, label)
        self.next_group = max(self.next_group, group + 1)


def bucket_split(phash_hex: str, val_frac: float, test_frac: float = TEST_FRAC) -> str:
    """Split from the hash alone, so it does not depend on the order images arrive."""
    u = (int(phash_hex, 16) % 10_000) / 10_000
    return "val" if u < val_frac else "test" if u < val_frac + test_frac else "train"


def resplit(df: pd.DataFrame, val_frac: float = VAL_FRAC) -> pd.DataFrame:
    """Give every group the split `decide` would give it today (from its first image's hash).
    Val groups stay val; the test split is carved out of train. Only before training on the old split."""
    ok = df["status"] == "ok"
    first = df[ok].groupby("group")["phash"].first()
    df.loc[ok, "split"] = df.loc[ok, "group"].map(first.map(lambda h: bucket_split(h, val_frac)))
    return df


def decide(label: str, ph: int, ah: int, idx: Index, hamming: int, val_frac: float):
    """-> (status, group, split) for one new, valid image."""
    if len(idx.ph):
        d = np.minimum(hamming_to_all(idx.ph, np.uint64(ph)), hamming_to_all(idx.ah, np.uint64(ah)))
        near = d <= hamming
        if near.any():
            if len(set(idx.split[near])) > 1:
                return "split_conflict", -1, ""     # near copies in two splits: would leak
            if label not in set(idx.label[near]):
                return "label_conflict", -1, ""     # same picture filed under another class
            best = np.flatnonzero(near)[np.argmin(d[near])]
            return "ok", int(idx.group[best]), str(idx.split[best])
    return "ok", idx.next_group, bucket_split(f"{ph:016x}", val_frac)


def _sha1(p: Path) -> str:
    h = hashlib.sha1()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _hash_image(p: Path):
    import imagehash
    from PIL import Image
    with Image.open(p) as im:
        im.load()
        w, h = im.size
        return str(imagehash.phash(im, hash_size=8)), str(imagehash.average_hash(im, hash_size=8)), w, h


def _rel(p: Path, root: Path) -> str:
    try:
        return str(p.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(p.resolve())


def _inspect(p: Path) -> tuple[str, str, str, int, int] | None:
    """(sha1, phash, ahash, width, height), or None if the file is not a readable image."""
    try:
        return (_sha1(p), *_hash_image(p))
    except Exception:
        return None


def classify(label: str, info, known_sha: set, idx: Index, hamming: int, val_frac: float):
    """-> (status, group, split) for one new file."""
    if info is None:
        return "bad_image", -1, ""
    sha, ph, ah, w, h = info
    if min(w, h) < MIN_SIDE:
        return "too_small", -1, ""
    if sha in known_sha:
        return "duplicate_exact", -1, ""
    return decide(label, int(ph, 16), int(ah, 16), idx, hamming, val_frac)


def _append(df: pd.DataFrame, rows: list[dict]) -> pd.DataFrame:
    return pd.concat([df, pd.DataFrame(rows, columns=COLUMNS)], ignore_index=True)


def prepare(raw: Path, root: Path, val_frac: float = VAL_FRAC, hamming: int = HAMMING) -> pd.DataFrame:
    df = load_manifest(root)
    known_paths = set(df["path"])
    known_sha = set(df.loc[df["status"] == "ok", "sha1"])
    idx = Index(df[df["status"] == "ok"])
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")

    print(f"Listing images in {raw} ...", flush=True)
    files = sorted(p for p in raw.rglob("*") if p.suffix.lower() in EXTS)
    loose = {p for p in files if len(p.relative_to(raw).parts) < 2}
    for p in sorted(loose):
        print(f"[skip] {p}: put images in raw/<Class>/ (class = folder name)")
    todo = [(p, rel) for p in files if p not in loose and (rel := _rel(p, root)) not in known_paths]

    new_rows = []
    for i, (p, rel) in enumerate(tqdm(todo, desc="checking new images", unit="img"), 1):
        label = p.relative_to(raw).parts[0]
        info = _inspect(p)
        status, group, split = classify(label, info, known_sha, idx, hamming, val_frac)
        sha, ph, ah = info[:3] if info else ("", "", "")
        if status == "ok":  # later files in this same batch must see it as existing
            known_sha.add(sha)
            idx.add(int(ph, 16), int(ah, 16), group, split, label)
        new_rows.append(dict(path=rel, label=label, sha1=sha, phash=ph, ahash=ah, group=group, split=split,
                             status=status, added_at=now))
        if i % SAVE_EVERY == 0:
            save_manifest(_append(df, new_rows), root)

    if new_rows:
        df = _append(df, new_rows)
        save_manifest(df, root)
    print(f"{len(files)} files scanned, {len(new_rows)} new.")
    return df


def count_leaks(df: pd.DataFrame, hamming: int = HAMMING, chunk: int = 512) -> int:
    """Near-copy pairs of accepted images that sit in different splits (train/val/test). Must be 0."""
    ok = df[df["status"] == "ok"].reset_index(drop=True)
    if ok.empty:
        return 0
    ph = np.array([int(h, 16) for h in ok["phash"]], dtype=np.uint64)
    ah = np.array([int(h, 16) for h in ok["ahash"]], dtype=np.uint64)
    split, bad = ok["split"].to_numpy(dtype=object), 0
    for s in range(0, len(ok), chunk):
        e = min(len(ok), s + chunk)
        near = np.minimum(popcount(ph[s:e, None] ^ ph[None, :]), popcount(ah[s:e, None] ^ ah[None, :])) <= hamming
        ii, jj = np.nonzero(near)
        keep = (ii + s) < jj
        bad += int((split[ii[keep] + s] != split[jj[keep]]).sum())
    return bad


def report(df: pd.DataFrame) -> None:
    print("\nstatus by class:\n", df.groupby(["label", "status"]).size().unstack(fill_value=0))
    ok = df[df["status"] == "ok"]
    if len(ok):
        print("\naccepted images by class / split (and independent groups):")
        t = ok.groupby(["label", "split"]).size().unstack(fill_value=0)
        t["groups"] = ok.groupby("label")["group"].nunique()
        print(t)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", help="folder with <Class>/ subfolders of images")
    ap.add_argument("--data-root", default=None, help="where manifest.csv lives (default $MAIZE_DATA_ROOT or ./data)")
    ap.add_argument("--val-frac", type=float, default=VAL_FRAC)
    ap.add_argument("--hamming", type=int, default=HAMMING)
    ap.add_argument("--verify", action="store_true", help="only check the manifest for leakage between splits")
    ap.add_argument("--resplit", action="store_true", help="re-derive train/val/test for the existing manifest")
    a = ap.parse_args(argv)
    root = data_root(a.data_root)
    if a.verify:
        df = load_manifest(root)
        n = count_leaks(df, a.hamming)
        print(f"{n} near-copy pairs cross splits." + ("" if n == 0 else "  <-- LEAK"))
        raise SystemExit(1 if n else 0)
    if a.resplit:
        df = resplit(load_manifest(root), a.val_frac)
        save_manifest(df, root)
    elif a.raw and Path(a.raw).is_dir():
        df = prepare(Path(a.raw), root, a.val_frac, a.hamming)
    else:
        ap.error(f"--raw folder not found: {a.raw}" if a.raw else "--raw is required (or --verify / --resplit)")
    report(df)
    n = count_leaks(df, a.hamming)
    print(f"\nleak check: {n} near-copy pairs cross splits.")
    raise SystemExit(1 if n else 0)
