"""Incremental data cleaning: drop new images into a raw folder, run this, done.

    raw/<Class>/*.jpg   ->   data/manifest.csv   (which images are usable, group, train/val)

Idempotent and append-only:
  * files already in the manifest are skipped, so re-running does nothing;
  * existing images NEVER change group or split when new data arrives (no leakage drift);
  * a new image that is a near-copy of an existing one inherits that image's split;
  * unusable images are recorded with a reason instead of silently dropped.

Near-copy rule = your notebook's: min(pHash, aHash Hamming distance) <= 6 bits.
(Notebook used complete-linkage clustering over all images at once; here a new image joins
an existing group if it is within the threshold of ANY member: slightly stricter grouping,
which is the safe direction for leakage.)

    python -m maize.prepare --raw /path/to/cv_datasets_cleaned      # first run: your current folder
    python -m maize.prepare --raw data/raw                          # later: new drops
    python -m maize.prepare --verify                                # leakage check on the manifest
"""
from __future__ import annotations

import argparse
import hashlib
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .data import COLUMNS, data_root, load_manifest, save_manifest

EXTS = {".jpg", ".jpeg", ".png"}
MIN_SIDE = 50
HAMMING = 6
VAL_FRAC = 0.15
_POP8 = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def hamming_to_all(arr: np.ndarray, q: np.uint64) -> np.ndarray:
    x = np.bitwise_xor(arr, q)
    return _POP8[x.view(np.uint8).reshape(-1, 8)].sum(1)


class Index:
    """Hashes of every accepted image, kept in memory while processing a batch."""

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


def bucket_split(phash_hex: str, val_frac: float) -> str:
    """Deterministic: the same group always lands in the same split, whatever order data arrives."""
    return "val" if (int(phash_hex, 16) % 10_000) / 10_000 < val_frac else "train"


def decide(label: str, ph: int, ah: int, idx: Index, hamming: int, val_frac: float):
    """-> (status, group, split) for one new, valid image."""
    if len(idx.ph):
        d = np.minimum(hamming_to_all(idx.ph, np.uint64(ph)), hamming_to_all(idx.ah, np.uint64(ah)))
        near = d <= hamming
        if near.any():
            if len(set(idx.split[near])) > 1:
                return "split_conflict", -1, ""     # bridges train and val: using it would leak
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


def prepare(raw: Path, root: Path, val_frac: float = VAL_FRAC, hamming: int = HAMMING) -> pd.DataFrame:
    df = load_manifest(root)
    known_paths = set(df["path"])
    known_sha = set(df.loc[df["status"] == "ok", "sha1"])
    idx = Index(df[df["status"] == "ok"])
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")

    files = sorted(p for p in raw.rglob("*") if p.suffix.lower() in EXTS)
    new_rows = []
    for p in files:
        rel = _rel(p, root)
        if rel in known_paths:
            continue
        parts = p.relative_to(raw).parts
        if len(parts) < 2:
            print(f"[skip] {p}: put images in raw/<Class>/ (class = folder name)")
            continue
        label = parts[0]
        row = dict(path=rel, label=label, sha1="", phash="", ahash="", group=-1, split="", status="", added_at=now)
        try:
            row["sha1"] = _sha1(p)
            row["phash"], row["ahash"], w, h = _hash_image(p)
        except Exception:
            row["status"] = "bad_image"
            new_rows.append(row)
            continue
        if min(w, h) < MIN_SIDE:
            row["status"] = "too_small"
        elif row["sha1"] in known_sha:
            row["status"] = "duplicate_exact"
        else:
            row["status"], row["group"], row["split"] = decide(
                label, int(row["phash"], 16), int(row["ahash"], 16), idx, hamming, val_frac)
            if row["status"] == "ok":
                known_sha.add(row["sha1"])
                idx.add(int(row["phash"], 16), int(row["ahash"], 16), row["group"], row["split"], label)
                idx.next_group = max(idx.next_group, row["group"] + 1)
        new_rows.append(row)

    if new_rows:
        df = pd.concat([df, pd.DataFrame(new_rows, columns=COLUMNS)], ignore_index=True)
        save_manifest(df, root)
    print(f"{len(files)} files scanned, {len(new_rows)} new.")
    return df


def count_leaks(df: pd.DataFrame, hamming: int = HAMMING, chunk: int = 512) -> int:
    """Near-copy pairs of accepted images that sit in different splits. Must be 0."""
    ok = df[df["status"] == "ok"].reset_index(drop=True)
    if ok.empty:
        return 0
    ph = np.array([int(h, 16) for h in ok["phash"]], dtype=np.uint64)
    ah = np.array([int(h, 16) for h in ok["ahash"]], dtype=np.uint64)
    split, bad = ok["split"].to_numpy(dtype=object), 0
    for s in range(0, len(ok), chunk):
        e = min(len(ok), s + chunk)
        dp = _POP8[np.bitwise_xor(ph[s:e, None], ph[None, :]).view(np.uint8).reshape(e - s, len(ok), 8)].sum(-1)
        da = _POP8[np.bitwise_xor(ah[s:e, None], ah[None, :]).view(np.uint8).reshape(e - s, len(ok), 8)].sum(-1)
        near = np.minimum(dp, da) <= hamming
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
    ap.add_argument("--verify", action="store_true", help="only check the manifest for train/val leakage")
    a = ap.parse_args(argv)
    root = data_root(a.data_root)
    if a.verify:
        df = load_manifest(root)
        n = count_leaks(df, a.hamming)
        print(f"{n} near-copy pairs cross train/val." + ("" if n == 0 else "  <-- LEAK"))
        raise SystemExit(1 if n else 0)
    if not a.raw:
        ap.error("--raw is required (or use --verify)")
    df = prepare(Path(a.raw), root, a.val_frac, a.hamming)
    report(df)
    n = count_leaks(df, a.hamming)
    print(f"\nleak check: {n} near-copy pairs cross train/val.")
    raise SystemExit(1 if n else 0)


if __name__ == "__main__":
    main()
