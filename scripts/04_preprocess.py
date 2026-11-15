"""Leaf-mask every image and cache 224x224 image + mask PNGs (the training pipelines read these).

  python scripts/04_preprocess.py --workers 4
"""
import argparse
from multiprocessing import Pool

import cv2
import pandas as pd
from tqdm import tqdm

from maize.config import data_root
from maize.constants import CACHE_SIZE
from maize.data.io import load_manifest, save_manifest
from maize.data.quality import read_rgb
from maize.masking import MASK_VERSION
from maize.preprocess import prepare

ap = argparse.ArgumentParser()
ap.add_argument("--workers", type=int, default=2)
ap.add_argument("--force", action="store_true")
args = ap.parse_args()

root = data_root()
out = root / "processed"
out.mkdir(exist_ok=True)


def work(item):
    id_, path = item
    ip, mp = out / f"{id_}.png", out / f"{id_}_mask.png"
    rgb = read_rgb(path)
    p = prepare(rgb, CACHE_SIZE)
    cv2.imwrite(str(ip), cv2.cvtColor(p.image, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(mp), p.mask)  # values {0,1}
    return id_, p.leaf_fraction


df = load_manifest()
todo = [(r.id, r.abs_path) for r in df.itertuples()
        if args.force or not (out / f"{r.id}_mask.png").exists() or "leaf_fraction" not in df or pd.isna(r.leaf_fraction)]
print(f"{len(todo)} images to process (mask version {MASK_VERSION})")
with Pool(args.workers) as pool:
    lf = dict(tqdm(pool.imap_unordered(work, todo, chunksize=8), total=len(todo)))

if "leaf_fraction" not in df:
    df["leaf_fraction"] = float("nan")
df["leaf_fraction"] = df["id"].map(lf).fillna(df["leaf_fraction"])
df["rel_proc"] = df["id"].map(lambda i: f"processed/{i}.png")
df["rel_mask"] = df["id"].map(lambda i: f"processed/{i}_mask.png")
df["mask_version"] = MASK_VERSION
df["mask_flag"] = (df["leaf_fraction"] < 0.10) | (df["leaf_fraction"] > 0.95)
save_manifest(df)
print(f"{int(df['mask_flag'].sum())} images flagged (leaf_fraction <0.10 or >0.95). Eyeball a few:")
print(df[df["mask_flag"]].groupby("label").size())
