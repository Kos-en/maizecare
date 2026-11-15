"""Scan the raw image folders, drop bad files, record stats + perceptual hashes -> manifest.csv.

  python scripts/01_build_manifest.py --config configs/data.yaml
"""
import argparse
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from maize.config import data_root, load_config
from maize.data.hashing import compute_hashes
from maize.data.io import manifest_path
from maize.data.quality import VALID_EXT, check_file, image_stats

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="configs/data.yaml")
args = ap.parse_args()
cfg = load_config(args.config)
root = data_root()

rows, dropped = [], 0
for src in cfg["sources"]:
    for folder, label in src["classes"].items():
        d = root / src["path"] / folder
        files = sorted(p for p in d.rglob("*") if p.suffix.lower() in VALID_EXT) if d.exists() else []
        if not files:
            print(f"[warn] nothing found in {d}")
        for p in tqdm(files, desc=f"{src['name']}/{folder}"):
            rgb = check_file(p, cfg["quality"]["min_side"])
            if rgb is None:
                dropped += 1
                continue
            ah, ph = compute_hashes(rgb)
            rows.append({"path": str(p.relative_to(root)), "label": label, "source": src["name"],
                         **image_stats(rgb), "ahash": ah, "phash": ph})

df = pd.DataFrame(rows)
df.insert(0, "id", [f"img{i:06d}" for i in range(len(df))])
df.to_csv(manifest_path(), index=False)
print(f"\nWrote {manifest_path()}  ({len(df)} images, {dropped} dropped)")
print(df.groupby(["source", "label"]).size().unstack(fill_value=0))
print("\nREMINDER: sample ~20 images per class and check labels by eye (proposal 3.4).")
