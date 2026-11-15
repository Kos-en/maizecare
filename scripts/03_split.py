"""Near-duplicate groups + leakage-free test/CV split (proposal 3.4).

  python scripts/03_split.py --inspect      # look at sample pairs to choose the Hamming cut-offs
  python scripts/03_split.py                # build groups, split, verify no leakage, write manifest
"""
import argparse

import cv2
import numpy as np

from maize.config import data_root, load_config
from maize.data.hashing import group_ids, hex_to_u64, near_pairs, pair_distances
from maize.data.io import load_manifest, save_manifest
from maize.data.split import find_leaks, make_splits, summarize

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="configs/data.yaml")
ap.add_argument("--inspect", action="store_true")
args = ap.parse_args()
cfg = load_config(args.config)
h, sp = cfg["hashing"], cfg["split"]
df = load_manifest()
ah, ph = hex_to_u64(df["ahash"]), hex_to_u64(df["phash"])

if args.inspect:
    # candidate pairs at a generous cut-off, shown side by side bucketed by distance
    pairs = near_pairs(ah, ph, 10, 12, "either")
    d = pair_distances(ah, ph, pairs)
    out = data_root() / "eda" / "pairs"
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    for dist in range(0, 13, 2):
        idx = np.nonzero(np.maximum(d[:, 0], d[:, 1]) // 2 * 2 == dist)[0]
        if not len(idx):
            continue
        rows = []
        for k in rng.choice(idx, min(6, len(idx)), replace=False):
            i, j = pairs[k]
            a, b = (cv2.resize(cv2.imread(df.loc[x, "abs_path"]), (200, 200)) for x in (i, j))
            rows.append(np.hstack([a, b]))
        cv2.imwrite(str(out / f"dist_{dist:02d}.jpg"), np.vstack(rows))
    print(f"{len(pairs)} candidate pairs. Montages in {out}: find the distance where pairs stop being the "
          "same photo/scene, then set hashing.ahash_max / phash_max in configs/data.yaml.")
    raise SystemExit

pairs = near_pairs(ah, ph, h["ahash_max"], h["phash_max"], h["mode"])
df["group_id"] = group_ids(len(df), pairs)
sizes = df["group_id"].value_counts()
print(f"{len(pairs)} near-copy pairs -> {df['group_id'].nunique()} groups for {len(df)} images; "
      f"largest group = {sizes.iloc[0]} images ({100 * sizes.iloc[0] / len(df):.1f}% of data)")

df = make_splits(df, sp["test_frac"], sp["n_folds"], cfg["seed"])
leaks = find_leaks(df, h["ahash_max"], h["phash_max"], h["mode"])
assert leaks.empty, f"LEAK: {len(leaks)} near-copy pairs cross sets:\n{leaks.head()}"
print("\nLeak check passed: no near-copy pair crosses test / fold boundaries.\n")
print(summarize(df))
save_manifest(df)
print("\nManifest updated with group_id, split, fold.")
