"""Mask quality vs hand-drawn masks (proposal 3.4 step 5): IoU overall, by class, and on
blurred / darkened copies.

  python scripts/04b_mask_check.py --sample 100        # writes the list of images to hand-mask
  # draw masks (white = leaf incl. lesions) -> $MAIZE_DATA_ROOT/handmasks/<id>.png, then:
  python scripts/04b_mask_check.py
"""
import argparse

import cv2
import numpy as np
import pandas as pd

from maize.config import data_root
from maize.data.degrade import DEGRADATIONS
from maize.data.io import load_manifest
from maize.data.quality import read_rgb
from maize.masking import mask_iou
from maize.preprocess import pad_square_resize, prepare

ap = argparse.ArgumentParser()
ap.add_argument("--sample", type=int, default=0)
args = ap.parse_args()
root = data_root()
df = load_manifest()
hm_dir = root / "handmasks"

if args.sample:
    per = max(1, args.sample // df["label"].nunique())
    s = df.groupby("label", group_keys=False).apply(lambda g: g.sample(min(len(g), per), random_state=0))
    hm_dir.mkdir(exist_ok=True)
    s[["id", "path", "label"]].to_csv(root / "handmask_sample.csv", index=False)
    print(f"Wrote {root / 'handmask_sample.csv'} ({len(s)} images). Draw masks into {hm_dir}/<id>.png")
    raise SystemExit

rows = []
for r in df.itertuples():
    hp = hm_dir / f"{r.id}.png"
    if not hp.exists():
        continue
    hand = (cv2.imread(str(hp), cv2.IMREAD_GRAYSCALE) > 127).astype(np.uint8)
    hand = pad_square_resize(hand, 224, cv2.INTER_NEAREST)
    rgb = read_rgb(r.abs_path)
    for name, deg in DEGRADATIONS.items():
        rows.append({"id": r.id, "label": r.label, "condition": name,
                     "iou": mask_iou(prepare(deg(rgb)).mask, hand)})
res = pd.DataFrame(rows)
if res.empty:
    raise SystemExit(f"No hand masks found in {hm_dir}")
print("IoU overall by condition:\n", res.groupby("condition")["iou"].agg(["mean", "std", "count"]).round(3))
print("\nIoU by class x condition:\n", res.pivot_table(index="label", columns="condition", values="iou").round(3))
(root / "eda").mkdir(exist_ok=True)
res.to_csv(root / "eda" / "mask_iou.csv", index=False)
print("\nIf a class is poor: tune MaskParams (hue range, touch_frac), bump MASK_VERSION, rerun 04_preprocess.py --force.")
