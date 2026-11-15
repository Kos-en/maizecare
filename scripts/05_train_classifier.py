"""Train one classifier backbone on one fold (or all folds). Resumable: finished folds are skipped.

  python scripts/05_train_classifier.py --config configs/classifier/mobilenetv2.yaml --fold 0
  python scripts/05_train_classifier.py --config configs/classifier/resnet50.yaml --fold all --set masked=false
"""
import argparse

import pandas as pd

from maize.config import load_config
from maize.data.io import load_manifest
from maize.train.classifier import train_one

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True)
ap.add_argument("--fold", default="all", help="0..K-1 or 'all'")
ap.add_argument("--set", nargs="*", default=[], help="overrides, e.g. masked=false phase2.epochs=5")
ap.add_argument("--force", action="store_true")
args = ap.parse_args()

cfg = load_config(args.config, args.set)
df = load_manifest()
assert "fold" in df and (df["split"] == "pool").any(), "Run 03_split.py and 04_preprocess.py first."
folds = sorted(df.loc[df["split"] == "pool", "fold"].unique()) if args.fold == "all" else [int(args.fold)]
res = [train_one(cfg, int(f), df, args.force) for f in folds]
t = pd.DataFrame(res)
print(t[["backbone", "masked", "fold", "val_macro_f1", "params"]])
print("mean macro-F1: %.4f  sd: %.4f" % (t["val_macro_f1"].mean(), t["val_macro_f1"].std()))
