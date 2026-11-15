"""Train an autoencoder on healthy leaves (training folds only).

  python scripts/06_train_ae.py --config configs/autoencoder/cae_ssim.yaml --fold 0
"""
import argparse

from maize.config import load_config
from maize.data.io import load_manifest
from maize.train.autoencoder import train_one

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True)
ap.add_argument("--fold", default="0", help="0..K-1 or 'all'")
ap.add_argument("--set", nargs="*", default=[])
ap.add_argument("--force", action="store_true")
args = ap.parse_args()

cfg = load_config(args.config, args.set)
df = load_manifest()
folds = sorted(df.loc[df["split"] == "pool", "fold"].unique()) if args.fold == "all" else [int(args.fold)]
for f in folds:
    print(train_one(cfg, int(f), df, args.force))
