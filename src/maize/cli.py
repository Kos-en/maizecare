"""The `maize` command.

The data pipeline. Every image goes through it before any training, and new photos get the same treatment:
    maize prepare --raw RAW --data-root ROOT
        1. check, group and split the images            -> ROOT/manifest.csv
        2. resize each usable image and find its leaf   -> ROOT/cache_<size>.npz, ROOT/mask_<size>.npz
        Re-run after adding photos to RAW/<Class>/: only the new files are read.

Training and evaluation, on what prepare produced:
    maize classifier  --data-root ROOT --out MODELS     -> MODELS/classifiers.csv, MODELS/clf_<backbone>/
    maize autoencoder --data-root ROOT --out MODELS     -> MODELS/ae/
    maize evaluate    --data-root ROOT --out MODELS [--classifier MODELS/clf_<backbone>]
                                                        -> MODELS/ae/calibration.json, metrics.json, figures
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

ALL_BACKBONES = ["mobilenet_v2", "mobilenet_v3_small", "efficientnet_b0", "resnet50"]


def prepare(a):
    from .cache import load
    from .data import prepare, report, select
    from .preprocess import AE_SIZE, CLS_SIZE

    if not (a.raw and a.raw.is_dir()):
        raise SystemExit(f"--raw must be the folder with Healthy/ MLB/ MLND/ MSV/ inside (got {a.raw}).")
    df, pairs = prepare(a.raw, a.data_root)
    report(df, pairs)
    for size in (AE_SIZE, CLS_SIZE):
        load(select(df), a.data_root, size)  # caches image + leaf mask; only photos not cached yet are read


def usable_manifest(a):
    """The manifest, or a clear message if prepare has not produced any usable images yet."""
    from .data import load_manifest, select

    df = load_manifest(a.data_root)
    if select(df).empty:
        raise SystemExit(f"No usable images in {a.data_root}/manifest.csv: run `maize prepare` first.")
    return df


def classifier(a):
    from .classifier import train_all

    table = train_all(usable_manifest(a), a.data_root, a.out, a.backbones, a.epochs)
    table.to_csv(a.out / "classifiers.csv", index=False)
    print("\n" + table.to_string(index=False))


def autoencoder(a):
    from .autoencoder import train

    train(usable_manifest(a), a.data_root, a.out / "ae", a.ae_epochs)


def evaluate(a):
    from .evaluate import run

    run(usable_manifest(a), a.data_root, a.out / "ae", a.classifier)


COMMANDS = {"prepare": prepare, "classifier": classifier, "autoencoder": autoencoder, "evaluate": evaluate}


def main(argv=None):
    ap = argparse.ArgumentParser(prog="maize", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=COMMANDS)
    ap.add_argument("--raw", type=Path, help="prepare: folder with one subfolder of images per class")
    ap.add_argument("--data-root", type=Path, default=Path(os.environ.get("MAIZE_DATA_ROOT", "data")),
                    help="where manifest.csv and the caches live")
    ap.add_argument("--out", type=Path, default=Path("models"), help="where models and results go")
    ap.add_argument("--backbones", nargs="+", choices=ALL_BACKBONES, default=ALL_BACKBONES)
    ap.add_argument("--epochs", type=int, default=10, help="classifier epochs per phase")
    ap.add_argument("--ae-epochs", type=int, default=30)
    ap.add_argument("--classifier", type=Path, help="evaluate: a trained classifier folder, for the baseline")
    a = ap.parse_args(argv)
    COMMANDS[a.command](a)
