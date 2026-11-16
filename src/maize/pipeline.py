"""Run the whole pipeline, or one step of it.

    maize all --raw RAW --data-root ROOT --out MODELS     # every step below, in order (GPU)
    maize prepare --raw RAW --data-root ROOT              # only the data step (CPU is enough)
    maize evaluate --data-root ROOT --out MODELS          # any single step

Steps and what they write:
    prepare      clean, group and split the images     ROOT/manifest.csv
    classifier   train and compare the backbones       MODELS/classifiers.csv, MODELS/clf_<backbone>/
    autoencoder  train on healthy train leaves         MODELS/ae/
    evaluate     calibrate on val, report on test      MODELS/ae/calibration.json, metrics.json, figures
    release      collect what the web app needs        MODELS/release/

Everything is read from and written to ROOT and MODELS. On Colab both are on Drive, so if the
session dies, rerun the remaining steps. Image caches (ROOT/cache_*.npz) make later runs fast.
"""
from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

import pandas as pd

from .data import load_manifest

ALL_BACKBONES = ["mobilenet_v2", "mobilenet_v3_small", "efficientnet_b0", "resnet50"]


def prepare(a):
    from .data import prepare, report
    if not (a.raw and a.raw.is_dir()):
        raise SystemExit(f"--raw must be the folder with Healthy/ MLB/ MLND/ MSV/ inside (got {a.raw}).")
    report(prepare(a.raw, a.data_root))


def classifier(a):
    from .models import train_classifiers
    table = train_classifiers(load_manifest(a.data_root), a.data_root, a.out, a.backbones, a.epochs)
    table.to_csv(a.out / "classifiers.csv", index=False)
    print("\n" + table.to_string(index=False))


def autoencoder(a):
    from .models import train_autoencoder
    train_autoencoder(load_manifest(a.data_root), a.data_root, a.out / "ae", a.ae_epochs)


def evaluate(a):
    from .evaluate import run
    run(load_manifest(a.data_root), a.data_root, a.out / "ae")


def release(a):
    """The classifier with the best val macro-F1, plus the autoencoder and its calibration."""
    table = pd.read_csv(a.out / "classifiers.csv")
    best = table.loc[table["val_macro_f1"].idxmax(), "backbone"]
    folder = a.out / "release"
    folder.mkdir(parents=True, exist_ok=True)
    for f in (a.out / f"clf_{best}" / "classifier.onnx", a.out / f"clf_{best}" / "classifier.json",
              a.out / "ae" / "model.onnx", a.out / "ae" / "calibration.json"):
        shutil.copy(f, folder)
    print(f"Release in {folder} (classifier: {best}). Copy these 4 files into the repo's release/ to deploy.")


STEPS = {"prepare": prepare, "classifier": classifier, "autoencoder": autoencoder,
         "evaluate": evaluate, "release": release}


def main(argv=None):
    ap = argparse.ArgumentParser(prog="maize", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=[*STEPS, "all"])
    ap.add_argument("--raw", type=Path, help="folder with one subfolder per class")
    ap.add_argument("--data-root", type=Path, default=Path(os.environ.get("MAIZE_DATA_ROOT", "data")),
                    help="where manifest.csv and the image caches live")
    ap.add_argument("--out", type=Path, default=Path("models"), help="where models and results go")
    ap.add_argument("--backbones", nargs="+", choices=ALL_BACKBONES, default=ALL_BACKBONES)
    ap.add_argument("--epochs", type=int, default=10, help="classifier epochs per phase")
    ap.add_argument("--ae-epochs", type=int, default=30)
    a = ap.parse_args(argv)

    steps = list(STEPS) if a.step == "all" else [a.step]
    for name in steps:
        print(f"\n=== {name} ===", flush=True)
        STEPS[name](a)
