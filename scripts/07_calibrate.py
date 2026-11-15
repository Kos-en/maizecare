"""Fit the severity scaling (Eq. 3.2) and the classifier's retake threshold on the VALIDATION fold.

  python scripts/07_calibrate.py --ae-run runs/ae/cae_ssim/fold0 --clf-run runs/clf/mobilenetv2_masked/fold0
Writes severity_calibration.json into the AE run and confidence_threshold.json into the classifier run.
"""
import argparse
import json
from pathlib import Path

import keras
import numpy as np

import maize.models.autoencoder  # noqa: F401  (registers custom layers/loss for loading)
from maize import severity as sev
from maize.config import write_json
from maize.constants import CLASS_TO_ID, DISEASES, HEALTHY
from maize.data.io import load_ae_arrays, load_manifest
from maize.evaluate.classification import pick_confidence_threshold

ap = argparse.ArgumentParser()
ap.add_argument("--ae-run", required=True)
ap.add_argument("--clf-run")
ap.add_argument("--target-acc", type=float, default=0.95, help="accuracy wanted on accepted predictions")
args = ap.parse_args()

df = load_manifest()
pool = df[df["split"] == "pool"]

ae_dir = Path(args.ae_run)
fold = json.loads((ae_dir / "metrics.json").read_text())["fold"]
ae = keras.saving.load_model(ae_dir / "best.keras", compile=False)


def errors(sub):
    x, m = load_ae_arrays(sub)
    xh = ae.predict(x, batch_size=64, verbose=0)
    return np.array([sev.masked_error(a, b, c) for a, b, c in zip(x, xh, m)])


val, train = pool[pool["fold"] == fold], pool[pool["fold"] != fold]
healthy_val = errors(val[val["label"] == HEALTHY])
diseased_val = errors(val[val["label"] != HEALTHY])
diseased_train = {d: errors(train[train["label"] == d]) for d in DISEASES}
cal = sev.fit_calibration(healthy_val, diseased_val, diseased_train)
sev.save_calibration(cal, ae_dir / "severity_calibration.json")
print(json.dumps(cal, indent=2))

if args.clf_run:
    clf_dir = Path(args.clf_run)
    probs = np.load(clf_dir / "val_probs.npy")
    y = val["label"].map(CLASS_TO_ID).values  # same order as training's val set
    conf = pick_confidence_threshold(y, probs, args.target_acc)
    write_json(clf_dir / "confidence_threshold.json", conf)
    print("retake threshold:", conf)
