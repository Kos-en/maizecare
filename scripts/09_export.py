"""Export the chosen models to TFLite and assemble the release folder the Render image ships.

  python scripts/09_export.py --clf-run runs/clf/mobilenetv3small_masked/fold0 \
      --ae-run runs/ae/cae_ssim/fold0 --out artifacts/release-lite --mode int8

Writes classifier_cam.tflite (probabilities + conv features), autoencoder.tflite (fp16),
head_W.npy (for CAM), severity_calibration.json and pipeline_config.json, plus export_report.json
(size, speed, macro-F1 and prediction agreement before/after quantisation).
"""
import argparse
import json
import shutil
import time
from pathlib import Path

import keras
import numpy as np

import maize.models.autoencoder  # noqa: F401
import maize.models.classifier  # noqa: F401
from maize.config import write_json
from maize.constants import CLASS_TO_ID
from maize.data.io import load_clf_array, load_manifest
from maize.evaluate.classification import summarize
from maize.export import tflite as ex
from maize.masking import MASK_VERSION
from maize.models.classifier import cam_model, head_weights
from maize.serving.backends import TFLiteModel, split_cam_outputs
from maize.serving.pipeline import DEFAULT_CFG

ap = argparse.ArgumentParser()
ap.add_argument("--clf-run", required=True)
ap.add_argument("--ae-run", required=True)
ap.add_argument("--out", default="artifacts/release-lite")
ap.add_argument("--mode", choices=["fp16", "int8"], default="int8")
args = ap.parse_args()
out, clf_dir, ae_dir = Path(args.out), Path(args.clf_run), Path(args.ae_run)
out.mkdir(parents=True, exist_ok=True)

masked = json.loads((clf_dir / "config.json").read_text())["masked"]
df = load_manifest()
train = df[(df["split"] == "pool")]
test = df[df["split"] == "test"].reset_index(drop=True)

clf = keras.saving.load_model(clf_dir / "best.keras", compile=False)
rep = load_clf_array(train.sample(min(200, len(train)), random_state=0), masked)
blob = ex.convert(cam_model(clf), args.mode, ex.representative_images(rep))
size_mb = ex.save(blob, out / "classifier_cam.tflite")
np.save(out / "head_W.npy", head_weights(clf))

ae = keras.saving.load_model(ae_dir / "best.keras", compile=False)
ae_mb = ex.save(ex.convert(ae, "fp16"), out / "autoencoder.tflite")
shutil.copy(ae_dir / "severity_calibration.json", out / "severity_calibration.json")

conf_path = clf_dir / "confidence_threshold.json"
thr = json.loads(conf_path.read_text())["threshold"] if conf_path.exists() else DEFAULT_CFG["confidence_threshold"]
cfg = {"classifier_masked": masked, "confidence_threshold": thr, "mask_version": MASK_VERSION,
       "blur_min": DEFAULT_CFG["blur_min"], "brightness_min": DEFAULT_CFG["brightness_min"],
       "brightness_max": DEFAULT_CFG["brightness_max"]}
write_json(out / "pipeline_config.json", cfg)

# before/after quantisation on the TEST set
y = test["label"].map(CLASS_TO_ID).values
x = load_clf_array(test, masked)
p_keras = clf.predict(x, batch_size=32, verbose=0)
tfl = TFLiteModel(out / "classifier_cam.tflite")
t0, probs = time.perf_counter(), []
for i in range(len(x)):
    probs.append(split_cam_outputs(tfl(x[i:i + 1]))[1])
ms = (time.perf_counter() - t0) / len(x) * 1000
p_tfl = np.stack(probs)
report = {
    "mode": args.mode, "classifier_mb": size_mb, "autoencoder_mb": ae_mb, "tflite_ms_per_image_this_machine": ms,
    "keras": {k: v for k, v in summarize(y, p_keras).items() if k in ("macro_f1", "accuracy")},
    "tflite": {k: v for k, v in summarize(y, p_tfl).items() if k in ("macro_f1", "accuracy")},
    "prediction_agreement": float((p_keras.argmax(1) == p_tfl.argmax(1)).mean()),
}
write_json(out / "export_report.json", report)
print(json.dumps(report, indent=2))
print(f"\nRelease folder ready: {out}. Commit it (small files) and push to deploy on Render.")
