"""classify -> (if diseased) grade pipeline served by the web app (proposal 3.7).

Steps: quality gate -> mask+resize (same module as training) -> classify (retake if unsure)
-> autoencoder severity -> CAM + error heat maps -> advice. TFLite only; no TensorFlow import.
"""
from __future__ import annotations

import base64
import json
from pathlib import Path

import cv2
import numpy as np

from .. import severity as sev
from ..constants import AE_SIZE, CLASSES, HEALTHY
from ..data.quality import image_stats
from ..masking import MASK_VERSION, apply_mask
from ..preprocess import downscale, prepare
from .advice import advice_for
from .backends import TFLiteModel, split_cam_outputs

DEFAULT_CFG = {
    "classifier_masked": True,
    "confidence_threshold": 0.7,
    "blur_min": 20.0,           # set from EDA percentiles of sharp training photos
    "brightness_min": 40.0,
    "brightness_max": 240.0,
    "leaf_fraction_min": 0.10,
    "leaf_fraction_max": 0.95,
    "mask_version": MASK_VERSION,
}


def _png_b64(rgb: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    return base64.b64encode(buf.tobytes()).decode()


def _overlay(rgb: np.ndarray, heat01: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    h = cv2.resize(np.clip(heat01, 0, 1).astype(np.float32), (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
    color = cv2.cvtColor(cv2.applyColorMap((h * 255).astype(np.uint8), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
    return (rgb * (1 - alpha) + color * alpha).astype(np.uint8)


class MaizePipeline:
    def __init__(self, clf, ae, head_w, calibration, cfg):
        self.clf, self.ae, self.head_w, self.cal = clf, ae, head_w, calibration
        self.cfg = {**DEFAULT_CFG, **cfg}
        if self.cfg["mask_version"] != MASK_VERSION:
            raise RuntimeError(f"Models trained with mask {self.cfg['mask_version']} but code has {MASK_VERSION}")

    @classmethod
    def from_dir(cls, d: str | Path) -> "MaizePipeline":
        d = Path(d)
        cfg = json.loads((d / "pipeline_config.json").read_text())
        return cls(TFLiteModel(d / "classifier_cam.tflite"), TFLiteModel(d / "autoencoder.tflite"),
                   np.load(d / "head_W.npy"), sev.load_calibration(d / "severity_calibration.json"), cfg)

    # -- helpers -------------------------------------------------------------------------
    def _retake(self, reason: str, **extra) -> dict:
        return {"status": "retake", "reason": reason, **extra}

    def predict(self, data: bytes) -> dict:
        arr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if arr is None:
            return self._retake("We could not read that image. Please upload a JPEG or PNG photo.")
        rgb = downscale(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))
        c = self.cfg

        # 1. quality gate
        st = image_stats(rgb)
        if st["blur"] < c["blur_min"]:
            return self._retake("The photo is blurry. Hold the phone steady and retake it.", quality=st)
        if st["brightness"] < c["brightness_min"]:
            return self._retake("The photo is too dark. Move to better light and retake it.", quality=st)
        if st["brightness"] > c["brightness_max"]:
            return self._retake("The photo is too bright. Avoid direct glare and retake it.", quality=st)

        # 2. mask + resize: same code as the training cache
        p = prepare(rgb)
        if not c["leaf_fraction_min"] <= p.leaf_fraction <= c["leaf_fraction_max"]:
            return self._retake("We could not find a clear leaf. Fill the frame with one leaf and retake it.",
                                quality={**st, "leaf_fraction": p.leaf_fraction})

        # 3. classify (raw 0-255 in; backbone preprocessing is inside the model)
        clf_in = apply_mask(p.image, p.mask) if c["classifier_masked"] else p.image
        feats, probs = split_cam_outputs(self.clf(clf_in[None].astype(np.float32)))
        k = int(probs.argmax())
        label, conf = CLASSES[k], float(probs[k])
        if conf < c["confidence_threshold"]:
            return self._retake("We are not confident about this leaf. Retake the photo closer, in good light.",
                                confidence=conf)

        cam = np.maximum(feats @ self.head_w[:, k], 0)
        cam = cam / cam.max() if cam.max() > 0 else cam
        result = {
            "status": "ok", "disease": label, "confidence": conf,
            "probabilities": dict(zip(CLASSES, map(float, probs))),
            "cam_png": _png_b64(_overlay(p.image, cam)),
            "severity": None,
        }

        # 4. severity for diseased leaves
        grade = None
        if label != HEALTHY:
            x = cv2.resize(apply_mask(p.image, p.mask), (AE_SIZE, AE_SIZE), interpolation=cv2.INTER_AREA) / 255.0
            m = cv2.resize(p.mask, (AE_SIZE, AE_SIZE), interpolation=cv2.INTER_NEAREST)
            x = (x * m[..., None]).astype(np.float32)
            recon = self.ae(x[None])[0][0]
            e = sev.masked_error(x, recon, m)
            grade = sev.grade(e, label, self.cal)
            emap = sev.error_map(x, recon, m)
            emap = emap / emap.max() if emap.max() > 0 else emap
            base = cv2.resize(p.image, (AE_SIZE, AE_SIZE), interpolation=cv2.INTER_AREA)
            result["severity"] = {"score": sev.score(e, self.cal), "grade": grade, "error": e,
                                  "heatmap_png": _png_b64(_overlay(base, emap))}

        result["advice"] = advice_for(label, grade)
        return result
