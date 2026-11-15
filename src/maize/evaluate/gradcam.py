"""Real Grad-CAM (training env only). The web service uses plain CAM from TFLite; for a
GAP->Dense head the two agree up to a ReLU/scale, and `cam_vs_gradcam_corr` measures that."""
from __future__ import annotations

import numpy as np
import tensorflow as tf

from ..models.classifier import cam_model, head_weights


def gradcam(model, x: np.ndarray, class_idx: int | None = None) -> np.ndarray:
    """x: (1,H,W,3) raw 0-255. Returns (h,w) heat map in [0,1] at feature-map resolution."""
    cm = cam_model(model)
    k = int(tf.argmax(cm(x, training=False)[1][0])) if class_idx is None else class_idx
    # gradient of the class probability w.r.t. the last conv feature maps
    with tf.GradientTape() as tape:
        feats, probs = cm(x, training=False)
        tape.watch(feats)
        score = probs[0, k]
    grads = tape.gradient(score, feats)[0]
    w = tf.reduce_mean(grads, axis=(0, 1))
    cam = tf.nn.relu(tf.reduce_sum(feats[0] * w, axis=-1)).numpy()
    return cam / cam.max() if cam.max() > 0 else cam


def cam(model, x: np.ndarray, class_idx: int | None = None) -> np.ndarray:
    """What serving computes: feats @ W[:, class] (no gradients needed)."""
    feats, probs = cam_model(model)(x, training=False)
    k = int(tf.argmax(probs[0])) if class_idx is None else class_idx
    c = np.maximum(feats[0].numpy() @ head_weights(model)[:, k], 0)
    return c / c.max() if c.max() > 0 else c


def cam_vs_gradcam_corr(model, x: np.ndarray) -> float:
    a, b = cam(model, x).ravel(), gradcam(model, x).ravel()
    return float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else float("nan")
