"""Keras -> TFLite (fp16 or int8) with float inputs/outputs, so serving needs no (de)quant code
for the model boundary. Goes through SavedModel (`model.export`), the path that works with Keras 3.

int8 = post-training quantisation with a representative set. Inputs/outputs stay float32
("full integer with float fallback"), weights/activations are int8 internally.
"""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import tensorflow as tf


def convert(model, mode: str, rep_data: Callable[[], Iterable[np.ndarray]] | None = None) -> bytes:
    tmp = tempfile.mkdtemp()
    try:
        model.export(tmp)
        conv = tf.lite.TFLiteConverter.from_saved_model(tmp)
        if mode == "fp16":
            conv.optimizations = [tf.lite.Optimize.DEFAULT]
            conv.target_spec.supported_types = [tf.float16]
        elif mode == "int8":
            if rep_data is None:
                raise ValueError("int8 needs a representative dataset")
            conv.optimizations = [tf.lite.Optimize.DEFAULT]
            conv.representative_dataset = rep_data
        elif mode != "fp32":
            raise ValueError(mode)
        return conv.convert()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def save(blob: bytes, path: str | Path) -> float:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(blob)
    return len(blob) / 1e6  # MB


def representative_images(images: np.ndarray, n: int = 200):
    """images: (N,H,W,C) float32 EXACTLY as the model receives them (e.g. 0-255 masked)."""
    def gen():
        for i in np.random.default_rng(0).permutation(len(images))[:n]:
            yield [images[i:i + 1].astype(np.float32)]
    return gen
