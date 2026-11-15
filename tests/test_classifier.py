"""Classifier build/train/export on tiny random data (no ImageNet download). Needs keras + torch."""
import os

import numpy as np
import pytest

os.environ.setdefault("KERAS_BACKEND", "torch")
pytest.importorskip("torch")
pytest.importorskip("keras")
pytest.importorskip("sklearn")
ort = pytest.importorskip("onnxruntime")

from maize.classifier import Batches, build, class_weights


def test_balanced_class_weights():
    w = class_weights(np.array([0, 0, 0, 1]), 2)
    assert w[1] == pytest.approx(3 * w[0])


def test_train_step_and_onnx_export_match(tmp_path):
    model, _ = build("mobilenet_v3_small", 4, weights=None)
    x = np.random.default_rng(0).integers(0, 255, (8, 224, 224, 3), dtype=np.uint8)
    y = np.array([0, 1, 2, 3] * 2)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
    model.fit(Batches(x, y, 4, shuffle=True), epochs=1, verbose=0)

    probs = model.predict(x[:2].astype("float32"), verbose=0)
    assert probs.shape == (2, 4) and np.allclose(probs.sum(1), 1, atol=1e-5)

    model.export(str(tmp_path / "classifier.onnx"), format="onnx")
    sess = ort.InferenceSession(str(tmp_path / "classifier.onnx"), providers=["CPUExecutionProvider"])
    onnx_probs = sess.run(None, {sess.get_inputs()[0].name: x[:2].astype("float32")})[0]
    assert np.allclose(onnx_probs, probs, atol=1e-4)
