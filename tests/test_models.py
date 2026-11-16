"""Both Keras models on tiny random data (no ImageNet download). Needs keras + torch, so CI skips it."""
import os

import numpy as np
import pytest

os.environ.setdefault("KERAS_BACKEND", "torch")
pytest.importorskip("torch")
pytest.importorskip("keras")
pytest.importorskip("sklearn")
ort = pytest.importorskip("onnxruntime")

from maize.models import Batches, build_autoencoder, build_classifier, class_weights, reconstruction_errors


def run_onnx(path, x):
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    return session.run(None, {session.get_inputs()[0].name: x})[0]


def random_images(n, size):
    return np.random.default_rng(0).integers(0, 255, (n, size, size, 3), dtype=np.uint8)


def test_class_weights_favour_the_rare_class():
    w = class_weights(np.array([0, 0, 0, 1]))
    assert w[1] == pytest.approx(3 * w[0])


def test_classifier_trains_and_onnx_matches(tmp_path):
    model, _ = build_classifier("mobilenet_v3_small", 4, weights=None)
    x, y = random_images(8, 224), np.array([0, 1, 2, 3] * 2)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
    model.fit(Batches(x, y, 4, shuffle=True), epochs=1, verbose=0)

    probs = model.predict(x[:2].astype("float32"), verbose=0)
    assert probs.shape == (2, 4) and np.allclose(probs.sum(1), 1, atol=1e-5)
    model.export(str(tmp_path / "classifier.onnx"), format="onnx")
    assert np.allclose(run_onnx(tmp_path / "classifier.onnx", x[:2].astype("float32")), probs, atol=1e-4)


def test_autoencoder_errors_and_onnx_match(tmp_path):
    model = build_autoencoder()
    x = random_images(3, 128)
    recon = model.predict(x.astype("float32"), verbose=0)
    assert recon.shape == x.shape and recon.min() >= 0 and recon.max() <= 1

    errors = reconstruction_errors(model, x)
    assert np.allclose(errors, ((x / 255.0 - recon) ** 2).mean(axis=(1, 2, 3)), atol=1e-5)
    model.export(str(tmp_path / "model.onnx"), format="onnx")
    assert np.allclose(run_onnx(tmp_path / "model.onnx", x.astype("float32")), recon, atol=1e-4)
