"""The masked autoencoder on tiny random data. Needs keras + torch, so CI skips it."""
import os

import numpy as np
import pytest

os.environ.setdefault("KERAS_BACKEND", "torch")
pytest.importorskip("torch")
pytest.importorskip("keras")
ort = pytest.importorskip("onnxruntime")

from maize.autoencoder import Autoencoder, masked_loss, training_targets
from maize.preprocess import mask_background, reconstruction_error


def leaves(n=4):
    rng = np.random.default_rng(0)
    images = rng.integers(0, 255, (n, 128, 128, 3), dtype=np.uint8)
    masks = np.zeros((n, 128, 128), np.uint8)
    masks[:, 32:96, 32:96] = 1  # the leaf is the middle square
    return images, masks


def test_loss_ignores_pixels_outside_the_leaf():
    images, masks = leaves(2)
    y = training_targets(images, masks)
    assert y.shape == (2, 128, 128, 4)
    perfect = (mask_background(images, masks) / 255.0).astype("float32")
    wrong_outside = perfect.copy()
    wrong_outside[:, :10] = 1.0                       # garbage in the background only
    assert float(np.asarray(masked_loss(y, perfect)).max()) == pytest.approx(0, abs=1e-6)
    assert float(np.asarray(masked_loss(y, wrong_outside)).max()) == pytest.approx(0, abs=1e-6)


def test_error_matches_the_loss_and_counts_leaf_pixels_only():
    images, masks = leaves(2)
    recons = np.random.default_rng(1).random((2, 128, 128, 3)).astype("float32")
    from_numpy = reconstruction_error(images, recons, masks)
    from_keras = np.asarray(masked_loss(training_targets(images, masks), recons))
    assert np.allclose(from_numpy, from_keras, atol=1e-5)


def test_train_save_load_and_onnx_match(tmp_path):
    images, masks = leaves()
    autoencoder = Autoencoder.new()
    history = autoencoder.fit((images, masks), (images, masks), epochs=1)
    assert "val_loss" in history

    recons = autoencoder.reconstruct(images, masks)
    assert recons.shape == images.shape and recons.min() >= 0 and recons.max() <= 1
    errors = autoencoder.errors(images, masks)
    assert np.allclose(errors, reconstruction_error(images, recons, masks), atol=1e-5)

    autoencoder.save(tmp_path)
    assert np.allclose(Autoencoder.load(tmp_path).errors(images, masks), errors, atol=1e-5)
    session = ort.InferenceSession(str(tmp_path / "model.onnx"), providers=["CPUExecutionProvider"])
    onnx_recons = session.run(None, {session.get_inputs()[0].name: mask_background(images, masks).astype("float32")})[0]
    assert np.allclose(onnx_recons, recons, atol=1e-4)
