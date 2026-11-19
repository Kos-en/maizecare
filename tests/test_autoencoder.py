"""The masked autoencoders on tiny random data. Needs keras + torch, so CI skips it."""
import json
import os

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault("KERAS_BACKEND", "torch")
pytest.importorskip("torch")
pytest.importorskip("keras")
ort = pytest.importorskip("onnxruntime")

from maize.autoencoder import KINDS, Autoencoder, chosen, compare, masked_loss, separation, training_targets
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


@pytest.mark.parametrize("kind", KINDS)
def test_train_save_load_and_onnx_match(kind, tmp_path):
    images, masks = leaves()
    autoencoder = Autoencoder.new(kind)
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


def test_vae_draws_and_adds_its_kl_penalty_only_while_training():
    import keras
    images, masks = leaves(2)
    model = Autoencoder.new("vae").model
    x = mask_background(images, masks).astype("float32")

    def run(training):
        return keras.ops.convert_to_numpy(model(x, training=training))

    first, second = run(True), run(True)
    assert len(model.losses) == 1                    # replaced each call, not piled up
    assert not np.allclose(first, second)            # a new random draw each time

    scored = run(False)
    assert len(model.losses) == 0                    # scoring uses the mean: no penalty
    assert np.allclose(scored, run(False))


def test_separation_measures_each_disease_against_healthy():
    labels = pd.Series(["Healthy"] * 3 + ["MSV"] * 3 + ["MLND"] * 3)
    errors = pd.Series([0.1, 0.2, 0.3, 0.5, 0.6, 0.7, 0.2, 0.25, 0.3])
    result = separation(labels, errors)
    assert result["auroc_MSV"] == 1.0
    assert result["auroc_MLND"] < 1.0
    assert result["mean_auroc"] == pytest.approx((result["auroc_MSV"] + result["auroc_MLND"]) / 2)
    assert result["worst_auroc"] == result["auroc_MLND"]
    assert result["healthy_median_error"] == pytest.approx(0.2)


def test_the_pair_with_the_best_weakest_disease_is_chosen(tmp_path):
    validation = {"conv": {"plain": (0.86, 0.84, 0.87), "smoothed": (0.92, 0.90, 0.72)},
                  "vae": {"plain": (0.88, 0.86, 0.82), "smoothed": (0.88, 0.89, 0.80)}}
    for kind, by_error in validation.items():
        (tmp_path / f"ae_{kind}").mkdir()
        results = {error: {"auroc_MLB": a, "auroc_MLND": b, "auroc_MSV": c, "mean_auroc": (a + b + c) / 3}
                   for error, (a, b, c) in by_error.items()}
        (tmp_path / f"ae_{kind}" / "validation.json").write_text(json.dumps(results))
    table = compare(tmp_path)
    assert list(zip(table["kind"], table["error"]))[:2] == [("conv", "plain"), ("vae", "plain")]
    assert chosen(tmp_path) == (tmp_path / "ae_conv", "plain")


def test_no_trained_autoencoder_is_a_clear_error(tmp_path):
    with pytest.raises(SystemExit, match="maize-train-autoencoder"):
        chosen(tmp_path)
