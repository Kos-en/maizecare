"""TensorFlow smoke tests: skipped automatically where TF is not installed."""
import numpy as np
import pytest

pytest.importorskip("tensorflow")
import keras

from maize.models.autoencoder import MaskedReconLoss, build_cae, build_vae
from maize.models.classifier import BACKBONES, build_classifier, cam_model, head_weights, set_phase


@pytest.mark.parametrize("backbone", list(BACKBONES))
def test_classifier_builds_and_cam_matches_head(backbone):
    m = build_classifier(backbone, weights=None, img_size=96)
    x = np.random.default_rng(0).uniform(0, 255, (2, 96, 96, 3)).astype("float32")
    probs = m.predict(x, verbose=0)
    assert probs.shape == (2, 4) and np.allclose(probs.sum(1), 1, atol=1e-4)
    feats, p2 = cam_model(m).predict(x, verbose=0)
    assert np.allclose(probs, p2, atol=1e-4)
    # logits from CAM path: GAP(feats) @ W + b must reproduce the softmax (dropout is off at inference)
    w, b = m.get_layer("head").get_weights()
    z = feats.mean((1, 2)) @ w + b
    soft = np.exp(z - z.max(1, keepdims=True))
    assert np.allclose(soft / soft.sum(1, keepdims=True), probs, atol=1e-3)
    set_phase(m, 1)
    assert not m.get_layer("backbone").trainable
    set_phase(m, 2, 10)
    assert m.get_layer("backbone").trainable


@pytest.mark.parametrize("builder", [build_cae, build_vae])
def test_autoencoder_trains_one_step_with_masked_loss(builder):
    ae = builder()
    x = np.random.default_rng(0).random((4, 128, 128, 3)).astype("float32")
    mask = np.ones((4, 128, 128, 1), "float32")
    mask[:, :, 64:] = 0                                   # right half is background
    y = np.concatenate([x, mask], -1)
    ae.compile("adam", MaskedReconLoss(0.5))
    h = ae.fit(x, y, epochs=1, batch_size=2, verbose=0)
    assert np.isfinite(h.history["loss"][0])
    assert ae.predict(x, verbose=0).shape == x.shape


def test_masked_loss_ignores_background():
    t = np.zeros((1, 16, 16, 4), "float32")
    t[..., 3] = 0.0
    t[:, :, :8, 3] = 1.0
    pred = np.zeros((1, 16, 16, 3), "float32")
    pred[:, :, 8:] = 1.0                                  # wrong only in background
    assert float(MaskedReconLoss(0.0)(t, pred)) == pytest.approx(0.0, abs=1e-6)
