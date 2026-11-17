"""The classifier on tiny random data (no ImageNet download). Needs keras + torch, so CI skips it."""
import os

import numpy as np
import pytest

os.environ.setdefault("KERAS_BACKEND", "torch")
pytest.importorskip("torch")
pytest.importorskip("keras")
pytest.importorskip("sklearn")
ort = pytest.importorskip("onnxruntime")

from maize.classifier import Batches, Classifier, class_weights


def run_onnx(path, x):
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    return session.run(None, {session.get_inputs()[0].name: x})[0]


def test_class_weights_favour_the_rare_class():
    w = class_weights(np.array([0, 0, 0, 1]))
    assert w[1] == pytest.approx(3 * w[0])


def test_best_macro_f1_restores_the_best_epoch():
    from maize.classifier import BestMacroF1
    import keras
    model = keras.Sequential([keras.Input((2,)), keras.layers.Dense(2, activation="softmax")])
    val = Batches(np.array([[1, 0], [0, 1]], np.uint8), np.array([0, 1]), 2, shuffle=False)
    callback = BestMacroF1(val)
    callback.set_model(model)
    model.layers[0].set_weights([np.array([[5.0, -5], [-5, 5]]), np.zeros(2)])   # perfect
    callback.on_epoch_end(0, {})
    model.layers[0].set_weights([np.array([[-5.0, 5], [5, -5]]), np.zeros(2)])   # all wrong
    callback.on_epoch_end(1, {})
    callback.on_train_end()
    assert callback.best_f1 == 1.0
    assert (model.predict(val.x.astype("float32"), verbose=0).argmax(1) == [0, 1]).all()


def test_classifier_trains_scores_and_onnx_matches(tmp_path):
    x = np.random.default_rng(0).integers(0, 255, (8, 224, 224, 3), dtype=np.uint8)
    data = Batches(x, np.array([0, 1, 2, 3] * 2), 4, shuffle=False)
    classifier = Classifier("mobilenet_v3_small", ["Healthy", "MLB", "MLND", "MSV"], weights=None)
    history = classifier.fit(data, data, epochs=2)
    assert set(history["phase"]) == {"head", "finetune"} and "val_macro_f1" in history

    probs = classifier.predict(data)
    assert probs.shape == (8, 4) and np.allclose(probs.sum(1), 1, atol=1e-5)
    assert 0 <= classifier.scores(data)["macro_f1"] <= 1

    classifier.save(tmp_path, {})
    assert np.allclose(run_onnx(tmp_path / "classifier.onnx", x[:2].astype("float32")), probs[:2], atol=1e-4)
