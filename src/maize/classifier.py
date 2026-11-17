"""Stage 1: which disease is it? (proposal 3.5.1)

An ImageNet backbone with a new head: global average pooling, dropout 0.3, softmax. Phase 1 trains the
head with the backbone frozen (Adam 1e-3); phase 2 fine-tunes the top of the backbone (1e-5). Class
weights balance the classes. Early stopping on val loss decides when to stop; the weights kept are
those of the epoch with the best val macro-F1. Several backbones are trained on the same split and
compared: pick on val, report test.

Input: RGB 224 x 224, values 0-255, the whole photo (not masked). Exported to ONNX for the web app.
"""
from __future__ import annotations

import os

os.environ.setdefault("KERAS_BACKEND", "torch")

import json
from pathlib import Path

import keras
import numpy as np
import pandas as pd
from keras import layers
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from .batches import SEED, Batches
from .cache import load_images
from .data import select
from .preprocess import CLS_SIZE

# name -> (constructor, input preprocessing; None = the backbone does its own)
BACKBONES = {
    "mobilenet_v2": (keras.applications.MobileNetV2, keras.applications.mobilenet_v2.preprocess_input),
    "mobilenet_v3_small": (keras.applications.MobileNetV3Small, None),
    "efficientnet_b0": (keras.applications.EfficientNetB0, None),
    "resnet50": (keras.applications.ResNet50, keras.applications.resnet50.preprocess_input),
}
MIN_CONFIDENCE = 0.6  # below this the app asks for a better photo instead of guessing
UNFREEZE = 40         # backbone layers fine-tuned in phase 2
PHASES = (("head", 1e-3, 0), ("finetune", 1e-5, UNFREEZE))  # (name, learning rate, layers unfrozen)


def build(backbone: str, n_classes: int, weights: str | None = "imagenet") -> tuple[keras.Model, keras.Model]:
    """Returns (model, backbone). The backbone is returned so training can freeze and unfreeze it."""
    make, preprocess = BACKBONES[backbone]
    base = make(include_top=False, weights=weights, input_shape=(CLS_SIZE, CLS_SIZE, 3))
    image = keras.Input((CLS_SIZE, CLS_SIZE, 3), name="image")
    x = layers.Lambda(preprocess)(image) if preprocess else image
    x = base(x, training=False)  # BatchNorm stays in inference mode, also while fine-tuning
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    probs = layers.Dense(n_classes, activation="softmax", name="probs")(x)
    return keras.Model(image, probs), base


def augmentation() -> list[layers.Layer]:
    """Random changes applied while training only; the exported model has none. No hue changes:
    yellowing is the disease signal."""
    return [
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
        layers.RandomContrast(0.1),
        layers.RandomBrightness(0.2),                                                  # darker / lighter photos
        layers.RandomGaussianBlur(factor=0.3, kernel_size=5, value_range=(0, 255)),    # out-of-focus photos
        # mild noise (about 4 of 255) stands in for JPEG compression, which Keras has no layer for;
        # GaussianNoise only accepts 0-1 values, hence the rescaling around it
        layers.Rescaling(1 / 255), layers.GaussianNoise(0.015), layers.Rescaling(255),
    ]


def class_weights(y: np.ndarray) -> dict[int, float]:
    """Rarer classes count more, so every class matters equally to the loss."""
    counts = np.bincount(y)
    return {c: len(y) / (len(counts) * n) for c, n in enumerate(counts)}


def expected_calibration_error(probs: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    """How far confidence is from accuracy, averaged over confidence bins."""
    confidence, correct = probs.max(1), probs.argmax(1) == y
    bin_of = np.minimum((confidence * bins).astype(int), bins - 1)
    gaps = [abs(correct[bin_of == b].mean() - confidence[bin_of == b].mean()) * (bin_of == b).mean()
            for b in np.unique(bin_of)]
    return float(sum(gaps))


class BestMacroF1(keras.callbacks.Callback):
    """Keeps the weights of the epoch with the best val macro-F1 (proposal 3.5.1), across both phases."""

    def __init__(self, val: Batches):
        super().__init__()
        self.val, self.best_f1, self.best_weights = val, -1.0, []

    def on_epoch_end(self, epoch, logs=None):
        f1 = f1_score(self.val.y, self.model.predict(self.val, verbose=0).argmax(1), average="macro")
        logs["val_macro_f1"] = f1
        print(f"  val macro-F1 {f1:.4f}", flush=True)
        if f1 > self.best_f1:
            self.best_f1 = f1
            self.best_weights = [(v, keras.ops.convert_to_numpy(v)) for v in self.model.weights]

    def on_train_end(self, logs=None):
        for variable, value in self.best_weights:
            variable.assign(value)


class Classifier:
    """One backbone with its head: train it in two phases, score it, save it."""

    def __init__(self, backbone: str, labels: list[str], weights: str | None = "imagenet"):
        keras.utils.set_random_seed(SEED)
        self.backbone, self.labels = backbone, labels
        self.model, self.base = build(backbone, len(labels), weights)

    def fit(self, train: Batches, val: Batches, epochs: int) -> pd.DataFrame:
        trainer = keras.Sequential([keras.Input((CLS_SIZE, CLS_SIZE, 3)), *augmentation(), self.model])
        best = BestMacroF1(val)
        history = []
        for phase, lr, unfreeze in PHASES:
            self.base.trainable = unfreeze > 0
            for layer in self.base.layers[:-unfreeze or None]:
                layer.trainable = False
            trainer.compile(optimizer=keras.optimizers.Adam(lr), loss="sparse_categorical_crossentropy",
                            metrics=["accuracy"])
            stop = keras.callbacks.EarlyStopping(patience=3)
            h = trainer.fit(train, validation_data=val, epochs=epochs, class_weight=class_weights(train.y),
                            callbacks=[best, stop], verbose=2)
            history.append(pd.DataFrame(h.history).assign(phase=phase))
        return pd.concat(history, ignore_index=True)

    def predict(self, data: Batches) -> np.ndarray:
        return self.model.predict(data, verbose=0)

    def scores(self, data: Batches) -> dict:
        probs = self.predict(data)
        pred = probs.argmax(1)
        all_classes = list(range(len(self.labels)))
        return {
            "macro_f1": float(f1_score(data.y, pred, average="macro")),
            "ece": expected_calibration_error(probs, data.y),
            "per_class": classification_report(data.y, pred, labels=all_classes, target_names=self.labels,
                                               output_dict=True, zero_division=0),
            "confusion_matrix": confusion_matrix(data.y, pred, labels=all_classes).tolist(),
        }

    def save(self, folder: Path, info: dict) -> None:
        """classifier.onnx for the web app, classifier.json with labels, settings and scores."""
        folder.mkdir(parents=True, exist_ok=True)
        self.model.export(str(folder / "classifier.onnx"), format="onnx")
        info = {"backbone": self.backbone, "labels": self.labels, "img_size": CLS_SIZE,
                "min_confidence": MIN_CONFIDENCE, **info}
        (folder / "classifier.json").write_text(json.dumps(info, indent=2))


def split_batches(df: pd.DataFrame, root: Path, labels: list[str], batch_size: int) -> dict[str, Batches]:
    """train / val / test batches of every usable image (224 px, from the cache)."""
    rows = select(df)
    images = load_images(rows, root, CLS_SIZE)
    y = rows["label"].map(labels.index).to_numpy()
    split = rows["split"].to_numpy()
    return {s: Batches(images[split == s], y[split == s], batch_size, shuffle=s == "train")
            for s in ("train", "val", "test")}


def train_all(df: pd.DataFrame, root: Path, out: Path, backbones: list[str], epochs: int,
              batch_size: int = 32) -> pd.DataFrame:
    """Train each backbone on the same split; writes out/clf_<name>/. Returns one comparison row per backbone."""
    labels = sorted(select(df)["label"].unique())
    data = split_batches(df, root, labels, batch_size)
    table = []
    for name in backbones:
        print(f"\n--- classifier: {name} ---", flush=True)
        folder = out / f"clf_{name}"
        classifier = Classifier(name, labels)
        history = classifier.fit(data["train"], data["val"], epochs)
        val, test = classifier.scores(data["val"]), classifier.scores(data["test"])
        classifier.save(folder, {"val": val, "test": test})
        history.to_csv(folder / "history.csv", index=False)
        table.append({"backbone": name, "val_macro_f1": val["macro_f1"], "test_macro_f1": test["macro_f1"],
                      "test_ece": test["ece"], "size_mb": (folder / "classifier.onnx").stat().st_size / 1e6,
                      **{f"test_recall_{c}": test["per_class"][c]["recall"] for c in labels}})
        keras.backend.clear_session()
    return pd.DataFrame(table)
