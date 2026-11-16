"""The two models (proposal 3.5). Both are Keras and take RGB images, channels last, values 0-255.

Classifier (stage 1): an ImageNet backbone with a new head (global average pooling, dropout 0.3,
softmax). Phase 1 trains the head with the backbone frozen (Adam 1e-3); phase 2 fine-tunes the top
of the backbone (1e-5). Class weights balance the classes; early stopping on val loss keeps the
best epoch. Several backbones are trained on the same split and compared.

Autoencoder (stage 2): trained on healthy leaves only. The encoder has 4 stride-2 conv blocks,
32 -> 256 filters (128 px -> 8 px); the decoder mirrors it. It outputs the reconstruction in 0-1.

Both are exported to ONNX, which is what the web app runs.
"""
from __future__ import annotations

import os

os.environ.setdefault("KERAS_BACKEND", "torch")

import json
import math
from pathlib import Path

import keras
import numpy as np
import pandas as pd
from keras import layers
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from .cache import load_images
from .data import HEALTHY, select
from .preprocess import AE_SIZE, CLS_SIZE

# name -> (constructor, input preprocessing; None = the backbone does its own)
BACKBONES = {
    "mobilenet_v2": (keras.applications.MobileNetV2, keras.applications.mobilenet_v2.preprocess_input),
    "mobilenet_v3_small": (keras.applications.MobileNetV3Small, None),
    "efficientnet_b0": (keras.applications.EfficientNetB0, None),
    "resnet50": (keras.applications.ResNet50, keras.applications.resnet50.preprocess_input),
}
MIN_CONFIDENCE = 0.6  # below this the app asks for a better photo instead of guessing
UNFREEZE = 40         # backbone layers fine-tuned in phase 2
SEED = 42


class Batches(keras.utils.PyDataset):
    """Feeds uint8 images to Keras as float32 batches, so the dataset never sits in memory as floats."""

    def __init__(self, x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool):
        super().__init__()
        self.x, self.y, self.batch_size, self.shuffle = x, y, batch_size, shuffle
        self.order = np.arange(len(x))
        self.on_epoch_end()

    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)

    def __getitem__(self, i):
        idx = self.order[i * self.batch_size:(i + 1) * self.batch_size]
        return self.x[idx].astype("float32"), self.y[idx]

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.order)


# ---- classifier ---------------------------------------------------------------------------------
def build_classifier(backbone: str, n_classes: int, weights: str | None = "imagenet"):
    """Returns (model, backbone). The backbone is returned so training can freeze/unfreeze it."""
    make, preprocess = BACKBONES[backbone]
    base = make(include_top=False, weights=weights, input_shape=(CLS_SIZE, CLS_SIZE, 3))
    image = keras.Input((CLS_SIZE, CLS_SIZE, 3), name="image")
    x = layers.Lambda(preprocess)(image) if preprocess else image
    x = base(x, training=False)  # BatchNorm stays in inference mode, also while fine-tuning
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    probs = layers.Dense(n_classes, activation="softmax", name="probs")(x)
    return keras.Model(image, probs), base


def with_augmentation(model: keras.Model) -> keras.Model:
    """Random flips, rotation, zoom, contrast and darkening while training; the exported model has none."""
    return keras.Sequential([
        keras.Input((CLS_SIZE, CLS_SIZE, 3)),
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
        layers.RandomContrast(0.1),
        layers.RandomBrightness(0.2),
        model,
    ])


def class_weights(y: np.ndarray) -> dict[int, float]:
    """Rarer classes count more, so every class matters equally to the loss."""
    counts = np.bincount(y)
    return {c: len(y) / (len(counts) * n) for c, n in enumerate(counts)}


def fit_classifier(model, base, train: Batches, val: Batches, epochs: int) -> pd.DataFrame:
    trainer = with_augmentation(model)
    history = []
    for phase, lr, unfreeze in (("head", 1e-3, 0), ("finetune", 1e-5, UNFREEZE)):
        base.trainable = unfreeze > 0
        for layer in base.layers[:-unfreeze or None]:
            layer.trainable = False
        trainer.compile(optimizer=keras.optimizers.Adam(lr), loss="sparse_categorical_crossentropy",
                        metrics=["accuracy"])
        stop = keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)
        h = trainer.fit(train, validation_data=val, epochs=epochs, class_weight=class_weights(train.y),
                        callbacks=[stop], verbose=2)
        history.append(pd.DataFrame(h.history).assign(phase=phase))
    return pd.concat(history, ignore_index=True)


def expected_calibration_error(probs: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    """How far confidence is from accuracy, averaged over confidence bins (sets the retake threshold)."""
    confidence, correct = probs.max(1), probs.argmax(1) == y
    bin_of = np.minimum((confidence * bins).astype(int), bins - 1)
    gaps = [abs(correct[bin_of == b].mean() - confidence[bin_of == b].mean()) * (bin_of == b).mean()
            for b in np.unique(bin_of)]
    return float(sum(gaps))


def classifier_scores(model, data: Batches, labels: list[str]) -> dict:
    probs = model.predict(data, verbose=0)
    pred = probs.argmax(1)
    all_classes = list(range(len(labels)))
    return {
        "macro_f1": float(f1_score(data.y, pred, average="macro")),
        "ece": expected_calibration_error(probs, data.y),
        "per_class": classification_report(data.y, pred, labels=all_classes, target_names=labels,
                                           output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(data.y, pred, labels=all_classes).tolist(),
    }


def train_classifiers(df: pd.DataFrame, root: Path, out: Path, backbones: list[str], epochs: int,
                      batch_size: int = 32) -> pd.DataFrame:
    """Train each backbone on the same split. Writes out/clf_<name>/ and returns one row per backbone.
    Pick the best on val; test is scored once at the end and only reported."""
    rows = select(df)
    labels = sorted(rows["label"].unique())
    images = load_images(rows, root, CLS_SIZE)
    y = rows["label"].map(labels.index).to_numpy()
    split = rows["split"].to_numpy()
    data = {s: Batches(images[split == s], y[split == s], batch_size, shuffle=s == "train")
            for s in ("train", "val", "test")}
    del images

    table = []
    for name in backbones:
        print(f"\n--- classifier: {name} ---", flush=True)
        keras.utils.set_random_seed(SEED)
        model, base = build_classifier(name, len(labels))
        folder = out / f"clf_{name}"
        folder.mkdir(parents=True, exist_ok=True)

        history = fit_classifier(model, base, data["train"], data["val"], epochs)
        val, test = classifier_scores(model, data["val"], labels), classifier_scores(model, data["test"], labels)
        model.export(str(folder / "classifier.onnx"), format="onnx")

        info = {"backbone": name, "labels": labels, "img_size": CLS_SIZE, "min_confidence": MIN_CONFIDENCE,
                "val": val, "test": test}
        (folder / "classifier.json").write_text(json.dumps(info, indent=2))
        history.to_csv(folder / "history.csv", index=False)
        table.append({"backbone": name, "val_macro_f1": val["macro_f1"], "test_macro_f1": test["macro_f1"],
                      "test_ece": test["ece"], "size_mb": (folder / "classifier.onnx").stat().st_size / 1e6,
                      **{f"test_recall_{c}": test["per_class"][c]["recall"] for c in labels}})
        keras.backend.clear_session()
    return pd.DataFrame(table)


# ---- autoencoder --------------------------------------------------------------------------------
def build_autoencoder() -> keras.Model:
    image = keras.Input((AE_SIZE, AE_SIZE, 3), name="image")
    x = layers.Rescaling(1 / 255)(image)
    for filters in (32, 64, 128, 256):  # 128 -> 64 -> 32 -> 16 -> 8 px
        x = layers.Conv2D(filters, 3, strides=2, padding="same", activation="relu")(x)
    for filters in (128, 64, 32):       # 8 -> 16 -> 32 -> 64 px
        x = layers.Conv2DTranspose(filters, 3, strides=2, padding="same", activation="relu")(x)
    recon = layers.Conv2DTranspose(3, 3, strides=2, padding="same", activation="sigmoid", name="recon")(x)
    return keras.Model(image, recon)


def reconstruction_loss(y, recon):
    """Mean squared error per image, with the 0-255 target scaled to 0-1 like the output."""
    target = keras.ops.cast(y, "float32") / 255.0
    return keras.ops.mean(keras.ops.square(target - recon), axis=(1, 2, 3))


def reconstruction_errors(model: keras.Model, images: np.ndarray, chunk: int = 1024) -> np.ndarray:
    """Error e(x) per image, in the order given. Done in chunks to keep memory low."""
    errors = []
    for start in range(0, len(images), chunk):
        x = images[start:start + chunk].astype("float32")
        recon = model.predict(x, batch_size=64, verbose=0)
        errors.append(((x / 255.0 - recon) ** 2).mean(axis=(1, 2, 3)))
    return np.concatenate(errors)


def train_autoencoder(df: pd.DataFrame, root: Path, out: Path, epochs: int, batch_size: int = 64) -> None:
    """Healthy train leaves only; healthy val leaves for early stopping. Writes out/autoencoder.keras + model.onnx."""
    train = load_images(select(df, HEALTHY, "train"), root, AE_SIZE)
    val = load_images(select(df, HEALTHY, "val"), root, AE_SIZE)
    print(f"Autoencoder: {len(train)} healthy train, {len(val)} healthy val images", flush=True)

    keras.utils.set_random_seed(SEED)
    model = build_autoencoder()
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss=reconstruction_loss)
    stop = keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
    h = model.fit(Batches(train, train, batch_size, shuffle=True), validation_data=Batches(val, val, batch_size, False),
                  epochs=epochs, callbacks=[stop], verbose=2)

    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(h.history).to_csv(out / "history.csv", index=False)
    model.save(out / "autoencoder.keras")
    model.export(str(out / "model.onnx"), format="onnx")


def load_autoencoder(folder: Path) -> keras.Model:
    return keras.saving.load_model(folder / "autoencoder.keras", compile=False)
