"""Stage 1: which disease is it? Fine-tunes a Keras ImageNet CNN on every accepted image in the
manifest (same leak-free train/val split as the autoencoder).

    maize-train-classifier --backbone mobilenet_v3_small --out models/clf
    maize-train-classifier --backbone resnet50 --out models/clf_resnet50     # proposal baseline

Backbones: mobilenet_v2, mobilenet_v3_small, efficientnet_b0, resnet50. Keras runs on the PyTorch
backend, so Colab needs one framework and the model exports to ONNX for the web app.
Writes classifier.onnx, classifier.json (labels + validation metrics), history.csv, confusion.png.
"""
from __future__ import annotations

import os

os.environ.setdefault("KERAS_BACKEND", "torch")

import argparse
import json
import math
from pathlib import Path

import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras import layers
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from .cache import load_images
from .data import data_root, load_manifest, select
from .preprocess import CLS_SIZE

plt.switch_backend("Agg")

# name -> (constructor, preprocessing inside the model; None = the backbone already includes it)
BACKBONES = {
    "mobilenet_v2": (keras.applications.MobileNetV2, keras.applications.mobilenet_v2.preprocess_input),
    "mobilenet_v3_small": (keras.applications.MobileNetV3Small, None),
    "efficientnet_b0": (keras.applications.EfficientNetB0, None),
    "resnet50": (keras.applications.ResNet50, keras.applications.resnet50.preprocess_input),
}
MIN_CONFIDENCE = 0.6  # below this the app asks for a better photo instead of guessing


class Batches(keras.utils.PyDataset):
    """uint8 images -> float32 batches, so the whole dataset never sits in memory as float."""

    def __init__(self, x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool):
        super().__init__()
        self.x, self.y, self.batch_size, self.shuffle = x, y, batch_size, shuffle
        self.order = np.random.permutation(len(x)) if shuffle else np.arange(len(x))

    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)

    def __getitem__(self, i):
        idx = self.order[i * self.batch_size:(i + 1) * self.batch_size]
        return self.x[idx].astype("float32"), self.y[idx]

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.order)


def build(backbone: str, n_classes: int, weights: str | None = "imagenet"):
    """Input: RGB 224x224 floats in 0-255. Output: class probabilities. Returns (model, backbone)."""
    make, prep = BACKBONES[backbone]
    base = make(include_top=False, weights=weights, input_shape=(CLS_SIZE, CLS_SIZE, 3))
    base.trainable = False
    inp = keras.Input((CLS_SIZE, CLS_SIZE, 3), name="image")
    x = layers.Lambda(prep)(inp) if prep else inp
    x = base(x, training=False)  # BatchNorm stays in inference mode, also while fine-tuning
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    out = layers.Dense(n_classes, activation="softmax", name="probs")(x)
    return keras.Model(inp, out), base


def with_augmentation(model: keras.Model) -> keras.Model:
    """Random flips/rotation/zoom/contrast during training only; the exported model has none."""
    augment = keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"), layers.RandomRotation(0.1),
        layers.RandomZoom(0.1), layers.RandomContrast(0.1),
    ])
    inp = keras.Input((CLS_SIZE, CLS_SIZE, 3))
    return keras.Model(inp, model(augment(inp)))


def class_weights(y: np.ndarray, n: int) -> dict[int, float]:
    counts = np.bincount(y, minlength=n)
    return {i: len(y) / (n * max(c, 1)) for i, c in enumerate(counts)}


def fit(model, base, train: Batches, val: Batches, weights: dict, a) -> pd.DataFrame:
    """Train the new head with the backbone frozen, then fine-tune the top of the backbone."""
    trainer = with_augmentation(model)
    stop = keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)
    history = []
    for lr, epochs, unfreeze in [(1e-3, a.epochs_head, 0), (1e-5, a.epochs_finetune, a.unfreeze)]:
        base.trainable = unfreeze > 0
        for layer in base.layers[:-unfreeze or None]:
            layer.trainable = False
        trainer.compile(optimizer=keras.optimizers.Adam(lr), loss="sparse_categorical_crossentropy",
                        metrics=["accuracy"])
        h = trainer.fit(train, validation_data=val, epochs=epochs, class_weight=weights, callbacks=[stop], verbose=2)
        history.append(pd.DataFrame(h.history).assign(phase="head" if unfreeze == 0 else "finetune"))
    return pd.concat(history, ignore_index=True)


def plot_confusion(cm: list, labels: list[str], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(cm, cmap="Greens")
    ax.set(xticks=range(len(labels)), yticks=range(len(labels)), xticklabels=labels, yticklabels=labels,
           xlabel="Predicted", ylabel="True", title="Test confusion matrix")
    for (i, j), v in np.ndenumerate(np.array(cm)):
        ax.text(j, i, v, ha="center", va="center")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def evaluate(model, data: Batches, y: np.ndarray, labels: list[str]) -> dict:
    pred = model.predict(data, verbose=0).argmax(1)
    cm = confusion_matrix(y, pred, labels=range(len(labels)))
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro")),
        "per_class": classification_report(y, pred, labels=range(len(labels)), target_names=labels,
                                           output_dict=True, zero_division=0),
        "confusion_matrix": cm.tolist(),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--out", default="models/clf")
    ap.add_argument("--backbone", choices=sorted(BACKBONES), default="mobilenet_v3_small")
    ap.add_argument("--epochs-head", type=int, default=10)
    ap.add_argument("--epochs-finetune", type=int, default=10)
    ap.add_argument("--unfreeze", type=int, default=40, help="backbone layers to fine-tune in phase 2")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args(argv)

    keras.utils.set_random_seed(a.seed)
    root, out = data_root(a.data_root), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    df = load_manifest(root)
    labels = sorted(select(df)["label"].unique())
    rows = {s: select(df, split=s) for s in ("train", "val", "test")}
    if any(r.empty for r in rows.values()):
        raise SystemExit("Need train, val and test images. Run `maize-prepare --raw ...` (or --resplit) first.")
    y = {s: r["label"].map(labels.index).to_numpy() for s, r in rows.items()}
    print(f"classes={labels}  " + "  ".join(f"{s}={len(r)}" for s, r in rows.items()) + f"  backbone={a.backbone}")

    data = {s: Batches(load_images(r, root, CLS_SIZE), y[s], a.batch_size, shuffle=s == "train")
            for s, r in rows.items()}
    model, base = build(a.backbone, len(labels))
    fit(model, base, data["train"], data["val"], class_weights(y["train"], len(labels)), a).to_csv(
        out / "history.csv", index=False)

    # val picked the epochs (early stopping); test is untouched until now and gives the reported scores
    metrics = {s: evaluate(model, data[s], y[s], labels) for s in ("val", "test")}
    plot_confusion(metrics["test"]["confusion_matrix"], labels, out / "confusion.png")
    print(f"TEST accuracy={metrics['test']['accuracy']:.3f}  macro-F1={metrics['test']['macro_f1']:.3f}"
          f"   (val accuracy={metrics['val']['accuracy']:.3f})")
    model.export(str(out / "classifier.onnx"), format="onnx")
    info = {"labels": labels, "img_size": CLS_SIZE, "backbone": a.backbone, "min_confidence": MIN_CONFIDENCE,
            "args": vars(a), "metrics": metrics}
    (out / "classifier.json").write_text(json.dumps(info, indent=2))
    print(f"Saved classifier to {out}")
