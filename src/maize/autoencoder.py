"""Stage 2: severity. A convolutional autoencoder trained on healthy leaves only, inside the leaf mask
(proposal 3.4 step 7, 3.5.2 and eq. 3.2).

Encoder: 4 stride-2 conv blocks, 32 -> 256 filters (128 px -> 8 px), then a 1x1 conv down to LATENT = 32
channels. The bottleneck holds 8 x 8 x 32 = 2,048 numbers for a 128 x 128 x 3 = 49,152-value image:
24x compression, so the model cannot just copy its input. The decoder mirrors the encoder.

Background (outside the leaf mask) is set to 0 at the input and ignored by the loss and the error,
so only leaf tissue counts. A diseased leaf reconstructs badly; evaluate.py turns its error into a
0-1 severity score.

    maize-train-autoencoder ROOT MODELS     -> MODELS/ae/
"""
from __future__ import annotations

import os

os.environ.setdefault("KERAS_BACKEND", "torch")

from pathlib import Path

import keras
import numpy as np
import pandas as pd
from keras import layers

from maize.pipeline import HEALTHY, command_args, load, load_usable_manifest, select
from maize.preprocess import AE_SIZE, mask_background, reconstruction_error

LATENT = 32       # bottleneck channels
MIN_LEAF = 0.10   # healthy training images with less leaf than this (no leaf found) are skipped
EPOCHS = 30       # early stopping usually ends sooner
SEED = 42
MODEL_FILE = "autoencoder.keras"


def build() -> keras.Model:
    """Input: masked RGB image 0-255. Output: its reconstruction in 0-1."""
    image = keras.Input((AE_SIZE, AE_SIZE, 3), name="image")
    x = layers.Rescaling(1 / 255)(image)
    for filters in (32, 64, 128, 256):  # 128 -> 64 -> 32 -> 16 -> 8 px
        x = layers.Conv2D(filters, 3, strides=2, padding="same", activation="relu")(x)
    x = layers.Conv2D(LATENT, 1, activation="relu", name="bottleneck")(x)
    for filters in (256, 128, 64, 32):  # 8 -> 16 -> 32 -> 64 -> 128 px
        x = layers.Conv2DTranspose(filters, 3, strides=2, padding="same", activation="relu")(x)
    recon = layers.Conv2D(3, 3, padding="same", activation="sigmoid", name="recon")(x)
    return keras.Model(image, recon)


def masked_loss(y, recon):
    """Mean squared error over leaf pixels only. y has 4 channels: the RGB target (0-255) and the mask (0/255)."""
    target = keras.ops.cast(y[..., :3], "float32") / 255.0
    mask = keras.ops.cast(y[..., 3:], "float32") / 255.0
    squared = keras.ops.square(target - recon) * mask
    leaf_values = 3 * keras.ops.sum(mask, axis=(1, 2, 3)) + 1e-6
    return keras.ops.sum(squared, axis=(1, 2, 3)) / leaf_values


def training_targets(images: np.ndarray, masks: np.ndarray) -> np.ndarray:
    """y for masked_loss: the masked image plus the mask as a 4th channel, uint8 (N, H, W, 4)."""
    return np.concatenate([mask_background(images, masks), masks[..., None] * 255], axis=-1)


class Autoencoder:
    """The Keras model plus everything done with it: train, reconstruct, measure errors, save."""

    def __init__(self, model: keras.Model):
        self.model = model

    @classmethod
    def new(cls) -> "Autoencoder":
        keras.utils.set_random_seed(SEED)
        model = build()
        model.compile(optimizer=keras.optimizers.Adam(1e-3), loss=masked_loss)
        return cls(model)

    @classmethod
    def load(cls, folder: Path) -> "Autoencoder":
        return cls(keras.saving.load_model(Path(folder) / MODEL_FILE, compile=False))

    def fit(self, train: tuple[np.ndarray, np.ndarray], val: tuple[np.ndarray, np.ndarray], epochs: int,
            batch_size: int = 64) -> pd.DataFrame:
        """train, val: (images, masks). Early stopping on val loss keeps the best epoch.
        Healthy leaves at 128 px are small enough to pass to Keras as plain arrays."""
        def inputs_and_targets(images, masks):
            return mask_background(images, masks).astype("float32"), training_targets(images, masks)

        stop = keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
        history = self.model.fit(*inputs_and_targets(*train), validation_data=inputs_and_targets(*val),
                                 batch_size=batch_size, epochs=epochs, callbacks=[stop], verbose=2)
        return pd.DataFrame(history.history)

    def reconstruct(self, images: np.ndarray, masks: np.ndarray) -> np.ndarray:
        return self.model.predict(mask_background(images, masks).astype("float32"), batch_size=64, verbose=0)

    def errors(self, images: np.ndarray, masks: np.ndarray, chunk: int = 1024) -> np.ndarray:
        """e(x) per image, in the order given. In chunks, so memory stays low."""
        errors = []
        for start in range(0, len(images), chunk):
            part = slice(start, start + chunk)
            recons = self.reconstruct(images[part], masks[part])
            errors.append(reconstruction_error(images[part], recons, masks[part]))
        return np.concatenate(errors)

    def save(self, folder: Path) -> None:
        """autoencoder.keras for evaluation, model.onnx for the web app."""
        folder.mkdir(parents=True, exist_ok=True)
        self.model.save(folder / MODEL_FILE)
        self.model.export(str(folder / "model.onnx"), format="onnx")


def train(df: pd.DataFrame, root: Path, out: Path, epochs: int = EPOCHS) -> None:
    """Healthy train leaves for training, healthy val leaves for early stopping. Writes to out/."""
    images, masks = load(select(df, HEALTHY, "train"), root, AE_SIZE)
    has_leaf = masks.mean(axis=(1, 2)) >= MIN_LEAF
    print(f"Autoencoder: skipping {int((~has_leaf).sum())} of {len(has_leaf)} healthy train images "
          f"with no leaf found (leaf fraction < {MIN_LEAF}).", flush=True)
    val = load(select(df, HEALTHY, "val"), root, AE_SIZE)

    autoencoder = Autoencoder.new()
    history = autoencoder.fit((images[has_leaf], masks[has_leaf]), val, epochs)
    autoencoder.save(out)
    history.to_csv(out / "history.csv", index=False)


def main():
    root, out = (Path(a) for a in command_args("maize-train-autoencoder DATA_ROOT MODELS", required=2)[:2])
    train(load_usable_manifest(root), root, out / "ae")
