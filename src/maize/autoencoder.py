"""Stage 2: severity. Two autoencoders are trained on healthy leaves only, inside the leaf mask, and the one
whose error best separates healthy from diseased validation leaves is used (proposal 3.4 step 7, 3.5.2, eq. 3.2).

    conv   convolutional autoencoder
    vae    variational autoencoder of the same shape (Kingma & Welling 2014)

Encoder: 4 stride-2 conv blocks, 32 -> 256 filters (128 px -> 8 px), then a 1x1 conv down to LATENT = 32
channels. The code holds 8 x 8 x 32 = 2,048 numbers for a 128 x 128 x 3 = 49,152-value image: 24x
compression, so the model cannot just copy its input. The decoder mirrors the encoder.

The VAE's encoder gives a mean and a spread for every code number. While training, the decoder rebuilds
the leaf from a random draw within that spread and a KL penalty keeps the code close to a standard normal;
this makes it harder to copy diseased leaves well. When scoring (and in the web app) it uses the mean.

Background (outside the leaf mask) is set to 0 at the input and ignored by the loss and the error,
so only leaf tissue counts. A diseased leaf reconstructs badly; evaluate.py turns its error into a
0-1 severity score.

    maize-train-autoencoder ROOT MODELS [KIND ...]     default: both kinds
        -> MODELS/ae_<kind>/ and MODELS/autoencoders.csv (the comparison; the top row is used)
"""
from __future__ import annotations

import os

os.environ.setdefault("KERAS_BACKEND", "torch")

import json
from pathlib import Path

import keras
import numpy as np
import pandas as pd
from keras import layers, ops

from maize.evaluate import auroc
from maize.pipeline import HEALTHY, command_args, load, load_usable_manifest, select
from maize.preprocess import AE_SIZE, mask_background, reconstruction_error

KINDS = ("conv", "vae")
LATENT = 32       # code channels
MIN_LEAF = 0.10   # healthy training images with less leaf than this (no leaf found) are skipped
EPOCHS = 30       # early stopping usually ends sooner
SEED = 42
MODEL_FILE = "autoencoder.keras"
# Weight of the VAE's KL penalty (a mean over code numbers) against the squared error (a mean over leaf
# values): 2,048 code numbers / 49,152 image values, the balance of the original VAE loss, where both are sums.
BETA = (AE_SIZE // 16) ** 2 * LATENT / (AE_SIZE * AE_SIZE * 3)


# ---- the networks --------------------------------------------------------------------------------
def encode(image):
    x = layers.Rescaling(1 / 255)(image)
    for filters in (32, 64, 128, 256):  # 128 -> 64 -> 32 -> 16 -> 8 px
        x = layers.Conv2D(filters, 3, strides=2, padding="same", activation="relu")(x)
    return x


def decode(code):
    x = code
    for filters in (256, 128, 64, 32):  # 8 -> 16 -> 32 -> 64 -> 128 px
        x = layers.Conv2DTranspose(filters, 3, strides=2, padding="same", activation="relu")(x)
    return layers.Conv2D(3, 3, padding="same", activation="sigmoid", name="recon")(x)


@keras.saving.register_keras_serializable(package="maize")
class Sample(layers.Layer):
    """The VAE code. While training: a random draw around the mean, and the KL penalty. Otherwise: the mean."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.seed = keras.random.SeedGenerator(SEED)

    def call(self, mean, log_var, training=None):
        if not training:
            return mean
        log_var = ops.clip(log_var, -10.0, 10.0)
        kl = -0.5 * ops.mean(1 + log_var - ops.square(mean) - ops.exp(log_var))
        self.add_loss(BETA * kl)
        return mean + ops.exp(0.5 * log_var) * keras.random.normal(ops.shape(mean), seed=self.seed)


def build(kind: str = "conv") -> keras.Model:
    """Input: masked RGB image 0-255. Output: its reconstruction in 0-1."""
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}, not {kind!r}")
    image = keras.Input((AE_SIZE, AE_SIZE, 3), name="image")
    x = encode(image)
    if kind == "vae":
        mean = layers.Conv2D(LATENT, 1, name="code_mean")(x)
        log_var = layers.Conv2D(LATENT, 1, name="code_log_var")(x)
        code = Sample(name="code")(mean, log_var)
    else:
        code = layers.Conv2D(LATENT, 1, activation="relu", name="code")(x)
    return keras.Model(image, decode(code), name=f"ae_{kind}")


def masked_loss(y, recon):
    """Mean squared error over leaf pixels only. y has 4 channels: the RGB target (0-255) and the mask (0/255)."""
    target = ops.cast(y[..., :3], "float32") / 255.0
    mask = ops.cast(y[..., 3:], "float32") / 255.0
    squared = ops.square(target - recon) * mask
    leaf_values = 3 * ops.sum(mask, axis=(1, 2, 3)) + 1e-6
    return ops.sum(squared, axis=(1, 2, 3)) / leaf_values


def training_targets(images: np.ndarray, masks: np.ndarray) -> np.ndarray:
    """y for masked_loss: the masked image plus the mask as a 4th channel, uint8 (N, H, W, 4)."""
    return np.concatenate([mask_background(images, masks), masks[..., None] * 255], axis=-1)


class Autoencoder:
    """The Keras model plus everything done with it: train, reconstruct, measure errors, save."""

    def __init__(self, model: keras.Model):
        self.model = model

    @classmethod
    def new(cls, kind: str = "conv") -> "Autoencoder":
        keras.utils.set_random_seed(SEED)
        model = build(kind)
        model.compile(optimizer=keras.optimizers.Adam(1e-3), loss=masked_loss)  # the VAE adds its KL itself
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


# ---- choosing between them -----------------------------------------------------------------------
def separation(labels: pd.Series, errors: pd.Series) -> dict:
    """How well the errors of one autoencoder separate healthy from each disease (val leaves)."""
    healthy = errors[labels == HEALTHY]
    aurocs = {f"auroc_{d}": auroc(healthy, errors[labels == d]) for d in sorted(set(labels) - {HEALTHY})}
    return {"healthy_median_error": float(healthy.median()), **aurocs,
            "mean_auroc": float(np.mean(list(aurocs.values())))}


def compare(models: Path) -> pd.DataFrame:
    """One row per trained autoencoder (MODELS/ae_<kind>/validation.json), best mean AUROC first."""
    rows = [{"kind": f.parent.name.removeprefix("ae_"), **json.loads(f.read_text())}
            for f in Path(models).glob("ae_*/validation.json")]
    if not rows:
        raise SystemExit(f"No trained autoencoder in {models}: run maize-train-autoencoder first.")
    return pd.DataFrame(rows).sort_values("mean_auroc", ascending=False, ignore_index=True)


def chosen(models: Path) -> Path:
    """The folder of the autoencoder that separates healthy from diseased val leaves best."""
    return Path(models) / f"ae_{compare(models)['kind'][0]}"


# ---- the step ------------------------------------------------------------------------------------
def healthy_training_leaves(df: pd.DataFrame, root: Path) -> tuple[np.ndarray, np.ndarray]:
    images, masks = load(select(df, HEALTHY, "train"), root, AE_SIZE)
    has_leaf = masks.mean(axis=(1, 2)) >= MIN_LEAF
    print(f"Autoencoder: skipping {int((~has_leaf).sum())} of {len(has_leaf)} healthy train images "
          f"with no leaf found (leaf fraction < {MIN_LEAF}).", flush=True)
    return images[has_leaf], masks[has_leaf]


def train(df: pd.DataFrame, root: Path, out: Path, kinds=KINDS, epochs: int = EPOCHS) -> pd.DataFrame:
    """Healthy train leaves to learn, healthy val leaves for early stopping, all val leaves to compare.
    Writes out/ae_<kind>/ for each kind and out/autoencoders.csv."""
    train_leaves = healthy_training_leaves(df, root)
    healthy_val = load(select(df, HEALTHY, "val"), root, AE_SIZE)
    val_rows = select(df, split="val")
    val_images, val_masks = load(val_rows, root, AE_SIZE)

    for kind in kinds:
        print(f"\n--- autoencoder: {kind} ---", flush=True)
        folder = out / f"ae_{kind}"
        autoencoder = Autoencoder.new(kind)
        history = autoencoder.fit(train_leaves, healthy_val, epochs)
        autoencoder.save(folder)
        history.to_csv(folder / "history.csv", index=False)
        errors = pd.Series(autoencoder.errors(val_images, val_masks), index=val_rows.index)
        (folder / "validation.json").write_text(json.dumps(separation(val_rows["label"], errors), indent=2))
        keras.backend.clear_session()

    table = compare(out)
    table.to_csv(out / "autoencoders.csv", index=False)
    print("\n" + table.to_string(index=False))
    return table


def main():
    root, out, *kinds = command_args("maize-train-autoencoder DATA_ROOT MODELS [KIND ...]", required=2)
    root, out = Path(root), Path(out)
    train(load_usable_manifest(root), root, out, kinds or KINDS)
