"""tf.data pipelines over the preprocessed cache (scripts/04_preprocess.py).

Cache = 224x224 RGB PNG + 224x224 mask PNG per image. Augmentation is online, TRAIN ONLY,
so every augmented copy stays in the same set as its original.
Background is black (0) wherever the mask removes it.
"""
from __future__ import annotations

import pandas as pd
import tensorflow as tf
import keras

from ..constants import AE_SIZE, CLASS_TO_ID

AUTOTUNE = tf.data.AUTOTUNE


def _load(proc_path, mask_path):
    img = tf.io.decode_png(tf.io.read_file(proc_path), channels=3)
    msk = tf.io.decode_png(tf.io.read_file(mask_path), channels=1)
    # mask PNG stores {0,1}; stack as 4th channel scaled to 0/255 so geometric aug treats it like pixels
    return tf.concat([tf.cast(img, tf.float32), tf.cast(msk, tf.float32) * 255.0], axis=-1)


def _maybe(p, fn, x):
    return tf.cond(tf.random.uniform(()) < p, lambda: fn(x), lambda: x)


def _box_blur(x):
    k = tf.random.shuffle(tf.constant([3, 5]))[0]
    return tf.cond(k == 3,
                   lambda: tf.nn.avg_pool2d(x[None], 3, 1, "SAME")[0],
                   lambda: tf.nn.avg_pool2d(x[None], 5, 1, "SAME")[0])


def _jpeg(x):
    q = tf.random.uniform((), 40, 95, dtype=tf.int32)
    return tf.cast(tf.image.adjust_jpeg_quality(tf.cast(tf.round(tf.clip_by_value(x, 0, 255)), tf.uint8), q), tf.float32)


def build_augmenter(cfg: dict):
    """Returns fn(x4) -> x4 where x4 = HxWx4 (RGB 0-255, mask 0/255). Mild colour only:
    yellowing is part of the disease signal, so no hue shifts or strong colour correction."""
    geo = keras.Sequential([
        keras.layers.RandomFlip("horizontal_and_vertical"),
        keras.layers.RandomRotation(cfg.get("rotation", 0.1), fill_mode="constant", fill_value=0.0),
        keras.layers.RandomZoom(cfg.get("zoom", 0.15), fill_mode="constant", fill_value=0.0),
    ])

    def fn(x4):
        x4 = geo(x4, training=True)
        rgb, m = x4[..., :3], tf.cast(x4[..., 3:] > 127.0, tf.float32) * 255.0
        rgb = rgb * tf.random.uniform((), cfg.get("dark_min", 0.55), 1.1)               # darkening
        rgb = tf.image.random_contrast(rgb, 0.85, 1.15)
        rgb = rgb * tf.random.uniform((1, 1, 3), 0.95, 1.05)                              # mild channel gain
        rgb = _maybe(cfg.get("blur_p", 0.2), _box_blur, rgb)
        rgb = _maybe(cfg.get("jpeg_p", 0.3), _jpeg, rgb)
        rgb = tf.clip_by_value(rgb, 0.0, 255.0)
        return tf.concat([rgb, m], axis=-1)
    return fn


def _paths(df: pd.DataFrame):
    return df["proc_path"].astype(str).tolist(), df["mask_path"].astype(str).tolist()


def make_clf_dataset(df: pd.DataFrame, *, batch_size: int, training: bool, masked: bool,
                     aug_cfg: dict | None = None, seed: int = 42) -> tf.data.Dataset:
    """Yields (image float32 0-255 [224,224,3], label int). Backbone preprocessing is INSIDE the model."""
    p, m = _paths(df)
    y = df["label"].map(CLASS_TO_ID).astype("int32").values
    ds = tf.data.Dataset.from_tensor_slices((p, m, y))
    if training:
        ds = ds.shuffle(len(df), seed=seed, reshuffle_each_iteration=True)
    aug = build_augmenter(aug_cfg or {}) if training else None

    def f(pp, mm, yy):
        x4 = _load(pp, mm)
        if aug is not None:
            x4 = aug(x4)
        rgb = x4[..., :3]
        if masked:
            rgb = rgb * (x4[..., 3:] / 255.0)
        return rgb, yy

    return ds.map(f, num_parallel_calls=AUTOTUNE).batch(batch_size).prefetch(AUTOTUNE)


def _to_ae(x4):
    x4 = tf.image.resize(x4, (AE_SIZE, AE_SIZE), method="area")
    rgb = x4[..., :3] / 255.0
    m = tf.cast(x4[..., 3:] / 255.0 > 0.5, tf.float32)
    return rgb * m, m


def make_ae_dataset(df: pd.DataFrame, *, batch_size: int, training: bool,
                    aug_cfg: dict | None = None, seed: int = 42) -> tf.data.Dataset:
    """Healthy leaves only. Yields (x, y) with x in [0,1] masked and y = concat[x, mask] (loss needs the mask)."""
    p, m = _paths(df)
    ds = tf.data.Dataset.from_tensor_slices((p, m))
    if training:
        ds = ds.shuffle(len(df), seed=seed, reshuffle_each_iteration=True)
    aug = build_augmenter(aug_cfg or {}) if training else None

    def f(pp, mm):
        x4 = _load(pp, mm)
        if aug is not None:
            x4 = aug(x4)
        x, msk = _to_ae(x4)
        return x, tf.concat([x, msk], axis=-1)

    return ds.map(f, num_parallel_calls=AUTOTUNE).batch(batch_size).prefetch(AUTOTUNE)


def make_ae_eval_dataset(df: pd.DataFrame, *, batch_size: int) -> tf.data.Dataset:
    """Yields (x, mask) for computing reconstruction errors (no augmentation, no labels)."""
    p, m = _paths(df)

    def f(pp, mm):
        return _to_ae(_load(pp, mm))

    return tf.data.Dataset.from_tensor_slices((p, m)).map(f, num_parallel_calls=AUTOTUNE).batch(batch_size).prefetch(AUTOTUNE)
