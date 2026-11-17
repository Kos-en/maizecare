"""Feeds cached uint8 images to Keras as float32 batches. Shared by the classifier and the autoencoder."""
from __future__ import annotations

import os

os.environ.setdefault("KERAS_BACKEND", "torch")

import math

import keras
import numpy as np

SEED = 42  # one seed for every training run, so results can be repeated


class Batches(keras.utils.PyDataset):
    """x: uint8 images (N, H, W, C). y: labels or targets (N, ...). Converts one batch at a time,
    so the whole dataset never sits in memory as floats."""

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
