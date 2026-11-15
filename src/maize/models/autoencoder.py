"""Stage 2: convolutional autoencoder and VAE (proposal 3.5.2).

Encoder: four stride-2 conv blocks (32..256 filters), decoder mirrors it. Input/output in
[0,1], 128x128, background pixels already zeroed. The loss ignores pixels outside the leaf mask.
At inference the VAE uses the mean (no sampling), so the TFLite export is deterministic.
"""
from __future__ import annotations

import keras
import tensorflow as tf
from keras import layers

from ..constants import AE_SIZE


def _enc_block(x, f):
    x = layers.Conv2D(f, 3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    return layers.LeakyReLU(0.1)(x)


def _dec_block(x, f):
    x = layers.Conv2DTranspose(f, 3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    return layers.LeakyReLU(0.1)(x)


@keras.saving.register_keras_serializable(package="maize")
class Sampling(layers.Layer):
    """z = mu + sigma*eps in training (adds KL loss); z = mu at inference."""

    def __init__(self, kl_weight=1e-4, **kw):
        super().__init__(**kw)
        self.kl_weight = kl_weight

    def call(self, inputs, training=None):
        mu, logvar = inputs
        if training:
            kl = -0.5 * tf.reduce_mean(tf.reduce_sum(1 + logvar - tf.square(mu) - tf.exp(logvar), axis=-1))
            self.add_loss(self.kl_weight * kl)
            return mu + tf.exp(0.5 * logvar) * tf.random.normal(tf.shape(mu))
        return mu

    def get_config(self):
        return {**super().get_config(), "kl_weight": self.kl_weight}


def build_cae(img_size=AE_SIZE, filters=(32, 64, 128, 256), latent_channels=16) -> keras.Model:
    inp = keras.Input((img_size, img_size, 3), name="x")
    x = inp
    for f in filters:
        x = _enc_block(x, f)
    code = layers.Conv2D(latent_channels, 1, name="code")(x)          # e.g. 8x8x16 bottleneck
    x = layers.Conv2D(filters[-1], 1)(code)
    for f in list(reversed(filters))[1:]:
        x = _dec_block(x, f)
    out = layers.Conv2DTranspose(3, 3, strides=2, padding="same", activation="sigmoid", name="recon")(x)
    return keras.Model(inp, out, name="cae")


def build_vae(img_size=AE_SIZE, filters=(32, 64, 128, 256), latent_dim=128, kl_weight=1e-4) -> keras.Model:
    inp = keras.Input((img_size, img_size, 3), name="x")
    x = inp
    for f in filters:
        x = _enc_block(x, f)
    side = img_size // 2 ** len(filters)
    x = layers.Flatten()(x)
    mu = layers.Dense(latent_dim, name="mu")(x)
    logvar = layers.Dense(latent_dim, name="logvar")(x)
    z = Sampling(kl_weight, name="z")([mu, logvar])
    x = layers.Dense(side * side * filters[-1])(z)
    x = layers.Reshape((side, side, filters[-1]))(x)
    for f in list(reversed(filters))[1:]:
        x = _dec_block(x, f)
    out = layers.Conv2DTranspose(3, 3, strides=2, padding="same", activation="sigmoid", name="recon")(x)
    return keras.Model(inp, out, name="vae")


@keras.saving.register_keras_serializable(package="maize")
class MaskedReconLoss(keras.losses.Loss):
    """y_true = concat[x (3 ch), mask (1 ch)]. (1-a)*masked MSE + a*(1-SSIM). a=0 is plain MSE."""

    def __init__(self, ssim_weight=0.0, **kw):
        super().__init__(**kw)
        self.ssim_weight = ssim_weight

    def call(self, y_true, y_pred):
        t, m = y_true[..., :3], y_true[..., 3:4]
        mse = tf.reduce_sum(m * tf.square(t - y_pred), axis=[1, 2, 3]) / (3.0 * tf.reduce_sum(m, axis=[1, 2, 3]) + 1e-6)
        if self.ssim_weight == 0:
            return mse
        ssim = tf.image.ssim(t * m, y_pred * m, max_val=1.0)
        return (1 - self.ssim_weight) * mse + self.ssim_weight * (1.0 - ssim)

    def get_config(self):
        return {**super().get_config(), "ssim_weight": self.ssim_weight}


def build_autoencoder(kind: str, **kw) -> keras.Model:
    return {"cae": build_cae, "vae": build_vae}[kind](**kw)
