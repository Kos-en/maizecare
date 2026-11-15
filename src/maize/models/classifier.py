"""Stage 1: four pretrained Keras backbones behind one identical head (proposal 3.5.1).

Every model takes RAW pixels 0-255 (float). Each backbone's own preprocessing is a layer
INSIDE the model, so training, TFLite and the web service never have to agree on it.
Head: GlobalAveragePooling -> Dropout(0.3) -> Dense(4, softmax)  (this exact head is what
lets the web service draw a CAM heat map from TFLite without gradients).
"""
from __future__ import annotations

import keras
from keras import ops

from ..constants import CLASSES, CLF_SIZE

# name -> (keras.applications class, preprocessing)
BACKBONES = {
    "resnet50": ("ResNet50", "caffe"),            # BGR + mean subtraction
    "mobilenetv2": ("MobileNetV2", "minus1_1"),   # [-1, 1]
    "mobilenetv3small": ("MobileNetV3Small", "builtin"),  # expects 0-255, rescales inside
    "efficientnetb0": ("EfficientNetB0", "builtin"),      # expects 0-255, rescales inside
}


@keras.saving.register_keras_serializable(package="maize")
class CaffePreprocess(keras.layers.Layer):
    """ResNet50 (Keras 'caffe' mode): RGB->BGR, subtract ImageNet BGR mean."""

    def call(self, x):
        x = ops.flip(x, axis=-1)
        return x - ops.convert_to_tensor([103.939, 116.779, 123.68], dtype=x.dtype)


def build_classifier(backbone: str, num_classes: int = len(CLASSES), img_size: int = CLF_SIZE,
                     dropout: float = 0.3, weights: str | None = "imagenet") -> keras.Model:
    app_name, prep = BACKBONES[backbone]
    inputs = keras.Input((img_size, img_size, 3), name="image")  # raw 0-255
    x = inputs
    if prep == "caffe":
        x = CaffePreprocess(name="preprocess")(x)
    elif prep == "minus1_1":
        x = keras.layers.Rescaling(1 / 127.5, offset=-1.0, name="preprocess")(x)
    kwargs = {"include_preprocessing": True} if prep == "builtin" else {}
    base = getattr(keras.applications, app_name)(
        include_top=False, weights=weights, input_shape=(img_size, img_size, 3), **kwargs)
    base._name = "backbone"
    feats = base(x)
    pooled = keras.layers.GlobalAveragePooling2D(name="gap")(feats)
    pooled = keras.layers.Dropout(dropout, name="dropout")(pooled)
    out = keras.layers.Dense(num_classes, activation="softmax", name="head")(pooled)
    return keras.Model(inputs, out, name=f"clf_{backbone}")


def set_phase(model: keras.Model, phase: int, unfreeze_last: int = 40) -> None:
    """Phase 1: backbone frozen (train head). Phase 2: unfreeze the top blocks; BatchNorm stays frozen."""
    base = model.get_layer("backbone")
    if phase == 1:
        base.trainable = False
        return
    base.trainable = True
    for layer in base.layers[:-unfreeze_last]:
        layer.trainable = False
    for layer in base.layers:
        if isinstance(layer, keras.layers.BatchNormalization):
            layer.trainable = False


def cam_model(model: keras.Model) -> keras.Model:
    """Same weights, two outputs: last conv feature maps (for CAM) and the class probabilities."""
    feats = model.get_layer("backbone").output
    return keras.Model(model.input, [feats, model.output], name=model.name + "_cam")


def head_weights(model: keras.Model):
    """(channels, classes) kernel of the Dense head: heatmap = feats @ W[:, class]."""
    return model.get_layer("head").get_weights()[0]
