"""TFLite runtime wrapper. Imports whichever runtime exists, so the same code runs in the
slim Render image (ai-edge-litert) and in the full training env (tensorflow)."""
from __future__ import annotations

import numpy as np


def _interpreter_cls():
    try:
        from ai_edge_litert.interpreter import Interpreter
        return Interpreter
    except ImportError:
        pass
    try:
        from tflite_runtime.interpreter import Interpreter
        return Interpreter
    except ImportError:
        pass
    import tensorflow as tf
    return tf.lite.Interpreter


class TFLiteModel:
    def __init__(self, path: str, num_threads: int = 1):
        self.interp = _interpreter_cls()(model_path=str(path), num_threads=num_threads)
        self.interp.allocate_tensors()
        self.inp = self.interp.get_input_details()[0]
        self.outs = self.interp.get_output_details()

    def __call__(self, x: np.ndarray) -> list[np.ndarray]:
        """x: float32 batch matching the model input. Returns all outputs (float32), model order."""
        x = np.ascontiguousarray(x, dtype=np.float32)
        if self.inp["dtype"] != np.float32:  # defensive: integer-IO models
            scale, zp = self.inp["quantization"]
            x = np.round(x / scale + zp).astype(self.inp["dtype"])
        self.interp.set_tensor(self.inp["index"], x)
        self.interp.invoke()
        return [self.interp.get_tensor(o["index"]).astype(np.float32) for o in self.outs]


def split_cam_outputs(outputs: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Classifier export has two outputs: conv features (4-D) and probabilities (2-D)."""
    feats = next(o for o in outputs if o.ndim == 4)
    probs = next(o for o in outputs if o.ndim == 2)
    return feats[0], probs[0]
