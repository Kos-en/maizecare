"""Config loading, paths and reproducibility helpers (no heavy imports at module level)."""
from __future__ import annotations

import json
import os
import platform
import random
from pathlib import Path

import numpy as np
import yaml


def data_root() -> Path:
    """Set MAIZE_DATA_ROOT on Colab (e.g. /content/data after copying from Drive)."""
    return Path(os.environ.get("MAIZE_DATA_ROOT", "data"))


def runs_root() -> Path:
    """Set MAIZE_RUNS_ROOT to a Drive folder on Colab so runs survive disconnects."""
    return Path(os.environ.get("MAIZE_RUNS_ROOT", "runs"))


def _deep_merge(a: dict, b: dict) -> dict:
    out = dict(a)
    for k, v in b.items():
        out[k] = _deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def _coerce(v: str):
    try:
        return yaml.safe_load(v)
    except yaml.YAMLError:
        return v


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict:
    """Load YAML; supports `_base: other.yaml` (relative) and `a.b=value` overrides."""
    path = Path(path)
    cfg = yaml.safe_load(path.read_text()) or {}
    base = cfg.pop("_base", None)
    if base:
        cfg = _deep_merge(load_config(path.parent / base), cfg)
    for item in overrides or []:
        key, _, val = item.partition("=")
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = _coerce(val)
    return cfg


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import keras
        keras.utils.set_random_seed(seed)
    except ImportError:
        pass


def env_info() -> dict:
    info = {"python": platform.python_version(), "platform": platform.platform()}
    try:
        import tensorflow as tf
        info["tensorflow"] = tf.__version__
        info["gpus"] = [g.name for g in tf.config.list_physical_devices("GPU")]
    except ImportError:
        pass
    return info


def write_json(path: str | Path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=float))
