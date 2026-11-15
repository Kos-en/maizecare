"""Two-phase classifier training for one backbone and one CV fold (proposal 3.5.1)."""
from __future__ import annotations

import json
from pathlib import Path

import keras
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, recall_score

from ..config import env_info, runs_root, set_seed, write_json
from ..constants import CLASSES, CLASS_TO_ID
from ..data.dataset import make_clf_dataset
from ..models.classifier import build_classifier, set_phase


def run_dir(cfg: dict, fold: int) -> Path:
    tag = "masked" if cfg["masked"] else "raw"
    return runs_root() / "clf" / f"{cfg['backbone']}_{tag}" / f"fold{fold}"


class MacroF1Best(keras.callbacks.Callback):
    """Keeps the weights with the best validation macro-F1 (the selection metric)."""

    def __init__(self, val_ds, y_val, path: Path):
        super().__init__()
        self.val_ds, self.y_val, self.path, self.best = val_ds, y_val, path, -1.0

    def on_epoch_end(self, epoch, logs=None):
        pred = self.model.predict(self.val_ds, verbose=0).argmax(1)
        f1 = f1_score(self.y_val, pred, average="macro")
        if logs is not None:
            logs["val_macro_f1"] = f1
        if f1 > self.best:
            self.best = f1
            self.model.save_weights(self.path)


def class_weights(y: np.ndarray, k: int) -> dict:
    counts = np.bincount(y, minlength=k).astype(float)
    return {i: float(len(y) / (k * c)) if c else 1.0 for i, c in enumerate(counts)}


def oversample(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    n = df["label"].value_counts().max()
    parts = [g.sample(n, replace=True, random_state=seed) for _, g in df.groupby("label")]
    return pd.concat(parts).sample(frac=1, random_state=seed)


def train_one(cfg: dict, fold: int, manifest: pd.DataFrame, force: bool = False) -> dict:
    out = run_dir(cfg, fold)
    if (out / "metrics.json").exists() and not force:
        print(f"[skip] {out} already done (use --force)")
        return json.loads((out / "metrics.json").read_text())
    out.mkdir(parents=True, exist_ok=True)
    set_seed(cfg["seed"] + fold)
    write_json(out / "config.json", cfg)
    write_json(out / "env.json", env_info())

    pool = manifest[manifest["split"] == "pool"]
    train_df, val_df = pool[pool["fold"] != fold], pool[pool["fold"] == fold]
    balance = cfg.get("balance", "class_weight")
    if balance == "oversample":
        train_df = oversample(train_df, cfg["seed"])
    y_tr = train_df["label"].map(CLASS_TO_ID).values
    y_val = val_df["label"].map(CLASS_TO_ID).values
    cw = class_weights(y_tr, len(CLASSES)) if balance == "class_weight" else None

    bs, masked = cfg["batch_size"], cfg["masked"]
    ds_tr = make_clf_dataset(train_df, batch_size=bs, training=True, masked=masked, aug_cfg=cfg.get("augment"), seed=cfg["seed"])
    ds_val = make_clf_dataset(val_df, batch_size=bs, training=False, masked=masked)

    model = build_classifier(cfg["backbone"], dropout=cfg["dropout"])
    best_path = out / "best.weights.h5"
    f1_cb = MacroF1Best(ds_val, y_val, best_path)
    history = []

    for phase, p in ((1, cfg["phase1"]), (2, cfg["phase2"])):
        set_phase(model, phase, cfg.get("unfreeze_last", 40))
        model.compile(optimizer=keras.optimizers.Adam(p["lr"]),
                      loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        cbs = [f1_cb,
               keras.callbacks.EarlyStopping(monitor="val_loss", patience=p["patience"], restore_best_weights=False),
               keras.callbacks.BackupAndRestore(str(out / f"backup_p{phase}"), delete_checkpoint=True)]
        h = model.fit(ds_tr, validation_data=ds_val, epochs=p["epochs"], class_weight=cw, callbacks=cbs, verbose=2)
        history.append({f"phase{phase}": h.history})

    model.load_weights(best_path)
    model.save(out / "best.keras")
    pred = model.predict(ds_val, verbose=0)
    yhat = pred.argmax(1)
    metrics = {
        "backbone": cfg["backbone"], "masked": masked, "fold": fold,
        "val_macro_f1": float(f1_score(y_val, yhat, average="macro")),
        "val_recall": dict(zip(CLASSES, map(float, recall_score(y_val, yhat, average=None, labels=range(len(CLASSES)))))),
        "n_train": len(train_df), "n_val": len(val_df), "params": int(model.count_params()),
    }
    write_json(out / "history.json", history)
    np.save(out / "val_probs.npy", pred)
    write_json(out / "metrics.json", metrics)
    return metrics
