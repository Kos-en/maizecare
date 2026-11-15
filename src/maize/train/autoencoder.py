"""Autoencoder training on HEALTHY leaves of the training folds only (proposal 3.4 / 3.5.2)."""
from __future__ import annotations

import json
from pathlib import Path

import keras
import pandas as pd

from ..config import env_info, runs_root, set_seed, write_json
from ..constants import HEALTHY
from ..data.dataset import make_ae_dataset
from ..models.autoencoder import MaskedReconLoss, build_autoencoder


def run_dir(cfg: dict, fold: int) -> Path:
    return runs_root() / "ae" / cfg["name"] / f"fold{fold}"


def train_one(cfg: dict, fold: int, manifest: pd.DataFrame, force: bool = False) -> dict:
    out = run_dir(cfg, fold)
    if (out / "metrics.json").exists() and not force:
        print(f"[skip] {out} already done (use --force)")
        return json.loads((out / "metrics.json").read_text())
    out.mkdir(parents=True, exist_ok=True)
    set_seed(cfg["seed"] + fold)
    write_json(out / "config.json", cfg)
    write_json(out / "env.json", env_info())

    pool = manifest[(manifest["split"] == "pool") & (manifest["label"] == HEALTHY)]
    train_df, val_df = pool[pool["fold"] != fold], pool[pool["fold"] == fold]
    bs = cfg["batch_size"]
    ds_tr = make_ae_dataset(train_df, batch_size=bs, training=True, aug_cfg=cfg.get("augment"), seed=cfg["seed"])
    ds_val = make_ae_dataset(val_df, batch_size=bs, training=False)

    kw = {"latent_channels": cfg["latent_channels"]} if cfg["kind"] == "cae" else \
         {"latent_dim": cfg["latent_dim"], "kl_weight": cfg["kl_weight"]}
    model = build_autoencoder(cfg["kind"], **kw)
    model.compile(optimizer=keras.optimizers.Adam(cfg["lr"]), loss=MaskedReconLoss(cfg["ssim_weight"]))
    cbs = [keras.callbacks.ModelCheckpoint(str(out / "best.keras"), monitor="val_loss", save_best_only=True),
           keras.callbacks.EarlyStopping(monitor="val_loss", patience=cfg["patience"]),
           keras.callbacks.BackupAndRestore(str(out / "backup"), delete_checkpoint=True)]
    h = model.fit(ds_tr, validation_data=ds_val, epochs=cfg["epochs"], callbacks=cbs, verbose=2)

    best = min(h.history["val_loss"])
    metrics = {"name": cfg["name"], "fold": fold, "best_val_loss": float(best),
               "epochs_run": len(h.history["loss"]), "n_train": len(train_df), "n_val": len(val_df)}
    write_json(out / "history.json", h.history)
    write_json(out / "metrics.json", metrics)
    return metrics
