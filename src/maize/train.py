"""Train the autoencoder on healthy leaves only (train split of the manifest).

    python -m maize.train --epochs 5 --out models/test

Writes healthy_autoencoder.pt, model.onnx, history.csv and meta.json to the output folder.
meta.json holds data_version, a hash of the images used, to trace a model to its data.
"""
from __future__ import annotations

import argparse
import hashlib
import random
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .autoencoder import ConvAutoencoder, LeafDataset, export_onnx, pick_device, save_model
from .data import HEALTHY, data_root, load_manifest, select


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def data_version(df: pd.DataFrame) -> str:
    """Hash of exactly which healthy images were used for training and validation."""
    h = df[(df["status"] == "ok") & (df["label"] == HEALTHY)]
    lines = sorted(f"{r.sha1}:{r.split}" for r in h.itertuples())
    return hashlib.sha1("\n".join(lines).encode()).hexdigest()[:12]


def git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL,
                                       text=True).strip()
    except Exception:
        return None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--out", default="models/ae")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args(argv)

    seed_everything(a.seed)
    device = pick_device()
    root = data_root(a.data_root)
    df = load_manifest(root)
    train_paths = select(df, root, label=HEALTHY, split="train")
    val_paths = select(df, root, label=HEALTHY, split="val")
    if not train_paths or not val_paths:
        raise SystemExit("No healthy train/val images in the manifest. Run `python -m maize.prepare --raw ...` first.")
    print(f"Device: {device}   healthy train={len(train_paths)}  val={len(val_paths)}")

    train_loader = DataLoader(LeafDataset(train_paths), batch_size=a.batch_size, shuffle=True, num_workers=a.workers)
    val_loader = DataLoader(LeafDataset(val_paths), batch_size=a.batch_size, shuffle=False, num_workers=a.workers)

    model = ConvAutoencoder().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=a.lr)
    criterion = nn.MSELoss()
    history = []

    for epoch in range(a.epochs):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            loss = criterion(model(batch), batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch.size(0)
        train_loss /= len(train_paths)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                val_loss += criterion(model(batch), batch).item() * batch.size(0)
        val_loss /= len(val_paths)

        history.append({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss})
        print(f"Epoch {epoch + 1:2d}/{a.epochs}  train_loss={train_loss:.5f}  val_loss={val_loss:.5f}")

    out = Path(a.out)
    meta = {"trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "data_version": data_version(df), "n_train": len(train_paths), "n_val": len(val_paths),
            "final_train_loss": history[-1]["train_loss"], "final_val_loss": history[-1]["val_loss"],
            "args": vars(a), "torch": torch.__version__, "git_commit": git_commit(), "device": device}
    save_model(model, out, meta)
    export_onnx(model, out / "model.onnx")
    pd.DataFrame(history).to_csv(out / "history.csv", index=False)
    print(f"Saved model to {out}  (data_version={meta['data_version']})")


if __name__ == "__main__":
    main()
