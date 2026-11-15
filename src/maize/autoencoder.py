"""Convolutional autoencoder (128x128 input, 8x8x32 bottleneck) plus dataset, save/load and ONNX export."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from .preprocess import IMG_SIZE, to_tensor

WEIGHTS = "healthy_autoencoder.pt"


def pick_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


class LeafDataset(Dataset):
    """images: (N,128,128,3) uint8 from cache.load_images."""

    def __init__(self, images: np.ndarray):
        self.images = images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return torch.from_numpy(to_tensor(self.images[idx]))


class ConvAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1), nn.ReLU(),     # 128 -> 64
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),    # 64 -> 32
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(),   # 32 -> 16
            nn.Conv2d(128, 32, 3, stride=2, padding=1), nn.ReLU(),   # 16 -> 8  (bottleneck)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 128, 4, stride=2, padding=1), nn.ReLU(),  # 8 -> 16
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1), nn.ReLU(),  # 16 -> 32
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), nn.ReLU(),   # 32 -> 64
            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1), nn.Sigmoid(),  # 64 -> 128
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


@torch.no_grad()
def reconstruction_errors(model, images: np.ndarray, device: str, batch_size: int = 64) -> np.ndarray:
    """Per-image mean squared reconstruction error, in the order of `images`."""
    model.eval()
    loader = DataLoader(LeafDataset(images), batch_size=batch_size, shuffle=False)
    errors = []
    for batch in loader:
        batch = batch.to(device)
        errors.extend(((model(batch) - batch) ** 2).mean(dim=[1, 2, 3]).cpu().numpy().tolist())
    return np.array(errors)


def save_model(model, out_dir: Path, meta: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / WEIGHTS)
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, default=str))


def load_model(model_dir: Path, device: str) -> ConvAutoencoder:
    model = ConvAutoencoder().to(device)
    model.load_state_dict(torch.load(Path(model_dir) / WEIGHTS, map_location=device))
    return model.eval()


def export_onnx(model, path: Path) -> None:
    """The web service runs this file with onnxruntime, so production needs no torch."""
    model = model.eval().cpu()
    dummy = torch.zeros(1, 3, IMG_SIZE, IMG_SIZE)
    kw = dict(input_names=["image"], output_names=["recon"], opset_version=17,
              dynamic_axes={"image": {0: "batch"}, "recon": {0: "batch"}})
    try:
        torch.onnx.export(model, dummy, str(path), dynamo=False, **kw)  # torch >= 2.5 needs dynamo=False
    except TypeError:
        torch.onnx.export(model, dummy, str(path), **kw)
