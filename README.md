# Maize leaf severity: healthy-only autoencoder

A convolutional autoencoder trained on Healthy leaves only. Its reconstruction error on a
diseased leaf (MLND / MSV / MLB) is the severity signal, so no severity labels are needed.
The repo is the notebook `maize_severity_autoencoder` as scripts, plus an incremental
data-cleaning step so new images can be added safely.

```
src/maize/
  prepare.py      clean, dedup and split new images -> data/manifest.csv   (no torch)
  data.py         manifest helpers
  autoencoder.py  model, dataset, reconstruction_errors, save/load/ONNX export
  train.py        train on the healthy train split, writes models/ae/
  evaluate.py     false-positive check, error by class, formula correlation, heat maps
  severity.py     the notebook's OpenCV formula severity (comparison only)
  preprocess.py   photo -> model input, shared by training and the web app
  serve.py        FastAPI app: /api/score + the built frontend              (no torch)
frontend/         plain TypeScript UI, compiled to static files
```

## Workflow

```bash
pip install -e ".[dev]"     # installs the maize-* commands; add ",train" if torch isn't installed (Colab has it)

# first run: point at your cleaned folder (Healthy/ MLND/ MSV/ MLB/ subfolders)
maize-prepare --raw "/path/to/cv_datasets_cleaned"
# later: drop new images into <raw>/<Class>/ and run the same command again

maize-train                 # notebook settings: 20 epochs, lr 1e-3, batch 64
maize-evaluate              # results, figures and calibration.json in models/ae/eval/
```
Each command takes `--help`.

Set `MAIZE_DATA_ROOT` (or `--data-root`) to where `manifest.csv` should live; it can be a Drive folder.
Retraining is manual. Each model records the `data_version` it was trained on in `meta.json`.

### What `prepare` guarantees
- Idempotent: files already in the manifest are skipped.
- Splits never drift: an existing image keeps its group and train/val split.
- Near-duplicates cannot leak: a new image within 6 bits (min of pHash/aHash, as in the notebook) of an
  existing one inherits its split. If it is near both a train and a val image it is quarantined.
- Bad data is recorded in the `status` column, not dropped: `bad_image`, `too_small`, `duplicate_exact`,
  `label_conflict` (same picture under another class), `split_conflict`.
- New groups get a split from their hash (`val_frac`, default 15%), so arrival order does not matter.
- `maize-prepare --verify` re-checks the whole manifest for leakage (exit code 1 if any).

## Web app

```
make train eval    # train writes model.onnx, eval writes calibration.json
make release       # copies model.onnx, calibration.json, meta.json into release/ (small; commit them)
make serve         # builds the frontend, serves http://localhost:8000
make docker        # the same in the production container
```
The model is exported to ONNX at train time and served with `onnxruntime`, so the image has no torch
and fits Render's free 512 MB. The app returns the reconstruction error and a 0-1 severity score
`(e - median healthy) / (p99 diseased - median healthy)`. It does not name the disease; that needs the
stage-1 classifier. Heat maps are a developer tool (`maize-evaluate`), not shown to users; production
monitoring is planned in [docs/observability.md](docs/observability.md).

Deploy on Render (free): push to GitHub, then Render -> New -> Blueprint -> select the repo.
Free instances sleep after 15 min idle (first request takes about a minute). Photos are never stored.

## Image cache

`train` and `evaluate` do not read photos directly. The first run resizes every image once (in parallel) and saves
them to `<data root>/cache_128.npz`, keyed by each image's sha1. Later runs load that single file in seconds,
and new images from `prepare` are added on demand. Delete the file if you change `preprocess.resize` or
`IMG_SIZE`. The cache is not in git (it lives with the data).

## Colab

Training and eval want the GPU. Keep everything inside one data root on Drive (the manifest stores relative paths):

```
MyDrive/maize_data/
  raw/Healthy|MLND|MSV|MLB/...     <- your images (move cv_datasets_cleaned here) + any new drops
  manifest.csv                     <- written by prepare
  cache_128.npz                    <- written by the first train/evaluate; reused by every later run
MyDrive/maize_models/ae/           <- trained weights
```

```python
from google.colab import drive; drive.mount("/content/drive")
!git clone <repo-url> /content/repo
%cd /content/repo
# Colab already has torch/opencv: install only our package
!pip install -q --no-deps -e . && pip install -q imagehash

import os
DRIVE = "/content/drive/MyDrive/maize_data"
os.environ["MAIZE_DATA_ROOT"] = DRIVE

!maize-prepare --raw {DRIVE}/raw        # update manifest (CPU; safe to re-run after adding images)
!maize-train --out /content/drive/MyDrive/maize_models/ae   # first run builds the cache (slow once)
!maize-evaluate --model-dir /content/drive/MyDrive/maize_models/ae
```
If a session dies, nothing is lost: the manifest, cache and models are on Drive.

## Differences from the notebook (on purpose)
- Per-image formula-correlation loop now computes all reconstruction errors in one batch (same numbers, much faster).
- Correlation is run for every disease class, not just MLND.
- Figures are saved to disk instead of `plt.show()`.
- Group assignment is incremental (see above) instead of one-off complete-linkage clustering.

## Not built yet (tag `old-keras-scaffold` has an earlier, larger TF/Keras scaffold for reference)
Stage-1 classifier (disease name + retake-if-unsure), leaf masking, a held-out test split. The serving
code is deliberately framework-neutral (ONNX), so a PyTorch classifier slots in next to the autoencoder.
