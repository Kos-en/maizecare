# Maize leaf severity: healthy-only autoencoder

A convolutional autoencoder trained on **Healthy** leaves only. Its reconstruction error on a
diseased leaf (MLND / MSV / MLB) is the severity signal: no severity labels needed.
This repo is the notebook `maize_severity_autoencoder` turned into scripts, plus an incremental
data-cleaning step so new images can be added safely.

```
src/maize/
  prepare.py      clean + dedup + split NEW images -> data/manifest.csv   (torch-free)
  data.py         manifest helpers
  autoencoder.py  model, dataset, reconstruction_errors, save/load        (torch)
  train.py        train on healthy train split, writes models/ae/
  evaluate.py     false-positive check, error by class, formula correlation, heat maps
  severity.py     the notebook's OpenCV formula severity (comparison signal only)
  preprocess.py   photo -> model input; shared by training AND the web app
  serve.py        FastAPI app: /api/score + serves the built frontend       (no torch)
frontend/         ~100 lines of TypeScript (no framework) -> compiled to static files
Dockerfile        builds the frontend, installs the slim server, ships release/
render.yaml       Render blueprint (free plan)
tests/            pytest (data-pipeline + web tests need no torch)
Makefile          make data | verify | train | eval | release | serve | docker | test
```

## Workflow

```bash
pip install -e ".[dev]"            # add ",train" if torch isn't installed (Colab has it)

# 1) FIRST RUN: point at your existing cleaned folder (Healthy/ MLND/ MSV/ MLB/ subfolders)
python -m maize.prepare --raw "/path/to/cv_datasets_cleaned"

# 2) later: drop new images into <raw>/<Class>/ and re-run the same command. That is the whole
#    "data cleans itself" step: safe to run any time, any number of times.
python -m maize.prepare --raw "/path/to/raw"

python -m maize.train              # same settings as the notebook (20 epochs, lr 1e-3, batch 64)
python -m maize.evaluate           # results + figures in models/ae/eval/
```

Set `MAIZE_DATA_ROOT` (or `--data-root`) to where `manifest.csv` should live; it can be a Drive folder.

### What `prepare` guarantees
- **Idempotent**: files already in the manifest are skipped.
- **Splits never drift**: an existing image keeps its group and train/val split forever.
- **Near-duplicates can't leak**: a new image within 6 bits (min of pHash/aHash, as in the notebook)
  of an existing one inherits its split. If it bridges train *and* val it is quarantined.
- **Bad data is recorded, not silently dropped** (`status` column): `bad_image`, `too_small`,
  `duplicate_exact`, `label_conflict` (same picture under another class), `split_conflict`.
- New groups get a deterministic split from their hash (`val_frac`, default 15%), so the result does not
  depend on arrival order.
- `python -m maize.prepare --verify` re-checks the whole manifest for leakage (exit code 1 if any).

Retraining is deliberately **manual** (`make train`): every model records the `data_version` it was
trained on in `meta.json`, so a result can always be traced to its exact dataset.

## Web app (production path)

```
make train eval    # train writes model.onnx, eval writes calibration.json
make release       # copies model.onnx + calibration.json + meta.json into release/  (commit these, they are small)
make serve         # local: builds the TypeScript, starts the app at http://localhost:8000
make docker        # same thing in the production container
```
`serve.py` starts, loads `release/`, and serves the compiled frontend from the same process, so there is
one container and one URL. The model is exported to ONNX at train time and served with `onnxruntime`
(no torch in production: small image, fits Render's free 512 MB). The app returns a reconstruction error,
a 0-1 severity score (`(e - median healthy) / (p99 diseased - median healthy)`) and an error heat map.
It does **not** name the disease yet: that needs the stage-1 classifier.

**Deploy on Render (free, no card):** push the repo to GitHub, Render -> New -> Blueprint -> select the repo.
Free instances sleep after 15 min idle (first request ~1 min) and have ~0.1 CPU. Photos are processed in
memory and never stored.

## Colab

Training/eval need the GPU; everything else can run anywhere. Layout on Drive (keep images INSIDE the data root,
because the manifest stores paths relative to it):

```
MyDrive/maize_data/
  raw/Healthy|MLND|MSV|MLB/...     <- your images (move cv_datasets_cleaned here) + any new drops
  manifest.csv                     <- written by prepare, persists across sessions
MyDrive/maize_models/ae/           <- trained weights
```

```python
from google.colab import drive; drive.mount("/content/drive")
!git clone <repo-url> /content/repo
%cd /content/repo
# Colab already has torch/opencv: install only our package, without dependencies
!pip install -q --no-deps -e . && pip install -q imagehash

import os
DRIVE = "/content/drive/MyDrive/maize_data"

# 1) update the manifest on Drive (CPU, safe to re-run after adding images)
os.environ["MAIZE_DATA_ROOT"] = DRIVE
!python -m maize.prepare --raw {DRIVE}/raw

# 2) copy images to fast local disk for training (Drive reads are slow); manifest paths still resolve
!mkdir -p /content/data && cp -r {DRIVE}/raw {DRIVE}/manifest.csv /content/data/
os.environ["MAIZE_DATA_ROOT"] = "/content/data"
!python -m maize.train --out /content/drive/MyDrive/maize_models/ae
!python -m maize.evaluate --model-dir /content/drive/MyDrive/maize_models/ae
```

## Differences from the notebook (on purpose)
- Per-image formula-correlation loop now computes all reconstruction errors in one batch (same numbers, much faster).
- Correlation is run for every disease class, not just MLND.
- Figures are saved to disk instead of `plt.show()`.
- Group assignment is incremental (see above) instead of one-off complete-linkage clustering.

## Not built yet (branch `main` has an earlier, larger TF/Keras scaffold for reference)
Stage-1 classifier (disease name + retake-if-unsure), leaf masking, a held-out test split. The serving
code is deliberately framework-neutral (ONNX), so a PyTorch classifier slots in next to the autoencoder.
