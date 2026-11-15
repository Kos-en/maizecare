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
tests/            pytest (data-pipeline tests need no torch)
Makefile          make data | verify | train | eval | test
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

## Not built yet (see git branch `main` for an earlier, larger TF/Keras scaffold)
Stage-1 classifier, leaf masking, test split, web service / Docker / Render. They depend on whether the
classifier is done in PyTorch (as the notebook) or Keras (as the proposal says).
