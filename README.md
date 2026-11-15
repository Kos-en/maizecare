# MaizeCare: maize leaf disease and severity

Two models behind one web app for farmers (English and Kiswahili):
1. **Classifier** (Keras CNN, transfer learning): Healthy / MLND / MSV / MLB, with a confidence so
   unclear photos get "please retake" instead of a guess.
2. **Autoencoder** (PyTorch) trained on Healthy leaves only. Its reconstruction error on a diseased
   leaf is the severity signal, so no severity labels are needed.

New images go through the same incremental cleaning step (`maize-prepare`) before either model sees them.

```
src/maize/
  prepare.py      clean, dedup and split new images -> data/manifest.csv   (no torch)
  data.py         manifest helpers
  autoencoder.py  model, dataset, reconstruction_errors, save/load/ONNX export
  train.py        train on the healthy train split, writes models/ae/
  classifier.py   Keras CNN (MobileNetV2 / V3 / EfficientNet-B0 / ResNet50), writes models/clf/
  evaluate.py     false-positive check, error by class, formula correlation, heat maps
  severity.py     the notebook's OpenCV formula severity (comparison only)
  preprocess.py   photo -> model input, shared by training and the web app
  serve.py        FastAPI app: /api/score + the built frontend              (no torch)
frontend/         plain TypeScript UI, compiled to static files
```

## Workflow

```bash
make install                # pip install -e ".[data,serve]" --group dev; add the train extra if torch isn't installed

# first run: point at your cleaned folder (Healthy/ MLND/ MSV/ MLB/ subfolders)
maize-prepare --raw "/path/to/cv_datasets_cleaned"
# later: drop new images into <raw>/<Class>/ and run the same command again

maize-train                 # autoencoder, notebook settings: 20 epochs, lr 1e-3, batch 64
maize-evaluate              # results, figures and calibration.json in models/ae/eval/
maize-train-classifier --backbone mobilenet_v2   # classifier: head then fine-tune, models/clf/
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
make train eval classifier   # model.onnx + calibration.json, classifier.onnx + classifier.json
make release       # copies those (and meta.json) into release/ (small; commit them)
make serve         # builds the frontend, serves http://localhost:8000
make docker        # the same in the production container
```
Both models are exported to ONNX at train time and served with `onnxruntime`, so the image has no
torch, TensorFlow or Keras and fits Render's free 512 MB. `/api/score` returns the predicted label,
its confidence (`confident` is false below `min_confidence` in `classifier.json`, default 0.6), and a
0-1 severity score `(e - median healthy) / (p99 diseased - median healthy)`. The UI shows Healthy for a
healthy label, otherwise grades the disease mild / moderate / severe from the score. All UI text is in
`frontend/public/i18n/{en,sw}.json`; the Kiswahili still needs a review by a native speaker and the
advice by an extension officer. Heat maps are a developer tool (`maize-evaluate`), not shown to users; production
monitoring is planned in [docs/observability.md](docs/observability.md).

## Dependencies, CI and deployment

`pyproject.toml` is the only place dependencies are declared, split by use: the base (numpy, OpenCV),
`data`, `train` and `serve` extras, and a `dev` group. `uv.lock` pins every package to an exact version
and hash; `make lock` refreshes it (only releases at least 7 days old). The Docker image installs just
the `serve` extra from the lock, so production runs exactly what CI tested.

`.github/workflows/ci.yml` on every push and PR: ruff, pytest, `pip-audit` on the shipped dependencies,
`npm audit` and the frontend build. On a push to `main` it then builds the image, pushes it to GitHub's
registry (`ghcr.io/<user>/<repo>`, tagged `latest` and the commit SHA) and tells Render to deploy that
exact image. Dependabot opens weekly update PRs.

One-time setup on Render (free plan):
1. Push to `main` once so the image exists in GHCR, then make the package public
   (GitHub -> Packages -> maize -> Package settings), or add GHCR credentials in Render.
2. Render -> New -> Web Service -> Existing image -> `ghcr.io/<user>/<repo>:latest`, plan Free,
   health check path `/health` (or put the URL in `render.yaml` and use New -> Blueprint).
3. Copy the service's Deploy Hook URL (Settings) into the GitHub repo secret `RENDER_DEPLOY_HOOK_URL`.
Free instances sleep after 15 min idle (first request takes about a minute). Photos are never stored.

## Image cache

`train` and `evaluate` do not read photos directly. The first run resizes every image once (in parallel) and saves
them to `<data root>/cache_128.npz`, keyed by each image's sha1. Later runs load that single file in seconds,
and new images from `prepare` are added on demand. The classifier uses its own `cache_224.npz`. Delete a cache if you
change `preprocess.resize`, `IMG_SIZE` or `CLS_SIZE`. The cache is not in git (it lives with the data).

## Colab

Training and eval want the GPU. Keep everything inside one data root on Drive (the manifest stores relative paths):

```
MyDrive/maize_data/
  raw/Healthy|MLND|MSV|MLB/...     <- your images (move cv_datasets_cleaned here) + any new drops
  manifest.csv                     <- written by prepare
  cache_128.npz, cache_224.npz     <- written by the first run of each model; reused afterwards
MyDrive/maize_models/ae/, clf/     <- trained models
```

```python
from google.colab import drive; drive.mount("/content/drive")
!git clone <repo-url> /content/repo
%cd /content/repo
# Colab already has torch, keras, sklearn and opencv: install only our package
!pip install -q --no-deps -e . && pip install -q imagehash onnx

import os
DRIVE = "/content/drive/MyDrive/maize_data"
os.environ["MAIZE_DATA_ROOT"] = DRIVE

!maize-prepare --raw {DRIVE}/raw        # update manifest (CPU; safe to re-run after adding images)
!maize-train --out /content/drive/MyDrive/maize_models/ae   # first run builds the cache (slow once)
!maize-evaluate --model-dir /content/drive/MyDrive/maize_models/ae
!maize-train-classifier --out /content/drive/MyDrive/maize_models/clf
```
Then copy `model.onnx`, `calibration.json`, `meta.json` (ae/) and `classifier.onnx`, `classifier.json`
(clf/) into `release/` and push; CI builds and deploys the new image.
If a session dies, nothing is lost: the manifest, cache and models are on Drive.

## Differences from the notebook (on purpose)
- Per-image formula-correlation loop now computes all reconstruction errors in one batch (same numbers, much faster).
- Correlation is run for every disease class, not just MLND.
- Figures are saved to disk instead of `plt.show()`.
- Group assignment is incremental (see above) instead of one-off complete-linkage clustering.

## Not built yet
Leaf masking, a held-out test split (the classifier is scored on the validation split), and the
monitoring in docs/observability.md. Tag `old-keras-scaffold` keeps an earlier TF/Keras scaffold.
