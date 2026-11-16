# MaizeCare: maize leaf disease and severity

Two Keras models behind one web app for farmers (English and Kiswahili), following the proposal:
1. **Classifier** (ImageNet backbone, transfer learning): Healthy / MLB / MLND / MSV, with a confidence so
   unclear photos get "please retake" instead of a guess.
2. **Autoencoder** trained on Healthy leaves only. Its reconstruction error on a diseased leaf is the
   severity signal, so no severity labels are needed.

```
src/maize/
  pipeline.py     the `maize` command: runs every step, or one step
  data.py         step 1: check, group, split the images -> manifest.csv          (no torch)
  models.py       steps 2-3: classifier and autoencoder (Keras), exported to ONNX
  evaluate.py     step 4: calibrate the severity score on val, report on test
  severity.py     the notebook's colour formula, used as the reference index
  cache.py        resized images cached in one file per size
  preprocess.py   photo -> model input, shared by training and the web app
  serve.py        FastAPI app: /api/score + the built frontend                    (no torch)
frontend/         plain TypeScript UI, compiled to static files
```

## The pipeline

```bash
maize all --raw RAW --data-root ROOT --out MODELS     # everything, in order (GPU)
maize prepare --raw RAW --data-root ROOT              # only the data step (CPU is enough)
maize classifier --data-root ROOT --out MODELS --backbones mobilenet_v2   # any single step
```

| step | what it does | writes |
|---|---|---|
| `prepare` | file check, near-copy groups, drop mixed-label groups, split | `ROOT/manifest.csv` |
| `classifier` | trains each backbone (head, then fine-tune), compares them | `MODELS/classifiers.csv`, `MODELS/clf_<backbone>/` |
| `autoencoder` | trains on healthy train leaves | `MODELS/ae/` |
| `evaluate` | calibrates the 0-1 score on val; AUROC, false-high rate, Spearman on test | `MODELS/ae/calibration.json`, `metrics.json`, `scores.csv`, figures |
| `release` | best classifier (by val macro-F1) + autoencoder + calibration | `MODELS/release/` |

`maize --help` lists the options (backbones, epochs).

### The data step (proposal 3.3-3.4)
1. **File check**: opens as JPEG/PNG and the shortest side is at least 50 px, else `bad_image` / `too_small`.
2. **Near copies**: min(pHash, aHash) Hamming distance <= 6. Chains of near copies form one group.
3. **Mixed labels**: a group holding two classes is dropped whole (`label_conflict`).
4. **Exact copies**: same file contents, keep one (`duplicate_exact`).
5. **Split by group**: 15% test, 15% val, 70% train, from the group's hash, so a group lands in the
   same split on every run and near copies can never be on both sides. A leak check confirms it.

Only new files are read (the slow part); steps 2-5 are redone on the whole dataset every run, so the
result does not depend on the order images arrived. Unusable images stay in the manifest with a status.

## Colab

Data, caches and models live on Drive, so a dead session loses nothing: rerun the remaining steps.
Use a GPU runtime (L4) for `maize all`; `maize prepare` alone runs fine on CPU.

```bash
cd /content && git clone https://github.com/Kos-en/maizecare repo && cd repo
pip install -q --no-deps -e . && pip install -q imagehash onnx onnxscript   # Colab has torch, keras, sklearn
D=/content/drive/MyDrive
nohup maize all --raw $D/cv_datasets_cleaned --data-root $D --out $D/maize_models > $D/pipeline.log 2>&1 &
tail -f $D/pipeline.log
```
The first run builds the image caches (`cache_224.npz`, `cache_128.npz`) once; later runs load them in seconds.

## Web app

```
make release       # copies MODELS/release/* into release/ (small; commit them)
make serve         # builds the frontend, serves http://localhost:8000
make docker        # the same in the production container
```
Both models are exported to ONNX at train time and served with `onnxruntime`, so the image has no
torch, TensorFlow or Keras and fits Render's free 512 MB. `/api/score` returns the predicted label,
its confidence (`confident` is false below `min_confidence` in `classifier.json`, default 0.6), and a
0-1 severity score `(e - median healthy) / (p99 diseased - median healthy)`. The UI shows Healthy for a
healthy label, otherwise grades the disease mild / moderate / severe from the score. All UI text is in
`frontend/public/i18n/{en,sw}.json`; the Kiswahili still needs a review by a native speaker and the
advice by an extension officer. Heat maps are a developer tool (`maize evaluate`, heatmaps.png), not shown to users; production
monitoring is planned in [docs/observability.md](docs/observability.md).

## Dependencies, CI and deployment

`pyproject.toml` is the only place dependencies are declared, split by use: the base (numpy, OpenCV),
`data`, `train` and `serve` extras, and a `dev` group. `uv.lock` pins every package to an exact version
and hash; `make lock` refreshes it (only releases at least 7 days old). The Docker image installs just
the `serve` extra from the lock, so production runs exactly what CI tested.

`.github/workflows/ci.yml` on every push and PR: ruff, pytest, `pip-audit` on the shipped dependencies,
`npm audit` and the frontend build. On a push to `main` it then builds the image, pushes it to GitHub's
registry (`ghcr.io/kos-en/maizecare`, tagged `latest` and the commit SHA) and tells Render to deploy that
exact image. Dependabot opens weekly update PRs.

One-time setup on Render (free plan):
1. Push to `main` once so the image exists in GHCR, then make the package public
   (GitHub -> Packages -> maize -> Package settings), or add GHCR credentials in Render.
2. Render -> New -> Web Service -> Existing image -> `ghcr.io/kos-en/maizecare:latest`, plan Free,
   health check path `/health` (or put the URL in `render.yaml` and use New -> Blueprint).
3. Copy the service's Deploy Hook URL (Settings) into the GitHub repo secret `RENDER_DEPLOY_HOOK_URL`.
Free instances sleep after 15 min idle (first request takes about a minute). Photos are never stored.

## Not built yet
From the proposal: leaf masking, VAE and SSIM-loss autoencoder variants, 5-fold cross-validation,
blurred / low-light test copies, Grad-CAM, TFLite quantisation, and the monitoring in docs/observability.md.
